import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed, WanTransformerBlock
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


class WanVAE(nn.Module):
    def __init__(self, model_path, frames: int = 4) -> None:
        super().__init__()
        self.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")

        self.frames = frames

        self.pad_chunk_size = 4
        self.pad_num = self.get_pad_num()
        self.latent_t_num = self.get_latent_t_num()

        self.register_buffer("latent_mean", torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1))
        self.register_buffer("latent_std", torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1))

    def get_pad_num(self):
        remainder = (self.frames - 1) % self.pad_chunk_size
        if remainder != 0:
            return self.pad_chunk_size - remainder
        else:
            return 0

    def get_latent_t_num(self):
        T_pad = self.frames + self.pad_num
        return (T_pad - 1) // 4 + 1

    def align_video(self, videos: torch.Tensor):
        """
        videos: [B, T, C, H, W]
        """
        if self.pad_num != 0:
            last_frame = videos[:, -1:, :, :, :]
            pad_frames = last_frame.repeat(1, self.pad_num, 1, 1, 1)
            videos = torch.cat([videos, pad_frames], dim=1)
        return videos

    def inverse_align_video(self, videos: torch.Tensor):
        """
        videos: [B, T, C, H, W]
        """
        if self.pad_num != 0:
            videos = videos[:, : -self.pad_num, :, :, :]
        return videos

    @torch.no_grad()
    def encode(self, videos: torch.Tensor):
        """
        Args:
            videos: [B, T, C, H, W]
        Returns:
            video_latents: [B, C', T', H', W']
        """
        videos = self.align_video(videos)
        videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        # Align input dtype with VAE (may differ when VAE is outside ZeRO module tree)
        vae_dtype = self.latent_mean.dtype
        videos = videos.to(dtype=vae_dtype)
        video_latents = self.vae.encode(videos).latent_dist.sample()

        mean = self.latent_mean.to(video_latents.dtype)
        std = self.latent_std.to(video_latents.dtype)

        video_latents = (video_latents - mean) / std
        return video_latents

    @torch.no_grad()
    def decode(self, video_latents: torch.Tensor):
        """
        Args:
            video_latents: [B, C', T', H', W']
        Returns:
            videos: [B, T, C, H, W]
        """
        # Align input dtype with VAE
        vae_dtype = self.latent_mean.dtype
        video_latents = video_latents.to(dtype=vae_dtype)

        mean = self.latent_mean.to(video_latents.dtype)
        std = self.latent_std.to(video_latents.dtype)

        video_latents = video_latents * std + mean
        videos = self.vae.decode(video_latents, return_dict=False)[0]

        videos = videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        videos = self.inverse_align_video(videos)
        return videos


class VideoEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class TimeVideoEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        video_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)

        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)

        self.act_fn = nn.SiLU()

        self.time_proj = nn.Linear(dim, time_proj_dim)

        self.video_embedder = VideoEmbedding(
            in_features=video_embed_dim,
            out_features=dim,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)

        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.video_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        self.condition_embedder = TimeVideoEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            video_embed_dim=text_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def init_weights(self) -> None:
        # 1. Materialize and initialize meta params from custom modules not in checkpoint
        for name, module in self.named_modules():
            has_meta = any(p.device.type == "meta" for p in module.parameters(recurse=False))
            if not has_meta:
                continue

            module.to_empty(device=torch.device("cpu"))
            overwatch.warning(f"Materializing meta module: {name}")

            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight.flatten(1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 2. Reinitialize NaN params from corrupted checkpoint weights
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and torch.isnan(module.weight).any():
                overwatch.warning(f"Reinitializing NaN: {name}")
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if torch.isnan(self.patch_embedding.weight).any():
            overwatch.warning("Reinitializing NaN: patch_embedding")
            nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
            nn.init.zeros_(self.patch_embedding.bias)

        # 3. Final safety check
        for name, param in self.named_parameters():
            if param.device.type == "meta":
                overwatch.error(f"Meta param still not materialized: {name}")
            elif torch.isnan(param).any():
                overwatch.error(f"NaN param: {name}, {param.shape}, {param.device}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                overwatch.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def test_wanvae(args, videos, device, dtype):
    vae = WanVAE(
        args.wanva.model_path,
        frames=args.data.frames,
    ).to(device=device, dtype=dtype)

    video_latents = vae.encode(videos)
    decode_videos = vae.decode(video_latents)

    print(f"videos.shape: {videos.shape}")
    print(f"video_latents.shape: {video_latents.shape}")
    print(f"decode_videos.shape: {decode_videos.shape}")
    return video_latents


def test_transformer3d(args, video_latents, device, dtype):
    batch_size = video_latents.shape[0]

    transformer3d, info = WanTransformer3DModel.from_pretrained(
        args.wanva.model_path,
        subfolder="transformer",
        patch_size=args.wanva.patch_size,
        num_attention_heads=args.wanva.num_attention_heads,
        text_dim=args.projector.output_align_dim,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
    )
    transformer3d.init_weights()
    transformer3d = transformer3d.to(device=device, dtype=dtype)

    timestep = torch.zeros((batch_size), dtype=torch.float32, device=device)

    encoder_hidden_states = torch.randn(
        (batch_size, args.projector.num_token, args.projector.output_align_dim), device=device, dtype=dtype
    )

    video_hidden_states = transformer3d(
        timestep=timestep,
        hidden_states=video_latents,
        encoder_hidden_states=encoder_hidden_states,
    ).sample

    print(f"video_hidden_states.shape: {video_hidden_states.shape}")


if __name__ == "__main__":
    from univam.utils.args import load_args
    from univam.utils.data import load_multi_datasets_form_json, set_seed

    args = load_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    batch_size = 2

    # get real data via Dataset
    eval_dataloader = load_multi_datasets_form_json(
        args.data,
        json_path=args.data.eval_json_path,
        local_batch_size=batch_size,
        num_workers=0,
        is_infinite=False,
        shuffle=False,
        drop_last=False,
        eval_sample_num=args.train.eval_sample_num,
        make_single_dataset=True,
    )
    data = next(iter(eval_dataloader))
    videos = data["videos"]

    video_latents = test_wanvae(args, videos, device, dtype)
    test_transformer3d(args, video_latents, device, dtype)


class FluxWanVAE(nn.Module):
    """
    混合 VAE：Flux Encoder + 额外映射层 + Wan Decoder
    导师方案："用 flux 试试吧，enc 在 vae enc 后面再套几层"
    """
    def __init__(self, flux_model_path: str = None, wan_model_path: str = None, frames: int = 4):
        super().__init__()

        # 默认使用 FLUX.1-schnell
        if flux_model_path is None:
            flux_model_path = "black-forest-labs/FLUX.1-schnell"

        overwatch.info(f"Loading Flux VAE encoder from: {flux_model_path}")

        # 1. 加载 Flux VAE (只用 encoder)
        from diffusers import AutoencoderKL
        try:
            flux_vae = AutoencoderKL.from_pretrained(
                flux_model_path,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            self.flux_encoder = flux_vae.encoder
        except Exception as e:
            overwatch.warning(f"Failed to load from subfolder 'vae', trying root: {e}")
            flux_vae = AutoencoderKL.from_pretrained(
                flux_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            self.flux_encoder = flux_vae.encoder

        self.flux_encoder.requires_grad_(False)
        self.flux_encoder.eval()
        overwatch.info("Flux encoder loaded and frozen")

        # 2. 加载 Wan VAE (只用 decoder)
        overwatch.info(f"Loading Wan VAE decoder from: {wan_model_path}")
        wan_vae = AutoencoderKLWan.from_pretrained(wan_model_path, subfolder="vae")
        self.wan_decoder = wan_vae
        self.wan_decoder.requires_grad_(False)
        self.wan_decoder.eval()
        overwatch.info("Wan decoder loaded and frozen")

        # Wan VAE 的标准化参数
        self.register_buffer("latent_mean", torch.tensor(wan_vae.config.latents_mean).view(1, -1, 1, 1, 1))
        self.register_buffer("latent_std", torch.tensor(wan_vae.config.latents_std).view(1, -1, 1, 1, 1))

        # 3. "套几层" - 额外的 encoder 层（可训练）
        # Flux: [B, 16, T, 32, 32] (假设256x256输入 → 32x32, 8x压缩)
        # → Wan: [B, 48, T', 16, 16] (16x压缩, 4x时间压缩)
        self.extra_encoder_layers = nn.Sequential(
            # 第1层：空间下采样 2x + 通道扩展
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # 第2层：通道扩展到 48
            nn.Conv3d(32, 48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),

            # 第3层：时间压缩 4x（匹配 Wan 的时间压缩）
            nn.Conv3d(48, 48, kernel_size=(4, 1, 1), stride=(4, 1, 1), padding=0),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
        )
        overwatch.info("Extra encoder layers initialized (trainable)")

        # 视频对齐参数（与原 WanVAE 一致）
        self.frames = frames
        self.pad_chunk_size = 4
        self.pad_num = self.get_pad_num()
        self.latent_t_num = self.get_latent_t_num()

    def get_pad_num(self):
        remainder = (self.frames - 1) % self.pad_chunk_size
        if remainder != 0:
            return self.pad_chunk_size - remainder
        else:
            return 0

    def get_latent_t_num(self):
        T_pad = self.frames + self.pad_num
        return (T_pad - 1) // 4 + 1

    def align_video(self, videos: torch.Tensor):
        """videos: [B, T, C, H, W]"""
        if self.pad_num != 0:
            last_frame = videos[:, -1:, :, :, :]
            pad_frames = last_frame.repeat(1, self.pad_num, 1, 1, 1)
            videos = torch.cat([videos, pad_frames], dim=1)
        return videos

    def inverse_align_video(self, videos: torch.Tensor):
        """videos: [B, T, C, H, W]"""
        if self.pad_num != 0:
            videos = videos[:, : -self.pad_num, :, :, :]
        return videos

    def encode(self, videos: torch.Tensor):
        """
        Args:
            videos: [B, T, C, H, W]
        Returns:
            video_latents: [B, 48, T', H/16, W/16]
        """
        videos = self.align_video(videos)
        B, T, C, H, W = videos.shape

        # Step 1: Flux encoder 逐帧编码（冻结）
        flux_latents = []
        with torch.no_grad():
            for t in range(T):
                frame = videos[:, t]  # [B, C, H, W]
                # Flux encoder: [B, C, H, W] → [B, 16, H/8, W/8]
                latent_t = self.flux_encoder(frame)
                flux_latents.append(latent_t)

        flux_latents = torch.stack(flux_latents, dim=2)  # [B, 16, T, H/8, W/8]

        # Step 2: 额外的 encoder 层（可训练）
        wan_latents = self.extra_encoder_layers(flux_latents)  # [B, 48, T', H/16, W/16]

        # Step 3: 标准化（使用 Wan VAE 的参数）
        mean = self.latent_mean.to(wan_latents.dtype)
        std = self.latent_std.to(wan_latents.dtype)
        wan_latents = (wan_latents - mean) / std

        return wan_latents

    @torch.no_grad()
    def decode(self, video_latents: torch.Tensor):
        """
        Args:
            video_latents: [B, 48, T', H/16, W/16]
        Returns:
            videos: [B, T, C, H, W]
        """
        # 反标准化
        mean = self.latent_mean.to(video_latents.dtype)
        std = self.latent_std.to(video_latents.dtype)
        video_latents = video_latents * std + mean

        # 使用 Wan decoder
        videos = self.wan_decoder.decode(video_latents, return_dict=False)[0]

        # [B, C, T, H, W] → [B, T, C, H, W]
        videos = videos.permute(0, 2, 1, 3, 4)
        videos = self.inverse_align_video(videos)

        return videos

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.flux_encoder.to(*args, **kwargs)
        self.wan_decoder.to(*args, **kwargs)
        return self

    def eval(self):
        super().eval()
        self.flux_encoder.eval()
        self.wan_decoder.eval()
        # extra_encoder_layers 在训练时会被单独设置
        return self

    def train(self, mode=True):
        super().train(mode)
        # 保持 encoder/decoder 冻结
        self.flux_encoder.eval()
        self.wan_decoder.eval()
        return self

    def requires_grad_(self, requires_grad: bool):
        # 只有 extra_encoder_layers 可训练
        self.extra_encoder_layers.requires_grad_(requires_grad)
        self.flux_encoder.requires_grad_(False)
        self.wan_decoder.requires_grad_(False)
        return self
