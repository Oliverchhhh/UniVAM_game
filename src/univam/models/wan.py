import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor, WanRotaryPosEmbed
from einops import rearrange

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
        video_timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        video_timestep = self.timesteps_proj(video_timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if video_timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            video_timestep = video_timestep.to(time_embedder_dtype)

        video_temb = self.time_embedder(video_timestep).type_as(encoder_hidden_states)
        video_timestep_proj = self.time_proj(self.act_fn(video_temb))

        encoder_hidden_states = self.video_embedder(encoder_hidden_states)

        return video_temb, video_timestep_proj, encoder_hidden_states


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        video_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        video_temb: torch.Tensor,
    ) -> torch.Tensor:
        # fmt: off
        if video_temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            (
                video_shift_msa,
                video_scale_msa,
                video_gate_msa,
                video_c_shift_msa,
                video_c_scale_msa,
                video_c_gate_msa,
            ) = self.scale_shift_table.unsqueeze(0) + video_temb.float().chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            video_shift_msa = video_shift_msa.squeeze(2)
            video_scale_msa = video_scale_msa.squeeze(2)
            video_gate_msa = video_gate_msa.squeeze(2)
            video_c_shift_msa = video_c_shift_msa.squeeze(2)
            video_c_scale_msa = video_c_scale_msa.squeeze(2)
            video_c_gate_msa = video_c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            (
                video_shift_msa,
                video_scale_msa,
                video_gate_msa,
                video_c_shift_msa,
                video_c_scale_msa,
                video_c_gate_msa,
            ) = (self.scale_shift_table + video_temb.float()).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(video_hidden_states.float()) * (1 + video_scale_msa) + video_shift_msa).type_as(video_hidden_states)

        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)

        video_hidden_states = (video_hidden_states.float() + attn_output * video_gate_msa).type_as(video_hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(video_hidden_states.float()).type_as(video_hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        video_hidden_states = video_hidden_states + attn_output

        # 3. Feed-forward
        video_norm_hidden_states = (self.norm3(video_hidden_states.float()) * (1 + video_c_scale_msa) + video_c_shift_msa).type_as(video_hidden_states)
        video_ff_output = self.ffn(video_norm_hidden_states)
        video_hidden_states = (video_hidden_states.float() + video_ff_output.float() * video_c_gate_msa).type_as(video_hidden_states)
        # fmt: on

        return video_hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        in_channels: int = 48,
        out_channels: int = 48,
        video_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len=1024,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        self.condition_embedder = TimeVideoEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            video_embed_dim=video_dim,
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

    def init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and torch.isnan(module.weight).any():
                overwatch.warning(f"Reinitializing: {name}")
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if torch.isnan(self.patch_embedding.weight).any():
            overwatch.warning("Reinitializing: patch_embedding")
            nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
            nn.init.zeros_(self.patch_embedding.bias)

        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                overwatch.error(f"NaN param: {name}, {param.shape}, {param.device}")

    def forward(
        self,
        video_timestep: torch.Tensor,
        video_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        B, C, T, H, W = video_hidden_states.shape

        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = T // p_t
        post_patch_height = H // p_h
        post_patch_width = W // p_w

        rotary_emb = self.rope(video_hidden_states)

        video_hidden_states = self.patch_embedding(video_hidden_states)
        video_hidden_states = video_hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if video_timestep.ndim == 2:
            ts_seq_len = video_timestep.shape[1]
            video_timestep = video_timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        video_temb, video_timestep_proj, encoder_hidden_states = (
            self.condition_embedder(
                video_timestep=video_timestep,
                encoder_hidden_states=encoder_hidden_states,
            )
        )

        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            video_timestep_proj = video_timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            video_timestep_proj = video_timestep_proj.unflatten(1, (6, -1))

        for block in self.blocks:
            video_hidden_states = block(
                video_hidden_states=video_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_emb=rotary_emb,
                video_temb=video_timestep_proj,
            )

        # fmt: off
        if video_temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            video_shift, video_scale = (self.scale_shift_table.unsqueeze(0).to(video_temb.device) + video_temb.unsqueeze(2)).chunk(2, dim=2)
            video_shift = video_shift.squeeze(2)
            video_scale = video_scale.squeeze(2)
        else:
            # batch_size, inner_dim
            video_shift, video_scale = (self.scale_shift_table.to(video_temb.device) + video_temb.unsqueeze(1)).chunk(2, dim=1)

        video_shift = video_shift.to(video_hidden_states.device)
        video_scale = video_scale.to(video_hidden_states.device)

        video_hidden_states = (self.norm_out(video_hidden_states.float()) * (1.0 + video_scale) + video_shift).type_as(video_hidden_states)
        # fmt: on

        video_hidden_states = self.proj_out(video_hidden_states)
        video_hidden_states = rearrange(
            video_hidden_states,
            "b (t h w) (c p1 p2 p3) -> b c (t p1) (h p2) (w p3)",
            t=post_patch_num_frames,
            h=post_patch_height,
            w=post_patch_width,
            p1=p_t,
            p2=p_h,
            p3=p_w,
        )

        return video_hidden_states


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
        video_dim=args.projector.output_align_dim,
        low_cpu_mem_usage=False,
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
        video_timestep=timestep,
        video_hidden_states=video_latents,
        encoder_hidden_states=encoder_hidden_states,
    )

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
