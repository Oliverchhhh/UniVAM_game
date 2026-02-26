import inspect
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from flash_attn import flash_attn_func
from tqdm.auto import tqdm

from univam.models.deepstack import Qwen3VLVideoFeatureExtractor
from univam.models.projector import MLPProjector, QformerProjector
from univam.utils.data import check_tensor
from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def custom_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)


def get_mesh_id(f, h, w, f_w=1, f_shift=0, action=False):
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")

    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
    return grid_id


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class VAVAE(nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.video_vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")

        self.chunk_size = 4

        self.register_buffer("latent_mean", torch.tensor(self.video_vae.config.latents_mean).view(1, -1, 1, 1, 1))
        self.register_buffer("latent_std", torch.tensor(self.video_vae.config.latents_std).view(1, -1, 1, 1, 1))

    @torch.no_grad()
    def encode(self, videos: torch.Tensor, actions: torch.Tensor | None = None):
        """
        Joint VAE encode, assume that actions ~ N(?, ?)
            videos:  [B, T, C, H, W]
            actions: [B, T, C, chunk_size, 1]
        """
        B, T, C, H, W = videos.shape
        videos = videos.reshape(B * T, C, 1, H, W)
        video_latents = self.video_vae.encode(videos).latent_dist.sample()

        video_latents = rearrange(
            video_latents,
            "(b t) c 1 h w -> b c t h w",
            b=B,
            t=T,
        )

        mean = self.latent_mean.to(video_latents.dtype)
        std = self.latent_std.to(video_latents.dtype)

        video_latents = (video_latents - mean) / std

        if actions is None:
            return video_latents
        else:
            action_latents = actions
            return video_latents, action_latents

    @torch.no_grad()
    def decode(self, video_latents: torch.Tensor, action_latents: torch.Tensor | None = None):
        """
        Joint VAE decode, assume that actions ~ N(?, ?)
            videos:  [B, C, T, H, W]
            actions: [B, C, T, chunk_size, 1]
        """
        B, C, T, H, W = video_latents.shape

        mean = self.latent_mean.to(video_latents.dtype)
        std = self.latent_std.to(video_latents.dtype)

        video_latents = video_latents * std + mean

        video_latents = rearrange(video_latents, "b c t h w -> (b t) c 1 h w")
        videos = self.video_vae.decode(video_latents, return_dict=False)[0]

        videos = rearrange(
            videos,
            "(b t) c 1 h w -> b t c h w",
            b=B,
            t=T,
        )

        if action_latents is None:
            return videos
        else:
            actions = action_latents
            return videos, actions


class VideoEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_video: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_video)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


# TODO: modify to align action
class ActionEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_action: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_action)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class TimeVideoActionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        video_embed_dim: int,
        action_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)

        # video_embedder
        self.video_embedder = VideoEmbedding(
            in_features=video_embed_dim,
            out_features=dim,
        )

        # action_embedder
        self.action_embedder = None
        if action_embed_dim is not None:
            self.action_embedder = ActionEmbedding(action_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states_video: torch.Tensor,
        encoder_hidden_states_action: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states_video)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states_video = self.video_embedder(encoder_hidden_states_video)
        check_tensor(encoder_hidden_states_video, "encoder_hidden_states_video")
        if encoder_hidden_states_action is not None:
            encoder_hidden_states_action = self.action_embedder(encoder_hidden_states_action)

        return temb, timestep_proj, encoder_hidden_states_video, encoder_hidden_states_action


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3
        self.f_dim = self.attention_head_dim - self.h_dim - self.w_dim

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()

        self.register_buffer("f_freqs_base", f_freqs_base, persistent=False)
        self.register_buffer("h_freqs_base", h_freqs_base, persistent=False)
        self.register_buffer("w_freqs_base", w_freqs_base, persistent=False)

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (self.theta ** (torch.arange(0, self.f_dim, 2)[: (self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta ** (torch.arange(0, self.h_dim, 2)[: (self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta ** (torch.arange(0, self.w_dim, 2)[: (self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class WanAttention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        cross_attention_dim_head: Optional[int] = None,
        attn_mode: str = "sdpa",
    ):
        super().__init__()
        if attn_mode == "sdpa":
            self.attn_op = custom_sdpa
        elif attn_mode == "flash_attention_2":
            if torch.cuda.is_available():
                self.attn_op = flash_attn_func
            else:
                self.attn_op = custom_sdpa
                overwatch.warning("GPU not available. Fall back to `sdpa`")
        else:
            raise ValueError(f"Unsupported attention mode: {attn_mode}, only support torch and flashattn")

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rotary_emb: Optional[torch.Tensor] = None,
    ):
        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        hidden_states = self.attn_op(query, key, value)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        attn_mode: str = "sdpa",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = rearrange(
            temb_scale_shift_table,
            "b l n c -> b n l c",
        ).chunk(6, dim=1)

        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            None,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )

        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


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
        action_dim: Optional[int] = None,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        rope_max_seq_len=1024,
        attn_mode: str = "sdpa",
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        # self.patch_embedding_mlp = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], inner_dim)

        # 2. Condition embeddings
        self.condition_embedder = TimeVideoActionEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            video_embed_dim=video_dim,
            action_embed_dim=action_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(inner_dim, ffn_dim, num_attention_heads, cross_attn_norm, eps, attn_mode=attn_mode)
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # 5. action parts
        self.action_embedder = None
        self.action_proj_out = None
        if action_dim is not None:
            self.action_embedder = nn.Linear(action_dim, inner_dim)
            self.action_proj_out = nn.Linear(inner_dim, action_dim)

    def forward(
        self,
        grid_id: torch.Tensor,
        timestep: torch.Tensor,
        # TODO deside to cat action tokens or not
        hidden_states_video: torch.Tensor,
        encoder_hidden_states_video: torch.Tensor,
        hidden_states_action: Optional[torch.Tensor] = None,
        encoder_hidden_states_action: Optional[torch.Tensor] = None,
    ):
        B, C, T, H, W = hidden_states_video.shape

        hidden_states_video = self.patch_embedding(hidden_states_video)
        hidden_states = hidden_states_video.flatten(2).transpose(1, 2)

        if hidden_states_action is not None:
            # TODO follow the action encoder/tokenizer rules
            hidden_states_action = rearrange(
                hidden_states_action,
                "b c f h w -> b (f h w) c",
            )
            hidden_states_action = self.action_embedder(hidden_states_action)
            hidden_states = torch.cat([hidden_states, hidden_states_action], dim=1)
            action_token_nums = hidden_states_action[1]

        rotary_emb = self.rope(grid_id)[:, :, None]  # [B, tokens, 1, (H//p2)*(W//p3)]

        temb, timestep_proj, encoder_hidden_states_video, encoder_hidden_states_action = self.condition_embedder(
            timestep=timestep,
            encoder_hidden_states_video=encoder_hidden_states_video,
            encoder_hidden_states_action=encoder_hidden_states_action,
        )

        timestep_proj = timestep_proj[:, None, :]  # [B, 1, 6*inner_dim]
        timestep_proj = timestep_proj.expand(-1, hidden_states.shape[1], -1)  # [B, tokens, 6*inner_dim]
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states_video,
                timestep_proj,
                rotary_emb,
            )

        temb_scale_shift_table = self.scale_shift_table + temb[:, None, :]
        shift, scale = temb_scale_shift_table.chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)

        if hidden_states_action is not None:
            # TODO: maybe split hidden_states
            hidden_states_action = hidden_states[:, action_token_nums - 1 :, :]
            hidden_states_action = self.action_proj_out(hidden_states_action)
            hidden_states_action = rearrange(
                hidden_states_action,
                "b (t h w) c -> b c t h w",
                t=T,
                h=H,
                w=W,
            )
        else:
            hidden_states_video = self.proj_out(hidden_states)
            hidden_states_video = rearrange(
                hidden_states_video,
                "b (t h w) (c p1 p2 p3) -> b c (t p1) (h p2) (w p3)",
                t=T // self.patch_size[0],
                h=H // self.patch_size[1],
                w=W // self.patch_size[2],
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )

        return hidden_states_video, hidden_states_action


class Wan22VisionActionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.video_feature_extractor = Qwen3VLVideoFeatureExtractor(
            config.video_feature_extractor,
            frames=config.data.frames,
            image_size=config.data.image_size,
        )

        if config.projector.type == "mlp":
            self.projector = MLPProjector(
                config.projector,
                self.video_feature_extractor.patches,
                self.video_feature_extractor.out_hidden_size,
            )
        elif config.projector.type == "qformer":
            self.projector = QformerProjector(
                config.projector,
                self.video_feature_extractor.patches,
                self.video_feature_extractor.out_hidden_size,
            )
        else:
            raise ValueError(f"Unknown projector type '{config.projector.type}'. ")

        self.vavae = VAVAE(config.wanva.model_path)
        # self.vae_scale_factor_spatial = self.vavae.video_vae.config.scale_factor_spatial
        # self.vae_scale_factor_temporal = self.vavae.video_vae.config.scale_factor_temporal

        height, width = config.data.image_size
        self.frames = config.data.frames

        self.vae_height = height // self.vavae.video_vae.config.scale_factor_spatial
        self.vae_width = width // self.vavae.video_vae.config.scale_factor_spatial

        self.transformer3d = WanTransformer3DModel.from_pretrained(
            config.wanva.model_path,
            subfolder="transformer",
            patch_size=config.wanva.patch_size,
            num_attention_heads=config.wanva.num_attention_heads,
            attn_mode=config.wanva.attn_mode,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )

        target_modules = ["video_embedder.ff", "action_embedder.ff"]
        for name, module in self.transformer3d.named_modules():
            if (
                any(target in name for target in target_modules)
                and isinstance(module, torch.nn.Linear)
                and torch.isnan(module.weight).any()
            ):
                overwatch.warning(f"Reinitializing: f{name}")
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        for name, param in self.transformer3d.named_parameters():
            if torch.isnan(param).any():
                overwatch.error(f"NaN param: {name}, {param.shape}, {param.device}")

        self.num_channels_latents = self.transformer3d.config.in_channels

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.wanva.model_path, subfolder="scheduler")

        self.eval_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.wanva.model_path, subfolder="scheduler"
        )

        self.seed = getattr(config, "seed", 33)
        self.guidance_scale = getattr(config.wanva, "guidance_scale", 1.0)
        self.num_inference_steps = getattr(config.wanva, "num_inference_steps", 50)

        self.token_dropout = getattr(config.wanva, "token_dropout", False)
        self.num_token = config.projector.num_token

    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.dtype = next(self.transformer3d.parameters()).dtype
        return model_converted

    def train(self, *args):
        super().train(*args)
        self.set_trainable_params()

    def set_trainable_params(self):
        self.transformer3d.train()
        self.transformer3d.requires_grad_(True)

        self.projector.train()
        self.projector.requires_grad_(True)

        self.vavae.eval()
        self.vavae.requires_grad_(False)

        self.video_feature_extractor.eval()
        self.video_feature_extractor.requires_grad_(False)

    def _save_ckpt(self, model_dict: Dict, projector_model_dict: Dict, save_path: str, global_step: int) -> None:
        exclude_prefixes = ["vavae", "projector", "video_feature_extractor"]
        save_dict = {"model": {}, "global_step": global_step}
        for k, v in model_dict.items():
            if not any(k.startswith(prefix) for prefix in exclude_prefixes):
                save_dict["model"][k] = v
        torch.save(save_dict, os.path.join(save_path, "Wan22VAM.pth"))
        torch.save(projector_model_dict, os.path.join(save_path, "Projector.pth"))

    def _load_ckpt(self, load_path: str) -> int:
        assert os.path.exists(os.path.join(load_path, "Projector.pth")), f"Projector.pth not found in {load_path}"
        assert os.path.exists(os.path.join(load_path, "Wan22VAM.pth")), f"Wan22VAM.pth not found in {load_path}"
        overwatch.warning(f"loading checkpoints from {load_path}")

        def _log_missing_unexpected(title, missing_keys, unexpected_keys):
            def extract_top_level(keys):
                return sorted({k.split(".")[0] for k in keys})

            top_missing = extract_top_level(missing_keys)
            top_unexpected = extract_top_level(unexpected_keys)

            overwatch.warning(f"{title} - Missing top-level keys: {top_missing}")
            overwatch.warning(f"{title} - Unexpected top-level keys: {top_unexpected}")

        wanvam_ckpt = torch.load(os.path.join(load_path, "Wan22VAM.pth"), map_location="cpu")
        missing, unexpected = self.load_state_dict(wanvam_ckpt["model"], strict=False)
        _log_missing_unexpected("Wan22VAM", missing, unexpected)

        projector_ckpt = torch.load(os.path.join(load_path, "Projector.pth"), map_location="cpu")
        missing, unexpected = self.projector.load_state_dict(projector_ckpt, strict=False)
        _log_missing_unexpected("Projector", missing, unexpected)

        return wanvam_ckpt["global_step"]

    def progress_bar(self, iterable=None, total=None, use_tqdm=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, ncols=150, dynamic_ncols=False, disable=not use_tqdm, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, ncols=150, dynamic_ncols=False, disable=not use_tqdm, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def prepare_latents(self, batch_size, dtype, device, generator, latents=None):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            self.num_channels_latents,
            self.frames,
            self.vae_height,
            self.vae_width,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def _make_grid_id(self, latents, dtype, action=False):
        f = latents.shape[-3] // self.transformer3d.patch_size[0]
        h = latents.shape[-2] // self.transformer3d.patch_size[1]
        w = latents.shape[-1] // self.transformer3d.patch_size[2]

        grid_id = get_mesh_id(
            f=f,
            h=h,
            w=w,
            f_w=1,
            f_shift=0,
            action=action,
        ).unsqueeze(0)

        # assume videos are same shape
        grid_id = torch.cat([grid_id] * latents.shape[0], dim=0).to(device=latents.device, dtype=dtype)
        return grid_id

    def encode(self, videos: torch.Tensor, do_classifier_free_guidance: bool = False):
        dtype = next(self.projector.parameters()).dtype
        videos = videos.to(device=self.device, dtype=dtype)

        pixel_values_videos, video_grid_thw = self.video_feature_extractor.preprocess(videos)
        video_pooler_feature = self.video_feature_extractor(pixel_values_videos, video_grid_thw)
        check_tensor(video_pooler_feature, "video_pooler_feature")

        video_embeds = self.projector(video_pooler_feature)
        check_tensor(video_embeds, "video_embeds")

        if self.token_dropout:
            dropout_range = torch.randint(1, self.num_token + 1, ())
            video_embeds = video_embeds[:, :dropout_range]

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(video_embeds)
            video_embeds = torch.cat([negative_prompt_embeds, video_embeds])

        video_embeds = video_embeds.to(dtype=dtype)
        return video_embeds

    def train_step(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        videos: torch.Tensor = inputs["videos"]  # [B, T, C, H, W]

        batch_size = videos.shape[0]

        video_embeds = self.encode(videos)

        video_latents = self.vavae.encode(videos)
        video_latents = video_latents.to(dtype=self.dtype)
        check_tensor(video_latents, "video_latents")

        first_frame_latents = video_latents[:, 0:1, :, :, :]

        video_noise = torch.randn_like(video_latents, dtype=self.dtype)

        timestep_id = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,))
        timesteps = self.scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)

        sigmas = self.scheduler.sigmas[timestep_id].to(dtype=self.dtype, device=self.device)
        sigmas = sigmas.view(batch_size, 1, 1, 1, 1)
        video_noisy_latents = (1.0 - sigmas) * video_latents + sigmas * video_noise
        video_noisy_latents[:, 0:1, :, :, :] = first_frame_latents

        # Flow-Matching target
        target = video_noise - video_latents
        target[:, 0:1, :, :, :] = 0

        grid_ids = self._make_grid_id(video_noisy_latents, dtype=video_noisy_latents.dtype)
        model_pred_video_latents, _ = self.transformer3d(
            grid_id=grid_ids,
            timestep=timesteps,
            hidden_states_video=video_noisy_latents,
            encoder_hidden_states_video=video_embeds,
        )
        check_tensor(model_pred_video_latents, "model_pred_video_latents", check_bound=100, check_std=10)
        model_pred_video_latents[:, 0:1, :, :, :] = 0

        loss = torch.nn.functional.mse_loss(model_pred_video_latents, target, reduction="mean")

        outputs["loss"] = loss
        return outputs

    @torch.no_grad()
    def eval_step(self, inputs: Dict[str, Any], outputs: Dict[str, Any], use_tqdm: bool = True) -> Dict[str, Any]:
        videos = inputs["videos"]
        generator = inputs["generator"]

        do_classifier_free_guidance = self.guidance_scale > 1.0
        video_embeds = self.encode(videos, do_classifier_free_guidance=do_classifier_free_guidance)

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler=self.eval_scheduler,
            num_inference_steps=self.num_inference_steps,
            device=self.device,
            timesteps=None,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.eval_scheduler.order, 0)

        latents = self.prepare_latents(
            batch_size=videos.shape[0],
            dtype=self.dtype,
            device=self.device,
            generator=generator,
            latents=None,
        )

        first_frame = videos[:, 0:1, :, :, :]
        first_frame_latents = self.vavae.encode(first_frame)

        latents[:, :, 0:1, :, :] = first_frame_latents

        grid_ids = self._make_grid_id(latents, dtype=latents.dtype)

        with self.progress_bar(total=num_inference_steps, use_tqdm=use_tqdm) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                grid_model_input = torch.cat([grid_ids] * 2) if do_classifier_free_guidance else grid_ids

                timestep = t.expand(latent_model_input.shape[0])
                noise_pred_video, _ = self.transformer3d(
                    grid_id=grid_model_input,
                    timestep=timestep,
                    hidden_states_video=latent_model_input,
                    encoder_hidden_states_video=video_embeds,
                )

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred_video.chunk(2)
                    noise_pred_video = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.eval_scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.eval_scheduler.order == 0):
                    progress_bar.update()

        latents = latents.to(dtype=self.dtype)
        gen_videos = self.vavae.decode(latents)

        outputs["videos"] = gen_videos
        return outputs

    def forward(self, inputs, **kwargs):
        outputs = {}

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        inputs["generator"] = generator

        if self.training:
            outputs = self.train_step(inputs, outputs, **kwargs)
        else:
            outputs = self.eval_step(inputs, outputs, **kwargs)

        inputs.pop("generator", None)
        return outputs


class FlopsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, videos, **kwargs):
        inputs = {"videos": videos}
        return self.model(inputs, **kwargs)


def test_vavae(args, videos, device, dtype):
    va_vae = VAVAE(args.wanva.model_path).to(device=device, dtype=dtype)

    video_latents = va_vae.encode(videos)
    decode_videos = va_vae.decode(video_latents)

    print(f"videos.shape: {videos.shape}")
    print(f"video_latents.shape: {video_latents.shape}")
    print(f"decode_videos.shape: {decode_videos.shape}")
    return video_latents


def test_transformer3d(args, video_latents, device, dtype):
    batch_size = video_latents.shape[0]

    transformer3d = WanTransformer3DModel.from_pretrained(
        args.wanva.model_path,
        subfolder="transformer",
        patch_size=args.wanva.patch_size,
        num_attention_heads=args.wanva.num_attention_heads,
        video_tokens=args.projector.num_token,
        attn_mode=args.wanva.attn_mode,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )

    target_modules = ["video_embedder.ff", "action_embedder.ff"]
    for name, module in transformer3d.named_modules():
        if (
            any(target in name for target in target_modules)
            and isinstance(module, torch.nn.Linear)
            and torch.isnan(module.weight).any()
        ):
            overwatch.warning(f"Reinitializing: f{name}")
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    for name, param in transformer3d.named_parameters():
        if torch.isnan(param).any():
            overwatch.error(f"NaN param: {name}, {param.shape}, {param.device}")

    transformer3d = transformer3d.to(device=device, dtype=dtype)

    grid_id = get_mesh_id(
        f=video_latents.shape[-3] // args.wanva.patch_size[0],
        h=video_latents.shape[-2] // args.wanva.patch_size[1],
        w=video_latents.shape[-1] // args.wanva.patch_size[2],
        f_w=1,
        f_shift=0,
        action=False,
    ).unsqueeze(0)
    grid_id = torch.cat([grid_id] * batch_size, dim=0).to(device=device, dtype=dtype)

    timestep = torch.ones((batch_size), dtype=torch.float32, device=device) * 0

    encoder_hidden_states_video = torch.randn(
        (batch_size, args.projector.num_token, args.projector.output_align_dim), device=device, dtype=dtype
    )

    hidden_states_video, _ = transformer3d(
        grid_id=grid_id,
        timestep=timestep,
        hidden_states_video=video_latents,
        encoder_hidden_states_video=encoder_hidden_states_video,
    )

    print(f"hidden_states_video.shape: {hidden_states_video.shape}")


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis

    from univam.utils.args import load_args
    from univam.utils.data import VideoData, set_seed

    args = load_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    batch_size = 2

    # get real data via Dataset
    data = VideoData(args.data)
    data.video_paths = ["tests/examples/lingbot.mp4"]
    video = data.read_video_torchcodec(0, 0)
    video = video.unsqueeze(0)
    videos = torch.cat([video] * batch_size, dim=0).to(device=device, dtype=dtype)

    # video_latents = test_vavae(args, videos, device, dtype)

    # test_transformer3d(args, video_latents, device, dtype)

    # >>> start main test for Wan22VisionActionModel <<<
    model = Wan22VisionActionModel(args).to(device=device, dtype=dtype)
    model = FlopsWrapper(model)

    total_params = sum(p.numel() for p in model.parameters())

    # train part
    model.train()
    train_outputs = model(videos)
    train_flops = FlopCountAnalysis(model, videos).total()

    # eval part
    # model.eval()
    # eval_outputs = model(videos)
    # eval_flops = FlopCountAnalysis(model, videos).total()

    print(">>>>> general part <<<<<")
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Inputs shape: {videos.shape}")

    print(">>>>> train part <<<<<<")
    print(f"FLOPs: {train_flops / 1e9:.2f} GFLOPs")
    print(f"Loss: {train_outputs['loss']}")

    # print(">>>>> eval part <<<<<")
    # print(f"FLOPs: {eval_flops / 1e9:.2f} GFLOPs")
    # print(f"Output shape: {eval_outputs['videos'].shape}")
    # >>> end main test for Wan22VisionActionModel <<<
