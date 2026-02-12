import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from flash_attn import flash_attn_func


def custom_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)


def get_mesh_id(f, h, w, t, f_w=1, f_shift=0, action=False):
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
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)
    return grid_id


class VAVAE(nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.video_vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")

        self.pad_num = 0
        self.pad_chunk_size = 4

        self.register_buffer("latent_mean", torch.tensor(self.video_vae.config.latents_mean).view(1, -1, 1, 1, 1))
        self.register_buffer("latent_std", torch.tensor(self.video_vae.config.latents_std).view(1, -1, 1, 1, 1))

    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.dtype = next(self.parameters()).dtype
        return model_converted

    def align_video(self, videos):
        """
        尾部重复 padding, 使 T 满足 1 + N * pad_chunk_size
        """
        T = videos.shape[2]
        self.pad_num = 0

        remainder = (T - 1) % self.pad_chunk_size
        if remainder != 0:
            self.pad_num = self.pad_chunk_size - remainder
            last_frame = videos[:, :, -1:, :, :]  # [B, C, 1, H, W]
            pad_frames = last_frame.repeat(1, 1, self.pad_num, 1, 1)
            videos = torch.cat([videos, pad_frames], dim=2)

        return videos

    def inverse_align_video(self, latents):
        """
        去掉尾部 padding
        """
        latents = latents[:, :, : -self.pad_num, :, :]
        return latents

    def encode(self, videos: torch.Tensor, actions: torch.Tensor | None = None):
        """
        Joint VAE encode, assume that actions ~ N(?, ?)
            videos:  [B, C, T, H, W]
            actions: [B, C, T, chunk_size, 1]
        """
        videos = self.align_video(videos)
        video_latents = self.video_vae.encode(videos).latent_dist.sample()

        mean = self.latent_mean.to(video_latents.dtype)
        std = self.latent_std.to(video_latents.dtype)

        video_latents = (video_latents - mean) / std
        video_latents = video_latents.to(dtype=self.dtype)

        if actions is None:
            return video_latents
        else:
            action_latents = actions.to(dtype=self.dtype)
            return video_latents, action_latents

    def decode(self, video_latents: torch.Tensor, action_latents: torch.Tensor | None = None):
        """
        Joint VAE decode, assume that actions ~ N(?, ?)
            videos:  [B, C, F, H, W]
            actions: [B, C, F, chunk_size, 1]
        """
        mean = self.latent_mean.to(video_latents.dtype)
        std = self.latent_std.to(video_latents.dtype)

        video_latents = video_latents * std + mean
        video_latents = video_latents.to(dtype=self.dtype)
        videos = self.video_vae.decode(video_latents, return_dict=False)[0]

        videos = self.inverse_align_video(videos)

        if action_latents is None:
            return videos
        else:
            actions = action_latents.to(dtype=self.dtype)
            return videos, actions


class VideoEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_video: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            encoder_hidden_states_video = encoder_hidden_states_video + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_video)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


# TODO: modify to align action
class ActionEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_action: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_action.shape
            encoder_hidden_states_action = encoder_hidden_states_action.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_action = encoder_hidden_states_action + self.pos_embed

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
        video_tokens: Optional[int] = None,
        action_tokens: Optional[int] = None,
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
            pos_embed_seq_len=video_tokens,
        )

        # action_embedder
        self.action_embedder = None
        if action_embed_dim is not None:
            self.image_embedder = ActionEmbedding(action_embed_dim, dim, pos_embed_seq_len=action_tokens)

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
        elif attn_mode == "falsh_attention_2":
            self.attn_op = flash_attn_func
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
        video_tokens: Optional[int] = None,
        action_dim: Optional[int] = None,
        action_tokens: Optional[int] = None,
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
        self.patch_embedding_mlp = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], inner_dim)

        # 2. Condition embeddings
        self.condition_embedder = TimeVideoActionEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            video_embed_dim=video_dim,
            video_tokens=video_tokens,
            action_embed_dim=action_dim,
            action_tokens=action_tokens,
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
        timestep: torch.LongTensor,
        # TODO deside to cat action tokens or not
        hidden_states_video: torch.Tensor,
        encoder_hidden_states_video: torch.Tensor,
        hidden_states_action: Optional[torch.Tensor] = None,
        encoder_hidden_states_action: Optional[torch.Tensor] = None,
    ):
        B, C, T, H, W = hidden_states_video.shape
        hidden_states_video = rearrange(
            hidden_states_video,
            "b c (t p1) (h p2) (w p3) -> b (t h w) (c p1 p2 p3)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
        )

        hidden_states = hidden_states_video

        if hidden_states_action is not None:
            # TODO follow the action encoder/tokenizer rules
            hidden_states_action = rearrange(
                hidden_states_action,
                "b c f h w -> b (f h w) c",
            )
            hidden_states_action = self.action_embedder(hidden_states_action)
            hidden_states = torch.cat([hidden_states, hidden_states_action], dim=1)
            action_token_nums = hidden_states_action[1]

        hidden_states = self.patch_embedding_mlp(hidden_states)  # [B, tokens, dim]

        rotary_emb = self.rope(grid_id)  # [B, tokens, (H//p2)*(W//p3)]
        rotary_emb = rotary_emb[:, :, None]  # [B, tokens, 1, (H//p2)*(W//p3)]

        # latent_time_steps = torch.repeat_interleave(
        #     timesteps,
        #     (H // self.patch_size[1]) * (W // self.patch_size[2]),
        #     dim=1,
        # )  # [B, tokens]

        temb, timestep_proj, encoder_hidden_states_video, encoder_hidden_states_action = self.condition_embedder(
            timestep=timestep,
            encoder_hidden_states_video=encoder_hidden_states_video,
            encoder_hidden_states_action=encoder_hidden_states_action,
        )
        
        timestep_proj = timestep_proj[:, None, :]                            # [B, 1, 6*inner_dim]
        timestep_proj = timestep_proj.expand(-1, hidden_states.shape[1], -1) # [B, tokens, 6*inner_dim]
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states_video,
                timestep_proj,
                rotary_emb,
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, None, :]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
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
            hidden_states = self.proj_out(hidden_states)
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

        self.vae = VAVAE(config.model_path)


if __name__ == "__main__":
    from einops import rearrange

    from univam.utils.args import load_args
    from univam.utils.data import VideoData

    args = load_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    va_vae = VAVAE(args.wanva.model_path).to(device)

    # get real data via Dataset
    data = VideoData(args.data)
    data.video_paths = ["tests/examples/lingbot.mp4"]
    video = data.read_video_torchcodec(0, 0)
    video = video.unsqueeze(0)
    videos = torch.cat([video] * 1, dim=0)
    videos = rearrange(videos, "b f c h w -> b c f h w")

    video_latents = va_vae.encode(videos)
    decode_videos = va_vae.decode(video_latents)

    print(f"videos.shape: {videos.shape}")
    print(f"video_latents.shape: {video_latents.shape}")
    print(f"decode_videos.shape: {decode_videos.shape}")

    DiT, loading_info = WanTransformer3DModel.from_pretrained(
        args.wanva.model_path,
        subfolder="transformer",
        video_tokens=8,
        output_loading_info=True,
        low_cpu_mem_usage=False,
    )
    
    DiT = DiT.to(device)

    grid_id = get_mesh_id(
        f=video_latents.shape[-3] // 1,
        h=video_latents.shape[-2] // 2,
        w=video_latents.shape[-1] // 2,
        t=0,
        f_w=1,
        f_shift=0,
        action=False,
    ).unsqueeze(0)

    timesteps = torch.ones([video_latents.shape[2]], dtype=torch.float32, device=device) * 0

    timestep = torch.ones((video_latents.shape[0]), dtype=torch.float32, device=device) * 0

    encoder_hidden_states_video = torch.randn((1, 8, 4096), device=device)

    hidden_states_video, hidden_states_action = DiT(
        grid_id=grid_id,
        timestep=timestep,
        hidden_states_video=video_latents,
        encoder_hidden_states_video=encoder_hidden_states_video,
    )

    print(f"hidden_states_video.shape: {hidden_states_video.shape}")

    # from diffusers import WanPipeline

    # pipe = WanPipeline.from_pretrained(args.wanva.model_path)
    # pipe(
    #     num_frames=5,
    #     height=256,
    #     width=256,
    #     prompt_embeds=torch.randn(1, 8, 4096),
    #     negative_prompt_embeds=torch.randn(1, 8, 4096),
    # )
    # print(1)
