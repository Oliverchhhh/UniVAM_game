"""WoG-style future representation learning on top of a frozen NitroGen VA model."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .actions import ACTION_DIM, ACTION_HORIZON


def _dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype {name!r}") from exc


def _load_wog_future_encoder_class(wog_root: Path):
    """Load WoG's two vision files without importing its TensorFlow/RLDS stack."""
    vision_dir = wog_root / "prismatic" / "models" / "backbones" / "vision"
    package_paths = {
        "prismatic": wog_root / "prismatic",
        "prismatic.models": wog_root / "prismatic" / "models",
        "prismatic.models.backbones": wog_root / "prismatic" / "models" / "backbones",
        "prismatic.models.backbones.vision": vision_dir,
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    def load(name: str, path: Path):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load("prismatic.models.backbones.vision.base_vision", vision_dir / "base_vision.py")
    load("prismatic.models.backbones.vision.vae", vision_dir / "vae.py")
    future_module = load("prismatic.models.backbones.vision.dinosiglip_vit", vision_dir / "dinosiglip_vit.py")
    return future_module.FutureEnc


class FutureQFormer(nn.Module):
    """18 learned queries cross-attend to pooled DINOv2 and Wan-VAE tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_layers: int = 6,
        num_queries: int = ACTION_HORIZON,
        output_dim: int = 64,
        dino_dim: int = 1024,
        wan_dim: int = 784,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dino_proj = nn.Identity() if dino_dim == hidden_dim else nn.Linear(dino_dim, hidden_dim)
        self.wan_proj = nn.Linear(wan_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, hidden_dim))
        self.query_time_embedding = nn.Parameter(torch.empty(1, num_queries, hidden_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.query_time_embedding, std=0.02)

    def forward(self, dino_tokens: torch.Tensor, wan_tokens: torch.Tensor) -> torch.Tensor:
        memory = torch.cat([self.dino_proj(dino_tokens), self.wan_proj(wan_tokens)], dim=1)
        queries = self.query_tokens + self.query_time_embedding
        queries = queries.expand(memory.shape[0], -1, -1)
        return self.output_proj(self.decoder(tgt=queries, memory=memory))


class FrozenWoGFutureEncoder(nn.Module):
    """Frozen DINOv2 (future frames) and Wan VAE (current+future clip)."""

    def __init__(self, wog_root: str | Path, dtype: str = "bfloat16", pool_size: int = 8) -> None:
        super().__init__()
        wog_root = Path(wog_root).resolve()
        FutureEnc = _load_wog_future_encoder_class(wog_root)
        self.encoder = FutureEnc(default_image_size=224)
        self.pool_size = pool_size
        target_dtype = _dtype(dtype)
        self.encoder.dino_featurizer.to(dtype=target_dtype)
        # WanVAE is a wrapper rather than nn.Module, so FutureEnc.to() cannot move it.
        self.encoder.vae.dtype = target_dtype
        self.encoder.vae.model.to(dtype=target_dtype)
        self.encoder.vae.mean = self.encoder.vae.mean.to(dtype=target_dtype)
        self.encoder.vae.std = self.encoder.vae.std.to(dtype=target_dtype)
        self.encoder.vae.scale = [self.encoder.vae.mean, 1.0 / self.encoder.vae.std]
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Frozen encoders must never enable dropout/stat updates when the parent trains.
        super().train(False)
        self.encoder.dino_featurizer.eval()
        self.encoder.vae.model.eval()
        return self

    @torch.no_grad()
    def forward(self, frames_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frames_01.shape[1] != 5:
            raise ValueError(f"Expected five keyframes, got {tuple(frames_01.shape)}")
        bsz = frames_01.shape[0]
        frames = F.interpolate(
            frames_01.flatten(0, 1), size=(224, 224), mode="bicubic", align_corners=False
        ).view(bsz, 5, 3, 224, 224)
        mean = frames.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = frames.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        normalized = (frames - mean) / std
        base = {"dino": normalized[:, 0]}
        future = {"dino": normalized[:, 1:].flatten(0, 1)}
        dino, wan = self.encoder(future, base)

        _, patches, dim = dino.shape
        side = int(patches**0.5)
        if side * side != patches:
            raise RuntimeError(f"DINO token count {patches} is not square")
        dino = dino.view(bsz * 4, side, side, dim).permute(0, 3, 1, 2)
        dino = F.adaptive_avg_pool2d(dino, (self.pool_size, self.pool_size))
        dino = dino.permute(0, 2, 3, 1).reshape(bsz, 4 * self.pool_size**2, dim)
        return dino.detach(), wan.detach()


class ConditionCrossAttentionAdapter(nn.Module):
    """Low-rank residual cross-attention from Action-DiT tokens to future tokens."""

    def __init__(self, model_dim: int = 1024, condition_dim: int = 64, inner_dim: int = 256, heads: int = 8):
        super().__init__()
        if inner_dim % heads:
            raise ValueError("inner_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = inner_dim // heads
        self.norm = nn.LayerNorm(model_dim)
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.q_proj = nn.Linear(model_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(condition_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(condition_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, model_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = x.shape
        return x.view(bsz, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.q_proj(self.norm(hidden)))
        condition = self.condition_norm(condition)
        key = self._heads(self.k_proj(condition))
        value = self._heads(self.v_proj(condition))
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).flatten(2)
        return hidden + torch.tanh(self.gate) * self.out_proj(attended)


def load_frozen_nitrogen(
    checkpoint_path: str | Path,
    nitrogen_root: str | Path,
    siglip_path: str | Path,
    device: torch.device,
    dtype: str = "bfloat16",
):
    """Load the released checkpoint without the inference-only no-grad wrapper."""
    nitrogen_root = Path(nitrogen_root).resolve()
    if str(nitrogen_root) not in sys.path:
        sys.path.insert(0, str(nitrogen_root))
    os.environ["SIGLIP_LOCAL_PATH"] = str(Path(siglip_path).resolve())
    # Some shared servers export DEBUG=release, while NitroGen parses DEBUG as int.
    if not os.environ.get("DEBUG", "0").isdigit():
        os.environ["DEBUG"] = "0"
    from nitrogen.cfg import CkptConfig
    from nitrogen.flow_matching_transformer.nitrogen import NitroGen

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True)
    ckpt_config = CkptConfig.model_validate(checkpoint["ckpt_config"])
    if ckpt_config.model_cfg.action_horizon != ACTION_HORIZON or ckpt_config.model_cfg.action_dim != ACTION_DIM:
        raise ValueError(
            f"Expected NitroGen action shape 18x25, got "
            f"{ckpt_config.model_cfg.action_horizon}x{ckpt_config.model_cfg.action_dim}"
        )
    model = NitroGen(config=ckpt_config.model_cfg, game_mapping=None)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.requires_grad_(False).eval().to(device=device, dtype=_dtype(dtype))
    return model, ckpt_config


class FutureConditionedNitroGen(nn.Module):
    """Frozen NitroGen with trainable adapters after its four visual-attention blocks."""

    def __init__(
        self,
        nitrogen: nn.Module,
        condition_dim: int = 64,
        adapter_dim: int = 256,
        adapter_heads: int = 8,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.nitrogen = nitrogen
        self.gradient_checkpointing = gradient_checkpointing
        n_cross = sum(
            not (idx % 2 == 1 and nitrogen.model.config.interleave_self_attention)
            for idx in range(len(nitrogen.model.transformer_blocks))
        )
        self.adapters = nn.ModuleList(
            [ConditionCrossAttentionAdapter(1024, condition_dim, adapter_dim, adapter_heads) for _ in range(n_cross)]
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.nitrogen.eval()
        self.adapters.train(mode)
        return self

    def _current_visual(self, current_pixels: torch.Tensor) -> torch.Tensor:
        ng = self.nitrogen
        with torch.no_grad():
            vision = ng.encode_images(current_pixels[:, None])
            # One non-dropped image and no game/separator tokens: prepare_input_embs
            # reduces exactly to this flattening operation.
            vl_embs = vision.flatten(1, 2)
            return ng.vl_self_attention_model(vl_embs)

    def _run_block(self, block, hidden, visual, temb, is_self_attention: bool):
        context = None if is_self_attention else visual

        def fn(h, t):
            return block(
                h,
                attention_mask=None,
                encoder_hidden_states=context,
                encoder_attention_mask=None,
                temb=t,
            )

        if self.gradient_checkpointing and self.training and hidden.requires_grad:
            return activation_checkpoint(fn, hidden, temb, use_reentrant=False)
        return fn(hidden, temb)

    def flow_loss(
        self,
        current_pixels: torch.Tensor,
        actions: torch.Tensor,
        actions_mask: torch.Tensor,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ng = self.nitrogen
        bsz = actions.shape[0]
        visual = self._current_visual(current_pixels)
        noise = torch.randn_like(actions) if noise is None else noise
        if time is None:
            time = ng.sample_time(bsz, actions.device, actions.dtype)
        broadcast_t = time[:, None, None]
        noisy = (1 - broadcast_t) * noise + broadcast_t * actions
        target_velocity = actions - noise
        discrete_t = (time * ng.num_timestep_buckets).long()

        embodiment = torch.zeros(bsz, dtype=torch.long, device=actions.device)
        with torch.no_grad():
            hidden = ng.action_encoder(noisy, discrete_t, embodiment)
            if ng.config.add_pos_embed:
                positions = torch.arange(ACTION_HORIZON, device=actions.device)
                hidden = hidden + ng.position_embedding(positions)[None].to(hidden.dtype)

        dit = ng.model
        temb = dit.timestep_encoder(discrete_t)
        adapter_idx = 0
        for idx, block in enumerate(dit.transformer_blocks):
            is_self = idx % 2 == 1 and dit.config.interleave_self_attention
            hidden = self._run_block(block, hidden, visual, temb, is_self)
            if not is_self:
                hidden = self.adapters[adapter_idx](hidden, condition)
                adapter_idx += 1

        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden = dit.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        model_output = dit.proj_out_2(hidden)
        prediction = ng.action_decoder(model_output, embodiment)[:, -ACTION_HORIZON:]

        element_loss = F.mse_loss(prediction, target_velocity, reduction="none")
        mask = actions_mask.to(element_loss.dtype)
        loss = (element_loss * mask).sum() / mask.sum().clamp_min(1)
        return loss, {"noise": noise.detach(), "time": time.detach(), "prediction": prediction.detach()}


class Stage1FutureConditionModel(nn.Module):
    def __init__(
        self,
        *,
        nitrogen_checkpoint: str,
        nitrogen_root: str,
        siglip_path: str,
        wog_root: str,
        device: torch.device,
        dtype: str = "bfloat16",
        condition_dim: int = 64,
        qformer_hidden: int = 1024,
        qformer_layers: int = 6,
        adapter_dim: int = 256,
        condition_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.condition_dropout = condition_dropout
        self.compute_dtype = _dtype(dtype)
        self.future_encoder = FrozenWoGFutureEncoder(wog_root, dtype=dtype)
        self.qformer = FutureQFormer(
            hidden_dim=qformer_hidden,
            num_layers=qformer_layers,
            num_queries=ACTION_HORIZON,
            output_dim=condition_dim,
        )
        nitrogen, self.ckpt_config = load_frozen_nitrogen(
            nitrogen_checkpoint, nitrogen_root, siglip_path, device, dtype
        )
        self.policy = FutureConditionedNitroGen(
            nitrogen,
            condition_dim=condition_dim,
            adapter_dim=adapter_dim,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.to(device=device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.future_encoder.train(False)
        self.policy.nitrogen.eval()
        return self

    def preprocess(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames = frames.to(dtype=self.compute_dtype) / 255.0
        current = F.interpolate(frames[:, 0], size=(256, 256), mode="bicubic", align_corners=False)
        current = (current - 0.5) / 0.5
        return current, frames

    def encode_condition(self, frames_01: torch.Tensor) -> torch.Tensor:
        dino, wan = self.future_encoder(frames_01)
        return self.qformer(dino.to(self.compute_dtype), wan.to(self.compute_dtype))

    def forward(self, frames: torch.Tensor, actions: torch.Tensor, actions_mask: torch.Tensor) -> torch.Tensor:
        current, frames_01 = self.preprocess(frames)
        condition = self.encode_condition(frames_01)
        if self.training and self.condition_dropout > 0:
            keep = torch.rand(condition.shape[0], 1, 1, device=condition.device) >= self.condition_dropout
            condition = condition * keep
        loss, _ = self.policy.flow_loss(
            current, actions.to(self.compute_dtype), actions_mask, condition
        )
        return loss

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            "qformer": self.qformer.state_dict(),
            "adapters": self.policy.adapters.state_dict(),
        }

    def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
        self.qformer.load_state_dict(state["qformer"])
        self.policy.adapters.load_state_dict(state["adapters"])

    def parameter_report(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "nitrogen_frozen": sum(p.numel() for p in self.policy.nitrogen.parameters()),
            "future_encoder_frozen": sum(p.numel() for p in self.future_encoder.parameters()),
        }
