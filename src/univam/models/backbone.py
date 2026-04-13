from typing import Any, Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusers import SD3Transformer2DModel
from diffusers.utils import is_torch_version
from einops import rearrange


class DiTConditionHead(nn.Module):
    def __init__(self, pooled_dim: int = 2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4096, pooled_dim),
            nn.SiLU(),
            nn.Linear(pooled_dim, pooled_dim),
        )
        self.norm = nn.LayerNorm(normalized_shape=pooled_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, compress_tokens: torch.Tensor):
        # [bsz, num_token, 4096]
        pooled_embeds = compress_tokens.mean(dim=1)
        pooled_embeds = self.mlp(pooled_embeds)
        pooled_embeds = self.norm(pooled_embeds)
        return pooled_embeds


class SD3TransformerBackbone(SD3Transformer2DModel):
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
    ):
        super().__init__(
            sample_size,
            patch_size,
            in_channels,
            num_layers,
            attention_head_dim,
            num_attention_heads,
            joint_attention_dim,
            caption_projection_dim,
            pooled_projection_dim,
            out_channels,
            pos_embed_max_size,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        add_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
    ) -> torch.FloatTensor:
        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        hidden_states_len = hidden_states.shape[1]
        if add_hidden_states is not None:
            add_hidden_states = self.pos_embed(add_hidden_states)
            hidden_states = torch.concat([hidden_states, add_hidden_states], dim=1)

        temb = self.time_text_embed(timestep, pooled_projections)  # timestep
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states[:, :hidden_states_len]

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )

        return output


class VisionBackbone(nn.Module):
    def __init__(
        self,
        img_size: List[int] = [224, 224],
        model_name: str = "vit_large_patch14_dinov2",
        pretrained: bool = True,
        local_ckpt: str = None,
        out_indices: Tuple[int] = (-1,),
    ) -> None:
        super().__init__()
        if local_ckpt:
            cfg = {"file": local_ckpt, "input_size": (3, *img_size)}
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                features_only=True,
                out_indices=out_indices,
                pretrained_cfg_overlay=cfg,
            )
        else:
            cfg = {"input_size": (3, *img_size)}
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                pretrained_cfg_overlay=cfg,
            )

        assert hasattr(self.model, "feature_info"), (
            "Could not infer vision backbone output channels. Ensure timm version supports features_only=True"
        )

        self.transforms = self.get_transforms()
        self.reduction = self.model.feature_info.reduction()[0]
        self.patches = (img_size[0] // self.reduction) * (img_size[1] // self.reduction)
        self.channels = self.model.feature_info.channels()[-1]

    def get_transforms(self) -> T.Normalize:
        mean, std = self.model.default_cfg["mean"], self.model.default_cfg["std"]
        transforms = T.Normalize(mean, std)
        return transforms

    def forward(self, images: torch.Tensor):
        features = self.model(images)[-1]
        features = rearrange(features, "b c h w -> b (h w) c")
        return features


if __name__ == "__main__":
    import os
    from pathlib import Path

    from fvcore.nn import FlopCountAnalysis

    from univam.utils.args import load_args

    args = load_args()

    model_name = "vit_large_patch16_dinov3.lvd1689m"
    local_ckpt = Path(os.environ.get("PRETRAINED_MODEL_PATH", "./models")) / "timm" / model_name / "pytorch_model.bin"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = VisionBackbone(
        img_size=args.data.image_size,
        model_name=model_name,
        local_ckpt=local_ckpt,
        pretrained=False,
    ).to(device)

    images = torch.randn(1, 3, *args.data.image_size)
    features = backbone(images)
    total_params = sum(p.numel() for p in backbone.parameters())
    flops = FlopCountAnalysis(backbone, images).total()

    print(f"Model name: {model_name}")
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Inputs shape: {images.shape}")
    print(f"Features shape: {features.shape}")
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
