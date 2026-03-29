import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock

from univam.utils.data import check_tensor


class MLPProjector(nn.Module):
    def __init__(self, args, patches: int, channels: int) -> None:
        super().__init__()

        self.patches = patches
        self.channels = channels

        self.hidden_dim: int = args.hidden_dim
        self.cross_attention_dim: int = args.cross_attention_dim
        self.output_align_dim: int = args.output_align_dim

        self.num_token: int = args.num_token
        self.num_attn_layers: int = args.num_attn_layers
        self.num_attn_compress_layers: int = args.num_attn_compress_layers
        self.compress_dims = self._generate_compress_dims()

        self.compress_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=1,
                    bias=True,
                )
                for in_dim, out_dim in zip(self.compress_dims[:-1], self.compress_dims[1:])
            ]
        )
        self.compress_layers.append(
            nn.Conv1d(
                in_channels=self.compress_dims[-1],
                out_channels=self.compress_dims[-1],
                kernel_size=1,
                bias=True,
            )
        )
        self.attn_layers = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.channels,
                    num_attention_heads=8,
                    attention_head_dim=self.channels // 8,
                    dropout=0.1,
                    cross_attention_dim=self.channels,
                )
                for _ in range(self.num_attn_layers)
            ]
        )
        self.attn_compress_layers = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.hidden_dim,
                    num_attention_heads=8,
                    attention_head_dim=self.cross_attention_dim // 8,
                    dropout=0.1,
                    cross_attention_dim=self.cross_attention_dim,
                )
                for _ in range(self.num_attn_compress_layers)
            ]
        )
        self.qkv_layer = nn.Linear(self.channels, self.hidden_dim + self.cross_attention_dim)
        self.output_align_mlp = nn.Linear(self.hidden_dim, self.output_align_dim)

        self.compress_align_conv = nn.Conv1d(
            in_channels=self.patches,
            out_channels=self.compress_dims[0],
            kernel_size=1,
            bias=True,
        )

        self.compress_norm = nn.LayerNorm(normalized_shape=self.hidden_dim, eps=1e-6, elementwise_affine=False)
        self.norm = nn.LayerNorm(normalized_shape=self.output_align_dim, eps=1e-6, elementwise_affine=False)

    def _generate_compress_dims(self):
        start_exp = self.patches.bit_length() - 1
        end_exp = self.num_token.bit_length() - 1
        exps = np.linspace(start_exp, end_exp, self.num_attn_compress_layers)
        exps = np.ceil(exps).astype(int)
        dims = [2**e for e in exps]
        return dims

    def forward(self, image_embeddings: torch.Tensor):
        hidden_states = image_embeddings.clone()
        for transformer_block in self.attn_layers:
            hidden_states = transformer_block(
                hidden_states=hidden_states,
                encoder_hidden_states=image_embeddings,
            )
            check_tensor(hidden_states, "self-attn")

        hidden_states = self.qkv_layer(hidden_states)
        q = hidden_states[:, :, : self.hidden_dim]

        encoder_hidden_states = hidden_states[:, :, self.hidden_dim :]
        hidden_states = self.compress_align_conv(q)

        for compress_block, transformer_block in zip(self.compress_layers, self.attn_compress_layers):
            hidden_states = transformer_block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = self.compress_norm(hidden_states)
            hidden_states = compress_block(hidden_states)
            check_tensor(hidden_states, "compressor")

        compressed_embeds = self.output_align_mlp(hidden_states)
        compressed_embeds = self.norm(compressed_embeds)

        return compressed_embeds


class QformerBlock(nn.Module):
    """Qformer block: Self-Attn on Q tokens + Cross-Attn with image embeddings"""

    def __init__(self, dim, num_heads, cross_dim):
        super().__init__()
        self.cross_attn = BasicTransformerBlock(
            dim=dim,
            num_attention_heads=num_heads,
            attention_head_dim=dim // num_heads,
            dropout=0.1,
            cross_attention_dim=cross_dim,
            norm_elementwise_affine=False,
            norm_eps=1e-7,
        )

    def forward(self, q_tokens, image_embeddings):
        # self attention (in diffusers) + cross attention
        q_tokens = self.cross_attn(
            hidden_states=q_tokens,
            encoder_hidden_states=image_embeddings,
        )

        return q_tokens


class QformerProjector(nn.Module):
    def __init__(self, args, patches: int, channels: int) -> None:
        super().__init__()

        self.patches = patches
        self.channels = channels

        self.hidden_dim = args.hidden_dim
        self.output_align_dim = args.output_align_dim

        self.num_query_token = args.num_token
        self.num_attn_layers = args.num_attn_layers

        self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_token, self.hidden_dim))

        self.qformer_layers = nn.ModuleList(
            [
                QformerBlock(
                    dim=self.hidden_dim,
                    num_heads=8,
                    cross_dim=self.channels,
                )
                for _ in range(self.num_attn_layers)
            ]
        )

        self.output_align_mlp = nn.Linear(self.hidden_dim, self.output_align_dim)

        self.norm = nn.LayerNorm(self.output_align_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, image_embeddings: torch.Tensor):
        bsz = image_embeddings.size(0)

        q_tokens = self.query_tokens.expand(bsz, -1, -1)

        for idx, block in enumerate(self.qformer_layers):
            q_tokens = block(q_tokens, image_embeddings)
            check_tensor(q_tokens, f"qformer{idx}", check_bound=1e4, check_std=5e3)

        compressed_embeds = self.output_align_mlp(q_tokens)
        compressed_embeds = self.norm(compressed_embeds)

        return compressed_embeds


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis

    from univam.utils.args import load_args

    args = load_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patches = 2 * 15 * 20
    channels = 24 * 128

    if args.projector.type == "mlp":
        model = MLPProjector(
            args.projector,
            patches=patches,
            channels=channels,
        ).to(device)
        print(f"Token compression process: {model.compress_dims}")
    elif args.projector.type == "qformer":
        model = QformerProjector(
            args.projector,
            patches=patches,
            channels=channels,
        ).to(device)
    else:
        raise ValueError(f"Unknown projector type '{args.projector.type}'. ")

    batch_size = 2

    video_pooler_feature = torch.randn(batch_size, patches, channels).to(device)

    compressed_embeds = model(video_pooler_feature)

    total_params = sum(p.numel() for p in model.parameters())
    flops = FlopCountAnalysis(model, video_pooler_feature).total()

    print(f"Model type: {args.projector.type}")
    print(f"Video_pooler_feature shape: {video_pooler_feature.shape}")
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Compressed_embeds shape: {compressed_embeds.shape}")
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
