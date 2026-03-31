# Action Expert Model
# DiT-style architecture with cross-attention injection from video model


import numpy as np
import torch
import torch.nn as nn
from diffusers.models.normalization import FP32LayerNorm

from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    """
    Get 1D positional embedding in the form of sin and cos.

    Args:
        embed_dim (int): output dimension for each position.
        pos (ndarray | tensor): a list of positions to be encoded, size (M,).
    Returns:
        out (tensor): resulting positional embedding, size (M, D).
    """
    import numpy as np

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return torch.from_numpy(emb).float()


def generate_compress_dims(in_features, out_features, num_layers, reverse=False):
    if reverse:
        in_features, out_features = out_features, in_features
    start_exp = in_features.bit_length() - 1
    end_exp = out_features.bit_length()
    exps = np.linspace(start_exp, end_exp, num_layers)
    exps = np.ceil(exps).astype(int)
    dims = [2**e for e in exps]
    if reverse:
        return dims[::-1]
    return dims


class ActionEncoder(nn.Module):
    """Encoder for action-only sequences (no state)."""

    def __init__(self, config, out_features):
        super().__init__()
        self.in_features = config.action_dim
        self.out_features = out_features
        self.num_layers = config.num_layers

        self.hidden_dims = generate_compress_dims(
            in_features=self.in_features,
            out_features=self.out_features,
            num_layers=config.num_layers - 3,
            reverse=True,
        )

        self.action_encoder = self.build_mlp()

        # Positional embeddings for action tokens
        max_seq_len = config.chunk_size
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.out_features, np.arange(max_seq_len))
        self.register_buffer("pos_embedding", pos_embed.unsqueeze(0))

    def build_mlp(self):
        modules = [nn.Linear(self.in_features, self.hidden_dims[0])]
        for i in range(self.num_layers - 4):
            modules.append(nn.SiLU())
            modules.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
        modules.append(nn.SiLU())
        modules.append(nn.Linear(self.hidden_dims[-1], self.out_features))
        modules.append(nn.SiLU())
        modules.append(nn.Linear(self.out_features, self.out_features))
        return nn.Sequential(*modules)

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        encoded = self.action_encoder(action_tokens)  # [B, chunk_size, dim]
        seq_len = encoded.shape[1]
        encoded = encoded + self.pos_embedding[:, :seq_len, :]
        return encoded


class ActionDecoder(nn.Module):
    """Final layer to decode action predictions."""

    def __init__(self, config, in_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = config.action_dim
        self.num_layers = 3

        self.hidden_dims = generate_compress_dims(
            in_features=self.in_features,
            out_features=self.out_features,
            num_layers=self.num_layers - 1,
        )

        self.action_head = self.build_mlp()

        self.norm = FP32LayerNorm(self.out_features, eps=1e-6, elementwise_affine=False)

    def build_mlp(self):
        modules = [nn.Linear(self.in_features, self.hidden_dims[0])]
        for i in range(self.num_layers - 2):
            modules.append(nn.SiLU())
            modules.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
        modules.append(nn.SiLU())
        modules.append(nn.Linear(self.hidden_dims[-1], self.out_features))
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode action predictions.

        Args:
            x: Features [B, chunk_size, dim]

        Returns:
            Action predictions [B, chunk_size, action_dim]
        """
        x = self.action_head(x)
        return self.norm(x)


if __name__ == "__main__":
    from univam.utils.args import load_args
    from univam.utils.data import set_seed
    from univam.utils.dataloaders.hdf5 import EpisodeData

    args = load_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    batch_size = 2
    hidden_dim = 3072

    # get real data via Dataset
    data = EpisodeData(args.data, 24)
    data.video_paths = ["tests/examples/sample_episode.hdf5"]
    data.dataset_name = ["XVLA"]
    video, action, timestep = data.read_episode(0, 0)

    video = video.unsqueeze(0)
    videos = torch.cat([video] * batch_size, dim=0).to(device=device, dtype=dtype)
    action = action.unsqueeze(0)
    actions = torch.cat([action] * batch_size, dim=0).to(device=device, dtype=dtype)
    actions = actions.reshape(batch_size, -1, actions.shape[-1])
    actions = actions[:, :-1, :]

    action_encoder = ActionEncoder(args.action, hidden_dim)
    action_latents = action_encoder(actions)

    action_decoder = ActionDecoder(args.action, hidden_dim)
    actions_decoded = action_decoder(action_latents)

    print(f"actions.shape: {actions.shape}")
    print(f"action_latents.shape: {action_latents.shape}")
    print(f"actions_decoded.shape: {actions_decoded.shape}")
