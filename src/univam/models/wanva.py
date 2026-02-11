import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan


class VAVAE(nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.video_vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")

        self.register_buffer("latent_mean", torch.tensor(self.video_vae.config.latents_mean).view(1, -1, 1, 1, 1))
        self.register_buffer("latent_std", torch.tensor(self.video_vae.config.latents_std).view(1, -1, 1, 1, 1))

    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.dtype = next(self.parameters()).dtype
        return model_converted

    def encode(self, videos: torch.Tensor, actions: torch.Tensor | None = None):
        """
        Joint VAE encode, assume that actions ~ N(?, ?)
        """
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
        """
        mean = self.latent_mean.to(video_latents.dtype)
        std = self.latent_std.to(video_latents.dtype)

        video_latents = video_latents * std + mean
        video_latents = video_latents.to(dtype=self.dtype)
        videos = self.video_vae.decode(video_latents, return_dict=False)[0]

        if action_latents is None:
            return videos
        else:
            actions = action_latents.to(dtype=self.dtype)
            return videos, actions


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
