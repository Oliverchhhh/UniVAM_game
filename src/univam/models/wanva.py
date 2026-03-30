import os
from typing import Any, Dict

import torch
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from univam.models.projector import MLPProjector, QformerProjector
from univam.models.scheduler import FlowMatchScheduler
from univam.models.wan import WanTransformer3DModel, WanVAE
from univam.utils.data import check_tensor
from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def sample_timestep_id(
    batch_size,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    num_train_timesteps: int = 1000,
    device: torch.device = torch.device("cpu"),
):
    u = torch.rand(size=[batch_size], device=device)
    u = u * (max_timestep_bd - min_timestep_bd) + min_timestep_bd
    timestep_id = (u * num_train_timesteps).clamp(min=0, max=num_train_timesteps - 1).to(torch.int64)
    return timestep_id


class Wan22VisionActionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.wanvae = WanVAE(
            config.wanva.model_path,
            frames=config.data.frames,
        )
        # self.vae_scale_factor_spatial = self.wanvae.vae.config.scale_factor_spatial
        # self.vae_scale_factor_temporal = self.wanvae.vae.config.scale_factor_temporal

        height, width = config.data.image_size
        self.latent_t_num = self.wanvae.latent_t_num
        self.num_channels_latents = self.wanvae.vae.config.z_dim

        self.vae_height = height // self.wanvae.vae.config.scale_factor_spatial
        self.vae_width = width // self.wanvae.vae.config.scale_factor_spatial

        self.patch_size = config.wanva.patch_size

        self.transformer3d = WanTransformer3DModel.from_pretrained(
            config.wanva.model_path,
            subfolder="transformer",
            patch_size=config.wanva.patch_size,
            num_attention_heads=config.wanva.num_attention_heads,
            video_dim=config.projector.output_align_dim,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )
        self.transformer3d.init_weights()

        self.patch_embedding = nn.Conv3d(
            self.wanvae.vae.config.z_dim,
            self.transformer3d.inner_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        patches = (
            (self.latent_t_num // self.patch_size[0])
            * (self.vae_height // self.patch_size[1])
            * (self.vae_width // self.patch_size[2])
        )

        if config.projector.type == "mlp":
            self.projector = MLPProjector(
                config.projector,
                patches=patches,
                channels=self.transformer3d.inner_dim,
            )
        elif config.projector.type == "qformer":
            self.projector = QformerProjector(
                config.projector,
                patches=patches,
                channels=self.transformer3d.inner_dim,
            )
        else:
            raise ValueError(f"Unknown projector type '{config.projector.type}'. ")

        self.seed = getattr(config, "seed", 33)
        self.guidance_scale = getattr(config.wanva, "guidance_scale", 1.0)
        self.num_inference_steps = getattr(config.wanva, "num_inference_steps", 100)

        self.scheduler = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        )
        self.scheduler.set_timesteps(num_inference_steps=1000, training=True)

        self.eval_scheduler = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_inference_steps=config.wanva.num_inference_steps,
        )

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

        self.wanvae.eval()
        self.wanvae.requires_grad_(False)

    def _save_ckpt(self, model_dict: Dict, projector_model_dict: Dict, save_path: str, global_step: int) -> None:
        exclude_prefixes = ["wanvae", "projector", "video_feature_extractor"]
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
            def extract_top_level(keys, k=1):
                if k <= 0:
                    raise ValueError("k must be >= 1")
                return sorted({".".join(key.split(".")[:k]) for key in keys})

            top_missing = extract_top_level(missing_keys, k=1)
            top_unexpected = extract_top_level(unexpected_keys, k=2)

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
            self.latent_t_num,
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

    def encode(self, video_latents: torch.Tensor, do_classifier_free_guidance: bool = False):
        dtype = next(self.projector.parameters()).dtype
        video_latents = video_latents.to(device=self.device, dtype=dtype)
        video_latents = self.patch_embedding(video_latents)
        video_latents = video_latents.flatten(2).transpose(1, 2)

        video_embeds = self.projector(video_latents)
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

        video_latents = self.wanvae.encode(videos)
        video_latents = video_latents.to(dtype=self.dtype)

        video_embeds = self.encode(video_latents)

        video_noise = torch.randn_like(video_latents, dtype=self.dtype)

        timestep_id = sample_timestep_id(
            batch_size=videos.shape[0],
            num_train_timesteps=self.scheduler.num_train_timesteps,
        )
        timesteps = self.scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)

        video_noisy_latents = self.scheduler.add_noise(video_latents, video_noise, timesteps, timestep_id)

        target = self.scheduler.training_target(video_latents, video_noise, timesteps)

        model_pred_video_latents, _ = self.transformer3d(
            video_timestep=timesteps,
            video_hidden_states=video_noisy_latents,
            encoder_hidden_states=video_embeds,
        )
        check_tensor(model_pred_video_latents, "model_pred_video_latents", check_bound=100, check_std=10)

        loss = self.scheduler.calculate_loss(
            model_pred_video_latents,
            target,
            timestep=timesteps,
            timestep_id=timestep_id,
        )

        outputs["loss"] = loss
        return outputs

    @torch.no_grad()
    def eval_step(self, inputs: Dict[str, Any], outputs: Dict[str, Any], use_tqdm: bool = True) -> Dict[str, Any]:
        videos = inputs["videos"]
        generator = inputs["generator"]

        video_latents = self.wanvae.encode(videos)

        do_classifier_free_guidance = self.guidance_scale > 1.0
        video_embeds = self.encode(video_latents, do_classifier_free_guidance=do_classifier_free_guidance)

        self.eval_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.eval_scheduler.timesteps

        timesteps = timesteps.to(self.device)

        latents = self.prepare_latents(
            batch_size=videos.shape[0],
            dtype=self.dtype,
            device=self.device,
            generator=generator,
            latents=None,
        )

        with self.progress_bar(total=self.num_inference_steps, use_tqdm=use_tqdm) as progress_bar:
            for t in timesteps:
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                timestep = t.expand(latent_model_input.shape[0])
                noise_pred_video, _ = self.transformer3d(
                    video_timestep=timestep,
                    video_hidden_states=latent_model_input,
                    encoder_hidden_states=video_embeds,
                )

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred_video.chunk(2)
                    noise_pred_video = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.eval_scheduler.step(noise_pred_video, t, latents)

                progress_bar.update()

        latents = latents.to(dtype=self.dtype)
        gen_videos = self.wanvae.decode(latents)

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


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis

    from univam.utils.args import load_args
    from univam.utils.data import set_seed
    from univam.utils.dataloaders.video import VideoData

    args = load_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    batch_size = 2

    # get real data via Dataset
    data = VideoData(args.data)
    data.video_paths = ["tests/examples/lingbot.mp4"]
    video = data.read_video_decord(0, 0)
    video = video.unsqueeze(0)
    videos = torch.cat([video] * batch_size, dim=0).to(device=device, dtype=dtype)

    # >>> start main test for Wan22VisionActionModel <<<
    model = Wan22VisionActionModel(args).to(device=device, dtype=dtype)
    model = FlopsWrapper(model)

    total_params = sum(p.numel() for p in model.parameters())

    # train part
    model.train()
    train_outputs = model(videos)
    train_flops = FlopCountAnalysis(model, videos).total()

    # eval part
    model.eval()
    eval_outputs = model(videos)
    eval_flops = FlopCountAnalysis(model, videos).total()

    print(">>>>> general part <<<<<")
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Inputs shape: {videos.shape}")

    print(">>>>> train part <<<<<<")
    print(f"FLOPs: {train_flops / 1e9:.2f} GFLOPs")
    print(f"Loss: {train_outputs['loss']}")

    print(">>>>> eval part <<<<<")
    print(f"FLOPs: {eval_flops / 1e9:.2f} GFLOPs")
    print(f"Output shape: {eval_outputs['videos'].shape}")
    # >>> end main test for Wan22VisionActionModel <<<
