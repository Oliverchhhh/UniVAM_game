import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from univam.models.action import ActionDecoder, ActionEncoder
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
        vae_hw = self.wanvae.vae.config.scale_factor_spatial

        height, width = config.data.image_size
        latent_t_num = self.wanvae.latent_t_num

        vae_height = height // vae_hw
        vae_width = width // vae_hw

        self.patch_size = config.wanva.patch_size
        self.projector_patch_size = config.projector.patch_size

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
            config.projector.output_align_dim,
            kernel_size=self.projector_patch_size,
            stride=self.projector_patch_size,
        )

        pt, ph, pw = self.projector_patch_size
        wt, wh, ww = self.patch_size = config.wanva.patch_size
        assert latent_t_num % (pt * wt) == 0, f"{latent_t_num=} must be divisible by {pt*wt=}"

        assert height % (ph * vae_hw * wh) == 0, f"{height=} must be divisible by {ph*vae_hw*wh=}"

        assert width % (pw * vae_hw * ww) == 0, f"{width=} must be divisible by {pw*vae_hw*ww=}"

        vae_num_tokens = (latent_t_num // pt) * (vae_height // ph) * (vae_width // pw)

        if config.projector.type == "mlp":
            self.projector = MLPProjector(
                config.projector,
                patches=vae_num_tokens,
                channels=config.projector.output_align_dim,
            )
        elif config.projector.type == "qformer":
            self.projector = QformerProjector(
                config.projector,
                patches=vae_num_tokens,
                channels=config.projector.output_align_dim,
            )
        else:
            raise ValueError(f"Unknown projector type '{config.projector.type}'. ")

        self.gamma = getattr(config.train, "gamma", 10)

        self.action_encoder = ActionEncoder(config.action, self.transformer3d.inner_dim)
        self.action_decoder = ActionDecoder(config.action, self.transformer3d.inner_dim)

        self.seed = getattr(config, "seed", 33)
        self.guidance_scale = getattr(config.wanva, "guidance_scale", 1.0)
        self.num_inference_steps = getattr(config.wanva, "num_inference_steps", 100)

        self.video_scheduler = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        )
        self.video_scheduler.set_timesteps(num_inference_steps=1000, training=True)

        self.action_scheduler = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        )
        self.action_scheduler.set_timesteps(num_inference_steps=1000, training=True)

        self.eval_scheduler = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_inference_steps=config.wanva.num_inference_steps,
        )

        self.token_dropout = getattr(config.wanva, "token_dropout", False)
        self.num_token = config.projector.num_token

        self.train_with_action = False

    def set_train_mode(self, use_action=False):
        self.train_with_action = use_action

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
        exclude_prefixes = ["wanvae", "projector"]
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

    def prepare_latents(self, shape, dtype, device, generator, latents=None):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != shape[0]:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {shape[0]}. Make sure the batch size matches the length of the generators."
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
        actions: Optional[torch.Tensor] = None
        if self.train_with_action and inputs.get("actions", None) is not None:
            actions = inputs["actions"]
            actions = actions.reshape(actions.shape[0], -1, actions.shape[-1])
            actions = actions[:, :-1, :]

        video_latents = self.wanvae.encode(videos)
        video_latents = video_latents.to(dtype=self.dtype)

        video_embeds = self.encode(video_latents)

        # video part flow matching
        video_noise = torch.randn_like(video_latents, dtype=self.dtype)

        video_timestep_id = sample_timestep_id(
            batch_size=videos.shape[0],
            num_train_timesteps=self.video_scheduler.num_train_timesteps,
        )
        video_timesteps = self.video_scheduler.timesteps[video_timestep_id].to(dtype=self.dtype, device=self.device)

        video_noisy_latents = self.video_scheduler.add_noise(
            video_latents, video_noise, video_timesteps, video_timestep_id
        )

        video_target = self.video_scheduler.training_target(video_latents, video_noise, video_timesteps)

        # action part flow matching
        action_timesteps = None
        action_noisy_latents = None
        if actions is not None:
            action_noise = torch.randn_like(actions, dtype=self.dtype)

            action_timestep_id = sample_timestep_id(
                batch_size=actions.shape[0],
                num_train_timesteps=self.action_scheduler.num_train_timesteps,
            )
            action_timesteps = self.action_scheduler.timesteps[action_timestep_id].to(
                dtype=self.dtype, device=self.device
            )

            action_noisy_latents = self.action_scheduler.add_noise(
                actions, action_noise, action_timesteps, action_timestep_id
            )

            action_target = self.action_scheduler.training_target(actions, action_noise, action_timesteps)

            action_noisy_latents = self.action_encoder(action_noisy_latents)

        video_pred_latents, action_pred_latents = self.transformer3d(
            video_timestep=video_timesteps,
            action_timestep=action_timesteps,
            video_hidden_states=video_noisy_latents,
            action_hidden_states=action_noisy_latents,
            encoder_hidden_states=video_embeds,
        )
        check_tensor(video_pred_latents, "video_pred_latents", check_bound=100, check_std=10)

        video_loss = self.video_scheduler.calculate_loss(
            video_pred_latents,
            video_target,
            timestep=video_timesteps,
            timestep_id=video_timestep_id,
        )

        if actions is not None:
            check_tensor(action_pred_latents, "action_pred_latents", check_bound=100, check_std=10)
            action_pred_latents = self.action_decoder(action_pred_latents)
            action_loss = self.action_scheduler.calculate_loss(
                action_pred_latents,
                action_target,
                timestep=action_timesteps,
                timestep_id=action_timestep_id,
            )
            loss = video_loss + self.gamma * action_loss
            outputs["loss_video"] = video_loss
            outputs["loss_action"] = action_loss
            outputs["loss"] = loss
            return outputs

        outputs["loss"] = video_loss
        return outputs

    @torch.no_grad()
    def eval_step(self, inputs: Dict[str, Any], outputs: Dict[str, Any], use_tqdm: bool = True) -> Dict[str, Any]:
        videos = inputs["videos"]
        generator = inputs["generator"]
        actions: Optional[torch.Tensor] = None
        if self.train_with_action and inputs.get("actions", None) is not None:
            actions = inputs["actions"]
            actions = actions.reshape(actions.shape[0], -1, actions.shape[-1])
            actions = actions[:, :-1, :]

        video_latents = self.wanvae.encode(videos)

        do_classifier_free_guidance = self.guidance_scale > 1.0
        video_embeds = self.encode(video_latents, do_classifier_free_guidance=do_classifier_free_guidance)

        self.eval_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.eval_scheduler.timesteps

        timesteps = timesteps.to(self.device)

        video_latents = self.prepare_latents(
            shape=video_latents.shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        action_latents = None
        if actions is not None:
            action_latents = self.prepare_latents(
                shape=actions.shape,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )
            action_latents = self.action_encoder(action_latents)

        with self.progress_bar(total=self.num_inference_steps, use_tqdm=use_tqdm) as progress_bar:
            for t in timesteps:
                video_latent_input = torch.cat([video_latents] * 2) if do_classifier_free_guidance else video_latents
                action_latent_input = None
                if actions is not None:
                    action_latent_input = (
                        torch.cat([action_latents] * 2) if do_classifier_free_guidance else action_latents
                    )

                timestep = t.expand(video_latent_input.shape[0])
                video_noise_pred, action_noise_pred = self.transformer3d(
                    video_timestep=timestep,
                    action_timestep=timestep,
                    video_hidden_states=video_latent_input,
                    action_hidden_states=action_latent_input,
                    encoder_hidden_states=video_embeds,
                )

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = video_noise_pred.chunk(2)
                    video_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if actions is not None:
                        noise_pred_uncond, noise_pred_text = action_noise_pred.chunk(2)
                        action_noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                video_latents = self.eval_scheduler.step(video_noise_pred, t, video_latents)
                if action_latents is not None:
                    action_latents = self.eval_scheduler.step(action_noise_pred, t, action_latents)

                progress_bar.update()

        video_latents = video_latents.to(dtype=self.dtype)
        gen_videos = self.wanvae.decode(video_latents)
        if actions is not None:
            action_latents = action_latents.to(dtype=self.dtype)
            pred_actions = self.action_decoder(action_latents)
            outputs["actions"] = pred_actions
            outputs["input_actions"] = actions

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

    def forward(self, videos, actions=None, **kwargs):
        inputs = {"videos": videos, "actions": actions}
        return self.model(inputs, **kwargs)


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis

    from univam.utils.args import load_args
    from univam.utils.data import set_seed
    from univam.utils.dataloaders.hdf5 import EpisodeData

    args = load_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    batch_size = 2

    # get real data via Dataset
    data = EpisodeData(args.data, 24)
    data.video_paths = ["tests/examples/sample_episode.hdf5"]
    data.dataset_name = ["XVLA"]
    video, action, timestep = data.read_episode(0, 0)

    video = video.unsqueeze(0)
    videos = torch.cat([video] * batch_size, dim=0).to(device=device, dtype=dtype)
    action = action.unsqueeze(0)
    actions = torch.cat([action] * batch_size, dim=0).to(device=device, dtype=dtype)

    # >>> start main test for Wan22VisionActionModel <<<
    model = Wan22VisionActionModel(args).to(device=device, dtype=dtype)
    model.set_train_mode(use_action=True)
    model = FlopsWrapper(model)

    total_params = sum(p.numel() for p in model.parameters())

    # train part
    model.train()
    train_outputs = model(videos, actions)
    train_flops = FlopCountAnalysis(model, (videos, actions)).total()

    # eval part
    model.eval()
    eval_outputs = model(videos, actions)
    eval_flops = FlopCountAnalysis(model, (videos, actions)).total()

    print(">>>>> general part <<<<<")
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Inputs shape: {videos.shape}")

    print(">>>>> train part <<<<<<")
    print(f"FLOPs: {train_flops / 1e9:.2f} GFLOPs")
    print(f"Loss: {train_outputs['loss']}")

    print(">>>>> eval part <<<<<")
    # print(f"FLOPs: {eval_flops / 1e9:.2f} GFLOPs")
    print(f"Output shape: {eval_outputs['videos'].shape}")
    # >>> end main test for Wan22VisionActionModel <<<
