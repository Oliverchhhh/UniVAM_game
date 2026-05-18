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


class Wan22VisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # Keep VAE out of nn.Module tree so ZeRO-3 won't partition it.
        # Its internal temporal feature caching is incompatible with parameter sharding.
        wanvae = WanVAE(
            config.wanva.model_path,
            frames=config.data.frames,
        )
        self.__dict__["wanvae"] = wanvae  # bypass nn.Module.__setattr__
        vae_hw = self.wanvae.vae.config.scale_factor_spatial

        height, width = config.data.image_size
        latent_t_num = self.wanvae.latent_t_num

        vae_height = height // vae_hw
        vae_width = width // vae_hw

        self.patch_size = config.wanva.patch_size
        self.proj_patch_size = config.projector.patch_size

        self.transformer3d = WanTransformer3DModel.from_pretrained(
            config.wanva.model_path,
            subfolder="transformer",
            patch_size=config.wanva.patch_size,
            num_attention_heads=config.wanva.num_attention_heads,
            text_dim=config.projector.output_align_dim,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
        )
        self.transformer3d.init_weights()  # also materializes meta params from custom modules

        self._use_lora = getattr(config, "lora", None) is not None and config.lora.enable
        if self._use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.lora_alpha,
                target_modules=[
                    "attn1.to_q",
                    "attn1.to_k",
                    "attn1.to_v",
                    "attn1.to_out.0",
                    "attn2.to_q",
                    "attn2.to_k",
                    "attn2.to_v",
                    "attn2.to_out.0",
                ],
                modules_to_save=[
                    "condition_embedder",
                    "patch_embedding",
                    "proj_out",
                ],
            )
            self.transformer3d = get_peft_model(self.transformer3d, lora_config)
            overwatch.warning(
                f"LoRA enabled: r={config.lora.r}, alpha={config.lora.lora_alpha}, "
                f"trainable params={sum(p.numel() for p in self.transformer3d.parameters() if p.requires_grad):,}"
            )

        self.vae_proj = nn.Conv3d(
            self.wanvae.vae.config.z_dim,
            config.projector.output_align_dim,
            kernel_size=self.proj_patch_size,
            stride=self.proj_patch_size,
        )

        pt, ph, pw = self.proj_patch_size
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
        self.wanvae.to(*args, **kwargs)  # VAE is not a registered submodule
        self.device = next(self.parameters()).device
        self.dtype = next(self.transformer3d.parameters()).dtype
        return model_converted

    def train(self, *args):
        super().train(*args)
        self.set_trainable_params()

    def set_trainable_params(self):
        self.transformer3d.train()
        if not self._use_lora:
            self.transformer3d.requires_grad_(True)

        self.vae_proj.train()
        self.vae_proj.requires_grad_(True)

        self.projector.train()
        self.projector.requires_grad_(True)

        self.wanvae.eval()
        self.wanvae.requires_grad_(False)

    def _save_ckpt(self, model_dict: Dict, projector_dict: Dict, save_path: str, global_step: int) -> None:
        if self._use_lora:
            # Wan22VM.pth: only Wan22VisionModel-level trainable params (vae_proj).
            # Frozen transformer base weights are reloaded from from_pretrained.
            _outer_exclude = ("wanvae", "projector", "transformer3d")
            base_dict = {}
            for k, v in model_dict.items():
                if not any(k.startswith(p) for p in _outer_exclude):
                    base_dict[k] = v
            torch.save({"model": base_dict, "global_step": global_step},
                       os.path.join(save_path, "Wan22VM.pth"))

            # LoraAdapter.pth: LoRA low-rank + modules_to_save.
            # Use model_dict keys (have full .default suffix), strip transformer3d.
            # prefix so they match PeftModel internal keys on load.
            lora_state = {}
            for k, v in model_dict.items():
                if any(p in k for p in ("lora_", "original_module", "modules_to_save")):
                    lora_state[k[len("transformer3d."):]] = v.cpu()
            torch.save(lora_state, os.path.join(save_path, "LoraAdapter.pth"))
        else:
            # Full training: filter out wanvae/projector, save rest
            exclude_prefixes = ["wanvae", "projector"]
            base_dict = {}
            for k, v in model_dict.items():
                if any(k.startswith(prefix) for prefix in exclude_prefixes):
                    continue
                base_dict[k] = v
            torch.save({"model": base_dict, "global_step": global_step},
                       os.path.join(save_path, "Wan22VM.pth"))

        torch.save(projector_dict, os.path.join(save_path, "Projector.pth"))

    def _load_ckpt(self, load_path: str) -> int:
        assert os.path.exists(os.path.join(load_path, "Projector.pth")), f"Projector.pth not found in {load_path}"

        ckpt_name = "Wan22VM.pth"
        ckpt_path = os.path.join(load_path, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_name} not found in {load_path}")

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

        wanvam_ckpt = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(wanvam_ckpt["model"], strict=False)
        _log_missing_unexpected("Wan22VM", missing, unexpected)

        projector_ckpt = torch.load(os.path.join(load_path, "Projector.pth"), map_location="cpu")
        missing, unexpected = self.projector.load_state_dict(projector_ckpt, strict=False)
        _log_missing_unexpected("Projector", missing, unexpected)

        if self._use_lora:
            lora_path = os.path.join(load_path, "LoraAdapter.pth")
            if os.path.exists(lora_path):
                lora_state = torch.load(lora_path, map_location="cpu")
                # LoraAdapter.pth only contains LoRA + modules_to_save (subset).
                self.transformer3d.load_state_dict(lora_state, strict=False)
                overwatch.warning(f"Loaded LoRA adapter from {lora_path}")
            else:
                overwatch.warning(
                    f"LoRA enabled but no LoraAdapter.pth found in {load_path}; "
                    "LoRA weights will be randomly initialized"
                )

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
        video_latents = self.vae_proj(video_latents)
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

        # video flow matching
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

        video_pred_latents = self.transformer3d(
            timestep=video_timesteps,
            hidden_states=video_noisy_latents,
            encoder_hidden_states=video_embeds,
        ).sample
        check_tensor(video_pred_latents, "video_pred_latents", check_bound=100, check_std=10)

        video_loss = self.video_scheduler.calculate_loss(
            video_pred_latents,
            video_target,
            timestep=video_timesteps,
            timestep_id=video_timestep_id,
        )

        outputs["loss"] = video_loss
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

        video_latents = self.prepare_latents(
            shape=video_latents.shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

        with self.progress_bar(total=self.num_inference_steps, use_tqdm=use_tqdm) as progress_bar:
            for t in timesteps:
                video_latent_input = torch.cat([video_latents] * 2) if do_classifier_free_guidance else video_latents

                timestep = t.expand(video_latent_input.shape[0])
                video_noise_pred = self.transformer3d(
                    timestep=timestep,
                    hidden_states=video_latent_input,
                    encoder_hidden_states=video_embeds,
                ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = video_noise_pred.chunk(2)
                    video_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                video_latents = self.eval_scheduler.step(video_noise_pred, t, video_latents)

                progress_bar.update()

        video_latents = video_latents.to(dtype=self.dtype)
        gen_videos = self.wanvae.decode(video_latents)

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

    # >>> start main test for Wan22VisionModel <<<
    model = Wan22VisionModel(args).to(device=device, dtype=dtype)
    model = FlopsWrapper(model)

    total_params = sum(p.numel() for p in model.parameters())

    # train part
    model.train()
    train_outputs = model(videos)
    train_flops = FlopCountAnalysis(model, (videos,)).total()

    # eval part
    model.eval()
    eval_outputs = model(videos)
    eval_flops = FlopCountAnalysis(model, (videos,)).total()

    print(">>>>> general part <<<<<")
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Inputs shape: {videos.shape}")

    print(">>>>> train part <<<<<<")
    print(f"FLOPs: {train_flops / 1e9:.2f} GFLOPs")
    print(f"Loss: {train_outputs['loss']}")

    print(">>>>> eval part <<<<<")
    # print(f"FLOPs: {eval_flops / 1e9:.2f} GFLOPs")
    print(f"Output shape: {eval_outputs['videos'].shape}")
    # >>> end main test for Wan22VisionModel <<<
