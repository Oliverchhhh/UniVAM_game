import os
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as DIST
import torchvision.transforms as T
from accelerate import Accelerator
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from univam.models.wanva import Wan22VisionModel
from univam.utils.data import check_tensor, complex_to_device, fp32_to_bf16, move_to_cuda
from univam.utils.files import ensure_directory, ensure_dirname
from univam.utils.metrics import Meter, Timer, calculate_psnr, calculate_ssim, get_parameters
from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


class Trainer:
    def __init__(self, args, model: Wan22VisionModel, optimizer=None, lr_scheduler=None) -> None:
        self.model: Wan22VisionModel = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.local_rank = overwatch.local_rank()
        self.rank = overwatch.rank()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.epoch = -1
        self.global_step = -1

        self.eval_before_train = True

        self.resume = args.resume
        self.resume_path = args.resume_path
        self.do_train = args.do_train

        self.num_iters = args.train.num_iters
        self.epochs = args.train.epochs
        self.eval_step = args.train.eval_step
        self.save_step = args.train.save_step
        self.local_batch_size = args.train.local_batch_size
        self.gradient_accumulate_steps = args.train.gradient_accumulate_steps
        self.iter_per_ep = None

        self.seed = args.seed
        self.task_name = args.task_name
        self.fps = args.data.fps
        self.image_size = args.data.image_size
        self.log_dir = os.path.join(args.log_dir, args.task_name)
        self.ckpt_save_dir = os.path.join(args.train.ckpt_save_dir, args.task_name)

        if overwatch.is_rank_zero() and args.do_train:
            ensure_directory(self.log_dir)
            ensure_directory(self.ckpt_save_dir)

        OmegaConf.resolve(args)
        if overwatch.is_rank_zero():
            OmegaConf.save(args, os.path.join(self.ckpt_save_dir, "config.yaml"))

    def move_model_to_device(self) -> None:
        self.model.to(self.device)
        if self.optimizer is not None:
            if isinstance(self.optimizer, list):
                for i in range(len(self.optimizer)):
                    self.optimizer[i].load_state_dict(
                        complex_to_device(self.optimizer[i].state_dict(), device=self.device)
                    )
            else:
                self.optimizer.load_state_dict(complex_to_device(self.optimizer.state_dict(), device=self.device))

    def prepare_dist_model(self) -> None:
        self.accelerator = Accelerator(
            log_with="tensorboard",
            mixed_precision="bf16",
            project_dir=self.log_dir,
            gradient_accumulation_steps=self.gradient_accumulate_steps,
        )
        self.accelerator.init_trackers("train")
        # self.accelerator.even_batches = False
        if overwatch.is_rank_zero():
            self.writer = self.accelerator.get_tracker("tensorboard").writer

        self.device = self.accelerator.device

        # ensure lr scheduler is checkpointed by accelerate
        if self.lr_scheduler is not None:
            self.accelerator.register_for_checkpointing(self.lr_scheduler)

        if not self.do_train:
            self.model.eval()
        overwatch.info(f"Successfully built models with {get_parameters(self.model)} parameters")

        if self.resume:
            assert os.path.exists(self.resume_path)

            overwatch.warning(f"Resuming from {self.resume_path}")
            self.load_checkpoint(self.resume_path)

    def forward_step(self, inputs, **kwargs) -> Dict[str, Any]:
        outputs = self.model(inputs, **kwargs)
        return outputs

    def backward_step(self, loss) -> None:
        if hasattr(self, "accelerator") and self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def prepare_batch(self, batch) -> Dict[str, Any]:
        batch = move_to_cuda(batch)
        batch = fp32_to_bf16(batch)
        return batch

    def step(self, optimizer_idx=-1) -> None:
        if hasattr(self, "accelerator") and self.accelerator is not None:
            if not self.accelerator.sync_gradients:
                return

        if optimizer_idx >= 0 and isinstance(self.optimizer, list):
            optimizer = self.optimizer[optimizer_idx]
        else:
            optimizer = self.optimizer

        optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        optimizer.zero_grad()

    def reduce_mean(self, v) -> float:
        world_size = overwatch.world_size()
        if world_size < 2:
            return v
        else:
            t = v.clone().detach().to(self.device)
            if hasattr(self, "accelerator") and self.accelerator is not None:
                gathered = self.accelerator.gather(t)
                return gathered.float().mean().item()
            else:
                DIST.all_reduce(t)
                t = t.float() / world_size
                return t.item()

    def save_checkpoint(self) -> None:
        save_path = os.path.join(self.ckpt_save_dir, str(self.global_step))
        overwatch.warning(f"Saving checkpoint to {save_path}")

        self.accelerator.wait_for_everyone()
        # get_state_dict is a collective operation; all ranks must participate.
        # With ZeRO-3 it may return None on non-zero ranks.
        full_state_dict = self.accelerator.get_state_dict(self.model)
        model_dict = {}
        projector_dict = {}
        if full_state_dict is not None:
            for k, v in full_state_dict.items():
                if k.startswith("projector."):
                    projector_dict[k[len("projector.") :]] = v
                else:
                    model_dict[k] = v

        if overwatch.is_rank_zero():
            ensure_directory(save_path)
            self.model._save_ckpt(model_dict, projector_dict, save_path, self.global_step)

        # Save optimizer / scheduler / RNG state for full training resume.
        # accelerate handles both normal and DeepSpeed ZeRO formats internally.
        # train_state_dir = os.path.join(save_path, "train_state")
        # self.accelerator.save_state(train_state_dir, safe_serialization=False)

    def load_checkpoint(self, load_path) -> None:
        global_step = self.model._load_ckpt(load_path)
        self.global_step = global_step

    def _resume_training_state(self) -> None:
        """Load optimizer, scheduler and RNG state. Must be called after ``prepare()``."""
        train_state_dir = os.path.join(self.resume_path, "train_state")
        if os.path.isdir(train_state_dir):
            self.accelerator.load_state(train_state_dir)
            overwatch.warning(f"Resumed training state from {train_state_dir}")
        else:
            overwatch.warning(
                f"No training state found at {train_state_dir}. "
                "Optimizer and scheduler will be initialized from scratch (model weights loaded, this is fine)."
            )

    def setup_model_for_training(self) -> None:
        if overwatch.is_rank_zero():
            overwatch.warning(f"Existing dirs detected {self.log_dir}")
            ensure_dirname(self.log_dir, override=False)

        self.model.set_trainable_params()
        self.prepare_dist_model()

    def train_eval_by_iter(self, train_loader, eval_loader=None, use_tqdm=True) -> None:
        self.model, self.optimizer, train_loader = self.accelerator.prepare(self.model, self.optimizer, train_loader)

        if self.resume:
            self._resume_training_state()

        if self.num_iters is not None:
            overwatch.warning("Start train & val phase...")
        else:
            overwatch.warning("Skip train & val phase...")
            return
        overwatch.warning(
            f"Train examples: {len(train_loader.dataset)},\n"
            f"Val examples: {len(eval_loader.dataset)}, {len(eval_loader)}\n"
            f"epochs: {self.epochs}, iters: {self.num_iters}, \n"
            f"eval_step: {self.eval_step}, save_step: {self.save_step},\n"
            f"global_batch_size: {self.local_batch_size * overwatch.world_size() * self.gradient_accumulate_steps}, local_batch_size: {self.local_batch_size}."
        )

        # Train & Eval phase
        train_pbar = tqdm(total=self.num_iters, disable=not use_tqdm, ncols=150, dynamic_ncols=False)
        train_meter = Meter()

        if self.global_step > 0:
            train_pbar.update(self.global_step)
        else:
            self.global_step = 0

        if self.eval_before_train:
            if eval_loader:
                eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
                overwatch.info(f"[Rank {self.rank}] Valid before train. Time: {eval_time}\n{eval_meter.avg}")

        self.model.train()

        last_log_time = time.time()

        while True:
            train_iter = iter(train_loader)
            while self.global_step < self.num_iters:
                try:
                    inputs = next(train_iter)
                except StopIteration:
                    # overwatch.warning("Reaching end of the train_loader, terminating training loop")
                    break

                self.epoch = (self.global_step + 1) // self.iter_per_ep

                with self.accelerator.accumulate(self.model):
                    inputs = self.prepare_batch(inputs)
                    check_tensor(inputs, "inputs(prepare_batch)")
                    outputs = self.forward_step(inputs)
                    check_tensor(outputs["loss"], "loss", check_bound=10, check_std=10)
                    self.backward_step(outputs["loss"])
                    self.step()

                if self.accelerator.sync_gradients:
                    loss_to_log = outputs["loss"].item()

                    metric_and_loss = {k: v for k, v in outputs.items() if k.split("_")[0] in ["metric", "loss"]}
                    for k, v in metric_and_loss.items():
                        metric_and_loss[k] = self.reduce_mean(v)
                    train_meter.update(metric_and_loss)
                    train_pbar.set_description("Metering: " + str(train_meter))

                    self.accelerator.log(metric_and_loss, step=self.global_step)

                    self.global_step += 1
                    train_pbar.update(1)

                    if self.global_step % 10 == 0:
                        current_time = time.time()
                        elapsed = current_time - last_log_time
                        avg_time_per_step = elapsed / 10.0 if self.global_step > 0 else 0

                        current_lr = self.lr_scheduler.get_lr()
                        if isinstance(current_lr, list):
                            current_lr = current_lr[0]

                        if overwatch.is_rank_zero():
                            overwatch.info(
                                f"Step: {self.global_step}/{self.num_iters} | "
                                f"Loss: {loss_to_log:.4f} | "
                                f"LR: {current_lr:.2e} | "
                                f"Time/Step: {avg_time_per_step:.4f}s"
                            )

                        last_log_time = current_time

                    if self.global_step % self.save_step == 0 and self.global_step != 0:
                        overwatch.warning("Saving model...")
                        self.save_checkpoint()

                    if self.global_step % self.eval_step == 0 and self.global_step != 0:
                        overwatch.warning("Evaluating...")
                        if eval_loader:
                            # sample a single round training sample to test whether over-fitting
                            eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
                            overwatch.info(
                                f"[Rank {self.rank}] Valid Step: {self.global_step}, Time: {eval_time}\n{eval_meter.avg}"
                            )
                        # torch.cuda.empty_cache()

                        # Update metric with eval metrics
                        train_meter = Meter()

                        last_log_time = time.time()

            if self.global_step >= self.num_iters:
                break

        if self.global_step % self.save_step != 0:
            overwatch.warning("Saving model...")
            self.save_checkpoint()

        if self.global_step % self.eval_step != 0:
            overwatch.warning("Evaluating...")
            if eval_loader:
                eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
                overwatch.info(
                    f"[Rank {self.rank}] Valid Step: {self.global_step}, Time: {eval_time}\n{eval_meter.avg}"
                )

    def eval_fn(self, eval_loader, use_tqdm=True):
        self.model.eval()
        eval_meter = Meter()
        eval_timer = Timer()

        label_videos = []
        pred_videos = []

        # ensure all ranks iterate the same number of batches to prevent gather deadlock
        # when dataset size is not divisible by world_size and drop_last=False
        n_batches = len(eval_loader)
        if DIST.is_initialized() and self.accelerator.num_processes > 1:
            n_batches_tensor = torch.tensor([n_batches], device=self.accelerator.device)
            DIST.all_reduce(n_batches_tensor, op=DIST.ReduceOp.MIN)
            n_batches = int(n_batches_tensor.item())

        with torch.no_grad():
            if overwatch.is_rank_zero():
                eval_loader = tqdm(eval_loader, total=n_batches, ncols=150, dynamic_ncols=False)
            for batch_idx, inputs in enumerate(eval_loader):
                if batch_idx >= n_batches:
                    break
                inputs = self.prepare_batch(inputs)
                outputs = self.forward_step(inputs, use_tqdm=use_tqdm)
                metric_and_loss = {k: v for k, v in outputs.items() if k.split("_")[0] in ["metric", "loss"]}

                for k, v in metric_and_loss.items():
                    metric_and_loss[k] = self.reduce_mean(v)
                eval_meter.update(metric_and_loss)

                label_video = inputs["videos"]
                pred_video = outputs["videos"]

                label_video = torch.clamp((label_video + 1) / 2, 0, 1)
                pred_video = torch.clamp((pred_video + 1) / 2, 0, 1)

                label_videos.append(label_video)
                pred_videos.append(pred_video)

            label_videos = torch.cat(label_videos, dim=0)
            pred_videos = torch.cat(pred_videos, dim=0)

            psnr = calculate_psnr(pred_videos, label_videos)
            ssim = calculate_ssim(pred_videos, label_videos)

            psnr = self.reduce_mean(psnr)
            ssim = self.reduce_mean(ssim)

            eval_meter.update(
                {
                    "val/psnr": psnr,
                    "val/ssim": ssim,
                }
            )

            overwatch.info(f"PSNR: {eval_meter.avg['val/psnr']:.4f}")
            overwatch.info(f"SSIM: {eval_meter.avg['val/ssim']:.4f}")
            # overwatch.info(f"rFID: {calculate_rfid(pred_imgs, label_imgs):.4f}")

            if overwatch.is_rank_zero():
                self.accelerator.log(
                    {
                        "val/psnr": eval_meter.avg["val/psnr"],
                        "val/ssim": eval_meter.avg["val/ssim"],
                    },
                    step=self.global_step,
                )

                video_path = os.path.join(self.log_dir, "videos", str(self.global_step))
                gt_video_path = os.path.join(video_path, "gt")
                pred_video_path = os.path.join(video_path, "pred")
                ensure_directory(video_path)
                ensure_directory(gt_video_path)
                ensure_directory(pred_video_path)

                to_tensor = T.ToTensor()
                gt_concat_tensors = []
                pred_concat_tensors = []

                for i in range(pred_videos.shape[0]):
                    gt_np = (label_videos[i].permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)
                    pred_np = (pred_videos[i].permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)

                    gt_frames = [Image.fromarray(frame) for frame in gt_np]
                    pred_frames = [Image.fromarray(frame) for frame in pred_np]

                    widths, heights = zip(*(img.size for img in gt_frames))
                    total_width = sum(widths)
                    max_height = max(heights)
                    gt_concat = Image.new("RGB", (total_width, max_height))
                    x_offset = 0
                    for img in gt_frames:
                        gt_concat.paste(img, (x_offset, 0))
                        x_offset += img.size[0]
                    gt_concat.save(os.path.join(gt_video_path, f"{i:02d}_gt_video.jpg"))
                    gt_concat_tensors.append(to_tensor(gt_concat))

                    widths, heights = zip(*(img.size for img in pred_frames))
                    total_width = sum(widths)
                    max_height = max(heights)
                    pred_concat = Image.new("RGB", (total_width, max_height))
                    x_offset = 0
                    for img in pred_frames:
                        pred_concat.paste(img, (x_offset, 0))
                        x_offset += img.size[0]
                    pred_concat.save(os.path.join(pred_video_path, f"{i:02d}_pred_video.jpg"))
                    pred_concat_tensors.append(to_tensor(pred_concat))

                    if pred_concat.size != gt_concat.size:
                        pred_concat = pred_concat.resize(gt_concat.size)
                    w, h = gt_concat.size
                    final_img = Image.new("RGB", (w, h * 2))
                    final_img.paste(gt_concat, (0, 0))
                    final_img.paste(pred_concat, (0, h))
                    final_img.save(os.path.join(video_path, f"{i:02d}.jpg"))

                gt_concat_batch = torch.stack(gt_concat_tensors, dim=0)
                pred_concat_batch = torch.stack(pred_concat_tensors, dim=0)

                self.writer.add_images("validation/gt", gt_concat_batch, self.global_step, dataformats="NCHW")
                self.writer.add_images("validation/pred", pred_concat_batch, self.global_step, dataformats="NCHW")

        eval_time = eval_timer.elapse(True)

        self.model.train()
        return eval_meter, eval_time
