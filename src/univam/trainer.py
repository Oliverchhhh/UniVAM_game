import os
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as DIST
import torchvision.transforms as T
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from univam.models.wanva import Wan22VisionActionModel
from univam.utils.data import check_tensor, complex_to_device, fp32_to_bf16, move_to_cuda
from univam.utils.files import ensure_directory, ensure_dirname
from univam.utils.metrics import Meter, Timer, calculate_psnr, calculate_ssim, get_parameters
from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


class Trainer:
    def __init__(self, args, model: Wan22VisionActionModel, criterion=None, optimizer=None, lr_scheduler=None) -> None:
        self.model: Wan22VisionActionModel = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.local_rank = overwatch.local_rank()
        self.rank = overwatch.rank()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.epoch = -1
        self.global_step = -1

        self.eval_before_train = False

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
        self.img_size = args.data.img_size
        self.log_dir = os.path.join(args.log_dir, args.task_name, args.projector.type)
        self.ckpt_save_dir = os.path.join(args.train.ckpt_save_dir, args.task_name, args.projector.type)

        if overwatch.is_rank_zero() and args.do_train:
            ensure_directory(self.log_dir)

    def move_model_to_cuda(self) -> None:
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
        overwatch.warning(f"Saving models to {save_path}")

        self.accelerator.wait_for_everyone()
        if overwatch.is_rank_zero():
            ensure_directory(save_path)
            model_dict = self.accelerator.get_state_dict(self.model)
            projector_model_dict = self.accelerator.get_state_dict(self.model.projector)
            self.model._save_ckpt(model_dict, projector_model_dict, save_path, self.global_step)

    def load_checkpoint(self, load_path) -> None:
        global_step = self.model._load_ckpt(load_path)
        self.global_step = global_step

    def setup_model_for_training(self) -> None:
        if overwatch.is_rank_zero():
            overwatch.warning(f"Existing dirs detected {self.log_dir}")
            ensure_dirname(self.log_dir, override=False)

        self.model.set_trainable_params()
        self.prepare_dist_model()

    def train_eval_by_iter(self, train_loader, eval_loader=None, use_tqdm=True) -> None:
        self.model, self.optimizer, train_loader = self.accelerator.prepare(self.model, self.optimizer, train_loader)

        if self.num_iters:
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

        if self.eval_before_train and self.global_step == 0:
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
                    outputs = self.forward_step(inputs, criterion=self.criterion)
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
                    torch.cuda.empty_cache()

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

        label_imgs = []
        pred_imgs = []

        with torch.no_grad():
            eval_loader = (
                tqdm(eval_loader, total=len(eval_loader), ncols=150, dynamic_ncols=False) if use_tqdm else eval_loader
            )
            for inputs in eval_loader:
                inputs = self.prepare_batch(inputs)
                outputs = self.forward_step(inputs)
                metric_and_loss = {k: v for k, v in outputs.items() if k.split("_")[0] in ["metric", "loss"]}

                for k, v in metric_and_loss.items():
                    metric_and_loss[k] = self.reduce_mean(v)
                eval_meter.update(metric_and_loss)

                label_img = inputs["images"]

                pred_img = self.model.inv_vae_transform(outputs["images"])
                pred_img = torch.clamp(pred_img, 0, 1)

                label_imgs.append(label_img)
                pred_imgs.append(pred_img)

            label_imgs = torch.cat(label_imgs, dim=0)
            pred_imgs = torch.cat(pred_imgs, dim=0)

            psnr = calculate_psnr(pred_imgs, label_imgs)
            ssim = calculate_ssim(pred_imgs, label_imgs)

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

            np_images = np.stack([np.asarray(img.permute(0, 2, 1).cpu().float()) for img in pred_imgs])
            np_gt_images = np.stack([np.asarray(img.permute(0, 2, 1).cpu().float()) for img in label_imgs])

            toimg = T.ToPILImage()

            images = [toimg(img.cpu().float()) for img in pred_imgs]
            gt_images = [toimg(img.cpu().float()) for img in label_imgs]

            if overwatch.is_rank_zero():
                self.accelerator.log(
                    {
                        "val/psnr": eval_meter.avg["val/psnr"],
                        "val/ssim": eval_meter.avg["val/ssim"],
                    },
                    step=self.global_step,
                )

                image_path = os.path.join(self.log_dir, "images", str(self.global_step))
                ensure_directory(os.path.join(image_path))
                for i in range(len(images)):
                    images[i].save(os.path.join(image_path, f"{i}_pred.jpeg"))
                    gt_images[i].save(os.path.join(image_path, f"{i}_gt.jpeg"))

                if overwatch.is_rank_zero():
                    self.writer.add_images("validation/pred", np_images, self.global_step, dataformats="NCWH")
                    self.writer.add_images("validation/gt", np_gt_images, self.global_step, dataformats="NCWH")

        eval_time = eval_timer.elapse(True)

        self.model.train()
        return eval_meter, eval_time

    def manually_eval(self, images, batch_size=64):
        self.model.eval()

        label_imgs = images
        toimg = T.ToPILImage()
        transforms = T.Compose([T.Resize(self.img_size, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(os.path.join(image_path))

        with torch.no_grad():
            for start_idx in range(0, len(images), batch_size):
                end_idx = min(start_idx + batch_size, len(images))
                batch_images = images[start_idx:end_idx]

                tensor_images = torch.stack([transforms(image).to(self.device) for image in batch_images])
                inputs = {"images": tensor_images}

                inputs = self.prepare_batch(inputs)
                outputs = self.forward_step(inputs)

                pred_imgs = self.model.inv_vae_transform(outputs["images"])
                pred_imgs = torch.clamp(pred_imgs, 0, 1)

                overwatch.info(f"PSNR: {calculate_psnr(pred_imgs, tensor_images):.4f}")
                overwatch.info(f"SSIM: {calculate_ssim(pred_imgs, tensor_images):.4f}")
                # overwatch.info(f"rFID: {calculate_rfid(pred_imgs, tensor_images):.4f}")

                pred_imgs = [toimg(pred_img.squeeze().cpu()) for pred_img in pred_imgs]

                for idx, pred_img in enumerate(pred_imgs):
                    pred_img.save(os.path.join(image_path, f"{start_idx + idx}_pred.jpeg"))
                    label_imgs[idx].save(os.path.join(image_path, f"{start_idx + idx}_gt.jpeg"))

    def interpolation_eval(
        self,
        image1,
        image2,
        tokens=None,
        num_interpolation=5,
        batch_size=None,
        to_video=False,
        fps=10,
        name="interpolation.mp4",
    ):
        """
        对压缩token进行线性插值
        """
        self.model.eval()

        transforms = T.Compose([T.Resize(self.img_size, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])

        with torch.no_grad():
            image1 = transforms(image1).to(self.device).unsqueeze(0)
            image2 = transforms(image2).to(self.device).unsqueeze(0)

            inputs1 = self.prepare_batch(image1)
            inputs2 = self.prepare_batch(image2)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)

            outputs = self.model.interpolation_eval(
                inputs1,
                inputs2,
                generator,
                tokens=tokens,
                num_interpolation=num_interpolation,
                batch_size=batch_size,
            )

        toimg = T.ToPILImage()

        images = []
        for pred_image in outputs:
            pred_image = self.model.inv_vae_transform(pred_image)
            pred_image = torch.clamp(pred_image, 0, 1)
            images.append(toimg(pred_image.cpu()))

        if to_video:
            import imageio

            video_path = os.path.join(self.log_dir, "images", str(self.global_step))
            ensure_directory(video_path)
            save_path = os.path.join(video_path, name)
            imageio.mimsave(save_path, images, fps=fps)
            return

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(image_path)
        for i in range(len(images)):
            images[i].save(os.path.join(image_path, f"interpolation_{i}.jpeg"))

        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        combined_image = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        # 保存拼接后的图像
        combined_image.save(os.path.join(image_path, f"combined_step_{self.global_step}.jpeg"))

    def visualize_token(self, images, batch_size=64, token=0, visualize=False, name="test"):
        self.model.eval()

        transforms = T.Compose([T.Resize(self.img_size, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])
        image_embeddings = []
        with torch.no_grad():
            for start_idx in range(0, len(images), batch_size):
                end_idx = min(start_idx + batch_size, len(images))
                batch_images = images[start_idx:end_idx]

                tensor_images = torch.stack([transforms(image).to(self.device) for image in batch_images])

                projector_images = self.model.projector_feature_extractor(tensor_images)
                image_embedding = self.model.encode(projector_images)
                image_embeddings.append(image_embedding)

        image_embeddings = torch.cat(image_embeddings, dim=0)

        X = image_embeddings[:, token, :]

        X = X - X.mean(dim=0, keepdim=True)

        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        explained_var = S**2
        explained_ratio = explained_var / explained_var.sum()

        for i in range(5):
            overwatch.info(f"Token {token}: PC{i + 1}: {explained_ratio[i].item():.4f}")

        lambda_ = explained_var
        effective_dim = (lambda_.sum() ** 2) / (lambda_**2).sum()
        overwatch.info(f"Effective dimension: {effective_dim.item()}")

        if visualize:
            import matplotlib.pyplot as plt

            V2 = Vh[:2]
            Z = X @ V2.T  # [N, 2]

            Z_np = Z.cpu().numpy()

            plt.figure(figsize=(6, 6))
            plt.scatter(Z_np[:, 0], Z_np[:, 1], s=5, alpha=0.6)

            for i in range(Z_np.shape[0]):
                plt.text(Z_np[i, 0], Z_np[i, 1], str(i), fontsize=6, alpha=0.8)

            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA of Image Embeddings")
            plt.axis("equal")
            plt.savefig(f"Token{token}_PCA_{name}.png")
            plt.close()

    def delta_interpolation(self, image, start, end):
        """
        进行delta插值
        """
        self.model.eval()

        toimg = T.ToPILImage()
        transforms = T.Compose([T.Resize(self.img_size, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])
        size = image.size

        with torch.no_grad():
            start_inputs = transforms(start).to(self.device).unsqueeze(0)
            end_inputs = transforms(end).to(self.device).unsqueeze(0)
            image_inputs = transforms(image).to(self.device).unsqueeze(0)

            start_inputs = self.prepare_batch(start_inputs)
            end_inputs = self.prepare_batch(end_inputs)
            image_inputs = self.prepare_batch(image_inputs)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)

            outputs = self.model.delta_interpolation(
                image_inputs,
                start_inputs,
                end_inputs,
                generator,
            )

        pred_image = self.model.inv_vae_transform(outputs).squeeze(0)
        pred_image = torch.clamp(pred_image, 0, 1)
        pred_image = toimg(pred_image.cpu())

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(os.path.join(image_path))

        pred_image.save(os.path.join(image_path, f"delta_interpolation_{self.global_step}.jpeg"))

        images = [start.resize(size), end.resize(size), image, pred_image.resize(size)]
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        combined_image = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        combined_image.save(os.path.join(image_path, f"delta_interpolation_combined_{self.global_step}.jpeg"))
