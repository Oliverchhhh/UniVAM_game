"""DDP training entry point for Stage-I future-condition representation learning."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .dataset import CupheadFutureIterableDataset
from .model import Stage1FutureConditionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Checkpoint path, or 'auto' for output_dir/latest.pt")
    parser.add_argument("--dry-run", action="store_true", help="Build one batch/model and stop after one update")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world, torch.device("cuda", local_rank)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


def seed_everything(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "model": unwrap(model).trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer, scheduler) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_trainable_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])


@torch.no_grad()
def validate(model, loader, device, max_batches: int) -> dict[str, float]:
    core = unwrap(model)
    core.eval()
    totals = {"correct": torch.zeros((), device=device), "shuffled": torch.zeros((), device=device), "zero": torch.zeros((), device=device)}
    count = torch.zeros((), device=device)
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        frames = batch["frames"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True, dtype=core.compute_dtype)
        mask = batch["actions_mask"].to(device, non_blocking=True)
        # Match the BF16 mixed-precision path used by training.  Without
        # autocast, BF16 visual features reach FP32 adapter/Q-Former weights
        # and the first scheduled validation fails with a dtype mismatch.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            current, frames_01 = core.preprocess(frames)
            condition = core.encode_condition(frames_01)
            noise = torch.randn_like(actions)
            time_sample = core.policy.nitrogen.sample_time(actions.shape[0], device, actions.dtype)
            variants = {
                "correct": condition,
                "shuffled": condition.roll(1, dims=0) if condition.shape[0] > 1 else condition.flip(1),
                "zero": torch.zeros_like(condition),
            }
            for name, cond in variants.items():
                loss, _ = core.policy.flow_loss(current, actions, mask, cond, noise=noise, time=time_sample)
                totals[name] += loss
        count += 1
    for value in [*totals.values(), count]:
        if dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    count = count.clamp_min(1)
    core.train()
    return {name: float((value / count).cpu()) for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    rank, local_rank, world, device = setup_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s rank={rank} %(levelname)s %(message)s",
    )
    seed_everything(int(config.get("seed", 43)), rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_cfg = config["data"]
    train_set = CupheadFutureIterableDataset(
        data_cfg["root"], "train", stride=int(data_cfg.get("train_stride", 4)),
        video_name=data_cfg.get("video_name", "256x256.mp4"), seed=int(config.get("seed", 43)),
    )
    val_set = CupheadFutureIterableDataset(
        data_cfg["root"], "val", stride=int(data_cfg.get("val_stride", 17)),
        video_name=data_cfg.get("video_name", "256x256.mp4"), seed=int(config.get("seed", 43)), repeat=True,
    )
    loader_kwargs = dict(
        batch_size=int(config["train"].get("micro_batch_size", 1)),
        num_workers=int(data_cfg.get("workers", 2)),
        pin_memory=True,
        persistent_workers=int(data_cfg.get("workers", 2)) > 0,
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)) if int(data_cfg.get("workers", 2)) > 0 else None,
    )
    train_loader = DataLoader(train_set, **loader_kwargs)
    val_loader = DataLoader(val_set, **loader_kwargs)

    model_cfg = config["model"]
    model = Stage1FutureConditionModel(
        nitrogen_checkpoint=model_cfg["nitrogen_checkpoint"],
        nitrogen_root=model_cfg["nitrogen_root"],
        siglip_path=model_cfg["siglip_path"],
        wog_root=model_cfg["wog_root"],
        device=device,
        dtype=model_cfg.get("dtype", "bfloat16"),
        condition_dim=int(model_cfg.get("condition_dim", 64)),
        qformer_hidden=int(model_cfg.get("qformer_hidden", 1024)),
        qformer_layers=int(model_cfg.get("qformer_layers", 6)),
        adapter_dim=int(model_cfg.get("adapter_dim", 256)),
        condition_dropout=float(model_cfg.get("condition_dropout", 0.1)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
    )
    if rank == 0:
        logging.info("parameter report: %s", json.dumps(model.parameter_report()))
    model.train()
    if world > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False,
        )

    train_cfg = config["train"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(train_cfg.get("learning_rate", 1e-4)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])), weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    max_steps = 1 if args.dry_run else int(train_cfg.get("max_steps", 50000))
    warmup = int(train_cfg.get("warmup_steps", 1000))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(step, 1) / max(warmup, 1)
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return max(float(train_cfg.get("min_lr_ratio", 0.1)), 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    output_dir = Path(config.get("output_dir", "runs/stage1_future_condition")).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    start_step = 0
    resume = args.resume
    if resume == "auto":
        resume = str(output_dir / "latest.pt")
    if resume and Path(resume).is_file():
        start_step = load_checkpoint(Path(resume), model, optimizer, scheduler)
        logging.info("resumed %s at optimizer step %d", resume, start_step)

    accumulation = int(train_cfg.get("gradient_accumulation", 8))
    clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 20))
    val_every = int(train_cfg.get("val_every", 1000))
    save_every = int(train_cfg.get("save_every", 1000))
    val_batches = int(train_cfg.get("val_batches", 64))
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(train_loader)
    step = start_step
    micro_step = 0
    loss_window = 0.0
    window_start = time.time()

    while step < max_steps:
        batch = next(data_iter)
        frames = batch["frames"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        masks = batch["actions_mask"].to(device, non_blocking=True)
        sync_now = (micro_step + 1) % accumulation == 0
        sync_context = contextlib.nullcontext() if sync_now or world == 1 else model.no_sync()
        with sync_context:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(frames, actions, masks) / accumulation
            loss.backward()
        loss_window += float(loss.detach()) * accumulation
        micro_step += 1
        if not sync_now:
            continue

        torch.nn.utils.clip_grad_norm_(trainable, clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_every == 0:
            mean_loss = reduce_mean(torch.tensor(loss_window / (log_every * accumulation), device=device))
            if rank == 0:
                elapsed = time.time() - window_start
                peak = torch.cuda.max_memory_allocated(device) / 2**30
                logging.info(
                    "step=%d loss=%.6f lr=%.3e %.2f opt_step/s peak_alloc=%.2fGiB",
                    step, float(mean_loss), scheduler.get_last_lr()[0], log_every / elapsed, peak,
                )
                with (output_dir / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps({"step": step, "train_loss": float(mean_loss), "peak_alloc_gib": peak}) + "\n")
            loss_window = 0.0
            window_start = time.time()
            torch.cuda.reset_peak_memory_stats(device)

        # Persist progress before running validation so a validation-only
        # failure can always resume from the just-completed optimizer step.
        if rank == 0 and save_every > 0 and step % save_every == 0:
            save_checkpoint(output_dir / "latest.pt", model, optimizer, scheduler, step, config)
            archive_every = int(train_cfg.get("archive_every", 5000))
            if archive_every > 0 and step % archive_every == 0:
                save_checkpoint(output_dir / f"step_{step:07d}.pt", model, optimizer, scheduler, step, config)
        if dist.is_initialized() and save_every > 0 and step % save_every == 0:
            dist.barrier()

        if val_every > 0 and step % val_every == 0:
            metrics = validate(model, val_loader, device, val_batches)
            if rank == 0:
                logging.info("validation step=%d %s", step, json.dumps(metrics))
                with (output_dir / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps({"step": step, **{f"val_{k}": v for k, v in metrics.items()}}) + "\n")

    if rank == 0 and not args.dry_run:
        save_checkpoint(output_dir / "final.pt", model, optimizer, scheduler, step, config)
        save_checkpoint(output_dir / "latest.pt", model, optimizer, scheduler, step, config)
    elif rank == 0:
        logging.info("dry-run completed; no checkpoint written")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
