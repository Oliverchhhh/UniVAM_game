#!/usr/bin/env python3
"""
Flow-style UniVAM video feature precompute.

For each frame t, encode a 5-frame clip [t, t+1, t+2, t+3, t+4] into 2 condition
tokens via WanVAE + Projector (same as encode_videos.py).

Output per video (default: univam_flow_features.pt):
  features   -> [T, num_token, hidden_dim] float16, T ~= native frame count (~1200)
  tail_mask  -> [T] bool, True for tail-handled positions
  meta       -> dict with native_frames / window_size / tail_mode / video_path

Note: training n_seq_timesteps=200 is a chunk length, not total video length.
Precomputed dinov3/vjepa2 features are [1200, D]; this script matches that per-frame alignment.

Tail handling when t cannot access 4 future frames:
  pad_last     (default): pad with the last frame, one token per frame
  last_window: use the final 5 frames for all remaining positions
  skip_tail:   only encode frames with a full future window (shorter output)

Example:
  PRETRAINED_MODEL_PATH=/path/to/models \\
  RESUME_PATH=./univam_cuphead_140000 \\
  python encode_videos_flow.py \\
      --data_folder /path/to/cuphead_dataset_converted \\
      --video_name 256x256.mp4 \\
      --config_path configs/debug.yaml
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import decord
import torch
from PIL.Image import Resampling
from torchvision.transforms import functional as F

from univam.models.wanva import Wan22VisionModel
from univam.trainer import Trainer
from dotenv import load_dotenv
from omegaconf import OmegaConf

from univam.utils.data import set_seed
from univam.utils.dataloaders.video import normalize_video
from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def load_video_frames(
    video_path: str,
    image_size: list[int],
    num_frames: int = 0,
    min_frames: int = 5,
) -> torch.Tensor:
    """
    Load frames from video. Returns [T, C, H, W] in [-1, 1].

    num_frames:
      0  -> use all native frames (default, aligned with dinov3 precompute [T, D])
      >0 -> uniformly subsample to this many frames (debug / downsample only)
    """
    decord.bridge.set_bridge("torch")
    reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
    orig_n = len(reader)
    if orig_n < min_frames:
        raise ValueError(f"{video_path}: only {orig_n} frames, need at least {min_frames}")

    if num_frames <= 0 or num_frames >= orig_n:
        indices = list(range(orig_n))
    elif num_frames == 1:
        indices = [0]
    else:
        indices = [round(i / (num_frames - 1) * (orig_n - 1)) for i in range(num_frames)]

    frames = reader.get_batch(indices).permute(0, 3, 1, 2)  # [T, C, H, W]
    if list(frames.shape[-2:]) != image_size:
        frames = F.resize(frames, image_size, interpolation=Resampling.BICUBIC)
    frames = normalize_video(frames)
    return frames


def build_window_indices(
    start_t: int,
    num_frames: int,
    window_size: int,
    tail_mode: str,
) -> tuple[list[int], bool]:
    """Return frame indices for one flow window and whether tail handling was used."""
    future_needed = window_size - 1
    if start_t + future_needed < num_frames:
        return list(range(start_t, start_t + window_size)), False

    if tail_mode == "skip_tail":
        raise StopIteration

    if tail_mode == "last_window":
        start = max(0, num_frames - window_size)
        return list(range(start, num_frames)), True

    if tail_mode == "pad_last":
        indices = list(range(start_t, num_frames))
        last_idx = num_frames - 1
        while len(indices) < window_size:
            indices.append(last_idx)
        return indices, True

    raise ValueError(f"Unknown tail_mode: {tail_mode}")


def iter_flow_windows(
    num_frames: int,
    window_size: int,
    tail_mode: str,
):
    """Yield (t, frame_indices, is_tail) for each output position."""
    if tail_mode == "skip_tail":
        last_t = num_frames - window_size
        if last_t < 0:
            return
        for t in range(last_t + 1):
            indices, is_tail = build_window_indices(t, num_frames, window_size, tail_mode)
            yield t, indices, is_tail
        return

    for t in range(num_frames):
        indices, is_tail = build_window_indices(t, num_frames, window_size, tail_mode)
        yield t, indices, is_tail


def build_window_batch(
    frames: torch.Tensor,
    window_size: int,
    tail_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Build all flow windows from sampled frames.

    Returns:
        windows: [N, window_size, C, H, W]
        tail_mask: [N] bool
        positions: list[int], frame index aligned with each window
    """
    num_frames = frames.shape[0]
    pieces = []
    tail_flags = []
    positions = []

    for t, indices, is_tail in iter_flow_windows(num_frames, window_size, tail_mode):
        pieces.append(frames[indices])
        tail_flags.append(is_tail)
        positions.append(t)

    if not pieces:
        raise ValueError(f"Cannot build any flow windows from {num_frames} frames")

    windows = torch.stack(pieces, dim=0)
    tail_mask = torch.tensor(tail_flags, dtype=torch.bool)
    return windows, tail_mask, positions


@torch.no_grad()
def encode_windows(
    model: Wan22VisionModel,
    windows: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    """Encode [N, T, C, H, W] windows to [N, num_token, hidden_dim]."""
    outputs = []
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(device=device, dtype=dtype)
        video_latents = model.wanvae.encode(batch)
        video_embeds = model.encode(video_latents)
        outputs.append(video_embeds.cpu())

    return torch.cat(outputs, dim=0)


def process_video(
    model: Wan22VisionModel,
    video_path: str,
    args,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    frames = load_video_frames(
        video_path,
        args.image_size,
        num_frames=args.num_frames,
        min_frames=args.window_size,
    )
    windows, tail_mask, positions = build_window_batch(frames, args.window_size, args.tail_mode)
    features = encode_windows(model, windows, device, dtype, args.batch_windows)
    features = features.to(torch.float16)

    return {
        "features": features,
        "tail_mask": tail_mask,
        "positions": torch.tensor(positions, dtype=torch.int32),
        "meta": {
            "video_path": video_path,
            "native_frames": int(frames.shape[0]),
            "num_frames": int(frames.shape[0]),
            "subsampled": args.num_frames > 0,
            "window_size": args.window_size,
            "tail_mode": args.tail_mode,
            "num_token": int(features.shape[1]),
            "hidden_dim": int(features.shape[2]),
            "output_positions": len(positions),
        },
    }


def build_model(config, resume_path: str, device: torch.device, dtype: torch.dtype) -> Wan22VisionModel:
    model = Wan22VisionModel(config).to(device=device, dtype=dtype)
    model.eval()

    trainer = Trainer(args=config, model=model)
    trainer.load_checkpoint(resume_path)
    trainer.move_model_to_device()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Flow-style UniVAM video feature precompute")
    parser.add_argument("--config_path", default="configs/debug.yaml", type=str)
    parser.add_argument("--data_folder", required=True, help="Root folder to search videos recursively")
    parser.add_argument("--video_name", default="256x256.mp4")
    parser.add_argument("--output_name", default="univam_flow_features.pt")
    parser.add_argument("--resume_path", default=None, help="Checkpoint dir with Wan22VM.pth and Projector.pth")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help="0=use all native frames (~1200@60fps); >0=uniform subsample for debug only",
    )
    parser.add_argument("--window_size", type=int, default=5, help="Current frame + future frames")
    parser.add_argument(
        "--tail_mode",
        default="pad_last",
        choices=["pad_last", "last_window", "skip_tail"],
        help="How to handle positions that cannot access full future window",
    )
    parser.add_argument("--batch_windows", type=int, default=16, help="Number of 5-frame windows per GPU batch")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_config(config_path: str):
    load_dotenv()
    OmegaConf.register_new_resolver("oc.load", lambda path: OmegaConf.load(path))
    return OmegaConf.load(config_path)


def main():
    cli_args = parse_args()
    config = load_config(cli_args.config_path)
    config.data.image_size = [256, 256]
    cli_args.image_size = list(config.data.image_size)

    resume_path = cli_args.resume_path or os.environ.get("RESUME_PATH")
    if resume_path is None:
        raise ValueError("Please set --resume_path or RESUME_PATH")

    set_seed(config.seed)
    device = torch.device(f"cuda:{cli_args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    overwatch.info("Building model...")
    model = build_model(config, resume_path, device, dtype)

    video_paths = sorted(glob.glob(os.path.join(cli_args.data_folder, "**", cli_args.video_name), recursive=True))
    overwatch.info(f"Found {len(video_paths)} videos named {cli_args.video_name}")

    if cli_args.num_workers > 1:
        video_paths = video_paths[cli_args.worker_id :: cli_args.num_workers]
        overwatch.info(f"Worker {cli_args.worker_id}/{cli_args.num_workers}: {len(video_paths)} videos")

    if cli_args.skip_existing:
        before = len(video_paths)
        video_paths = [
            p
            for p in video_paths
            if not os.path.exists(os.path.join(os.path.dirname(p), cli_args.output_name))
        ]
        overwatch.info(f"Skipped {before - len(video_paths)} existing outputs, remaining {len(video_paths)}")

    if not video_paths:
        overwatch.info("All done.")
        return

    processed = errors = 0
    t0 = time.time()
    for video_path in video_paths:
        output_path = os.path.join(os.path.dirname(video_path), cli_args.output_name)
        try:
            result = process_video(model, video_path, cli_args, device, dtype)
            torch.save(result, output_path)
            processed += 1
            shape = tuple(result["features"].shape)
            tail_count = int(result["tail_mask"].sum().item())
            overwatch.info(
                f"Saved {output_path} shape={shape} tail={tail_count} <- {video_path}"
            )
            if processed % 20 == 0:
                elapsed = time.time() - t0
                overwatch.info(
                    f"progress {processed}/{len(video_paths)}, "
                    f"{elapsed / processed:.2f}s/video, "
                    f"eta {(len(video_paths) - processed) * elapsed / processed / 60:.1f}min"
                )
        except Exception as e:
            overwatch.error(f"failed {video_path}: {e}")
            errors += 1

    elapsed = time.time() - t0
    overwatch.info(
        f"done: processed={processed}, errors={errors}, "
        f"elapsed={elapsed:.1f}s, avg={elapsed / max(processed, 1):.2f}s/video"
    )


if __name__ == "__main__":
    main()
