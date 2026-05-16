"""
Extract per-episode MP4 videos from LeRobot v3.0 format datasets.

Example:
    python scripts/datasets/extract_lerobot_videos.py \\
        -d /inspire/qb-ilm/project/robot-reasoning/public/cyh/datasets/lerobot \\
        -o ./output_videos \\
        -n libero_goal_image
"""

import argparse
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.io import write_video


load_dotenv()


def read_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def extract_dataset_videos(
    dataset_path: Path,
    output_dir: Path,
    camera_key: str = "observation.images.image",
    fps: int | None = None,
) -> int:
    """
    Extract all episodes of a LeRobot dataset as MP4 videos.

    Returns the number of episodes extracted.
    """
    dataset_name = dataset_path.name
    info = read_info(dataset_path)

    if fps is None:
        fps = info["fps"]

    if camera_key not in info["features"]:
        print(
            f"  [WARN] Camera key '{camera_key}' not found in {dataset_name}. "
            f"Available: {[k for k, v in info['features'].items() if v.get('dtype') in ('image', 'video')]}"
        )
        return 0

    print(f"  [{dataset_name}] Loading metadata and parquet data ...")
    ds = LeRobotDataset(str(dataset_path), root=str(dataset_path))
    print(f"  [{dataset_name}] Triggering hf_dataset load ...")
    _ = ds[0]
    print(
        f"  [{dataset_name}] Dataset loaded ({info['total_episodes']} episodes, {info['total_frames']} frames, {fps} fps)"
    )

    episode_output_dir = output_dir / dataset_name
    episode_output_dir.mkdir(parents=True, exist_ok=True)

    episodes = ds.meta.episodes
    total = len(episodes)

    print(f"  [{dataset_name}] Starting extraction ({total} episodes) ...")
    for ep_idx in range(total):
        ep = episodes[ep_idx]
        start = ep["dataset_from_index"]
        end = ep["dataset_to_index"]
        n_frames = end - start

        batch = ds.hf_dataset[int(start) : int(end)]
        frames: list[torch.Tensor] = batch[camera_key]

        video = torch.stack(frames).mul(255).byte()  # [T, C, H, W] uint8
        video = video.permute(0, 2, 3, 1).numpy()  # [T, H, W, C] uint8

        out_path = episode_output_dir / f"episode_{ep_idx:06d}.mp4"
        write_video(str(out_path), video, fps=fps, video_codec="libx264")

        print(f"    [{dataset_name}] episode {ep_idx:04d}/{total} | {n_frames} frames | {out_path.name}")

    print(f"  -> Saved {total} videos to {episode_output_dir}")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-episode MP4 videos from LeRobot v3.0 datasets",
    )
    default_data_root = Path(os.environ.get("DATASETS_PATH", ".")) / "lerobot"
    parser.add_argument(
        "-d",
        "--data-root",
        type=Path,
        default=default_data_root,
        help=f"Root directory containing LeRobot dataset subdirectories (default: {default_data_root})",
    )
    default_output_root = Path(os.environ.get("DATASETS_PATH", ".")) / "lerobot_videos"
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=default_output_root,
        help=f"Directory to save output videos (default: {default_output_root})",
    )
    parser.add_argument(
        "-c",
        "--camera-key",
        default="observation.images.image",
        help="Camera feature key to extract (default: observation.images.image)",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        default=None,
        help="Output video FPS (default: use dataset's native FPS)",
    )
    parser.add_argument(
        "-n",
        "--datasets",
        nargs="*",
        default=None,
        help="Specific dataset names to process (default: all subdirs in data-root)",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root not found: {data_root}")

    if args.datasets:
        dataset_paths = [data_root / name for name in args.datasets]
    else:
        dataset_paths = sorted(d for d in data_root.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists())

    if not dataset_paths:
        print(f"No LeRobot datasets found under {data_root}")
        return

    print(f"Found {len(dataset_paths)} dataset(s)")
    total_episodes = 0

    for ds_path in dataset_paths:
        print(f"\nProcessing: {ds_path.name}")
        try:
            n = extract_dataset_videos(
                ds_path,
                args.output_root,
                camera_key=args.camera_key,
                fps=args.fps,
            )
            total_episodes += n
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")

    print(f"\nDone. Total episodes extracted: {total_episodes}")


if __name__ == "__main__":
    main()
