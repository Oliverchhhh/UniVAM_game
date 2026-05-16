"""
Extract per-episode MP4 videos from a LeRobot v2.0 format dataset.

Example:
    python scripts/datasets/extract_lerobot_v2_videos.py

LeRobot v2.0 differs from v3.0:
  - Each episode is a standalone parquet file (not chunked frames)
  - Episodes metadata is stored in meta/episodes.jsonl (not parquet)
  - Camera keys are typically "image" / "wrist_image"
"""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from dotenv import load_dotenv
from PIL import Image
from torchvision.io import write_video


load_dotenv()


def read_info(dataset_path: Path) -> dict:
    with open(dataset_path / "meta" / "info.json", encoding="utf-8") as f:
        return json.load(f)


def read_episodes(dataset_path: Path) -> list[dict]:
    episodes = []
    ep_path = dataset_path / "meta" / "episodes.jsonl"
    with open(ep_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def extract_dataset_videos(
    dataset_path: Path,
    output_dir: Path,
    camera_key: str = "image",
    fps: int | None = None,
) -> int:
    dataset_name = dataset_path.name
    info = read_info(dataset_path)
    chunks_size = info["chunks_size"]
    episodes = read_episodes(dataset_path)

    if fps is None:
        fps = info["fps"]

    if camera_key not in info["features"]:
        available = [k for k, v in info["features"].items() if v.get("dtype") == "image"]
        print(f"  [WARN] Camera key '{camera_key}' not found. Available: {available}")
        return 0

    total = len(episodes)
    print(f"  [{dataset_name}] {total} episodes, {info['total_frames']} frames, {fps} fps")

    episode_output_dir = output_dir / dataset_name
    episode_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{dataset_name}] Starting extraction ({total} episodes) ...")
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // chunks_size

        parquet_path = dataset_path / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        table = pq.read_table(parquet_path, columns=[camera_key])

        frames = []
        for row in table.column(camera_key):
            img_bytes = row.as_py()["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frames.append(np.array(img))  # [H, W, C] uint8

        video = np.stack(frames)  # [T, H, W, C] uint8

        out_path = episode_output_dir / f"episode_{ep_idx:06d}.mp4"
        write_video(str(out_path), video, fps=fps, video_codec="libx264")

        print(f"    [{dataset_name}] episode {ep_idx:04d}/{total - 1} | {len(frames)} frames | {out_path.name}")

    print(f"  -> Saved {total} videos to {episode_output_dir}")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-episode MP4 videos from a LeRobot v2.0 dataset",
    )
    default_data_root = Path(os.environ.get("DATASETS_PATH", ".")) / "physical-intelligence" / "libero"
    parser.add_argument(
        "-d",
        "--data-root",
        type=Path,
        default=default_data_root,
        help=f"Path to the LeRobot v2.0 dataset directory (default: {default_data_root})",
    )
    default_output_root = Path(os.environ.get("DATASETS_PATH", ".")) / "physical-intelligence" / "libero_videos"
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
        default="image",
        help="Camera feature key to extract (default: image, also available: wrist_image)",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        default=None,
        help="Output video FPS (default: use dataset's native FPS)",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root not found: {data_root}")
    if not (data_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Not a LeRobot v2.0 dataset: {data_root}")

    extract_dataset_videos(
        data_root,
        args.output_root,
        camera_key=args.camera_key,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
