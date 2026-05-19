"""
Extract per-episode MP4 videos from the Bridge-V2 RLDS dataset.

Bridge-V2 (ericonaldo/Bridge-V2) is an RLDS-format dataset with:
  - Each tfrecord shard contains multiple full episodes (not chunked frames)
  - Camera keys in steps/observation: image_0, image_1, image_2, image_3
  - image_2/image_3 are optional (may be dummy if has_image_2/has_image_3 is False)
  - Splits: train, val

Example:
    python scripts/datasets/extract_rlds_bridge_v2_videos.py
    python scripts/datasets/extract_rlds_bridge_v2_videos.py -c image_1 -s 0 10
"""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image
from torchvision.io import write_video

load_dotenv()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_example(example_proto: tf.train.Example) -> dict:
    """Parse a single episode tf.train.Example into a flat dict.

    RLDS stores each feature as a list of values (one per timestep for steps/*).
    Scalar features (episode_metadata/*) have a single value.
    """
    result = {}
    for key, feature in example_proto.features.feature.items():
        kind = feature.WhichOneof("kind")
        if kind == "bytes_list":
            result[key] = list(feature.bytes_list.value)
        elif kind == "float_list":
            result[key] = list(feature.float_list.value)
        elif kind == "int64_list":
            result[key] = list(feature.int64_list.value)
        else:
            result[key] = []
    return result


def extract_dataset_videos(
    dataset_path: Path,
    output_dir: Path,
    camera_key: str = "image_0",
    fps: int = 5,
    split: str = "train",
    shard_start: int = 0,
    shard_end: int | None = None,
) -> int:
    info = read_dataset_info(dataset_path)
    splits_info = {s["name"]: s for s in info["splits"]}
    if split not in splits_info:
        available = list(splits_info.keys())
        print(f"  [WARN] Split '{split}' not found. Available: {available}")
        return 0

    split_info = splits_info[split]
    num_shards = len(split_info["shardLengths"])
    if shard_end is None:
        shard_end = num_shards
    shard_end = min(shard_end, num_shards)

    template = split_info["filepathTemplate"]
    dataset_name = info["name"]

    feature_camera_key = f"steps/observation/{camera_key}"

    episode_output_dir = output_dir / f"{dataset_name}_{split}"
    episode_output_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    for shard_idx in range(shard_start, shard_end):
        shard_name = template.replace("{DATASET}", dataset_name).replace("{SPLIT}", split)
        shard_name = shard_name.replace("{FILEFORMAT}", "tfrecord")
        shard_name = shard_name.replace("{SHARD_X_OF_Y}", f"{shard_idx:05d}-of-{num_shards:05d}")
        shard_path = dataset_path / shard_name

        if not shard_path.exists():
            print(f"  [WARN] Shard not found, skipping: {shard_path}")
            continue

        raw_ds = tf.data.TFRecordDataset(str(shard_path))
        num_in_shard = int(split_info["shardLengths"][shard_idx])
        print(f"  [{dataset_name}] shard {shard_idx:04d}/{num_shards - 1} ({shard_path.name})")

        for ep_i, record in enumerate(raw_ds):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            data = parse_example(example)

            if feature_camera_key not in data:
                print(f"    [{dataset_name}] shard {shard_idx} ep {ep_i}: no {feature_camera_key}, skip")
                continue

            img_bytes_list = data[feature_camera_key]
            if not img_bytes_list:
                continue

            frames = []
            for img_bytes in img_bytes_list:
                if not img_bytes:
                    continue
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                frames.append(np.array(img))  # [H, W, C] uint8

            if not frames:
                continue

            video = np.stack(frames)  # [T, H, W, C] uint8

            # Use episode_id from metadata if available, otherwise use shard_local index
            if "episode_metadata/episode_id" in data:
                ep_id = int(data["episode_metadata/episode_id"][0])
            else:
                ep_id = ep_i

            out_path = episode_output_dir / f"shard{shard_idx:04d}_ep{ep_id:06d}.mp4"
            write_video(str(out_path), video, fps=fps, video_codec="libx264")
            total_episodes += 1

        print(f"    [{dataset_name}] shard {shard_idx:04d}: {num_in_shard} expected, processed")

    print(f"  -> Saved {total_episodes} videos to {episode_output_dir}")
    return total_episodes


def read_dataset_info(dataset_path: Path) -> dict:
    # Try versioned directory first, then root
    for candidate in [dataset_path, dataset_path.parent]:
        info_path = candidate / "dataset_info.json"
        if info_path.exists():
            with open(info_path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"dataset_info.json not found under {dataset_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-episode MP4 videos from Bridge-V2 RLDS dataset",
    )
    default_data_root = Path(
        os.environ.get("DATASETS_PATH", ".")
    ) / "ericonaldo" / "Bridge-V2" / "1.0.0"
    parser.add_argument(
        "-d", "--data-root", type=Path, default=default_data_root,
        help=f"Path to the RLDS dataset version directory (default: {default_data_root})",
    )
    default_output_root = Path(
        os.environ.get("DATASETS_PATH", ".")
    ) / "ericonaldo" / "bridge_v2_videos"
    parser.add_argument(
        "-o", "--output-root", type=Path, default=default_output_root,
        help=f"Directory to save output videos (default: {default_output_root})",
    )
    parser.add_argument(
        "-c", "--camera-key", default="image_0",
        help="Camera key in steps/observation (default: image_0; available: image_0, image_1, image_2, image_3)",
    )
    parser.add_argument(
        "-f", "--fps", type=int, default=5,
        help="Output video FPS (default: 5)",
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split to extract (default: train; available: train, val)",
    )
    parser.add_argument(
        "-s", "--shards", type=int, nargs=2, default=None,
        metavar=("START", "END"),
        help="Range of shards to process (default: all)",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    if not data_root.is_dir():
        # Try without version suffix
        alt = data_root.parent
        if alt.is_dir() and (alt / "dataset_info.json").exists():
            data_root = alt
        else:
            raise NotADirectoryError(f"Data root not found: {args.data_root}")

    shard_start = args.shards[0] if args.shards else 0
    shard_end = args.shards[1] if args.shards else None

    extract_dataset_videos(
        data_root,
        args.output_root,
        camera_key=args.camera_key,
        fps=args.fps,
        split=args.split,
        shard_start=shard_start,
        shard_end=shard_end,
    )


if __name__ == "__main__":
    main()
