"""Streaming dataset for future-conditioned NitroGen Stage-I training."""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .actions import ACTION_HORIZON, frame_action
from .proto import video_annotation_pb2

try:
    from torchcodec.decoders import VideoDecoder
except Exception:
    VideoDecoder = None


KEYFRAME_OFFSETS = (0, 4, 8, 12, 17)


@dataclass(frozen=True)
class ChunkRecord:
    source_id: str
    chunk_dir: Path
    proto_path: Path
    video_path: Path


def source_split(source_id: str, seed: int = 43) -> str:
    """Stable source-video split, preventing neighboring chunks from leaking."""
    digest = hashlib.blake2b(f"{seed}:{source_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 1000
    if bucket < 900:
        return "train"
    if bucket < 950:
        return "val"
    return "test"


def discover_chunks(root: str | Path, video_name: str = "256x256.mp4") -> list[ChunkRecord]:
    root = Path(root)
    records: list[ChunkRecord] = []
    for proto in sorted(root.glob("*/*/annotation.proto")):
        video = proto.parent / video_name
        if video.is_file():
            records.append(ChunkRecord(proto.parent.parent.name, proto.parent, proto, video))
    if not records:
        raise FileNotFoundError(f"No */*/annotation.proto + {video_name} pairs found under {root}")
    return records


def _load_annotations(path: Path):
    annotation = video_annotation_pb2.VideoAnnotation()
    annotation.ParseFromString(path.read_bytes())
    return annotation.frame_annotations


def _decode_keyframes(decoder, indices: Sequence[int]) -> torch.Tensor:
    """Return uint8/float video frames as [5,C,H,W] across torchcodec versions."""
    try:
        batch = decoder.get_frames_at(list(indices))
        frames = batch.data if hasattr(batch, "data") else batch
    except (AttributeError, TypeError):
        frames = torch.stack([decoder[i] for i in indices])
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise RuntimeError(f"Unexpected decoded frame shape: {tuple(frames.shape)}")
    return frames


class CupheadFutureIterableDataset(IterableDataset):
    """Streams `(I_t, I_t+4, I_t+8, I_t+12, I_t+17, a_t+1:t+18)` samples."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        stride: int = 4,
        video_name: str = "256x256.mp4",
        seed: int = 43,
        repeat: bool | None = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        self.stride = stride
        self.seed = seed
        self.repeat = split == "train" if repeat is None else repeat
        self.records = [r for r in discover_chunks(root, video_name) if source_split(r.source_id, seed) == split]
        if not self.records:
            raise RuntimeError(f"No chunks assigned to split={split}")

    def _sharded_records(self) -> tuple[list[ChunkRecord], int]:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        shard_id = rank * workers + worker_id
        num_shards = world * workers
        return self.records[shard_id::num_shards], shard_id

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | str | int]]:
        if VideoDecoder is None:
            raise RuntimeError("torchcodec with FFmpeg support is required")
        records, shard_id = self._sharded_records()
        epoch = 0
        while True:
            ordered = list(records)
            rng = random.Random(self.seed + 1009 * epoch + shard_id)
            if self.split == "train":
                rng.shuffle(ordered)
            for record in ordered:
                try:
                    annotations = _load_annotations(record.proto_path)
                    decoder = VideoDecoder(str(record.video_path), device="cpu", num_ffmpeg_threads=1)
                    n_frames = min(len(decoder), len(annotations))
                    # Need image t+17 and actions t+1 ... t+18 inclusive.
                    anchors = list(range(0, n_frames - ACTION_HORIZON, self.stride))
                    if self.split == "train":
                        rng.shuffle(anchors)
                    for anchor in anchors:
                        frames = _decode_keyframes(decoder, [anchor + x for x in KEYFRAME_OFFSETS])
                        action_and_mask = [frame_action(annotations[i]) for i in range(anchor + 1, anchor + 19)]
                        actions = torch.stack([x[0] for x in action_and_mask])
                        masks = torch.stack([x[1] for x in action_and_mask])
                        if not masks.any():
                            continue
                        yield {
                            "frames": frames,
                            "actions": actions,
                            "actions_mask": masks,
                            "source_id": record.source_id,
                            "chunk": record.chunk_dir.name,
                            "anchor": anchor,
                        }
                except Exception as exc:
                    logging.warning("Skipping %s: %s", record.chunk_dir, exc)
            epoch += 1
            if not self.repeat:
                return
