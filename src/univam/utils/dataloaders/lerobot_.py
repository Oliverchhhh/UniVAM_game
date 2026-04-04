import bisect
import itertools
import json
import random
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


CAMERA_KEYS = {
    "libero": ["observation.images.image"],
    "robocasa": ["image"],
    "robotwin": ["observation.images.cam_high"],
}

ACTION_KEYS = {
    "libero": ["action"],
    "robocasa": ["actions"],
    "robotwin": ["action"],
}


def save_image(img: torch.Tensor, path: str = "./output.png"):
    img = (img + 1) / 2
    img = img.mul(255).byte()
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = np.ascontiguousarray(img_np)
    Image.fromarray(img_np).save(path)


def resolve_root(repo_id: str, root=None) -> Path:
    return Path(root) if root else HF_LEROBOT_HOME / repo_id


def read_info(repo_id: str, root=None) -> dict:
    info_path = resolve_root(repo_id, root) / "meta" / "info.json"
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def build_delta_timestamps(
    camera_keys: list[str],
    action_keys: list[str],
    num_frames: int,
    target_fps: int,
    orig_fps: int,
    chunk_size: int,
) -> dict[str, list[float]]:
    """
    Build delta_timestamps:

    - camera: `T` timestamps
    - action: `T * chunk_size` timestamps
        for `i` frame: `t = i/target_fps`, continuously take `chunk_size` original frame actions
        => `t = i/target_fps + j/orig_fps` j ∈ [0, chunk_size)
    """
    dt_t = 1.0 / target_fps
    dt_o = 1.0 / orig_fps

    camera_ts = [round(i * dt_t, 6) for i in range(num_frames)]

    action_ts = [round(i * dt_t + j * dt_o, 6) for i in range(num_frames) for j in range(chunk_size)]

    delta = dict.fromkeys(camera_keys, camera_ts)
    delta.update(dict.fromkeys(action_keys, action_ts))
    return delta


def build_valid_indices(
    episodes_meta: list[dict],
    window_orig: int,  # 窗口覆盖的原始帧数 = num_frames * chunk_size
) -> list[list[int]]:
    """
    只保留在 episode 内部能完整采到整个窗口的起始帧绝对索引。
    对应 EpisodeData 中的 valid = max(0, total_num_frames - self.frames + 1) 逻辑。
    """
    valid = []
    for ep in episodes_meta:
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        last_valid = ep_end - window_orig  # 最后一个合法起始
        if last_valid >= ep_start:
            valid.append(list(range(ep_start, last_valid + 1)))
    return valid


class SingleResampledDataset(Dataset):
    """
    单个 LeRobot 数据集的时序重采样封装

    输出 shape:
      camera keys : [T, C, H, W]
      action keys : [T, chunk_size, action_dim]
    """

    def __init__(
        self,
        repo_id: str,
        root=None,
        num_frames: int = 5,
        target_fps: int = 24,
        image_size: Tuple[int, int] = (256, 256),
        episodes: list[int] | None = None,
        dataset: str = "libero",
        eval_sample_num: int | None = None,
    ):
        info = read_info(repo_id, root)
        orig_fps: int = info["fps"]

        if orig_fps % target_fps != 0:
            raise ValueError(
                f"target_fps ({target_fps}) 必须是数据集 orig_fps ({orig_fps}) 的因数，"
                f"当前 {orig_fps} % {target_fps} = {orig_fps % target_fps} ≠ 0。"
            )

        self.orig_fps = orig_fps
        self.target_fps = target_fps
        self.num_frames = num_frames
        self.chunk_size = orig_fps // target_fps  # action_chunk

        # info["features"]
        self.camera_keys = CAMERA_KEYS.get(dataset)
        self.action_keys = ACTION_KEYS.get(dataset)

        # camera_fps = info["features"][self.camera_keys[0]]["fps"]
        # action_fps = info["features"][self.action_keys[0]]["fps"]
        # assert camera_fps == action_fps == orig_fps, (
        #     f"Camera FPS({camera_fps}) and action FPS({action_fps}) should be equal to the overall FPS({orig_fps})."
        # )

        delta_timestamps = build_delta_timestamps(
            camera_keys=self.camera_keys,
            action_keys=self.action_keys,
            num_frames=num_frames,
            target_fps=target_fps,
            orig_fps=orig_fps,
            chunk_size=self.chunk_size,
        )

        image_transforms = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

        self._ds = LeRobotDataset(
            repo_id,
            root=root,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=1.0 / orig_fps - 1e-4,  # 略小于原始帧间隔
        )

        episode_meta = self._ds.meta.episodes

        if episodes is not None:
            episode_meta = episode_meta.filter(lambda x: x["episode_index"] in set(episodes))

        # 窗口需要覆盖的原始帧数：最后一个 action chunk 的末尾
        # 最后一帧 t = (T-1)/target_fps，其最后一个 action 在 t + (chunk_size-1)/orig_fps
        # 折算到原始帧数 = (T-1)*chunk_size + chunk_size = T*chunk_size
        window_orig = num_frames * self.chunk_size
        all_valid = build_valid_indices(episode_meta, window_orig)

        if eval_sample_num is not None:
            self._valid_indices = []
            for sub_list in all_valid:
                self._valid_indices.extend(random.sample(sub_list, eval_sample_num))
        else:
            self._valid_indices = list(itertools.chain.from_iterable(all_valid))

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        item = self._ds[self._valid_indices[idx]]

        # action: [T*chunk_size, D] → [T, chunk_size, D]
        for key in list(item):
            t = item[key]
            if key in self.action_keys and isinstance(t, torch.Tensor) and t.dim() == 2:
                item[key] = t.reshape(self.num_frames, self.chunk_size, -1)

        item["video"] = item.pop(self.camera_keys[0])
        item["action"] = item.pop(self.action_keys[0])

        return item


class ResampledLeRobotDataset(Dataset):
    """
    组合多个 LeRobot 数据集

    DataLoader 输出：
        videos   : [B, T, C, H, W]
        actions  : [B, T, chunk_size, action_dim]
        timesteps: [B, T, chunk_size]   （原始帧索引，与 EpisodeData 一致）
    """

    def __init__(
        self,
        config,
        eval_episode_num: int = 2,
        eval_sample_num: int | None = None,
        action_chunk_size: int | None = None,
    ):
        self.num_frames = config.frames
        self.target_fps = config.fps
        self.image_size = config.image_size

        self.eval_episode_num = eval_episode_num
        self.eval_sample_num = eval_sample_num

        self._user_action_chunk_size = action_chunk_size  # 用户意图（固定不变）
        self.action_chunk_size: int | None = action_chunk_size  # 实际生效值（add 后更新）
        self._dataset_chunk_sizes: list[int] = []  # 各数据集自身的 chunk_size

        self._datasets: list[SingleResampledDataset] = []
        self._cum_lengths: list[int] = []  # 累积长度，用于 O(log N) 查找

    def add(self, metadata_path: str) -> "ResampledLeRobotDataset":
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip() != ""]

        assert len(lines) == 1, (
            f"Lerobot data must require jsonl to have only one row. Currently, there are {len(lines)} rows"
        )

        metadata = json.loads(lines[0])
        repo_id = metadata["repo_id"]
        root = metadata.get("root")
        dataset = metadata["dataset"]

        info = read_info(repo_id, root)
        total = info["total_episodes"]
        if self.eval_episode_num > total:
            raise ValueError(f"eval_episode_num ({self.eval_episode_num}) > all episodes ({total})")
        if self.eval_sample_num is None:
            episodes = list(range(total))[: -(self.eval_episode_num - 1)]
        else:
            episodes = list(range(total - self.eval_episode_num, total))

        ds = SingleResampledDataset(
            repo_id=repo_id,
            root=root,
            num_frames=self.num_frames,
            target_fps=self.target_fps,
            image_size=self.image_size,
            episodes=episodes,
            dataset=dataset,
            eval_sample_num=self.eval_sample_num,
        )
        self._datasets.append(ds)
        prev = self._cum_lengths[-1] if self._cum_lengths else 0
        self._cum_lengths.append(prev + len(ds))

        ds_chunk = ds.chunk_size
        self._dataset_chunk_sizes.append(ds_chunk)

        if self._user_action_chunk_size is not None:
            if ds_chunk % self._user_action_chunk_size != 0:
                raise ValueError(
                    f"指定的 action_chunk_size ({self._user_action_chunk_size}) "
                    f"必须是所有数据集 chunk_size 的公因数，"
                    f"但数据集 {repo_id} 的 chunk_size={ds_chunk} 不能被 "
                    f"{self._user_action_chunk_size} 整除。"
                )
            # self.action_chunk_size 保持指定值不变
        else:
            self.action_chunk_size = reduce(gcd, self._dataset_chunk_sizes)

        overwatch.info(f"{len(ds):,} data loaded from {repo_id} | total: {self._cum_lengths[-1]:,}")
        return self

    def __len__(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def __getitem__(self, idx: int) -> dict:
        ds_idx = bisect.bisect_right(self._cum_lengths, idx)
        offset = self._cum_lengths[ds_idx - 1] if ds_idx > 0 else 0

        item = self._datasets[ds_idx][idx - offset]

        # 将当前数据集的 action [T, ds_chunk, D] 子采样到 [T, action_chunk_size, D]
        ds_chunk = self._datasets[ds_idx].chunk_size
        if ds_chunk != self.action_chunk_size:
            step = ds_chunk // self.action_chunk_size
            item["action"] = item["action"][:, ::step, :]  # 取每个 chunk 的第 0, step, 2*step... 帧
        item["action"] = item["action"][..., :7]

        item["dataset_index"] = torch.tensor(ds_idx)
        return item

    def __repr__(self) -> str:
        lines = [
            f"ResampledLeRobotDataset("
            f"num_frames={self.num_frames}, target_fps={self.target_fps}, "
            f"action_chunk_size={self.action_chunk_size})"
        ]
        for i, ds in enumerate(self._datasets):
            step = ds.chunk_size // self.action_chunk_size
            lines.append(
                f"  [{i}] repo={ds._ds.repo_id} | "
                f"orig_fps={ds.orig_fps} | ds_chunk={ds.chunk_size} | "
                f"subsample_step={step} | samples={len(ds):,}"
            )
        lines.append(f"  Total: {len(self):,} samples")
        return "\n".join(lines)


def collate_fn(inputs):
    """
    videos    : [B, T, C, H, W]
    actions   : [B, T, chunk_size, action_dim]
    """
    videos = torch.stack([x["video"] for x in inputs])
    actions = torch.stack([x["action"] for x in inputs]) if "action" in inputs[0].keys() else None

    return {
        "videos": videos,
        "actions": actions,
    }


if __name__ == "__main__":
    from univam.utils.args import load_args
    from univam.utils.data import set_seed

    args = load_args()
    set_seed(args.seed)

    dataset = ResampledLeRobotDataset(
        args.data,
        eval_episode_num=2,
        # eval_sample_num=args.train.eval_sample_num,
    )
    dataset.add(metadata_path="jsons/train_debug_part_0_lerobot.jsonl")

    dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
    data = next(iter(dataloader))

    print(f"Dataset length: {len(dataset)}")
    print(f"Video shape: {data['videos'].shape}")
    print(f"Action shape: {data['actions'].shape}")
