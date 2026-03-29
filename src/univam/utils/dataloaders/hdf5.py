import json
import os
import random
from collections import OrderedDict

import cv2
import h5py
import jsonlines
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)

VIDEO_KEYS = {
    "robotwin": "/observation/head_camera/rgb",
    "XVLA": "/observations/images/cam_high",
    "libero": "/primary_images_jpeg",
}

ACTION_KEYS = {
    "robotwin": "/joint_action/vector",
    "XVLA": "/action",
    "libero": "/actions",
}


def save_image(img: torch.Tensor, path: str = "./output.png"):
    """
    Save a uint8 tensor image (C, H, W) to PNG.

    Args:
        img: torch.Tensor, shape (3, H, W), dtype=torch.uint8
        path: output file path (e.g., 'xxx.png')
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch.Tensor")

    if img.dtype != torch.uint8:
        raise ValueError(f"Expected dtype=torch.uint8, got {img.dtype}")

    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected shape (3, H, W), got {img.shape}")

    # (C, H, W) -> (H, W, C)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = np.ascontiguousarray(img_np)
    Image.fromarray(img_np).save(path)


def decode_hdf5_images(img_bytes):
    """
    img_bytes: ndarray shape (B,)
    return: torch tensor [B, C, H, W]
    """
    imgs = []

    for b in img_bytes:
        img_np = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        imgs.append(img)

    return torch.stack(imgs, dim=0)


def get_q01_q99():
    global q01, q99
    if not os.path.exists("./jsons/latent_action_q01_q99.json"):
        raise ValueError("./jsons/latent_action_q01_q99.json not found")
    with open("./jsons/latent_action_q01_q99.json", "r") as f:
        stats = json.load(f)
        if "latent_action" in stats:
            q01 = torch.tensor(stats["latent_action"]["q01"], dtype=torch.float32)
            q99 = torch.tensor(stats["latent_action"]["q99"], dtype=torch.float32)
        else:
            q01 = torch.tensor(stats["q01"], dtype=torch.float32)
            q99 = torch.tensor(stats["q99"], dtype=torch.float32)


def normalize_action(action):
    min_val = q01.to(action.device)
    max_val = q99.to(action.device)
    denom = max_val - min_val
    denom[denom == 0] = 1.0

    action = 2 * (action - min_val) / denom - 1
    action = torch.clamp(action, -1, 1)
    return action


def denormalize_action(action):
    min_val = q01.to(action.device)
    max_val = q99.to(action.device)
    denom = max_val - min_val

    action = (action + 1) / 2 * denom + min_val
    return action


def normalize_video(video: torch.Tensor):
    """
    Normalize the video frames.

    Args:
        video (torch.Tensor): [B, T, C, H, W] or [T, C, H, W]
        mean (torch.Tensor): [C]
        std (torch.Tensor): [C]

    Returns:
        torch.Tensor: normalized video tensor
    """
    video = video.to(dtype=torch.float32)

    mean = torch.tensor([127.5, 127.5, 127.5]).to(video)
    std = torch.tensor([127.5, 127.5, 127.5]).to(video)

    if video.dim() == 4:
        # [T, C, H, W]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif video.dim() == 5:
        # [B, T, C, H, W]
        mean = mean.view(1, 1, -1, 1, 1)
        std = std.view(1, 1, -1, 1, 1)
    else:
        raise ValueError(f"Unsupported video shape: {video.shape}")

    return (video - mean) / std


def denormalize_video(video: torch.Tensor):
    """
    Denormalize the video frames.

    Args:
        video (torch.Tensor): [B, T, C, H, W] or [T, C, H, W]

    Returns:
        torch.Tensor: denormalized video tensor
    """
    video = video.to(dtype=torch.float32)

    video = (video.add(1).mul(127.5)).clamp(0, 255).to(torch.uint8)

    return video


class ResampledHDF5Decoder:
    """
    HDF5 episode decoder with temporal resampling
    """

    def __init__(
        self,
        path: str,
        orig_fps: int,
        target_fps: int,
        dataset: str = "robotwin",
    ):
        self.path = path
        self.file = h5py.File(path, "r")
        self.dataset = dataset

        if dataset not in VIDEO_KEYS or dataset not in ACTION_KEYS:
            raise ValueError(f"Unsupported dataset: {dataset}")

        video_key = VIDEO_KEYS.get(dataset)
        action_key = ACTION_KEYS.get(dataset)

        self.video = self.file[video_key]
        self.action = self.file[action_key]

        if dataset == "libero":
            self.action = np.tile(self.action, (1, 2))

        self.orig_fps = orig_fps
        self.target_fps = target_fps

        if target_fps > orig_fps:
            raise ValueError("target_fps should not exceed orig_fps")

        self.orig_total_frames = self.video.shape[0]

        self.duration = self.orig_total_frames / self.orig_fps
        self.new_total_frames = max(1, int(round(self.duration * self.target_fps)))

        self.action_chunk = int(self.orig_fps // self.target_fps)

        self._metadata = self._build_metadata()

    def _build_metadata(self):
        class Meta:
            pass

        m = Meta()
        m.fps = self.target_fps
        m.num_frames = self.new_total_frames

        return m

    @property
    def metadata(self):
        return self._metadata

    def _map_indices(self, target_indices):
        step = self.orig_fps / self.target_fps
        mapped = []
        for idx in target_indices:
            orig_idx = int(round(idx * step))
            orig_idx = min(self.orig_total_frames - 1, max(0, orig_idx))
            mapped.append(orig_idx)
        return mapped

    def get_frames_and_actions(self, indices):
        mapped = self._map_indices(indices)
        video = decode_hdf5_images(self.video[mapped])

        if self.dataset == "libero":
            video = video[:, [2, 1, 0], :, :]

        actions = []
        timesteps = []
        for idx in mapped:
            start = idx
            end = start + self.action_chunk

            a = self.action[start:end]
            a = torch.from_numpy(a)
            t = torch.arange(start, end)

            actions.append(a)
            timesteps.append(t)

        actions = torch.stack(actions)
        timestep = torch.stack(timesteps)
        return video, actions, timestep


class EpisodeData(Dataset):
    def __init__(
        self,
        config,
        orig_fps: int,
        flip_p: float = 0.5,
        eval_sample_num=None,
        cache_size: int = 128,
    ):
        self.flip_p = flip_p
        self.orig_fps = orig_fps
        self.eval_sample_num = eval_sample_num

        self.fps = config.fps
        self.frames = config.frames
        self.image_size = config.image_size

        self.length = 0
        self.video_paths = []
        self.dataset_name = []
        self.video_lengths = []
        self.video_start_indices = []

        self.decoder_cache = OrderedDict()
        self.cache_size = cache_size
        get_q01_q99()

    def add(self, metadata_path):
        this_length = 0
        this_video_paths = []
        this_dataset_name = []
        this_video_lengths = []
        this_video_start_indices = []

        with open(metadata_path, "r", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                this_video_paths.append(item["hdf5"])
                this_dataset_name.append(item["dataset"])

        for path, dataset_name in zip(this_video_paths, this_dataset_name):
            decoder = self.build_decoder(path, dataset_name)

            total_num_frames = decoder.metadata.num_frames

            valid = max(0, total_num_frames - self.frames + 1)
            if valid <= 0:
                this_video_lengths.append(0)
                this_video_start_indices.append([])
                continue

            if self.eval_sample_num is not None:
                k = min(self.eval_sample_num, valid)
                starts = random.sample(range(valid), k)
                this_video_lengths.append(k)
                this_video_start_indices.append(starts)
            else:
                this_video_lengths.append(valid)
                this_video_start_indices.append(None)
            # overwatch.info(f"{valid} data loaded from {path}")

        this_length = sum(this_video_lengths)

        self.length += this_length

        self.video_paths.extend(this_video_paths)
        self.dataset_name.extend(this_dataset_name)
        self.video_lengths.extend(this_video_lengths)
        self.video_start_indices.extend(this_video_start_indices)

        overwatch.info(f"{this_length} data loaded from {metadata_path}")

    def idx_to_video_and_frame(self, idx):
        video_idx = -1
        total_frame = 0

        while idx - total_frame >= 0:
            video_idx += 1
            total_frame += self.video_lengths[video_idx]

        local_idx = idx - (total_frame - self.video_lengths[video_idx])

        if self.eval_sample_num is not None:
            start_frame = self.video_start_indices[video_idx][local_idx]
        else:
            start_frame = local_idx

        return video_idx, start_frame

    def build_decoder(self, path: str, dataset_name: str):
        if path in self.decoder_cache:
            decoder = self.decoder_cache.pop(path)
            self.decoder_cache[path] = decoder
            return decoder

        decoder = ResampledHDF5Decoder(
            path,
            orig_fps=self.orig_fps,
            target_fps=self.fps,
            dataset=dataset_name,
        )

        self.decoder_cache[path] = decoder

        if len(self.decoder_cache) > self.cache_size:
            self.decoder_cache.popitem(last=False)

        return decoder

    def read_episode(self, video_idx, start_frame):
        path = self.video_paths[video_idx]
        dataset_name = self.dataset_name[video_idx]
        decoder = self.build_decoder(path, dataset_name)

        indices = list(range(start_frame, start_frame + self.frames))
        video, actions, timestep = decoder.get_frames_and_actions(indices)

        video = self.apply_transformations(video)
        actions = normalize_action(actions)
        return video, actions, timestep

    def apply_transformations(self, video):
        video = F.resize(video, self.image_size, interpolation=Resampling.BICUBIC)

        if random.random() < self.flip_p:
            video = F.hflip(video)

        video = normalize_video(video)
        return video

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            video_idx, start_frame = self.idx_to_video_and_frame(idx)
            try:
                video, action, timestep = self.read_episode(video_idx, start_frame)
                return {
                    "video": video,
                    "action": action,
                    "timestep": timestep,
                }
            except Exception:
                overwatch.error(f"read {self.video_paths[video_idx]}, start_frame {start_frame} error")
                idx = random.randint(0, self.length - 1)


def collate_fn(inputs):
    videos = torch.stack([x["video"] for x in inputs])
    actions = torch.stack([x["action"] for x in inputs]) if "action" in inputs[0].keys() else None
    timesteps = torch.stack([x["timestep"] for x in inputs]) if "timestep" in inputs[0].keys() else None

    return {
        "videos": videos,
        "actions": actions,
        "timesteps": timesteps,
    }


if __name__ == "__main__":
    from univam.utils.args import load_args
    from univam.utils.data import set_seed

    args = load_args()
    set_seed(args.seed)

    dataset = EpisodeData(args.data, 24)
    dataset.add(metadata_path="jsons/train_debug_part_0_hdf5.jsonl")

    dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
    data = next(iter(dataloader))

    print(f"Dataset length: {len(dataset)}")
    print(f"Video shape: {data['videos'].shape}")
    print(f"Action shape: {data['actions'].shape}")
    print(f"Timestep shape: {data['timesteps'].shape}")
