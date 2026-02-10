import os
import random

import jsonlines
import numpy as np
import torch
from PIL.Image import Resampling
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import functional as F

from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def set_seed(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def complex_to_device(complex, device, non_blocking=False):
    if complex is None:
        return complex
    if isinstance(complex, torch.Tensor):
        return complex.to(device, non_blocking=non_blocking)
    elif isinstance(complex, dict):
        return {k: complex_to_device(v, device, non_blocking=non_blocking) for k, v in complex.items()}
    elif isinstance(complex, list) or isinstance(complex, tuple):
        return [complex_to_device(e, device, non_blocking=non_blocking) for e in complex]
    elif (
        isinstance(complex, str) or isinstance(complex, bytes) or isinstance(complex, int) or isinstance(complex, float)
    ):
        return complex
    else:
        raise ValueError("Unsupported complex", complex)


def fp32_to_fp16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.half)
    elif isinstance(batch, list):
        new_batch = [fp32_to_fp16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_fp16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def fp32_to_bf16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.bfloat16)
    elif isinstance(batch, list):
        new_batch = [fp32_to_bf16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_bf16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_bf16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def move_to_cuda(batch):
    if not torch.cuda.is_available():
        return batch
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def check_tensor(obj, name, check_bound=1e4, check_std=1e3, _visited=None, force_output=False):
    if not int(os.environ.get("CHECK_TENSOR", 1)):
        return
    if _visited is None:
        _visited = set()
    if id(obj) in _visited:
        return False
    _visited.add(id(obj))

    # list / tuple
    if isinstance(obj, (list, tuple)):
        problem = False
        for i, v in enumerate(obj):
            if check_tensor(v, f"{name}[{i}]", _visited=_visited):
                problem = True
        return problem

    # dict
    if isinstance(obj, dict):
        problem = False
        for k, v in obj.items():
            if check_tensor(v, f"{name}['{k}']", _visited=_visited):
                problem = True
        return problem

    # not tensor
    if not isinstance(obj, torch.Tensor):
        return False

    t = obj
    problem_found = False

    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()

    if has_nan:
        overwatch.error(f"<{name}> 检测到 NaN")
        problem_found = True
    if has_inf:
        overwatch.error(f"<{name}> 检测到 Inf")
        problem_found = True

    try:
        minv = t.min().item()
        maxv = t.max().item()
        meanv = t.mean().item()
        stdv = t.std().item() if t.numel() > 1 else 0.0
    except Exception as e:
        overwatch.error(f"<{name}> 无法计算统计值: {e}")
        return True

    if abs(maxv) > float(check_bound) or abs(minv) > float(check_bound):
        overwatch.error(f"<{name}> 数值过大 (|value| > {check_bound}), 可能导致梯度爆炸")
        problem_found = True

    if stdv > float(check_std):
        overwatch.error(f"<{name}> 标准差过大 (std={stdv}), 数值不稳定")
        problem_found = True

    if t.numel() > 1 and stdv < 1e-12:
        overwatch.error(f"<{name}> 标准差过小 (=0), 张量可能塌陷/全常数")
        problem_found = True

    if torch.all(t == 0):
        overwatch.error(f"<{name}> 张量全为 0")
        problem_found = True

    if t.numel() > 1 and torch.all(t == t.flatten()[0]):
        overwatch.error(f"<{name}> 张量全为常数")
        problem_found = True

    if t.requires_grad and t.is_leaf and t.grad is not None:
        g = t.grad
        if torch.isnan(g).any():
            overwatch.error(f"<{name}> 梯度存在 NaN")
            problem_found = True
        if torch.isinf(g).any():
            overwatch.error(f"<{name}> 梯度存在 Inf")
            problem_found = True
        if g.abs().max().item() > 1e6:
            overwatch.error(f"<{name}> 梯度爆炸 (grad > 1e6)")
            problem_found = True

    # if problem_found:
    if force_output or problem_found:
        overwatch.error(f"<{name}> 基本信息: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
        overwatch.error(f"<{name}> 数值统计: min={minv}, max={maxv}, mean={meanv}, std={stdv}")

    return problem_found


class VideoData(Dataset):
    def __init__(self, config, flip_p: float = 0.5, device="cpu"):
        self.flip_p = flip_p
        self.device = device

        self.frames = config.frames
        self.image_size = config.image_size

        self.length = 0
        self.video_paths = []
        self.video_lengths = []

        self.video_mean = torch.tensor([127.5, 127.5, 127.5])
        self.video_std = torch.tensor([127.5, 127.5, 127.5])

    def add(self, metadata_path):
        this_length = 0
        this_video_paths = []
        this_video_lengths = []
        with open(metadata_path, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                this_video_paths.append(item["video"])

        for video_path in this_video_paths:
            decoder = VideoDecoder(
                video_path,
                seek_mode="exact",
                num_ffmpeg_threads=0,
                device=self.device,
            )
            total_num_frames = decoder.metadata.num_frames
            this_video_lengths.append(max(0, total_num_frames - self.frames + 1))

        this_length = sum(this_video_lengths)
        self.length += this_length
        self.video_paths.extend(this_video_paths)
        self.video_lengths.extend(this_video_lengths)

        overwatch.info(f"{this_length} data loaded from {metadata_path}")

    def idx_to_video_and_frame(self, idx: int):
        video_idx = -1
        total_frame = 0
        start_frame = 0

        while idx - total_frame >= 0:
            video_idx += 1
            total_frame += self.video_lengths[video_idx]

        start_frame = idx - total_frame + self.video_lengths[video_idx]
        return video_idx, start_frame

    def read_video_torchcodec(self, video_idx: int, start_frame: int):
        """
        Decode the video with torchcodec decoder.

        Args:
            video_path (`str`):
                Path to the video file.

        Returns:
            torch.Tensor
        """
        video_path = self.video_paths[video_idx]

        decoder = VideoDecoder(
            video_path,
            # Interestingly `exact` mode takes less than approximate when we load the whole video
            seek_mode="exact",
            # Allow FFmpeg decide on the number of threads for efficiency
            num_ffmpeg_threads=0,
            device=self.device,
        )

        indices = list(range(start_frame, start_frame + self.frames))
        video = decoder.get_frames_at(indices=indices).data.contiguous()
        video = self.apply_transformations(video)
        return video

    def apply_transformations(self, video: torch.Tensor):
        """
        Apply flip and reshape to the frames in the video.
        """
        video = F.resize(video, self.image_size, interpolation=Resampling.BICUBIC)
        if random.random() < self.flip_p:
            video = F.hflip(video)

        video = F.normalize(video.to(dtype=torch.float32), self.video_mean, self.video_std)
        return video

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            video_idx, start_frame = self.idx_to_video_and_frame(idx)
            try:
                video = self.read_video_torchcodec(video_idx, start_frame)
                inputs = {"video": video}
                break
            except Exception:
                overwatch.error(f"read {self.video_paths[video_idx]}, start_frame: {start_frame} error")
                idx = random.randint(0, self.length - 1)
        return inputs
