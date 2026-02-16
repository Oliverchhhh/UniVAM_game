import json
import math
import os
import random

# from functools import lru_cache
import jsonlines
import numpy as np
import torch
import torch.distributed as dist
from PIL.Image import Resampling
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler
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


def get_loader_info(dataset_len, epochs, bsz, gradient_accumulate_steps):
    images_per_gpu = bsz
    images_per_batch = bsz * overwatch.world_size()
    iter_per_ep = dataset_len // (bsz * overwatch.world_size() * gradient_accumulate_steps)
    num_iters = iter_per_ep * epochs
    loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
    return loader_info


class ResampledVideoDecoder:
    """
    A wrapper over VideoDecoder that provides temporal resampling
    to a target fps using strict linear time mapping.
    """

    def __init__(self, decoder: VideoDecoder, target_fps: float):
        self.decoder = decoder
        self.target_fps = target_fps

        meta = decoder.metadata
        self.orig_fps = float(meta.average_fps_from_header)
        self.orig_total_frames = int(meta.num_frames)

        if (target_fps - self.orig_fps) > 0.1:
            raise ValueError(f"Target fps: {target_fps} should not be larger than original fps: {self.orig_fps}")

        self.duration = self.orig_total_frames / self.orig_fps

        self.new_total_frames = max(1, int(round(self.duration * self.target_fps)))

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
        """
        将 target-fps 时间轴上的 indices
        映射到原视频时间轴
        """
        if self.new_total_frames == 1:
            return [0] * len(target_indices)

        mapped = []

        for idx in target_indices:
            orig_idx = round(idx / (self.new_total_frames - 1) * (self.orig_total_frames - 1))
            orig_idx = min(self.orig_total_frames - 1, max(0, orig_idx))
            mapped.append(orig_idx)

        return mapped

    def get_frames_at(self, indices):
        """
        indices 是 target-fps 时间轴上的索引
        """
        mapped_indices = self._map_indices(indices)
        return self.decoder.get_frames_at(mapped_indices)


class VideoData(Dataset):
    def __init__(self, config, flip_p: float = 0.5, device="cpu", eval_sample_num=None):
        self.flip_p = flip_p
        self.device = device
        self.eval_sample_num = eval_sample_num

        self.fps = config.fps
        self.frames = config.frames
        self.image_size = config.image_size

        self.length = 0
        self.video_paths = []
        self.video_lengths = []
        self.video_start_indices = []

        self.video_mean = torch.tensor([127.5, 127.5, 127.5])
        self.video_std = torch.tensor([127.5, 127.5, 127.5])

    def add(self, metadata_path):
        this_length = 0
        this_video_paths = []
        this_video_lengths = []
        this_video_start_indices = []

        with open(metadata_path, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                this_video_paths.append(item["video"])

        for video_path in this_video_paths:
            decoder = self.build_video_decoder(video_path)
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

        this_length = sum(this_video_lengths)

        self.length += this_length
        self.video_paths.extend(this_video_paths)
        self.video_lengths.extend(this_video_lengths)
        self.video_start_indices.extend(this_video_start_indices)

        overwatch.info(f"{this_length} data loaded from {metadata_path}")

    def idx_to_video_and_frame(self, idx: int):
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

    @staticmethod
    # @lru_cache(maxsize=16)
    def _build_video_decoder(video_path, target_fps, device="cpu"):
        decoder = VideoDecoder(
            video_path,
            # Interestingly `exact` mode takes less than approximate when we load the whole video
            seek_mode="exact",
            # Allow FFmpeg decide on the number of threads for efficiency
            num_ffmpeg_threads=0,
            device=device,
        )
        return ResampledVideoDecoder(decoder, target_fps)

    def build_video_decoder(self, video_path):
        return self._build_video_decoder(video_path, self.fps, self.device)

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

        decoder = self.build_video_decoder(video_path)
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


def collate_fn(inputs):
    videos = torch.stack([input["video"] for input in inputs])
    return {"videos": videos}


def worker_init_fn(worker_id):
    seed = 33 + worker_id
    np.random.seed(seed)
    random.seed(seed)


class InfiniteDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        """
        无限循环分布式采样器。
        :param dataset: 数据集
        :param num_replicas: 总共的设备数量
        :param rank: 当前设备的 rank
        :param shuffle: 是否随机打乱
        """
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self._epoch = 0

    def __iter__(self):
        """
        无限循环返回索引。
        """
        while True:
            self.set_epoch(self._epoch)
            self._epoch += 1
            indices = super().__iter__()
            yield from indices

    def __len__(self):
        return len(self.dataset)


class InfiniteMultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, sample_per_dataset, shuffle=True):
        """
        多任务批量采样器，支持 Lightning 的分布式模式。
        :param datasets: 多个数据集的列表
        :param batch_size: 每个 batch 的大小
        :param drop_last: 是否丢弃最后一个不足 batch_size 的 batch
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_datasets = len(self.datasets)
        self.samples_per_dataset = sample_per_dataset
        # self.remaining_samples = batch_size % self.num_datasets
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]

        self.cumulative_sizes = [0] + self.dataset_lengths

        for i in range(1, len(self.cumulative_sizes)):
            self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

        self.cur_idx = 0

        # 为每个数据集创建无限采样器
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.samplers = [
            InfiniteDistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=shuffle)
            for dataset in datasets
        ]
        self.iterators = [iter(sampler) for sampler in self.samplers]

    def __iter__(self):
        """
        无限生成每个 batch 的样本索引。
        """
        while True:
            batch = []
            for i in range(len(self.iterators)):
                iterator = self.iterators[i]
                for _ in range(self.samples_per_dataset[i]):
                    batch.append(next(iterator) + self.cumulative_sizes[i])
            yield batch

    def __len__(self):
        return sum(self.dataset_lengths)


class FiniteMultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, sample_per_dataset, drop_last=False, shuffle=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.samples_per_dataset = sample_per_dataset
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = [0] + self.dataset_lengths

        for i in range(1, len(self.cumulative_sizes)):
            self.cumulative_sizes[i] += self.cumulative_sizes[i - 1]

        self.drop_last = drop_last
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1

        # 初始化标准分布式采样器
        self.samplers = [
            DistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=shuffle)
            for dataset in datasets
        ]
        self.iterators = [iter(sampler) for sampler in self.samplers]
        # 计算每个数据集还剩多少样本
        self.remaining_samples = [len(sampler) for sampler in self.samplers]

    def __iter__(self):
        iterators = [iter(sampler) for sampler in self.samplers]
        remaining_samples = self.remaining_samples.copy()

        while sum(remaining_samples) > 0:
            batch = []
            for i, iterator in enumerate(iterators):
                num_samples = min(self.samples_per_dataset[i], remaining_samples[i])
                for _ in range(num_samples):
                    try:
                        idx = next(iterator)
                        batch.append(idx + self.cumulative_sizes[i])
                        remaining_samples[i] -= 1
                    except StopIteration:
                        remaining_samples[i] = 0
                        break  # 当前dataset采样完毕

            if len(batch) == 0:
                break

            # 根据drop_last判断batch大小
            if self.drop_last and len(batch) < self.batch_size:
                break

            yield batch

    def __len__(self):
        # 总的batch数量（近似值）
        total_samples = sum(self.dataset_lengths)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class MultiDatasetWrapper(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_lengths = [len(ds) for ds in datasets]

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, index):
        cumulative_sizes = 0
        for dataset, length in zip(self.datasets, self.dataset_lengths):
            if index < cumulative_sizes + length:
                return dataset[index - cumulative_sizes]
            cumulative_sizes += length
        raise IndexError("Index out of range")


def load_unsampler_datasets_from_json(
    config,
    json_path,
    flip_p,
    local_batch_size,
    num_workers=8,
    is_infinite=True,
    shuffle=True,
    drop_last=False,
    eval_sample_num=None,
    device="cpu",
):
    dataset = VideoData(config, flip_p=flip_p, device=device, eval_sample_num=eval_sample_num)

    with open(json_path, "r") as f:
        meta_infos = json.load(f)
    dataset_paths = meta_infos["datasets"]

    for dataset_path in dataset_paths:
        dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
        dataset.add(dataset_path)

    rank = dist.get_rank() if dist.is_initialized() else 0
    num_replicas = dist.get_world_size() if dist.is_initialized() else 1

    if is_infinite:
        sampler = InfiniteDistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
        )

    return dataloader


def load_multi_datasets_form_json(
    config,
    json_path,
    flip_p,
    local_batch_size,
    num_workers=8,
    is_infinite=True,
    shuffle=True,
    drop_last=False,
    eval_sample_num=None,
    make_single_dataset=False,
    device="cpu",
):
    if make_single_dataset:
        return load_unsampler_datasets_from_json(
            config=config,
            json_path=json_path,
            flip_p=flip_p,
            local_batch_size=local_batch_size,
            num_workers=num_workers,
            is_infinite=is_infinite,
            shuffle=shuffle,
            drop_last=drop_last,
            eval_sample_num=eval_sample_num,
            device=device,
        )

    with open(json_path, "r") as f:
        meta_infos = json.load(f)
    dataset_paths = meta_infos["datasets"]
    ratios = meta_infos["ratios"]

    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    assert len(ratios) == len(dataset_paths), "Each dataset must have a corresponding ratio"

    datasets = []

    for dataset_path in dataset_paths:
        dataset_path = os.path.join(os.path.dirname(json_path), dataset_path)
        dataset = VideoData(config, flip_p=flip_p, device=device, eval_sample_num=eval_sample_num)
        dataset.add(dataset_path)
        datasets.append(dataset)

    sample_per_dataset = [max(1, math.floor(r * local_batch_size)) for r in ratios]

    total = sum(sample_per_dataset)
    if total < local_batch_size:
        sample_per_dataset[-1] += local_batch_size - total
    elif total > local_batch_size:
        sample_per_dataset[-1] -= total - local_batch_size

    wrapped_dataset = MultiDatasetWrapper(datasets)

    if is_infinite:
        batch_sampler = InfiniteMultiTaskBatchSampler(
            datasets, local_batch_size, sample_per_dataset=sample_per_dataset, shuffle=shuffle
        )
    else:
        batch_sampler = FiniteMultiTaskBatchSampler(
            datasets, local_batch_size, sample_per_dataset=sample_per_dataset, shuffle=shuffle, drop_last=drop_last
        )

    dataloader = DataLoader(
        wrapped_dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    return dataloader


if __name__ == "__main__":
    from univam.utils.args import load_args

    args = load_args()

    # test for single dataset
    dataset = VideoData(args.data)
    dataset.add(metadata_path="jsons/train_debug_part_0.jsonl")

    dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
    data = next(iter(dataloader))

    print(f"Dataset length: {len(dataset)}")
    print(f"Video shape: {data['videos'].shape}")

    # test for multi datasets / dataloader
    dataloader = load_multi_datasets_form_json(
        args.data,
        json_path=args.data.train_json_path,
        flip_p=0,
        local_batch_size=32,
        num_workers=0,
        is_infinite=False,
        shuffle=False,
        drop_last=False,
        make_single_dataset=True,
    )

    data = next(iter(dataloader))

    print(f"Dataloader length (Batches): {len(dataloader)}")
    print(f"Video shape: {data['videos'].shape}")
