import random

import decord
import jsonlines
import torch
from PIL.Image import Resampling
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def normalize_video(video: torch.Tensor):
    """
    Normalize the video frames.

    Args:
        video (torch.Tensor): [B, T, C, H, W] or [T, C, H, W]

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
        mean (torch.Tensor): [C]
        std (torch.Tensor): [C]

    Returns:
        torch.Tensor: denormalized video tensor
    """
    video = video.to(dtype=torch.float32)

    video = (video.add(1).mul(127.5)).clamp(0, 255).to(torch.uint8)

    return video


class ResampledVideoDecoder:
    def __init__(self, video_path: str, target_fps: float):
        decord.bridge.set_bridge("torch")
        self.target_fps = target_fps
        self.video = decord.VideoReader(video_path, ctx=decord.cpu(0))

        self.orig_total_frames = len(self.video)
        # decord 有 average FPS 属性，但不一定总是准确，可用 metadata
        try:
            self.orig_fps = float(self.video.get_avg_fps())
        except Exception:
            self.orig_fps = target_fps  # fallback

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
        if self.new_total_frames == 1:
            return [0] * len(target_indices)

        mapped = []
        for idx in target_indices:
            orig_idx = round(idx / (self.new_total_frames - 1) * (self.orig_total_frames - 1))
            orig_idx = min(self.orig_total_frames - 1, max(0, orig_idx))
            mapped.append(orig_idx)
        return mapped

    def get_frames_at(self, indices):
        mapped_indices = self._map_indices(indices)
        frames = self.video.get_batch(mapped_indices)  # [N, H, W, C]
        frames = frames.permute(0, 3, 1, 2)  # [N, C, H, W]
        return frames


class VideoData(Dataset):
    def __init__(self, config, flip_p: float = 0.5, eval_sample_num=None):
        self.flip_p = flip_p
        self.eval_sample_num = eval_sample_num

        self.fps = config.fps
        self.frames = config.frames
        self.image_size = config.image_size

        self.length = 0
        self.video_paths = []
        self.video_lengths = []
        self.video_start_indices = []

    def add(self, metadata_path):
        this_length = 0
        this_video_paths = []
        this_video_lengths = []
        this_video_start_indices = []

        with open(metadata_path, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                this_video_paths.append(item["video"])

        for video_path in this_video_paths:
            decoder = self._build_video_decoder(video_path, target_fps=self.fps)
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
    def _build_video_decoder(video_path, target_fps):
        return ResampledVideoDecoder(video_path, target_fps)

    def read_video_decord(self, video_idx: int, start_frame: int):
        """
        Decode the video using ResampledVideoDecoder.

        Args:
            video_idx (int): index of the video in self.video_paths
            start_frame (int): start frame index on target-fps timeline

        Returns:
            torch.Tensor: [T, C, H, W] video clip
        """
        video_path = self.video_paths[video_idx]

        decoder = self._build_video_decoder(video_path, target_fps=self.fps)

        indices = list(range(start_frame, start_frame + self.frames))
        video = decoder.get_frames_at(indices=indices).contiguous()  # [T, C, H, W]

        video = self.apply_transformations(video)
        return video

    def apply_transformations(self, video: torch.Tensor):
        """
        Apply flip and reshape to the frames in the video.
        """
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
                video = self.read_video_decord(video_idx, start_frame)
                inputs = {"video": video}
                break
            except Exception:
                overwatch.error(f"read {self.video_paths[video_idx]}, start_frame: {start_frame} error")
                idx = random.randint(0, self.length - 1)
        return inputs


def collate_fn(inputs):
    videos = torch.stack([input["video"] for input in inputs])
    return {"videos": videos}


if __name__ == "__main__":
    from univam.utils.args import load_args
    from univam.utils.data import set_seed

    args = load_args()
    set_seed(args.seed)

    dataset = VideoData(args.data)
    dataset.add(metadata_path="jsons/train_debug_part_0_video.jsonl")

    dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn, shuffle=True, drop_last=False)
    data = next(iter(dataloader))

    print(f"Dataset length: {len(dataset)}")
    print(f"Video shape: {data['videos'].shape}")
