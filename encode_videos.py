"""
    VIDEO_DIR=/path/to/videos RESUME_PATH=./ckpts/cuphead/140000 \
    python encode_videos.py --config_path configs/debug.yaml
"""

import os

import torch
from PIL.Image import Resampling
from torchvision.transforms import functional as F

from univam.models.wanva import Wan22VisionModel
from univam.trainer import Trainer
from univam.utils.args import load_args
from univam.utils.data import set_seed
from univam.utils.dataloaders.video import ResampledVideoDecoder, normalize_video
from univam.utils.files import ensure_directory
from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif")


def list_videos(video_dir):
    files = [
        os.path.join(video_dir, f)
        for f in sorted(os.listdir(video_dir))
        if f.lower().endswith(VIDEO_EXTS)
    ]
    return files


def read_clip(video_path, frames, fps, image_size):
    """重采样到 fps，取前 frames 帧，做与训练一致的预处理。返回 [T, C, H, W]，归一化到 [-1,1]。"""
    decoder = ResampledVideoDecoder(video_path, target_fps=fps)  # 5fps
    total = decoder.metadata.num_frames                          # 取5帧

    if total < frames:
        raise ValueError(
            f"{video_path}: 重采样到 {fps}fps 后只有 {total} 帧，不足 {frames} 帧（视频太短，需 ≥ {frames / fps:.1f} 秒）"
        )

    idxs = list(range(frames))                  # 取重采样后的前5帧
    video = decoder.get_frames_at(idxs)         # [T, C, H, W]
    video = F.resize(video, image_size, interpolation=Resampling.BICUBIC)
    video = normalize_video(video)              # -> [-1, 1], float32
    return video


@torch.no_grad()
def main(args, video_dir, out_dir):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    overwatch.info("Building model...")
    model = Wan22VisionModel(args).to(device=device, dtype=dtype)
    model.eval()

    trainer = Trainer(args=args, model=model)
    trainer.load_checkpoint(args.resume_path)
    trainer.move_model_to_device()

    frames = args.data.frames
    fps = args.data.fps
    image_size = args.data.image_size

    if out_dir is not None:
        ensure_directory(out_dir)
    video_files = list_videos(video_dir)
    overwatch.info(f"Found {len(video_files)} videos in {video_dir}")

    for video_path in video_files:
        try:
            clip = read_clip(video_path, frames, fps, image_size)    # [T, C, H, W]
        except Exception as e:
            overwatch.error(f"Skip {video_path}: {e}")
            continue

        videos = clip.unsqueeze(0).to(device=device, dtype=dtype)   # [1, T, C, H, W]

        video_latents = model.wanvae.encode(videos)                 # ① VAE 潜变量
        video_embeds = model.encode(video_latents)                  # ③ 条件 token [1, num_token, hidden]

        # 默认存到视频同目录、同名 (video.mp4 -> video.pt)；如果设置OUT_DIR，存到指定目录
        if out_dir is None:
            out_path = os.path.splitext(video_path)[0] + ".pt"
        else:
            name = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(out_dir, f"{name}.pt")

        torch.save(video_embeds[0].cpu(), out_path)
        overwatch.info(f"Saved {out_path}  shape={tuple(video_embeds[0].shape)}  <- {video_path}")

    overwatch.info("Done.")


if __name__ == "__main__":
    args = load_args()
    args.data.image_size = [256, 256]

    args.resume_path = os.environ.get("RESUME_PATH", args.resume_path)
    video_dir = os.environ.get("VIDEO_DIR", getattr(args.data, "video_dir", None))
    out_dir = os.environ.get("OUT_DIR", None)  # 不设则存到每个视频的同目录

    assert video_dir is not None, "请通过环境变量 VIDEO_DIR 指定视频目录"
    main(args, video_dir, out_dir)
