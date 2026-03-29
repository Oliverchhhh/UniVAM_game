import os

import numpy as np
import torch
from PIL import Image

from univam.models.wanva import Wan22VisionActionModel
from univam.trainer import Trainer
from univam.utils.args import load_args
from univam.utils.data import load_multi_datasets_form_json, set_seed
from univam.utils.files import ensure_directory
from univam.utils.metrics import calculate_psnr, calculate_ssim
from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def main(args):
    overwatch.info("Loading datasets...")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    eval_dataloader = load_multi_datasets_form_json(
        args.data,
        json_path=args.data.eval_json_path,
        flip_p=0,
        local_batch_size=args.train.local_batch_size,
        num_workers=args.data.num_workers,
        is_infinite=False,
        shuffle=False,
        drop_last=False,
        eval_sample_num=args.train.eval_sample_num,
        make_single_dataset=True,
    )

    overwatch.info("Building model...")
    model = Wan22VisionActionModel(args).to(device=device, dtype=dtype)
    model.eval()

    trainer = Trainer(args=args, model=model)
    trainer.load_checkpoint(args.resume_path)
    trainer.move_model_to_device()

    label_videos = []
    pred_videos = []

    for batch in eval_dataloader:
        inputs = trainer.prepare_batch(batch)
        outputs = model(inputs)

        label_video = inputs["videos"]
        pred_video = outputs["videos"]

        label_video = torch.clamp((label_video + 1) / 2, 0, 1)
        pred_video = torch.clamp((pred_video + 1) / 2, 0, 1)

        label_videos.append(label_video)
        pred_videos.append(pred_video)

    label_videos = torch.cat(label_videos, dim=0)
    pred_videos = torch.cat(pred_videos, dim=0)

    psnr = calculate_psnr(pred_videos, label_videos)
    ssim = calculate_ssim(pred_videos, label_videos)

    overwatch.info(f"PSNR: {psnr:.4f}")
    overwatch.info(f"SSIM: {ssim:.4f}")

    video_path = os.path.join("tests", "eval")
    gt_video_path = os.path.join(video_path, "gt")
    pred_video_path = os.path.join(video_path, "pred")
    ensure_directory(video_path)
    ensure_directory(gt_video_path)
    ensure_directory(pred_video_path)

    for i in range(pred_videos.shape[0]):
        gt_np = (label_videos[i].permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)
        pred_np = (pred_videos[i].permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)

        gt_frames = [Image.fromarray(frame) for frame in gt_np]
        pred_frames = [Image.fromarray(frame) for frame in pred_np]

        widths, heights = zip(*(img.size for img in gt_frames))
        total_width = sum(widths)
        max_height = max(heights)
        gt_concat = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in gt_frames:
            gt_concat.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        gt_concat.save(os.path.join(gt_video_path, f"{i:02d}_gt_video.jpg"))

        widths, heights = zip(*(img.size for img in pred_frames))
        total_width = sum(widths)
        max_height = max(heights)
        pred_concat = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in pred_frames:
            pred_concat.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        pred_concat.save(os.path.join(pred_video_path, f"{i:02d}_pred_video.jpg"))

        if pred_concat.size != gt_concat.size:
            pred_concat = pred_concat.resize(gt_concat.size)
        w, h = gt_concat.size
        final_img = Image.new("RGB", (w, h * 2))
        final_img.paste(gt_concat, (0, 0))
        final_img.paste(pred_concat, (0, h))
        final_img.save(os.path.join(video_path, f"{i:02d}.jpg"))


if __name__ == "__main__":
    args = load_args()
    args.image_size = [480, 640]
    args.data.eval_json_path = os.environ.get("EVAL_JSON_PATH", args.data.eval_json_path)
    args.resume_path = os.environ.get("RESUME_PATH", args.resume_path)
    main(args)
