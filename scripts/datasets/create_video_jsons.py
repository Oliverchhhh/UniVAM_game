import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


random.seed(33)


def create_split_jsonl(
    train_video_dirs: List[str],
    eval_video_dirs: List[str],
    dataset_name: str,
    shared_train_num: int = 3,
    eval_num: int = 3,
):
    video_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def collect_videos(dirs):
        dir_to_videos = defaultdict(list)
        for video_dir in dirs:
            for root, _, files in os.walk(video_dir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in video_EXTENSIONS:
                        abs_path = os.path.abspath(os.path.join(root, file))
                        dir_to_videos[video_dir].append(abs_path)
            overwatch.info(f"Collected {len(dir_to_videos[video_dir])} videos from {video_dir}")
        return dir_to_videos

    # collect all videos
    train_dir_to_videos = collect_videos(train_video_dirs)
    eval_dir_to_videos = collect_videos(eval_video_dirs)

    overwatch.info(
        f"Collection completed: {sum(len(v) for v in train_dir_to_videos.values())} train videos, "
        f"{sum(len(v) for v in eval_dir_to_videos.values())} eval videos in total."
    )

    os.makedirs("./jsons", exist_ok=True)

    eval_videos_entries = []
    cnt = 0
    # 处理 train_video_dirs：全部写入 train.jsonl，每个目录抽取 shared_train_num 个写入 eval.jsonl
    for video_dir, videos in train_dir_to_videos.items():
        train_videos_set = set()
        if len(videos) < shared_train_num:
            raise ValueError(f"训练目录 {video_dir} 中图像不足 {shared_train_num}，无法抽取 eval 用图像。")

        # 生成名字后缀（取目录名）
        suffix = f"part_{cnt}"
        train_jsonl_path = f"./jsons/train_{dataset_name}_{suffix}.jsonl"
        cnt += 1

        train_videos_set.update(videos)
        shared_for_eval = random.sample(videos, shared_train_num)
        for video in shared_for_eval:
            eval_videos_entries.append({"video": video, "//": "from-train-shared"})

        with open(train_jsonl_path, "w", encoding="utf-8") as f_train:
            for video in sorted(train_videos_set):
                f_train.write(json.dumps({"video": video}) + "\n")

        overwatch.info(
            f"Train part {suffix}: wrote {len(train_videos_set)} videos to {train_jsonl_path}, {shared_train_num} shared with eval."
        )

    # 处理 eval_video_dirs：每个目录抽取 eval_num 个 eval-only video
    for video_dir, videos in eval_dir_to_videos.items():
        if len(videos) < eval_num:
            raise ValueError(f"评估目录 {video_dir} 中图像不足 {eval_num}，无法抽取 eval 用图像。")
        eval_only = random.sample(videos, eval_num)
        for video in eval_only:
            eval_videos_entries.append({"video": video, "//": "from-eval-only"})

    eval_jsonl_path = f"./jsons/eval_{dataset_name}.jsonl"
    with open(eval_jsonl_path, "w", encoding="utf-8") as f_eval:
        for entry in eval_videos_entries:
            f_eval.write(json.dumps(entry) + "\n")

    overwatch.info(f"Eval: {len(eval_videos_entries)} videos written to {eval_jsonl_path}")

    datasets = [f"train_{dataset_name}_part_{i}.jsonl" for i in range(cnt)]
    ratios = [1 / cnt for _ in range(cnt)]
    train_info = {"datasets": datasets, "ratios": ratios}
    train_json_path = f"./jsons/train_{dataset_name}.json"
    with open(train_json_path, "w", encoding="utf-8") as f_train:
        json.dump(train_info, f_train, indent=4)
    overwatch.info(f"Train config written to {train_json_path}, {cnt} parts.")

    datasets = [f"eval_{dataset_name}.jsonl"]
    ratios = [1]
    eval_info = {"datasets": datasets, "ratios": ratios}
    eval_json_path = f"./jsons/eval_{dataset_name}.json"
    with open(eval_json_path, "w", encoding="utf-8") as f_eval:
        json.dump(eval_info, f_eval, indent=4)
    overwatch.info(f"Eval config written to {eval_json_path}")


if __name__ == "__main__":
    load_dotenv()

    dataset_path = Path(os.environ.get("DATASETS_PATH", "./datasets"))

    train_video_dirs = [
    Path("/mnt/workspace/datasets/games/cuphead_supplementary"),
        ]
    eval_video_dirs = [
            Path("/mnt/workspace/datasets/games/cuphead_supplementary"),
        ]


    create_split_jsonl(train_video_dirs, eval_video_dirs, "fractal", shared_train_num=1, eval_num=1)
