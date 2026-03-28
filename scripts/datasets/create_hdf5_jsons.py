import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


random.seed(33)


def create_split_jsonl(
    train_hdf5_map: Dict[str, str],
    eval_hdf5_map: Dict[str, str],
    dataset_name: str,
    shared_train_num: int = 3,
    eval_num: int = 3,
):
    hdf5_EXTENSIONS = {".hdf5"}

    def collect_hdf5s(hdf5_map):
        dir_to_hdf5s = defaultdict(list)

        for ds_name, hdf5_dir in hdf5_map.items():
            for root, _, files in os.walk(hdf5_dir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in hdf5_EXTENSIONS:
                        abs_path = os.path.abspath(os.path.join(root, file))
                        dir_to_hdf5s[hdf5_dir].append((abs_path, ds_name))

            overwatch.info(f"Collected {len(dir_to_hdf5s[hdf5_dir])} hdf5s from {hdf5_dir} (dataset={ds_name})")

        return dir_to_hdf5s

    # collect
    train_dir_to_hdf5s = collect_hdf5s(train_hdf5_map)
    eval_dir_to_hdf5s = collect_hdf5s(eval_hdf5_map)

    overwatch.info(
        f"Collection completed: {sum(len(v) for v in train_dir_to_hdf5s.values())} train hdf5s, "
        f"{sum(len(v) for v in eval_dir_to_hdf5s.values())} eval hdf5s in total."
    )

    os.makedirs("./jsons", exist_ok=True)

    eval_hdf5s_entries = []
    cnt = 0

    # ===== train 部分 =====
    for hdf5_dir, hdf5s in train_dir_to_hdf5s.items():
        if len(hdf5s) < shared_train_num:
            raise ValueError(f"{hdf5_dir} 不足 {shared_train_num}")

        suffix = f"part_{cnt}"
        train_jsonl_path = f"./jsons/train_{dataset_name}_{suffix}.jsonl"
        cnt += 1

        shared_for_eval = random.sample(hdf5s, shared_train_num)

        for path, ds_name in shared_for_eval:
            eval_hdf5s_entries.append({"hdf5": path, "dataset": ds_name, "//": "from-train-shared"})

        with open(train_jsonl_path, "w", encoding="utf-8") as f_train:
            for path, ds_name in sorted(hdf5s):
                f_train.write(json.dumps({"hdf5": path, "dataset": ds_name}) + "\n")

        overwatch.info(f"Train {suffix}: {len(hdf5s)} samples, shared {shared_train_num}")

    # ===== eval-only =====
    for hdf5_dir, hdf5s in eval_dir_to_hdf5s.items():
        if len(hdf5s) < eval_num:
            raise ValueError(f"{hdf5_dir} 不足 {eval_num}")

        eval_only = random.sample(hdf5s, eval_num)

        for path, ds_name in eval_only:
            eval_hdf5s_entries.append({"hdf5": path, "dataset": ds_name, "//": "from-eval-only"})

    # 写 eval jsonl
    eval_jsonl_path = f"./jsons/eval_{dataset_name}.jsonl"
    with open(eval_jsonl_path, "w", encoding="utf-8") as f_eval:
        for entry in eval_hdf5s_entries:
            f_eval.write(json.dumps(entry) + "\n")

    overwatch.info(f"Eval: {len(eval_hdf5s_entries)} samples")

    # train config
    datasets = [f"train_{dataset_name}_part_{i}.jsonl" for i in range(cnt)]
    ratios = [1 / cnt for _ in range(cnt)]
    with open(f"./jsons/train_{dataset_name}.json", "w") as f:
        json.dump({"datasets": datasets, "ratios": ratios}, f, indent=4)

    # eval config
    with open(f"./jsons/eval_{dataset_name}.json", "w") as f:
        json.dump({"datasets": [f"eval_{dataset_name}.jsonl"], "ratios": [1]}, f, indent=4)


if __name__ == "__main__":
    load_dotenv()

    dataset_path = Path(os.environ.get("DATASETS_PATH", "./datasets"))

    train_hdf5_map = {
        "robotwin": dataset_path / "robotwin/train",
        "XVLA": dataset_path / "XVLA-Soft-Fold/train",
        "libero": dataset_path / "LIBERO-Cosmos-Policy/all_episodes/train",
    }

    eval_hdf5_map = {
        "robotwin": dataset_path / "robotwin/eval",
        "XVLA": dataset_path / "XVLA-Soft-Fold/eval",
        "libero": dataset_path / "LIBERO-Cosmos-Policy/all_episodes/eval",
    }

    create_split_jsonl(
        train_hdf5_map,
        eval_hdf5_map,
        dataset_name="mix",
        shared_train_num=1,
        eval_num=1,
    )
