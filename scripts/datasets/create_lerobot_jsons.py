import json
import os
import random
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from univam.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)
random.seed(33)


def create_split_jsonl(
    train_dataset_ls: List[Tuple[str, Path]],
    eval_dataset_ls: List[Tuple[str, Path]],
    dataset_name: str,
):
    os.makedirs("./jsons", exist_ok=True)

    cnt = 0
    for ds_name, root_path in train_dataset_ls:
        suffix = f"part_{cnt}"
        train_jsonl_path = f"./jsons/train_{dataset_name}_{suffix}.jsonl"
        cnt += 1

        with open(train_jsonl_path, "w", encoding="utf-8") as f_train:
            f_train.write(
                json.dumps(
                    {
                        "repo_id": str(root_path),
                        "dataset": ds_name,
                    },
                    ensure_ascii=False,
                )
            )

    # train config
    datasets = [f"train_{dataset_name}_part_{i}.jsonl" for i in range(cnt)]
    ratios = [1 / cnt for _ in range(cnt)]

    with open(f"./jsons/train_{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": datasets,
                "ratios": ratios,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    cnt = 0
    for ds_name, root_path in eval_dataset_ls:
        suffix = f"part_{cnt}"
        eval_jsonl_path = f"./jsons/eval_{dataset_name}_{suffix}.jsonl"
        cnt += 1

        with open(eval_jsonl_path, "w", encoding="utf-8") as f_eval:
            f_eval.write(
                json.dumps(
                    {
                        "repo_id": str(root_path),
                        "dataset": ds_name,
                    },
                    ensure_ascii=False,
                )
            )

    # eval config
    datasets = [f"eval_{dataset_name}_part_{i}.jsonl" for i in range(cnt)]
    ratios = [1 / cnt for _ in range(cnt)]

    with open(f"./jsons/eval_{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": datasets,
                "ratios": ratios,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    load_dotenv()

    dataset_path = Path(os.environ.get("DATASETS_PATH", "./datasets"))

    robotwin_tasks = [
        "blocks_ranking_size",
        "shake_bottle",
        "stack_blocks_three",
        "place_a2b_left",
        "move_can_pot",
        "place_mouse_pad",
        "place_object_stand",
        "move_pillbottle_pad",
        "handover_block",
        "scan_object",
        "place_phone_stand",
        "click_alarmclock",
        "dump_bin_bigbin",
        "place_object_scale",
        "stamp_seal",
        "adjust_bottle",
        "place_bread_basket",
        "pick_diverse_bottles",
        "stack_bowls_three",
        "move_playingcard_away",
        "handover_mic",
        "hanging_mug",
        "rotate_qrcode",
        "place_fan",
        "press_stapler",
        "turn_switch",
        "put_object_cabinet",
        "open_microwave",
        "place_container_plate",
        "place_burger_fries",
        "stack_bowls_two",
        "place_shoe",
        "place_can_basket",
        "move_stapler_pad",
        "place_a2b_right",
        "pick_dual_bottles",
        "click_bell",
        "place_bread_skillet",
        "lift_pot",
        "beat_block_hammer",
        "shake_bottle_horizontally",
        "put_bottles_dustbin",
        "place_empty_cup",
        "blocks_ranking_rgb",
        "place_dual_shoes",
        "open_laptop",
        "place_object_basket",
        "grab_roller",
        "place_cans_plasticbox",
        "stack_blocks_two",
    ]

    robotwin_dataset_clean = [
        ("robotwin", dataset_path / f"hxma/RoboTwin-LeRobot-v3.0/{task_name}/aloha-agilex_clean_50/")
        for task_name in robotwin_tasks
    ]
    robotwin_dataset_random = [
        ("robotwin", dataset_path / f"hxma/RoboTwin-LeRobot-v3.0/{task_name}/aloha-agilex_randomized_500/")
        for task_name in robotwin_tasks
    ]

    train_dataset_ls = [
        ("libero", dataset_path / "lerobot/libero_10_image"),
        ("libero", dataset_path / "lerobot/libero_goal_image"),
        ("libero", dataset_path / "lerobot/libero_object_image"),
        ("libero", dataset_path / "lerobot/libero_spatial_image"),
        ("robocasa", dataset_path / "iAyoD/robocasa_mobile_close_drawer_256_hybrid"),
        ("robocasa", dataset_path / "iAyoD/robocasa_mobile_close_single_door_256_hybrid"),
        ("robocasa", dataset_path / "iAyoD/robocasa_mobile_turn_on_microwave_256_hybrid"),
    ]
    train_dataset_ls.extend(robotwin_dataset_clean)
    train_dataset_ls.extend(robotwin_dataset_random)

    eval_dataset_ls = [
        ("libero", dataset_path / "lerobot/libero_10_image"),
        ("libero", dataset_path / "lerobot/libero_goal_image"),
        ("libero", dataset_path / "lerobot/libero_object_image"),
        ("libero", dataset_path / "lerobot/libero_spatial_image"),
        ("robotwin", dataset_path / "hxma/RoboTwin-LeRobot-v3.0/pick_diverse_bottles/aloha-agilex_randomized_500/"),
        ("robocasa", dataset_path / "iAyoD/robocasa_mobile_close_drawer_256_hybrid"),
        ("robocasa", dataset_path / "iAyoD/robocasa_mobile_close_single_door_256_hybrid"),
        ("robocasa", dataset_path / "iAyoD/robocasa_mobile_turn_on_microwave_256_hybrid"),
    ]

    create_split_jsonl(
        train_dataset_ls,
        eval_dataset_ls,
        dataset_name="debug",
    )
