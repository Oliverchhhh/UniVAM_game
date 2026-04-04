import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv()
local_dir = Path(os.environ.get("DATASETS_PATH", "./datasets"))

repo_ids = [
    "lerobot/libero_spatial_image",
    "lerobot/libero_object_image",
    "lerobot/libero_goal_image",
    "lerobot/libero_10_image",
    "hxma/RoboTwin-LeRobot-v3.0",
    "iAyoD/robocasa_mobile_turn_on_microwave_256_hybrid",
    "iAyoD/robocasa_mobile_close_drawer_256_hybrid",
    "iAyoD/robocasa_mobile_close_single_door_256_hybrid",
]

for repo_id in repo_ids:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir / repo_id,
    )
