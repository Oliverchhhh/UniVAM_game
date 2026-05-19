import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv()
local_dir = Path(os.environ.get("DATASETS_PATH", "./datasets"))

repo_ids = [
    "physical-intelligence/libero",
    "ericonaldo/Bridge-V2",
    "ucasmichael/fractal20220817_data"
]

for repo_id in repo_ids:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir / repo_id,
    )
