"""Upload specified checkpoints to Hugging Face repo STA-I-R/video_stamo.

Usage:
    python scripts/upload_to_hf.py

Make sure you have logged in first:
    huggingface-cli login
"""

import os
from huggingface_hub import HfApi

REPO_ID = "STA-I-R/video_stamo"

CHECKPOINTS = [
    {
        "local_dir": "ckpts/libero/110k_2",
        "repo_dir": "libero/110k_2",
    },
    {
        "local_dir": "ckpts/fractal/76000",
        "repo_dir": "fractal/76000",
    },
    {
        "local_dir": "ckpts/bridgev2/137k",
        "repo_dir": "bridgev2/137k",
    },
]

# Also upload the config.yaml for each task
TASK_CONFIGS = [
    ("ckpts/libero/config.yaml", "libero/config.yaml"),
    ("ckpts/fractal/config.yaml", "fractal/config.yaml"),
    ("ckpts/bridgev2/config.yaml", "bridgev2/config.yaml"),
]


def main():
    # 移除镜像站，确保直连 HF 官方服务器上传
    os.environ.pop("HF_ENDPOINT", None)
    os.environ.pop("HF_MIRROR", None)

    api = HfApi()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for item in CHECKPOINTS:
        local_path = os.path.join(project_root, item["local_dir"])
        print(f"Uploading {item['local_dir']} -> {item['repo_dir']} ...")
        api.upload_folder(
            folder_path=local_path,
            repo_id=REPO_ID,
            path_in_repo=item["repo_dir"],
            commit_message=f"Upload checkpoint: {item['repo_dir']}",
        )
        print(f"  Done: {item['repo_dir']}")

    for local_rel, repo_path in TASK_CONFIGS:
        local_path = os.path.join(project_root, local_rel)
        if os.path.exists(local_path):
            print(f"Uploading config {local_rel} -> {repo_path}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                commit_message=f"Upload config: {repo_path}",
            )

    print("All uploads complete.")


if __name__ == "__main__":
    main()
