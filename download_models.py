from modelscope import snapshot_download
from dotenv import load_dotenv
import os


load_dotenv()
cache_dir = os.environ.get("PRETRAINED_MODEL_PATH", "./models")

snapshot_download(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    cache_dir=cache_dir,
)

snapshot_download(
    "Qwen/Qwen3-VL-8B-Instruct",
    cache_dir=cache_dir
)
