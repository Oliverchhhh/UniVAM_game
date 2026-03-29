import os

from dotenv import load_dotenv
from modelscope import snapshot_download


load_dotenv()
cache_dir = os.environ.get("PRETRAINED_MODEL_PATH", "./models")

snapshot_download(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    cache_dir=cache_dir,
)
