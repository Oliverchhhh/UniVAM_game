import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope import snapshot_download


load_dotenv()
cache_dir = os.environ.get("PRETRAINED_MODEL_PATH", "./models")

snapshot_download(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    cache_dir=cache_dir,
)

# local_dir = os.path.join(cache_dir, "timm", "vit_large_patch16_dinov3.lvd1689m")

# hf_snapshot_download(
#     "timm/vit_large_patch16_dinov3.lvd1689m",
#     local_dir=local_dir,
#     endpoint="https://hf-mirror.com",
#     allow_patterns="pytorch_model.bin",
# )
