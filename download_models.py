import os

from dotenv import load_dotenv
from modelscope import snapshot_download

from univam.models.deepstack import convert_qwen3vl_to_vfe_ckpt
from univam.utils.files import ensure_directory


load_dotenv()
cache_dir = os.environ.get("PRETRAINED_MODEL_PATH", "./models")

snapshot_download(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    cache_dir=cache_dir,
)

qwen_vl = snapshot_download(
    "Qwen/Qwen3-VL-8B-Instruct",
    cache_dir=cache_dir,
)

vfe_path = os.path.join(cache_dir, "Qwen/Qwen3-VL-VideoFeatureExtractor/Qwen3-VL-VideoFeatureExtractor-8b.pt")

ensure_directory(os.path.basename(vfe_path))

convert_qwen3vl_to_vfe_ckpt(
    qwen_vl,
    vfe_path,
)
