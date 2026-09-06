#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/stage1_storage_env.sh"
if [[ -z "${CONDA_EXE:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
  elif [[ -x /root/miniconda3/bin/conda ]]; then
    CONDA_EXE=/root/miniconda3/bin/conda
  elif [[ -x /opt/conda/bin/conda ]]; then
    CONDA_EXE=/opt/conda/bin/conda
  else
    echo "Conda was not found. Install Miniconda or set CONDA_EXE." >&2
    exit 1
  fi
fi
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CONDA_CHANNEL="${CONDA_CHANNEL:-https://repo.anaconda.com/pkgs/main}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

eval "$("${CONDA_EXE}" shell.bash hook)"
if [[ ! -f "${CONDA_ENV_PREFIX}/conda-meta/history" ]]; then
  # Do not inherit potentially stale global mirrors (the old TUNA conda-forge
  # endpoint currently returns HTTP 403 on this host).
  conda create -y -p "${CONDA_ENV_PREFIX}" --override-channels -c "${CONDA_CHANNEL}" python=3.12 pip
fi
conda activate "${CONDA_ENV_PREFIX}"

python -m pip install --upgrade pip wheel setuptools -i "${PIP_INDEX_URL}"
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "${TORCH_INDEX_URL}"
python -m pip install -r "${REPO_ROOT}/requirements-stage1.txt" -i "${PIP_INDEX_URL}"

# TorchCodec loads FFmpeg through its shared libraries, so having only the
# Python wheel is not sufficient. AutoDL images normally run as root and can
# install the matching system libraries with apt. Keep a conda-forge fallback
# for non-root machines.
if ! python -c 'import torchcodec' >/dev/null 2>&1; then
  echo "TorchCodec cannot load FFmpeg; installing FFmpeg runtime libraries..."
  if [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
    command -v ldconfig >/dev/null 2>&1 && ldconfig
  else
    conda install -y -c conda-forge 'ffmpeg>=6,<8'
  fi
fi

python - <<'PY'
import torch
import torchcodec
import timm
import transformers

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchcodec", torchcodec.__version__)
print("timm", timm.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
PY

# The environment is self-contained; downloaded conda packages are no longer
# needed and otherwise consume data-disk space.
conda clean -a -y

CONDA_BASE="$("${CONDA_EXE}" info --base)"
echo "Conda environment is ready. In every new terminal run:"
echo "  source ${CONDA_BASE}/etc/profile.d/conda.sh"
echo "  conda activate ${CONDA_ENV_PREFIX}"
