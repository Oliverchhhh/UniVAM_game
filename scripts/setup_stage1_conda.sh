#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
CONDA_ENV_NAME="${CONDA_ENV_NAME:-nitrogen-stage1}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CONDA_CHANNEL="${CONDA_CHANNEL:-https://repo.anaconda.com/pkgs/main}"

eval "$("${CONDA_EXE}" shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  # Do not inherit potentially stale global mirrors (the old TUNA conda-forge
  # endpoint currently returns HTTP 403 on this host).
  conda create -y -n "${CONDA_ENV_NAME}" --override-channels -c "${CONDA_CHANNEL}" python=3.12 pip
fi
conda activate "${CONDA_ENV_NAME}"

python -m pip install --upgrade pip wheel setuptools -i "${PIP_INDEX_URL}"
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "${TORCH_INDEX_URL}"
python -m pip install -r "${REPO_ROOT}/requirements-stage1.txt" -i "${PIP_INDEX_URL}"

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

echo "Conda environment ready: conda activate ${CONDA_ENV_NAME}"
