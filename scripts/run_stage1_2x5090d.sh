#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/stage1_future_condition_2x5090d.yaml}"
ASSET_ROOT="${ASSET_ROOT:-/root/cuphead-action-assets}"
CUPHEAD_ACTION_DATA_ROOT="${CUPHEAD_ACTION_DATA_ROOT:-/root/cuphead-action-data}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-/root/stage1-runs/stage1_future_condition_2x5090d}"
STAGE1_NITROGEN_CKPT="${STAGE1_NITROGEN_CKPT:-${ASSET_ROOT}/checkpoints/nitrogen/ng.pt}"
STAGE1_NITROGEN_ROOT="${STAGE1_NITROGEN_ROOT:-${REPO_ROOT}/third_party/NitroGen-real-time}"
STAGE1_SIGLIP_PATH="${STAGE1_SIGLIP_PATH:-${ASSET_ROOT}/checkpoints/siglip2-large-patch16-256}"
STAGE1_WOG_ROOT="${STAGE1_WOG_ROOT:-${REPO_ROOT}/third_party/WoG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
export ASSET_ROOT CUPHEAD_ACTION_DATA_ROOT STAGE1_OUTPUT_DIR
export STAGE1_NITROGEN_CKPT STAGE1_NITROGEN_ROOT STAGE1_SIGLIP_PATH STAGE1_WOG_ROOT
export PYTHONPATH="${REPO_ROOT}:${STAGE1_NITROGEN_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export DEBUG=0

if [[ -n "${PROXY_PORT:-}" ]]; then
  export HTTP_PROXY="http://127.0.0.1:${PROXY_PORT}"
  export HTTPS_PROXY="http://127.0.0.1:${PROXY_PORT}"
  export http_proxy="${HTTP_PROXY}"
  export https_proxy="${HTTPS_PROXY}"
fi

for required in \
  "${CUPHEAD_ACTION_DATA_ROOT}" \
  "${STAGE1_NITROGEN_CKPT}" \
  "${STAGE1_SIGLIP_PATH}/model.safetensors" \
  "${STAGE1_WOG_ROOT}/pretrained/vision/dinov2_weights.pth" \
  "${STAGE1_WOG_ROOT}/pretrained/vision/Wan2.1_VAE.pth"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required Stage-I asset: ${required}" >&2
    echo "Run: bash scripts/prepare_cuphead_action_assets.sh" >&2
    exit 1
  fi
done

exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  -m stage1_future_condition.train --config "${CONFIG}" --resume auto "$@"
