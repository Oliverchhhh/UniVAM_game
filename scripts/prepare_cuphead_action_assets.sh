#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_REPO="${HF_REPO:-ch1415926/cuphead-action}"
ASSET_ROOT="${ASSET_ROOT:-/root/cuphead-action-assets}"
CUPHEAD_ACTION_DATA_ROOT="${CUPHEAD_ACTION_DATA_ROOT:-/root/cuphead-action-data}"

mkdir -p "${ASSET_ROOT}" "${CUPHEAD_ACTION_DATA_ROOT}"
hf download "${HF_REPO}" --repo-type dataset --local-dir "${ASSET_ROOT}"

if [[ -f "${ASSET_ROOT}/manifest/sha256sums.txt" ]]; then
  (cd "${ASSET_ROOT}" && sha256sum -c manifest/sha256sums.txt)
fi

for shard in "${ASSET_ROOT}"/data_shards/*.tar; do
  marker="${CUPHEAD_ACTION_DATA_ROOT}/.$(basename "${shard}").extracted"
  [[ -f "${marker}" ]] && continue
  tar -xf "${shard}" -C "${CUPHEAD_ACTION_DATA_ROOT}"
  touch "${marker}"
done

mkdir -p "${REPO_ROOT}/third_party/WoG/pretrained/vision"
ln -sfn "${ASSET_ROOT}/checkpoints/wog/dinov2_weights.pth" \
  "${REPO_ROOT}/third_party/WoG/pretrained/vision/dinov2_weights.pth"
ln -sfn "${ASSET_ROOT}/checkpoints/wog/Wan2.1_VAE.pth" \
  "${REPO_ROOT}/third_party/WoG/pretrained/vision/Wan2.1_VAE.pth"

echo "Assets ready"
echo "  assets: ${ASSET_ROOT}"
echo "  data:   ${CUPHEAD_ACTION_DATA_ROOT}"
echo "  chunks: $(find "${CUPHEAD_ACTION_DATA_ROOT}" -name annotation.proto | wc -l)"
