#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/stage1_storage_env.sh"
HF_REPO="${HF_REPO:-ch1415926/cuphead-action}"
EXPECTED_CHUNKS="${EXPECTED_CHUNKS:-4793}"
DELETE_DATA_SHARDS_AFTER_EXTRACT="${DELETE_DATA_SHARDS_AFTER_EXTRACT:-1}"
READY_MARKER="${ASSET_ROOT}/.stage1_assets_ready"

mkdir -p "${ASSET_ROOT}" "${CUPHEAD_ACTION_DATA_ROOT}"
chunk_count="$(find "${CUPHEAD_ACTION_DATA_ROOT}" -name annotation.proto | wc -l)"
assets_complete=1
for required_asset in \
  "${ASSET_ROOT}/checkpoints/nitrogen/ng.pt" \
  "${ASSET_ROOT}/checkpoints/siglip2-large-patch16-256/model.safetensors" \
  "${ASSET_ROOT}/checkpoints/wog/dinov2_weights.pth" \
  "${ASSET_ROOT}/checkpoints/wog/Wan2.1_VAE.pth"; do
  [[ -f "${required_asset}" ]] || assets_complete=0
done
if [[ ! -f "${READY_MARKER}" || "${chunk_count}" -ne "${EXPECTED_CHUNKS}" || "${assets_complete}" -ne 1 ]]; then
  # Fetch the small checksum manifest first. Download checkpoints once, then
  # stream data shards one at a time so compressed and extracted copies do not
  # occupy the data disk simultaneously.
  hf download "${HF_REPO}" manifest/sha256sums.txt \
    --repo-type dataset --local-dir "${ASSET_ROOT}"
  checksum_manifest="${ASSET_ROOT}/manifest/sha256sums.txt"

  download_and_verify() {
    local expected_sha="$1"
    local relative_path="$2"
    local local_path="${ASSET_ROOT}/${relative_path}"
    if [[ ! -f "${local_path}" ]] || \
       ! (cd "${ASSET_ROOT}" && echo "${expected_sha}  ${relative_path}" | sha256sum -c - >/dev/null 2>&1); then
      hf download "${HF_REPO}" "${relative_path}" \
        --repo-type dataset --local-dir "${ASSET_ROOT}"
    fi
    (cd "${ASSET_ROOT}" && echo "${expected_sha}  ${relative_path}" | sha256sum -c -)
  }

  while read -r expected_sha relative_path; do
    [[ "${relative_path}" == checkpoints/* ]] || continue
    download_and_verify "${expected_sha}" "${relative_path}"
  done < "${checksum_manifest}"

  while read -r expected_sha relative_path; do
    [[ "${relative_path}" == data_shards/*.tar ]] || continue
    shard="${ASSET_ROOT}/${relative_path}"
    marker="${CUPHEAD_ACTION_DATA_ROOT}/.$(basename "${relative_path}").extracted"
    if [[ -f "${marker}" ]]; then
      [[ "${DELETE_DATA_SHARDS_AFTER_EXTRACT}" == 1 ]] && rm -f "${shard}"
      continue
    fi
    download_and_verify "${expected_sha}" "${relative_path}"
    tar -xf "${shard}" -C "${CUPHEAD_ACTION_DATA_ROOT}"
    touch "${marker}"
    [[ "${DELETE_DATA_SHARDS_AFTER_EXTRACT}" == 1 ]] && rm -f "${shard}"
  done < "${checksum_manifest}"

  chunk_count="$(find "${CUPHEAD_ACTION_DATA_ROOT}" -name annotation.proto | wc -l)"
  if [[ "${chunk_count}" -ne "${EXPECTED_CHUNKS}" ]]; then
    echo "Expected ${EXPECTED_CHUNKS} chunks after extraction, found ${chunk_count}" >&2
    exit 1
  fi
  touch "${READY_MARKER}"
fi

# Any verified archives retained from an interrupted or older deployment
# duplicate extracted data and can always be recovered from Hugging Face.
if [[ "${DELETE_DATA_SHARDS_AFTER_EXTRACT}" == 1 ]]; then
  find "${ASSET_ROOT}/data_shards" -maxdepth 1 -type f -name '*.tar' -delete 2>/dev/null || true
fi

mkdir -p "${REPO_ROOT}/third_party/WoG/pretrained/vision"
ln -sfn "${ASSET_ROOT}/checkpoints/wog/dinov2_weights.pth" \
  "${REPO_ROOT}/third_party/WoG/pretrained/vision/dinov2_weights.pth"
ln -sfn "${ASSET_ROOT}/checkpoints/wog/Wan2.1_VAE.pth" \
  "${REPO_ROOT}/third_party/WoG/pretrained/vision/Wan2.1_VAE.pth"

echo "Assets ready"
echo "  storage: ${STAGE1_STORAGE_ROOT}"
echo "  assets: ${ASSET_ROOT}"
echo "  data:   ${CUPHEAD_ACTION_DATA_ROOT}"
echo "  chunks: ${chunk_count}"
