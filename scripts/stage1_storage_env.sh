#!/usr/bin/env bash
# Shared storage layout for AutoDL. Source this file before manual commands;
# the Stage-I helper scripts source it automatically.

STAGE1_STORAGE_ROOT="${STAGE1_STORAGE_ROOT:-/root/autodl-tmp/cuphead-stage1}"
if [[ "${STAGE1_STORAGE_ROOT}" == /root/autodl-tmp/* ]]; then
  if [[ ! -d /root/autodl-tmp ]]; then
    echo "Data disk path /root/autodl-tmp does not exist; refusing to use the system disk." >&2
    return 1 2>/dev/null || exit 1
  fi
  data_device="$(df -P /root/autodl-tmp | awk 'NR == 2 {print $1}')"
  root_device="$(df -P / | awk 'NR == 2 {print $1}')"
  if [[ -n "${data_device}" && "${data_device}" == "${root_device}" ]]; then
    echo "/root/autodl-tmp is on the same filesystem as /; refusing system-disk deployment." >&2
    return 1 2>/dev/null || exit 1
  fi
fi
ASSET_ROOT="${ASSET_ROOT:-${STAGE1_STORAGE_ROOT}/assets}"
CUPHEAD_ACTION_DATA_ROOT="${CUPHEAD_ACTION_DATA_ROOT:-${STAGE1_STORAGE_ROOT}/data}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${STAGE1_STORAGE_ROOT}/runs/stage1_future_condition_2x5090d}"
LOG_ROOT="${LOG_ROOT:-${STAGE1_STORAGE_ROOT}/runs}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${STAGE1_STORAGE_ROOT}/conda-env}"

# Redirect every large or potentially unbounded cache away from the system disk.
HF_HOME="${HF_HOME:-${STAGE1_STORAGE_ROOT}/cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${STAGE1_STORAGE_ROOT}/cache/conda-pkgs}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STAGE1_STORAGE_ROOT}/cache/xdg}"
TORCH_HOME="${TORCH_HOME:-${STAGE1_STORAGE_ROOT}/cache/torch}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${STAGE1_STORAGE_ROOT}/cache/pip}"
TMPDIR="${TMPDIR:-${STAGE1_STORAGE_ROOT}/tmp}"

export STAGE1_STORAGE_ROOT ASSET_ROOT CUPHEAD_ACTION_DATA_ROOT
export STAGE1_OUTPUT_DIR LOG_ROOT CONDA_ENV_PREFIX
export HF_HOME HF_HUB_CACHE HF_XET_CACHE CONDA_PKGS_DIRS
export XDG_CACHE_HOME TORCH_HOME PIP_CACHE_DIR TMPDIR

# Setting PROXY_PORT is enough for Python requests, Hugging Face, git and curl.
# This is intentionally repeated whenever the file is sourced because exports
# from an earlier terminal do not survive opening a new terminal.
if [[ -n "${PROXY_PORT:-}" ]]; then
  HTTP_PROXY="http://127.0.0.1:${PROXY_PORT}"
  HTTPS_PROXY="${HTTP_PROXY}"
  http_proxy="${HTTP_PROXY}"
  https_proxy="${HTTPS_PROXY}"
  export HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
fi

mkdir -p \
  "${ASSET_ROOT}" "${CUPHEAD_ACTION_DATA_ROOT}" "${LOG_ROOT}" \
  "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${CONDA_PKGS_DIRS}" \
  "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}"
