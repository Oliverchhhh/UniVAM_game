#!/usr/bin/env bash
# Run Stage-I with a persistent GPU guard, stall detection, and checkpoint resume.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/stage1_storage_env.sh"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-0,1}"
RUN_DIR="${RUN_DIR:-${STAGE1_OUTPUT_DIR:-${LOG_ROOT}/stage1_future_condition_2x5090d}}"
TRAIN_LOG="${TRAIN_LOG:-${LOG_ROOT}/stage1_training_supervised.log}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-${LOG_ROOT}/stage1_supervisor.log}"
GUARD_LOG="${GUARD_LOG:-${LOG_ROOT}/stage1_gpu_guard.log}"
GUARD_GIB_PER_GPU="${GUARD_GIB_PER_GPU:-18}"
STALL_SECONDS="${STALL_SECONDS:-600}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-30}"
MAX_AUTOMATIC_RESTARTS="${MAX_AUTOMATIC_RESTARTS:-3}"

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "The Stage-I environment is not active. Run: conda activate ${CONDA_ENV_PREFIX}" >&2
  exit 1
fi

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "${SUPERVISOR_LOG}"
}

GUARD_PID=""
TRAIN_PID=""
cleanup() {
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM -- "-${TRAIN_PID}" 2>/dev/null || true
  fi
  if [[ -n "${GUARD_PID}" ]] && kill -0 "${GUARD_PID}" 2>/dev/null; then
    kill -TERM "${GUARD_PID}" 2>/dev/null || true
    wait "${GUARD_PID}" 2>/dev/null || true
  fi
}
trap 'cleanup; exit 130' INT TERM

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

log "starting GPU guard on physical GPUs ${PHYSICAL_GPUS}"
python -u scripts/hold_stage1_gpus.py \
  --gib-per-gpu "${GUARD_GIB_PER_GPU}" >>"${GUARD_LOG}" 2>&1 &
GUARD_PID=$!
sleep 5
if ! kill -0 "${GUARD_PID}" 2>/dev/null; then
  wait "${GUARD_PID}" || true
  log "GPU guard failed; cards are unavailable or do not have enough free memory"
  exit 1
fi

restart_count=0
while true; do
  log "launching Stage-I (automatic restart ${restart_count}/${MAX_AUTOMATIC_RESTARTS})"
  setsid env CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}" \
    STAGE1_OUTPUT_DIR="${RUN_DIR}" \
    bash scripts/run_stage1_2x5090d.sh >>"${TRAIN_LOG}" 2>&1 &
  TRAIN_PID=$!
  launch_epoch=$(date +%s)

  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep 30
    now=$(date +%s)
    metrics="${RUN_DIR}/metrics.jsonl"
    if [[ -f "${metrics}" ]]; then
      last_progress=$(stat -c %Y "${metrics}")
    else
      last_progress=${launch_epoch}
    fi
    if (( now - last_progress > STALL_SECONDS && now - launch_epoch > STALL_SECONDS )); then
      log "no metric update for more than ${STALL_SECONDS}s; terminating stalled process group ${TRAIN_PID}"
      kill -TERM -- "-${TRAIN_PID}" 2>/dev/null || true
      sleep 15
      kill -KILL -- "-${TRAIN_PID}" 2>/dev/null || true
      break
    fi
  done

  wait "${TRAIN_PID}"
  status=$?
  TRAIN_PID=""
  if (( status == 0 )); then
    log "Stage-I completed successfully"
    if [[ -n "${STAGE2_CMD:-}" ]]; then
      log "starting configured Stage-II command"
      bash -lc "${STAGE2_CMD}" >>"${LOG_ROOT}/stage2_training.log" 2>&1
      stage2_status=$?
      log "Stage-II exited with status ${stage2_status}"
    else
      log "STAGE2_CMD is not configured; keeping GPUs reserved"
      wait "${GUARD_PID}"
    fi
    break
  fi

  restart_count=$((restart_count + 1))
  log "Stage-I exited with status ${status}; GPU guard remains active"
  if (( restart_count > MAX_AUTOMATIC_RESTARTS )); then
    log "automatic restart limit reached; keeping GPUs reserved for manual diagnosis"
    wait "${GUARD_PID}"
    break
  fi
  sleep "${RESTART_DELAY_SECONDS}"
done

cleanup
