#!/usr/bin/env bash
# 4-GPU parallel launcher for encode_videos_flow.py
#
# Usage:
#   bash scripts/encode_videos_flow_4gpu.sh --detach   # 推荐：nohup 后台续跑
#   bash scripts/encode_videos_flow_4gpu.sh            # 前台运行（终端关闭会中断）
#   bash scripts/encode_videos_flow_4gpu.sh --status   # 查看状态
#   bash scripts/encode_videos_flow_4gpu.sh --stop     # 停止任务
#
# Optional overrides:
#   PRETRAINED_MODEL_PATH=/home/user/rjt/UniVAM \
#   RESUME_PATH=./univam_cuphead_140000 \
#   DATA_FOLDER=./cuphead_dataset_converted_20260501 \
#   BATCH_WINDOWS=32 \
#   bash scripts/encode_videos_flow_4gpu.sh --detach

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NUM_GPUS=4
PYTHON="${PYTHON:-/home/user/anaconda3/envs/univam/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/debug.yaml}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/home/user/rjt/UniVAM}"
RESUME_PATH="${RESUME_PATH:-./univam_cuphead_140000}"
DATA_FOLDER="${DATA_FOLDER:-./cuphead_dataset_converted_20260501}"
VIDEO_NAME="${VIDEO_NAME:-256x256.mp4}"
OUTPUT_NAME="${OUTPUT_NAME:-univam_flow_features.pt}"
BATCH_WINDOWS="${BATCH_WINDOWS:-32}"
LOG_DIR="${LOG_DIR:-./logs/encode_flow_4gpu}"
MASTER_PID_FILE="${LOG_DIR}/master.pid"
RUN_LOG="${LOG_DIR}/run.log"

mkdir -p "$LOG_DIR"

is_running() {
  if [[ -f "${MASTER_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${MASTER_PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  pgrep -f "encode_videos_flow.py.*--data_folder ${DATA_FOLDER}" >/dev/null 2>&1
}

cmd_status() {
  if is_running; then
    echo "Status: RUNNING"
    [[ -f "${MASTER_PID_FILE}" ]] && echo "Master PID: $(cat "${MASTER_PID_FILE}")"
    echo "Worker PIDs:"
    pgrep -af "encode_videos_flow.py.*--data_folder ${DATA_FOLDER}" || true
    echo "Saved counts (log):"
    for f in "${LOG_DIR}"/gpu*.log; do
      [[ -f "$f" ]] && echo "  $(basename "$f"): $(grep -c 'Saved' "$f" 2>/dev/null || echo 0)"
    done
  else
    echo "Status: NOT RUNNING"
  fi
}

cmd_stop() {
  if is_running; then
    echo "Stopping encode workers..."
    pkill -f "encode_videos_flow.py.*--data_folder ${DATA_FOLDER}" || true
    if [[ -f "${MASTER_PID_FILE}" ]]; then
      local pid
      pid="$(cat "${MASTER_PID_FILE}")"
      kill "${pid}" 2>/dev/null || true
      rm -f "${MASTER_PID_FILE}"
    fi
    sleep 2
    cmd_status
  else
    echo "No running job found."
  fi
}

cmd_detach() {
  if is_running; then
    echo "Job already running. Use --status to check."
    exit 1
  fi

  echo "Starting detached job (nohup)..."
  nohup env \
    PYTHON="${PYTHON}" \
    PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH}" \
    RESUME_PATH="${RESUME_PATH}" \
    DATA_FOLDER="${DATA_FOLDER}" \
    VIDEO_NAME="${VIDEO_NAME}" \
    OUTPUT_NAME="${OUTPUT_NAME}" \
    BATCH_WINDOWS="${BATCH_WINDOWS}" \
    LOG_DIR="${LOG_DIR}" \
    bash "${BASH_SOURCE[0]}" --run \
    >> "${RUN_LOG}" 2>&1 &

  echo $! > "${MASTER_PID_FILE}"
  echo "Master PID: $(cat "${MASTER_PID_FILE}")"
  echo "Run log: ${RUN_LOG}"
  echo "Worker logs: ${LOG_DIR}/gpu{0,1,2,3}.log"
  echo "Check status: bash scripts/encode_videos_flow_4gpu.sh --status"
}

cmd_run() {
  export PRETRAINED_MODEL_PATH
  export RESUME_PATH

  {
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') resume/start =========="
    echo "ROOT_DIR=$ROOT_DIR"
    echo "DATA_FOLDER=$DATA_FOLDER"
    echo "RESUME_PATH=$RESUME_PATH"
    echo "PRETRAINED_MODEL_PATH=$PRETRAINED_MODEL_PATH"
    echo "LOG_DIR=$LOG_DIR"
    echo "Launching $NUM_GPUS workers..."
  } | tee -a "${RUN_LOG}"

  PIDS=()
  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    LOG_FILE="${LOG_DIR}/gpu${GPU_ID}.log"
    echo "  GPU ${GPU_ID} -> worker_id=${GPU_ID}, log=${LOG_FILE}" | tee -a "${RUN_LOG}"

    {
      echo ""
      echo "========== $(date '+%Y-%m-%d %H:%M:%S') GPU ${GPU_ID} resume =========="
    } >> "${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" encode_videos_flow.py \
      --config_path "${CONFIG_PATH}" \
      --data_folder "${DATA_FOLDER}" \
      --video_name "${VIDEO_NAME}" \
      --output_name "${OUTPUT_NAME}" \
      --resume_path "${RESUME_PATH}" \
      --batch_windows "${BATCH_WINDOWS}" \
      --worker_id "${GPU_ID}" \
      --num_workers "${NUM_GPUS}" \
      --gpu 0 \
      --skip_existing \
      >> "${LOG_FILE}" 2>&1 &

    PIDS+=("$!")
  done

  echo "PIDs: ${PIDS[*]}" | tee -a "${RUN_LOG}"

  FAIL=0
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "Worker GPU ${i} failed. See ${LOG_DIR}/gpu${i}.log" | tee -a "${RUN_LOG}"
      FAIL=1
    else
      echo "Worker GPU ${i} done." | tee -a "${RUN_LOG}"
    fi
  done

  rm -f "${MASTER_PID_FILE}"

  if [[ "${FAIL}" -ne 0 ]]; then
    echo "Some workers failed." | tee -a "${RUN_LOG}"
    exit 1
  fi

  echo "All ${NUM_GPUS} workers finished. $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${RUN_LOG}"
}

case "${1:-}" in
  --detach) cmd_detach ;;
  --run)    cmd_run ;;
  --status) cmd_status ;;
  --stop)   cmd_stop ;;
  "")       cmd_run ;;
  *)
    echo "Unknown option: $1"
    echo "Usage: bash scripts/encode_videos_flow_4gpu.sh [--detach|--status|--stop]"
    exit 1
    ;;
esac
