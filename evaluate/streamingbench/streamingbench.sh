#!/usr/bin/env bash
# StreamingBench evaluation launcher for Qwen3.5 + PredictMem
#
# Usage:
#   bash evaluate/streamingbench/streamingbench.sh \
#     --model-path /path/to/model --method predictmem --num-gpus 1 --dry-run
set -euo pipefail

RUN_NAME="streamingbench_run"
MODEL_PATH="/data/model_weights_public/Qwen/Qwen3.5-9B"
TASK_CSV=""
VIDEO_DIR=""
RESULT_DIR=""
METHOD="baseline"
PREDICTMEM_RUNTIME="none"
PREDICTMEM_KEEP_RATIO="0.10"
NUM_GPUS=1
MAX_NUM_FRAMES=256
MAX_PIXELS=200704  # 256*28*28
FPS="1.0"
TIME_WINDOW=""
MAX_NEW_TOKENS=128
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --task-csv) TASK_CSV="$2"; shift 2 ;;
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --predictmem-runtime) PREDICTMEM_RUNTIME="$2"; shift 2 ;;
    --keep-ratio) PREDICTMEM_KEEP_RATIO="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --max-num-frames) MAX_NUM_FRAMES="$2"; shift 2 ;;
    --max-pixels) MAX_PIXELS="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --time-window) TIME_WINDOW="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="--dry_run"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$METHOD" == "predictmem" ]]; then
  PREDICTMEM_RUNTIME="${PREDICTMEM_RUNTIME:-plugin}"
else
  PREDICTMEM_RUNTIME="${PREDICTMEM_RUNTIME:-none}"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="${RESULT_DIR:-eval_results/streamingbench/${RUN_NAME}_${TIMESTAMP}}"

echo "============================================"
echo "StreamingBench Evaluation"
echo "============================================"
echo "Run name:     ${RUN_NAME}"
echo "Model:        ${MODEL_PATH}"
echo "Method:       ${METHOD}"
echo "Task CSV:     ${TASK_CSV}"
echo "Video dir:    ${VIDEO_DIR}"
echo "Result dir:   ${RESULT_DIR}"
echo "Num GPUs:     ${NUM_GPUS}"
echo "============================================"

if [[ -z "${TASK_CSV}" ]] || [[ -z "${VIDEO_DIR}" ]]; then
  echo "ERROR: --task-csv and --video-dir are required"
  exit 1
fi

mkdir -p "${RESULT_DIR}"

ARGS=(
  --run_name "${RUN_NAME}"
  --model_path "${MODEL_PATH}"
  --task_csv "${TASK_CSV}"
  --video_dir "${VIDEO_DIR}"
  --result_dir "${RESULT_DIR}"
  --method "${METHOD}"
  --predictmem_runtime "${PREDICTMEM_RUNTIME}"
  --predictmem_keep_ratio "${PREDICTMEM_KEEP_RATIO}"
  --fps "${FPS}"
  --max_num_frames "${MAX_NUM_FRAMES}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)

if [[ -n "${TIME_WINDOW}" ]]; then
  ARGS+=(--time_window_size "${TIME_WINDOW}")
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  ARGS+=(--multi_gpu --num_gpus "${NUM_GPUS}")
fi

if [[ -n "${DRY_RUN}" ]]; then
  ARGS+=(--dry_run)
fi

PYTHONPATH="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/models"
export PYTHONPATH

python evaluate/streamingbench/streamingbench.py "${ARGS[@]}"

# Auto-score
python evaluate/streamingbench/score.py \
  --result_dir "${RESULT_DIR}" \
  --model_name "${RUN_NAME}"

echo "Done. Results in: ${RESULT_DIR}"
