#!/usr/bin/env bash
# OVO-Bench evaluation launcher for Qwen3.5 + PredictMem
#
# Usage:
#   bash evaluate/ovobench/ovobench.sh --model-path /path/to/model --method predictmem --num-gpus 1 --max-samples 2
#   bash evaluate/ovobench/ovobench.sh --model-path /path/to/model --method baseline --num-gpus 8
set -euo pipefail

RUN_NAME="ovobench_run"
MODEL_PATH="/data/model_weights_public/Qwen/Qwen3.5-9B"
TASK_JSON="evaluate/ovobench/ovo_bench_new.json"
VIDEO_DIR="/data/qinian_workspace/OVO-Bench"
RESULT_DIR=""
METHOD="baseline"
PREDICTMEM_RUNTIME="none"
PREDICTMEM_KEEP_RATIO="0.10"
NUM_GPUS=1
MAX_SAMPLES=""
SAMPLE_IDS=""
TASK=""
TIME_WINDOW=""
FPS="1.0"
MAX_NEW_TOKENS=16
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --task-json) TASK_JSON="$2"; shift 2 ;;
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --predictmem-runtime) PREDICTMEM_RUNTIME="$2"; shift 2 ;;
    --keep-ratio) PREDICTMEM_KEEP_RATIO="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --sample-ids) SAMPLE_IDS="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --time-window) TIME_WINDOW="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="--dry_run"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Resolve method defaults
if [[ "$METHOD" == "predictmem" ]]; then
  PREDICTMEM_RUNTIME="${PREDICTMEM_RUNTIME:-plugin}"
else
  PREDICTMEM_RUNTIME="${PREDICTMEM_RUNTIME:-none}"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="${RESULT_DIR:-eval_results/ovobench/${RUN_NAME}_${TIMESTAMP}}"

echo "============================================"
echo "OVO-Bench Evaluation"
echo "============================================"
echo "Run name:     ${RUN_NAME}"
echo "Model:        ${MODEL_PATH}"
echo "Method:       ${METHOD}"
echo "Pred runtime: ${PREDICTMEM_RUNTIME}"
echo "Keep ratio:   ${PREDICTMEM_KEEP_RATIO}"
echo "Result dir:   ${RESULT_DIR}"
echo "Task JSON:    ${TASK_JSON}"
echo "Video dir:    ${VIDEO_DIR}"
echo "Num GPUs:     ${NUM_GPUS}"
echo "Max samples:  ${MAX_SAMPLES:-all}"
echo "============================================"

mkdir -p "${RESULT_DIR}"

# Build args
ARGS=(
  --run_name "${RUN_NAME}"
  --model_path "${MODEL_PATH}"
  --task_json "${TASK_JSON}"
  --video_dir "${VIDEO_DIR}"
  --result_dir "${RESULT_DIR}"
  --method "${METHOD}"
  --predictmem_runtime "${PREDICTMEM_RUNTIME}"
  --predictmem_keep_ratio "${PREDICTMEM_KEEP_RATIO}"
  --fps "${FPS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)

if [[ -n "${TASK}" ]]; then
  ARGS+=(--task ${TASK})
fi
if [[ -n "${SAMPLE_IDS}" ]]; then
  ARGS+=(--sample_ids ${SAMPLE_IDS})
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
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

python evaluate/ovobench/ovobench.py "${ARGS[@]}"

# Auto-score
python evaluate/ovobench/score.py \
  --result_dir "${RESULT_DIR}" \
  --run_name "${RUN_NAME}"

echo "Done. Results in: ${RESULT_DIR}"
