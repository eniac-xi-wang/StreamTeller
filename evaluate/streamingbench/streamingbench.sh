#!/usr/bin/env bash
# ===========================================================================
# StreamingBench evaluation — official entry point for Qwen3.5 + PredictMem
#
# Usage:
#   bash evaluate/streamingbench/streamingbench.sh --dry-run \
#     --task-csv /path/to/Real_Time_Visual_Understanding.csv \
#     --video-dir /path/to/StreamingBench/data/real
#   bash evaluate/streamingbench/streamingbench.sh --method predictmem \
#     --task-csv ... --video-dir ... --max-num-frames 64
#
# All experiment-level configuration lives here.  Python has NO hardcoded
# model paths, V-JEPA checkpoints, or data paths for formal experiments.
# ===========================================================================
set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────
# Configuration (override via CLI flags)
# ────────────────────────────────────────────────────────────────────────────

REPO_ROOT="/root/stream/StreamTeller"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- Model / dependency paths --
MODEL_PATH="/data/model_weights_public/Qwen/Qwen3.5-9B"
JEPA_CHECKPOINT="/data/model_weights_public/jepa/jeap_vitl_16_256.pt"
VJEPA_SRC="${REPO_ROOT}/site-packages/vjepa2"
LOCAL_MODELS_DIR="${REPO_ROOT}/models"

# -- Benchmark data (required) --
TASK_CSV=""
VIDEO_DIR=""

# -- Output --
RESULT_ROOT="${REPO_ROOT}/eval_results"
RUN_NAME="streamingbench_run"
RESULT_DIR=""

# -- PredictMem / V-JEPA --
METHOD="predictmem"                         # baseline | predictmem
PREDICTMEM_RUNTIME=""                       # empty = auto: predictmem→plugin, baseline→none
PREDICTMEM_KEEP_RATIO="0.10"
WINDOW_FRAMES=16
STRIDE_FRAMES=2
TAIL_KEEP_FRAMES=4
DROP_BOOTSTRAP=true

# -- Video sampling --
FPS="1.0"
QWEN_SIZE=512
JEPA_SIZE=256
MAX_NUM_FRAMES=256
MAX_PIXELS=200704   # 256*28*28
TIME_WINDOW=""

# -- Generation / runtime --
MAX_NEW_TOKENS=128
DEVICE="cuda"
TORCH_DTYPE="bfloat16"
NUM_GPUS=1
DISABLE_THINKING=true
DRY_RUN=false

# -- Misc --
LOG_PATH=""
OUTPUT_JSONL=""

# ────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --jepa-checkpoint) JEPA_CHECKPOINT="$2"; shift 2 ;;
    --vjepa-src) VJEPA_SRC="$2"; shift 2 ;;
    --task-csv) TASK_CSV="$2"; shift 2 ;;
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --predictmem-runtime) PREDICTMEM_RUNTIME="$2"; shift 2 ;;
    --keep-ratio) PREDICTMEM_KEEP_RATIO="$2"; shift 2 ;;
    --window-frames) WINDOW_FRAMES="$2"; shift 2 ;;
    --stride-frames) STRIDE_FRAMES="$2"; shift 2 ;;
    --tail-keep-frames) TAIL_KEEP_FRAMES="$2"; shift 2 ;;
    --drop-bootstrap) DROP_BOOTSTRAP=true; shift ;;
    --no-drop-bootstrap) DROP_BOOTSTRAP=false; shift ;;
    --fps) FPS="$2"; shift 2 ;;
    --qwen-size) QWEN_SIZE="$2"; shift 2 ;;
    --jepa-size) JEPA_SIZE="$2"; shift 2 ;;
    --max-num-frames) MAX_NUM_FRAMES="$2"; shift 2 ;;
    --max-pixels) MAX_PIXELS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --disable-thinking) DISABLE_THINKING=true; shift ;;
    --enable-thinking) DISABLE_THINKING=false; shift ;;
    --time-window) TIME_WINDOW="$2"; shift 2 ;;
    --log-path) LOG_PATH="$2"; shift 2 ;;
    --output-jsonl) OUTPUT_JSONL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ────────────────────────────────────────────────────────────────────────────
# Validate required params
# ────────────────────────────────────────────────────────────────────────────
if [[ -z "${TASK_CSV}" ]] || [[ -z "${VIDEO_DIR}" ]]; then
  echo "ERROR: --task-csv and --video-dir are required"
  echo "  --task-csv   path to Real_Time_Visual_Understanding.csv"
  echo "  --video-dir  path to StreamingBench data/real directory"
  exit 1
fi

# ────────────────────────────────────────────────────────────────────────────
# Auto-resolve predictmem_runtime
# ────────────────────────────────────────────────────────────────────────────
if [[ -z "${PREDICTMEM_RUNTIME}" ]]; then
  if [[ "${METHOD}" == "predictmem" ]]; then
    PREDICTMEM_RUNTIME="plugin"
  else
    PREDICTMEM_RUNTIME="none"
  fi
fi

# ────────────────────────────────────────────────────────────────────────────
# Result directory
# ────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/streamingbench/${RUN_NAME}_${TIMESTAMP}}"
mkdir -p "${RESULT_DIR}"

# ────────────────────────────────────────────────────────────────────────────
# Environment
# ────────────────────────────────────────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}:${LOCAL_MODELS_DIR}:${VJEPA_SRC}:${PYTHONPATH:-}"

GIT_COMMIT=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_STATUS=$(git -C "${REPO_ROOT}" status --short 2>/dev/null | head -5 | tr '\n' ' ' || echo "unknown")

# ────────────────────────────────────────────────────────────────────────────
# Print & save config
# ────────────────────────────────────────────────────────────────────────────
echo "============================================"
echo "StreamingBench Evaluation"
echo "============================================"
echo "Repo:         ${REPO_ROOT}"
echo "Git commit:   ${GIT_COMMIT}"
echo "Run name:     ${RUN_NAME}"
echo "Model:        ${MODEL_PATH}"
echo "JEPA ckpt:    ${JEPA_CHECKPOINT}"
echo "V-JEPA src:   ${VJEPA_SRC}"
echo "Method:       ${METHOD}"
echo "Pred runtime: ${PREDICTMEM_RUNTIME}"
echo "Keep ratio:   ${PREDICTMEM_KEEP_RATIO}"
echo "Window:       ${WINDOW_FRAMES}f / stride ${STRIDE_FRAMES} / tail ${TAIL_KEEP_FRAMES}"
echo "Drop boot:    ${DROP_BOOTSTRAP}"
echo "FPS:          ${FPS}"
echo "Qwen/JEPA:    ${QWEN_SIZE}/${JEPA_SIZE}"
echo "Max frames:   ${MAX_NUM_FRAMES}"
echo "Max pixels:   ${MAX_PIXELS}"
echo "Time window:  ${TIME_WINDOW:-full}"
echo "Task CSV:     ${TASK_CSV}"
echo "Video dir:    ${VIDEO_DIR}"
echo "Result dir:   ${RESULT_DIR}"
echo "Num GPUs:     ${NUM_GPUS}"
echo "Device:       ${DEVICE}"
echo "Torch dtype:  ${TORCH_DTYPE}"
echo "============================================"

# Save run_config.json
RUN_CONFIG="${RESULT_DIR}/run_config.json"
python3 -c "
import json, subprocess
repo = '${REPO_ROOT}'
try:
    commit = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
except Exception:
    commit = '${GIT_COMMIT}'
cfg = {
    'repo_root': '${REPO_ROOT}',
    'model_path': '${MODEL_PATH}',
    'jepa_checkpoint_path': '${JEPA_CHECKPOINT}',
    'vjepa_src_path': '${VJEPA_SRC}',
    'method': '${METHOD}',
    'predictmem_runtime': '${PREDICTMEM_RUNTIME}',
    'predictmem_keep_ratio': ${PREDICTMEM_KEEP_RATIO},
    'window_frames': ${WINDOW_FRAMES},
    'stride_frames': ${STRIDE_FRAMES},
    'tail_keep_frames': ${TAIL_KEEP_FRAMES},
    'drop_bootstrap': '${DROP_BOOTSTRAP}' == 'true',
    'fps': ${FPS},
    'qwen_size': ${QWEN_SIZE},
    'jepa_size': ${JEPA_SIZE},
    'max_num_frames': ${MAX_NUM_FRAMES},
    'max_pixels': ${MAX_PIXELS},
    'time_window_size': '${TIME_WINDOW}',
    'max_new_tokens': ${MAX_NEW_TOKENS},
    'device': '${DEVICE}',
    'torch_dtype': '${TORCH_DTYPE}',
    'num_gpus': ${NUM_GPUS},
    'disable_thinking': '${DISABLE_THINKING}' == 'true',
    'task_csv': '${TASK_CSV}',
    'video_dir': '${VIDEO_DIR}',
    'result_dir': '${RESULT_DIR}',
    'git_commit': commit,
}
with open('${RUN_CONFIG}', 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'run_config.json saved: ${RUN_CONFIG}')
"

# Save run_command.sh
cp "$0" "${RESULT_DIR}/run_command.sh" 2>/dev/null || true

if ${DRY_RUN}; then
  echo "DRY RUN — exiting without executing"
  exit 0
fi

# ────────────────────────────────────────────────────────────────────────────
# Build Python args
# ────────────────────────────────────────────────────────────────────────────
ARGS=(
  --run_name "${RUN_NAME}"
  --model_path "${MODEL_PATH}"
  --task_csv "${TASK_CSV}"
  --video_dir "${VIDEO_DIR}"
  --result_dir "${RESULT_DIR}"
  --method "${METHOD}"
  --predictmem_runtime "${PREDICTMEM_RUNTIME}"
  --predictmem_keep_ratio "${PREDICTMEM_KEEP_RATIO}"
  --jepa_checkpoint_path "${JEPA_CHECKPOINT}"
  --vjepa_src_path "${VJEPA_SRC}"
  --window_frames "${WINDOW_FRAMES}"
  --stride_frames "${STRIDE_FRAMES}"
  --tail_keep_frames "${TAIL_KEEP_FRAMES}"
  --fps "${FPS}"
  --qwen_size "${QWEN_SIZE}"
  --jepa_size "${JEPA_SIZE}"
  --max_num_frames "${MAX_NUM_FRAMES}"
  --max_pixels "${MAX_PIXELS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --device "${DEVICE}"
  --torch_dtype "${TORCH_DTYPE}"
)

${DROP_BOOTSTRAP} && ARGS+=(--drop_bootstrap) || ARGS+=(--no_drop_bootstrap)

if [[ -n "${TIME_WINDOW}" ]]; then
  ARGS+=(--time_window_size "${TIME_WINDOW}")
fi
if [[ -n "${LOG_PATH}" ]]; then
  ARGS+=(--log_path "${LOG_PATH}")
fi
if [[ -n "${OUTPUT_JSONL}" ]]; then
  ARGS+=(--output_jsonl "${OUTPUT_JSONL}")
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  ARGS+=(--multi_gpu --num_gpus "${NUM_GPUS}")
fi

# ────────────────────────────────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────────────────────────────────
python -m evaluate.streamingbench.streamingbench "${ARGS[@]}"

# Auto-score
python evaluate/streamingbench/score.py \
  --result_dir "${RESULT_DIR}" \
  --model_name "${RUN_NAME}"

echo ""
echo "Done. Results in: ${RESULT_DIR}"
echo "  run_config.json:  ${RESULT_DIR}/run_config.json"
echo "  summary.md:       ${RESULT_DIR}/results/summary.md"
