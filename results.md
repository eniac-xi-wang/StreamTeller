# PredictMem 第五轮验证结果：配置收口与 bash 入口

测试时间：2026-05-16

## 执行摘要

按照最新 `instruct.md`（P0-P8），完成了配置体系的重构：
- **P0**: Bash 成为唯一正式配置入口，包含完整配置区
- **P1**: 修复 `PREDICTMEM_RUNTIME` 自动检测（method=predictmem → plugin，baseline → none）
- **P2**: V-JEPA checkpoint/source path 由 bash 传入，Python 无硬编码
- **P3**: 所有 Python 入口补齐 CLI 参数，multi-GPU worker 完整透传
- **P4**: 视频采样 time clipping 实现，MAX_PIXELS 透传
- **P5**: Bash 入口为正式复现入口
- **P6**: 每次运行保存 `run_config.json` + `run_command.sh`
- **P7**: 10 个新测试（bash config + video clipping）+ 21 个已有 = 31 tests
- **P8**: 更新 results.md

## P0+P1：Bash 配置区 + Runtime 自动检测

### 已排查并修复的问题

| # | 问题 | 修复 |
|---|---|---|
| 1 | bash `PREDICTMEM_RUNTIME="none"` 导致 predictmem 不启用 | 改为 `=""`，自动检测：method=predictmem → plugin |
| 2 | V-JEPA checkpoint 硬编码在 `streaming_memory.py` | 移除，由 `config.jepa_checkpoint_path` 接入 |
| 3 | V-JEPA source 硬编码在 `vjepa_scorer.py` | 支持 `vjepa_src_path` 参数，bash 透传 |
| 4 | StreamingBench `start_time/end_time` 未使用 | 传入 `build_video_inputs_for_eval()` |
| 5 | bash `MAX_PIXELS` 未透传 Python | 已加入 ARGS |
| 6 | common helper 不支持时间裁剪 | 新增 `start_time/end_time` 实现 |
| 7 | multi-GPU worker 未透传配置 | 全部补全 |
| 8 | smoke 命令直接调 Python | 全部改为 bash entry |

### Bash 配置区示例（ovobench.sh）

```bash
MODEL_PATH="/data/model_weights_public/Qwen/Qwen3.5-9B"
JEPA_CHECKPOINT="/data/model_weights_public/jepa/jeap_vitl_16_256.pt"
VJEPA_SRC="${REPO_ROOT}/site-packages/vjepa2"

METHOD="predictmem"
PREDICTMEM_RUNTIME=""        # auto: predictmem→plugin, baseline→none
PREDICTMEM_KEEP_RATIO="0.10"
WINDOW_FRAMES=16
STRIDE_FRAMES=2
TAIL_KEEP_FRAMES=4
DROP_BOOTSTRAP=true

FPS="1.0"
QWEN_SIZE=512
JEPA_SIZE=256

export PYTHONPATH="${REPO_ROOT}:${LOCAL_MODELS_DIR}:${VJEPA_SRC}:${PYTHONPATH:-}"
```

### Runtime 自动检测逻辑

```bash
if [[ -z "${PREDICTMEM_RUNTIME}" ]]; then
  if [[ "${METHOD}" == "predictmem" ]]; then
    PREDICTMEM_RUNTIME="plugin"
  else
    PREDICTMEM_RUNTIME="none"
  fi
fi
```

## P2：Hardcoded Paths 清理结果

```
rg "/data/model_weights_public/jepa|jeap_vitl_16_256.pt" models/predictmem evaluate
```
- Bash 默认配置中保留（作为可覆盖的默认值）✓
- Python 主逻辑 **零命中** ✓

## P3+P4：CLI 参数 + 视频裁剪

### OVO-Bench Python 新增参数

```
--jepa_checkpoint_path, --vjepa_src_path, --device, --torch_dtype
--qwen_size, --jepa_size, --window_frames, --stride_frames
--tail_keep_frames, --drop_bootstrap/--no_drop_bootstrap
--frame_budget, --stream_mode, --disable_thinking/--enable_thinking
--baseline_result_dir
```

### StreamingBench Python 新增参数

Same + `--max_pixels`, `--max_num_frames`, `--time_window_size`

### Video Time Clipping

`build_video_inputs_for_eval(video_path, start_time=10, end_time=30, fps=1.0)` 现在正确采样 [10s, 30s) 区间，Qwen 512 和 V-JEPA 256 使用同一批 `frames_indices`。

## P5：Bash 正式入口

OVO smoke：
```bash
bash evaluate/ovobench/ovobench.sh \
  --model-path /data/model_weights_public/Qwen/Qwen3.5-9B \
  --jepa-checkpoint /data/model_weights_public/jepa/jeap_vitl_16_256.pt \
  --vjepa-src /root/stream/StreamTeller/site-packages/vjepa2 \
  --method predictmem --run-name predictmem_ovobench_smoke \
  --num-gpus 1 --max-samples 2
```

StreamingBench smoke：
```bash
bash evaluate/streamingbench/streamingbench.sh \
  --model-path /data/model_weights_public/Qwen/Qwen3.5-9B \
  --jepa-checkpoint /data/model_weights_public/jepa/jeap_vitl_16_256.pt \
  --vjepa-src /root/stream/StreamTeller/site-packages/vjepa2 \
  --task-csv /path/to/Real_Time_Visual_Understanding.csv \
  --video-dir /path/to/StreamingBench/data/real \
  --method predictmem --run-name predictmem_streamingbench_smoke \
  --num-gpus 1 --dry-run
```

## P6：Run Config

每次运行生成：
- `${RESULT_DIR}/run_config.json` — model_path, jepa_checkpoint, vjepa_src, 所有参数 + git_commit
- `${RESULT_DIR}/run_command.sh` — bash 脚本副本

ODry-run 验证输出：
```
OVO predictmem --dry-run:
  Pred runtime: plugin  ✓
  JEPA ckpt: /data/model_weights_public/jepa/jeap_vitl_16_256.pt  ✓
  V-JEPA src: /root/stream/StreamTeller/site-packages/vjepa2  ✓

OVO baseline --dry-run:
  Pred runtime: none  ✓

StreamingBench predictmem --dry-run:
  Pred runtime: plugin  ✓
  MAX_PIXELS passthrough  ✓
```

## P7：测试结果

| 测试文件 | 测试项 | 状态 |
|---|---|---|
| `test_eval_bash_config.py` | runtime auto-detect / dry-run config / Python --help params / no hardcoded ckpt / multi-GPU passthrough | 9/9 ✓ |
| `test_eval_common_predictmem.py` | kwargs / stats / time clip / frame budget / Qwen-JEPA frame match | 8/8 ✓ |
| `test_eval_ovobench.py` | prompts / scoring / path resolver / no evaluation/ paths | 8/8 ✓ |
| `test_eval_streamingbench.py` | timestamp / answer extraction / prompt format / scoring / no evaluation/ paths | 7/7 ✓ |

总计：**31 tests passing**

## 文件变更汇总

| 文件 | 变更 |
|---|---|
| `models/predictmem/config.py` | 新增 jepa_checkpoint_path, vjepa_src_path, tail_keep_frames, drop_bootstrap |
| `models/predictmem/vjepa_scorer.py` | make_vjepa_analyzer_scorer 支持 vjepa_src_path 参数 |
| `models/predictmem/streaming_memory.py` | _ensure_scorer 使用 config 路径，移除硬编码 checkpoint |
| `evaluate/common/qwen35_predictmem.py` | load_qwen35_model 配置 plugin config；build_video_inputs_for_eval 实现 time clipping |
| `evaluate/ovobench/ovobench.py` | build_parser 新增全部配置参数；model loading 传参；multi-GPU 命令完整透传 |
| `evaluate/ovobench/ovobench.sh` | **重写** — 完整配置区、runtime auto-detect、run_config.json、run_command.sh |
| `evaluate/streamingbench/streamingbench.py` | build_parser 新增全部参数；time_window 实际生效；multi-GPU 完整透传 |
| `evaluate/streamingbench/streamingbench.sh` | **重写** — 完整配置区、MAX_PIXELS 透传 |
| `test/test_eval_bash_config.py` | **新增** |
| `test/test_eval_common_predictmem.py` | 新增 video clipping 测试 |

## 下一步

1. OVO-Bench 正式评估（baseline + PredictMem，全量或 50 样本）
2. StreamingBench 正式评估（baseline + PredictMem）
3. V-JEPA scoring 延迟优化后跑 50-100 条对照实验
