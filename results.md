# PredictMem 第四轮验证结果：评估框架建立

测试时间：2026-05-16

## 执行摘要

按照最新 `instruct.md`（P0-P6），参考 FluxMem evaluation 结构，完成了：
- **P0+P1**: `evaluate/` 目录结构 + 公共 Qwen3.5/PredictMem helper
- **P2**: OVO-Bench 评估入口（ovobench.py, score.py, ovobench.sh）
- **P3**: StreamingBench 评估入口（streamingbench.py, score.py, streamingbench.sh）
- **P4**: Baseline vs PredictMem 对照支持（--method + --baseline_result_dir）
- **P5**: 18 个新测试（common helper, OVO, StreamingBench）
- **P6**: OVO-Bench smoke test + 更新 results.md

## P0+P1：文件布局 + 公共 helper

### 新增文件结构

```
evaluate/
  common/
    __init__.py                        # 导出 load_qwen35_model, etc.
    qwen35_predictmem.py               # 公共加载/采样/generate helper
  ovobench/
    ovo_bench_new.json                 # OVO-Bench 数据（已存在）
    ovobench.py                        # 主评估入口
    score.py                           # 合并 + 打分 + summary
    ovobench.sh                        # 启动脚本
  streamingbench/
    streamingbench.py                  # 主评估入口
    score.py                           # 合并 + 打分 + summary
    streamingbench.sh                  # 启动脚本
```

### common/qwen35_predictmem.py 提供

| 函数 | 用途 |
|---|---|
| `load_qwen35_model()` | 加载 Qwen3.5-9B + PredictMem 插件初始化 |
| `load_qwen35_processor()` | 加载 processor（fps=1, do_resize=False） |
| `build_video_inputs_for_eval()` | 视频采样 → Qwen 512 + V-JEPA 256 |
| `generate_qwen35_response()` | 单样本生成（baseline 或 PredictMem plugin） |
| `extract_predictmem_stats()` | 从 model 提取 predictmem_last_stats |

关键约束：
- baseline → `use_predictmem=False`，不传 `predictmem_frames_256`
- predictmem plugin → `use_predictmem=True` + `predictmem_frames_256` + `predictmem_keep_ratio`
- 不使用 offline score cache
- processor 参数：`do_sample_frames=False, do_resize=False, fps=1`

## P2：OVO-Bench 评估入口

### ovobench.py

支持功能：
- Backward (EPM, ASI, HLD) / Realtime (STU, OJR, ATR, ACR, OCR, FPD) / Forward (REC, SSR, CRR) 任务分类
- 视频路径解析（兼容 chunked_videos 和 OVO 根目录）
- Single-GPU + Multi-GPU 模式
- `--method baseline|predictmem --predictmem_runtime plugin|none`
- Per-sample JSONL 输出含完整 PredictMem stats

### score.py

- 合并 outputs/*.jsonl
- 按 backward/realtime/forward 分类计算 accuracy
- 聚合 PredictMem token/latency 统计
- 生成 `results_merged.json`, `score_merged.json`, `summary.json`, `summary.md`
- 支持 `--baseline_result_dir` 计算 token_compression / qwen_only_speedup / e2e_speedup

### ovobench.sh

Shell 启动脚本：`--model-path`, `--method`, `--num-gpus`, `--max-samples`, 等

## P3：StreamingBench 评估入口

### streamingbench.py

- CSV 读取 + timestamp MM:SS / HH:MM:SS 解析
- 视频路径解析（`sample_{id}/video.mp4`）
- Single-GPU + Multi-GPU 模式
- Prompt 模板：多选题（A-D letters）+ 开放题

### score.py

- 合并 multi-GPU outputs
- 按 task_type 计算 accuracy + avg latency/memory
- 聚合 PredictMem stats
- 生成 `scores.json`, `summary.json`, `summary.md`

### streamingbench.sh

Shell 启动脚本：`--task-csv`, `--video-dir`, `--method`, `--num-gpus`, `--max-num-frames`, 等

## P4：Baseline vs PredictMem 对照

两个 bench 都支持：

```bash
# Baseline
python -m evaluate.ovobench.ovobench --method baseline ...

# PredictMem
python -m evaluate.ovobench.ovobench --method predictmem --predictmem_runtime plugin --predictmem_keep_ratio 0.10 ...
```

score.py 在传入 `--baseline_result_dir` 后计算：
- token_compression = avg_original_video_tokens / avg_kept_video_tokens
- qwen_only_speedup = baseline_avg_latency / avg(qwen_latency_excluding_vjepa)
- e2e_speedup = baseline_avg_latency / predictmem_avg_total_latency

输出目录约定：
```
eval_results/ovobench/baseline_ovobench_{timestamp}/
eval_results/ovobench/predictmem_ovobench_{timestamp}/
eval_results/streamingbench/baseline_streamingbench_{timestamp}/
eval_results/streamingbench/predictmem_streamingbench_{timestamp}/
```

## P5：测试覆盖

| 测试文件 | 测试项 | 状态 |
|---|---|---|
| `test_eval_common_predictmem.py` | baseline kwargs / plugin kwargs / stats extraction / stats structure | 4/4 ✓ |
| `test_eval_ovobench.py` | MC prompt / REC prompt / SSR prompt / video path / score_all / REC scoring / SSR scoring / no hardcoded evaluation/ | 8/8 ✓ |
| `test_eval_streamingbench.py` | time_to_seconds / extract_answer / format_prompt / format_prompt no opts / score calc / video path / no evaluation/ | 7/7 ✓ |
| 已有所有测试 | 回归验证 | 45/45 ✓ |

总计：**67 tests passing**

## P6：Smoke Test

### OVO-Bench PredictMem Smoke (2 samples, EPM+ASI+HLD)

命令：
```
PYTHONPATH=/root/stream/StreamTeller/models python -m evaluate.ovobench.ovobench \
  --model_path /data/model_weights_public/Qwen/Qwen3.5-9B \
  --method predictmem --predictmem_runtime plugin --predictmem_keep_ratio 0.10 \
  --run_name predictmem_ovobench_smoke --max_samples 2 --max_new_tokens 16 --fps 1.0
```

结果：
- EPM 100% (2/2 correct)
- Avg keep ratio: 13.75%
- Avg scoring latency: 63.2s
- Avg total latency: 66.5s
- Avg peak memory: 15174MB

输出目录：
```
eval_results/ovobench/predictmem_ovobench_smoke_20260516_112540/
  outputs/predictmem_ovobench_smoke.jsonl
  log/predictmem_ovobench_smoke.log
  results/results_merged.json
  results/score_merged.json
  results/summary.json
  results/summary.md          ← 可直接读的评估结论
```

## 文件变更汇总

| 文件 | 变更 |
|---|---|
| `evaluate/common/__init__.py` | **新增** |
| `evaluate/common/qwen35_predictmem.py` | **新增** — 公共 helper |
| `evaluate/ovobench/ovobench.py` | **新增** — OVO-Bench 主评估入口 |
| `evaluate/ovobench/score.py` | **新增** — OVO-Bench 打分 + summary |
| `evaluate/ovobench/ovobench.sh` | **新增** — OVO-Bench 启动脚本 |
| `evaluate/streamingbench/streamingbench.py` | **新增** — StreamingBench 主评估入口 |
| `evaluate/streamingbench/score.py` | **新增** — StreamingBench 打分 + summary |
| `evaluate/streamingbench/streamingbench.sh` | **新增** — StreamingBench 启动脚本 |
| `test/test_eval_common_predictmem.py` | **新增** |
| `test/test_eval_ovobench.py` | **新增** |
| `test/test_eval_streamingbench.py` | **新增** |

## 下一步

1. OVO-Bench smoke test 完成并验证 autoscoring
2. StreamingBench 数据路径确认后跑 smoke
3. Baseline + PredictMem 对照实验（OVO-Bench 全量 + StreamingBench 全量）
4. V-JEPA scoring 延迟优化后跑 50-100 条正式实验
