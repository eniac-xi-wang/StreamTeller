# StreamTeller / PredictMem 项目总结

## 项目概述

**目标**：在 streaming video 场景下，用 V-JEPA prediction loss 做 Qwen3.5-VL 视觉 token 剪枝——高 loss patch 保留，低 loss patch 丢弃，降低 LLM 输入 token 数，从而减少 prefill 延迟和 KV cache 显存。

**核心思路**：FluxMem-like 的 streaming memory 机制。V-JEPA 对每个 tubelet（2帧）算 predict loss，通过 t-digest 在线估计阈值，只保留 top-10% 高 loss 的 visual patches，丢弃冗余 token。

---

## 架构

```
StreamTeller/
├── models/
│   ├── predictmem/              # PredictMem 插件（与 Qwen 解耦）
│   │   ├── config.py            #   配置 dataclass
│   │   ├── vjepa_scorer.py      #   V-JEPA encoder/predictor，算 predict loss
│   │   ├── streaming_memory.py  #   FluxMem-like 插件，在线 scoring + pruning
│   │   ├── token_pruner.py      #   按 keep mask 裁剪 inputs_embeds/pos_ids/attn
│   │   ├── vision_inputs.py     #   视频解码 + resize (Qwen 512 + JEPA 256)
│   │   ├── streaming_sampler.py #   流式逐 tubelet 采样（P2 compact memory）
│   │   ├── qwen_visual_chunk.py #   单 tubelet Qwen visual tower（P2）
│   │   ├── compact_memory.py    #   Compact memory 核心（P2，未完成端到端）
│   │   └── legacy/              #   旧模块（token_mapping, frame_plan 等）
│   └── qwen3_5/                 # Qwen3.5-9B VL 本地实现
│       └── modeling_qwen3_5.py  #   接入 PredictMem + visual chunking + compact
├── evaluate/                    # 评测框架
│   ├── common/
│   │   ├── qwen35_predictmem.py #   模型加载 / 视频采样 / 推理（共享）
│   │   └── memory_debug.py      #   GPU 显存 trace 工具（P0）
│   ├── ovobench/                # OVO-Bench 评测
│   │   ├── ovobench.sh          #   bash 配置入口
│   │   ├── ovobench.py          #   Python 主逻辑
│   │   └── score.py             #   分数计算
│   └── streamingbench/          # StreamingBench 评测
│       ├── streamingbench.sh    #   bash 配置入口
│       ├── streamingbench.py    #   Python 主逻辑
│       └── score.py             #   分数计算
├── test/                        # 13 个测试文件
├── results.md                   # 历轮验证结果
└── site-packages/               # 参考实现
    ├── FluxMem/                  #   FluxMem 参考
    └── vjepa2/                   #   V-JEPA 参考
```

---

## 关键设计决策

### Token 映射
- V-JEPA 输入：16 frames × 256 × 256，temporal_patch=2 → grid [8,16,16] = 2048 tokens
- Qwen 输入：16 frames × 512 × 512，temporal_patch=2 → grid [8,32,32]，merge=2 → 2048 LLM tokens
- 一一对应：`qwen_llm_video_token_id == vjepa_token_id`

### 边界策略
- Tubelet 0（frames 0-1）：始终丢弃（无历史上下文）
- 最后 4 帧（tail）：始终全保留（安全缓冲）
- 中间 tubelet：V-JEPA loss + t-digest，保留 top ~10% 高 loss patches

### 窗口策略
- Phase 1（tubelets 1-7）：expanding window（从 frame 0 开始，长度从 4 增长到 16）
- Phase 2（tubelets 8+）：standard sliding window（16 帧窗口，stride 2）

### 配置体系
- bash 脚本是唯一正式配置入口
- Python 无硬编码路径
- 每次运行自动保存 `run_config.json` + `run_command.sh`

---

## 当前状态

### 已完成 ✓

| 功能 | 状态 |
|---|---|
| V-JEPA analyzer-aligned scorer | ✓（与 Survey 零数值差异）|
| PredictMem 插件（后剪枝） | ✓ |
| Expanding + sliding 窗口 | ✓ |
| Boundary policy（tubelet 0 drop + tail full keep） | ✓ |
| t-digest 在线阈值估计 | ✓ |
| TokenPruner（保留非 video token） | ✓ |
| OVO-Bench + StreamingBench 评测框架 | ✓ |
| Bash 配置入口 + 参数全透传 | ✓ |
| Per-sample 统计（tokens, latency, memory） | ✓ |
| Memory trace 工具（P0） | ✓ |
| Visual chunking（P1） | ✓（bash→Python→model 全透传，3 个 bug 已修）|
| Compact memory 核心模块（P2） | ✓（代码完成，端到端待集成）|
| Slim stats（record_keep_masks flag） | ✓ |
| 11+ 个单元测试全部通过 | ✓ |

### 待完成

| 任务 | 说明 |
|---|---|
| P2 端到端集成 | `generate_with_compact_memory()` 已实现，需要在 ovobench.py 中接入 |
| 实验 A/B/C/D | 后剪枝 / chunking / compact / 长视频重复测试（需要 GPU） |
| V-JEPA scoring 延迟优化 | 微批处理、混合精度 |
| StreamingBench 实际跑分 | 数据路径已确认，需要运行 |

### 已知问题

1. **显存高水位**：当前后剪枝路径先构造完整 `pixel_values_videos` → full visual tower → full `inputs_embeds`，再剪枝。长视频（>1000帧）峰值显存可达 65GB。统计的 `peak_memory_mb`（avg 26GB）只覆盖 per-sample 增量，不包括模型权重基线（~18GB）和 PyTorch 缓存残留。
2. **V-JEPA scoring 占延迟主导**：avg 28.8s scoring vs 1.9s Qwen（30.7s 总延迟），scoring 占 94%。
3. **Visual chunking 有 3 个已修复的 bug**（`prepare_inputs_for_generation` 不转发新参数、`get_video_features` 装饰器干扰返回类型、`self.visual()` 不返回 tensor），验证实验待跑。
4. **Forward 任务 prompt 缺少上下文**（REC/SSR/CRR 比 Survey 参考少很多指令），尚未修改。

---

## OVO-Bench 全量跑分结果

运行：`predictmem_ovobench_20260516_151804`（1468 samples，2 GPU）

| 指标 | 值 |
|---|---|
| **Overall Accuracy** | 55.78% |
| Real-Time Visual Perception | 70.40% (STU 61.8, OJR 66.9, ATR 72.4, ACR 60.6, OCR 82.6, FPD 78.2) |
| Backward Tracing | 46.93% (EPM 56.6, ASI 67.6, HLD 16.7) |
| Forward Active Responding | 50.01% (REC 33.0, SSR 63.8, CRR 53.3) |
| Avg original video tokens | 30,216 |
| Avg kept video tokens | 3,726 |
| Avg keep ratio | 17.0% (6.6x compression) |
| Avg scoring latency | 28.8s |
| Avg total latency | 30.7s |
| Avg peak memory | 26,566 MB |
| Max peak memory（最大视频） | 64,960 MB（1695帧，217K tokens） |

---

## 使用方式

### OVO-Bench
```bash
# Smoke test（2 samples）
bash evaluate/ovobench/ovobench.sh --method predictmem --max-samples 2 --num-gpus 1

# 全量（所有 task）
bash evaluate/ovobench/ovobench.sh --method predictmem --num-gpus 2

# Visual chunking（降低显存）
bash evaluate/ovobench/ovobench.sh --method predictmem --video-chunk-t 8 --max-samples 20
```

### StreamingBench
```bash
bash evaluate/streamingbench/streamingbench.sh \
  --method predictmem \
  --task-csv /data/qinian_workspace/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv \
  --video-dir /data/qinian_workspace/StreamingBench/data/real \
  --max-num-frames 0 --num-gpus 1
```

### 单元测试
```bash
cd /root/stream/StreamTeller
export PYTHONPATH="$PWD:$PWD/evaluate:$PWD/models:$PWD/site-packages/vjepa2"
for f in test/test_*.py; do python "$f" && echo "PASS: $f"; done
# 结果: 11 pass, 0 fail, 2 skip
```
