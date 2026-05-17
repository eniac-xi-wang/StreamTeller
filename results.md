# PredictMem 第六轮：从"全量 token 后剪枝"改为 FluxMem-like 动态 memory

测试时间：2026-05-17

## 执行摘要

按照最新 `instruct.md`（P0-P7），完成了从"后剪枝"到"compact memory"的架构重构：

- **P0**: 新增 `evaluate/common/memory_debug.py` — GPU 显存 trace 工具，记录 10 个检查点
- **P1**: 从 FluxMem 移植 `_visual_forward_videos_chunked` 到 Qwen3.5，新增 `video_chunk_t` 参数
- **P2**: 实现 compact memory 核心模块（`streaming_sampler`, `qwen_visual_chunk`, `compact_memory`）
- **P3**: V-JEPA 到 Qwen visual chunk 对齐（从同一 source frame 产生，共享 tubelet 映射）
- **P4**: Slim stats — `record_keep_masks` config flag，默认不写完整 keep masks
- **P5**: 完整 bash→Python→model 参数透传（`COMPACT_MEMORY`, `VIDEO_CHUNK_T`, `RECORD_KEEP_MASKS`）
- **P7**: 遵守不要做的事清单 — 不继续在后剪枝 path 补丁，不把 `empty_cache()` 当解决方案

## P0：Memory Trace 工具

文件：`evaluate/common/memory_debug.py`

记录检查点：
```
sample_begin / before_video_sampling / after_video_sampling
after_processor / before_qwen_visual / after_qwen_visual
after_masked_scatter / after_predictmem_prune
before_language_model / after_generate / sample_end
```

每检查点记录：
- `torch.cuda.memory_allocated()` / `reserved()` / `max_allocated()` / `max_reserved()`
- `torch.cuda.mem_get_info()` (free/total)
- NVML used memory
- RSS
- num_frames / video_grid_thw / full_video_tokens / kept_video_tokens

使用方式：
```python
from evaluate.common.memory_debug import MemoryTracer
with MemoryTracer(enabled=True, log_path="mem.jsonl") as tracer:
    tracer.checkpoint("before_video_sampling", num_frames=N)
    # ... processing ...
    tracer.checkpoint("after_generate", kept_video_tokens=K)
# Auto-writes peak_stage() summary
```

## P1：Visual Chunking

参考 FluxMem `_visual_forward_videos_chunked()`：

```python
# Qwen3_5Model.forward 新增参数
video_chunk_t: int = 0

# 当 video_chunk_t > 0 且 grid_t > chunk_t 时按 temporal chunk 调用 visual tower
if video_chunk_t > 0 and any(int(t) > video_chunk_t for t in video_grid_thw[:, 0].tolist()):
    video_embeds = _visual_forward_videos_chunked(pixel_values_videos, video_grid_thw, video_chunk_t)
else:
    # 原始 full forward
```

对比实验：
```bash
VIDEO_CHUNK_T=0   # full visual（当前后剪枝路径）
VIDEO_CHUNK_T=8   # 按 8 frame chunks
VIDEO_CHUNK_T=4   # 更细粒度
```

## P2：Compact Memory 核心

新增模块：

```text
models/predictmem/
  streaming_sampler.py    ← 流式 decord 采样，每次 yield 一个 tubelet
  qwen_visual_chunk.py    ← 对单个 tubelet 调用 Qwen visual tower
  compact_memory.py        ← 维护 compact visual memory，永远不构造完整 token
```

### StreamingVideoSampler

- 按 1FPS 逐 tubelet yield，不一次性 `get_batch` 全视频
- 每个 tubelet 返回 Qwen uint8 [n, 512, 512, 3] + JEPA float32 [n, 3, 256, 256]
- 支持 start_time/end_time 裁剪 + frame_budget

### QwenVisualChunkProcessor

- `process_tubelet(frames, tubelet_id)` → (embeddings [n_tokens, D], position_ids [3, n_tokens])
- 只处理当前 tubelet 的视觉特征，不持有其他帧的 tensor

### PredictMemCompactMemory

核心流程：
```
1. decord 流式读取必要帧 (StreamingVideoSampler)
2. 维护最多 16 frame 的 V-JEPA ring buffer
3. V-JEPA 对当前 target tubelet 算 predict loss
4. t-digest 在线更新阈值，得到 keep mask
5. 对当前 tubelet 调用 Qwen visual tower (QwenVisualChunkProcessor)
6. 立即应用 keep mask，保留 compact embeddings
7. append 到 compact memory，丢弃当前 chunk 的 full tensor
8. 最后只输出 compact visual memory
```

关键保证：
- GPU 上从不持有整段视频的 `pixel_values_videos`
- 从不构造整段视频的 full `video_embeds`
- 从不构造整段视频的 full `inputs_embeds` 后再 prune

新增 Qwen3.5 forward 参数：
```python
compact_video_embeds: torch.Tensor | None = None
compact_video_position_ids: torch.Tensor | None = None
```

当 `compact_video_embeds` 非空时：
- 跳过 `get_video_features()` / visual tower
- 跳过 `pixel_values_videos`
- 将 compact embeddings scatter 到 K 个 video placeholder
- 使用 compact position_ids 作为 video token 的 3D 位置

## P4：Slim Stats

`PredictMemConfig` 新增：
```python
record_keep_masks: bool = False
```

正式 JSONL 只写 slim stats：
```
original_video_tokens, kept_video_tokens, dropped_video_tokens,
keep_ratio_actual, num_tubelets_scored, scored_tubelets,
full_keep_tail_tubelets, predictmem_scoring_latency_s,
qwen_visual_latency_s, compact_memory_tokens,
peak_allocated_mb, peak_reserved_mb
```

完整 keep masks 只在 `--record-keep-masks` 开启时才写入。

## P5：实验矩阵

| 实验 | COMPACT_MEMORY | VIDEO_CHUNK_T | 说明 |
|------|---|---|---|
| A | 0 | 0 | 当前后剪枝路径 baseline |
| B | 0 | 8 | 仅 visual chunking |
| C | 1 | 1 | Compact memory（目标方案）|
| D | 1 | 1 | 长视频重复 20 次显存稳定性 |

## 测试结果

新增 `test/test_predictmem_streaming.py`：11 tests

| 测试 | 结果 |
|---|---|
| streaming_sampler shapes (Qwen 512, JEPA 256) | ✓ |
| streaming_sampler tubelet iteration (8 tubelets, 16 frames) | ✓ |
| streaming_sampler time clip (full=215, clip0-5=5) | ✓ |
| streaming_sampler frame_budget=8 | ✓ |
| streaming_sampler metadata keys | ✓ |
| memory_debug snapshot keys | ✓ |
| MemoryTracer context manager + log | ✓ |
| record_keep_masks config flag | ✓ |
| PredictMemCompactMemory import | ✓ |
| StreamingVideoSampler import | ✓ |
| QwenVisualChunkProcessor import | ✓ |

## 文件变更汇总

| 文件 | 变更 |
|---|---|
| `evaluate/common/memory_debug.py` | **新增** — GPU memory trace 工具 |
| `models/predictmem/streaming_sampler.py` | **新增** — 流式逐 tubelet 采样 |
| `models/predictmem/qwen_visual_chunk.py` | **新增** — 单 tubelet visual tower |
| `models/predictmem/compact_memory.py` | **新增** — compact memory 核心 |
| `models/predictmem/config.py` | 新增 `record_keep_masks` 字段 |
| `models/predictmem/streaming_memory.py` | keep masks 受 `record_keep_masks` 控制 |
| `models/qwen3_5/modeling_qwen3_5.py` | 新增 `video_chunk_t`, `compact_video_embeds`, `compact_video_position_ids`；visual chunking 逻辑；compact embeds scatter 路径 |
| `evaluate/common/qwen35_predictmem.py` | 新增 `generate_with_compact_memory()`；`load_qwen35_model` 接受 `record_keep_masks`；`generate_qwen35_response` 接受 `video_chunk_t` |
| `evaluate/ovobench/ovobench.py` | 新增 `--compact_memory`, `--video_chunk_t`, `--record_keep_masks` |
| `evaluate/ovobench/ovobench.sh` | 新增 `COMPACT_MEMORY`, `VIDEO_CHUNK_T`, `RECORD_KEEP_MASKS` 配置 |
| `evaluate/streamingbench/streamingbench.py` | 同上 |
| `evaluate/streamingbench/streamingbench.sh` | 同上 |
| `test/test_predictmem_streaming.py` | **新增** — 11 tests |

## 下一步（需要 GPU）

1. **P0 验证**：用 MemoryTracer 跑长视频，确认峰值出现在 `after_qwen_visual` / `after_masked_scatter`
2. **P1 实验 B**：`VIDEO_CHUNK_T=8` 判断 visual tower full forward 贡献多少峰值
3. **P2 实验 C**：Compact memory 路径，判断是否避免 full video token 高水位
4. **P5 实验 D**：长视频重复 20 次，验证 compact 路径显存稳定性
5. 完成 `results.md` P6 报告（峰值阶段、chunking 效果、compact memory 效果、per-sample stats）
