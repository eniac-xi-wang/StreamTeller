# PredictMem 第二轮验证结果：expanding windows、analyzer 对齐、清理

测试时间：2026-05-16

## 执行摘要

按照最新 `instruct.md`（P0-P7），完成了三方向修复：
- **P0+P4**: 主脚本重写为 plugin-only，移除所有 offline/cache/FramePlan 代码
- **P1**: 前 14 帧不再全保留，改用 expanding windows 覆盖 tubelets 1-7
- **P2**: V-JEPA scorer 对齐 Survey analyzer（MultiSeqWrapper、num_mask_tokens=10、ImageNet 归一化）
- **P3**: `models/predictmem` 包清理，legacy 文件移至 `scripts/legacy/`
- **P5**: 新增 16 个测试，覆盖所有关键校验点
- **P6+P7**: Sample 0 端到端验证通过

## P0+P4：主脚本 plugin-only + CLI

`scripts/run_predictmem_ovobench.py` 已重写：

- 移除所有 `ScoreCache`、`FramePlan`、`sample_video_1fps_decord`、`score_vjepa_windows`、`global_scores` 引用
- Plugin 路径只做：`build_predictmem_video_inputs() → processor → model.generate(predictmem_frames_256=...)`
- 新增 `--predictmem_runtime plugin | legacy_offline | none` CLI 参数
- 默认：`predictmem` 方法 → `plugin`，其他方法 → `none`
- 原脚本备份至 `scripts/legacy/run_predictmem_ovobench.py`

验收：
```
rg "ScoreCache|FramePlan|sample_video_1fps_decord" scripts/run_predictmem_ovobench.py → 无命中（仅注释）
```

## P1：前 14 帧不再全保留 — expanding windows

### 修复内容

新增 `iter_predictmem_windows(T)` 函数（`streaming_memory.py`）：

Phase 1 — Expanding windows (tubelets 1-7):
```
target frames: [2,3], [4,5], ..., [14,15]
window lengths: 4, 6, 8, 10, 12, 14, 16
window start: 0 (always)
```

Phase 2 — Standard sliding (tubelets 8+):
```
window length: 16, stride: 2
window start: 2, 4, 6, ...
target: last tubelet of each window
```

t-digest warmup 策略：
- Digest 样本不足时改用 **local quantile** (`torch.quantile(loss, 1 - keep_ratio)`)，不再 keep_all
- Digest 更新在 keep/drop **决策之后**，避免偷看当前 tubelet

### 结果 (Sample 0, 215 frames, 108 tubelets)

| 指标 | 值 |
|---|---|
| num_tubelets_scored | 106 |
| num_tubelets_unscored | 2 (tubelet 0 bootstrap + tubelet 107 boundary) |
| num_tubelets_warmup | 2 |
| early_scored_tubelets | [1, 2, 3, 4, 5, 6, 7, ... 106] ✓ |
| first_full_keep_tubelets | [] (无 blanket warmup) ✓ |
| tdigest_samples | 566 |
| window_mode | expanding+sliding |

tubelet 0 (frames 0-1) 无历史上下文，无法 scoring，默认全保留。tubelet 107 是边界 tubelet（帧 214-215 不完整），无法形成完整 16-frame 窗口覆盖。除这两个 bootstrap/boundary 特殊情况外，所有 tubelet 均通过 expanding 或 sliding window 评分。

## P2：V-JEPA scorer 对齐 Survey analyzer

### 修复内容

`vjepa_scorer.py` 重大更新：

- 新增 `make_vjepa_analyzer_scorer()`（analyzer-compatible builder）
- 使用 `MultiSeqWrapper` 包装 encoder，`PredictorMultiSeqWrapper` 包装 predictor
- `num_mask_tokens=10`（对齐 analyzer）
- `torch.load(..., weights_only=True)` + `"module.backbone."` prefix 清洗
- 加载后打印 `missing_keys`、`unexpected_keys`、`num_mask_tokens`、`wrapper_type`
- 新增 `score_latest_tubelet_variable()` 支持变长窗口（expanding: 4/6/8/10/12/14/16 frames）
- `make_vjepa_encoder_predictor()` 保持向后兼容（委托给 analyzer scorer）

`vision_inputs.py`：
- V-JEPA tensor 改用 **ImageNet normalization** `(0.485,0.456,0.406)/(0.229,0.224,0.225)`，对齐 analyzer
- Qwen 512 frames 保持 uint8 [0,255]

### 验证输出

```
[make_vjepa_analyzer_scorer] encoder missing_keys=0, unexpected_keys=0
[make_vjepa_analyzer_scorer] target_encoder missing_keys=0, unexpected_keys=0
[make_vjepa_analyzer_scorer] predictor missing_keys=0, unexpected_keys=0
[make_vjepa_analyzer_scorer] predictor.num_mask_tokens=10
[make_vjepa_analyzer_scorer] wrapper_type=MultiSeqWrapper/PredictorMultiSeqWrapper
```

## P3：包清理

`models/predictmem/__init__.py` 主路径导出：
```python
PredictMemConfig, PredictMemStreamingMemory, build_predictmem_video_inputs,
VJEPAPredictLossScorer, TokenPruner
```

Legacy 模块保留为 optional try/except 导入。

脚本清理：
```
scripts/legacy/run_predictmem_ovobench.py      ← 旧版主入口备份
scripts/legacy/precompute_predictmem_scores.py  ← 旧 offline 脚本备份
scripts/legacy/smoke_qwen_predictmem.py         ← 旧 smoke 脚本备份
scripts/legacy/visualize_predictmem_scores.py   ← 旧可视化脚本备份
```

## P5：新增测试

| 测试文件 | 测试项 | 状态 |
|---|---|---|
| `test_predictmem_plugin.py` | iter_predictmem_windows T=16/20/30, local quantile, digest update, should_skip | 6/6 ✓ |
| `test_predictmem_vjepa_analyzer_parity.py` | num_mask_tokens=10, ImageNet norm, variable window masks, backward compat | 4/4 ✓ |
| `test_predictmem_cleanup.py` | mainline exports, no legacy imports, legacy preserved, analyzer scorer usage | 5/5 ✓ |
| 已有 M0-M2/M3-M5 测试 | 回归验证 | 14/14 ✓ |

总计：**29 tests passing**

## P6：Sample 0 端到端验证

### 命令

```bash
PYTHONPATH=/root/stream/StreamTeller/models python scripts/run_predictmem_ovobench.py \
  --model_path /data/model_weights_public/Qwen/Qwen3.5-9B \
  --method predictmem --predictmem_runtime plugin --stream_mode full \
  --predictmem_keep_ratio 0.10 --disable_thinking \
  --max_new_tokens 16 --max_samples 1 \
  --output results/plugin_predictmem_sample0_v2.jsonl --device cuda
```

### 结果

| Method | Tokens | Latency | Peak Mem | Answer | Score |
|---|---|---|---|---|---|
| plugin predictmem | 27648→3983 (14.4%) | 98.6s | 13974MB | C | 1 |

### 验收清单

| 条件 | 状态 |
|---|---|
| `predictmem_runtime = plugin` | ✓ |
| 无 score/cache JSON/pt 文件生成 | ✓ (0 new cache files) |
| `use_predictmem=True` 不依赖外部 score 文件 | ✓ |
| `num_tubelets_unscored <= 2` (bootstrap+boundary) | ✓ (2) |
| `early_scored_tubelets` 包含 1-7 | ✓ |
| `predictor.num_mask_tokens == 10` | ✓ |
| `wrapper_type == MultiSeqWrapper/PredictorMultiSeqWrapper` | ✓ |
| `missing_keys == 0, unexpected_keys == 0` | ✓ |
| expanding + sliding window 模式 | ✓ |
| 主脚本 plugin-only，无 legacy import | ✓ |
| 包清理完成，主线无 ScoreCache/FramePlan | ✓ |
| ImageNet normalization | ✓ |
| QA 输出单次回答 | ✓ |
| 前 14 帧不再全保留（expanding windows 覆盖） | ✓ |

### 延迟分析

98.6s 延迟中，~95.4s 用于 V-JEPA scoring（106 windows × ~0.9s/window）。
相比上一版（102.8s / 100 windows），per-window 延迟从 ~1.0s 降至 ~0.9s（analyzer-aligned scorer 略快）。

优化方向：
1. V-JEPA micro-batch
2. 减小 stream 长度（`--frame_budget 64`）
3. Mixed precision / torch.compile

## 文件变更

| 文件 | 变更 |
|---|---|
| `models/predictmem/vjepa_scorer.py` | **重写**：analyzer-compatible scorer (MultiSeqWrapper, num_mask_tokens=10) |
| `models/predictmem/streaming_memory.py` | **重写**：expanding+sliding windows, local quantile warmup |
| `models/predictmem/vision_inputs.py` | **更新**：V-JEPA tensor ImageNet normalization |
| `models/predictmem/__init__.py` | **更新**：主线导出，legacy optional |
| `models/qwen3_5/modeling_qwen3_5.py` | **小修**：存储 predictmem_last_stats |
| `scripts/run_predictmem_ovobench.py` | **重写**：plugin-only，移除所有 offline 遗留代码 |
| `scripts/legacy/` | **新增**：4 个旧脚本备份 |
| `test/test_predictmem_plugin.py` | **新增** |
| `test/test_predictmem_vjepa_analyzer_parity.py` | **新增** |
| `test/test_predictmem_cleanup.py` | **新增** |

## 下一步

1. **优化 V-JEPA scoring 延迟**：micro-batch scoring（多窗口并行）、限制 stream 长度
2. **50-100 条实验**：使用 `--frame_budget 64` 降低单样本时间至 < 30s
3. **FluxMem baseline**：实现 FluxMem-recent/mid 作为对照
4. **Analyzer parity 数值验证**：随机抽 3 个窗口对比 PredictMem loss vs analyzer loss
