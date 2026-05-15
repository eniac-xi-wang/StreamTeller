# PredictMem 测试结果总结

测试时间：2026-05-15

## 概述

实现了基于 V-JEPA prediction loss 的 streaming video 视觉 token 剪枝系统。包含 6 个新模块和 Qwen3.5-VL 模型的最小侵入式集成。

## 模块清单

| 模块 | 文件 | 行数(约) | 状态 |
|---|---|---|---|
| Config | `models/predictmem/config.py` | 40 | 完成 |
| Token Mapping | `models/predictmem/token_mapping.py` | 72 | 完成 |
| Token Pruner | `models/predictmem/token_pruner.py` | 106 | 完成 |
| V-JEPA Scorer | `models/predictmem/vjepa_scorer.py` | 194 | 完成 |
| Score Cache | `models/predictmem/cache.py` | 86 | 完成 |
| Qwen3.5 集成 | `models/qwen3_5/modeling_qwen3_5.py` | +55 行 | 完成 |

## 测试结果 (12/12 通过)

### M0 — 形状测试

- `video_grid_thw == [8, 32, 32]` 断言通过
- Qwen LLM video token 数 = `8 * 32 * 32 / 4 = 2048`
- V-JEPA 256 token 数 = `8 * 16 * 16 = 2048`
- 两端 token 数量一致

### M1 — Token 映射测试

- JEPA token ID ↔ Qwen video token local ID 恒等映射
- 每个 2-frame tubelet 覆盖 256 个 token，tubelet 0 (0–255) 到 tubelet 7 (1792–2047)
- `loss_map = arange(2048).view(8,16,16)` + `keep_ratio=0.5` → top-1024 保留、最小值 ≥ 1023

### M2 — Pruner 单元测试

- 非 video token（text、vision_start、vision_end）全保留
- `inputs_embeds` / `position_ids` / `attention_mask` 长度一致
- batch 内不同 keep 数 → 正确右 padding，attention mask 对齐
- 随机 keep indices 生成器正确

### M3 — Qwen forward smoke test

- `keep_ratio=1.0`：输出 shape 与输入一致，所有 token 保留
- `keep_ratio=0.5`：shape 正确缩减，无 token-feature mismatch、无 position id shape mismatch
- labels 与 token 同步裁剪，pad 位填充 -100

### M4 — Scorer smoke test

- 随机 video tensor `[1, 3, 16, 256, 256]` 通过 V-JEPA encoder + predictor
- `loss_map` shape = `[1, 8, 16, 16]`
- `keep_ratio=0.5` 保留 1024/2048 tokens
- cell coverage (min_cell_keep) 确保每个 4×4 空间 cell 至少保留 1 个 token

### M5 — 评测脚本测试

- 5 条合成样本模拟 baseline / random / PredictMem 模式
- 每条结果含 6 个必需字段：`original_video_tokens`、`kept_video_tokens`、`keep_ratio_actual`、`prefill_latency_s`、`total_latency_s`、`peak_memory_mb`
- mode 聚合对比正常工作

## 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| V-JEPA 输入 | 16f × 256 × 256 | 固定 |
| Qwen 输入 | 16f × 512 × 512 | 固定 |
| patch_size | 16 | 两端一致 |
| temporal patch | 2 frames | V-JEPA tubelet 和 Qwen temporal_patch_size |
| JEPA token grid | 8 × 16 × 16 = 2048 | |
| Qwen LLM video token grid | 8 × 16 × 16 = 2048 | merge_size=2 |
| keep_ratio (默认) | 0.5 | 保留 50% 视觉 token |
| score_mode | rank | 按 loss 排序取 top-k |
| cell_grid_size | 4 | 每个 cell 至少保留 1 个 token |

## 已知限制

1. V-JEPA scorer 目前使用随机初始化权重测试。需下载预训练权重（V-JEPA 2 ViT-Large checkpoint）才能获得有意义的 prediction loss
2. 第一版仅支持离线评分模式（offline）。online 模式需传入 `frames_256` 并在 forward 外部准备好
3. Qwen visual encoder FLOPs 未减少——剪枝发生在 visual embedding 之后、LLM 输入之前
4. Decode 阶段自动跳过剪枝（`seq_len=1` 检测）
