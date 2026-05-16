# PredictMem 10-Video Experiment Summary

## Aggregate Metrics

| Metric | Value |
|---|---|
| Samples | 10 |
| Avg baseline score | 0.5 |
| Avg plugin score | 0.6 |
| Avg keep_ratio_actual | 0.1838 |
| Avg token_compression | 6.63x |
| Avg qwen_only_speedup | 2.83x |
| Avg e2e_speedup | 0.34x |
| Avg V-JEPA scoring latency | 29.433s |
| Avg total latency | 31.795s |
| Avg peak memory | 14730.2MB |

## Per-Sample Results

| ID | Task | B→P Score | Orig Tok | Kept Tok | Keep% | Tok Comp | Qwen Spd | E2E Spd | V-JEPA Lat | Total Lat |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | EPM | 1→1 | 27648 | 3946 | 14.3% | 7.01x | 2.59x | 0.09x | 93.919s | 97.432s |
| 1 | EPM | 1→1 | 31232 | 4135 | 13.2% | 7.55x | 3.0x | 0.28x | 29.634s | 32.676s |
| 2 | EPM | 0→0 | 34048 | 4311 | 12.7% | 7.9x | 3.38x | 0.28x | 32.241s | 35.183s |
| 3 | EPM | 0→0 | 47360 | 5650 | 11.9% | 8.38x | 3.53x | 0.29x | 45.431s | 49.456s |
| 4 | EPM | 0→0 | 15360 | 2099 | 13.7% | 7.32x | 3.0x | 0.29x | 13.778s | 15.273s |
| 5 | EPM | 1→1 | 23808 | 3272 | 13.7% | 7.28x | 3.17x | 0.28x | 22.071s | 24.245s |
| 6 | EPM | 0→1 | 24832 | 3382 | 13.6% | 7.34x | 3.16x | 0.28x | 23.111s | 25.38s |
| 7 | EPM | 1→1 | 32000 | 4211 | 13.2% | 7.6x | 3.33x | 0.28x | 30.344s | 33.142s |
| 8 | EPM | 0→0 | 1024 | 538 | 52.5% | 1.9x | 1.12x | 0.91x | 0.123s | 0.641s |
| 9 | EPM | 1→1 | 5120 | 1280 | 25.0% | 4.0x | 1.98x | 0.37x | 3.683s | 4.524s |

### Speedup Definitions

- **token_compression**: original_video_tokens / kept_video_tokens
- **qwen_only_speedup**: baseline_latency / (total_latency - vjepa_scoring_latency)
- **e2e_speedup**: baseline_latency / total_latency