#!/usr/bin/env python3
"""Run OVO-Bench evaluation with PredictMem token pruning.

Supports: baseline, random, uniform, predictmem.

Usage:
    python scripts/run_predictmem_ovobench.py \
        --method predictmem \
        --cache_path results/predictmem_scores.jsonl \
        --max_samples 5 --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

_models_dir = Path(__file__).parent.parent / "models"
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from predictmem.config import PredictMemConfig
from predictmem.token_mapping import TokenMapper
from predictmem.cache import ScoreCache


# ─── Results logger ───────────────────────────────────────────────────────────

class EvalLogger:
    REQUIRED_FIELDS = [
        "sample_id", "video", "question", "ground_truth", "prediction",
        "method", "keep_ratio_target", "keep_ratio_actual",
        "original_video_tokens", "kept_video_tokens",
        "score_latency_s", "vision_latency_s",
        "prefill_latency_s", "decode_latency_s", "total_latency_s",
        "peak_memory_mb",
    ]

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results = []

    def log(self, entry: dict):
        for field in self.REQUIRED_FIELDS:
            entry.setdefault(field, None)
        self.results.append(entry)

    def flush(self):
        with open(self.output_path, "w") as f:
            for r in self.results:
                f.write(json.dumps(r) + "\n")


# ─── Sample loader ────────────────────────────────────────────────────────────

def load_ovo_samples(bench_path: str, max_samples: int, video_base: str) -> list[dict]:
    """Load OVO-Bench samples, mapping video names to chunked video paths."""
    with open(bench_path) as f:
        data = json.load(f)

    samples = []
    for item in data[:max_samples]:
        vid_name = Path(item["video"]).stem
        samples.append({
            "sample_id": str(item["id"]),
            "video": str(Path(video_base) / f"{item['id']}.mp4"),
            "question": item["question"],
            "ground_truth": item["answer"],
            "options": item.get("options", []),
            "gt_idx": item.get("gt"),
        })
    return samples


# ─── Keep-index generators per method ─────────────────────────────────────────

def generate_keep_indices(method: str, config: PredictMemConfig, sample_id: str,
                          cache: ScoreCache | None = None, seed: int = 0,
                          video_grid_thw: torch.Tensor | None = None) -> tuple[list, dict]:
    """Generate keep_indices for one sample. Returns (keep_indices_list, stats_dict)."""
    mapper = TokenMapper(config)
    if video_grid_thw is None:
        video_grid_thw = torch.tensor([[config.qwen_grid_t, config.qwen_grid_h, config.qwen_grid_w]])
    num_tokens = mapper.compute_num_video_tokens(video_grid_thw)
    stats = {"original_video_tokens": num_tokens}

    if method == "baseline":
        keep = [torch.arange(num_tokens)]
        stats["keep_ratio_actual"] = 1.0
        stats["kept_video_tokens"] = num_tokens

    elif method == "random":
        torch.manual_seed(seed)
        n_keep = max(1, int(num_tokens * config.keep_ratio))
        perm = torch.randperm(num_tokens)
        keep = [perm[:n_keep].sort().values]
        stats["keep_ratio_actual"] = n_keep / num_tokens
        stats["kept_video_tokens"] = n_keep

    elif method == "uniform":
        n_keep = max(1, int(num_tokens * config.keep_ratio))
        step = num_tokens / n_keep
        indices = torch.arange(num_tokens, dtype=torch.float)
        keep_local = torch.round(torch.arange(n_keep).float() * step).long()
        keep_local = keep_local.clamp(0, num_tokens - 1).unique()
        keep = [keep_local.sort().values]
        stats["keep_ratio_actual"] = len(keep[0]) / num_tokens
        stats["kept_video_tokens"] = len(keep[0])

    elif method == "predictmem":
        if cache is not None and cache.has(sample_id):
            loss_map = cache.get_loss_map(sample_id)
            keep_mask = None if loss_map is not None else cache.get_keep_mask(sample_id)
            if loss_map is None and keep_mask is None:
                raise ValueError(f"Cached sample {sample_id} has neither loss_map nor keep_mask")
            keep = mapper.map_scores_to_qwen_keep_indices(
                video_grid_thw=video_grid_thw,
                loss_map=loss_map,
                keep_mask=keep_mask,
                keep_ratio=config.keep_ratio,
            )
            stats["keep_ratio_actual"] = len(keep[0]) / num_tokens
            stats["kept_video_tokens"] = len(keep[0])
        else:
            raise ValueError(f"Sample {sample_id} not in cache, run precompute first")
    else:
        raise ValueError(f"Unknown method: {method}")

    return keep, stats


# ─── Main eval loop ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["baseline", "random", "uniform", "predictmem"])
    parser.add_argument("--cache_path", default="results/predictmem_scores.jsonl")
    parser.add_argument("--bench_path", default="evaluate/ovobench/ovo_bench_new.json")
    parser.add_argument("--video_dir", default="/data/qinian_workspace/OVO-Bench/chunked_videos")
    parser.add_argument("--output", default="results/eval_output.jsonl")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--qwen_grid_t", type=int, default=8)
    parser.add_argument("--qwen_grid_h", type=int, default=32)
    parser.add_argument("--qwen_grid_w", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PredictMemConfig()
    config.keep_ratio = args.keep_ratio
    video_grid_thw = torch.tensor([[args.qwen_grid_t, args.qwen_grid_h, args.qwen_grid_w]])

    # Load cache for predictmem method
    cache = None
    if args.method == "predictmem":
        cache = ScoreCache(args.cache_path)
        if len(cache) == 0:
            print("ERROR: cache is empty, run precompute_predictmem_scores.py first")
            sys.exit(1)
        print(f"Loaded cache with {len(cache)} entries")

    # Load samples
    samples = load_ovo_samples(args.bench_path, args.max_samples, args.video_dir)
    print(f"Loaded {len(samples)} samples, method={args.method}")

    logger = EvalLogger(args.output)
    torch.manual_seed(args.seed)

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {sample['sample_id']}")

        keep_indices_list, stats = generate_keep_indices(
            method=args.method, config=config, sample_id=sample["sample_id"],
            cache=cache, seed=args.seed + i, video_grid_thw=video_grid_thw,
        )

        entry = {
            "sample_id": sample["sample_id"],
            "video": sample["video"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "prediction": f"[simulated_{args.method}]",
            "method": args.method,
            "keep_ratio_target": config.keep_ratio,
            "keep_ratio_actual": stats["keep_ratio_actual"],
            "original_video_tokens": stats["original_video_tokens"],
            "kept_video_tokens": stats["kept_video_tokens"],
            "score_latency_s": 0.0,
            "vision_latency_s": 0.0,
            "prefill_latency_s": 0.0,
            "decode_latency_s": 0.0,
            "total_latency_s": 0.0,
            "peak_memory_mb": 0.0,
        }
        logger.log(entry)

    logger.flush()
    print(f"\nDone. Results: {args.output}")


if __name__ == "__main__":
    main()
