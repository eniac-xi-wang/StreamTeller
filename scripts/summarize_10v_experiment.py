#!/usr/bin/env python3
"""Summarize the 10-video experiment: baseline vs plugin PredictMem.

Computes:
  - token_compression = original_video_tokens / kept_video_tokens
  - qwen_only_speedup = baseline_latency / qwen_latency_excluding_vjepa
  - e2e_speedup = baseline_latency / predictmem_total_latency

Usage:
    python scripts/summarize_10v_experiment.py \
        --baseline results/baseline_10.jsonl \
        --plugin results/predictmem_plugin_10.jsonl \
        --output-json results/predictmem_10_summary.json \
        --output-md results/predictmem_10_summary.md
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_speedups(baseline: list[dict], plugin: list[dict]) -> dict:
    """Compute per-sample and aggregate speedup metrics."""
    # Index by sample_id
    bl_by_id = {str(e["sample_id"]): e for e in baseline}
    pl_by_id = {str(e["sample_id"]): e for e in plugin}

    common_ids = sorted(set(bl_by_id.keys()) & set(pl_by_id.keys()),
                        key=lambda x: int(x) if x.isdigit() else 0)

    if not common_ids:
        print("WARNING: No common sample_ids between baseline and plugin")
        return {"samples": [], "aggregate": {}}

    samples = []
    for sid in common_ids:
        bl = bl_by_id[sid]
        pl = pl_by_id[sid]

        bl_lat = bl.get("total_latency_s", 0)
        pl_lat = pl.get("total_latency_s", 0)
        vjepa_lat = pl.get("predictmem_scoring_latency_s", 0)
        qwen_only_lat = pl.get("qwen_latency_excluding_vjepa_s", pl_lat - vjepa_lat)

        orig_tokens = pl.get("expected_video_tokens", 0)
        kept_tokens = pl.get("predictmem_stats", {}).get("kept_video_tokens", 0)
        if kept_tokens == 0:
            # Try top-level
            kept_tokens = pl.get("kept_video_tokens", 0)
        keep_ratio = pl.get("predictmem_stats", {}).get("keep_ratio_actual", 0)
        if keep_ratio == 0:
            keep_ratio = kept_tokens / orig_tokens if orig_tokens > 0 else 1.0

        token_compression = orig_tokens / kept_tokens if kept_tokens > 0 else 1.0
        qwen_only_speedup = bl_lat / qwen_only_lat if qwen_only_lat > 0 else 0
        e2e_speedup = bl_lat / pl_lat if pl_lat > 0 else 0

        pm_stats = pl.get("predictmem_stats", {})
        samples.append({
            "sample_id": sid,
            "task": pl.get("task", ""),
            "baseline_score": bl.get("score", 0),
            "plugin_score": pl.get("score", 0),
            "baseline_latency_s": round(bl_lat, 3),
            "plugin_total_latency_s": round(pl_lat, 3),
            "vjepa_scoring_latency_s": round(vjepa_lat, 3),
            "qwen_only_latency_s": round(qwen_only_lat, 3),
            "peak_memory_mb": round(pl.get("peak_memory_mb", 0), 1),
            "original_video_tokens": orig_tokens,
            "kept_video_tokens": kept_tokens,
            "keep_ratio_actual": round(keep_ratio, 4),
            "token_compression": round(token_compression, 2),
            "qwen_only_speedup": round(qwen_only_speedup, 2),
            "e2e_speedup": round(e2e_speedup, 2),
            "dropped_bootstrap_tubelets": pm_stats.get("dropped_bootstrap_tubelets", []),
            "full_keep_tail_tubelets": pm_stats.get("full_keep_tail_tubelets", []),
            "num_tubelets_scored": pm_stats.get("num_tubelets_scored", 0),
        })

    n = len(samples)
    agg = {
        "num_samples": n,
        "avg_baseline_score": round(sum(s["baseline_score"] for s in samples) / n, 3) if n else 0,
        "avg_plugin_score": round(sum(s["plugin_score"] for s in samples) / n, 3) if n else 0,
        "avg_keep_ratio_actual": round(sum(s["keep_ratio_actual"] for s in samples) / n, 4) if n else 0,
        "avg_token_compression": round(sum(s["token_compression"] for s in samples) / n, 2) if n else 0,
        "avg_qwen_only_speedup": round(sum(s["qwen_only_speedup"] for s in samples) / n, 2) if n else 0,
        "avg_e2e_speedup": round(sum(s["e2e_speedup"] for s in samples) / n, 2) if n else 0,
        "avg_vjepa_scoring_latency_s": round(sum(s["vjepa_scoring_latency_s"] for s in samples) / n, 3) if n else 0,
        "avg_total_latency_s": round(sum(s["plugin_total_latency_s"] for s in samples) / n, 3) if n else 0,
        "avg_peak_memory_mb": round(sum(s["peak_memory_mb"] for s in samples) / n, 1) if n else 0,
    }

    return {"samples": samples, "aggregate": agg}


def format_md(summary: dict) -> str:
    """Render summary as markdown table."""
    agg = summary["aggregate"]
    samples = summary["samples"]

    lines = [
        "# PredictMem 10-Video Experiment Summary",
        "",
        "## Aggregate Metrics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Samples | {agg['num_samples']} |",
        f"| Avg baseline score | {agg['avg_baseline_score']} |",
        f"| Avg plugin score | {agg['avg_plugin_score']} |",
        f"| Avg keep_ratio_actual | {agg['avg_keep_ratio_actual']} |",
        f"| Avg token_compression | {agg['avg_token_compression']}x |",
        f"| Avg qwen_only_speedup | {agg['avg_qwen_only_speedup']}x |",
        f"| Avg e2e_speedup | {agg['avg_e2e_speedup']}x |",
        f"| Avg V-JEPA scoring latency | {agg['avg_vjepa_scoring_latency_s']}s |",
        f"| Avg total latency | {agg['avg_total_latency_s']}s |",
        f"| Avg peak memory | {agg['avg_peak_memory_mb']}MB |",
        "",
        "## Per-Sample Results",
        "",
        "| ID | Task | B→P Score | Orig Tok | Kept Tok | Keep% | Tok Comp | Qwen Spd | E2E Spd | V-JEPA Lat | Total Lat |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for s in samples:
        lines.append(
            f"| {s['sample_id']} | {s['task']} | {s['baseline_score']}→{s['plugin_score']} | "
            f"{s['original_video_tokens']} | {s['kept_video_tokens']} | "
            f"{s['keep_ratio_actual']:.1%} | {s['token_compression']}x | "
            f"{s['qwen_only_speedup']}x | {s['e2e_speedup']}x | "
            f"{s['vjepa_scoring_latency_s']}s | {s['plugin_total_latency_s']}s |"
        )

    lines.extend([
        "",
        "### Speedup Definitions",
        "",
        "- **token_compression**: original_video_tokens / kept_video_tokens",
        "- **qwen_only_speedup**: baseline_latency / (total_latency - vjepa_scoring_latency)",
        "- **e2e_speedup**: baseline_latency / total_latency",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--output-json", default="results/predictmem_10_summary.json")
    parser.add_argument("--output-md", default="results/predictmem_10_summary.md")
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    plugin = load_jsonl(args.plugin)
    print(f"Loaded: {len(baseline)} baseline, {len(plugin)} plugin entries")

    summary = compute_speedups(baseline, plugin)

    # Write JSON
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {output_json}")

    # Write MD
    md_text = format_md(summary)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w") as f:
        f.write(md_text)
    print(f"Wrote: {output_md}")

    # Print quick summary
    agg = summary["aggregate"]
    print(f"\nQuick: avg_score={agg['avg_baseline_score']:.2f}→{agg['avg_plugin_score']:.2f}, "
          f"compression={agg['avg_token_compression']}x, "
          f"qwen_spd={agg['avg_qwen_only_speedup']}x, e2e_spd={agg['avg_e2e_speedup']}x")


if __name__ == "__main__":
    main()
