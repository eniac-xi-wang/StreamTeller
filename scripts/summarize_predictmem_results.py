#!/usr/bin/env python3
"""Summarize PredictMem evaluation results by method.

Usage:
    python scripts/summarize_predictmem_results.py --input results/eval_output.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        print("No results found.")
        return

    # Group by method
    by_method = {}
    for r in results:
        m = r.get("method", "unknown")
        by_method.setdefault(m, []).append(r)

    print(f"{'method':<14} {'n':>4} {'score':>8} {'kept_tok':>9} {'keep_ratio':>10} "
          f"{'score_lat':>9} {'prefill_lat':>10} {'total_lat':>9} {'peak_mem':>9}")
    print("-" * 90)

    summary_rows = []
    for method in sorted(by_method.keys()):
        rows = by_method[method]
        n = len(rows)
        avg_kept = sum(r.get("kept_video_tokens", 0) for r in rows) / n
        avg_ratio = sum(r.get("keep_ratio_actual", 0) for r in rows) / n
        avg_prefill = sum(r.get("prefill_latency_s", 0) or 0 for r in rows) / n
        avg_total = sum(r.get("total_latency_s", 0) or 0 for r in rows) / n
        avg_peak = sum(r.get("peak_memory_mb", 0) or 0 for r in rows) / n
        avg_score_lat = sum(r.get("score_latency_s", 0) or 0 for r in rows) / n

        # Score placeholder (simulated)
        score = "N/A"

        print(f"{method:<14} {n:>4} {score:>8} {avg_kept:>9.0f} {avg_ratio:>10.3f} "
              f"{avg_score_lat:>9.3f} {avg_prefill:>10.3f} {avg_total:>9.3f} {avg_peak:>9.0f}")

        summary_rows.append({
            "method": method,
            "n": n,
            "avg_kept_tokens": round(avg_kept, 1),
            "avg_keep_ratio": round(avg_ratio, 3),
            "avg_prefill_latency_s": round(avg_prefill, 3),
            "avg_total_latency_s": round(avg_total, 3),
            "avg_peak_memory_mb": round(avg_peak, 0),
            "avg_score_latency_s": round(avg_score_lat, 3),
        })

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary_rows, f, indent=2)
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
