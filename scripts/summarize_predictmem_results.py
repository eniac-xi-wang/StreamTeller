#!/usr/bin/env python3
"""Summarize PredictMem evaluation results by method with OVO scoring.

Handles OVO-format fields: task, response, ground_truth_letter, options, score.
Also supports legacy format with prediction/ground_truth fields.

Usage:
    python scripts/summarize_predictmem_results.py --input results/prompt_smoke_baseline_5.jsonl
"""

import argparse
import json
from pathlib import Path


def summarize(input_path: str, detail: bool = False):
    results = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        print("No results found.")
        return []

    # Score may already be computed in new format; if not, compute from legacy fields
    for r in results:
        if "score" not in r:
            # Legacy scoring fallback
            pred = r.get("prediction", "")
            gt = r.get("ground_truth", "")
            npred = " ".join((pred or "").lower().strip().split())
            ngt = " ".join((gt or "").lower().strip().split())
            r["score"] = 1 if ngt and npred and ngt in npred else 0

    # Group by method
    by_method = {}
    for r in results:
        m = r.get("method", "unknown")
        by_method.setdefault(m, []).append(r)

    header = (f"{'method':<14} {'n':>4} {'score':>8} {'kept_tok':>9} {'keep_ratio':>10} "
              f"{'score_lat':>9} {'prefill_lat':>10} {'total_lat':>9} {'peak_mem':>9}")
    sep = "-" * len(header)

    print(header)
    print(sep)

    summary_rows = []
    for method in sorted(by_method.keys()):
        rows = by_method[method]
        n = len(rows)
        scores = [r.get("score", 0) for r in rows]
        total_score = sum(scores)
        avg_kept = sum(r.get("kept_video_tokens", 0) for r in rows) / max(n, 1)
        avg_ratio = sum(r.get("keep_ratio_actual", 0) for r in rows) / max(n, 1)
        avg_prefill = sum(r.get("prefill_latency_s", 0) or 0 for r in rows) / max(n, 1)
        avg_total = sum(r.get("total_latency_s", 0) or 0 for r in rows) / max(n, 1)
        avg_peak = sum(r.get("peak_memory_mb", 0) or 0 for r in rows) / max(n, 1)
        avg_score_lat = sum(r.get("score_latency_s", 0) or 0 for r in rows) / max(n, 1)
        avg_cache_lat = sum(r.get("cache_build_latency_s", 0) or 0 for r in rows) / max(n, 1)

        score_str = f"{total_score}/{n}"

        print(f"{method:<14} {n:>4} {score_str:>8} {avg_kept:>9.0f} {avg_ratio:>10.3f} "
              f"{avg_score_lat:>9.3f} {avg_prefill:>10.3f} {avg_total:>9.3f} {avg_peak:>9.0f}")

        row = {
            "method": method,
            "n": n,
            "score": score_str,
            "score_rate": round(total_score / n, 3) if n > 0 else 0.0,
            "avg_kept_tokens": round(avg_kept, 1),
            "avg_keep_ratio": round(avg_ratio, 3),
            "avg_score_latency_s": round(avg_score_lat, 3),
            "avg_cache_build_latency_s": round(avg_cache_lat, 3),
            "avg_prefill_latency_s": round(avg_prefill, 3),
            "avg_decode_latency_s": round(
                sum(r.get("decode_latency_s", 0) or 0 for r in rows) / max(n, 1), 3),
            "avg_total_latency_s": round(avg_total, 3),
            "avg_peak_memory_mb": round(avg_peak, 0),
        }
        summary_rows.append(row)

    # Per-sample details
    if detail:
        print(f"\n{'=' * 80}")
        print("Per-sample details:")
        for r in results:
            s = r.get("score", 0)
            flag = "PASS" if s else "FAIL"
            task = r.get("task", "?")
            resp = r.get("response", r.get("prediction", ""))
            gt = r.get("ground_truth_letter", r.get("ground_truth", ""))
            raw = r.get("raw_response", "")
            print(f"  [{flag}] {r['method']}/{r['sample_id']} ({task}): "
                  f"resp='{resp}' gt='{gt}' raw='{raw[:60]}'")

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--detail", action="store_true",
                        help="Print per-sample scoring details.")
    args = parser.parse_args()

    summary_rows = summarize(args.input, detail=args.detail)

    if args.output and summary_rows:
        with open(args.output, "w") as f:
            json.dump(summary_rows, f, indent=2)
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
