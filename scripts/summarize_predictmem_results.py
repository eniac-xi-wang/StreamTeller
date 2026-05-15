#!/usr/bin/env python3
"""Summarize PredictMem evaluation results by method with answer scoring.

Scoring: score = 1 if normalized_ground_truth appears in normalized_prediction, else 0.

Usage:
    python scripts/summarize_predictmem_results.py --input results/real_eval.jsonl
"""

import argparse
import json
from pathlib import Path


def normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    if s is None:
        return ""
    return " ".join(s.lower().strip().split())


def compute_score(prediction: str, ground_truth: str) -> int:
    """OVO answer scoring: 1 if normalized ground_truth is a substring of normalized prediction."""
    npred = normalize(prediction)
    ngt = normalize(ground_truth)
    if not ngt or not npred:
        return 0
    return 1 if ngt in npred else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--detail", action="store_true",
                        help="Print per-sample scoring details.")
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

    # Score each result
    for r in results:
        r["score"] = compute_score(r.get("prediction", ""), r.get("ground_truth", ""))

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
        avg_kept = sum(r.get("kept_video_tokens", 0) for r in rows) / n
        avg_ratio = sum(r.get("keep_ratio_actual", 0) for r in rows) / n
        avg_prefill = sum(r.get("prefill_latency_s", 0) or 0 for r in rows) / n
        avg_total = sum(r.get("total_latency_s", 0) or 0 for r in rows) / n
        avg_peak = sum(r.get("peak_memory_mb", 0) or 0 for r in rows) / n
        avg_score_lat = sum(r.get("score_latency_s", 0) or 0 for r in rows) / n

        score_str = f"{total_score}/{n}"

        print(f"{method:<14} {n:>4} {score_str:>8} {avg_kept:>9.0f} {avg_ratio:>10.3f} "
              f"{avg_score_lat:>9.3f} {avg_prefill:>10.3f} {avg_total:>9.3f} {avg_peak:>9.0f}")

        summary_rows.append({
            "method": method,
            "n": n,
            "score": f"{total_score}/{n}",
            "score_rate": round(total_score / n, 3) if n > 0 else 0.0,
            "avg_kept_tokens": round(avg_kept, 1),
            "avg_keep_ratio": round(avg_ratio, 3),
            "avg_score_latency_s": round(avg_score_lat, 3),
            "avg_prefill_latency_s": round(avg_prefill, 3),
            "avg_decode_latency_s": round(sum(r.get("decode_latency_s", 0) or 0 for r in rows) / n, 3),
            "avg_total_latency_s": round(avg_total, 3),
            "avg_peak_memory_mb": round(avg_peak, 0),
        })

    # Per-sample details
    if args.detail:
        print(f"\n{'=' * 80}")
        print("Per-sample details:")
        for r in results:
            score = r.get("score", 0)
            flag = "PASS" if score else "FAIL"
            print(f"  [{flag}] {r['method']}/{r['sample_id']}: "
                  f"pred='{r.get('prediction','')[:80]}' | gt='{r.get('ground_truth','')[:60]}'")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary_rows, f, indent=2)
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
