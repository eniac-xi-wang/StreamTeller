#!/usr/bin/env python3
"""StreamingBench scoring — merge, compute accuracy, generate summary.

Usage:
    python evaluate/streamingbench/score.py --result_dir <path> --model_name <name> [--baseline_result_dir <path>]
"""

import argparse
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


def _extract_answer(response: str) -> str | None:
    if not response:
        return None
    response = response.strip()
    patterns = [
        r"answer is ([A-D])",
        r"^([A-D])[\.:\s]",
        r"\(([A-D])\)",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    if response and response[0] in "ABCD":
        return response[0]
    m = re.search(r"[A-D]", response)
    return m.group(0) if m else None


def merge_outputs(result_dir: Path) -> Path:
    output_dir = result_dir / "output"
    gpu_files = sorted(output_dir.glob("gpu_*.jsonl"))
    if not gpu_files:
        any_files = sorted(result_dir.rglob("*.jsonl"))
        if not any_files:
            raise FileNotFoundError(f"No JSONL files in {result_dir}")
        return any_files[0]
    if len(gpu_files) == 1:
        return gpu_files[0]
    merged = result_dir / "output" / "results.jsonl"
    with open(merged, "w") as fout:
        for jf in gpu_files:
            with open(jf) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
    return merged


def calculate_scores(jsonl_file: Path) -> dict[str, Any]:
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0, "total_latency": [], "peak_memory": []})
    pm_agg: dict[str, list] = defaultdict(list)

    with open(jsonl_file) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task_type = data.get("task_type", "unknown")

            stats["overall"]["total"] += 1
            stats[task_type]["total"] += 1

            if data.get("total_latency_s"):
                stats["overall"]["total_latency"].append(data["total_latency_s"])
                stats[task_type]["total_latency"].append(data["total_latency_s"])
            if data.get("peak_memory_mb"):
                stats["overall"]["peak_memory"].append(data["peak_memory_mb"])
                stats[task_type]["peak_memory"].append(data["peak_memory_mb"])

            # PredictMem stats
            pm = data.get("predictmem_stats")
            if pm:
                pm_agg["original_tokens"].append(pm.get("original_video_tokens", 0))
                pm_agg["kept_tokens"].append(pm.get("kept_video_tokens", 0))
                pm_agg["keep_ratio"].append(pm.get("keep_ratio_actual", 0))
                pm_agg["scoring_latency"].append(pm.get("predictmem_scoring_latency_s", 0))

            # Accuracy
            if data.get("correct") is True:
                stats["overall"]["correct"] += 1
                stats[task_type]["correct"] += 1
            elif "predicted_answer" in data and "answer" in data:
                pred = str(data["predicted_answer"]).strip().lower() if data["predicted_answer"] else ""
                ans = str(data["answer"]).strip().lower()
                if pred == ans:
                    stats["overall"]["correct"] += 1
                    stats[task_type]["correct"] += 1

    results = OrderedDict()
    for task_type in sorted(stats.keys()):
        s = stats[task_type]
        if s["total"] == 0:
            continue
        avg_lat = round(sum(s["total_latency"]) / len(s["total_latency"]), 3) if s["total_latency"] else 0
        avg_mem = round(sum(s["peak_memory"]) / len(s["peak_memory"]), 1) if s["peak_memory"] else 0
        results[task_type] = OrderedDict([
            ("total", s["total"]),
            ("correct", s["correct"]),
            ("accuracy", round(s["correct"] / s["total"] * 100, 2)),
            ("avg_latency_s", avg_lat),
            ("avg_peak_memory_mb", avg_mem),
        ])

    # Aggregate PredictMem stats
    pm_summary = {}
    if pm_agg["original_tokens"]:
        pm_summary = {
            "num_samples": len(pm_agg["original_tokens"]),
            "avg_original_video_tokens": round(sum(pm_agg["original_tokens"]) / len(pm_agg["original_tokens"]), 1),
            "avg_kept_video_tokens": round(sum(pm_agg["kept_tokens"]) / len(pm_agg["kept_tokens"]), 1),
            "avg_keep_ratio": round(sum(pm_agg["keep_ratio"]) / len(pm_agg["keep_ratio"]), 4),
            "avg_scoring_latency_s": round(sum(pm_agg["scoring_latency"]) / len(pm_agg["scoring_latency"]), 3),
        }

    return {"scores": results, "predictmem_stats": pm_summary}


def build_summary_md(scores: dict, model_name: str) -> str:
    lines = [
        f"# StreamingBench Summary — {model_name}",
        "",
        "| Task Type | Total | Correct | Accuracy | Avg Latency | Avg Memory |",
        "|---|---|---|---|---|---|",
    ]
    sc = scores.get("scores", {})
    for task_type, s in sc.items():
        lines.append(
            f"| {task_type} | {s['total']} | {s['correct']} | "
            f"{s['accuracy']:.2f}% | {s['avg_latency_s']}s | {s['avg_peak_memory_mb']}MB |"
        )

    pm = scores.get("predictmem_stats", {})
    if pm:
        lines.extend([
            "",
            "## PredictMem Statistics",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Avg original tokens | {pm.get('avg_original_video_tokens', 'N/A')} |",
            f"| Avg kept tokens | {pm.get('avg_kept_video_tokens', 'N/A')} |",
            f"| Avg keep ratio | {pm.get('avg_keep_ratio', 'N/A')} |",
            f"| Avg scoring latency | {pm.get('avg_scoring_latency_s', 'N/A')}s |",
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="StreamingBench scoring")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--baseline_result_dir", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    results_dir = result_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    jsonl_file = merge_outputs(result_dir)
    print(f"Scoring file: {jsonl_file}")

    scores = calculate_scores(jsonl_file)

    # Write scores
    score_path = results_dir / "scores.json"
    with open(score_path, "w") as f:
        json.dump(scores, f, indent=2)

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"model_name": args.model_name, **scores}, f, indent=2)

    md_text = build_summary_md(scores, args.model_name)
    md_path = results_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(md_text)

    print(f"Scores:    {score_path}")
    print(f"Summary:   {summary_path}")
    print(f"Markdown:  {md_path}")

    # Print
    for task_type, s in scores["scores"].items():
        print(f"  {task_type}: {s['correct']}/{s['total']} = {s['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
