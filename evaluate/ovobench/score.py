#!/usr/bin/env python3
"""OVO-Bench scoring + summary — merge outputs, compute accuracy, generate reports.

Usage:
    python evaluate/ovobench/score.py --result_dir <path> --run_name <name> [--baseline_result_dir <path>]
"""

import argparse
import json
import os
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REALTIME_TASKS = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]


# ── Scoring logic ──────────────────────────────────────────────────────────

def _score_mc(response: str | None, gt: str) -> int:
    if not response:
        return 0
    return int(str(gt).upper() in response.upper())


def _score_rec(response: str | None, gt) -> int:
    if not response:
        return 0
    nums = re.findall(r"\d+", str(response))
    return 1 if nums and "".join(nums) == str(gt) else 0


def _score_yesno(response: str | None, gt: str) -> int:
    if not response:
        return 0
    return int(gt.lower() in response.lower())


def score_all(results: dict[str, list[dict]]) -> dict:
    """Compute per-task accuracy and category averages."""
    eval_results = {}

    for category, task_list in [
        ("backward", BACKWARD_TASKS),
        ("realtime", REALTIME_TASKS),
        ("forward", FORWARD_TASKS),
    ]:
        eval_results[category] = {"tasks": OrderedDict(), "average": None}
        category_scores = []

        for task_name in task_list:
            if task_name not in results:
                continue
            task_data = results[task_name]
            if not task_data:
                continue

            if task_name in BACKWARD_TASKS + REALTIME_TASKS:
                scores = []
                for item in task_data:
                    scores.append(_score_mc(item.get("response", ""), str(item.get("ground_truth", ""))))
            elif task_name == "REC":
                scores = []
                for item in task_data:
                    for ti in item.get("test_info", []):
                        scores.append(_score_rec(ti.get("response", ""), ti.get("count", 0)))
            elif task_name == "SSR":
                scores = []
                for item in task_data:
                    for ti in item.get("test_info", []):
                        if (ti.get("response", "").upper() == "N" and ti.get("type") == 0) or \
                           (ti.get("response", "").upper() == "Y" and ti.get("type") == 1):
                            scores.append(1)
                        else:
                            gt = "No" if ti.get("type") == 0 else "Yes"
                            scores.append(_score_yesno(ti.get("response", ""), gt))
            elif task_name == "CRR":
                scores = []
                for item in task_data:
                    for ti in item.get("test_info", []):
                        if (ti.get("response", "").upper() == "N" and ti.get("type") == 0) or \
                           (ti.get("response", "").upper() == "Y" and ti.get("type") == 1):
                            scores.append(1)
                        else:
                            gt = "No" if ti.get("type") == 0 else "Yes"
                            scores.append(_score_yesno(ti.get("response", ""), gt))

            if scores:
                acc = 100 * sum(scores) / len(scores)
                eval_results[category]["tasks"][task_name] = round(acc, 2)
                category_scores.append(acc)

        if category_scores:
            eval_results[category]["average"] = round(sum(category_scores) / len(category_scores), 2)

    overall = []
    for cat in ["backward", "realtime", "forward"]:
        if eval_results[cat]["average"] is not None:
            overall.append(eval_results[cat]["average"])
    eval_results["Overall Avg."] = round(sum(overall) / len(overall), 2) if overall else None

    return eval_results


# ── Token / latency stats ──────────────────────────────────────────────────

def compute_predictmem_stats(merged: dict[str, list[dict]]) -> dict[str, Any]:
    """Aggregate PredictMem token/latency statistics across samples."""
    all_stats = []
    for task_name, items in merged.items():
        for item in items:
            pm = item.get("predictmem_stats")
            if pm:
                all_stats.append({
                    "original_video_tokens": pm.get("original_video_tokens", 0),
                    "kept_video_tokens": pm.get("kept_video_tokens", 0),
                    "keep_ratio_actual": pm.get("keep_ratio_actual", 0),
                    "predictmem_scoring_latency_s": pm.get("predictmem_scoring_latency_s", 0),
                    "num_tubelets_scored": pm.get("num_tubelets_scored", 0),
                })
            total_lat = item.get("total_latency_s", 0)
            if total_lat:
                all_stats.append({"total_latency_s": total_lat, "peak_memory_mb": item.get("peak_memory_mb", 0)})

    n = len(all_stats)
    if n == 0:
        return {}

    return {
        "num_samples_with_stats": n,
        "avg_original_video_tokens": round(sum(s.get("original_video_tokens", 0) for s in all_stats) / max(1, sum(1 for s in all_stats if s.get("original_video_tokens"))), 1),
        "avg_kept_video_tokens": round(sum(s.get("kept_video_tokens", 0) for s in all_stats) / max(1, sum(1 for s in all_stats if s.get("kept_video_tokens"))), 1),
        "avg_keep_ratio_actual": round(sum(s.get("keep_ratio_actual", 0) for s in all_stats) / max(1, sum(1 for s in all_stats if s.get("keep_ratio_actual"))), 4),
        "avg_predictmem_scoring_latency_s": round(sum(s.get("predictmem_scoring_latency_s", 0) for s in all_stats) / max(1, sum(1 for s in all_stats if s.get("predictmem_scoring_latency_s"))), 3),
        "avg_total_latency_s": round(sum(s.get("total_latency_s", 0) for s in all_stats) / max(1, sum(1 for s in all_stats if s.get("total_latency_s"))), 3),
        "avg_peak_memory_mb": round(sum(s.get("peak_memory_mb", 0) for s in all_stats) / max(1, sum(1 for s in all_stats if s.get("peak_memory_mb"))), 1),
    }


# ── Summary MD ─────────────────────────────────────────────────────────────

def build_summary_md(scores: dict, pm_stats: dict, run_name: str) -> str:
    lines = [
        f"# OVO-Bench Summary — {run_name}",
        "",
        "## Accuracy",
        "",
        "| Category | Task | Accuracy |",
        "|---|---|---|",
    ]
    mapping = [
        ("Real-Time Visual Perception", "realtime", REALTIME_TASKS),
        ("Backward Tracing", "backward", BACKWARD_TASKS),
        ("Forward Active Responding", "forward", FORWARD_TASKS),
    ]
    for cat_name, cat_key, tasks in mapping:
        if cat_key in scores and scores[cat_key].get("tasks"):
            for t in tasks:
                if t in scores[cat_key]["tasks"]:
                    lines.append(f"| {cat_name} | {t} | {scores[cat_key]['tasks'][t]:.2f}% |")
            avg = scores[cat_key].get("average")
            if avg is not None:
                lines.append(f"| **{cat_name} Avg** | | **{avg:.2f}%** |")

    overall = scores.get("Overall Avg.")
    if overall is not None:
        lines.append(f"| | | |")
        lines.append(f"| **Overall Avg** | | **{overall:.2f}%** |")

    if pm_stats:
        lines.extend([
            "",
            "## PredictMem Statistics",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Avg original video tokens | {pm_stats.get('avg_original_video_tokens', 'N/A')} |",
            f"| Avg kept video tokens | {pm_stats.get('avg_kept_video_tokens', 'N/A')} |",
            f"| Avg keep ratio | {pm_stats.get('avg_keep_ratio_actual', 'N/A')} |",
            f"| Avg scoring latency | {pm_stats.get('avg_predictmem_scoring_latency_s', 'N/A')}s |",
            f"| Avg total latency | {pm_stats.get('avg_total_latency_s', 'N/A')}s |",
            f"| Avg peak memory | {pm_stats.get('avg_peak_memory_mb', 'N/A')}MB |",
        ])

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main(result_dir: str, run_name: str, baseline_result_dir: str | None = None):
    result_dir = Path(result_dir)
    output_dir = result_dir / "outputs"
    results_dir = result_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Merge outputs
    merged: dict[str, list[dict]] = defaultdict(list)
    jsonl_files = sorted(output_dir.glob("*.jsonl")) if output_dir.exists() else []
    for jf in jsonl_files:
        if not jf.name.startswith(run_name):
            continue
        if "token_drop" in jf.name or "token_memory" in jf.name or "predictmem_stats" in jf.name:
            continue
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if not line or "task" not in line:
                    continue
                item = json.loads(line)
                task = item.get("task", "")
                merged[task].append(item)

    # Score
    scores = score_all(merged)

    # Reorder + rename
    new_scores = OrderedDict()
    for cat_name, cat_key, tasks in [
        ("Real-Time Visual Perception", "realtime", REALTIME_TASKS),
        ("Backward Tracing", "backward", BACKWARD_TASKS),
        ("Forward Active Responding", "forward", FORWARD_TASKS),
    ]:
        if cat_key in scores:
            section = OrderedDict()
            for t in tasks:
                if t in scores[cat_key].get("tasks", {}):
                    section[t] = scores[cat_key]["tasks"][t]
            section["Avg"] = scores[cat_key].get("average")
            new_scores[cat_name] = section
    new_scores["Overall"] = {"Avg": scores.get("Overall Avg.")}

    # Write score files
    merged_path = results_dir / "results_merged.json"
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)

    score_path = results_dir / "score_merged.json"
    with open(score_path, "w") as f:
        json.dump(new_scores, f, indent=2)

    # Compute PredictMem stats
    pm_stats = compute_predictmem_stats(merged)

    # If baseline provided, compute speedups
    if baseline_result_dir:
        bl_dir = Path(baseline_result_dir) / "results" / "results_merged.json"
        if bl_dir.exists():
            with open(bl_dir) as f:
                bl_merged = json.load(f)
            pm_stats["_baseline_provided"] = True
            bl_total_lat = []
            for items in bl_merged.values():
                for item in items:
                    if item.get("total_latency_s"):
                        bl_total_lat.append(item["total_latency_s"])
            if bl_total_lat:
                bl_avg_lat = sum(bl_total_lat) / len(bl_total_lat)
                pm_stats["baseline_avg_total_latency_s"] = round(bl_avg_lat, 3)
                if pm_stats.get("avg_total_latency_s"):
                    pm_stats["e2e_speedup"] = round(bl_avg_lat / pm_stats["avg_total_latency_s"], 2)
                if pm_stats.get("avg_kept_video_tokens") and pm_stats.get("avg_original_video_tokens"):
                    pm_stats["token_compression"] = round(pm_stats["avg_original_video_tokens"] / pm_stats["avg_kept_video_tokens"], 2)
                qwen_lat = pm_stats.get("avg_total_latency_s", 0) - pm_stats.get("avg_predictmem_scoring_latency_s", 0)
                if qwen_lat > 0:
                    pm_stats["qwen_only_speedup"] = round(bl_avg_lat / qwen_lat, 2)

    # Summary JSON
    summary = OrderedDict([
        ("run_name", run_name),
        ("accuracy", new_scores),
        ("predictmem_stats", pm_stats),
    ])
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Summary MD
    md_text = build_summary_md(scores, pm_stats, run_name)
    md_path = results_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(md_text)

    print(f"Scores written to: {score_path}")
    print(f"Summary written to: {summary_path}")
    print(f"Markdown written to: {md_path}")

    # Print summary to stdout
    for cat_name, tasks in new_scores.items():
        if cat_name == "Overall":
            print(f"\nOverall Avg: {tasks['Avg']:.2f}%")
        else:
            print(f"\n{cat_name}:")
            for t, acc in tasks.items():
                if t != "Avg":
                    print(f"  {t}: {acc:.2f}%")
            avg = tasks.get("Avg")
            if avg is not None:
                print(f"  Average: {avg:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OVO-Bench scoring")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--baseline_result_dir", default=None)
    args = parser.parse_args()
    main(args.result_dir, args.run_name, args.baseline_result_dir)
