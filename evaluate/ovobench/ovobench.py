#!/usr/bin/env python3
"""OVO-Bench evaluation for Qwen3.5 + PredictMem.

Supports single-GPU and multi-GPU execution. Backward / Realtime / Forward
task types, baseline and PredictMem methods.

Reference: site-packages/FluxMem/evaluation/ovobench/ovobench.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

# Ensure evaluate/ is importable for common helpers
_eval_dir = Path(__file__).parent.parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

from common.qwen35_predictmem import (
    load_qwen35_model,
    load_qwen35_processor,
    build_video_inputs_for_eval,
    generate_qwen35_response,
)

# ─── Task classification ───────────────────────────────────────────────────

BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REALTIME_TASKS = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]
ALL_TASKS = BACKWARD_TASKS + REALTIME_TASKS + FORWARD_TASKS

# ─── Prompt builder ────────────────────────────────────────────────────────

def build_prompt(task: str, question: str, options: list, anno: dict | None = None, index: int | None = None) -> str:
    if task in BACKWARD_TASKS + REALTIME_TASKS:
        letters = [chr(65 + i) for i in range(len(options))]
        formatted = "; ".join(f"{l}. {opt}" for l, opt in zip(letters, options)) + ";"
        return (
            f"Question: {question}\n"
            f"Options:\n{formatted}\n\n"
            f"Respond only with the letter corresponding to your chosen option "
            f"(e.g., A, B, C).\n"
            f"Do not include any additional text or explanation in your response."
        )
    elif task == "REC":
        activity = anno.get("activity", "the action") if anno else "the action"
        return (
            f"How many times was {activity} performed in the video?\n\n"
            f"Respond only with a number.\n"
            f"Do not include any additional text or explanation in your response."
        )
    elif task == "SSR":
        step = anno["test_info"][index]["step"] if anno else "this step"
        return (
            f"Is the following step being performed at the indicated time?\n"
            f"Step: {step}\n\n"
            f"Respond only with Yes or No.\n"
            f"Do not include any additional text or explanation in your response."
        )
    elif task == "CRR":
        q = anno.get("question", "") if anno else ""
        return (
            f"Question: {q}\n\n"
            f"Respond only with Yes or No.\n"
            f"Do not include any additional text or explanation in your response."
        )
    return f"Question: {question}\n\nAnswer concisely."


# ─── Video path resolution ─────────────────────────────────────────────────

def resolve_video_path(video_dir: str, item_id: int, index: int | None = None) -> str:
    """Resolve video path for an OVO sample.

    If video_dir points to a directory named 'chunked_videos' or contains
    .mp4 files directly, use it as-is. Otherwise append 'chunked_videos'.
    """
    vd = Path(video_dir)
    if vd.name == "chunked_videos" or any(vd.glob("*.mp4")):
        base = vd
    else:
        base = vd / "chunked_videos"

    if index is not None:
        path = base / f"{item_id}_{index}.mp4"
    else:
        path = base / f"{item_id}.mp4"
    return str(path)


# ─── Sample distribution (multi-GPU) ───────────────────────────────────────

def distribute_samples(samples: list[dict], num_gpus: int) -> list[list[dict]]:
    """Distribute samples across GPUs balancing by sample count."""
    items = []
    for s in samples:
        task = s["task"]
        if task in FORWARD_TASKS:
            count = len(s.get("test_info", []))
        else:
            count = 1
        items.append({"sample": s, "count": count})

    items.sort(key=lambda x: x["count"], reverse=True)
    gpu_items = [[] for _ in range(num_gpus)]
    gpu_counts = [0] * num_gpus
    for item in items:
        idx = gpu_counts.index(min(gpu_counts))
        gpu_items[idx].append(item["sample"])
        gpu_counts[idx] += item["count"]

    logger = logging.getLogger(__name__)
    logger.info("Sample distribution across GPUs:")
    for i, (samples_gpu, cnt) in enumerate(zip(gpu_items, gpu_counts)):
        task_counts = defaultdict(int)
        for s in samples_gpu:
            task_counts[s["task"]] += 1
        task_str = ", ".join(f"{t}:{n}" for t, n in sorted(task_counts.items()))
        logger.info(f"  GPU {i}: {len(samples_gpu)} IDs, {cnt} samples — {task_str}")
    return gpu_items


# ─── Main run (single GPU) ─────────────────────────────────────────────────

def run_single(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name
    result_dir = Path(args.result_dir) if args.result_dir else Path(f"eval_results/ovobench/{run_name}_{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "outputs").mkdir(exist_ok=True)
    (result_dir / "log").mkdir(exist_ok=True)

    log_path = args.log_path or str(result_dir / "log" / f"{run_name}.log")
    output_jsonl = args.output_jsonl or str(result_dir / "outputs" / f"{run_name}.jsonl")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"OVO-Bench: run_name={run_name}, method={args.method}")
    logger.info(f"model_path={args.model_path}, result_dir={result_dir}")

    # Load model and processor
    model = load_qwen35_model(args.model_path, "cuda")
    processor = load_qwen35_processor(args.model_path, fps=args.fps)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    # Load task data
    with open(args.task_json) as f:
        all_tasks = json.load(f)

    tasks = [t for t in all_tasks if t["task"] in args.task]
    if args.sample_ids:
        ids_set = set(int(s) for s in args.sample_ids)
        tasks = [t for t in tasks if t["id"] in ids_set]
    if args.max_samples:
        tasks = tasks[: args.max_samples]

    logger.info(f"Processing {len(tasks)} samples")

    for item in tqdm(tasks, desc=f"[{run_name}]"):
        item_id = item["id"]
        task = item["task"]

        try:
            if task in BACKWARD_TASKS + REALTIME_TASKS:
                prompt = build_prompt(task, item["question"], item.get("options", []))
                video_path = resolve_video_path(args.video_dir, item_id)
                if not Path(video_path).exists():
                    logger.warning(f"Video not found: {video_path}")
                    continue

                qwen_frames, jepa_tensor, meta = build_video_inputs_for_eval(
                    video_path, fps=args.fps, qwen_size=args.qwen_size,
                    jepa_size=args.jepa_size, frame_budget=args.frame_budget,
                )

                response, stats = generate_qwen35_response(
                    model, processor, prompt,
                    qwen_frames=qwen_frames, video_metadata=meta,
                    method=args.method,
                    predictmem_runtime=args.predictmem_runtime,
                    predictmem_frames_256=jepa_tensor if args.method == "predictmem" else None,
                    predictmem_keep_ratio=args.predictmem_keep_ratio,
                    fps=args.fps, max_new_tokens=args.max_new_tokens,
                    disable_thinking=True,
                )

                entry = {
                    "id": item_id,
                    "video": str(video_path),
                    "task": task,
                    "question": item["question"],
                    "response": response.strip(),
                    "ground_truth": chr(65 + item["gt"]) if isinstance(item.get("gt"), int) else item.get("gt", ""),
                    "method": args.method,
                    "predictmem_runtime": args.predictmem_runtime,
                    "fps": args.fps,
                    "num_frames": stats["num_frames"],
                    "total_latency_s": stats["total_latency_s"],
                    "peak_memory_mb": stats["peak_memory_mb"],
                    "video_grid_thw": stats["video_grid_thw"],
                    "expected_video_tokens": stats["expected_video_tokens"],
                    "predictmem_stats": stats.get("predictmem_stats"),
                }

            elif task in FORWARD_TASKS:
                entry = dict(item)
                for i, ti in enumerate(item.get("test_info", [])):
                    prompt = build_prompt(task, "", [], anno=item, index=i)
                    video_path = resolve_video_path(args.video_dir, item_id, index=i)
                    if not Path(video_path).exists():
                        logger.warning(f"Video not found: {video_path}")
                        continue

                    qwen_frames, jepa_tensor, meta = build_video_inputs_for_eval(
                        video_path, fps=args.fps, qwen_size=args.qwen_size,
                        jepa_size=args.jepa_size, frame_budget=args.frame_budget,
                    )

                    response, stats = generate_qwen35_response(
                        model, processor, prompt,
                        qwen_frames=qwen_frames, video_metadata=meta,
                        method=args.method,
                        predictmem_runtime=args.predictmem_runtime,
                        predictmem_frames_256=jepa_tensor if args.method == "predictmem" else None,
                        predictmem_keep_ratio=args.predictmem_keep_ratio,
                        fps=args.fps, max_new_tokens=args.max_new_tokens,
                        disable_thinking=True,
                    )

                    entry["test_info"][i]["response"] = response.strip()
                    entry["test_info"][i]["latency_s"] = stats["total_latency_s"]

                entry["method"] = args.method
                entry["predictmem_runtime"] = args.predictmem_runtime
            else:
                continue

            with open(output_jsonl, "a" if Path(output_jsonl).exists() else "w") as f:
                f.write(json.dumps(entry) + "\n")

        except Exception as e:
            logger.error(f"Error on sample {item_id} task={task}: {e}")

    logger.info(f"Done. Output: {output_jsonl}")


# ─── Multi-GPU run ─────────────────────────────────────────────────────────

def run_multi_gpu(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.result_dir) if args.result_dir else Path(f"eval_results/ovobench/{args.run_name}_{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "outputs").mkdir(exist_ok=True)
    (result_dir / "log").mkdir(exist_ok=True)

    with open(args.task_json) as f:
        all_tasks = json.load(f)
    tasks = [t for t in all_tasks if t["task"] in args.task]

    gpu_assignments = distribute_samples(tasks, args.num_gpus)

    if args.dry_run:
        for gpu_id, samples in enumerate(gpu_assignments):
            if samples:
                ids = [s["id"] for s in samples]
                print(f"GPU {gpu_id}: {len(samples)} IDs, sample IDs: {ids[:5]}..." if len(ids) > 5 else f"GPU {gpu_id}: {len(samples)} IDs, sample IDs: {ids}")
        return

    def launch_worker(gpu_id, gpu_samples):
        if not gpu_samples:
            return True
        ids = sorted({str(s["id"]) for s in gpu_samples})
        gpu_tasks = sorted({s["task"] for s in gpu_samples})

        log_path = result_dir / "log" / f"{args.run_name}_gpu{gpu_id}.log"
        output_jsonl = result_dir / "outputs" / f"{args.run_name}_gpu{gpu_id}.jsonl"

        cmd = [
            sys.executable, "-m", "evaluate.ovobench.ovobench",
            "--run_name", args.run_name,
            "--task", *gpu_tasks,
            "--sample_ids", *ids,
            "--model_path", args.model_path,
            "--task_json", args.task_json,
            "--video_dir", args.video_dir,
            "--fps", str(args.fps),
            "--qwen_size", str(args.qwen_size),
            "--jepa_size", str(args.jepa_size),
            "--result_dir", str(result_dir),
            "--log_path", str(log_path),
            "--output_jsonl", str(output_jsonl),
            "--method", args.method,
            "--predictmem_runtime", args.predictmem_runtime,
            "--predictmem_keep_ratio", str(args.predictmem_keep_ratio),
            "--max_new_tokens", str(args.max_new_tokens),
        ]
        if args.frame_budget:
            cmd += ["--frame_budget", str(args.frame_budget)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] Starting {len(ids)} IDs, tasks: {gpu_tasks}")
        res = subprocess.run(cmd, env=env)
        return res.returncode == 0

    print(f"Starting {args.num_gpus} GPU workers...")
    with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = [executor.submit(launch_worker, gpu_id, samples)
                   for gpu_id, samples in enumerate(gpu_assignments)]
        results = [f.result() for f in futures]

    if not all(results):
        print("WARNING: Some GPU workers failed")

    # Score
    score_cmd = [
        sys.executable, "evaluate/ovobench/score.py",
        "--result_dir", str(result_dir),
        "--run_name", args.run_name,
    ]
    subprocess.run(score_cmd)
    print(f"Done. Results in: {result_dir}")


# ─── CLI ───────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="OVO-Bench evaluation for Qwen3.5 + PredictMem")
    p.add_argument("--run_name", default="ovobench_run")
    p.add_argument("--model_path", default="/data/model_weights_public/Qwen/Qwen3.5-9B")
    p.add_argument("--task_json", default="evaluate/ovobench/ovo_bench_new.json")
    p.add_argument("--video_dir", default="/data/qinian_workspace/OVO-Bench")
    p.add_argument("--result_dir", default=None)
    p.add_argument("--method", choices=["baseline", "predictmem"], default="baseline")
    p.add_argument("--predictmem_runtime", choices=["plugin", "none"], default="none")
    p.add_argument("--predictmem_keep_ratio", type=float, default=0.10)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--qwen_size", type=int, default=512)
    p.add_argument("--jepa_size", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--frame_budget", type=int, default=0)
    p.add_argument("--task", nargs="+", choices=ALL_TASKS, default=ALL_TASKS)
    p.add_argument("--sample_ids", nargs="+", default=None)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--time_window_size", type=float, default=None)
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--log_path", default=None)
    p.add_argument("--output_jsonl", default=None)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.multi_gpu:
        run_multi_gpu(args)
    else:
        run_single(args)
