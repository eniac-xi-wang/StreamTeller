#!/usr/bin/env python3
"""StreamingBench evaluation for Qwen3.5 + PredictMem.

Reads a CSV of questions with timestamps, samples video clips up to each
timestamp, and runs QA.

Reference: site-packages/FluxMem/evaluation/streamingbench/streamingbench.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

_eval_dir = Path(__file__).parent.parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

from common.qwen35_predictmem import (
    load_qwen35_model,
    load_qwen35_processor,
    build_video_inputs_for_eval,
    generate_qwen35_response,
)

PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with some frames from a video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to the question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}

Options:
{options}

The best option is:"""

PROMPT_NO_OPTIONS = """You are an advanced video question-answering AI assistant. You have been provided with a video and a question related to the video. Your task is to carefully analyze the video and provide the answer to the question.

Question: {question}

Answer:"""


def _time_to_seconds(time_str: str) -> float:
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Invalid time format: {time_str}")


def _extract_answer(response: str) -> str:
    response = response.strip()
    patterns = [
        r"option\s*([A-D])",
        r"([A-D])\s*is\s*the\s*best",
        r"answer\s*is\s*([A-D])",
        r"([A-D])\)",
        r"^([A-D])$",
        r"option is\s*([A-D])",
        r"\(([A-D])\)",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(r"[A-D]", response)
    return m.group(0) if m else response


def _format_prompt(question: str, options_str) -> str:
    has_opts = options_str and str(options_str) != "nan" and not pd.isna(options_str)
    if not has_opts:
        return PROMPT_NO_OPTIONS.format(question=question)
    try:
        opts = eval(str(options_str))
    except Exception:
        opts = [o.strip() for o in str(options_str).split("\n") if o.strip()]
    formatted = []
    for i, opt in enumerate(opts):
        letter = chr(65 + i)
        if opt.startswith(f"{letter}.") or opt.startswith(f"{letter})"):
            formatted.append(opt)
        else:
            formatted.append(f"{letter}. {opt}")
    return PROMPT_TEMPLATE.format(question=question, options="\n".join(formatted))


def resolve_video_path(video_dir: str, sample_id: str) -> str:
    return str(Path(video_dir) / f"sample_{sample_id}" / "video.mp4")


def _split_dataframe(df: pd.DataFrame, num_gpus: int) -> list[pd.DataFrame]:
    if num_gpus <= 1:
        return [df]
    chunk = len(df) // num_gpus
    splits = []
    for i in range(num_gpus):
        start = i * chunk
        end = start + chunk if i < num_gpus - 1 else len(df)
        splits.append(df.iloc[start:end])
    return splits


# ── Single-GPU run ─────────────────────────────────────────────────────────

def run_single(args):
    ts = time.strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.result_dir) if args.result_dir else Path(f"eval_results/streamingbench/{args.run_name}_{ts}")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "output").mkdir(exist_ok=True)
    (result_dir / "log").mkdir(exist_ok=True)

    output_jsonl = args.output_jsonl or str(result_dir / "output" / f"results_{ts}.jsonl")
    log_path = args.log_path or str(result_dir / "log" / f"eval_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"StreamingBench: run_name={args.run_name}, method={args.method}")
    logger.info(f"model={args.model_path}, task_csv={args.task_csv}, video_dir={args.video_dir}")

    model = load_qwen35_model(
        args.model_path,
        device=getattr(args, "device", "cuda"),
        torch_dtype=getattr(args, "torch_dtype", "bfloat16"),
        jepa_checkpoint_path=getattr(args, "jepa_checkpoint_path", None),
        vjepa_src_path=getattr(args, "vjepa_src_path", None),
        predictmem_keep_ratio=args.predictmem_keep_ratio,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        tail_keep_frames=args.tail_keep_frames,
        drop_bootstrap=args.drop_bootstrap,
    )
    processor = load_qwen35_processor(args.model_path, fps=args.fps)
    logger.info(f"Model loaded")

    df = pd.read_csv(args.task_csv)
    logger.info(f"Loaded {len(df)} questions")

    with open(output_jsonl, "w") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{args.run_name}]"):
            question_id = row.question_id
            sample_id = question_id.split("_")[-2]
            video_path = resolve_video_path(args.video_dir, sample_id)
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue

            timestamp_sec = _time_to_seconds(row.time_stamp)

            # Compute start time
            start_time = 0.0
            if args.time_window_size and args.time_window_size > 0:
                start_time = max(0.0, timestamp_sec - float(args.time_window_size))

            prompt = _format_prompt(row.question, row.options)
            has_options = str(row.options) != "nan" and not pd.isna(row.options)

            try:
                # Build video inputs with time-window clipping
                qwen_frames, jepa_tensor, meta = build_video_inputs_for_eval(
                    video_path, fps=args.fps, qwen_size=args.qwen_size,
                    jepa_size=args.jepa_size,
                    frame_budget=args.max_num_frames,
                    start_time=start_time,
                    end_time=timestamp_sec,
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

                predicted = _extract_answer(response) if has_options else response.strip()
                correct = predicted == str(row.answer) if has_options else None

                result = {
                    "question_id": question_id,
                    "task_type": row.task_type,
                    "question": row.question,
                    "answer": str(row.answer),
                    "predicted_answer": predicted,
                    "response": response.strip(),
                    "has_options": has_options,
                    "correct": correct,
                    "method": args.method,
                    "predictmem_runtime": args.predictmem_runtime,
                    "timestamp_sec": timestamp_sec,
                    "video_start": start_time,
                    "video_end": timestamp_sec,
                    "fps": args.fps,
                    "num_frames": stats["num_frames"],
                    "total_latency_s": stats["total_latency_s"],
                    "peak_memory_mb": stats["peak_memory_mb"],
                    "video_grid_thw": stats["video_grid_thw"],
                    "expected_video_tokens": stats["expected_video_tokens"],
                    "predictmem_stats": stats.get("predictmem_stats"),
                }
                fout.write(json.dumps(result) + "\n")
                fout.flush()

            except Exception as e:
                logger.error(f"Error on {question_id}: {e}")

    logger.info(f"Done. Output: {output_jsonl}")


# ── Multi-GPU run ──────────────────────────────────────────────────────────

def run_multi_gpu(args):
    ts = time.strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.result_dir) if args.result_dir else Path(f"eval_results/streamingbench/{args.run_name}_{ts}")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "output").mkdir(exist_ok=True)
    (result_dir / "log").mkdir(exist_ok=True)

    df = pd.read_csv(args.task_csv)
    splits = _split_dataframe(df, args.num_gpus)
    temp_dir = result_dir / "tmp"
    temp_dir.mkdir(exist_ok=True)

    def launch_worker(gpu_id, split_df):
        if len(split_df) == 0:
            return True
        split_csv = temp_dir / f"split_{gpu_id}.csv"
        split_df.to_csv(split_csv, index=False)
        out_jsonl = result_dir / "output" / f"gpu_{gpu_id}.jsonl"
        log_path = result_dir / "log" / f"gpu_{gpu_id}.log"

        cmd = [
            sys.executable, "-m", "evaluate.streamingbench.streamingbench",
            "--model_path", args.model_path,
            "--task_csv", str(split_csv),
            "--video_dir", args.video_dir,
            "--run_name", args.run_name,
            "--output_jsonl", str(out_jsonl),
            "--log_path", str(log_path),
            "--method", args.method,
            "--predictmem_runtime", args.predictmem_runtime,
            "--predictmem_keep_ratio", str(args.predictmem_keep_ratio),
            "--fps", str(args.fps),
            "--max_num_frames", str(args.max_num_frames),
            "--max_new_tokens", str(args.max_new_tokens),
            "--window_frames", str(args.window_frames),
            "--stride_frames", str(args.stride_frames),
            "--tail_keep_frames", str(args.tail_keep_frames),
            "--max_pixels", str(args.max_pixels),
            "--device", getattr(args, "device", "cuda"),
            "--torch_dtype", getattr(args, "torch_dtype", "bfloat16"),
            "--worker",
        ]
        if args.drop_bootstrap:
            cmd.append("--drop_bootstrap")
        else:
            cmd.append("--no_drop_bootstrap")
        if getattr(args, "jepa_checkpoint_path", None):
            cmd += ["--jepa_checkpoint_path", args.jepa_checkpoint_path]
        if getattr(args, "vjepa_src_path", None):
            cmd += ["--vjepa_src_path", args.vjepa_src_path]
        if args.time_window_size is not None:
            cmd += ["--time_window_size", str(args.time_window_size)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] {len(split_df)} samples")
        res = subprocess.run(cmd, env=env)
        return res.returncode == 0

    if args.dry_run:
        for gid, split in enumerate(splits):
            print(f"GPU {gid}: {len(split)} samples")
        return

    print(f"Launching {args.num_gpus} GPU workers...")
    with ThreadPoolExecutor(max_workers=args.num_gpus) as pool:
        results = list(pool.map(lambda p: launch_worker(*p), enumerate(splits)))

    if not all(results):
        print("WARNING: Some workers failed")

    # Merge
    merged = result_dir / "output" / "results.jsonl"
    with open(merged, "w") as fout:
        for jf in sorted((result_dir / "output").glob("gpu_*.jsonl")):
            with open(jf) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
    print(f"Merged: {merged}")
    print(f"Done. Results in: {result_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="StreamingBench evaluation for Qwen3.5 + PredictMem")
    # Paths
    p.add_argument("--model_path", default="/data/model_weights_public/Qwen/Qwen3.5-9B")
    p.add_argument("--task_csv", default=None)
    p.add_argument("--video_dir", default=None)
    p.add_argument("--run_name", default="streamingbench_run")
    p.add_argument("--result_dir", default=None)
    p.add_argument("--jepa_checkpoint_path", default=None, help="V-JEPA checkpoint path")
    p.add_argument("--vjepa_src_path", default=None, help="V-JEPA source code path")
    # Method
    p.add_argument("--method", choices=["baseline", "predictmem"], default="baseline")
    p.add_argument("--predictmem_runtime", choices=["plugin", "none"], default="none")
    p.add_argument("--predictmem_keep_ratio", type=float, default=0.10)
    # PredictMem / V-JEPA params
    p.add_argument("--window_frames", type=int, default=16)
    p.add_argument("--stride_frames", type=int, default=2)
    p.add_argument("--tail_keep_frames", type=int, default=4)
    p.add_argument("--drop_bootstrap", action="store_true", default=True)
    p.add_argument("--no_drop_bootstrap", dest="drop_bootstrap", action="store_false")
    # Video sampling
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--qwen_size", type=int, default=512)
    p.add_argument("--jepa_size", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_num_frames", type=int, default=256)
    p.add_argument("--max_pixels", type=int, default=200704)  # 256*28*28
    p.add_argument("--time_window_size", type=float, default=None)
    # Generation
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", default="bfloat16")
    p.add_argument("--disable_thinking", action="store_true", default=True)
    # Multi-GPU
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--log_path", default=None)
    p.add_argument("--output_jsonl", default=None)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.multi_gpu and not args.worker:
        run_multi_gpu(args)
    else:
        run_single(args)
