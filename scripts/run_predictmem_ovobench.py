#!/usr/bin/env python3
"""Run a short OVO-Bench comparison with compact PredictMem or the baseline."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _path in (_repo_root, _models_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evaluate.common.qwen35_predictmem import (
    apply_chat_template as apply_chat_template_for_generation,
    build_video_inputs_for_eval,
    generate_with_compact_memory,
    load_qwen35_model,
    load_qwen35_processor,
    resolve_eval_runtime,
)


# ─── OVO Prompt Builder ─────────────────────────────────────────────────────

_MC_TASKS = {"EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"}
_YESNO_TASKS = {"SSR", "CRR"}
_NUMBER_TASKS = {"REC"}


def _option_letter(idx: int) -> str:
    return chr(ord("A") + idx)


def build_ovo_prompt(sample: dict) -> tuple[str, str, str]:
    task = sample["task"]
    question = sample.get("question", "")
    options = sample.get("options", [])
    answer = sample.get("answer", "")
    gt_idx = sample.get("gt")

    if task in _MC_TASKS and options:
        letters = [_option_letter(i) for i in range(len(options))]
        options_str = "; ".join(f"{l}. {opt}" for l, opt in zip(letters, options))
        prompt = (
            f"Question: {question}\n"
            f"Options:\n"
            f"{options_str}\n\n"
            f"Respond only with the letter corresponding to your chosen option "
            f"(e.g., {', '.join(letters[:3])}).\n"
            f"Do not include any additional text or explanation in your response."
        )
        gt_letter = _option_letter(gt_idx) if gt_idx is not None else ""
        return prompt, task, gt_letter

    elif task in _YESNO_TASKS:
        if question:
            prompt = (
                f"Question: {question}\n\n"
                f"Respond only with Yes or No.\n"
                f"Do not include any additional text or explanation in your response."
            )
        else:
            test_info = sample.get("test_info", [])
            if test_info:
                ti = test_info[0]
                step = ti.get("step", "this step")
                prompt = (
                    f"Is the following step being performed at the indicated time?\n"
                    f"Step: {step}\n\n"
                    f"Respond only with Yes or No.\n"
                    f"Do not include any additional text or explanation in your response."
                )
            else:
                prompt = "Respond only with Yes or No."
        gt_letter = ""
        if answer:
            gt_letter = "Yes" if "yes" in answer.lower() else answer
        elif sample.get("test_info"):
            ti_type = sample["test_info"][0].get("type")
            gt_letter = "Yes" if ti_type == 1 else "No"
        return prompt, task, gt_letter

    elif task in _NUMBER_TASKS:
        activity = sample.get("activity", "the action")
        prompt = (
            f"How many times was {activity} performed in the video?\n\n"
            f"Respond only with a number.\n"
            f"Do not include any additional text or explanation in your response."
        )
        gt_letter = ""
        test_info = sample.get("test_info", [])
        if test_info:
            gt_letter = str(test_info[0].get("count", ""))
        return prompt, task, gt_letter

    else:
        if question:
            prompt = f"Question: {question}\n\nAnswer concisely in a few words."
        else:
            prompt = "Describe what is happening in the video."
        gt_letter = answer
        return prompt, task, gt_letter


# ─── Response Parser ─────────────────────────────────────────────────────────

def parse_response(raw_response: str, task: str, num_options: int = 0) -> str:
    cleaned = raw_response.strip()

    if task in _MC_TASKS and num_options > 0:
        letter_match = re.search(r'\b([A-D])\b', cleaned, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
        return ""

    elif task in _YESNO_TASKS:
        yes_match = re.search(r'\b(Yes|Y)\b', cleaned, re.IGNORECASE)
        no_match = re.search(r'\b(No|N)\b', cleaned, re.IGNORECASE)
        if yes_match and not no_match:
            return "Yes"
        elif no_match and not yes_match:
            return "No"
        elif yes_match and no_match:
            return "Yes" if yes_match.start() < no_match.start() else "No"
        return ""

    elif task in _NUMBER_TASKS:
        num_match = re.search(r'\b(\d+)\b', cleaned)
        return num_match.group(1) if num_match else ""

    else:
        return cleaned


def score_parsed_response(parsed: str, ground_truth_letter: str, task: str) -> int:
    if not parsed or not ground_truth_letter:
        return 0

    if task in _MC_TASKS:
        return 1 if parsed.upper() == ground_truth_letter.upper().strip() else 0
    elif task in _YESNO_TASKS:
        p = parsed.strip().lower()
        g = ground_truth_letter.strip().lower()
        return 1 if p == g else 0
    elif task in _NUMBER_TASKS:
        return 1 if parsed.strip() == ground_truth_letter.strip() else 0
    else:
        npred = " ".join(parsed.lower().split())
        ngt = " ".join(ground_truth_letter.lower().split())
        return 1 if ngt and npred and ngt in npred else 0


# ─── Model / Processor Loading ───────────────────────────────────────────────

def move_inputs_to_device(inputs: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def decode_new_tokens(processor, generated_ids: torch.Tensor, input_len: int) -> str:
    output_ids = generated_ids[0, input_len:]
    if hasattr(processor, "decode"):
        return processor.decode(output_ids, skip_special_tokens=True)
    return processor.tokenizer.decode(output_ids, skip_special_tokens=True)


# ─── Sample Loading ──────────────────────────────────────────────────────────

def load_ovo_samples(bench_path: str, max_samples: int, video_base: str) -> list[dict]:
    with open(bench_path) as f:
        data = json.load(f)
    samples = []
    for item in data[:max_samples]:
        samples.append({
            **item,
            "sample_id": str(item["id"]),
            "video_path": str(Path(video_base) / f"{item['id']}.mp4"),
        })
    return samples


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--method", required=True,
                        choices=["baseline", "predictmem"])
    parser.add_argument("--predictmem_runtime", choices=["auto", "none", "compact"], default="auto")
    parser.add_argument("--jepa_checkpoint_path", default="/data/model_weights_public/jepa/jeap_vitl_16_256.pt")
    parser.add_argument("--vjepa_src_path", default=None)
    parser.add_argument("--bench_path", default="evaluate/ovobench/ovo_bench_new.json")
    parser.add_argument("--video_dir", default="/data/qinian_workspace/OVO-Bench/chunked_videos")
    parser.add_argument("--output", default="results/ovo_eval.jsonl")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--predictmem_keep_ratio", type=float, default=0.10)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--qwen_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--stream_mode", default="full",
                        choices=["full", "tail_budget", "uniform_budget", "first_budget"],
                        help="Stream sampling mode.")
    parser.add_argument("--frame_budget", type=int, default=0,
                        help="Max frames for budgeted stream modes (0=no limit).")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--visual_ablation", default="normal",
                        choices=["normal", "black", "shuffle", "text_only"])
    args = parser.parse_args()

    try:
        predictmem_runtime = resolve_eval_runtime(args.method, args.predictmem_runtime)
    except ValueError as exc:
        parser.error(str(exc))
    if args.method == "predictmem" and args.visual_ablation != "normal":
        parser.error("Frame-array ablations require --method baseline")

    model = load_qwen35_model(
        args.model_path, device=args.device, jepa_checkpoint_path=args.jepa_checkpoint_path,
        vjepa_src_path=args.vjepa_src_path, predictmem_keep_ratio=args.predictmem_keep_ratio,
        qwen_size=args.qwen_size, jepa_size=args.qwen_size // 2,
    )
    processor = load_qwen35_processor(args.model_path, args.fps)
    stream_mode = {"tail_budget": "recent", "uniform_budget": "uniform",
                   "first_budget": "full", "full": "full"}[args.stream_mode]
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.1f}B params, video_token_id={model.config.video_token_id}")

    samples = load_ovo_samples(args.bench_path, args.max_samples, args.video_dir)
    print(f"Running {len(samples)} samples, method={args.method}, "
          f"predictmem_runtime={predictmem_runtime}, stream_mode={args.stream_mode}, "
          f"keep_ratio={args.predictmem_keep_ratio}, max_new_tokens={args.max_new_tokens}")

    results = []
    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        task = sample["task"]

        print(f"\n[{i+1}/{len(samples)}] sample_id={sid} task={task} method={args.method}")

        try:
            prompt_text, task_type, gt_letter = build_ovo_prompt(sample)

            if args.method == "predictmem":
                raw_response, stats = generate_with_compact_memory(
                    model, processor, prompt_text, sample["video_path"], fps=args.fps,
                    frame_budget=args.frame_budget, stream_mode=stream_mode, device=args.device,
                    predictmem_keep_ratio=args.predictmem_keep_ratio,
                    max_new_tokens=args.max_new_tokens, disable_thinking=args.disable_thinking,
                )
                pm_stats = stats["predictmem_stats"]
                num_frames_np = stats["num_frames"]
                grid = stats["video_grid_thw"]
                num_qwen_video_tokens = stats["expected_video_tokens"]
                total_latency = stats["total_latency_s"]
                peak_memory = stats["peak_memory_mb"]
            else:
                qwen_frames, _, video_metadata, _ = build_video_inputs_for_eval(
                    sample["video_path"], fps=args.fps, qwen_size=args.qwen_size,
                    jepa_size=args.qwen_size // 2, frame_budget=args.frame_budget, stream_mode=stream_mode,
                )
                num_frames_np = len(qwen_frames)
                # ── Visual ablation ──
                frames_for_model = qwen_frames.copy()
                if args.visual_ablation == "black":
                    frames_for_model = np.zeros_like(frames_for_model)
                elif args.visual_ablation == "shuffle":
                    perm = np.random.permutation(len(frames_for_model))
                    frames_for_model = frames_for_model[perm]

                # ── Build messages ──
                content_blocks = []
                if args.visual_ablation != "text_only":
                    content_blocks.append({"type": "video", "video": frames_for_model, "fps": args.fps})
                content_blocks.append({"type": "text", "text": prompt_text})
                messages = [{"role": "user", "content": content_blocks}]

                text = apply_chat_template_for_generation(processor, messages, args.disable_thinking)

                if args.visual_ablation == "text_only":
                    inputs = processor(text=[text], return_tensors="pt")
                else:
                    inputs = processor(
                        text=[text], videos=[frames_for_model],
                        video_metadata=[video_metadata],
                        do_sample_frames=False, do_resize=False, fps=args.fps,
                        return_tensors="pt",
                    )
                inputs = move_inputs_to_device(inputs, args.device)

                # ── Compute token counts ──
                grid = inputs["video_grid_thw"].tolist() if "video_grid_thw" in inputs else []
                num_qwen_video_tokens = sum(t * (h // 2) * (w // 2) for t, h, w in grid)

                # ── Generate ──
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=False, temperature=None, top_p=None,
                    )

                t1 = time.perf_counter()
                total_latency = t1 - t0
                raw_response = decode_new_tokens(processor, generated_ids, inputs["input_ids"].shape[1])
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if args.device == "cuda" else 0.0

                pm_stats = None

            # Compute derived latency fields
            vjepa_latency = pm_stats.get("predictmem_scoring_latency_s", 0.0) if pm_stats else 0.0
            qwen_only_latency = total_latency - vjepa_latency

            # ── Parse and score ──
            num_options = len(sample.get("options", []))
            parsed_response = parse_response(raw_response, task, num_options)
            score = score_parsed_response(parsed_response, gt_letter, task)

            # ── Build output entry ──
            entry = {
                "sample_id": sid,
                "task": task,
                "video": sample["video_path"],
                "question": sample.get("question", ""),
                "answer_text": sample.get("answer", ""),
                "ground_truth_letter": gt_letter,
                "options": sample.get("options", []),
                "response": parsed_response,
                "raw_response": raw_response,
                "score": score,
                "method": args.method,
                "predictmem_runtime": predictmem_runtime,
                "stream_mode": args.stream_mode,
                "frame_budget": args.frame_budget,
                "fps": args.fps,
                "num_frames": num_frames_np,
                "qwen_size": args.qwen_size,
                "video_grid_thw": grid,
                "expected_video_tokens": num_qwen_video_tokens,
                "keep_ratio_target": args.predictmem_keep_ratio,
                "total_latency_s": round(total_latency, 4),
                "predictmem_scoring_latency_s": round(vjepa_latency, 4),
                "qwen_latency_excluding_vjepa_s": round(qwen_only_latency, 4),
                "peak_memory_mb": round(peak_memory, 1),
                "visual_ablation": args.visual_ablation,
                "disable_thinking": args.disable_thinking,
            }
            if pm_stats is not None:
                entry["predictmem_stats"] = pm_stats

            results.append(entry)
            print(f"  Raw: '{raw_response.strip()}'")
            print(f"  Parsed: '{parsed_response}' | GT: '{gt_letter}' | Score: {score}")
            print(f"  Latency: {total_latency:.3f}s, Peak mem: {peak_memory:.0f}MB")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            prompt_text, task_type, gt_letter = build_ovo_prompt(sample)
            results.append({
                "sample_id": sid, "task": task, "video": sample["video_path"],
                "question": sample.get("question", ""),
                "answer_text": sample.get("answer", ""),
                "ground_truth_letter": gt_letter,
                "options": sample.get("options", []),
                "response": f"[error: {str(e)[:100]}]", "raw_response": "", "score": 0,
                "method": args.method,
                "predictmem_runtime": predictmem_runtime,
                "stream_mode": args.stream_mode,
                "frame_budget": args.frame_budget,
                "visual_ablation": args.visual_ablation,
                "disable_thinking": args.disable_thinking,
            })

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults: {len(results)} entries -> {args.output}")


if __name__ == "__main__":
    main()
