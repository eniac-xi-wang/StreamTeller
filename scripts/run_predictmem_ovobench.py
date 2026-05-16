#!/usr/bin/env python3
"""PredictMem plugin evaluator for OVO-Bench — FluxMem-like in-model path.

This is the MAIN entry point. The plugin path runs V-JEPA scoring inside
Qwen3.5 prefill with expanding + sliding windows. No offline cache, no
ScoreCache, no FramePlan, no precomputed global_scores.

Usage:
    # Baseline (no pruning)
    python scripts/run_predictmem_ovobench.py \
        --model_path /data/model_weights_public/Qwen/Qwen3.5-9B \
        --method baseline --stream_mode full \
        --max_samples 1 --disable_thinking --max_new_tokens 16 \
        --output results/baseline_sample0.jsonl --device cuda

    # Plugin PredictMem
    python scripts/run_predictmem_ovobench.py \
        --model_path /data/model_weights_public/Qwen/Qwen3.5-9B \
        --method predictmem --predictmem_runtime plugin --stream_mode full \
        --predictmem_keep_ratio 0.10 --disable_thinking \
        --max_new_tokens 16 --max_samples 1 \
        --output results/plugin_predictmem_sample0.jsonl --device cuda
"""

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

from predictmem.config import PredictMemConfig
from predictmem.vision_inputs import build_predictmem_video_inputs


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

def load_qwen35_model(model_path: str, device: str):
    from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else device,
    )


def load_qwen35_processor(model_path: str, fps: float):
    from transformers import AutoProcessor, AutoTokenizer
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        from models.qwen3_5.processing_qwen3_vl import Qwen3_5Processor
        from models.qwen3_5.video_processing_qwen3_vl import Qwen3_5VideoProcessor
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            video_processor = Qwen3_5VideoProcessor.from_pretrained(model_path, fps=fps)
        except Exception:
            video_processor = Qwen3_5VideoProcessor(fps=fps)
        processor = Qwen3_5Processor(
            image_processor=None, tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=getattr(tokenizer, "chat_template", None),
        )
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = fps
        processor.video_processor.do_resize = False
    return processor


def move_inputs_to_device(inputs: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def apply_chat_template_for_generation(processor, messages, disable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


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
                        choices=["baseline", "random", "uniform", "predictmem"])
    parser.add_argument("--predictmem_runtime", default=None,
                        choices=["plugin", "legacy_offline", "none"],
                        help="PredictMem runtime mode. Default: plugin for predictmem, none for others.")
    parser.add_argument("--bench_path", default="evaluate/ovobench/ovo_bench_new.json")
    parser.add_argument("--video_dir", default="/data/qinian_workspace/OVO-Bench/chunked_videos")
    parser.add_argument("--output", default="results/plugin_eval.jsonl")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--predictmem_keep_ratio", type=float, default=0.10)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--qwen_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
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

    # Resolve runtime mode
    if args.predictmem_runtime is None:
        if args.method == "predictmem":
            predictmem_runtime = "plugin"
        else:
            predictmem_runtime = "none"
    else:
        predictmem_runtime = args.predictmem_runtime

    config = PredictMemConfig()
    config.keep_ratio = args.predictmem_keep_ratio
    config.fps = args.fps
    config.qwen_size = args.qwen_size
    config.__post_init__()

    # Load model
    print(f"Loading Qwen3.5 from {args.model_path}...")
    model = load_qwen35_model(args.model_path, args.device)
    model.eval()
    processor = load_qwen35_processor(args.model_path, args.fps)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.1f}B params, video_token_id={model.config.video_token_id}")

    samples = load_ovo_samples(args.bench_path, args.max_samples, args.video_dir)
    print(f"Running {len(samples)} samples, method={args.method}, "
          f"predictmem_runtime={predictmem_runtime}, stream_mode={args.stream_mode}, "
          f"keep_ratio={config.keep_ratio}, max_new_tokens={args.max_new_tokens}")

    results = []
    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        task = sample["task"]
        seed = args.seed + i

        print(f"\n[{i+1}/{len(samples)}] sample_id={sid} task={task} method={args.method}")

        try:
            prompt_text, task_type, gt_letter = build_ovo_prompt(sample)

            # ── Plugin path: build_predictmem_video_inputs ──
            predictmem_frames_256 = None
            predictmem_stats = None

            if predictmem_runtime == "plugin" and args.method == "predictmem":
                qwen_frames, predictmem_frames_256, video_metadata = build_predictmem_video_inputs(
                    sample["video_path"],
                    fps=config.fps,
                    qwen_size=config.qwen_size,
                    jepa_size=config.jepa_size,
                    num_frames=args.frame_budget if args.frame_budget > 0 else None,
                )
                num_frames_np = qwen_frames.shape[0]
                print(f"  Plugin video: {num_frames_np} frames, "
                      f"jepa_tensor={list(predictmem_frames_256.shape)}")
            else:
                # Non-predictmem or legacy: just use decord for Qwen frames
                import decord
                decord.bridge.set_bridge("torch")
                vr = decord.VideoReader(sample["video_path"])
                total_frames = len(vr)
                source_fps = float(vr.get_avg_fps() or config.fps)
                duration = total_frames / source_fps if source_fps > 0 else 0.0
                total_1fps = max(1, int(duration * config.fps))

                if args.stream_mode == "full":
                    if args.frame_budget > 0:
                        total_1fps = min(args.frame_budget, total_1fps)
                    times_s = [i / config.fps for i in range(total_1fps)]
                elif args.stream_mode == "tail_budget":
                    budget = args.frame_budget or 64
                    total_1fps = min(budget, total_1fps)
                    start_s = max(0.0, duration - total_1fps / config.fps)
                    times_s = [start_s + i / config.fps for i in range(total_1fps)]
                elif args.stream_mode == "first_budget":
                    budget = args.frame_budget or 64
                    total_1fps = min(budget, total_1fps)
                    times_s = [i / config.fps for i in range(total_1fps)]
                elif args.stream_mode == "uniform_budget":
                    budget = args.frame_budget or 64
                    total_1fps = min(budget, total_1fps)
                    times_s = [i * duration / total_1fps for i in range(total_1fps)]
                else:
                    times_s = [i / config.fps for i in range(total_1fps)]

                source_indices = [
                    min(total_frames - 1, max(0, int(round(t * source_fps))))
                    for t in times_s
                ]
                frames_raw = vr.get_batch(source_indices)
                if hasattr(frames_raw, "asnumpy"):
                    frames_raw = frames_raw.asnumpy()
                elif isinstance(frames_raw, torch.Tensor):
                    frames_raw = frames_raw.numpy()
                qwen_frames = np.asarray(frames_raw, dtype=np.uint8)
                num_frames_np = qwen_frames.shape[0]
                print(f"  Frames: {num_frames_np}")

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
                content_blocks.append({"type": "video", "video": frames_for_model, "fps": config.fps})
            content_blocks.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content_blocks}]

            text = apply_chat_template_for_generation(processor, messages, args.disable_thinking)

            if args.visual_ablation == "text_only":
                inputs = processor(text=[text], return_tensors="pt")
            else:
                video_metadata = {
                    "total_num_frames": num_frames_np,
                    "fps": float(config.fps),
                    "duration": num_frames_np / float(config.fps),
                    "frames_indices": list(range(num_frames_np)),
                    "height": int(frames_for_model.shape[1]),
                    "width": int(frames_for_model.shape[2]),
                    "video_backend": "decord",
                }
                inputs = processor(
                    text=[text], videos=[frames_for_model],
                    video_metadata=[video_metadata],
                    do_sample_frames=False, do_resize=False, fps=config.fps,
                    return_tensors="pt",
                )
            inputs = move_inputs_to_device(inputs, args.device)

            # ── Compute token counts ──
            video_grid_thw = inputs["video_grid_thw"].detach().cpu()
            grid_t = int(video_grid_thw[0, 0].item())
            grid_h = int(video_grid_thw[0, 1].item())
            grid_w = int(video_grid_thw[0, 2].item())
            merge = config.qwen_merge_size
            num_qwen_video_tokens = grid_t * (grid_h // merge) * (grid_w // merge)
            print(f"  video_grid_thw: {video_grid_thw.tolist()} -> {num_qwen_video_tokens} Qwen video tokens")

            # ── Build generate kwargs ──
            generate_kwargs = {}
            if args.method == "predictmem" and predictmem_runtime == "plugin":
                generate_kwargs["use_predictmem"] = True
                generate_kwargs["predictmem_frames_256"] = predictmem_frames_256
                generate_kwargs["predictmem_keep_ratio"] = config.keep_ratio
            elif args.method == "baseline":
                generate_kwargs["use_predictmem"] = False
            else:
                # random / uniform: use legacy keep_indices path
                generate_kwargs["use_predictmem"] = True
                num_video_tokens = num_qwen_video_tokens
                if args.method == "random":
                    torch.manual_seed(seed)
                    n_keep = max(1, int(num_video_tokens * config.keep_ratio))
                    perm = torch.randperm(num_video_tokens)
                    keep = [perm[:n_keep].sort().values]
                elif args.method == "uniform":
                    n_keep = max(1, int(num_video_tokens * config.keep_ratio))
                    step = num_video_tokens / n_keep
                    keep_local = torch.round(torch.arange(n_keep).float() * step).long().clamp(0, num_video_tokens - 1).unique()
                    keep = [keep_local.sort().values]
                generate_kwargs["predictmem_keep_indices"] = keep
                kept = keep[0].shape[0]
                keep_ratio_actual = kept / num_video_tokens
                print(f"  Keep: {kept}/{num_video_tokens} ({keep_ratio_actual:.1%})")

            # ── Generate ──
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=None, top_p=None,
                    **generate_kwargs,
                )

            t1 = time.perf_counter()
            total_latency = t1 - t0
            raw_response = decode_new_tokens(processor, generated_ids, inputs["input_ids"].shape[1])
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if args.device == "cuda" else 0.0

            # Collect PredictMem plugin stats
            pm_stats = getattr(getattr(model, "model", None), "predictmem_last_stats", None)

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
                "fps": config.fps,
                "num_frames": num_frames_np,
                "qwen_size": config.qwen_size,
                "video_grid_thw": video_grid_thw.tolist(),
                "expected_video_tokens": num_qwen_video_tokens,
                "keep_ratio_target": config.keep_ratio,
                "visual_ablation": args.visual_ablation,
                "disable_thinking": args.disable_thinking,
                "total_latency_s": round(total_latency, 4),
                "peak_memory_mb": round(peak_memory, 1),
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
