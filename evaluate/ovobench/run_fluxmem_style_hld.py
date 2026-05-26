#!/usr/bin/env python3
"""FluxMem-style frame sampling + Qwen3.5 baseline — HLD evaluation.

Uses FluxMem's smart_nframes (cap 256, linspace sampling, smart_resize
with 200K pixel budget) but runs Qwen3.5 for inference.  Strips out
all FluxMem/streaming memory logic — pure frame-count + resize comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Ensure project modules are importable ────────────────────────────────
_repo = Path(__file__).parent.parent.parent
for _d in [_repo, _repo / "models", _repo / "evaluate"]:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from evaluate.common.qwen35_predictmem import apply_chat_template
from evaluate.ovobench.ovobench import (
    resolve_video_path,
    build_prompt,
    BACKWARD_TASKS,
    REALTIME_TASKS,
    FORWARD_TASKS,
)

# ── FluxMem-style frame sampling (model-agnostic) ────────────────────────

def fluxmem_style_sample(video_path: str, fps: float = 1.0,
                         min_frames: int = 4, max_frames: int = 256,
                         qwen_factor: int = 32):
    """Decode a video the way FluxMem does, but return Qwen3.5-compatible frames.

    Uses ``smart_nframes`` logic (duration × fps, clamped, floored to even)
    and linspace frame indices.  Resize is done with Qwen3.5's factor=32.
    """
    import decord
    from models.predictmem.resize_utils import smart_resize_keep_aspect

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)
    source_fps = float(vr.get_avg_fps() or fps)
    duration = total_frames / source_fps if source_fps > 0 else 0.0

    # ── smart_nframes (FluxMem logic) ──
    nframes = int(duration * fps)
    nframes = max(min_frames, min(nframes, max_frames))
    # floor to even (Qwen requirement)
    nframes = (nframes // 2) * 2
    nframes = max(2, min(nframes, total_frames))

    # ── linspace indices (FluxMem logic) ──
    indices = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    # ── decode + resize ──
    frames_raw = vr.get_batch(indices)
    if hasattr(frames_raw, "asnumpy"):
        frames_raw = torch.from_numpy(frames_raw.asnumpy())
    elif not isinstance(frames_raw, torch.Tensor):
        frames_raw = torch.from_numpy(np.asarray(frames_raw))
    frames_raw = frames_raw.to(dtype=torch.uint8)  # [N, H, W, 3]

    source_h, source_w = int(frames_raw.shape[1]), int(frames_raw.shape[2])

    # smart_resize_keep_aspect with Qwen3.5 factor=32, budget=512²
    qwen_pixels = 512 * 512
    qwen_h, qwen_w = smart_resize_keep_aspect(
        source_h, source_w, factor=qwen_factor,
        min_pixels=qwen_pixels, max_pixels=qwen_pixels,
    )

    frames_chw = frames_raw.permute(0, 3, 1, 2).float()
    if frames_chw.shape[-2:] != (qwen_h, qwen_w):
        qwen_chw = F.interpolate(frames_chw, size=(qwen_h, qwen_w),
                                  mode="bilinear", align_corners=False)
    else:
        qwen_chw = frames_chw
    qwen_chw = qwen_chw.clamp(0, 255)
    qwen_uint8 = qwen_chw.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()

    meta = {
        "total_num_frames": nframes,
        "fps": float(fps),
        "duration": duration,
        "frames_indices": indices,
        "height": qwen_h,
        "width": qwen_w,
        "video_backend": "decord",
    }

    return qwen_uint8, meta, nframes


# ── Model inference ──────────────────────────────────────────────────────

@torch.no_grad()
def run_one_sample(model, processor, prompt: str, frames: np.ndarray,
                   meta: dict, fps: float, max_new_tokens: int = 16):
    content_blocks = [
        {"type": "video", "video": frames, "fps": fps},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content_blocks}]
    text = apply_chat_template(processor, messages, disable_thinking=True)

    inputs = processor(
        text=[text],
        videos=[frames],
        video_metadata=[meta],
        do_sample_frames=False,
        do_resize=False,
        fps=fps,
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    t1 = time.perf_counter()

    output_ids = generated_ids[0, input_len:]
    response = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
    return response.strip(), t1 - t0


# ── Scoring ──────────────────────────────────────────────────────────────

def score_hld(response: str | None, gt_letter: str) -> int:
    if not response:
        return 0
    response_upper = response.strip().upper()
    gt_upper = gt_letter.strip().upper()
    # match first A-D letter in response
    import re
    m = re.search(r"\b([A-D])\b", response_upper)
    if m:
        return 1 if m.group(1) == gt_upper else 0
    return 1 if gt_upper in response_upper else 0


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/model_weights_public/Qwen/Qwen3.5-9B")
    parser.add_argument("--task_json", default="evaluate/ovobench/ovo_bench_new.json")
    parser.add_argument("--video_dir", default="/data/qinian_workspace/OVO-Bench")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_ids", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load model
    print("Loading Qwen3.5-9B...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    from transformers import AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        from models.qwen3_5.processing_qwen3_vl import Qwen3_5Processor
        from models.qwen3_5.video_processing_qwen3_vl import Qwen3_5VideoProcessor
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        video_processor = Qwen3_5VideoProcessor.from_pretrained(args.model_path, fps=args.fps)
        processor = Qwen3_5Processor(
            image_processor=None, tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=getattr(tokenizer, "chat_template", None),
        )
    processor.video_processor.fps = args.fps
    processor.video_processor.do_resize = False

    # Load HLD tasks
    with open(args.task_json) as f:
        all_tasks = json.load(f)
    hld_tasks = [t for t in all_tasks if t["task"] == "HLD"]
    if args.sample_ids:
        ids_set = set(int(s) for s in args.sample_ids)
        hld_tasks = [t for t in hld_tasks if t["id"] in ids_set]
    if args.max_samples:
        hld_tasks = hld_tasks[:args.max_samples]

    print(f"HLD samples: {len(hld_tasks)}")
    print(f"Sampling: min_frames={args.min_frames}, max_frames={args.max_frames}, fps={args.fps}")
    print(f"Resize: 512² budget, factor=32")
    print("=" * 60)

    correct = 0
    total = 0
    total_latency = 0.0

    output_path = args.output or f"eval_results/ovobench/fluxmem_style_hld_{int(time.time())}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for item in hld_tasks:
        item_id = item["id"]
        video_path = resolve_video_path(args.video_dir, item_id)
        if not os.path.exists(video_path):
            print(f"  [{item_id}] video not found: {video_path}")
            continue

        prompt = build_prompt("HLD", item["question"], item.get("options", []))
        gt_letter = chr(65 + item["gt"])

        try:
            frames, meta, nframes = fluxmem_style_sample(
                video_path, fps=args.fps,
                min_frames=args.min_frames, max_frames=args.max_frames,
            )
            response, latency = run_one_sample(
                model, processor, prompt, frames, meta, fps=args.fps,
                max_new_tokens=args.max_new_tokens,
            )
            correct += score_hld(response, gt_letter)
            total += 1
            total_latency += latency

            status = "✓" if score_hld(response, gt_letter) else "✗"
            print(f"  [{item_id}] {status} nframes={nframes} pred={response[:40]} gt={gt_letter}")

            with open(output_path, "a") as f:
                f.write(json.dumps({
                    "id": item_id, "task": "HLD",
                    "response": response, "ground_truth": gt_letter,
                    "correct": score_hld(response, gt_letter),
                    "nframes": nframes, "latency_s": round(latency, 3),
                }) + "\n")

        except Exception as e:
            print(f"  [{item_id}] ERROR: {e}")
            total += 1

    acc = 100 * correct / total if total > 0 else 0
    print("=" * 60)
    print(f"HLD Accuracy: {acc:.1f}% ({correct}/{total})")
    print(f"Avg latency: {total_latency/total:.2f}s" if total > 0 else "")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
