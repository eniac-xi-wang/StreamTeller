#!/usr/bin/env python3
"""Quick test + visualize pipeline for a single OVO sample.

Runs plugin PredictMem on one sample, then generates highlight MP4 + keepmask JSON.

Usage:
    python scripts/run_single_sample_viz.py --sample_id 509 --device cuda
    python scripts/run_single_sample_viz.py --sample_id 509 --frame_budget 64 --device cuda
"""

import argparse
import json
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

VIDEO_DIR = "/data/qinian_workspace/OVO-Bench/chunked_videos"
BENCH_PATH = "evaluate/ovobench/ovo_bench_new.json"


def load_sample(sample_id: str) -> dict:
    with open(BENCH_PATH) as f:
        data = json.load(f)
    for item in data:
        if str(item["id"]) == str(sample_id):
            return {
                **item,
                "sample_id": str(item["id"]),
                "video_path": str(Path(VIDEO_DIR) / f"{item['id']}.mp4"),
            }
    raise ValueError(f"Sample {sample_id} not found in {BENCH_PATH}")


def build_prompt(sample: dict) -> str:
    task = sample["task"]
    question = sample["question"]
    options = sample.get("options", [])
    answer = sample.get("answer", "")

    if options and task not in {"SSR", "CRR", "REC"}:
        letters = [chr(ord("A") + i) for i in range(len(options))]
        opts = "; ".join(f"{l}. {opt}" for l, opt in zip(letters, options))
        prompt = (
            f"Question: {question}\n"
            f"Options:\n{opts}\n\n"
            f"Respond only with the letter corresponding to your chosen option "
            f"(e.g., {', '.join(letters[:3])}).\n"
            f"Do not include any additional text or explanation in your response."
        )
    else:
        prompt = f"Question: {question}\n\nAnswer concisely in a few words."
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", required=True, help="OVO-Bench sample ID (e.g. 509)")
    parser.add_argument("--model_path", default="/data/model_weights_public/Qwen/Qwen3.5-9B")
    parser.add_argument("--predictmem_keep_ratio", type=float, default=0.10)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--frame_budget", type=int, default=0, help="Max frames (0=all)")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/single_test")
    args = parser.parse_args()

    sid = args.sample_id
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PredictMemConfig()
    config.keep_ratio = args.predictmem_keep_ratio
    config.fps = args.fps
    config.__post_init__()

    # ── Load sample ──
    sample = load_sample(sid)
    prompt = build_prompt(sample)
    print(f"Sample {sid} | task={sample['task']} | video={sample['video_path']}")
    print(f"  Q: {sample['question']}")
    print(f"  Options: {sample.get('options', [])}")
    print(f"  GT: {sample.get('answer', '')} (option {sample.get('gt', '')})")

    # ── Load video ──
    t_load = time.perf_counter()
    qwen_frames, predictmem_frames_256, _ = build_predictmem_video_inputs(
        sample["video_path"],
        fps=config.fps,
        qwen_size=config.qwen_size,
        jepa_size=config.jepa_size,
        num_frames=args.frame_budget if args.frame_budget > 0 else None,
    )
    num_frames = qwen_frames.shape[0]
    duration_s = num_frames / config.fps
    jepa_shape = list(predictmem_frames_256.shape)
    load_time = time.perf_counter() - t_load
    print(f"\n  Video: {num_frames} frames @ {config.fps} FPS = {duration_s:.0f}s")
    print(f"  JEPA tensor: {jepa_shape} (ImageNet normalized)")
    print(f"  Load time: {load_time:.2f}s")

    # ── Load model ──
    print(f"\nLoading Qwen3.5 from {args.model_path}...")
    from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else args.device,
    )
    model.eval()

    from transformers import AutoProcessor, AutoTokenizer
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        from models.qwen3_5.processing_qwen3_vl import Qwen3_5Processor
        from models.qwen3_5.video_processing_qwen3_vl import Qwen3_5VideoProcessor
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        try:
            video_processor = Qwen3_5VideoProcessor.from_pretrained(args.model_path, fps=config.fps)
        except Exception:
            video_processor = Qwen3_5VideoProcessor(fps=config.fps)
        processor = Qwen3_5Processor(
            image_processor=None, tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=getattr(tokenizer, "chat_template", None),
        )
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = config.fps
        processor.video_processor.do_resize = False

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.1f}B params, video_token_id={model.config.video_token_id}")

    # ── Build chat messages ──
    content = [
        {"type": "video", "video": qwen_frames, "fps": config.fps},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(f"  Chat template length: {len(text)} chars")
    print(f"  Chat has <|video_pad|>: {'<|video_pad|>' in text}")

    # ── Process inputs ──
    video_metadata = {
        "total_num_frames": num_frames,
        "fps": float(config.fps),
        "duration": duration_s,
        "frames_indices": list(range(num_frames)),
        "height": int(qwen_frames.shape[1]),
        "width": int(qwen_frames.shape[2]),
        "video_backend": "decord",
    }
    inputs = processor(
        text=[text], videos=[qwen_frames],
        video_metadata=[video_metadata],
        do_sample_frames=False, do_resize=False, fps=config.fps,
        return_tensors="pt",
    )
    inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    video_grid_thw = inputs["video_grid_thw"].detach().cpu()
    grid_t, grid_h, grid_w = (int(v) for v in video_grid_thw[0].tolist())
    merge = config.qwen_merge_size
    num_video_tokens = grid_t * (grid_h // merge) * (grid_w // merge)
    input_len = inputs["input_ids"].shape[1]
    print(f"  video_grid_thw: {video_grid_thw.tolist()}")
    print(f"  Video tokens: {num_video_tokens}, Total tokens: {input_len}")

    # ── Generate ──
    print(f"\nGenerating (PredictMem plugin, keep_ratio={config.keep_ratio})...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            use_predictmem=True,
            predictmem_frames_256=predictmem_frames_256,
            predictmem_keep_ratio=config.keep_ratio,
        )

    t1 = time.perf_counter()
    total_latency = t1 - t0
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    output_ids = generated_ids[0, input_len:]
    raw_response = processor.decode(output_ids, skip_special_tokens=True) if hasattr(processor, "decode") else processor.tokenizer.decode(output_ids, skip_special_tokens=True)

    # ── Get plugin stats ──
    pm_stats = getattr(getattr(model, "model", None), "predictmem_last_stats", {})
    vjepa_latency = pm_stats.get("predictmem_scoring_latency_s", 0)
    qwen_only_latency = total_latency - vjepa_latency
    keep_masks = pm_stats.get("predictmem_keep_masks", {})

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"RESULTS — Sample {sid}")
    print(f"{'='*60}")
    print(f"  Response:       '{raw_response.strip()}'")
    print(f"  Ground truth:   '{sample['answer']}'")
    print(f"  Task:           {sample['task']}")
    print(f"\n  Total latency:      {total_latency:.3f}s")
    print(f"  V-JEPA scoring:     {vjepa_latency:.3f}s ({vjepa_latency/total_latency*100:.0f}%)")
    print(f"  Qwen-only latency:  {qwen_only_latency:.3f}s ({qwen_only_latency/total_latency*100:.0f}%)")
    print(f"  Peak memory:        {peak_memory:.0f}MB")
    print(f"\n  Video tokens:       {num_video_tokens}")
    print(f"  Kept tokens:        {pm_stats.get('kept_video_tokens', '?')}")
    print(f"  Keep ratio actual:  {pm_stats.get('keep_ratio_actual', '?'):.1%}")
    token_compression = num_video_tokens / pm_stats.get('kept_video_tokens', 1) if pm_stats.get('kept_video_tokens') else 1
    print(f"  Token compression:  {token_compression:.1f}x")
    print(f"\n  num_tubelets_scored:    {pm_stats.get('num_tubelets_scored', '?')}")
    print(f"  dropped_bootstrap:      {pm_stats.get('dropped_bootstrap_tubelets', [])}")
    print(f"  full_keep_tail_tubelets: {pm_stats.get('full_keep_tail_tubelets', [])}")
    print(f"  full_keep_tail_frames:   {pm_stats.get('full_keep_tail_frames', [])}")
    if pm_stats.get('scored_tubelets'):
        scored = pm_stats['scored_tubelets']
        print(f"  scored tubelets range:   [{min(scored)}, {max(scored)}]")
    if keep_masks.get('tubelets'):
        modes = {}
        for t in keep_masks['tubelets']:
            m = t.get('mode', '?')
            modes[m] = modes.get(m, 0) + 1
        print(f"  per-tubelet mode dist:   {sorted(modes.items())}")

    # ── Save results JSONL ──
    entry = {
        "sample_id": sid,
        "task": sample["task"],
        "video": sample["video_path"],
        "question": sample["question"],
        "answer_text": sample["answer"],
        "options": sample.get("options", []),
        "response": raw_response.strip(),
        "raw_response": raw_response,
        "method": "predictmem",
        "predictmem_runtime": "plugin",
        "num_frames": num_frames,
        "video_grid_thw": video_grid_thw.tolist(),
        "expected_video_tokens": num_video_tokens,
        "keep_ratio_target": config.keep_ratio,
        "total_latency_s": round(total_latency, 4),
        "predictmem_scoring_latency_s": round(vjepa_latency, 4),
        "qwen_latency_excluding_vjepa_s": round(qwen_only_latency, 4),
        "peak_memory_mb": round(peak_memory, 1),
        "predictmem_stats": pm_stats,
    }
    jsonl_path = output_dir / f"sample_{sid}_result.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\n  Result saved: {jsonl_path}")

    # ── Generate visualization ──
    print(f"\nGenerating highlight video...")
    # Import from render_predictmem_highlight (same directory)
    sys.path.insert(0, str(Path(__file__).parent))
    from render_predictmem_highlight import render_highlight_video

    try:
        video_out = str(output_dir / f"sample_{sid}_highlight.mp4")
        render_highlight_video(
            sample["video_path"], keep_masks, num_frames, sid, video_out, fps=config.fps
        )
        size_mb = Path(video_out).stat().st_size / 1e6
        print(f"  Highlight video: {video_out} ({size_mb:.1f}MB)")

        keepmask_out = str(output_dir / f"sample_{sid}_keepmask.json")
        with open(keepmask_out, "w") as f:
            json.dump(keep_masks, f, indent=2)
        print(f"  Keepmask JSON:   {keepmask_out}")
    except Exception as e:
        print(f"  Visualization error: {e}")

    print(f"\n{'='*60}")
    print(f"Done! Outputs in {output_dir}/")
    for f in sorted(output_dir.glob(f"sample_{sid}*")):
        size = f.stat().st_size
        print(f"  {f.name} ({size/1e6:.1f}MB)" if size > 1e6 else f"  {f.name} ({size}B)")


if __name__ == "__main__":
    main()
