#!/usr/bin/env python3
"""Quick test + visualize pipeline for a single OVO sample.

Runs compact PredictMem on one sample, then generates highlight MP4 + keepmask JSON.

Usage:
    python scripts/run_single_sample_viz.py --sample_id 509 --device cuda
    python scripts/run_single_sample_viz.py --sample_id 509 --frame_budget 64 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path


_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _path in (_repo_root, _models_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evaluate.common.qwen35_predictmem import (
    generate_with_compact_memory,
    load_qwen35_model,
    load_qwen35_processor,
)

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
    parser.add_argument("--jepa_checkpoint_path", default="/data/model_weights_public/jepa/jeap_vitl_16_256.pt")
    parser.add_argument("--vjepa_src_path", default=None)
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

    # ── Load sample ──
    sample = load_sample(sid)
    prompt = build_prompt(sample)
    print(f"Sample {sid} | task={sample['task']} | video={sample['video_path']}")
    print(f"  Q: {sample['question']}")
    print(f"  Options: {sample.get('options', [])}")
    print(f"  GT: {sample.get('answer', '')} (option {sample.get('gt', '')})")

    model = load_qwen35_model(
        args.model_path, device=args.device, jepa_checkpoint_path=args.jepa_checkpoint_path,
        vjepa_src_path=args.vjepa_src_path, predictmem_keep_ratio=args.predictmem_keep_ratio,
        record_keep_masks=True,
    )
    processor = load_qwen35_processor(args.model_path, args.fps)
    raw_response, stats = generate_with_compact_memory(
        model, processor, prompt, sample["video_path"], fps=args.fps,
        frame_budget=args.frame_budget, predictmem_keep_ratio=args.predictmem_keep_ratio,
        max_new_tokens=args.max_new_tokens, device=args.device,
    )
    pm_stats = stats["predictmem_stats"]
    keep_masks = pm_stats["predictmem_keep_masks"]
    num_frames = stats["num_frames"]
    print(f"Response: {raw_response.strip()}")
    print(f"Kept {pm_stats['kept_video_tokens']} of {stats['expected_video_tokens']} video tokens")

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
        "predictmem_runtime": "compact",
        "keep_ratio_target": args.predictmem_keep_ratio,
        **stats,
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
            sample["video_path"], keep_masks, num_frames, sid, video_out, fps=args.fps
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
