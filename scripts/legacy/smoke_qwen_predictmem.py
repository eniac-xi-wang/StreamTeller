#!/usr/bin/env python3
"""Single-sample smoke test for PredictMem with the local Qwen3.5 model.

The script samples exactly 16 frames at 1 FPS with decord, feeds Qwen a 512px
video tensor, and expects the default 2048-token video grid:

    video_grid_thw = [8, 32, 32]
    video_tokens = 8 * 32 * 32 / 2^2 = 2048

Usage:
    python scripts/smoke_qwen_predictmem.py \
      --model_path /data/model_weights_public/Qwen/Qwen3.5-VL-2B-Instruct \
      --video /data/qinian_workspace/OVO-Bench/chunked_videos/0.mp4 \
      --question "Who did I communicate to when chopping egg plants?" \
      --cache_path results/predictmem_scores.jsonl \
      --sample_id 0 \
      --method predictmem \
      --fps 1 \
      --num_frames 16 \
      --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _path in (_repo_root, _models_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from predictmem.cache import ScoreCache
from predictmem.config import PredictMemConfig
from predictmem.token_mapping import TokenMapper
from predictmem.video_sampling import sample_video_1fps_decord


def load_qwen35_model(model_path: str, device: str):
    """Load the local Qwen3.5 implementation with PredictMem kwargs enabled."""
    from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    return Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else device,
    )


def load_qwen35_processor(model_path: str, fps: float):
    """Load a Qwen3.5-compatible processor and force the project FPS default."""
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
            image_processor=None,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=getattr(tokenizer, "chat_template", None),
        )

    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = fps
        processor.video_processor.do_resize = False
    return processor


def move_inputs_to_device(inputs: dict, device: str) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def decode_new_tokens(processor, generated_ids: torch.Tensor, input_len: int) -> str:
    output_ids = generated_ids[0, input_len:]
    if hasattr(processor, "decode"):
        return processor.decode(output_ids, skip_special_tokens=True)
    return processor.tokenizer.decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--cache_path", default="results/predictmem_scores.jsonl")
    parser.add_argument("--sample_id", default="0")
    parser.add_argument("--method", default="predictmem", choices=["baseline", "predictmem"])
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--qwen_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--allow_dynamic_grid", action="store_true")
    args = parser.parse_args()

    if args.fps != 1.0:
        print(f"WARNING: project default is 1 FPS; using explicit --fps {args.fps}")

    device = args.device
    config = PredictMemConfig()
    config.keep_ratio = args.keep_ratio
    config.window_frames = args.num_frames
    config.fps = args.fps
    config.qwen_size = args.qwen_size
    config.__post_init__()

    print(f"Loading Qwen3.5 from {args.model_path}...")
    model = load_qwen35_model(args.model_path, device)
    model.eval()
    processor = load_qwen35_processor(args.model_path, args.fps)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.1f}B params, video_token_id={model.config.video_token_id}")

    print(f"Sampling {args.num_frames} frames at {args.fps:g} FPS with decord from {args.video}...")
    video_sample = sample_video_1fps_decord(
        args.video,
        num_frames=args.num_frames,
        size=args.qwen_size,
        target_fps=args.fps,
    )
    frames_np = video_sample.frames_uint8
    print(
        f"  Frames shape: {frames_np.shape}, source_fps={video_sample.source_fps:.3f}, "
        f"indices={video_sample.source_indices[:4]}...{video_sample.source_indices[-4:]}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames_np, "fps": args.fps},
                {"type": "text", "text": args.question},
            ],
        }
    ]

    print("Processing inputs...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        videos=[frames_np],
        video_metadata=[video_sample.qwen_metadata()],
        do_sample_frames=False,
        do_resize=False,
        fps=args.fps,
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(inputs, device)
    print(f"  input_ids shape: {inputs['input_ids'].shape}")
    if "pixel_values_videos" in inputs:
        pvv = inputs["pixel_values_videos"]
        print(f"  pixel_values_videos shape: {pvv.shape if isinstance(pvv, torch.Tensor) else 'N/A'}")
    if "video_grid_thw" not in inputs:
        print("ERROR: processor did not return video_grid_thw")
        sys.exit(1)

    mapper = TokenMapper(config)
    video_grid_thw = inputs["video_grid_thw"].detach().cpu()
    num_qwen_video_tokens = mapper.compute_num_video_tokens(video_grid_thw)
    print(f"  video_grid_thw: {video_grid_thw.tolist()} -> {num_qwen_video_tokens} Qwen video tokens")
    if not args.allow_dynamic_grid:
        try:
            mapper.assert_video_grid_thw(video_grid_thw, expected_t=config.qwen_grid_t)
        except AssertionError as exc:
            print(f"ERROR: unexpected Qwen grid under 1FPS/16-frame/512px contract: {exc}")
            print("       Use --allow_dynamic_grid only for diagnostic runs.")
            sys.exit(1)

    if args.method == "predictmem":
        cache = ScoreCache(args.cache_path)
        if not cache.has(args.sample_id):
            print(f"ERROR: sample {args.sample_id} not in cache")
            sys.exit(1)
        loss_map = cache.get_loss_map(args.sample_id)
        keep_mask = None if loss_map is not None else cache.get_keep_mask(args.sample_id)
        keep_indices = mapper.map_scores_to_qwen_keep_indices(
            video_grid_thw=video_grid_thw,
            loss_map=loss_map,
            keep_mask=keep_mask,
            keep_ratio=args.keep_ratio,
        )
        n_kept = keep_indices[0].shape[0]
        print(f"  PredictMem: {n_kept}/{num_qwen_video_tokens} tokens kept ({n_kept/num_qwen_video_tokens:.1%})")
    else:
        keep_indices = [torch.arange(num_qwen_video_tokens)]
        print(f"  Baseline: {num_qwen_video_tokens}/{num_qwen_video_tokens} tokens kept")

    print(f"\nGenerating ({args.method})...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    generate_kwargs = {}
    if args.method == "predictmem":
        generate_kwargs["use_predictmem"] = True
        generate_kwargs["predictmem_keep_indices"] = keep_indices

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            **generate_kwargs,
        )

    t1 = time.perf_counter()
    total_latency = t1 - t0
    output_text = decode_new_tokens(processor, generated_ids, inputs["input_ids"].shape[1])

    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_memory = 0.0

    kept = keep_indices[0].shape[0]
    result = {
        "sample_id": args.sample_id,
        "video": args.video,
        "question": args.question,
        "prediction": output_text,
        "method": args.method,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "qwen_size": args.qwen_size,
        "source_fps": video_sample.source_fps,
        "source_indices": video_sample.source_indices,
        "video_grid_thw": video_grid_thw.tolist(),
        "keep_ratio_target": args.keep_ratio,
        "keep_ratio_actual": kept / num_qwen_video_tokens,
        "original_video_tokens": num_qwen_video_tokens,
        "kept_video_tokens": kept,
        "score_latency_s": 0.0,
        "vision_latency_s": 0.0,
        "prefill_latency_s": round(total_latency, 4),
        "decode_latency_s": 0.0,
        "total_latency_s": round(total_latency, 4),
        "peak_memory_mb": round(peak_memory, 1),
    }

    print(f"\n{'=' * 60}")
    print(f"Method:      {args.method}")
    print(f"Prediction:  {output_text[:200]}")
    print(f"Kept tokens: {kept}/{num_qwen_video_tokens} ({kept/num_qwen_video_tokens:.1%})")
    print(f"Grid:        {video_grid_thw.tolist()}")
    print(f"Total latency: {total_latency:.3f}s")
    print(f"Peak mem:      {peak_memory:.0f}MB")
    print(f"{'=' * 60}")

    output_path = f"results/smoke_{args.method}_{args.sample_id}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
