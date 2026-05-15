#!/usr/bin/env python3
"""Real Qwen3.5 evaluator for PredictMem on OVO-Bench.

Loads Qwen3.5-9B, samples videos at 1FPS with decord, and runs all four
methods (baseline, random, uniform, predictmem) with real model.generate().

Usage:
    python scripts/run_predictmem_ovobench.py \
        --model_path /data/model_weights_public/Qwen/Qwen3.5-9B \
        --cache_path results/predictmem_scores_1fps_5.jsonl \
        --max_samples 5 --device cuda
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

from predictmem.config import PredictMemConfig
from predictmem.token_mapping import TokenMapper
from predictmem.cache import ScoreCache
from predictmem.video_sampling import sample_video_1fps_decord


def load_qwen35_model(model_path: str, device: str):
    from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    return Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
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


def generate_keep_indices_for_method(method: str, config: PredictMemConfig,
                                      sample_id: str, cache: ScoreCache | None,
                                      seed: int, video_grid_thw: torch.Tensor,
                                      mapper: TokenMapper) -> tuple[list, dict]:
    """Generate keep_indices for one sample. Returns (keep_indices_list, stats_dict)."""
    num_tokens = mapper.compute_num_video_tokens(video_grid_thw)
    stats = {"original_video_tokens": num_tokens}

    if method == "baseline":
        keep = [torch.arange(num_tokens)]
        stats["keep_ratio_actual"] = 1.0
        stats["kept_video_tokens"] = num_tokens

    elif method == "random":
        torch.manual_seed(seed)
        n_keep = max(1, int(num_tokens * config.keep_ratio))
        perm = torch.randperm(num_tokens)
        keep = [perm[:n_keep].sort().values]
        stats["keep_ratio_actual"] = n_keep / num_tokens
        stats["kept_video_tokens"] = n_keep

    elif method == "uniform":
        n_keep = max(1, int(num_tokens * config.keep_ratio))
        step = num_tokens / n_keep
        keep_local = torch.round(torch.arange(n_keep).float() * step).long()
        keep_local = keep_local.clamp(0, num_tokens - 1).unique()
        keep = [keep_local.sort().values]
        stats["keep_ratio_actual"] = len(keep[0]) / num_tokens
        stats["kept_video_tokens"] = len(keep[0])

    elif method == "predictmem":
        if cache is not None and cache.has(sample_id):
            loss_map = cache.get_loss_map(sample_id)
            keep_mask = None if loss_map is not None else cache.get_keep_mask(sample_id)
            if loss_map is None and keep_mask is None:
                raise ValueError(f"Cached sample {sample_id} has neither loss_map nor keep_mask")
            keep = mapper.map_scores_to_qwen_keep_indices(
                video_grid_thw=video_grid_thw,
                loss_map=loss_map,
                keep_mask=keep_mask,
                keep_ratio=config.keep_ratio,
            )
            stats["keep_ratio_actual"] = len(keep[0]) / num_tokens
            stats["kept_video_tokens"] = len(keep[0])
        else:
            raise ValueError(f"Sample {sample_id} not in cache, run precompute first")
    else:
        raise ValueError(f"Unknown method: {method}")

    return keep, stats


def load_ovo_samples(bench_path: str, max_samples: int, video_base: str) -> list[dict]:
    with open(bench_path) as f:
        data = json.load(f)

    samples = []
    for item in data[:max_samples]:
        samples.append({
            "sample_id": str(item["id"]),
            "video": str(Path(video_base) / f"{item['id']}.mp4"),
            "question": item["question"],
            "ground_truth": item["answer"],
            "options": item.get("options", []),
            "gt_idx": item.get("gt"),
        })
    return samples


def run_one_sample(model, processor, sample: dict, method: str, config: PredictMemConfig,
                   cache: ScoreCache | None, seed: int, device: str,
                   mapper: TokenMapper, args) -> dict:
    """Run a single sample with real Qwen3.5 generate."""
    sample_id = sample["sample_id"]

    # Sample frames
    video_sample = sample_video_1fps_decord(
        sample["video"],
        num_frames=config.window_frames,
        size=config.qwen_size,
        target_fps=config.fps,
    )
    frames_np = video_sample.frames_uint8

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames_np, "fps": config.fps},
                {"type": "text", "text": sample["question"]},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        videos=[frames_np],
        video_metadata=[video_sample.qwen_metadata()],
        do_sample_frames=False,
        do_resize=False,
        fps=config.fps,
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(inputs, device)

    video_grid_thw = inputs["video_grid_thw"].detach().cpu()
    num_qwen_video_tokens = mapper.compute_num_video_tokens(video_grid_thw)

    # Generate keep indices for this method
    keep_indices, stats = generate_keep_indices_for_method(
        method=method, config=config, sample_id=sample_id,
        cache=cache, seed=seed, video_grid_thw=video_grid_thw, mapper=mapper,
    )

    # Run generation
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    generate_kwargs = {}
    if method != "baseline":
        generate_kwargs["use_predictmem"] = True
        generate_kwargs["predictmem_keep_indices"] = keep_indices
    else:
        generate_kwargs["use_predictmem"] = False

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

    kept = keep_indices[0].shape[0] if keep_indices else num_qwen_video_tokens

    return {
        "sample_id": sample_id,
        "video": sample["video"],
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "prediction": output_text,
        "method": method,
        "fps": config.fps,
        "num_frames": config.window_frames,
        "qwen_size": config.qwen_size,
        "source_fps": video_sample.source_fps,
        "source_indices": video_sample.source_indices,
        "video_grid_thw": video_grid_thw.tolist(),
        "keep_ratio_target": config.keep_ratio,
        "keep_ratio_actual": stats["keep_ratio_actual"],
        "original_video_tokens": stats["original_video_tokens"],
        "kept_video_tokens": stats["kept_video_tokens"],
        "score_latency_s": 0.0,
        "vision_latency_s": 0.0,
        "prefill_latency_s": round(total_latency, 4),
        "decode_latency_s": 0.0,
        "total_latency_s": round(total_latency, 4),
        "peak_memory_mb": round(peak_memory, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        default="/data/model_weights_public/Qwen/Qwen3.5-9B")
    parser.add_argument("--method", required=True,
                        choices=["baseline", "random", "uniform", "predictmem"])
    parser.add_argument("--cache_path", default="results/predictmem_scores.jsonl")
    parser.add_argument("--bench_path", default="evaluate/ovobench/ovo_bench_new.json")
    parser.add_argument("--video_dir", default="/data/qinian_workspace/OVO-Bench/chunked_videos")
    parser.add_argument("--output", default="results/real_eval.jsonl")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--qwen_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--allow_dynamic_grid", action="store_true")
    args = parser.parse_args()

    config = PredictMemConfig()
    config.keep_ratio = args.keep_ratio
    config.fps = args.fps
    config.window_frames = args.num_frames
    config.qwen_size = args.qwen_size
    config.__post_init__()

    # Load model
    print(f"Loading Qwen3.5 from {args.model_path}...")
    model = load_qwen35_model(args.model_path, args.device)
    model.eval()
    processor = load_qwen35_processor(args.model_path, args.fps)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.1f}B params, video_token_id={model.config.video_token_id}")

    mapper = TokenMapper(config)

    # Load cache for predictmem method
    cache = None
    if args.method == "predictmem":
        cache = ScoreCache(args.cache_path)
        if len(cache) == 0:
            print("ERROR: cache is empty, run precompute_predictmem_scores.py first")
            sys.exit(1)
        print(f"Loaded cache with {len(cache)} entries")

    # Load samples
    samples = load_ovo_samples(args.bench_path, args.max_samples, args.video_dir)
    print(f"Running {len(samples)} samples, method={args.method}")

    # Run evaluation
    results = []
    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        print(f"\n[{i+1}/{len(samples)}] sample_id={sid} method={args.method}")

        seed = args.seed + i
        try:
            entry = run_one_sample(
                model=model, processor=processor, sample=sample,
                method=args.method, config=config, cache=cache,
                seed=seed, device=args.device, mapper=mapper, args=args,
            )
            results.append(entry)
            print(f"  Prediction: {entry['prediction'][:120]}")
            print(f"  Kept: {entry['kept_video_tokens']}/{entry['original_video_tokens']} "
                  f"({entry['keep_ratio_actual']:.1%})")
            print(f"  Latency: {entry['total_latency_s']:.3f}s, Peak mem: {entry['peak_memory_mb']:.0f}MB")
            print(f"  Grid: {entry['video_grid_thw']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "sample_id": sid,
                "video": sample["video"],
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "prediction": f"[error: {str(e)[:100]}]",
                "method": args.method,
                "fps": config.fps,
                "num_frames": config.window_frames,
                "video_grid_thw": [[config.qwen_grid_t, config.qwen_grid_h, config.qwen_grid_w]],
                "keep_ratio_target": config.keep_ratio,
                "keep_ratio_actual": 0.0,
                "original_video_tokens": config.num_qwen_video_tokens,
                "kept_video_tokens": 0,
                "score_latency_s": 0.0,
                "vision_latency_s": 0.0,
                "prefill_latency_s": 0.0,
                "decode_latency_s": 0.0,
                "total_latency_s": 0.0,
                "peak_memory_mb": 0.0,
            })

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults written to {args.output} ({len(results)} entries)")


if __name__ == "__main__":
    main()
