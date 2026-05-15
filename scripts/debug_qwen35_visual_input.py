#!/usr/bin/env python3
"""Debug whether Qwen3.5 receives video inputs before generation.

Default mode is processor-only and does not load the 9B model. It checks:
- chat template contains the video placeholder
- processor expands placeholders to the expected number of video tokens
- video_grid_thw and pixel_values_videos are present and shape-consistent
- mm_token_type_ids marks the video token span

Use --model_probe only when GPU memory is available. It wraps generation hooks
to verify the first prefill receives pixel_values_videos.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _path in (_repo_root, _models_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from predictmem.config import PredictMemConfig
from scripts.run_predictmem_ovobench import (
    build_ovo_prompt,
    get_window_bounds,
    load_qwen35_model,
    load_qwen35_processor,
    move_inputs_to_device,
    sample_window,
)


def load_sample(bench_path: str, video_dir: str, sample_id: str) -> dict:
    with open(bench_path) as f:
        data = json.load(f)
    for item in data:
        if str(item["id"]) == str(sample_id):
            return {
                **item,
                "sample_id": str(item["id"]),
                "video_path": str(Path(video_dir) / f"{item['id']}.mp4"),
            }
    raise ValueError(f"sample_id={sample_id} not found in {bench_path}")


def apply_chat_template(processor, messages, disable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--bench_path", default="evaluate/ovobench/ovo_bench_new.json")
    parser.add_argument("--video_dir", default="/data/qinian_workspace/OVO-Bench/chunked_videos")
    parser.add_argument("--sample_id", default="0")
    parser.add_argument("--window_policy", default="anchor_end16", choices=["first16", "anchor_end16", "full_stream"])
    parser.add_argument("--max_stream_frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--qwen_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--model_probe", action="store_true")
    args = parser.parse_args()

    config = PredictMemConfig()
    config.fps = args.fps
    config.window_frames = args.num_frames
    config.qwen_size = args.qwen_size
    config.__post_init__()

    sample = load_sample(args.bench_path, args.video_dir, args.sample_id)
    processor = load_qwen35_processor(args.model_path, args.fps)
    prompt, task, gt = build_ovo_prompt(sample)
    window_start_s, window_end_s, video_duration_s = get_window_bounds(
        sample["video_path"],
        args.window_policy,
        config.window_frames / config.fps,
        args.fps,
        args.max_stream_frames,
    )
    video_sample, actual_start_s, actual_end_s = sample_window(
        sample["video_path"], window_start_s, window_end_s, config
    )
    frames_np = video_sample.frames_uint8

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames_np, "fps": config.fps},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = apply_chat_template(processor, messages, args.disable_thinking)
    inputs = processor(
        text=[text],
        videos=[frames_np],
        video_metadata=[video_sample.qwen_metadata()],
        do_sample_frames=False,
        do_resize=False,
        fps=config.fps,
        return_tensors="pt",
    )

    tokenizer = processor.tokenizer
    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    input_ids = inputs["input_ids"][0]
    video_grid_thw = inputs["video_grid_thw"]
    merge = config.qwen_merge_size
    expected_video_tokens = int(
        (video_grid_thw[:, 0] * (video_grid_thw[:, 1] // merge) * (video_grid_thw[:, 2] // merge)).sum().item()
    )
    actual_video_tokens = int((input_ids == video_token_id).sum().item())
    mm_video_tokens = None
    if "mm_token_type_ids" in inputs:
        mm_video_tokens = int((inputs["mm_token_type_ids"][0] == 2).sum().item())

    debug = {
        "sample_id": sample["sample_id"],
        "task": task,
        "ground_truth": gt,
        "processor_class": type(processor).__name__,
        "video_processor_class": type(processor.video_processor).__name__,
        "video_processor_fps": getattr(processor.video_processor, "fps", None),
        "video_processor_do_resize": getattr(processor.video_processor, "do_resize", None),
        "chat_contains_video_pad": "<|video_pad|>" in text,
        "chat_video_pad_count_before_processing": text.count("<|video_pad|>"),
        "chat_opens_thinking": text.rstrip().endswith("<think>"),
        "chat_has_empty_think_block": "<think>\n\n</think>" in text,
        "window_policy": args.window_policy,
        "window_start_s": actual_start_s,
        "window_end_s": actual_end_s,
        "video_duration_s": video_duration_s,
        "num_frames": int(frames_np.shape[0]),
        "source_indices_head": video_sample.source_indices[:4],
        "source_indices_tail": video_sample.source_indices[-4:],
        "video_grid_thw": video_grid_thw.tolist(),
        "pixel_values_videos_shape": list(inputs["pixel_values_videos"].shape),
        "input_ids_shape": list(inputs["input_ids"].shape),
        "video_token_id": video_token_id,
        "vision_start_token_id": vision_start_token_id,
        "vision_end_token_id": vision_end_token_id,
        "expected_llm_video_tokens": expected_video_tokens,
        "actual_input_video_tokens": actual_video_tokens,
        "vision_start_count": int((input_ids == vision_start_token_id).sum().item()),
        "vision_end_count": int((input_ids == vision_end_token_id).sum().item()),
        "mm_video_tokens": mm_video_tokens,
        "token_count_match": actual_video_tokens == expected_video_tokens == (mm_video_tokens or expected_video_tokens),
    }

    if args.model_probe:
        model = load_qwen35_model(args.model_path, args.device)
        model.eval()
        inputs_on_device = move_inputs_to_device(inputs, args.device)
        prepare_records = []
        forward_records = []

        original_prepare = model.prepare_inputs_for_generation
        original_forward = model.model.forward

        def wrapped_prepare(*pargs, **kwargs):
            prepared = original_prepare(*pargs, **kwargs)
            pvv = prepared.get("pixel_values_videos")
            prepare_records.append(
                {
                    "is_first_iteration": kwargs.get("is_first_iteration"),
                    "has_pixel_values_videos": pvv is not None,
                    "pixel_values_videos_shape": list(pvv.shape) if isinstance(pvv, torch.Tensor) else None,
                    "has_video_grid_thw": prepared.get("video_grid_thw") is not None,
                }
            )
            return prepared

        def wrapped_forward(*pargs, **kwargs):
            pvv = kwargs.get("pixel_values_videos")
            input_ids_arg = kwargs.get("input_ids")
            forward_records.append(
                {
                    "has_pixel_values_videos": pvv is not None,
                    "pixel_values_videos_shape": list(pvv.shape) if isinstance(pvv, torch.Tensor) else None,
                    "input_len": int(input_ids_arg.shape[1]) if isinstance(input_ids_arg, torch.Tensor) else None,
                }
            )
            return original_forward(*pargs, **kwargs)

        model.prepare_inputs_for_generation = wrapped_prepare
        model.model.forward = wrapped_forward
        with torch.no_grad():
            model.generate(
                **inputs_on_device,
                max_new_tokens=1,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_predictmem=False,
            )
        debug["model_probe"] = {
            "prepare_records": prepare_records,
            "forward_records": forward_records,
        }

    print(json.dumps(debug, indent=2))


if __name__ == "__main__":
    main()
