"""Shared Qwen3.5 + PredictMem helpers for OVO-Bench and StreamingBench.

All evaluation scripts use these functions to ensure consistent:
  - Model / processor loading
  - Video sampling (Qwen 512 + V-JEPA 256 from the same frame plan)
  - Generation kwargs (baseline vs PredictMem plugin)
  - Per-sample stats extraction
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ── Model / Processor Loading ──────────────────────────────────────────────

def load_qwen35_model(model_path: str, device: str = "cuda"):
    """Load Qwen3.5-9B with PredictMem plugin initialized."""
    import sys
    _models_dir = Path(__file__).parent.parent.parent / "models"
    if str(_models_dir) not in sys.path:
        sys.path.insert(0, str(_models_dir))

    from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else device,
    )
    model.eval()
    return model


def load_qwen35_processor(model_path: str, fps: float = 1.0):
    """Load Qwen3.5 processor with fixed fps and no auto-resize."""
    import sys
    _models_dir = Path(__file__).parent.parent.parent / "models"
    if str(_models_dir) not in sys.path:
        sys.path.insert(0, str(_models_dir))

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


# ── Video Input ────────────────────────────────────────────────────────────

def build_video_inputs_for_eval(
    video_path: str,
    fps: float = 1.0,
    qwen_size: int = 512,
    jepa_size: int = 256,
    frame_budget: int = 0,
    stream_mode: str = "full",
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[np.ndarray, torch.Tensor | None, dict]:
    """Sample a video once; return Qwen frames, V-JEPA tensor, and metadata.

    Args:
        video_path: path to mp4
        fps: target frame rate (default 1.0)
        qwen_size: Qwen input resolution (512)
        jepa_size: V-JEPA input resolution (256)
        frame_budget: max frames (0 = all)
        stream_mode: sampling mode (full, tail_budget, etc.)
        start_time: clip start in seconds (for StreamingBench)
        end_time: clip end in seconds (for StreamingBench)

    Returns:
        qwen_frames_uint8: [N, qwen_size, qwen_size, 3] uint8 RGB
        predictmem_frames_256: [N, 3, jepa_size, jepa_size] ImageNet-normalized
        video_metadata: dict
    """
    import sys
    _models_dir = Path(__file__).parent.parent.parent / "models"
    if str(_models_dir) not in sys.path:
        sys.path.insert(0, str(_models_dir))

    from predictmem.vision_inputs import build_predictmem_video_inputs

    num_frames = frame_budget if frame_budget > 0 else None
    qwen_frames, predictmem_frames_256, video_metadata = build_predictmem_video_inputs(
        video_path,
        fps=fps,
        qwen_size=qwen_size,
        jepa_size=jepa_size,
        num_frames=num_frames,
    )

    return qwen_frames, predictmem_frames_256, video_metadata


# ── Chat / Generation ──────────────────────────────────────────────────────

def apply_chat_template(processor, messages: list, disable_thinking: bool = True) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def generate_qwen35_response(
    model,
    processor,
    prompt: str,
    video_path: str | None = None,
    qwen_frames: np.ndarray | None = None,
    video_metadata: dict | None = None,
    method: str = "baseline",
    predictmem_runtime: str = "none",
    predictmem_frames_256: torch.Tensor | None = None,
    predictmem_keep_ratio: float = 0.10,
    fps: float = 1.0,
    max_new_tokens: int = 16,
    device: str = "cuda",
    disable_thinking: bool = True,
) -> tuple[str, dict]:
    """Run Qwen3.5 generation for one sample.

    Args:
        model: Qwen3_5ForConditionalGeneration
        processor: Qwen3_5Processor
        prompt: text prompt
        qwen_frames: [N, 512, 512, 3] uint8, pre-sampled Qwen frames
        video_metadata: dict with fps, duration, frames_indices, etc.
        method: baseline | predictmem
        predictmem_runtime: plugin | none
        predictmem_frames_256: [N, 3, 256, 256] ImageNet-normalized
        predictmem_keep_ratio: top fraction to keep (default 0.10)
        fps: FPS for metadata
        max_new_tokens: generation budget
        device: cuda | cpu
        disable_thinking: suppress Qwen3.5 <think> output

    Returns:
        (response_text, sample_stats_dict)
    """
    num_frames = qwen_frames.shape[0]
    duration = num_frames / fps

    # Build metadata
    if video_metadata is None:
        video_metadata = {
            "total_num_frames": num_frames,
            "fps": float(fps),
            "duration": duration,
            "frames_indices": list(range(num_frames)),
            "height": int(qwen_frames.shape[1]),
            "width": int(qwen_frames.shape[2]),
            "video_backend": "decord",
        }

    # Build messages and apply chat template
    content_blocks = [
        {"type": "video", "video": qwen_frames, "fps": fps},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content_blocks}]
    text = apply_chat_template(processor, messages, disable_thinking=disable_thinking)

    # Processor inputs
    inputs = processor(
        text=[text],
        videos=[qwen_frames],
        video_metadata=[video_metadata],
        do_sample_frames=False,
        do_resize=False,
        fps=fps,
        return_tensors="pt",
    )
    if device == "cuda":
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Build generate kwargs
    generate_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    if method == "predictmem" and predictmem_runtime == "plugin":
        generate_kwargs["use_predictmem"] = True
        generate_kwargs["predictmem_frames_256"] = predictmem_frames_256
        generate_kwargs["predictmem_keep_ratio"] = predictmem_keep_ratio
    else:
        generate_kwargs["use_predictmem"] = False

    # Generate
    input_len = inputs["input_ids"].shape[1]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    t1 = time.perf_counter()
    total_latency = t1 - t0
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Decode
    output_ids = generated_ids[0, input_len:]
    if hasattr(processor, "decode"):
        response = processor.decode(output_ids, skip_special_tokens=True)
    else:
        response = processor.tokenizer.decode(output_ids, skip_special_tokens=True)

    # Extract PredictMem stats
    pm_stats = extract_predictmem_stats(model)
    vjepa_latency = pm_stats.get("predictmem_scoring_latency_s", 0.0) if pm_stats else 0.0

    video_grid_thw = inputs["video_grid_thw"].detach().cpu()
    grid_t, grid_h, grid_w = (int(v) for v in video_grid_thw[0].tolist())
    merge = 2  # qwen_merge_size
    num_video_tokens = grid_t * (grid_h // merge) * (grid_w // merge)

    stats = {
        "total_latency_s": round(total_latency, 4),
        "predictmem_scoring_latency_s": round(vjepa_latency, 4),
        "qwen_latency_excluding_vjepa_s": round(total_latency - vjepa_latency, 4),
        "peak_memory_mb": round(peak_memory, 1),
        "video_grid_thw": video_grid_thw.tolist(),
        "expected_video_tokens": num_video_tokens,
        "fps": fps,
        "num_frames": num_frames,
        "predictmem_stats": pm_stats,
    }

    return response, stats


def extract_predictmem_stats(model) -> dict | None:
    """Extract PredictMem stats from model after generation."""
    return getattr(getattr(model, "model", None), "predictmem_last_stats", None)
