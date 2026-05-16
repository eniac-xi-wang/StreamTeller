"""Shared Qwen3.5 + PredictMem helpers for OVO-Bench and StreamingBench.

All evaluation scripts use these functions to ensure consistent:
  - Model / processor loading with configurable PredictMem plugin config
  - Video sampling with start/end clipping (Qwen 512 + V-JEPA 256)
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

def load_qwen35_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    jepa_checkpoint_path: str | None = None,
    vjepa_src_path: str | None = None,
    predictmem_keep_ratio: float = 0.10,
    window_frames: int = 16,
    stride_frames: int = 2,
    tail_keep_frames: int = 4,
    drop_bootstrap: bool = True,
):
    """Load Qwen3.5-9B and configure the PredictMem plugin from arguments.

    The plugin is initialized by Qwen3_5ForConditionalGeneration.__init__ with
    a default config.  We update that config here so the bash entry controls
    every setting.
    """
    import sys
    _models_dir = Path(__file__).parent.parent.parent / "models"
    if str(_models_dir) not in sys.path:
        sys.path.insert(0, str(_models_dir))

    from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    dtype = getattr(torch, torch_dtype, torch.bfloat16)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else device,
    )
    model.eval()

    # Configure the PredictMem plugin from bash-level settings
    pm = getattr(getattr(model, "model", None), "predictmem", None)
    if pm is not None:
        pm.config.jepa_checkpoint_path = jepa_checkpoint_path
        pm.config.vjepa_src_path = vjepa_src_path
        pm.config.keep_ratio = predictmem_keep_ratio
        pm.config.window_frames = window_frames
        pm.config.temporal_stride = stride_frames
        pm.config.tail_keep_frames = tail_keep_frames
        pm.config.drop_bootstrap = drop_bootstrap
        pm.config.__post_init__()

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


# ── Video Input (with start/end clipping) ──────────────────────────────────

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
    """Sample a video with optional time clipping; return Qwen frames + V-JEPA tensor.

    All frame indices are derived from source video at ``fps``.  When
    ``start_time`` / ``end_time`` are given the clip is restricted to that
    interval.  ``frame_budget`` further limits the total number of 1FPS frames.

    Returns:
        qwen_frames_uint8: [N, qwen_size, qwen_size, 3] uint8 RGB
        predictmem_frames_256: [N, 3, jepa_size, jepa_size] ImageNet-normalized
        video_metadata: dict
    """
    import decord
    import torch.nn.functional as F

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)
    source_fps = float(vr.get_avg_fps() or fps)
    duration = total_frames / source_fps if source_fps > 0 else 0.0

    # Resolve time boundaries
    clip_start = max(0.0, start_time) if start_time is not None else 0.0
    clip_end = min(duration, end_time) if end_time is not None else duration
    clip_duration = clip_end - clip_start

    # Build 1FPS frame times and map to source indices
    total_1fps = max(1, int(clip_duration * fps))
    if frame_budget and frame_budget > 0:
        total_1fps = min(frame_budget, total_1fps)

    times_s = [clip_start + i / fps for i in range(total_1fps)]
    source_indices = [
        min(total_frames - 1, max(0, int(round(t * source_fps))))
        for t in times_s
    ]

    # Read all frames
    frames_raw = vr.get_batch(source_indices)
    if hasattr(frames_raw, "asnumpy"):
        frames_raw = torch.from_numpy(frames_raw.asnumpy())
    elif not isinstance(frames_raw, torch.Tensor):
        frames_raw = torch.from_numpy(np.asarray(frames_raw))
    frames_raw = frames_raw.to(dtype=torch.uint8)  # [N, src_h, src_w, 3]

    # Qwen frames: resize to qwen_size, keep uint8
    frames_chw = frames_raw.permute(0, 3, 1, 2).float()
    if frames_chw.shape[-2:] != (qwen_size, qwen_size):
        qwen_chw = F.interpolate(frames_chw, size=(qwen_size, qwen_size),
                                  mode="bilinear", align_corners=False)
    else:
        qwen_chw = frames_chw
    qwen_chw = qwen_chw.clamp(0, 255)
    qwen_frames_uint8 = qwen_chw.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()

    # V-JEPA frames: resize to jepa_size, ImageNet normalize
    from predictmem.vision_inputs import IMAGENET_MEAN, IMAGENET_STD
    if frames_chw.shape[-2:] != (jepa_size, jepa_size):
        jepa_chw = F.interpolate(frames_chw, size=(jepa_size, jepa_size),
                                  mode="bilinear", align_corners=False)
    else:
        jepa_chw = frames_chw
    jepa_01 = jepa_chw / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=jepa_01.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=jepa_01.device).view(1, 3, 1, 1)
    predictmem_frames_256 = ((jepa_01 - mean) / std).contiguous().cpu()

    # Processor-safe metadata (only keys Qwen's VideoMetadata expects)
    video_metadata = {
        "total_num_frames": total_1fps,
        "fps": float(fps),
        "duration": clip_duration,
        "frames_indices": source_indices,
        "height": qwen_size,
        "width": qwen_size,
        "video_backend": "decord",
    }

    # Extended metadata for logging / traceability (not passed to processor)
    extra_meta = {
        "clip_start": clip_start,
        "clip_end": clip_end,
        "source_fps": source_fps,
    }

    return qwen_frames_uint8, predictmem_frames_256, video_metadata, extra_meta


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
    qwen_frames: np.ndarray,
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

    Returns:
        (response_text, sample_stats_dict)
    """
    num_frames = qwen_frames.shape[0]

    if video_metadata is None:
        video_metadata = {
            "total_num_frames": num_frames,
            "fps": float(fps),
            "duration": num_frames / fps,
            "frames_indices": list(range(num_frames)),
            "height": int(qwen_frames.shape[1]),
            "width": int(qwen_frames.shape[2]),
            "video_backend": "decord",
        }

    # Warn on config mismatch
    if method == "predictmem" and predictmem_runtime == "none":
        import logging
        logging.getLogger(__name__).warning(
            "method=predictmem but predictmem_runtime=none; plugin is disabled"
        )

    content_blocks = [
        {"type": "video", "video": qwen_frames, "fps": fps},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content_blocks}]
    text = apply_chat_template(processor, messages, disable_thinking=disable_thinking)

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

    input_len = inputs["input_ids"].shape[1]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    t1 = time.perf_counter()
    total_latency = t1 - t0
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

    output_ids = generated_ids[0, input_len:]
    if hasattr(processor, "decode"):
        response = processor.decode(output_ids, skip_special_tokens=True)
    else:
        response = processor.tokenizer.decode(output_ids, skip_special_tokens=True)

    pm_stats = extract_predictmem_stats(model)
    vjepa_latency = pm_stats.get("predictmem_scoring_latency_s", 0.0) if pm_stats else 0.0

    video_grid_thw = inputs["video_grid_thw"].detach().cpu()
    grid_t, grid_h, grid_w = (int(v) for v in video_grid_thw[0].tolist())
    merge = 2
    num_video_tokens = grid_t * (grid_h // merge) * (grid_w // merge)

    # Attach plugin config to stats for traceability
    if pm_stats:
        pm = getattr(getattr(model, "model", None), "predictmem", None)
        if pm is not None:
            pm_stats["jepa_checkpoint_path"] = pm.config.jepa_checkpoint_path
            pm_stats["vjepa_src_path"] = pm.config.vjepa_src_path
            pm_stats["window_frames"] = pm.config.window_frames
            pm_stats["stride_frames"] = pm.config.temporal_stride
            pm_stats["tail_keep_frames"] = pm.config.tail_keep_frames
            pm_stats["drop_bootstrap"] = pm.config.drop_bootstrap

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
    return getattr(getattr(model, "model", None), "predictmem_last_stats", None)
