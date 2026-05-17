"""Tests for streaming_sampler, qwen_visual_chunk, compact_memory modules."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

_repo_root = Path(__file__).parent.parent
for _p in (_repo_root, _repo_root / "models", _repo_root / "evaluate"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _video_path():
    p = "/data/qinian_workspace/OVO-Bench/chunked_videos/0.mp4"
    return p if os.path.exists(p) else None


# ── StreamingVideoSampler tests ──────────────────────────────────────────

def test_streaming_sampler_shapes():
    vp = _video_path()
    if vp is None:
        print("  (skip — no test video)")
        return

    from predictmem.streaming_sampler import StreamingVideoSampler

    sampler = StreamingVideoSampler(vp, fps=1.0, qwen_size=512, jepa_size=256, frame_budget=32)
    for tubelet in sampler:
        assert tubelet["qwen"].shape[1:] == (512, 512, 3), f"Qwen shape wrong: {tubelet['qwen'].shape}"
        assert tubelet["jepa"].shape[1:] == (3, 256, 256), f"JEPA shape wrong: {tubelet['jepa'].shape}"
        assert tubelet["qwen"].shape[0] == tubelet["jepa"].shape[0], "Frame count mismatch"
        n = tubelet["num_frames_in_tubelet"]
        assert 1 <= n <= 2, f"Expected 1-2 frames per tubelet, got {n}"
        break
    print(f"✓ streaming_sampler shapes: Qwen {tubelet['qwen'].shape}, JEPA {tubelet['jepa'].shape}")


def test_streaming_sampler_iteration():
    vp = _video_path()
    if vp is None:
        print("  (skip — no test video)")
        return

    from predictmem.streaming_sampler import StreamingVideoSampler

    sampler = StreamingVideoSampler(vp, fps=1.0, frame_budget=16)
    tubelets = list(sampler)
    assert len(tubelets) == sampler.num_tubelets
    total_frames = sum(t["num_frames_in_tubelet"] for t in tubelets)
    assert total_frames <= 16
    assert sampler.metadata["total_num_frames"] <= 16
    print(f"✓ streaming_sampler iteration: {len(tubelets)} tubelets, {total_frames} frames")


def test_streaming_sampler_time_clip():
    vp = _video_path()
    if vp is None:
        print("  (skip — no test video)")
        return

    from predictmem.streaming_sampler import StreamingVideoSampler

    full = StreamingVideoSampler(vp, fps=1.0)
    clipped = StreamingVideoSampler(vp, fps=1.0, start_time=0, end_time=5)

    assert clipped.num_frames <= full.num_frames
    assert clipped.num_tubelets <= full.num_tubelets
    print(f"✓ time clip: full={full.num_frames} frames, clip0-5={clipped.num_frames} frames")


def test_streaming_sampler_frame_budget():
    vp = _video_path()
    if vp is None:
        print("  (skip — no test video)")
        return

    from predictmem.streaming_sampler import StreamingVideoSampler

    sampler = StreamingVideoSampler(vp, fps=1.0, frame_budget=8)
    assert sampler.num_frames <= 8
    tubelets = list(sampler)
    total = sum(t["num_frames_in_tubelet"] for t in tubelets)
    assert total <= 8
    print(f"✓ frame_budget=8: {total} frames")


def test_streaming_sampler_metadata():
    vp = _video_path()
    if vp is None:
        print("  (skip — no test video)")
        return

    from predictmem.streaming_sampler import StreamingVideoSampler

    sampler = StreamingVideoSampler(vp, fps=1.0, frame_budget=8)
    meta = sampler.metadata
    for key in ["total_num_frames", "fps", "duration", "frames_indices", "height", "width", "video_backend"]:
        assert key in meta, f"Missing metadata key: {key}"
    extra = sampler.extra_meta
    assert "clip_start" in extra
    assert "num_tubelets" in extra
    print(f"✓ metadata complete: {sorted(meta.keys())}")


# ── Memory debug tests ───────────────────────────────────────────────────

def test_memory_snapshot():
    from evaluate.common.memory_debug import snapshot
    s = snapshot(0)
    for key in ["allocated_mb", "reserved_mb", "max_allocated_mb", "max_reserved_mb",
                "free_mb", "total_mb", "nvml_mb", "rss_mb"]:
        assert key in s, f"Missing snapshot key: {key}"
        assert isinstance(s[key], (int, float)), f"{key} is not numeric: {type(s[key])}"
    print("✓ memory snapshot has all keys")


def test_memory_tracer_context():
    from evaluate.common.memory_debug import MemoryTracer
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        log_path = f.name
    try:
        with MemoryTracer(enabled=True, log_path=log_path) as tracer:
            x = torch.zeros(1000, 1000, device="cuda")
            tracer.checkpoint("after_alloc", num_frames=16)
            del x

        assert len(tracer.records) >= 2  # sample_begin + after_alloc + sample_end
        assert tracer.records[0]["checkpoint"] == "sample_begin"
        peak = tracer.peak_stage()
        assert peak is not None
        assert peak["memory"]["allocated_mb"] > 0
        print(f"✓ MemoryTracer: {len(tracer.records)} checkpoints, peak at '{peak['checkpoint']}'")

        # Verify log file
        with open(log_path) as lf:
            lines = [line for line in lf if line.strip()]
        assert len(lines) == len(tracer.records)
    finally:
        os.unlink(log_path)


# ── Config tests ─────────────────────────────────────────────────────────

def test_config_record_keep_masks():
    from predictmem.config import PredictMemConfig
    cfg = PredictMemConfig()
    assert cfg.record_keep_masks is False
    cfg.record_keep_masks = True
    cfg.__post_init__()
    assert cfg.record_keep_masks is True
    print("✓ record_keep_masks config flag")


# ── Compact memory module smoke ────────────────────────────────────────

def test_compact_memory_module_exists():
    from predictmem.compact_memory import PredictMemCompactMemory
    assert PredictMemCompactMemory is not None
    print("✓ PredictMemCompactMemory importable")


def test_streaming_sampler_module_exists():
    from predictmem.streaming_sampler import StreamingVideoSampler, IMAGENET_MEAN, IMAGENET_STD
    assert len(IMAGENET_MEAN) == 3
    assert len(IMAGENET_STD) == 3
    print(f"✓ StreamingVideoSampler + ImageNet constants: mean={IMAGENET_MEAN[0]}, std={IMAGENET_STD[0]}")


def test_qwen_visual_chunk_module_exists():
    from predictmem.qwen_visual_chunk import QwenVisualChunkProcessor
    assert QwenVisualChunkProcessor is not None
    print("✓ QwenVisualChunkProcessor importable")


if __name__ == "__main__":
    test_streaming_sampler_shapes()
    test_streaming_sampler_iteration()
    test_streaming_sampler_time_clip()
    test_streaming_sampler_frame_budget()
    test_streaming_sampler_metadata()
    test_memory_snapshot()
    test_memory_tracer_context()
    test_config_record_keep_masks()
    test_compact_memory_module_exists()
    test_streaming_sampler_module_exists()
    test_qwen_visual_chunk_module_exists()
    print("\nAll streaming/compact tests passed!")
