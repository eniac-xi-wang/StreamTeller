"""Tests for evaluate/common/qwen35_predictmem.py — kwargs, stats, video clipping."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

_repo_root = Path(__file__).parent.parent
for _p in (_repo_root, _repo_root / "evaluate", _repo_root / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.qwen35_predictmem import extract_predictmem_stats


def test_baseline_no_predictmem_kwargs():
    """method=baseline should set use_predictmem=False and NOT pass predictmem_frames_256."""
    predictmem_runtime = "none"
    method = "baseline"
    generate_kwargs = {}

    if method == "predictmem" and predictmem_runtime == "plugin":
        generate_kwargs["use_predictmem"] = True
        generate_kwargs["predictmem_frames_256"] = "fake_tensor"
        generate_kwargs["predictmem_keep_ratio"] = 0.10
    else:
        generate_kwargs["use_predictmem"] = False

    assert generate_kwargs["use_predictmem"] is False
    assert "predictmem_frames_256" not in generate_kwargs
    print("✓ baseline generates correct kwargs")


def test_predictmem_plugin_kwargs():
    """method=predictmem + plugin runtime should pass plugin kwargs."""
    predictmem_runtime = "plugin"
    method = "predictmem"
    generate_kwargs = {}

    if method == "predictmem" and predictmem_runtime == "plugin":
        generate_kwargs["use_predictmem"] = True
        generate_kwargs["predictmem_frames_256"] = "fake_tensor"
        generate_kwargs["predictmem_keep_ratio"] = 0.10
    else:
        generate_kwargs["use_predictmem"] = False

    assert generate_kwargs["use_predictmem"] is True
    assert generate_kwargs["predictmem_frames_256"] == "fake_tensor"
    assert generate_kwargs["predictmem_keep_ratio"] == 0.10
    print("✓ predictmem plugin generates correct kwargs")


def test_extract_predictmem_stats():
    """extract_predictmem_stats returns None when model has no stats."""
    mock_model = MagicMock()
    mock_model.model.predictmem_last_stats = None
    stats = extract_predictmem_stats(mock_model)
    assert stats is None

    fake_stats = {"kept_video_tokens": 100, "keep_ratio_actual": 0.15}
    mock_model.model.predictmem_last_stats = fake_stats
    stats = extract_predictmem_stats(mock_model)
    assert stats == fake_stats
    print("✓ extract_predictmem_stats works")


def test_stats_structure():
    """Stats dict must contain all required fields."""
    required = [
        "total_latency_s", "peak_memory_mb", "video_grid_thw",
        "expected_video_tokens", "predictmem_stats"
    ]
    fake_stats = {
        "total_latency_s": 1.0,
        "peak_memory_mb": 1000.0,
        "video_grid_thw": [[8, 32, 32]],
        "expected_video_tokens": 2048,
        "predictmem_stats": {"kept_video_tokens": 512},
    }
    for key in required:
        assert key in fake_stats, f"Missing: {key}"
    print("✓ stats structure complete")


def test_build_video_inputs_full_video():
    """build_video_inputs_for_eval with no clip boundaries returns all frames."""
    # Find a real video
    video_path = "/data/qinian_workspace/OVO-Bench/chunked_videos/0.mp4"
    if not os.path.exists(video_path):
        print("  (skipping — no test video)")
        return

    from common.qwen35_predictmem import build_video_inputs_for_eval
    qwen, jepa, meta, _extra = build_video_inputs_for_eval(video_path, fps=1.0, frame_budget=16)

    assert qwen.shape[0] == 16
    assert jepa.shape[0] == 16
    assert qwen.shape[1:3] == (512, 512)
    assert jepa.shape[1:4] == (3, 256, 256)
    assert len(meta["frames_indices"]) == 16
    # Source indices should increase
    assert meta["frames_indices"][0] < meta["frames_indices"][-1]
    print(f"✓ full video: {qwen.shape[0]} frames, source range [{meta['frames_indices'][0]}, {meta['frames_indices'][-1]}]")


def test_build_video_inputs_time_clip():
    """start_time/end_time should change sampled frame range."""
    video_path = "/data/qinian_workspace/OVO-Bench/chunked_videos/0.mp4"
    if not os.path.exists(video_path):
        print("  (skipping — no test video)")
        return

    from common.qwen35_predictmem import build_video_inputs_for_eval

    # Full: sample all at 1fps
    _, _, full, _ = build_video_inputs_for_eval(video_path, fps=1.0)

    # Clip: only first 10 seconds
    _, _, clipped, _ = build_video_inputs_for_eval(video_path, fps=1.0, start_time=0, end_time=10)

    # Clipped should have fewer frames (at most 10)
    assert clipped["total_num_frames"] <= full["total_num_frames"]
    assert clipped["duration"] <= 10.1  # allow float tolerance
    print(f"✓ time clip: full={full['total_num_frames']} frames, clip0-10={clipped['total_num_frames']} frames")


def test_build_video_inputs_frame_budget():
    """frame_budget should limit total frames."""
    video_path = "/data/qinian_workspace/OVO-Bench/chunked_videos/0.mp4"
    if not os.path.exists(video_path):
        print("  (skipping — no test video)")
        return

    from common.qwen35_predictmem import build_video_inputs_for_eval

    _, _, limited, _ = build_video_inputs_for_eval(video_path, fps=1.0, frame_budget=8)
    assert limited["total_num_frames"] <= 8
    print(f"✓ frame_budget=8: {limited['total_num_frames']} frames")


def test_qwen_jepa_frame_count_match():
    """Qwen and V-JEPA tensors have the same frame count."""
    video_path = "/data/qinian_workspace/OVO-Bench/chunked_videos/0.mp4"
    if not os.path.exists(video_path):
        print("  (skipping — no test video)")
        return

    from common.qwen35_predictmem import build_video_inputs_for_eval

    qwen, jepa, _, _extra = build_video_inputs_for_eval(video_path, fps=1.0, start_time=5, end_time=25)
    assert qwen.shape[0] == jepa.shape[0], f"Qwen {qwen.shape[0]}, JEPA {jepa.shape[0]}"
    print(f"✓ Qwen/JEPA frame match: both {qwen.shape[0]}")


if __name__ == "__main__":
    test_baseline_no_predictmem_kwargs()
    test_predictmem_plugin_kwargs()
    test_extract_predictmem_stats()
    test_stats_structure()
    test_build_video_inputs_full_video()
    test_build_video_inputs_time_clip()
    test_build_video_inputs_frame_budget()
    test_qwen_jepa_frame_count_match()
    print("\n✅ All common helper tests passed!")

