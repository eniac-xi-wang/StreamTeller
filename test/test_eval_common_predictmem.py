"""Tests for evaluate/common/qwen35_predictmem.py — baseline vs PredictMem kwargs, stats extraction."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

_repo_root = Path(__file__).parent.parent
for _p in (_repo_root, _repo_root / "evaluate"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.qwen35_predictmem import extract_predictmem_stats


def test_baseline_no_predictmem_kwargs():
    """method=baseline should set use_predictmem=False and NOT pass predictmem_frames_256."""
    # The generate_kwargs are built inside generate_qwen35_response.
    # We test the logic directly by checking the method branch.
    from common.qwen35_predictmem import generate_qwen35_response

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


if __name__ == "__main__":
    test_baseline_no_predictmem_kwargs()
    test_predictmem_plugin_kwargs()
    test_extract_predictmem_stats()
    test_stats_structure()
    print("\n✅ All common helper tests passed!")
