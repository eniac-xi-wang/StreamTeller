"""Tests for evaluate/ovobench/ — prompts, answer parsing, path resolution, scoring."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_repo_root = Path(__file__).parent.parent
for _p in (_repo_root, _repo_root / "evaluate"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_build_prompt_mc():
    """Backward+Realtime tasks produce A-D multiple choice prompts."""
    from evaluate.ovobench.ovobench import build_prompt

    prompt = build_prompt("EPM", "What did I do?", ["eat", "sleep", "run", "walk"])
    assert "A. eat" in prompt
    assert "B. sleep" in prompt
    assert "Respond only with the letter" in prompt
    print("✓ MC prompt built correctly")


def test_build_prompt_rec():
    """REC task produces count prompt."""
    from evaluate.ovobench.ovobench import build_prompt

    anno = {"activity": "jumping"}
    prompt = build_prompt("REC", "", [], anno=anno)
    assert "jumping" in prompt
    assert "Respond only with a number" in prompt
    print("✓ REC prompt built correctly")


def test_build_prompt_ssr():
    """SSR task produces step verification prompt."""
    from evaluate.ovobench.ovobench import build_prompt

    anno = {"test_info": [{"step": "chop vegetables"}]}
    prompt = build_prompt("SSR", "", [], anno=anno, index=0)
    assert "chop vegetables" in prompt
    assert "Yes or No" in prompt or "Yes" in prompt
    print("✓ SSR prompt built correctly")


def test_resolve_video_path():
    """Video path resolver handles different directory layouts."""
    from evaluate.ovobench.ovobench import resolve_video_path

    # chunked_videos dir
    p = resolve_video_path("/data/OVO/chunked_videos", 42)
    assert str(p).endswith("42.mp4")

    # OVO root dir (no chunked_videos)
    p = resolve_video_path("/data/OVO", 42)
    assert "chunked_videos" in str(p)
    assert str(p).endswith("42.mp4")

    # Forward task with index
    p = resolve_video_path("/data/OVO/chunked_videos", 42, index=1)
    assert str(p).endswith("42_1.mp4")

    print("✓ video path resolver works")


def test_score_merged_output():
    """score.py can process merged results and produce score JSON."""
    from evaluate.ovobench.score import score_all

    merged = {
        "EPM": [
            {"response": "A", "ground_truth": "A"},
            {"response": "B", "ground_truth": "C"},
        ],
        "ASI": [
            {"response": "B", "ground_truth": "B"},
        ],
    }
    scores = score_all(merged)
    assert scores["backward"]["tasks"]["EPM"] == 50.0
    assert scores["backward"]["tasks"]["ASI"] == 100.0
    assert scores["backward"]["average"] == 75.0
    print("✓ score_all computes correct accuracy")


def test_score_rec_task():
    """REC scoring matches count."""
    from evaluate.ovobench.score import score_all

    merged = {
        "REC": [{
            "test_info": [
                {"response": "3", "count": 3},
                {"response": "three", "count": 3},
                {"response": "5", "count": 3},
            ]
        }],
    }
    scores = score_all(merged)
    assert scores["forward"]["tasks"]["REC"] > 0
    print("✓ REC scoring works")


def test_score_ssr_task():
    """SSR scoring handles type-based evaluation."""
    from evaluate.ovobench.score import score_all

    merged = {
        "SSR": [{
            "test_info": [
                {"response": "Y", "type": 1},  # correct
                {"response": "Yes", "type": 1},  # correct (contains "yes")
                {"response": "N", "type": 1},  # wrong
            ]
        }],
    }
    scores = score_all(merged)
    acc = scores["forward"]["tasks"]["SSR"]
    assert acc > 0
    print(f"✓ SSR scoring: {acc:.1f}% (expect > 0)")


def test_no_evaluation_hardcoded():
    """No executable code should reference evaluation/ (only docstrings)."""
    files_to_check = [
        "evaluate/ovobench/ovobench.py",
        "evaluate/ovobench/ovobench.sh",
        "evaluate/ovobench/score.py",
        "evaluate/streamingbench/streamingbench.py",
        "evaluate/streamingbench/streamingbench.sh",
        "evaluate/streamingbench/score.py",
    ]
    for fp in files_to_check:
        path = Path(__file__).parent.parent / fp
        if not path.exists():
            continue
        content = path.read_text()
        lines = [l for l in content.split("\n") if "evaluation/" in l
                 and not l.strip().startswith("#")
                 and not l.strip().startswith('"""')
                 and "Reference:" not in l
                 and "site-packages" not in l]
        assert len(lines) == 0, f"{fp} contains hardcoded evaluation/ path: {lines}"
    print("✓ no evaluation/ hardcoded paths")


if __name__ == "__main__":
    test_build_prompt_mc()
    test_build_prompt_rec()
    test_build_prompt_ssr()
    test_resolve_video_path()
    test_score_merged_output()
    test_score_rec_task()
    test_score_ssr_task()
    test_no_evaluation_hardcoded()
    print("\n✅ All OVO-Bench tests passed!")
