"""Tests for evaluate/streamingbench/ — timestamp conversion, answer extraction, scoring."""

import json
import sys
import tempfile
from pathlib import Path

_repo_root = Path(__file__).parent.parent
for _p in (_repo_root, _repo_root / "evaluate"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_time_to_seconds():
    """MM:SS and HH:MM:SS conversion."""
    from evaluate.streamingbench.streamingbench import _time_to_seconds

    assert _time_to_seconds("01:30") == 90
    assert _time_to_seconds("00:00") == 0
    assert _time_to_seconds("1:00:00") == 3600
    assert _time_to_seconds("0:05:30") == 330
    print("✓ _time_to_seconds handles all formats")


def test_extract_answer():
    """Answer extraction from various response formats."""
    from evaluate.streamingbench.streamingbench import _extract_answer

    assert _extract_answer("A") == "A"
    assert _extract_answer("The answer is B") == "B"
    assert _extract_answer("option C is the best") == "C"
    assert _extract_answer("I think (D)") == "D"
    assert _extract_answer("Option A is correct") == "A"
    assert _extract_answer("B is the best option") == "B"
    print("✓ _extract_answer handles all patterns")


def test_format_prompt_with_options():
    """Prompt formatting with multiple-choice options."""
    from evaluate.streamingbench.streamingbench import _format_prompt

    prompt = _format_prompt("What color?", "['red', 'blue', 'green', 'yellow']")
    assert "A. red" in prompt
    assert "B. blue" in prompt
    assert "C. green" in prompt
    assert "D. yellow" in prompt
    assert "Respond with only the letter" in prompt
    print("✓ _format_prompt with options works")


def test_format_prompt_without_options():
    """Prompt formatting for open-ended questions."""
    from evaluate.streamingbench.streamingbench import _format_prompt

    prompt = _format_prompt("Describe the scene.", "nan")
    assert "Describe the scene." in prompt
    assert "options:" not in prompt.lower() or "Options:" not in prompt
    print("✓ _format_prompt without options works")


def test_score_from_jsonl():
    """score.py can process fake JSONL results."""
    from evaluate.streamingbench.score import calculate_scores

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({
            "question_id": "q_001_123",
            "task_type": "Real-time Visual Understanding",
            "question": "What is shown?",
            "answer": "A",
            "predicted_answer": "A",
            "response": "A",
            "has_options": True,
            "correct": True,
            "total_latency_s": 5.0,
            "peak_memory_mb": 10000,
        }) + "\n")
        f.write(json.dumps({
            "question_id": "q_002_456",
            "task_type": "Real-time Visual Understanding",
            "question": "Who is shown?",
            "answer": "B",
            "predicted_answer": "C",
            "response": "C",
            "has_options": True,
            "correct": False,
            "total_latency_s": 4.0,
            "peak_memory_mb": 9000,
        }) + "\n")
        tmp_path = f.name

    try:
        scores = calculate_scores(Path(tmp_path))
        assert "overall" in scores["scores"]
        assert scores["scores"]["overall"]["accuracy"] == 50.0
        assert scores["scores"]["overall"]["total"] == 2
        assert scores["scores"]["overall"]["correct"] == 1
        print("✓ score calculation correct")
    finally:
        Path(tmp_path).unlink()


def test_resolve_video_path():
    """Video path follows StreamingBench convention."""
    from evaluate.streamingbench.streamingbench import resolve_video_path

    p = resolve_video_path("/data/streamingbench/data/real", "42")
    assert p.endswith("sample_42/video.mp4")
    print("✓ video path resolver works")


def test_no_evaluation_hardcoded():
    """No executable code should reference evaluation/ (only docstrings)."""
    files_to_check = [
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
    test_time_to_seconds()
    test_extract_answer()
    test_format_prompt_with_options()
    test_format_prompt_without_options()
    test_score_from_jsonl()
    test_resolve_video_path()
    test_no_evaluation_hardcoded()
    print("\n✅ All StreamingBench tests passed!")
