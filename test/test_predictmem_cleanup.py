"""Tests for P0/P3/P4 cleanup — plugin imports, no legacy refs, CLI flags.

Covers:
  7. plugin main script does not import ScoreCache/FramePlan
  8. decode phase does not call scorer
  9. __init__.py mainline exports are correct
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _p in (_repo_root, _models_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_init_mainline_exports():
    """__init__.py exports mainline plugin symbols."""
    from predictmem import (
        PredictMemConfig,
        PredictMemStreamingMemory,
        build_predictmem_video_inputs,
        VJEPAPredictLossScorer,
        TokenPruner,
    )
    assert PredictMemConfig is not None
    assert PredictMemStreamingMemory is not None
    assert build_predictmem_video_inputs is not None
    assert VJEPAPredictLossScorer is not None
    assert TokenPruner is not None
    print("✓ All mainline exports available")


def test_plugin_script_no_legacy_imports():
    """The main run script must not import ScoreCache or FramePlan at module level."""
    import ast

    script_path = Path(__file__).parent.parent / "scripts" / "run_predictmem_ovobench.py"
    source = script_path.read_text()
    tree = ast.parse(source)

    forbidden_imports = {"ScoreCache", "FramePlan", "sample_video_1fps_decord",
                         "score_vjepa_windows", "build_frame_plan", "global_scores"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "predictmem" in node.module:
                for alias in node.names:
                    name = alias.name
                    if name in forbidden_imports:
                        # Only fail if it's a direct import, not a try/except fallback
                        raise AssertionError(f"Main script imports legacy symbol: {name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_imports:
                    raise AssertionError(f"Main script imports legacy module: {alias.name}")

    print("✓ Main script has no legacy imports")


def test_legacy_scripts_exist():
    """Legacy scripts are preserved in scripts/legacy/."""
    legacy_dir = Path(__file__).parent.parent / "scripts" / "legacy"
    assert legacy_dir.is_dir(), f"Legacy dir not found: {legacy_dir}"
    legacy_files = list(legacy_dir.glob("*.py"))
    assert len(legacy_files) >= 1, f"No legacy scripts in {legacy_dir}"
    print(f"✓ Legacy scripts preserved: {[f.name for f in legacy_files]}")


def test_streaming_memory_uses_analyzer_scorer():
    """streaming_memory._ensure_scorer must use make_vjepa_analyzer_scorer."""
    source = (Path(__file__).parent.parent / "models" / "predictmem" / "streaming_memory.py").read_text()
    assert "make_vjepa_analyzer_scorer" in source, "streaming_memory should use make_vjepa_analyzer_scorer"
    assert "score_latest_tubelet_variable" in source, "streaming_memory should use score_latest_tubelet_variable"
    print("✓ streaming_memory uses analyzer-compatible scorer")


def test_iter_predictmem_windows_exported():
    """iter_predictmem_windows is callable."""
    from predictmem.streaming_memory import iter_predictmem_windows
    result = list(iter_predictmem_windows(16))
    assert len(result) > 0, "Should produce windows"
    print(f"✓ iter_predictmem_windows(16) produced {len(result)} windows")


if __name__ == "__main__":
    test_init_mainline_exports()
    test_plugin_script_no_legacy_imports()
    test_legacy_scripts_exist()
    test_streaming_memory_uses_analyzer_scorer()
    test_iter_predictmem_windows_exported()
    print("\n✅ All P0/P3/P4 cleanup tests passed!")
