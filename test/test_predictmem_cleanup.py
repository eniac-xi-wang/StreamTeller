"""Tests for P0 cleanup — file structure, exports, no legacy symbols in mainline.

Covers:
  1. models/predictmem mainline file list exactly matches P0 spec.
  2. __init__.py does not export legacy symbols.
  3. Top-level legacy scripts are absent.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _p in (_repo_root, _models_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MAINLINE_FILES = {
    "__init__.py",
    "config.py",
    "streaming_memory.py",
    "vjepa_scorer.py",
    "vision_inputs.py",
    "token_pruner.py",
    "compact_memory.py",
    "streaming_sampler.py",
    "qwen_visual_chunk.py",
}

LEGACY_SYMBOLS = {
    "ScoreCache", "FramePlan", "build_frame_plan",
    "sample_video_1fps_decord", "DecordVideoSample", "TokenMapper",
}


def test_mainline_file_list():
    """predictmem dir must contain only the 6 mainline files."""
    pm_dir = Path(__file__).parent.parent / "models" / "predictmem"
    actual = {f.name for f in pm_dir.iterdir() if f.is_file() and f.suffix == ".py"}
    expected = MAINLINE_FILES

    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing mainline files: {missing}"
    assert not extra, f"Unexpected files in predictmem: {extra}"

    # Verify legacy dir exists
    legacy_dir = pm_dir / "legacy"
    assert legacy_dir.is_dir(), "Legacy directory must exist"
    legacy_files = list(legacy_dir.glob("*.py"))
    assert len(legacy_files) >= 3, f"Expected at least 3 legacy files, got {len(legacy_files)}"

    print(f"✓ predictmem mainline: {sorted(actual)}, legacy: {[f.name for f in legacy_files]}")


def test_no_legacy_exports():
    """__init__.py must not export legacy symbols."""
    import predictmem
    for sym in LEGACY_SYMBOLS:
        assert not hasattr(predictmem, sym), f"__init__.py should not export {sym}"
    print("✓ No legacy symbols exported")


def test_no_top_level_legacy_scripts():
    """Top-level scripts/ must not contain the old offline entry points."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    legacy_names = {
        "precompute_predictmem_scores.py",
        "smoke_qwen_predictmem.py",
        "visualize_predictmem_scores.py",
    }
    for name in legacy_names:
        path = scripts_dir / name
        assert not path.exists(), f"Top-level scripts/{name} must not exist (moved to legacy/)"

    # But they should exist in legacy
    legacy_dir = scripts_dir / "legacy"
    for name in legacy_names:
        path = legacy_dir / name
        assert path.exists(), f"Legacy script {name} must exist in scripts/legacy/"

    print(f"✓ No legacy scripts in scripts/, preserved in scripts/legacy/")


def test_mainline_exports():
    """Mainline exports are accessible."""
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
    print("✓ Mainline exports accessible")


if __name__ == "__main__":
    test_mainline_file_list()
    test_no_legacy_exports()
    test_no_top_level_legacy_scripts()
    test_mainline_exports()
    print("\n✅ All P0 cleanup tests passed!")
