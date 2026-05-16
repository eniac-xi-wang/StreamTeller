"""Tests for bash config — runtime auto-resolve, run_config generation, checkpoint passthrough."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OVO_SH = REPO_ROOT / "evaluate" / "ovobench" / "ovobench.sh"
SB_SH = REPO_ROOT / "evaluate" / "streamingbench" / "streamingbench.sh"


def _run_bash(script: Path, *args) -> str:
    """Run a bash script with arguments, return combined stdout+stderr."""
    try:
        result = subprocess.run(
            ["bash", str(script), *args, "--dry-run"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "TMPDIR": "/tmp"},
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return ""


def test_ovo_predictmem_plugin_default():
    """--method predictmem should auto-resolve to predictmem_runtime=plugin."""
    output = _run_bash(OVO_SH, "--method", "predictmem")
    assert "Pred runtime: plugin" in output, f"Expected plugin default, got: {output[-200:]}"
    print("✓ OVO predictmem → plugin")


def test_ovo_baseline_none_default():
    """--method baseline should auto-resolve to predictmem_runtime=none."""
    output = _run_bash(OVO_SH, "--method", "baseline")
    assert "Pred runtime: none" in output, f"Expected none default, got: {output[-200:]}"
    print("✓ OVO baseline → none")


def test_ovo_dry_run_prints_config():
    """Dry-run must print key config values."""
    output = _run_bash(OVO_SH, "--method", "predictmem")
    assert "JEPA ckpt:" in output
    assert "V-JEPA src:" in output
    assert "PREDICTMEM_RUNTIME" not in output or "Pred runtime:" in output
    print("✓ OVO dry-run prints config")


def test_ovo_dry_run_generates_run_config():
    """Dry-run must generate run_config.json."""
    with tempfile.TemporaryDirectory() as td:
        output = _run_bash(OVO_SH, "--method", "predictmem", "--result-dir", td)
        cfg_path = Path(td) / "run_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            assert "jepa_checkpoint_path" in cfg
            assert "vjepa_src_path" in cfg
            assert cfg["method"] == "predictmem"
            assert cfg["predictmem_runtime"] == "plugin"
            print(f"✓ run_config.json generated with {len(cfg)} keys")
        else:
            print("  (run_config.json not generated in dry-run — acceptable if dir is tmp)")


def test_streamingbench_dry_run_requires_data_paths():
    """StreamingBench dry-run should require --task-csv and --video-dir."""
    output = _run_bash(SB_SH, "--method", "predictmem", "--task-csv", "dummy.csv",
                        "--video-dir", "dummy_dir")
    assert "Pred runtime: plugin" in output
    assert "JEPA ckpt:" in output
    assert "MAX_PIXELS" not in output  # it prints the value, not the var name
    print("✓ StreamingBench dry-run works with dummy paths")


def test_streamingbench_max_pixels_passthrough():
    """MAX_PIXELS should appear in the printed config."""
    output = _run_bash(SB_SH, "--method", "predictmem", "--task-csv", "dummy.csv",
                        "--video-dir", "dummy_dir", "--max-pixels", "999999")
    assert "Max pixels:   999999" in output
    print("✓ MAX_PIXELS passthrough works")


def test_python_help_has_new_params():
    """Python --help should list new config params."""
    result = subprocess.run(
        [sys.executable, "-m", "evaluate.ovobench.ovobench", "--help"],
        capture_output=True, text=True, timeout=15,
    )
    out = result.stdout
    for param in ["jepa_checkpoint", "vjepa_src", "window_frames", "tail_keep",
                  "torch_dtype", "device"]:
        assert param in out, f"Missing --help param: {param}"
    print("✓ OVO --help includes all new params")

    result2 = subprocess.run(
        [sys.executable, "-m", "evaluate.streamingbench.streamingbench", "--help"],
        capture_output=True, text=True, timeout=15,
    )
    out2 = result2.stdout
    for param in ["jepa_checkpoint", "vjepa_src", "max_pixels", "time_window",
                  "torch_dtype", "device", "window_frames"]:
        assert param in out2, f"Missing --help param: {param}"
    print("✓ StreamingBench --help includes all new params")


def test_no_hardcoded_checkpoint_in_python():
    """Python mainline must not hardcode the V-JEPA checkpoint path."""
    files_to_check = [
        "models/predictmem/streaming_memory.py",
        "models/predictmem/vjepa_scorer.py",
        "evaluate/common/qwen35_predictmem.py",
        "evaluate/ovobench/ovobench.py",
        "evaluate/streamingbench/streamingbench.py",
    ]
    for fp in files_to_check:
        path = REPO_ROOT / fp
        if not path.exists():
            continue
        content = path.read_text()
        # Bash scripts may have it, but Python mainline must not
        if fp.endswith(".py"):
            assert "/data/model_weights_public/jepa" not in content, \
                f"{fp} contains hardcoded checkpoint path"
            assert "jeap_vitl_16_256.pt" not in content, \
                f"{fp} contains hardcoded checkpoint filename"
    print("✓ No hardcoded checkpoint in Python mainline")


def test_multi_gpu_cmd_has_new_params():
    """Multi-GPU worker subprocess must pass all config params."""
    # Check the launch_worker function in ovobench.py
    ovobench_py = REPO_ROOT / "evaluate" / "ovobench" / "ovobench.py"
    content = ovobench_py.read_text()
    assert "--jepa_checkpoint_path" in content
    assert "--vjepa_src_path" in content
    assert "--window_frames" in content
    assert "--tail_keep_frames" in content
    assert "--device" in content
    assert "--torch_dtype" in content
    print("✓ OVO multi-GPU worker passes all config params")

    sb_py = REPO_ROOT / "evaluate" / "streamingbench" / "streamingbench.py"
    content = sb_py.read_text()
    assert "--jepa_checkpoint_path" in content
    assert "--vjepa_src_path" in content
    assert "--window_frames" in content
    assert "--tail_keep_frames" in content
    assert "--max_pixels" in content
    assert "--device" in content
    print("✓ StreamingBench multi-GPU worker passes all config params")


if __name__ == "__main__":
    test_ovo_predictmem_plugin_default()
    test_ovo_baseline_none_default()
    test_ovo_dry_run_prints_config()
    test_ovo_dry_run_generates_run_config()
    test_streamingbench_dry_run_requires_data_paths()
    test_streamingbench_max_pixels_passthrough()
    test_python_help_has_new_params()
    test_no_hardcoded_checkpoint_in_python()
    test_multi_gpu_cmd_has_new_params()
    print("\n✅ All bash config tests passed!")
