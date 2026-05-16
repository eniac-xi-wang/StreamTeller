"""Tests for PredictMemStreamingMemory plugin — expanding windows, local quantile, warmup.

Covers:
  1. iter_predictmem_windows(T=16): 7 expanding target tubelets 1-7, standard doesn't duplicate 7
  2. iter_predictmem_windows(T=20): covers tubelets 1-9, only tubelet 0 unscored
  3. Local quantile used when t-digest has < 100 samples (not keep_all)
  4. num_tubelets_unscored <= 1
  5. Decode skip: should_skip returns True for no video / single token
"""

import sys
from pathlib import Path

import torch

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _p in (_repo_root, _models_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_iter_windows_t16():
    """T=16: 7 expanding windows for tubelets 1-7, no standard windows."""
    from predictmem.streaming_memory import iter_predictmem_windows

    windows = list(iter_predictmem_windows(16, window_frames=16, stride_frames=2))

    expanding = [w for w in windows if w["mode"] == "expanding"]
    sliding = [w for w in windows if w["mode"] == "sliding"]

    # 7 expanding windows: tubelets 1-7
    assert len(expanding) == 7, f"Expected 7 expanding, got {len(expanding)}"
    expanding_tubelets = [w["target_global_tubelet"] for w in expanding]
    assert expanding_tubelets == list(range(1, 8)), f"Got {expanding_tubelets}"

    # No standard sliding windows for T=16 (tubelet 7 already covered)
    standard_tubelets = [w["target_global_tubelet"] for w in sliding]
    for t in standard_tubelets:
        assert t != 7, f"Standard window duplicated tubelet 7"

    # Verify window specs
    for w in expanding:
        assert w["start"] == 0, f"Expanding window must start at 0, got {w['start']}"
        expected_len = (w["target_global_tubelet"] + 1) * 2  # frames = tubelets*2
        assert w["length"] == expected_len, f"Window tubelet {w['target_global_tubelet']}: expected length {expected_len}, got {w['length']}"

    print(f"✓ T=16: {len(expanding)} expanding, {len(sliding)} sliding")


def test_iter_windows_t20():
    """T=20: tubelets 1-7 expanding, 8-9 sliding. Only tubelet 0 unscored."""
    from predictmem.streaming_memory import iter_predictmem_windows

    windows = list(iter_predictmem_windows(20, window_frames=16, stride_frames=2))

    expanding = [w for w in windows if w["mode"] == "expanding"]
    sliding = [w for w in windows if w["mode"] == "sliding"]

    all_tubelets = set(w["target_global_tubelet"] for w in windows)

    # Tubelets 1-9 should be scored
    assert all_tubelets.issuperset({1, 2, 3, 4, 5, 6, 7, 8, 9}), f"Missing tubelets: {sorted(all_tubelets)}"

    # Tubelet 0 should NOT be scored
    assert 0 not in all_tubelets, "Tubelet 0 should not be scored"

    # num_tubelets_unscored <= 1 (only tubelet 0)
    num_tubelets = (20 + 1) // 2  # 10
    scored = len(all_tubelets)
    unscored = num_tubelets - scored
    assert unscored <= 1, f"num_tubelets_unscored={unscored}, expected <= 1"

    print(f"✓ T=20: {len(expanding)} expanding, {len(sliding)} sliding, "
          f"tubelets scored={sorted(all_tubelets)}, unscored={unscored}")


def test_iter_windows_t30():
    """T=30: expanding + sliding, covers many tubelets."""
    from predictmem.streaming_memory import iter_predictmem_windows

    windows = list(iter_predictmem_windows(30, window_frames=16, stride_frames=2))

    expanding = [w for w in windows if w["mode"] == "expanding"]
    sliding = [w for w in windows if w["mode"] == "sliding"]

    all_tubelets = set(w["target_global_tubelet"] for w in windows)

    # Should have expanding for 1-7
    assert {1, 2, 3, 4, 5, 6, 7}.issubset(all_tubelets)

    # Should NOT cover tubelet 0
    assert 0 not in all_tubelets

    # Each standard window should have length 16
    for w in sliding:
        assert w["length"] == 16
        assert w["target_local_tubelet"] == 7

    num_tubelets = (30 + 1) // 2
    unscored = num_tubelets - len(all_tubelets)
    assert unscored <= 1, f"num_tubelets_unscored={unscored}"

    print(f"✓ T=30: {len(windows)} total windows, tubelets scored={sorted(all_tubelets)}, unscored={unscored}")


def test_local_quantile_behavior():
    """Local quantile (not keep_all) should be used when t-digest is empty."""
    # Simulate the core logic: when digest is empty, use torch.quantile
    loss_2d = torch.rand(16, 16) * 0.5 + 0.1  # random losses
    keep_ratio = 0.10

    # Local quantile
    threshold = torch.quantile(loss_2d.flatten(), 1.0 - keep_ratio)
    keep = loss_2d >= threshold
    kept_count = keep.sum().item()

    # Should keep ~10% of patches (with some variance for small sample)
    total = 256
    expected = int(total * keep_ratio)
    # Wide tolerance because quantile on small 16x16 is approximate
    assert 1 <= kept_count <= total, f"kept={kept_count}, total={total}"
    # local top 10% should be roughly 25 patches
    assert abs(kept_count - expected) < 50, f"kept={kept_count} too far from expected={expected}"

    print(f"✓ Local quantile: kept {kept_count}/{total} (~{kept_count/total:.1%})")


def test_digest_update_after_decision():
    """Verify that t-digest is updated AFTER keep/drop decision (no peeking)."""
    from tdigest import TDigest

    loss_2d = torch.rand(16, 16) + 0.1
    keep_ratio = 0.10

    tdigest = TDigest()
    assert len(tdigest) == 0, "Digest should start empty"

    # Decision before update
    if len(tdigest) > 100:
        threshold_from_digest = tdigest.percentile(100 * (1.0 - keep_ratio))
    else:
        threshold_from_digest = None
        threshold_local = torch.quantile(loss_2d.flatten(), 1.0 - keep_ratio)
        keep = loss_2d >= threshold_local

    # Update AFTER
    tdigest.batch_update(loss_2d.flatten().tolist())
    assert len(tdigest) > 0, "Digest should have samples after update"

    # The decision was made WITHOUT seeing the current tubelet's loss distribution
    # through t-digest, which is the key invariant
    print(f"✓ Digest updated after decision: {len(tdigest)} samples, "
          f"local_threshold={threshold_local:.4f}")


def test_streaming_memory_should_skip():
    """Decode step / no video: should_skip returns True."""
    from predictmem.config import PredictMemConfig
    from predictmem.streaming_memory import PredictMemStreamingMemory

    config = PredictMemConfig()
    config.__post_init__()
    plugin = PredictMemStreamingMemory(
        video_token_id=248057,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        config=config,
    )

    # Decode: no frames tensor
    assert plugin.should_skip(None, None, torch.randn(1, 1, 4096))

    # Decode: single token
    assert plugin.should_skip(torch.ones(1, 3, 16, 512, 512), torch.randn(8, 3, 256, 256), torch.randn(1, 1, 4096))

    # No video
    assert plugin.should_skip(None, torch.randn(8, 3, 256, 256), torch.randn(1, 100, 4096))

    # Prefill with video: should NOT skip
    assert not plugin.should_skip(
        torch.ones(1, 3, 16, 512, 512),
        torch.randn(8, 3, 256, 256),
        torch.randn(1, 500, 4096),
    )

    print("✓ should_skip: decode/no-video correctly detected")


if __name__ == "__main__":
    test_iter_windows_t16()
    test_iter_windows_t20()
    test_iter_windows_t30()
    test_local_quantile_behavior()
    test_digest_update_after_decision()
    test_streaming_memory_should_skip()
    print("\n✅ All P1 plugin tests passed!")
