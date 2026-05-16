"""Tests for P1 boundary token policy: tubelet 0 drop, tail full keep."""

import sys
from pathlib import Path

import torch

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _p in (_repo_root, _models_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_tubelet_0_dropped():
    """Tubelet 0 keep mask must be all False."""
    from predictmem.streaming_memory import iter_predictmem_windows

    # iter_predictmem_windows never yields tubelet 0
    for T in [16, 20, 30]:
        windows = list(iter_predictmem_windows(T))
        scored_tubelets = {w["target_global_tubelet"] for w in windows}
        assert 0 not in scored_tubelets, f"T={T}: tubelet 0 should not be scored"

    print("✓ Tubelet 0 excluded from scoring windows")


def test_tail_tubelets_skipped():
    """Tail tubelets should be skipped by iter_predictmem_windows."""
    from predictmem.streaming_memory import iter_predictmem_windows

    T = 30
    num_tubelets = (T + 1) // 2
    tail_keep = {num_tubelets - 2, num_tubelets - 1}  # tubelets 13, 14

    windows = list(iter_predictmem_windows(T, tail_keep_tubelets=tail_keep))
    scored_tubelets = {w["target_global_tubelet"] for w in windows}

    for t in tail_keep:
        assert t not in scored_tubelets, f"Tail tubelet {t} should be skipped"

    print(f"✓ Tail tubelets {tail_keep} excluded, scored={sorted(scored_tubelets)}")


def test_boundary_sets_computation():
    """Verify num_tubelets, tail_keep sets for various T values."""
    for T in [16, 20, 30, 215]:
        num_tubelets = (T + 1) // 2
        dropped = {0}
        tail_keep = {num_tubelets - 2, num_tubelets - 1} & set(range(num_tubelets))

        assert 0 in dropped
        assert len(tail_keep) <= 2
        for t in tail_keep:
            assert t >= 0

        if T >= 16:
            assert len(tail_keep) == 2, f"T={T}: expected 2 tail tubelets, got {len(tail_keep)}"

        tail_frames = []
        for t in tail_keep:
            tail_frames.extend([t * 2, t * 2 + 1])
        tail_frames = sorted(set(tail_frames) & set(range(T)))
        # Tail covers last 4 frames, but can be 3 if total frames is odd
        expected_tail = min(4, T)
        if T % 2 == 0:
            assert len(tail_frames) == 4, f"T={T}: expected 4 tail frames, got {len(tail_frames)}"
        else:
            assert len(tail_frames) in (3, 4), f"T={T}: expected 3-4 tail frames, got {len(tail_frames)}"

        print(f"  T={T}: tubelets={num_tubelets}, dropped={dropped}, tail_keep={tail_keep}, "
              f"tail_frames={tail_frames}")


def test_scored_tubelets_exclude_boundaries():
    """scored_tubelets must NOT include 0 or tail tubelets."""
    from predictmem.streaming_memory import iter_predictmem_windows

    T = 30
    num_tubelets = (T + 1) // 2  # 15
    dropped = {0}
    tail_keep = {num_tubelets - 2, num_tubelets - 1}  # {13, 14}
    skip = dropped | tail_keep

    windows = list(iter_predictmem_windows(T, tail_keep_tubelets=skip))
    scored = {w["target_global_tubelet"] for w in windows}

    assert 0 not in scored, "tubelet 0 must not be scored"
    assert 13 not in scored, "tail tubelet 13 must not be scored"
    assert 14 not in scored, "tail tubelet 14 must not be scored"

    # All other tubelets 1-12 should be scored
    for t in range(1, 13):
        assert t in scored, f"tubelet {t} should be scored"

    print(f"✓ scored_tubelets={sorted(scored)} (excl 0, 13, 14)")


def test_early_tubelets_scored():
    """Early tubelets 1-7 must be scored via expanding windows."""
    from predictmem.streaming_memory import iter_predictmem_windows

    T = 40
    num_tubelets = (T + 1) // 2
    tail_keep = {num_tubelets - 2, num_tubelets - 1}
    skip = {0} | tail_keep

    windows = list(iter_predictmem_windows(T, tail_keep_tubelets=skip))
    scored = {w["target_global_tubelet"] for w in windows}

    early_expected = {1, 2, 3, 4, 5, 6, 7}
    assert early_expected.issubset(scored), f"Missing early tubelets: {early_expected - scored}"

    # Check expanding windows
    expanding = [w for w in windows if w["mode"] == "expanding"]
    expanding_tubelets = {w["target_global_tubelet"] for w in expanding}
    assert expanding_tubelets == {1, 2, 3, 4, 5, 6, 7}, f"Expanding windows wrong: {expanding_tubelets}"

    print(f"✓ expanding windows cover {expanding_tubelets}")


if __name__ == "__main__":
    test_tubelet_0_dropped()
    test_tail_tubelets_skipped()
    test_boundary_sets_computation()
    test_scored_tubelets_exclude_boundaries()
    test_early_tubelets_scored()
    print("\n✅ All P1 boundary policy tests passed!")
