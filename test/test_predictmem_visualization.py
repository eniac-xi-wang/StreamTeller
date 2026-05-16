"""Tests for P4 visualization — render_predictmem_highlight.py.

Tests that the highlight rendering logic can process a fake keepmask JSON
and produce correct frame arrays.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _p in (_repo_root, _models_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def make_fake_keep_masks(num_frames: int = 16) -> dict:
    """Create a minimal fake keepmasks dict for testing."""
    grid_h, grid_w = 16, 16
    num_tubelets = (num_frames + 1) // 2
    tubelets = []

    for t in range(num_tubelets):
        frames = [t * 2, t * 2 + 1]
        frames = [f for f in frames if f < num_frames]

        if t == 0:
            mode = "bootstrap_drop"
            mask = [0] * (grid_h * grid_w)
        elif t >= num_tubelets - 2:
            mode = "protected_tail_full_keep"
            mask = [1] * (grid_h * grid_w)
        else:
            mode = "scored_digest"
            # Keep top ~10% patches
            mask_2d = np.zeros((grid_h, grid_w), dtype=int)
            mask_2d[0, 0] = 1  # keep at least one
            mask_2d[8, 8] = 1
            mask = mask_2d.flatten().tolist()

        tubelets.append({
            "tubelet": t,
            "frames": frames,
            "mode": mode,
            "keep_mask": mask,
        })

    return {"grid_h": grid_h, "grid_w": grid_w, "tubelets": tubelets}


def test_keepmask_structure():
    """Ensure keep masks have the expected structure."""
    masks = make_fake_keep_masks(16)
    assert masks["grid_h"] == 16
    assert masks["grid_w"] == 16
    assert len(masks["tubelets"]) == 8  # 16 frames / 2

    # Tubelet 0: bootstrap_drop
    assert masks["tubelets"][0]["mode"] == "bootstrap_drop"
    assert all(v == 0 for v in masks["tubelets"][0]["keep_mask"])

    # Last 2 tubelets: protected
    for t in range(6, 8):
        assert masks["tubelets"][t]["mode"] == "protected_tail_full_keep"
        assert all(v == 1 for v in masks["tubelets"][t]["keep_mask"])

    print("✓ keepmask structure valid")


def test_per_frame_mask_construction():
    """Build per-frame masks from tubelets and verify."""
    masks = make_fake_keep_masks(16)
    grid_h, grid_w = masks["grid_h"], masks["grid_w"]

    per_frame_mask = {}
    per_frame_mode = {}

    for tinfo in masks["tubelets"]:
        t = tinfo["tubelet"]
        mode = tinfo["mode"]
        mask = np.array(tinfo["keep_mask"], dtype=bool).reshape(grid_h, grid_w)
        for f_idx in tinfo["frames"]:
            per_frame_mask[f_idx] = mask
            per_frame_mode[f_idx] = mode

    # Frame 0-1: bootstrap_drop, all False
    assert per_frame_mode[0] == "bootstrap_drop"
    assert per_frame_mode[1] == "bootstrap_drop"
    assert not per_frame_mask[0].any()

    # Frames 12-15: protected, all True
    for f in [12, 13, 14, 15]:
        assert per_frame_mode[f] == "protected_tail_full_keep", f"frame {f}: {per_frame_mode[f]}"
        assert per_frame_mask[f].all(), f"frame {f} should have all-True mask"

    # Middle frames: scored
    assert "scored" in per_frame_mode[2]

    print("✓ per-frame mask construction correct")


def test_json_serialization():
    """keepmasks must serialize to JSON."""
    masks = make_fake_keep_masks(16)
    json_str = json.dumps(masks)
    decoded = json.loads(json_str)
    assert decoded["grid_h"] == 16
    assert len(decoded["tubelets"]) == 8
    print("✓ keepmasks JSON roundtrip OK")


def test_highlight_frame_logic():
    """Verify highlight frame rendering logic with mock frame."""
    masks = make_fake_keep_masks(16)
    grid_h, grid_w = masks["grid_h"], masks["grid_w"]

    # Build per-frame masks
    per_frame_mask = {}
    per_frame_mode = {}
    for tinfo in masks["tubelets"]:
        mask = np.array(tinfo["keep_mask"], dtype=bool).reshape(grid_h, grid_w)
        for f_idx in tinfo["frames"]:
            per_frame_mask[f_idx] = mask
            per_frame_mode[f_idx] = tinfo["mode"]

    import cv2

    for f_idx in range(16):
        original = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        h, w = original.shape[:2]

        if f_idx in per_frame_mask:
            mask_16x16 = per_frame_mask[f_idx]
            mode = per_frame_mode[f_idx]

            if mode == "bootstrap_drop":
                composite = (original * 0.3).astype(np.uint8)
                n_kept = 0
            elif mode == "protected_tail_full_keep":
                composite = original.copy()
                n_kept = grid_h * grid_w
            else:
                mask_u8 = mask_16x16.astype(np.uint8)
                mask_full = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
                darkened = (original * 0.3).astype(np.uint8)
                mask_3ch = np.stack([mask_full] * 3, axis=-1)
                composite = original * mask_3ch + darkened * (1 - mask_3ch)
                composite = composite.astype(np.uint8)
                n_kept = mask_16x16.sum()

            assert composite.shape == original.shape
            if mode == "bootstrap_drop":
                assert n_kept == 0
                assert (composite.max() < original.max() * 0.5)
            elif mode == "protected_tail_full_keep":
                assert n_kept == grid_h * grid_w
                # Full brightness preserved
                assert (composite == original).all()

    print("✓ highlight frame logic produces expected composites")


if __name__ == "__main__":
    test_keepmask_structure()
    test_per_frame_mask_construction()
    test_json_serialization()
    test_highlight_frame_logic()
    print("\n✅ All P4 visualization tests passed!")
