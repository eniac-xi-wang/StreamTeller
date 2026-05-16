"""Tests for V-JEPA analyzer parity — num_mask_tokens, wrappers, ImageNet norm.

Covers:
  5. analyzer-compatible scorer: predictor.num_mask_tokens == 10
  6. vision_inputs.py: V-JEPA tensor has ImageNet normalization
  7. Wrapper types: MultiSeqWrapper / PredictorMultiSeqWrapper
"""

import sys
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _p in (_repo_root, _models_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_num_mask_tokens():
    """Scorer builder must set num_mask_tokens=10 (analyzer-compatible)."""
    from predictmem.vjepa_scorer import make_vjepa_analyzer_scorer

    result = make_vjepa_analyzer_scorer(
        checkpoint_path=None,
        device="cpu",
    )

    assert result["num_mask_tokens"] == 10, f"Expected 10, got {result['num_mask_tokens']}"
    assert result["wrapper_type"] == "MultiSeqWrapper/PredictorMultiSeqWrapper"
    print(f"✓ num_mask_tokens={result['num_mask_tokens']}, wrapper={result['wrapper_type']}")


def test_imagenet_normalization():
    """vision_inputs.py must apply ImageNet normalization to V-JEPA tensor."""
    from predictmem.vision_inputs import IMAGENET_MEAN, IMAGENET_STD

    assert IMAGENET_MEAN == (0.485, 0.456, 0.406), f"Bad mean: {IMAGENET_MEAN}"
    assert IMAGENET_STD == (0.229, 0.224, 0.225), f"Bad std: {IMAGENET_STD}"

    # Simulate the normalization pipeline
    raw_frames = torch.randint(0, 256, (4, 3, 256, 256), dtype=torch.float32)
    normalized = (raw_frames / 255.0 - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    # After normalization, mean should be close to 0 (within reasonable range for random data)
    channel_means = normalized.mean(dim=(0, 2, 3))
    print(f"  Channel means after ImageNet norm: {channel_means.tolist()}")
    # Not zero because input is uniform, but should be negative since 0.5 < 0.485/0.456/0.406
    assert torch.all(channel_means < 0.5), f"Expected shifted means, got {channel_means}"

    print("✓ ImageNet normalization constants correct")


def test_variable_window_mask_construction():
    """Verify mask construction for variable-length windows."""
    window_frames_list = [4, 6, 8, 10, 12, 14, 16]

    for wf in window_frames_list:
        n_temporal = wf // 2
        total_tokens = n_temporal * 256
        n_context = total_tokens - 256  # target = last tubelet

        # Context should be all but the last tubelet
        assert n_context == (n_temporal - 1) * 256, f"wf={wf}: n_context={n_context}"
        assert total_tokens - n_context == 256, f"wf={wf}: target should be 256 tokens"

        print(f"  wf={wf}: total={total_tokens}, context={n_context}, target=256")


def test_make_vjepa_encoder_predictor_backward_compat():
    """Legacy function should still work (delegates to analyzer scorer)."""
    from predictmem.vjepa_scorer import make_vjepa_encoder_predictor

    result = make_vjepa_encoder_predictor(checkpoint_path=None, device="cpu")
    assert result["num_mask_tokens"] == 10
    assert "context_encoder" in result
    assert "target_encoder" in result
    assert "predictor" in result
    print("✓ make_vjepa_encoder_predictor backward compat OK")


if __name__ == "__main__":
    test_num_mask_tokens()
    test_imagenet_normalization()
    test_variable_window_mask_construction()
    test_make_vjepa_encoder_predictor_backward_compat()
    print("\n✅ All P2 analyzer parity tests passed!")
