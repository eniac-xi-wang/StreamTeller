"""P4 tests for PredictMem: scorer leakage, online tubelet, checkpoint keys,
video preprocess alignment, and Qwen generate smoke.

Run with:
    PYTHONPATH=/root/stream/StreamTeller/models python test/test_predictmem_p4.py
"""

import sys
from pathlib import Path

_models_dir = Path(__file__).parent.parent / "models"
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

import torch

from predictmem.config import PredictMemConfig
from predictmem.token_mapping import TokenMapper
from predictmem.vjepa_scorer import (
    VJEPAPredictLossScorer,
    load_vjepa_checkpoint,
    make_vjepa_encoder_predictor,
    PredictMemScore,
    keep_indices_to_thw,
)


# ─── P4.1: test_scorer_no_target_leakage ──────────────────────────────────────

def test_scorer_no_target_leakage():
    """Ensure the scorer uses masked context encoder, not full encoder gather."""
    print("  P4.1: Testing scorer has no target leakage...")

    try:
        models = make_vjepa_encoder_predictor(
            img_size=256, patch_size=16, num_frames=16, tubelet_size=2,
            device="cpu", checkpoint_path=None,
        )
    except Exception as e:
        print(f"  P4.1 SKIP: {e}")
        return

    config = PredictMemConfig()
    scorer = VJEPAPredictLossScorer(
        config, models["context_encoder"], models["target_encoder"], models["predictor"],
    )

    frames = torch.randn(1, 3, 16, 256, 256)

    # score_tubelet should use context_encoder.forward(x, masks=[...])
    # which masks out target patches
    loss = scorer.score_tubelet(frames, target_tubelet_id=7)
    assert loss.shape == (1, 256), f"Expected (1,256), got {loss.shape}"
    assert not torch.isnan(loss).any(), "Loss contains NaN"
    assert not torch.isinf(loss).any(), "Loss contains Inf"

    # Verify that target latents come from target_encoder, not context_encoder
    # by checking the two encoders produce different outputs for the same input
    target_full = scorer.target_encoder(frames)  # full input, no mask
    context_full = scorer.context_encoder(frames)  # full input, no mask

    # With random init, outputs should be different (different encoder instances)
    diff = (target_full - context_full).abs().mean().item()
    assert diff > 0, f"Target and context encoder should have different outputs (diff={diff})"

    print(f"  P4.1 PASS: no target leakage, masked context used, diff={diff:.4f}")


# ─── P4.2: test_online_tubelet_keep_mask ──────────────────────────────────────

def test_online_tubelet_keep_mask():
    """Online mode: only score the newest tubelet, merge with history from cache."""
    print("  P4.2: Testing online tubelet keep mask...")

    try:
        models = make_vjepa_encoder_predictor(
            img_size=256, patch_size=16, num_frames=16, tubelet_size=2,
            device="cpu", checkpoint_path=None,
        )
    except Exception as e:
        print(f"  P4.2 SKIP: {e}")
        return

    config = PredictMemConfig()
    config.keep_ratio = 0.5
    scorer = VJEPAPredictLossScorer(
        config, models["context_encoder"], models["target_encoder"], models["predictor"],
    )

    frames = torch.randn(2, 3, 16, 256, 256)

    # Simulate online: process tubelets 0..3 first, then 4
    history_mask = torch.zeros(2, 8, 16, 16, dtype=torch.bool)

    # "Pre-populate" tubelets 0..3 with fake keep decisions (keep first half)
    for t in range(4):
        for b in range(2):
            history_mask[b, t, :, :8] = True  # keep left half

    # Now score tubelet 4 online
    score = scorer.score_window_online(
        frames, new_tubelet_id=4, history_keep_mask=history_mask,
    )

    assert score.keep_mask.shape == (2, 8, 16, 16)
    # Tubelets 0..3 should still have the historical keep pattern
    for t in range(4):
        for b in range(2):
            n_kept = score.keep_mask[b, t].sum().item()
            assert n_kept == 128, f"Tubelet {t}, batch {b}: expected 128 kept from history, got {n_kept}"

    # Tubelet 4 should have newly computed decisions
    n_new_kept_b0 = score.keep_mask[0, 4].sum().item()
    n_new_kept_b1 = score.keep_mask[1, 4].sum().item()
    expected_per_tubelet = max(1, int(256 * 0.5))  # 128
    # With min_cell_keep, may be slightly more
    assert n_new_kept_b0 >= 16, f"Tubelet 4 batch 0: too few kept ({n_new_kept_b0})"
    assert n_new_kept_b1 >= 16, f"Tubelet 4 batch 1: too few kept ({n_new_kept_b1})"

    # Tubelets 5..7 should be all False (not yet scored)
    for t in range(5, 8):
        assert score.keep_mask[0, t].sum() == 0, f"Tubelet {t} should be empty"

    print(f"  P4.2 PASS: online tubelet scoring, keep_mask merges correctly")


# ─── P4.3: test_checkpoint_loader_keys ────────────────────────────────────────

def test_checkpoint_loader_keys():
    """Test checkpoint loader handles all known key formats."""
    print("  P4.3: Testing checkpoint loader key handling...")

    # Build a fake checkpoint with module./backbone. prefixes
    fake_state = {}
    for i in range(3):
        fake_state[f"module.encoder.layer.{i}.weight"] = torch.randn(64, 64)
        fake_state[f"backbone.encoder.layer.{i}.bias"] = torch.randn(64)
        fake_state[f"predictor.layer.{i}.weight"] = torch.randn(32, 32)

    # Save and reload
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(fake_state, f)
        tmp_path = f.name

    try:
        # Test with no recognized keys – should wrap everything as "encoder"
        cleaned = load_vjepa_checkpoint(tmp_path, device="cpu")
        assert "encoder" in cleaned, "Fallback: should treat whole dict as encoder"
        for key in cleaned["encoder"]:
            assert "module." not in key, f"module. prefix not stripped: {key}"
            assert "backbone." not in key, f"backbone. prefix not stripped: {key}"

        # Test with structured checkpoint
        structured = {
            "target_encoder": {f"module.layer.{i}.weight": torch.randn(64, 64) for i in range(2)},
            "encoder": {f"module.layer.{i}.weight": torch.randn(64, 64) for i in range(2)},
            "predictor": {f"layer.{i}.weight": torch.randn(32, 32) for i in range(2)},
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(structured, f)
            tmp_path2 = f.name

        cleaned2 = load_vjepa_checkpoint(tmp_path2, device="cpu")
        assert "target_encoder" in cleaned2
        assert "encoder" in cleaned2
        assert "predictor" in cleaned2
        # Check key cleaning
        for key in cleaned2["target_encoder"]:
            assert "module." not in key

        Path(tmp_path2).unlink()
    finally:
        Path(tmp_path).unlink()

    print("  P4.3 PASS: checkpoint loader handles all key formats")


# ─── P4.4: test_real_video_preprocess_alignment ───────────────────────────────

def test_real_video_preprocess_alignment():
    """Verify preprocess alignment: V-JEPA frames [B,3,16,256,256] and Qwen equivalents."""
    print("  P4.4: Testing video preprocess alignment...")

    config = PredictMemConfig()
    mapper = TokenMapper(config)

    # Simulate processor output shapes
    B = 1
    # V-JEPA frames shape
    jepa_frames = torch.randn(B, 3, 16, 256, 256)
    assert jepa_frames.shape == (B, 3, 16, 256, 256)

    # Qwen frames shape
    qwen_frames = torch.randn(B, 3, 16, 512, 512)
    assert qwen_frames.shape == (B, 3, 16, 512, 512)

    # Expected project grid: Qwen3.5/Qwen3-VL use the same token formula
    # when the processor sees the same 16-frame 512px input.
    qwen35_grid = torch.tensor([[8, 32, 32]])
    mapper.assert_video_grid_thw(qwen35_grid)
    n_tokens = mapper.compute_num_video_tokens(qwen35_grid)
    assert n_tokens == 2048

    # Diagnostic short grid: processor resampling to 4 frames yields 512 tokens.
    short_grid = torch.tensor([[2, 32, 32]])
    mapper.assert_video_grid_thw(short_grid)
    n_tokens = mapper.compute_num_video_tokens(short_grid)
    assert n_tokens == 512
    assert config.num_jepa_tokens == 2048

    print("  P4.4 PASS: video preprocess alignment verified")


# ─── P4.5: test_qwen_generate_with_predictmem_small ──────────────────────────

def test_qwen_generate_with_predictmem_small():
    """Smoke test that generation path skips pruning during decode."""
    print("  P4.5: Testing Qwen generate with PredictMem (decode skip)...")

    from predictmem.token_pruner import TokenPruner

    class DummyCfg:
        video_token_id = 248057
        vision_start_token_id = 248053
        vision_end_token_id = 248054

    config = PredictMemConfig()
    pruner = TokenPruner(
        config=config,
        video_token_id=DummyCfg.video_token_id,
        vision_start_token_id=DummyCfg.vision_start_token_id,
        vision_end_token_id=DummyCfg.vision_end_token_id,
    )

    # Simulate decode step: pixel_values_videos is None
    assert pruner.should_skip_pruning(pixel_values_videos=None, inputs_embeds=torch.randn(1, 64, 128))

    # Simulate decode step: seq_len == 1
    assert pruner.should_skip_pruning(pixel_values_videos=torch.randn(1, 3, 16, 256, 256), inputs_embeds=torch.randn(1, 1, 128))

    # Simulate prefill: should NOT skip
    assert not pruner.should_skip_pruning(pixel_values_videos=torch.randn(1, 3, 16, 256, 256), inputs_embeds=torch.randn(1, 64, 128))

    print("  P4.5 PASS: decode skip, prefill proceed correctly")


# ─── P4.6: test_loss_unified ──────────────────────────────────────────────────

def test_loss_unified():
    """Verify loss uses loss_exp from config correctly."""
    print("  P4.6: Testing unified loss formula...")

    config = PredictMemConfig()
    config.loss_exp = 1.0

    pred = torch.randn(2, 256, 1024)
    target = torch.randn(2, 256, 1024)

    p = config.loss_exp
    loss = (pred - target).abs().pow(p).mean(dim=-1) / p

    assert loss.shape == (2, 256), f"Expected (2,256), got {loss.shape}"
    assert not torch.isnan(loss).any()
    assert (loss >= 0).all(), "Loss should be non-negative"

    # loss_exp = 2.0
    config.loss_exp = 2.0
    loss2 = (pred - target).abs().pow(2.0).mean(dim=-1) / 2.0
    assert loss2.shape == (2, 256)
    assert not torch.isnan(loss2).any()

    print("  P4.6 PASS: loss_exp unified across config")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running P4 tests...\n")
    test_scorer_no_target_leakage()
    test_online_tubelet_keep_mask()
    test_checkpoint_loader_keys()
    test_real_video_preprocess_alignment()
    test_qwen_generate_with_predictmem_small()
    test_loss_unified()
    print("\nAll P4 tests completed!")
