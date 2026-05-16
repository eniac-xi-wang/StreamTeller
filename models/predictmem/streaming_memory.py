"""FluxMem-like PredictMem plugin that runs inside Qwen3.5 prefill.

This is the online main path — no offline cache, no score JSONL, no
precomputed global_scores.  V-JEPA scoring is done in-memory during
the first prefill using expanding windows for early frames and standard
sliding windows thereafter, aligned with the Survey analyzer.

A t-digest estimates the P90 loss threshold so that roughly the top
``keep_ratio`` (default 0.10) high-loss patches are kept per tubelet.
When t-digest samples are insufficient, local quantile is used (not keep_all).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tdigest import TDigest

from .config import PredictMemConfig
from .token_pruner import TokenPruner


def iter_predictmem_windows(T: int, window_frames: int = 16, stride_frames: int = 2):
    """Generate window specs for streaming PredictMem scoring.

    Phase 1 — Expanding windows (tubelets 1..7):
      Window always starts at frame 0. Length grows from 4 → 16.
      Target is always the LAST tubelet of the window.

    Phase 2 — Standard sliding windows (tubelets 8+):
      Window is ``window_frames`` long, stride ``stride_frames``.
      Target is the last tubelet (index 7) of each window.

    Tubelet 0 (frames 0-1) has no historical context and cannot be scored
    by expanding windows. It is returned with ``target_global_tubelet=0``
    and ``mode="bootstrap"`` — the caller may choose to keep_all or skip.

    Yields:
        dict with keys: start, length, target_global_tubelet,
        target_local_tubelet, mode
    """
    # Phase 1: expanding windows for tubelets 1..7
    # target_end is the last frame index of the target tubelet (1-indexed frame)
    # tubelet k covers frames [2k, 2k+1], target_end = 2k+1
    for target_end in range(3, min(window_frames, T), 2):
        wlen = target_end + 1  # frames 0..target_end inclusive
        target_global_tubelet = target_end // 2  # tubelet index
        yield {
            "start": 0,
            "length": wlen,
            "target_global_tubelet": target_global_tubelet,
            "target_local_tubelet": wlen // 2 - 1,  # last tubelet in window
            "mode": "expanding",
        }

    # Phase 2: standard sliding windows from start=2
    # We start at frame 2 to avoid re-scoring tubelet 7 (already scored by
    # the last expanding window). Each window scores the last tubelet (index 7).
    for s in range(2, T - window_frames + 1, stride_frames):
        tgt_global = s // 2 + window_frames // 2 - 1  # s//2 + 7
        if tgt_global >= (T + 1) // 2:
            continue
        yield {
            "start": s,
            "length": window_frames,
            "target_global_tubelet": tgt_global,
            "target_local_tubelet": window_frames // 2 - 1,
            "mode": "sliding",
        }

    # Bootstrap note: tubelet 0 (frames 0-1) has no window that can score it.
    # Callers should detect this and may keep_all for tubelet 0 only.


class PredictMemStreamingMemory(nn.Module):
    """Plugin that runs V-JEPA sliding-window scoring inside Qwen3.5 prefill.

    Instantiated once and stored as ``self.predictmem`` on ``Qwen3_5Model``.
    Uses the analyzer-compatible scorer with MultiSeqWrapper /
    PredictorMultiSeqWrapper, num_mask_tokens=10, and variable-length window
    support for expanding windows on early frames.
    """

    def __init__(
        self,
        video_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
        config: PredictMemConfig,
    ):
        super().__init__()
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.config = config

        # Lazy-loaded V-JEPA components (analyzer-compatible)
        self._scorer = None
        self._pruner = TokenPruner(
            config=None,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
        )

    def _ensure_scorer(self, device: torch.device):
        if self._scorer is not None:
            return
        from .vjepa_scorer import VJEPAPredictLossScorer, make_vjepa_analyzer_scorer

        checkpoint = "/data/model_weights_public/jepa/jeap_vitl_16_256.pt"
        models = make_vjepa_analyzer_scorer(checkpoint_path=checkpoint, device=str(device))
        self._scorer = VJEPAPredictLossScorer(
            self.config,
            models["context_encoder"],
            models["target_encoder"],
            models["predictor"],
            degraded=models["degraded"],
        )

    def process_memory_streaming(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        video_grid_thw: torch.Tensor,
        predictmem_frames_256: torch.Tensor,
        keep_ratio: float | None = None,
        window_frames: int = 16,
        stride_frames: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], dict]:
        """Run expanding + sliding window V-JEPA scoring and prune video tokens.

        Args:
            hidden_states: [B, L, D] dense after video embeds scattered
            position_ids: [3, B, L] M-RoPE positions
            attention_mask: [B, L] or None
            input_ids: [B, L]
            video_grid_thw: [1, 3] Qwen dense grid
            predictmem_frames_256: [T, 3, 256, 256] ImageNet-normalized float
            keep_ratio: fraction of patches to keep (default from config)
            window_frames: V-JEPA window size (16)
            stride_frames: stride between windows (2)

        Returns:
            pruned_hidden_states: [B, L_new, D]
            pruned_position_ids: [3, B, L_new]
            pruned_attention_mask: [B, L_new]
            kept_sequence_indices: list of [K_b] per batch
            stats: dict with kept/dropped counts, scoring latency, etc.
        """
        ratio = keep_ratio if keep_ratio is not None else self.config.keep_ratio
        device = hidden_states.device
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        self._ensure_scorer(device)

        # ── Build video keep mask from V-JEPA expanding + sliding windows ──
        T = predictmem_frames_256.shape[0]
        num_tubelets = (T + 1) // 2  # ceil(T/2)
        grid_h, grid_w = 16, 16

        # keep_mask per tubelet: True = keep (high loss)
        tubelet_keep = torch.ones(num_tubelets, grid_h, grid_w, dtype=torch.bool, device="cpu")
        tubelet_scored = torch.zeros(num_tubelets, dtype=torch.bool)

        # t-digest for global loss distribution (populated AFTER each decision)
        tdigest = TDigest()
        total_scored_tubelets = 0

        # Iterate windows (expanding + sliding)
        for spec in iter_predictmem_windows(T, window_frames, stride_frames):
            tg_tubelet = spec["target_global_tubelet"]
            if tg_tubelet >= num_tubelets:
                continue

            wlen = spec["length"]
            window_frames_tensor = predictmem_frames_256[spec["start"]:spec["start"] + wlen]
            # predictmem_frames_256 is [T, C, H, W]; V-JEPA needs [B, C, T, H, W]
            clip = window_frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, wlen, 256, 256]

            # Score latest tubelet via analyzer-compatible variable-length scorer
            loss = self._scorer.score_latest_tubelet_variable(
                clip.to(device), window_frames=wlen,
            )  # [1, 256]
            loss_2d = loss.squeeze(0).reshape(grid_h, grid_w).cpu()  # [16, 16]

            # ── Keep/drop decision ──
            tubelet_scored[tg_tubelet] = True
            total_scored_tubelets += 1

            if len(tdigest) > 100:
                # Enough global samples: use t-digest P90
                p_threshold = tdigest.percentile(100 * (1.0 - ratio))
                tubelet_keep[tg_tubelet] = loss_2d >= p_threshold
            else:
                # Warmup: use local quantile on this tubelet's loss
                threshold = torch.quantile(
                    loss_2d.flatten(), 1.0 - ratio
                )
                tubelet_keep[tg_tubelet] = loss_2d >= threshold

            # Update t-digest AFTER decision (avoid peeking at current tubelet)
            tdigest.batch_update(loss_2d.flatten().tolist())

        # ── Build video-local keep indices from per-tubelet masks ──
        video_grid_t = int(video_grid_thw[0, 0].item())
        total_video_tokens = video_grid_t * grid_h * grid_w
        video_keep_indices_list: list[torch.Tensor] = []

        B = hidden_states.shape[0]
        for _b in range(B):
            keep_list = []
            for t in range(video_grid_t):
                if t < num_tubelets and tubelet_scored[t]:
                    mask = tubelet_keep[t]  # [16, 16] bool
                    keep_flat = mask.flatten()
                else:
                    # Tubelet 0 (bootstrap) or beyond scored range: keep all
                    keep_flat = torch.ones(grid_h * grid_w, dtype=torch.bool)
                tubelet_offset = t * grid_h * grid_w
                kept_local = torch.where(keep_flat)[0] + tubelet_offset
                keep_list.append(kept_local)
            video_keep_indices_list.append(torch.cat(keep_list))

        # ── Prune using TokenPruner ──
        pruned_hidden, pruned_pos, pruned_attn = self._pruner.prune(
            input_ids=input_ids,
            inputs_embeds=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            video_keep_indices=video_keep_indices_list,
        )

        t1.record()
        torch.cuda.synchronize()
        scoring_latency = t0.elapsed_time(t1) / 1000.0

        kept = video_keep_indices_list[0].shape[0]
        num_tubelets_unscored = int((~tubelet_scored).sum().item())
        early_scored = [int(i) for i in torch.where(tubelet_scored)[0].tolist()]

        stats = {
            "original_video_tokens": total_video_tokens,
            "kept_video_tokens": kept,
            "dropped_video_tokens": total_video_tokens - kept,
            "keep_ratio_actual": round(kept / total_video_tokens, 4) if total_video_tokens > 0 else 1.0,
            "predictmem_scoring_latency_s": round(scoring_latency, 4),
            "num_tubelets_scored": total_scored_tubelets,
            "num_tubelets_unscored": num_tubelets_unscored,
            "num_tubelets_warmup": num_tubelets_unscored,
            "early_scored_tubelets": early_scored,
            "first_full_keep_tubelets": [0] if num_tubelets_unscored == 1 and not tubelet_scored[0].item() else [],
            "tdigest_samples": len(tdigest),
            "window_mode": "expanding+sliding",
        }

        return pruned_hidden, pruned_pos, pruned_attn, video_keep_indices_list, stats

    def should_skip(self, pixel_values_videos, predictmem_frames_256, inputs_embeds) -> bool:
        """Return True during decode steps or when no video signal is present."""
        if predictmem_frames_256 is None:
            return True
        if pixel_values_videos is None:
            return True
        if inputs_embeds.shape[1] == 1:
            return True
        return False
