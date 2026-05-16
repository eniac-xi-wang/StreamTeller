"""FluxMem-like PredictMem plugin that runs inside Qwen3.5 prefill.

This is the online main path — no offline cache, no score JSONL, no
precomputed global_scores.  V-JEPA scoring is done in-memory during
the first prefill using sliding windows over ``predictmem_frames_256``.

A t-digest estimates the P90 loss threshold so that roughly the top
``keep_ratio`` (default 0.10) high-loss patches are kept per tubelet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tdigest import TDigest

from .config import PredictMemConfig
from .token_pruner import TokenPruner


class PredictMemStreamingMemory(nn.Module):
    """Plugin that runs V-JEPA sliding-window scoring inside Qwen3.5 prefill.

    Instantiated once and stored as ``self.predictmem`` on ``Qwen3_5Model``.
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

        # Lazy-loaded V-JEPA components
        self._scorer = None
        self._tdigest: TDigest | None = None
        self._pruner = TokenPruner(
            config=None,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
        )

    def _ensure_scorer(self, device: torch.device):
        if self._scorer is not None:
            return
        from .vjepa_scorer import VJEPAPredictLossScorer, make_vjepa_encoder_predictor

        checkpoint = "/data/model_weights_public/jepa/jeap_vitl_16_256.pt"
        models = make_vjepa_encoder_predictor(checkpoint_path=checkpoint, device=str(device))
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
        """Run sliding-window V-JEPA scoring and prune video tokens.

        Args:
            hidden_states: [B, L, D] dense after video embeds scattered
            position_ids: [3, B, L] M-RoPE positions
            attention_mask: [B, L] or None
            input_ids: [B, L]
            video_grid_thw: [1, 3] Qwen dense grid
            predictmem_frames_256: [T, 3, 256, 256] float in [0,1]
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

        # ── Build video keep mask from V-JEPA sliding windows ──
        T = predictmem_frames_256.shape[0]  # number of 1FPS frames
        num_tubelets = (T + 1) // 2  # ceil(T/2)
        grid_h, grid_w = 16, 16

        # keep_mask per tubelet: True = keep (high loss)
        tubelet_keep = torch.ones(num_tubelets, grid_h, grid_w, dtype=torch.bool, device="cpu")
        tubelet_scored = torch.zeros(num_tubelets, dtype=torch.bool)

        # t-digest for loss distribution (used to estimate P90 threshold)
        tdigest = TDigest()

        # Sliding windows: score only the latest tubelet per window
        for local_start in range(0, T - window_frames + 1, stride_frames):
            target_global_tubelet = local_start // 2 + (window_frames // 2 - 1)  # +7 for 16-frame
            if target_global_tubelet >= num_tubelets:
                continue

            window_frames_tensor = predictmem_frames_256[local_start:local_start + window_frames]
            # predictmem_frames_256 is [T, C, H, W]; V-JEPA needs [B, C, T, H, W]
            window_frames_tensor = window_frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, 16, 256, 256]

            loss = self._scorer.score_tubelet(
                window_frames_tensor.to(device),
                target_tubelet_id=window_frames // 2 - 1,
            )  # [1, 256]
            loss = loss.squeeze(0).reshape(grid_h, grid_w).cpu()  # [16, 16]

            # Update t-digest and mark scored
            tdigest.batch_update(loss.flatten().tolist())
            tubelet_scored[target_global_tubelet] = True

            # Apply threshold if enough data
            if len(tdigest) > 100:
                p_threshold = tdigest.percentile(100 * (1.0 - ratio))
                tubelet_keep[target_global_tubelet] = loss >= p_threshold
            # else: keep all for this tubelet (warmup)

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
                    # warmup / beyond scored range: keep all
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
        stats = {
            "original_video_tokens": total_video_tokens,
            "kept_video_tokens": kept,
            "dropped_video_tokens": total_video_tokens - kept,
            "keep_ratio_actual": round(kept / total_video_tokens, 4) if total_video_tokens > 0 else 1.0,
            "predictmem_scoring_latency_s": round(scoring_latency, 4),
            "num_tubelets_scored": int(tubelet_scored.sum().item()),
            "num_tubelets_warmup": int((~tubelet_scored).sum().item()),
            "tdigest_samples": len(tdigest),
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
