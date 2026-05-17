"""FluxMem-like PredictMem plugin that runs inside Qwen3.5 prefill.

V-JEPA scoring uses expanding windows for early frames and standard sliding
windows thereafter. Token boundary policy:

  - Tubelet 0 (frames 0-1): always DROP (no historical context to score against)
  - Last 4 frames (last 2 tubelets): always FULL KEEP (tail safety buffer)
  - Middle tubelets: V-JEPA loss + t-digest top 10%
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tdigest import TDigest

from .config import PredictMemConfig
from .token_pruner import TokenPruner


def iter_predictmem_windows(
    T: int,
    window_frames: int = 16,
    stride_frames: int = 2,
    tail_keep_tubelets: set | None = None,
):
    """Generate window specs for streaming PredictMem scoring.

    Phase 1 — Expanding windows (tubelets 1..7):
      Window always starts at frame 0. Length grows from 4 to window_frames.
      Target is always the LAST tubelet of the window.

    Phase 2 — Standard sliding windows (tubelets 8+):
      Window is ``window_frames`` long, stride ``stride_frames``.

    Windows whose target tubelet is in ``tail_keep_tubelets`` are skipped.
    Tubelet 0 has no historical context and cannot be scored — it is not
    yielded by this iterator.

    Yields:
        dict with keys: start, length, target_global_tubelet,
        target_local_tubelet, mode
    """
    skip = tail_keep_tubelets or set()

    # Phase 1: expanding windows for tubelets 1..7
    for target_end in range(3, min(window_frames, T), 2):
        wlen = target_end + 1
        tg = target_end // 2
        if tg in skip:
            continue
        yield {
            "start": 0,
            "length": wlen,
            "target_global_tubelet": tg,
            "target_local_tubelet": wlen // 2 - 1,
            "mode": "expanding",
        }

    # Phase 2: standard sliding windows (start at frame 2 to avoid
    # re-scoring tubelet 7 via a standard window)
    for s in range(2, T - window_frames + 1, stride_frames):
        tg = s // 2 + window_frames // 2 - 1
        if tg >= (T + 1) // 2:
            continue
        if tg in skip:
            continue
        yield {
            "start": s,
            "length": window_frames,
            "target_global_tubelet": tg,
            "target_local_tubelet": window_frames // 2 - 1,
            "mode": "sliding",
        }


class PredictMemStreamingMemory(nn.Module):
    """Plugin that runs V-JEPA scoring inside Qwen3.5 prefill.

    Uses analyzer-compatible scorer with MultiSeqWrapper /
    PredictorMultiSeqWrapper (num_mask_tokens=10), variable-length support
    for expanding windows, and the boundary-token policy documented above.
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

        checkpoint = self.config.jepa_checkpoint_path
        if not checkpoint:
            raise ValueError(
                "PredictMem requires config.jepa_checkpoint_path to be set. "
                "Pass --jepa-checkpoint through the bash entry or set it programmatically."
            )
        models = make_vjepa_analyzer_scorer(
            checkpoint_path=checkpoint,
            device=str(device),
            vjepa_src_path=self.config.vjepa_src_path,
        )
        self._scorer = VJEPAPredictLossScorer(
            self.config,
            models["context_encoder"],
            models["target_encoder"],
            models["predictor"],
            degraded=models["degraded"],
        )

    # ── Public API ─────────────────────────────────────────────────────────

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
        """Run expanding+sliding window V-JEPA scoring and prune tokens.

        Returns:
            pruned_hidden_states, pruned_position_ids, pruned_attention_mask,
            kept_sequence_indices, stats
        """
        ratio = keep_ratio if keep_ratio is not None else self.config.keep_ratio
        device = hidden_states.device
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        self._ensure_scorer(device)

        T = predictmem_frames_256.shape[0]
        num_tubelets = (T + 1) // 2
        grid_h, grid_w = 16, 16

        # ── Boundary sets ──
        dropped_bootstrap_tubelets = [0]  # tubelet 0 always dropped
        tail_keep_tubelets = sorted({num_tubelets - 2, num_tubelets - 1} & set(range(num_tubelets)))
        tail_keep_frames = []
        for t in tail_keep_tubelets:
            tail_keep_frames.extend([t * 2, t * 2 + 1])
        tail_keep_frames = sorted(set(tail_keep_frames) & set(range(T)))

        # ── Per-tubelet keep/drop ──
        tubelet_keep = torch.ones(num_tubelets, grid_h, grid_w, dtype=torch.bool, device="cpu")
        tubelet_scored = torch.zeros(num_tubelets, dtype=torch.bool)
        tubelet_mode = {}  # tubelet_id -> str

        # Tubelet 0: drop
        tubelet_keep[0] = False
        tubelet_mode[0] = "bootstrap_drop"

        # Tail tubelets: full keep (no scoring)
        for t in tail_keep_tubelets:
            tubelet_keep[t] = True
            tubelet_mode[t] = "protected_tail_full_keep"

        # ── Score middle tubelets ──
        tdigest = TDigest()
        total_scored = 0
        skip_set = set(dropped_bootstrap_tubelets) | set(tail_keep_tubelets)

        for spec in iter_predictmem_windows(T, window_frames, stride_frames, tail_keep_tubelets=skip_set):
            tg_tubelet = spec["target_global_tubelet"]
            if tg_tubelet >= num_tubelets:
                continue
            if tg_tubelet in skip_set:
                continue

            wlen = spec["length"]
            window_frames_tensor = predictmem_frames_256[spec["start"]:spec["start"] + wlen]
            clip = window_frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)

            loss = self._scorer.score_latest_tubelet_variable(
                clip.to(device), window_frames=wlen,
            )
            loss_2d = loss.squeeze(0).reshape(grid_h, grid_w).cpu()

            tubelet_scored[tg_tubelet] = True
            total_scored += 1

            # ── Keep/drop decision (no peeking at current tubelet) ──
            if len(tdigest) > 100:
                p_threshold = tdigest.percentile(100 * (1.0 - ratio))
                mode_str = "scored_digest"
            else:
                p_threshold = float(torch.quantile(loss_2d.flatten(), 1.0 - ratio))
                mode_str = "scored_local_quantile"

            tubelet_keep[tg_tubelet] = loss_2d >= p_threshold
            tubelet_mode[tg_tubelet] = mode_str

            # Update t-digest AFTER decision
            tdigest.batch_update(loss_2d.flatten().tolist())

        # ── Build video-local keep indices ──
        video_grid_t = int(video_grid_thw[0, 0].item())
        total_video_tokens = video_grid_t * grid_h * grid_w
        video_keep_indices_list: list[torch.Tensor] = []

        # Build per-tubelet keep masks for serialization
        keep_masks_tubelets = []

        B = hidden_states.shape[0]
        for _b in range(B):
            keep_list = []
            for t in range(video_grid_t):
                if t < num_tubelets:
                    keep_flat = tubelet_keep[t].flatten()
                else:
                    keep_flat = torch.ones(grid_h * grid_w, dtype=torch.bool)
                tubelet_offset = t * grid_h * grid_w
                kept_local = torch.where(keep_flat)[0] + tubelet_offset
                keep_list.append(kept_local)

            video_keep_indices_list.append(torch.cat(keep_list))

        # Build serializable keep masks
        for t in range(num_tubelets):
            frames = [t * 2, t * 2 + 1]
            frames = [f for f in frames if f < T]
            keep_masks_tubelets.append({
                "tubelet": t,
                "frames": frames,
                "mode": tubelet_mode.get(t, "bootstrap_drop" if t == 0 else "protected_tail_full_keep"),
                "keep_mask": tubelet_keep[t].flatten().tolist(),
            })

        # ── Prune ──
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
        scored_list = [int(i) for i in torch.where(tubelet_scored)[0].tolist()]

        stats = {
            "original_video_tokens": total_video_tokens,
            "kept_video_tokens": kept,
            "dropped_video_tokens": total_video_tokens - kept,
            "keep_ratio_actual": round(kept / total_video_tokens, 4) if total_video_tokens > 0 else 1.0,
            "predictmem_scoring_latency_s": round(scoring_latency, 4),
            "num_tubelets_scored": total_scored,
            "num_tubelets_unscored": int((~tubelet_scored).sum().item()),
            "scored_tubelets": scored_list,
            "dropped_bootstrap_tubelets": dropped_bootstrap_tubelets,
            "full_keep_tail_tubelets": tail_keep_tubelets,
            "full_keep_tail_frames": tail_keep_frames,
            "num_tail_tubelets_kept_full": len(tail_keep_tubelets),
            "tdigest_samples": len(tdigest),
            "window_mode": "expanding+sliding",
        }
        if self.config.record_keep_masks:
            stats["predictmem_keep_masks"] = {
                "grid_h": grid_h,
                "grid_w": grid_w,
                "tubelets": keep_masks_tubelets,
            }

        return pruned_hidden, pruned_pos, pruned_attn, video_keep_indices_list, stats

    def should_skip(self, pixel_values_videos, predictmem_frames_256, inputs_embeds) -> bool:
        if predictmem_frames_256 is None:
            return True
        if pixel_values_videos is None:
            return True
        if inputs_embeds.shape[1] == 1:
            return True
        return False
