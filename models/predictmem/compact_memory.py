"""PredictMem compact memory — streaming visual token management.

Replaces the "full video → full visual tower → full inputs_embeds → prune"
pipeline with a streaming approach:

1. Stream frames tubelet-by-tubelet (never load entire video at once).
2. Maintain a V-JEPA ring buffer (max 16 frames of 256×256).
3. Score each tubelet via V-JEPA, get per-patch keep mask.
4. Call Qwen visual tower ONLY on the current tubelet/chunk.
5. Immediately apply keep mask, append pruned embeddings to compact memory.
6. Discard full pixel tensor / full visual embeds / loss tensor of current chunk.

At the end, compact memory contains only the kept visual tokens, ready to be
scattered into a minimal placeholder sequence for the language model.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from tdigest import TDigest

from .config import PredictMemConfig
from .streaming_sampler import StreamingVideoSampler


class PredictMemCompactMemory:
    """Streaming compact visual memory driven by V-JEPA prediction loss.

    Usage::

        cm = PredictMemCompactMemory(model, processor, config)
        cm.ingest_video_streaming(video_path, fps=1.0)
        result = cm.assemble()
        # result["visual_embeds"] → [K_total, hidden_dim]
        # result["visual_position_ids"] → [3, K_total]
        # result["stats"] → dict
    """

    def __init__(self, model, processor, config: PredictMemConfig):
        self.model = model
        self.processor = processor
        self.config = config

        self.compact_embeds: list[torch.Tensor] = []
        self.compact_positions: list[torch.Tensor] = []
        self.compact_meta: list[dict] = []

        self._jepa_buffer_frames: list[torch.Tensor] = []  # each [3, 256, 256] on CPU
        self._jepa_buffer_frame_ids: list[int] = []
        self._max_buffer = config.window_frames  # 16

        self._tdigest = TDigest()
        self._scorer = None

        self._tail_buffer_embeds: list[torch.Tensor] = []
        self._tail_buffer_positions: list[torch.Tensor] = []
        self._tail_keep_frames = config.tail_keep_frames  # 4

        self.stats: dict[str, Any] = {}
        self._total_video_tokens_full = 0
        self._total_kept_tokens = 0
        self._num_tubelets_scored = 0
        self._scored_tubelets: list[int] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def ingest_video_streaming(
        self,
        video_path: str,
        fps: float = 1.0,
        frame_budget: int = 0,
        start_time: float = 0.0,
        end_time: float | None = None,
        keep_ratio: float | None = None,
        device: str = "cuda",
    ) -> None:
        """Stream a video through PredictMem and build compact memory.

        This is the main entry point. It never loads the full video into
        GPU memory at once.
        """
        ratio = keep_ratio if keep_ratio is not None else self.config.keep_ratio
        self._ensure_scorer(device)

        sampler = StreamingVideoSampler(
            video_path, fps=fps,
            qwen_size=self.config.qwen_size,
            jepa_size=self.config.jepa_size,
            frame_budget=frame_budget,
            start_time=start_time, end_time=end_time,
        )
        num_tubelets = sampler.num_tubelets
        grid_h, grid_w = 16, 16
        tokens_per_tubelet = 2 * grid_h * grid_w  # 512
        self._total_video_tokens_full = num_tubelets * 2 * grid_h * grid_w

        # Boundary sets
        dropped_tubelets = {0}  # tubelet 0 always dropped
        tail_tubelets = set()
        if num_tubelets >= 2:
            tail_tubelets = {num_tubelets - 2, num_tubelets - 1} & set(range(num_tubelets))

        tail_frame_ids = set()
        for tt in tail_tubelets:
            tail_frame_ids.update([tt * 2, tt * 2 + 1])

        t0_total = time.perf_counter()
        scoring_latency = 0.0
        visual_latency = 0.0

        for tubelet_data in sampler:
            tid = tubelet_data["tubelet_id"]
            jepa_tensor = tubelet_data["jepa"]  # [n, 3, 256, 256]
            qwen_frames = tubelet_data["qwen"]  # [n, 512, 512, 3] uint8
            frame_ids = list(range(tid * 2, tid * 2 + jepa_tensor.shape[0]))

            is_tail = any(fid in tail_frame_ids for fid in frame_ids)

            if tid in dropped_tubelets:
                # Bootstrap tubelet: drop entirely
                self._add_to_jepa_buffer(jepa_tensor, frame_ids)
                self._total_video_tokens_full -= tokens_per_tubelet
                # Need embed for the placeholder count but mark for drop
                self.compact_meta.append({
                    "tubelet": tid, "frames": frame_ids,
                    "mode": "bootstrap_drop", "kept_tokens": 0,
                })
                continue

            if is_tail:
                # Tail tubelets: full keep, no scoring
                t_vis0 = time.perf_counter()
                embeds, positions = self._process_qwen_chunk(qwen_frames, tid, device)
                visual_latency += time.perf_counter() - t_vis0

                self._tail_buffer_embeds.append(embeds)
                self._tail_buffer_positions.append(positions)
                self._add_to_jepa_buffer(jepa_tensor, frame_ids)

                num_tokens = embeds.shape[0]
                self._total_kept_tokens += num_tokens
                self.compact_meta.append({
                    "tubelet": tid, "frames": frame_ids,
                    "mode": "protected_tail_full_keep", "kept_tokens": num_tokens,
                })
                continue

            # Middle tubelet: V-JEPA score → keep mask → Qwen visual → apply mask
            self._add_to_jepa_buffer(jepa_tensor, frame_ids)

            t_sc0 = time.perf_counter()
            keep_mask = self._score_current_tubelet(tid, ratio, grid_h, grid_w)
            scoring_latency += time.perf_counter() - t_sc0

            self._num_tubelets_scored += 1
            self._scored_tubelets.append(tid)

            t_vis0 = time.perf_counter()
            embeds, positions = self._process_qwen_chunk(qwen_frames, tid, device)
            visual_latency += time.perf_counter() - t_vis0

            # Apply keep mask: keep_mask is [grid_h, grid_w] per frame in tubelet
            n_frames = embeds.shape[0] // (grid_h * grid_w)
            kept_pieces = []
            kept_pos_pieces = []
            for f in range(n_frames):
                f_mask = keep_mask[f].flatten()  # [256]
                f_start = f * grid_h * grid_w
                f_end = f_start + grid_h * grid_w
                f_embeds = embeds[f_start:f_end]
                f_pos = positions[:, f_start:f_end]
                kept_pieces.append(f_embeds[f_mask])
                kept_pos_pieces.append(f_pos[:, f_mask])

            pruned_embeds = torch.cat(kept_pieces, dim=0) if kept_pieces else \
                embeds.new_empty((0, embeds.shape[1]))
            pruned_positions = torch.cat(kept_pos_pieces, dim=1) if kept_pos_pieces else \
                positions.new_empty((3, 0))

            self.compact_embeds.append(pruned_embeds)
            self.compact_positions.append(pruned_positions)

            kept_n = pruned_embeds.shape[0]
            self._total_kept_tokens += kept_n
            self.compact_meta.append({
                "tubelet": tid, "frames": frame_ids,
                "mode": "scored", "kept_tokens": kept_n,
            })

            # Free intermediate tensors
            del embeds, positions

        t_total = time.perf_counter() - t0_total

        # Append tail buffer (full keep, always at end)
        for emb, pos in zip(self._tail_buffer_embeds, self._tail_buffer_positions):
            self.compact_embeds.append(emb)
            self.compact_positions.append(pos)

        self._tail_buffer_embeds.clear()
        self._tail_buffer_positions.clear()

        self.stats = {
            "original_video_tokens": self._total_video_tokens_full,
            "kept_video_tokens": self._total_kept_tokens,
            "dropped_video_tokens": self._total_video_tokens_full - self._total_kept_tokens,
            "keep_ratio_actual": round(self._total_kept_tokens / max(1, self._total_video_tokens_full), 4),
            "num_tubelets_scored": self._num_tubelets_scored,
            "scored_tubelets": self._scored_tubelets,
            "full_keep_tail_tubelets": sorted(tail_tubelets),
            "num_tail_tubelets_kept_full": len(tail_tubelets),
            "predictmem_scoring_latency_s": round(scoring_latency, 4),
            "qwen_visual_latency_s": round(visual_latency, 4),
            "compact_memory_tokens": self._total_kept_tokens,
            "total_latency_s": round(t_total, 4),
            "tdigest_samples": len(self._tdigest),
            "num_tubelets": num_tubelets,
        }

    def assemble(self, device: str = "cuda") -> dict:
        """Assemble compact memory into the final format.

        Returns dict with keys:
            visual_embeds: [K_total, hidden_dim]
            visual_position_ids: [3, K_total]
            stats: dict
        """
        if not self.compact_embeds:
            return {
                "visual_embeds": torch.empty((0, self.model.config.text_config.hidden_size), device=device),
                "visual_position_ids": torch.empty((3, 0), dtype=torch.long, device=device),
                "stats": self.stats,
            }

        all_embeds = torch.cat([e.to(device) for e in self.compact_embeds], dim=0)
        all_positions = torch.cat([p.to(device) for p in self.compact_positions], dim=1)
        return {
            "visual_embeds": all_embeds,
            "visual_position_ids": all_positions,
            "stats": self.stats,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _add_to_jepa_buffer(self, jepa_tensor: torch.Tensor, frame_ids: list[int]):
        """Add frames to the V-JEPA ring buffer, evicting oldest if full."""
        for i in range(jepa_tensor.shape[0]):
            self._jepa_buffer_frames.append(jepa_tensor[i])
            self._jepa_buffer_frame_ids.append(frame_ids[i] if i < len(frame_ids) else -1)
        while len(self._jepa_buffer_frames) > self._max_buffer:
            self._jepa_buffer_frames.pop(0)
            self._jepa_buffer_frame_ids.pop(0)

    def _score_current_tubelet(
        self, tubelet_id: int, keep_ratio: float, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """Score the last tubelet in the JEPA buffer via V-JEPA.

        Returns:
            keep_mask: [2, grid_h, grid_w] bool tensor on CPU
        """
        from .vjepa_scorer import VJEPAPredictLossScorer

        scorer: VJEPAPredictLossScorer = self._scorer
        buf = torch.stack(self._jepa_buffer_frames)  # [W, 3, 256, 256]
        wlen = buf.shape[0]

        if wlen < 4:
            # Not enough context — keep all
            return torch.ones(2, grid_h, grid_w, dtype=torch.bool)

        clip = buf.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, W, 256, 256]
        loss = scorer.score_latest_tubelet_variable(clip, window_frames=wlen)
        loss_2d = loss.squeeze(0).reshape(2, grid_h, grid_w).cpu()  # [2, grid_h, grid_w]

        # Threshold: t-digest after warm-up, local quantile otherwise
        if len(self._tdigest) > 100:
            p_threshold = self._tdigest.percentile(100 * (1.0 - keep_ratio))
        else:
            p_threshold = float(torch.quantile(loss_2d.flatten(), 1.0 - keep_ratio))

        keep = loss_2d >= p_threshold

        # Update t-digest AFTER decision
        self._tdigest.batch_update(loss_2d.flatten().tolist())

        return keep  # [2, grid_h, grid_w]

    def _process_qwen_chunk(
        self, qwen_frames, tubelet_id: int, device: str = "cuda"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen visual tower on a single tubelet.

        Returns:
            (embeddings: [n_tokens, hidden_dim], position_ids: [3, n_tokens])
        """
        from .qwen_visual_chunk import QwenVisualChunkProcessor

        # Lazy init chunk processor
        if not hasattr(self, "_chunk_proc"):
            self._chunk_proc = QwenVisualChunkProcessor(self.model, self.processor, self.config)

        return self._chunk_proc.process_tubelet(qwen_frames, tubelet_id, device)

    def _ensure_scorer(self, device: str = "cuda"):
        if self._scorer is not None:
            return
        from .vjepa_scorer import VJEPAPredictLossScorer, make_vjepa_analyzer_scorer

        checkpoint = self.config.jepa_checkpoint_path
        if not checkpoint:
            raise ValueError("PredictMem requires config.jepa_checkpoint_path to be set.")
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
