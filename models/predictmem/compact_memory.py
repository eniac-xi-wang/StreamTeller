"""PredictMem compact memory — streaming visual token management.

Replaces the "full video → full visual tower → full inputs_embeds → prune"
pipeline with a streaming approach:

1. Stream frames tubelet-by-tubelet (never load entire video at once).
2. Maintain a V-JEPA ring buffer (max 16 aspect-preserving resized frames).
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

        self._jepa_buffer_frames: list[torch.Tensor] = []  # each [3, jepa_h, jepa_w] on CPU
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

    def _reset(self) -> None:
        self.compact_embeds.clear()
        self.compact_positions.clear()
        self.compact_meta.clear()
        self._jepa_buffer_frames.clear()
        self._jepa_buffer_frame_ids.clear()
        self._tdigest = TDigest()
        self._scorer = None
        self._tail_buffer_embeds.clear()
        self._tail_buffer_positions.clear()
        self.stats = {}
        self._total_video_tokens_full = 0
        self._total_kept_tokens = 0
        self._num_tubelets_scored = 0
        self._scored_tubelets = []

    # ── Public API ─────────────────────────────────────────────────────────

    def ingest_video_streaming(
        self,
        video_path: str,
        fps: float = 1.0,
        frame_budget: int = 0,
        stream_mode: str = "full",
        start_time: float = 0.0,
        end_time: float | None = None,
        keep_ratio: float | None = None,
        device: str = "cuda",
    ) -> None:
        """Stream a video through PredictMem and build compact memory.

        This is the main entry point. It never loads the full video into
        GPU memory at once.
        """
        self._reset()
        ratio = keep_ratio if keep_ratio is not None else self.config.keep_ratio

        sampler = StreamingVideoSampler(
            video_path, fps=fps,
            qwen_size=self.config.qwen_size,
            jepa_size=self.config.jepa_size,
            frame_budget=frame_budget,
            stream_mode=stream_mode,
            start_time=start_time, end_time=end_time,
        )
        num_tubelets = sampler.num_tubelets
        grid_h = sampler.jepa_h // self.config.patch_size
        grid_w = sampler.jepa_w // self.config.patch_size
        qwen_grid_h = sampler.qwen_h // self.config.patch_size
        qwen_grid_w = sampler.qwen_w // self.config.patch_size
        qwen_llm_h = qwen_grid_h // self.config.qwen_merge_size
        qwen_llm_w = qwen_grid_w // self.config.qwen_merge_size
        if (qwen_llm_h, qwen_llm_w) != (grid_h, grid_w):
            raise ValueError(
                "Compact JEPA/Qwen grids are not aligned: "
                f"JEPA={(grid_h, grid_w)} QwenLLM={(qwen_llm_h, qwen_llm_w)}"
            )
        self._ensure_scorer(device, img_size=(sampler.jepa_h, sampler.jepa_w))
        tokens_per_tubelet = grid_h * grid_w
        self._total_video_tokens_full = num_tubelets * tokens_per_tubelet

        # Boundary sets (three-tier)
        dropped_tubelets = {0} if self.config.drop_bootstrap and num_tubelets > 0 else set()
        tail_count = max(0, (self.config.tail_keep_frames + self.config.temporal_stride - 1) // self.config.temporal_stride)
        tail_tubelets = set(range(max(0, num_tubelets - tail_count), num_tubelets)) if tail_count else set()
        recent_count = max(0, self.config.recent_frames // self.config.temporal_stride)
        recent_tubelets = set(
            range(max(0, num_tubelets - recent_count), num_tubelets)
        ) - tail_tubelets if recent_count else set()

        t0_total = time.perf_counter()
        scoring_latency = 0.0
        visual_latency = 0.0

        for tubelet_data in sampler:
            tid = tubelet_data["tubelet_id"]
            jepa_tensor = tubelet_data["jepa"]  # [n, 3, jepa_h, jepa_w]
            qwen_frames = tubelet_data["qwen"]  # [n, qwen_h, qwen_w, 3] uint8
            frame_ids = list(tubelet_data["frame_indices"])
            times_s = list(tubelet_data["times_s"])
            tubelet_time = sum(times_s) / len(times_s) if times_s else float(tid * self.config.temporal_stride)
            is_tail = tid in tail_tubelets

            if tid in dropped_tubelets:
                # Bootstrap tubelet: drop entirely
                self._add_to_jepa_buffer(jepa_tensor, frame_ids)
                self.compact_meta.append({
                    "tubelet": tid, "frames": frame_ids, "times_s": times_s,
                    "timestamp_s": tubelet_time, "mode": "bootstrap_drop",
                    "tier": "bootstrap",
                    "kept_tokens": 0, "original_tokens": tokens_per_tubelet,
                })
                continue

            if is_tail:
                # Tail tubelets: full keep, no scoring
                t_vis0 = time.perf_counter()
                embeds, positions = self._process_qwen_chunk(qwen_frames, tid, device)
                visual_latency += time.perf_counter() - t_vis0

                if embeds.shape[0] != tokens_per_tubelet:
                    raise ValueError(
                        f"Compact tail chunk token mismatch: got {embeds.shape[0]}, expected {tokens_per_tubelet}"
                    )
                self.compact_embeds.append(embeds.detach().cpu())
                self.compact_positions.append(positions.detach().cpu())
                self._add_to_jepa_buffer(jepa_tensor, frame_ids)

                num_tokens = embeds.shape[0]
                self._total_kept_tokens += num_tokens
                self.compact_meta.append({
                    "tubelet": tid, "frames": frame_ids, "times_s": times_s,
                    "timestamp_s": tubelet_time, "mode": "protected_tail_full_keep",
                    "tier": "tail",
                    "kept_tokens": num_tokens, "original_tokens": tokens_per_tubelet,
                })
                del embeds, positions
                continue

            # Middle tubelet: V-JEPA score → keep mask → Qwen visual → apply mask
            self._add_to_jepa_buffer(jepa_tensor, frame_ids)

            tier = "old"  # default
            if tid == 0:
                keep_mask = torch.ones(grid_h, grid_w, dtype=torch.bool)
                mode = "bootstrap_full_keep"
                tier = "bootstrap"
            else:
                tubelet_ratio = self.config.recent_keep_ratio if tid in recent_tubelets else ratio
                tier = "recent" if tid in recent_tubelets else "old"
                t_sc0 = time.perf_counter()
                keep_mask = self._score_current_tubelet(tid, tubelet_ratio, grid_h, grid_w, device=device)
                scoring_latency += time.perf_counter() - t_sc0
                mode = f"scored_{tier}"
                self._num_tubelets_scored += 1
                self._scored_tubelets.append(tid)

            t_vis0 = time.perf_counter()
            embeds, positions = self._process_qwen_chunk(qwen_frames, tid, device)
            visual_latency += time.perf_counter() - t_vis0

            if embeds.shape[0] != tokens_per_tubelet:
                raise ValueError(
                    f"Compact scored chunk token mismatch: got {embeds.shape[0]}, expected {tokens_per_tubelet}"
                )
            keep_flat = keep_mask.flatten().to(device=embeds.device)
            pruned_embeds = embeds[keep_flat]
            pruned_positions = positions[:, keep_flat]

            self.compact_embeds.append(pruned_embeds.detach().cpu())
            self.compact_positions.append(pruned_positions.detach().cpu())

            kept_n = pruned_embeds.shape[0]
            self._total_kept_tokens += kept_n
            self.compact_meta.append({
                "tubelet": tid, "frames": frame_ids, "times_s": times_s,
                "timestamp_s": tubelet_time, "mode": mode,
                "tier": tier,
                "kept_tokens": kept_n, "original_tokens": tokens_per_tubelet,
            })

            # Free intermediate tensors
            del embeds, positions

        t_total = time.perf_counter() - t0_total

        self.stats = {
            "compact_memory": True,
            "original_video_tokens": self._total_video_tokens_full,
            "kept_video_tokens": self._total_kept_tokens,
            "dropped_video_tokens": self._total_video_tokens_full - self._total_kept_tokens,
            "keep_ratio_actual": round(self._total_kept_tokens / max(1, self._total_video_tokens_full), 4),
            "num_tubelets_scored": self._num_tubelets_scored,
            "scored_tubelets": self._scored_tubelets,
            "full_keep_tail_tubelets": sorted(tail_tubelets),
            "num_tail_tubelets_kept_full": len(tail_tubelets),
            "recent_tubelets": sorted(recent_tubelets),
            "num_recent_tubelets": len(recent_tubelets),
            "recent_keep_ratio": self.config.recent_keep_ratio,
            "predictmem_scoring_latency_s": round(scoring_latency, 4),
            "qwen_visual_latency_s": round(visual_latency, 4),
            "compact_memory_tokens": self._total_kept_tokens,
            "total_latency_s": round(t_total, 4),
            "tdigest_samples": len(self._tdigest),
            "num_tubelets": num_tubelets,
            "num_frames": sampler.num_frames,
            "source_resize_hw": [sampler.source_height, sampler.source_width],
            "qwen_resize_hw": [sampler.qwen_h, sampler.qwen_w],
            "jepa_resize_hw": [sampler.jepa_h, sampler.jepa_w],
            "qwen_grid_thw": [num_tubelets, qwen_grid_h, qwen_grid_w],
            "token_grid_hw": [grid_h, grid_w],
            "tokens_per_tubelet": tokens_per_tubelet,
            "stream_mode": stream_mode,
            "compact_tubelets": self.compact_meta,
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
        self,
        tubelet_id: int,
        keep_ratio: float,
        grid_h: int,
        grid_w: int,
        device: str | torch.device = "cuda",
    ) -> torch.Tensor:
        """Score the last tubelet in the JEPA buffer via V-JEPA.

        Returns:
            keep_mask: [grid_h, grid_w] bool tensor on CPU
        """
        from .vjepa_scorer import VJEPAPredictLossScorer

        scorer: VJEPAPredictLossScorer = self._scorer
        buf = torch.stack(self._jepa_buffer_frames)  # [W, 3, jepa_h, jepa_w]
        wlen = buf.shape[0]

        if wlen < 4:
            # Not enough context — keep all
            return torch.ones(grid_h, grid_w, dtype=torch.bool)

        clip = buf.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, 3, W, jepa_h, jepa_w]
        loss = scorer.score_latest_tubelet_variable(clip, window_frames=wlen)
        loss_2d = loss.squeeze(0).reshape(grid_h, grid_w).cpu()

        # Threshold: t-digest after warm-up, local quantile otherwise
        if len(self._tdigest) > 100:
            p_threshold = self._tdigest.percentile(100 * (1.0 - keep_ratio))
        else:
            p_threshold = float(torch.quantile(loss_2d.flatten(), 1.0 - keep_ratio))

        keep = loss_2d >= p_threshold

        # Update t-digest AFTER decision
        self._tdigest.batch_update(loss_2d.flatten().tolist())

        return keep

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

    def _ensure_scorer(self, device: str = "cuda", img_size: tuple[int, int] | None = None):
        if self._scorer is not None:
            return
        from .vjepa_scorer import VJEPAPredictLossScorer, make_vjepa_analyzer_scorer

        pm = getattr(getattr(self.model, "model", None), "predictmem", None)
        if pm is not None and hasattr(pm, "_ensure_scorer"):
            pm._ensure_scorer(torch.device(device), img_size=img_size)
            self._scorer = pm._scorer
            return

        checkpoint = self.config.jepa_checkpoint_path
        if not checkpoint:
            raise ValueError("PredictMem requires config.jepa_checkpoint_path to be set.")
        models = make_vjepa_analyzer_scorer(
            checkpoint_path=checkpoint,
            device=str(device),
            img_size=img_size or self.config.jepa_size,
            vjepa_src_path=self.config.vjepa_src_path,
        )
        self._scorer = VJEPAPredictLossScorer(
            self.config,
            models["context_encoder"],
            models["target_encoder"],
            models["predictor"],
            degraded=models["degraded"],
        )
