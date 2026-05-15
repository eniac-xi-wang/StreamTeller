"""Token mapping between V-JEPA scores and Qwen LLM video tokens.

V-JEPA scores are produced on a fixed 16-frame 256px grid:
    8 temporal tubelets x 16 x 16 = 2048 tokens.

Qwen video token counts depend on the actual processor output. Qwen3.5 with
``video_grid_thw=[8,32,32]`` has the same 8 x 16 x 16 LLM-token grid, while
Qwen3-VL with ``video_grid_thw=[2,32,32]`` has a 2 x 16 x 16 grid and needs
temporal aggregation from four V-JEPA tubelets into one Qwen token.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import PredictMemConfig


class TokenMapper:
    """Map fixed-grid JEPA losses into the Qwen video-token layout in use."""

    def __init__(self, config: PredictMemConfig):
        self.config = config
        self._validate_config()
        self.num_tokens = config.num_jepa_tokens

    def _validate_config(self):
        cfg = self.config
        assert cfg.qwen_merge_size > 0, f"Expected positive qwen_merge_size, got {cfg.qwen_merge_size}"
        assert cfg.jepa_grid_t == 8, f"Expected jepa_grid_t=8, got {cfg.jepa_grid_t}"
        assert cfg.jepa_grid_h == 16, f"Expected jepa_grid_h=16, got {cfg.jepa_grid_h}"
        assert cfg.jepa_grid_w == 16, f"Expected jepa_grid_w=16, got {cfg.jepa_grid_w}"

    @property
    def jepa_grid(self) -> tuple[int, int, int]:
        return self.config.jepa_grid_t, self.config.jepa_grid_h, self.config.jepa_grid_w

    def jepa_to_qwen_index(self, jepa_token_id: int, video_grid_thw: torch.Tensor | None = None) -> int:
        """Map one JEPA local token id to a Qwen video local token id.

        Without ``video_grid_thw`` this preserves the original Qwen3.5 identity
        behavior. When Qwen has fewer temporal tokens, multiple JEPA ids can map
        to the same Qwen id; use score aggregation for pruning decisions.
        """
        ids = torch.tensor([jepa_token_id], dtype=torch.long)
        return int(self.jepa_to_qwen_indices(ids, video_grid_thw=video_grid_thw)[0].item())

    def jepa_to_qwen_indices(
        self,
        jepa_token_ids: torch.Tensor,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map JEPA token local indices to Qwen video-token local indices."""
        if video_grid_thw is None:
            return jepa_token_ids

        jT, jH, jW = self.jepa_grid
        qT, qH, qW = self.qwen_llm_grid(video_grid_thw)

        ids = jepa_token_ids.to(dtype=torch.long)
        jt = ids // (jH * jW)
        rem = ids % (jH * jW)
        jh = rem // jW
        jw = rem % jW

        qt = torch.div(jt * qT, jT, rounding_mode="floor").clamp(max=qT - 1)
        qh = torch.div(jh * qH, jH, rounding_mode="floor").clamp(max=qH - 1)
        qw = torch.div(jw * qW, jW, rounding_mode="floor").clamp(max=qW - 1)
        return qt * (qH * qW) + qh * qW + qw

    def qwen_video_to_jepa_index(
        self,
        qwen_video_local_index: int,
        video_grid_thw: torch.Tensor | None = None,
    ) -> int:
        """Reverse map to the first JEPA token covered by a Qwen video token."""
        if video_grid_thw is None:
            return qwen_video_local_index

        jT, jH, jW = self.jepa_grid
        qT, qH, qW = self.qwen_llm_grid(video_grid_thw)
        qt = qwen_video_local_index // (qH * qW)
        rem = qwen_video_local_index % (qH * qW)
        qh = rem // qW
        qw = rem % qW

        jt = min((qt * jT) // qT, jT - 1)
        jh = min((qh * jH) // qH, jH - 1)
        jw = min((qw * jW) // qW, jW - 1)
        return jt * (jH * jW) + jh * jW + jw

    def get_tubelet_indices(self, tubelet_id: int) -> torch.Tensor:
        """Return flat JEPA token indices for a temporal tubelet (0..7)."""
        offset = tubelet_id * self.config.jepa_grid_h * self.config.jepa_grid_w
        return torch.arange(offset, offset + self.config.jepa_grid_h * self.config.jepa_grid_w)

    def assert_video_grid_thw(self, video_grid_thw: torch.Tensor, expected_t: int | None = None):
        """Validate Qwen processor ``video_grid_thw``.

        Both Qwen3.5 ``[8,32,32]`` and Qwen3-VL ``[2,32,32]`` are valid for this
        project. Spatial dimensions must still match the configured 512px Qwen
        input grid and be divisible by the Qwen merge size.
        """
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is required for dynamic JEPA->Qwen mapping")
        assert video_grid_thw.ndim == 2 and video_grid_thw.shape[1] == 3, (
            f"Expected video_grid_thw with shape [N,3], got {tuple(video_grid_thw.shape)}"
        )

        merge = self.config.qwen_merge_size
        for i, row in enumerate(video_grid_thw.detach().cpu()):
            t, h, w = (int(v) for v in row.tolist())
            assert t > 0, f"Row {i}: expected positive t, got {t}"
            if expected_t is not None:
                assert t == expected_t, f"Row {i}: expected t={expected_t}, got {t}"
            assert h == self.config.qwen_grid_h, f"Row {i}: expected h={self.config.qwen_grid_h}, got {h}"
            assert w == self.config.qwen_grid_w, f"Row {i}: expected w={self.config.qwen_grid_w}, got {w}"
            assert h % merge == 0 and w % merge == 0, (
                f"Row {i}: h/w must be divisible by merge={merge}, got {(h, w)}"
            )

    def qwen_llm_grid(self, video_grid_thw: torch.Tensor) -> tuple[int, int, int]:
        """Return one video's Qwen LLM-token grid ``(T, H, W)``."""
        self.assert_video_grid_thw(video_grid_thw)
        if video_grid_thw.shape[0] != 1:
            raise ValueError(
                "TokenMapper aggregation expects one video grid row per call. "
                "For batched data, call it once per sample/video."
            )
        t, h, w = (int(v) for v in video_grid_thw[0].detach().cpu().tolist())
        merge = self.config.qwen_merge_size
        return t, h // merge, w // merge

    def compute_num_video_tokens(self, video_grid_thw: torch.Tensor) -> int:
        """Compute the number of Qwen LLM video tokens.

        ``video_grid_thw`` may contain multiple videos. Token counts add across
        rows; they must not be multiplied across the whole tensor.
        """
        self.assert_video_grid_thw(video_grid_thw)
        merge = self.config.qwen_merge_size
        rows = video_grid_thw.to(dtype=torch.long)
        counts = rows[:, 0] * (rows[:, 1] // merge) * (rows[:, 2] // merge)
        return int(counts.sum().item())

    def aggregate_jepa_loss_to_qwen(
        self,
        loss_map: torch.Tensor,
        video_grid_thw: torch.Tensor,
        reduce: str = "mean",
    ) -> torch.Tensor:
        """Aggregate JEPA prediction losses to the actual Qwen video-token grid."""
        loss_map = self._as_batched_jepa_map(loss_map, "loss_map").float()
        return self._aggregate_jepa_map_to_qwen(loss_map, video_grid_thw, reduce=reduce)

    def aggregate_jepa_keep_to_qwen(
        self,
        keep_mask: torch.Tensor,
        video_grid_thw: torch.Tensor,
        reduce: str = "mean",
    ) -> torch.Tensor:
        """Aggregate a JEPA keep mask into Qwen-grid scores in ``[0,1]``."""
        keep_scores = self._as_batched_jepa_map(keep_mask, "keep_mask").float()
        return self._aggregate_jepa_map_to_qwen(keep_scores, video_grid_thw, reduce=reduce)

    def map_scores_to_qwen_keep_indices(
        self,
        *,
        video_grid_thw: torch.Tensor,
        loss_map: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
        keep_ratio: float | None = None,
        min_cell_keep: bool | None = None,
        cell_grid_size: int | None = None,
        reduce: str = "mean",
    ) -> list[torch.Tensor]:
        """Return Qwen local keep indices from JEPA loss/keep tensors.

        Prefer ``loss_map`` because it preserves score strength across temporal
        aggregation. ``keep_mask`` is a compatibility fallback for old caches.
        """
        if loss_map is None and keep_mask is None:
            raise ValueError("Either loss_map or keep_mask must be provided")

        if loss_map is not None:
            qwen_scores = self.aggregate_jepa_loss_to_qwen(loss_map, video_grid_thw, reduce=reduce)
        else:
            qwen_scores = self.aggregate_jepa_keep_to_qwen(keep_mask, video_grid_thw, reduce=reduce)

        return self.keep_indices_from_qwen_scores(
            qwen_scores,
            keep_ratio=keep_ratio,
            min_cell_keep=min_cell_keep,
            cell_grid_size=cell_grid_size,
        )

    def keep_indices_from_qwen_scores(
        self,
        qwen_scores: torch.Tensor,
        keep_ratio: float | None = None,
        min_cell_keep: bool | None = None,
        cell_grid_size: int | None = None,
    ) -> list[torch.Tensor]:
        """Select top-scoring Qwen video local indices from a Qwen-grid score map."""
        if qwen_scores.ndim == 3:
            qwen_scores = qwen_scores.unsqueeze(0)
        if qwen_scores.ndim != 4:
            raise ValueError(f"Expected qwen_scores [B,T,H,W] or [T,H,W], got {tuple(qwen_scores.shape)}")

        ratio = self.config.keep_ratio if keep_ratio is None else keep_ratio
        if not (0 < ratio <= 1):
            raise ValueError(f"keep_ratio must be in (0,1], got {ratio}")
        use_cell_keep = self.config.min_cell_keep if min_cell_keep is None else min_cell_keep
        cells = self.config.cell_grid_size if cell_grid_size is None else cell_grid_size

        keep_indices = []
        for scores in qwen_scores:
            total = scores.numel()
            n_keep = max(1, min(total, int(total * ratio)))
            keep_indices.append(self._topk_with_cell_coverage(scores, n_keep, use_cell_keep, cells))
        return keep_indices

    def _aggregate_jepa_map_to_qwen(
        self,
        values: torch.Tensor,
        video_grid_thw: torch.Tensor,
        reduce: str = "mean",
    ) -> torch.Tensor:
        if reduce not in {"mean", "max"}:
            raise ValueError(f"Unsupported reduce={reduce!r}; expected 'mean' or 'max'")

        jT, jH, jW = self.jepa_grid
        qT, qH, qW = self.qwen_llm_grid(video_grid_thw)

        if (jT, jH, jW) == (qT, qH, qW):
            return values.clone()

        if jT % qT == 0 and jH % qH == 0 and jW % qW == 0:
            t_factor = jT // qT
            h_factor = jH // qH
            w_factor = jW // qW
            grouped = values.view(values.shape[0], qT, t_factor, qH, h_factor, qW, w_factor)
            reduce_dims = (2, 4, 6)
            if reduce == "mean":
                return grouped.mean(dim=reduce_dims)
            return grouped.amax(dim=reduce_dims)

        # Fallback for future grids that are not exact integer downsampling of JEPA.
        mode = "trilinear"
        resized = F.interpolate(
            values.unsqueeze(1),
            size=(qT, qH, qW),
            mode=mode,
            align_corners=False,
        ).squeeze(1)
        return resized

    def _as_batched_jepa_map(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4:
            raise ValueError(f"Expected {name} [B,8,16,16] or [8,16,16], got {tuple(value.shape)}")
        if tuple(value.shape[-3:]) != self.jepa_grid:
            raise ValueError(f"Expected {name} trailing shape {self.jepa_grid}, got {tuple(value.shape[-3:])}")
        return value

    def _topk_with_cell_coverage(
        self,
        scores: torch.Tensor,
        n_keep: int,
        min_cell_keep: bool,
        cell_grid_size: int,
    ) -> torch.Tensor:
        flat_scores = scores.flatten()
        total = flat_scores.numel()
        if n_keep >= total:
            return torch.arange(total, device=scores.device, dtype=torch.long)

        if not min_cell_keep or cell_grid_size <= 0:
            return torch.topk(flat_scores, n_keep).indices.sort().values

        T, H, W = scores.shape
        cell_grid_size = max(1, min(cell_grid_size, H, W))
        selected: list[int] = []

        for t in range(T):
            for ci in range(cell_grid_size):
                h0 = math.floor(ci * H / cell_grid_size)
                h1 = math.floor((ci + 1) * H / cell_grid_size)
                for cj in range(cell_grid_size):
                    w0 = math.floor(cj * W / cell_grid_size)
                    w1 = math.floor((cj + 1) * W / cell_grid_size)
                    if h1 <= h0 or w1 <= w0:
                        continue
                    cell_scores = scores[t, h0:h1, w0:w1].reshape(-1)
                    best = int(torch.argmax(cell_scores).item())
                    dh = best // (w1 - w0)
                    dw = best % (w1 - w0)
                    selected.append(t * H * W + (h0 + dh) * W + (w0 + dw))

        selected_idx = torch.tensor(sorted(set(selected)), device=scores.device, dtype=torch.long)
        if selected_idx.numel() >= n_keep:
            selected_scores = flat_scores[selected_idx]
            return selected_idx[torch.topk(selected_scores, n_keep).indices].sort().values

        used = torch.zeros(total, device=scores.device, dtype=torch.bool)
        used[selected_idx] = True
        remaining_idx = torch.where(~used)[0]
        remaining_needed = n_keep - selected_idx.numel()
        extra_idx = remaining_idx[torch.topk(flat_scores[remaining_idx], remaining_needed).indices]
        return torch.cat([selected_idx, extra_idx]).sort().values

    def tubelet_id_to_token_range(self, tubelet_id: int) -> tuple[int, int]:
        """Return ``(start, end)`` JEPA local indices for a temporal tubelet."""
        n = self.config.jepa_grid_h * self.config.jepa_grid_w
        start = tubelet_id * n
        end = start + n
        return start, end
