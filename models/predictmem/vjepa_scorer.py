"""V-JEPA prediction loss scorer for PredictMem — analyzer-compatible.

Uses MultiSeqWrapper / PredictorMultiSeqWrapper (same as the Survey analyzer)
so that checkpoint loading, mask format, and forward paths are identical.

Key alignment points vs upstream analyzer:
  - num_mask_tokens=10
  - weights_only=True on torch.load
  - "module.backbone." prefix stripping (not "module." + "backbone." separately)
  - ImageNet normalization applied externally (see vision_inputs.py)
  - Variable-length window support for expanding windows
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .config import PredictMemConfig


@dataclass
class PredictMemScore:
    """Output of V-JEPA scoring for one window."""

    loss_map: torch.Tensor  # [B, T, 16, 16]
    keep_mask: torch.Tensor  # [B, T, 16, 16], bool
    keep_indices: list[torch.Tensor]  # per-sample flat local token indices


class VJEPAPredictLossScorer:
    """V-JEPA prediction loss scorer using MultiSeqWrapper-wrapped encoders.

    The scorer is created via :func:`make_vjepa_analyzer_scorer` which mirrors
    the Survey analyzer's build pipeline exactly.
    """

    def __init__(
        self,
        config: PredictMemConfig,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        degraded: bool = False,
    ):
        self.config = config
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.degraded = degraded
        self.grid_h = 16
        self.grid_w = 16
        self.num_tubelet_patches = 256

    # ── Analyzer-compatible: score latest tubelet, variable window length ──

    @torch.no_grad()
    def score_latest_tubelet_variable(
        self,
        clip: torch.Tensor,
        window_frames: int,
    ) -> torch.Tensor:
        """Score the latest (last) tubelet of a variable-length window.

        Uses the same mask / wrapper convention as the Survey analyzer:
          - Target encoder: full clip, no masks
          - Context encoder: all tokens EXCEPT the last tubelet
          - Predictor: context → target

        Args:
            clip: [B, 3, window_frames, 256, 256]  ImageNet-normalized
            window_frames: number of frames (4, 6, ..., 16 for expanding;
                          16 for standard sliding)

        Returns:
            loss: [B, 256] per-patch L1 prediction loss for the last tubelet
        """
        B, device = clip.shape[0], clip.device

        n_temporal = window_frames // 2  # tubelet_size=2
        total_tokens = n_temporal * self.num_tubelet_patches
        n_context = total_tokens - self.num_tubelet_patches  # all but last tubelet

        # Build flat context / target index masks (analyzer format)
        all_tokens = torch.arange(total_tokens, device=device)
        ctx_mask = all_tokens[:n_context]  # [N_ctx]
        tgt_mask = all_tokens[n_context:]  # [256]

        # Wrapper format: for each batch item, a list of mask variants
        # Shape: [[tensor(1, K)], ...] for B items, each with 1 variant
        clips_list = [clip[b:b + 1] for b in range(B)]
        masks_enc = [[ctx_mask.unsqueeze(0)] for _ in range(B)]
        masks_pred = [[tgt_mask.unsqueeze(0)] for _ in range(B)]

        # Target encoder: full input (no masks) → [tensor(1, total, D), ...]
        h_list = self.target_encoder(clips_list)
        h_target = torch.cat(
            [h_list[b][:, -self.num_tubelet_patches:, :] for b in range(B)], dim=0
        )  # [B, 256, D]

        # Context encoder: masked input → [[tensor(1, N_ctx, D)], ...]
        z = self.context_encoder(clips_list, masks=masks_enc)

        # Predictor: context → target
        pred_out = self.predictor(z, masks_enc, masks_pred)
        # pred_out: [[tensor(1, 256, D)], ...]
        pred = torch.cat([pred_out[b][0] for b in range(B)], dim=0)  # [B, 256, D]

        # Per-patch L1 loss (mean over feature dim)
        loss = torch.abs(pred - h_target).mean(dim=-1)  # [B, 256]

        return loss

    # ── Legacy single-tubelet API (used by offline paths) ──

    @torch.no_grad()
    def score_tubelet(
        self,
        frames_256: torch.Tensor,
        target_tubelet_id: int,
    ) -> torch.Tensor:
        """Score a single tubelet using the analyzer-compatible wrapper path.

        This is kept for backward compatibility with legacy offline scripts.
        New code should use ``score_latest_tubelet_variable``.
        """
        B, device = frames_256.shape[0], frames_256.device
        N = self.num_tubelet_patches
        tubelet_start = target_tubelet_id * N
        tubelet_end = tubelet_start + N
        total_tokens = 8 * N  # 16 frames → 8 tubelets

        ctx_indices = torch.cat([
            torch.arange(0, tubelet_start, device=device),
            torch.arange(tubelet_end, total_tokens, device=device),
        ])
        tgt_indices = torch.arange(tubelet_start, tubelet_end, device=device)

        clips_list = [frames_256[b:b + 1] for b in range(B)]
        masks_enc = [[ctx_indices.unsqueeze(0)] for _ in range(B)]
        masks_pred = [[tgt_indices.unsqueeze(0)] for _ in range(B)]

        h_list = self.target_encoder(clips_list)
        h_target = torch.cat(
            [h_list[b][:, tubelet_start:tubelet_end, :] for b in range(B)], dim=0
        )

        z = self.context_encoder(clips_list, masks=masks_enc)
        pred_out = self.predictor(z, masks_enc, masks_pred)
        pred = torch.cat([pred_out[b][0] for b in range(B)], dim=0)

        loss = torch.abs(pred - h_target).mean(dim=-1)
        return loss

    @torch.no_grad()
    def score_window(self, frames_256: torch.Tensor) -> PredictMemScore:
        """Score all 8 tubelets (offline mode, backward compat)."""
        B, device = frames_256.shape[0], frames_256.device

        all_losses = []
        for t in range(8):
            loss_t = self.score_tubelet(frames_256, target_tubelet_id=t)
            all_losses.append(loss_t)

        loss_map = torch.stack(all_losses, dim=1).view(B, 8, self.grid_h, self.grid_w)
        keep_mask, keep_indices = self._loss_to_keep_mask(loss_map)

        return PredictMemScore(
            loss_map=loss_map,
            keep_mask=keep_mask,
            keep_indices=keep_indices,
        )

    @torch.no_grad()
    def score_window_online(
        self,
        frames_256: torch.Tensor,
        new_tubelet_id: int,
        history_keep_mask: torch.Tensor | None = None,
    ) -> PredictMemScore:
        """Online single-tubelet scoring (backward compat)."""
        B, device = frames_256.shape[0], frames_256.device

        loss_new = self.score_tubelet(frames_256, target_tubelet_id=new_tubelet_id)

        loss_map = torch.zeros(B, 8, self.grid_h, self.grid_w, device=device)
        loss_map[:, new_tubelet_id, :, :] = loss_new.view(B, self.grid_h, self.grid_w)

        num_keep_per_tubelet = max(1, int(self.num_tubelet_patches * self.config.keep_ratio))
        keep_mask = torch.zeros(B, 8, self.grid_h, self.grid_w, dtype=torch.bool, device=device)
        keep_indices = []

        if history_keep_mask is not None:
            keep_mask = history_keep_mask.clone()
            keep_mask[:, new_tubelet_id, :, :] = False

        for b in range(B):
            new_loss = loss_new[b]
            if self.config.min_cell_keep:
                keep_local = self._topk_2d_with_cell_coverage(
                    new_loss.view(self.grid_h, self.grid_w), num_keep_per_tubelet
                )
            else:
                _, top = torch.topk(new_loss, num_keep_per_tubelet)
                keep_local = top

            keep_h = keep_local // self.grid_w
            keep_w = keep_local % self.grid_w
            keep_mask[b, new_tubelet_id, keep_h, keep_w] = True

            all_kept = torch.where(keep_mask[b].flatten())[0]
            keep_indices.append(all_kept.sort().values)

        return PredictMemScore(
            loss_map=loss_map,
            keep_mask=keep_mask,
            keep_indices=keep_indices,
        )

    def _topk_2d_with_cell_coverage(self, loss_2d: torch.Tensor, total_keep: int) -> torch.Tensor:
        G = self.config.cell_grid_size
        H, W = loss_2d.shape
        cells_h, cells_w = H // G, W // G

        cell_kept = set()
        for ci in range(G):
            for cj in range(G):
                h_start, h_end = ci * cells_h, (ci + 1) * cells_h
                w_start, w_end = cj * cells_w, (cj + 1) * cells_w
                cell_loss = loss_2d[h_start:h_end, w_start:w_end]
                best_local = cell_loss.argmax().item()
                best_h = best_local // cells_w
                best_w = best_local % cells_w
                best_global = (h_start + best_h) * W + (w_start + best_w)
                cell_kept.add(best_global)

        min_keeps = torch.tensor(sorted(cell_kept), device=loss_2d.device)
        remaining = total_keep - len(min_keeps)

        if remaining > 0:
            flat = loss_2d.flatten()
            mask = torch.ones(len(flat), dtype=torch.bool, device=loss_2d.device)
            mask[min_keeps] = False
            remaining_loss = flat[mask]
            remaining_indices = torch.arange(len(flat), device=loss_2d.device)[mask]
            _, top = torch.topk(remaining_loss, min(remaining, len(remaining_loss)))
            extra = remaining_indices[top]
            return torch.cat([min_keeps, extra]).sort().values

        return min_keeps

    def _loss_to_keep_mask(self, loss_map: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        B, T, H, W = loss_map.shape
        device = loss_map.device
        num_total = T * H * W
        num_keep = max(1, int(num_total * self.config.keep_ratio))

        keep_mask = torch.zeros(B, T, H, W, dtype=torch.bool, device=device)
        keep_indices = []

        for b in range(B):
            flat_loss = loss_map[b].flatten()
            if self.config.min_cell_keep:
                keep_indices_b = self._topk_with_cell_coverage(flat_loss, num_keep)
            elif self.config.score_mode == "rank":
                _, top_indices = torch.topk(flat_loss, num_keep)
                keep_indices_b = top_indices
            elif self.config.score_mode == "zscore":
                mean, std = flat_loss.mean(), flat_loss.std()
                z = (flat_loss - mean) / (std + 1e-8)
                keep_indices_b = torch.where(z > 0)[0]
                if len(keep_indices_b) > num_keep:
                    _, top = torch.topk(z[keep_indices_b], num_keep)
                    keep_indices_b = keep_indices_b[top]
            else:
                threshold = flat_loss.quantile(1.0 - self.config.keep_ratio)
                keep_indices_b = torch.where(flat_loss >= threshold)[0]

            keep_indices_b = keep_indices_b.sort().values
            keep_indices.append(keep_indices_b)

            thw = keep_indices_to_thw(keep_indices_b, H, W)
            keep_mask[b, thw[:, 0], thw[:, 1], thw[:, 2]] = True

        return keep_mask, keep_indices

    def _topk_with_cell_coverage(self, flat_loss: torch.Tensor, total_keep: int) -> torch.Tensor:
        G = self.config.cell_grid_size
        T = 8
        H, W = self.grid_h, self.grid_w
        cells_h, cells_w = H // G, W // G

        cell_kept = set()
        for ti in range(T):
            for ci in range(G):
                for cj in range(G):
                    h_start, h_end = ci * cells_h, (ci + 1) * cells_h
                    w_start, w_end = cj * cells_w, (cj + 1) * cells_w
                    cell_loss = flat_loss.view(T, H, W)[ti, h_start:h_end, w_start:w_end]
                    best_local = cell_loss.argmax().item()
                    best_h = best_local // cells_w
                    best_w = best_local % cells_w
                    best_global = ti * H * W + (h_start + best_h) * W + (w_start + best_w)
                    cell_kept.add(best_global)

        min_cell_keeps = torch.tensor(sorted(cell_kept), device=flat_loss.device)
        remaining = total_keep - len(min_cell_keeps)

        if remaining > 0:
            all_indices = torch.arange(len(flat_loss), device=flat_loss.device)
            mask = torch.ones(len(flat_loss), dtype=torch.bool, device=flat_loss.device)
            mask[min_cell_keeps] = False
            remaining_loss = flat_loss[mask]
            remaining_indices = all_indices[mask]
            _, top_remaining = torch.topk(remaining_loss, min(remaining, len(remaining_loss)))
            extra_keeps = remaining_indices[top_remaining]
            return torch.cat([min_cell_keeps, extra_keeps]).sort().values

        return min_cell_keeps


def keep_indices_to_thw(indices: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert flat indices to (t, h, w) coordinates."""
    area = H * W
    t = indices // area
    residual = indices % area
    h = residual // W
    w = residual % W
    return torch.stack([t, h, w], dim=-1)


# ─── Analyzer-compatible checkpoint loading ────────────────────────────────────

def _clean_state_dict(state_dict: dict) -> dict:
    """Strip 'module.backbone.' prefix (same as Survey analyzer)."""
    out = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.backbone.", "")
        out[new_k] = v
    return out


def load_vjepa_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    strict_encoder: bool = False,
    strict_predictor: bool = False,
) -> dict[str, dict]:
    """Load a V-JEPA checkpoint and extract recognized sub-model states.

    Uses ``weights_only=True`` for safety (same as the Survey analyzer).
    """
    checkpoint_path = Path(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    recognized = {}
    for key in ["encoder", "target_encoder", "ema_encoder", "predictor"]:
        if key in state_dict:
            recognized[key] = _clean_state_dict(state_dict[key])

    if not recognized:
        recognized["encoder"] = _clean_state_dict(state_dict)

    return recognized


def make_vjepa_analyzer_scorer(
    img_size: int = 256,
    patch_size: int = 16,
    num_frames: int = 16,
    tubelet_size: int = 2,
    embed_dim: int = 1024,
    encoder_depth: int = 24,
    encoder_num_heads: int = 16,
    predictor_embed_dim: int = 384,
    predictor_depth: int = 12,
    predictor_num_heads: int = 12,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    strict_encoder: bool = False,
    strict_predictor: bool = False,
) -> dict:
    """Build V-JEPA encoders + predictor wrapped like the Survey analyzer.

    Uses MultiSeqWrapper and PredictorMultiSeqWrapper so that the mask
    format, forward paths, and variable-length window handling match
    ``/root/stream/Survey/survey/vjepa_loss_analyzer/analyzer.py`` exactly.

    Returns:
        dict with keys: context_encoder, target_encoder, predictor, degraded,
        keys_found, missing_keys, unexpected_keys, num_mask_tokens, wrapper_type
    """
    import sys
    vjepa_src = Path(__file__).parent.parent.parent / "site-packages" / "vjepa2"
    if str(vjepa_src) not in sys.path:
        sys.path.insert(0, str(vjepa_src))
    from src.models import vision_transformer as vit_encoder
    from src.models import predictor as vit_predictor
    from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

    encoder_kwargs = dict(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
    )

    predictor_kwargs = dict(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=predictor_depth,
        num_heads=predictor_num_heads,
        use_mask_tokens=True,
        num_mask_tokens=10,  # analyzer-compatible
        use_rope=True,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
    )

    raw_encoder = vit_encoder.vit_large(**encoder_kwargs)
    raw_target_encoder = vit_encoder.vit_large(**encoder_kwargs)
    raw_predictor = vit_predictor.vit_predictor(**predictor_kwargs)

    keys_found = []
    missing_enc = []
    missing_target = []
    missing_pred = []
    unexpected_enc = []
    unexpected_target = []
    unexpected_pred = []

    degraded = False

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = load_vjepa_checkpoint(checkpoint_path, device=device)
        keys_found = list(ckpt.keys())

        # Load target encoder (prefer target_encoder > ema_encoder > encoder)
        target_key = None
        if "target_encoder" in ckpt:
            target_key = "target_encoder"
        elif "ema_encoder" in ckpt:
            target_key = "ema_encoder"
        elif "encoder" in ckpt:
            target_key = "encoder"
            degraded = True

        if target_key:
            msg = raw_target_encoder.load_state_dict(ckpt[target_key], strict=strict_encoder)
            missing_target = msg.missing_keys
            unexpected_target = msg.unexpected_keys

        # Load context encoder (prefer encoder > target_encoder)
        context_key = None
        if "encoder" in ckpt:
            context_key = "encoder"
        elif "target_encoder" in ckpt:
            context_key = "target_encoder"
            degraded = True

        if context_key:
            msg = raw_encoder.load_state_dict(ckpt[context_key], strict=strict_encoder)
            missing_enc = msg.missing_keys
            unexpected_enc = msg.unexpected_keys

        # Load predictor
        if "predictor" in ckpt:
            msg = raw_predictor.load_state_dict(ckpt["predictor"], strict=strict_predictor)
            missing_pred = msg.missing_keys
            unexpected_pred = msg.unexpected_keys

    # Wrap in MultiSeqWrapper / PredictorMultiSeqWrapper (analyzer style)
    context_encoder = MultiSeqWrapper(raw_encoder).to(device).eval()
    target_encoder = MultiSeqWrapper(raw_target_encoder).to(device).eval()
    predictor = PredictorMultiSeqWrapper(raw_predictor).to(device).eval()

    for p in target_encoder.parameters():
        p.requires_grad = False

    # Print diagnostics
    print(f"[make_vjepa_analyzer_scorer] encoder missing_keys={len(missing_enc)}, "
          f"unexpected_keys={len(unexpected_enc)}")
    print(f"[make_vjepa_analyzer_scorer] target_encoder missing_keys={len(missing_target)}, "
          f"unexpected_keys={len(unexpected_target)}")
    print(f"[make_vjepa_analyzer_scorer] predictor missing_keys={len(missing_pred)}, "
          f"unexpected_keys={len(unexpected_pred)}")
    print(f"[make_vjepa_analyzer_scorer] predictor.num_mask_tokens={raw_predictor.num_mask_tokens}")
    print(f"[make_vjepa_analyzer_scorer] wrapper_type=MultiSeqWrapper/PredictorMultiSeqWrapper")
    print(f"[make_vjepa_analyzer_scorer] keys_found={keys_found}")

    return {
        "context_encoder": context_encoder,
        "target_encoder": target_encoder,
        "predictor": predictor,
        "degraded": degraded,
        "keys_found": keys_found,
        "missing_keys": {
            "encoder": missing_enc,
            "target_encoder": missing_target,
            "predictor": missing_pred,
        },
        "unexpected_keys": {
            "encoder": unexpected_enc,
            "target_encoder": unexpected_target,
            "predictor": unexpected_pred,
        },
        "num_mask_tokens": raw_predictor.num_mask_tokens,
        "wrapper_type": "MultiSeqWrapper/PredictorMultiSeqWrapper",
    }


def make_vjepa_encoder_predictor(
    img_size: int = 256,
    patch_size: int = 16,
    num_frames: int = 16,
    tubelet_size: int = 2,
    embed_dim: int = 1024,
    encoder_depth: int = 24,
    encoder_num_heads: int = 16,
    predictor_embed_dim: int = 384,
    predictor_depth: int = 12,
    predictor_num_heads: int = 12,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    strict_encoder: bool = False,
    strict_predictor: bool = False,
) -> dict:
    """Legacy builder — delegates to ``make_vjepa_analyzer_scorer``.

    Kept for backward compatibility with offline/legacy scripts.
    """
    return make_vjepa_analyzer_scorer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        predictor_embed_dim=predictor_embed_dim,
        predictor_depth=predictor_depth,
        predictor_num_heads=predictor_num_heads,
        checkpoint_path=checkpoint_path,
        device=device,
        strict_encoder=strict_encoder,
        strict_predictor=strict_predictor,
    )
