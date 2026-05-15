"""V-JEPA prediction loss scorer for PredictMem.

Takes a 16-frame 256x256 video window, runs it through V-JEPA encoder + predictor,
and produces per-patch prediction loss. High-loss patches are harder to predict and
thus more likely to contain informative content worth keeping.

P0 fixes applied:
- Target encoder (EMA) vs context encoder separation (no cross-attention leakage)
- Masked context encoding: context encoder only sees context patches
- loss_exp in config, unified loss = mean(abs(pred - target) ** loss_exp) / loss_exp
- Online tubelet scoring: only produce keep mask for the current new tubelet
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .config import PredictMemConfig


@dataclass
class PredictMemScore:
    """Output of V-JEPA scoring for one window."""

    loss_map: torch.Tensor  # [B, 8, 16, 16]
    keep_mask: torch.Tensor  # [B, 8, 16, 16], bool
    keep_indices: list[torch.Tensor]  # per-sample flat local token indices


class VJEPAPredictLossScorer:
    """V-JEPA-based prediction loss scorer for video token pruning.

    Uses separate target encoder (EMA) and context encoder.
    Context encoder runs with target patches masked out to prevent leakage.

    Main entry point::

        scorer = VJEPAPredictLossScorer(config, context_encoder, target_encoder, predictor)
        score = scorer.score_window(frames_256)  # [B, 3, 16, 256, 256] -> PredictMemScore
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
        self.degraded = degraded  # True if context_encoder == target_encoder
        self.grid_t = config.jepa_grid_t  # 8
        self.grid_h = config.jepa_grid_h  # 16
        self.grid_w = config.jepa_grid_w  # 16
        self.num_patches = self.grid_t * self.grid_h * self.grid_w  # 2048
        self.num_tubelet_patches = self.grid_h * self.grid_w  # 256
        self.num_tubelets = self.grid_t  # 8

    @torch.no_grad()
    def score_tubelet(
        self,
        frames_256: torch.Tensor,
        target_tubelet_id: int,
    ) -> torch.Tensor:
        """Score a single tubelet: compute prediction loss for target_tubelet_id.

        Context = all patches EXCEPT the target tubelet's patches.
        Target encoder sees full input; context encoder sees context only.

        Args:
            frames_256: [B, 3, 16, 256, 256]
            target_tubelet_id: 0..7, which tubelet to predict

        Returns:
            loss: [B, 256] per-patch prediction loss for the target tubelet.
        """
        B, device = frames_256.shape[0], frames_256.device
        H, W = self.grid_h, self.grid_w
        N = self.num_tubelet_patches  # 256
        tubelet_start = target_tubelet_id * N
        tubelet_end = tubelet_start + N

        # Context mask: all patches except the target tubelet
        context_indices = torch.cat([
            torch.arange(0, tubelet_start, device=device),
            torch.arange(tubelet_end, self.num_patches, device=device),
        ])
        context_mask_tensor = context_indices.unsqueeze(0).expand(B, -1)  # [B, N_ctxt]

        # Target mask: the target tubelet patches
        target_indices = torch.arange(tubelet_start, tubelet_end, device=device)
        target_mask_tensor = target_indices.unsqueeze(0).expand(B, -1)  # [B, N]

        # Target encoder: full input, no mask  → target latents
        target_embeddings = self.target_encoder(frames_256)  # [B, 2048, D]
        target_tubelet_emb = target_embeddings[:, tubelet_start:tubelet_end, :]  # [B, 256, D]

        # Context encoder: masked input, only context patches
        # V-JEPA encoder.forward(x, masks=[...]) where masks is list of [B, K] keep indices
        context_embeddings = self.context_encoder(
            frames_256, masks=[context_mask_tensor]
        )  # [B, N_ctxt, D]

        # Predictor: context → target
        predictions = self.predictor(
            context_embeddings,
            masks_x=context_mask_tensor,
            masks_y=target_mask_tensor,
        )  # [B, N, D]

        # Loss: mean(abs(pred - target) ** loss_exp) / loss_exp
        p = self.config.loss_exp
        diff = predictions - target_tubelet_emb
        loss = diff.abs().pow(p).mean(dim=-1) / p  # [B, N]

        return loss

    @torch.no_grad()
    def score_window(self, frames_256: torch.Tensor) -> PredictMemScore:
        """Score a 16-frame video window in offline mode.

        Offline: scores all 8 tubelets, assembles full [8,16,16] loss map.

        Args:
            frames_256: [B, 3, 16, 256, 256]

        Returns:
            PredictMemScore with loss_map, keep_mask, keep_indices.
        """
        B, device = frames_256.shape[0], frames_256.device

        # Score each tubelet independently
        all_tubelet_losses = []
        for t in range(self.num_tubelets):
            loss_t = self.score_tubelet(frames_256, target_tubelet_id=t)  # [B, 256]
            all_tubelet_losses.append(loss_t)

        # Assemble full loss map: [B, 8, 16, 16]
        loss_map = torch.stack(all_tubelet_losses, dim=1).view(B, self.grid_t, self.grid_h, self.grid_w)

        # Generate keep mask
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
        """Online mode: only score the current new tubelet, merge with history.

        Args:
            frames_256: [B, 3, 16, 256, 256]
            new_tubelet_id: 0..7, the newly arrived 2-frame tubelet to score
            history_keep_mask: [B, 8, 16, 16] or None, previous tubelets' keep decisions

        Returns:
            PredictMemScore where keep_mask merges new decision with history.
        """
        B, device = frames_256.shape[0], frames_256.device

        # Score only the new tubelet
        loss_new = self.score_tubelet(frames_256, target_tubelet_id=new_tubelet_id)  # [B, 256]

        # Build loss map: zero for history, new loss for new tubelet
        loss_map = torch.zeros(B, self.grid_t, self.grid_h, self.grid_w, device=device)
        loss_map[:, new_tubelet_id, :, :] = loss_new.view(B, self.grid_h, self.grid_w)

        # Compute keep decision for new tubelet only
        num_keep_per_tubelet = max(1, int(self.num_tubelet_patches * self.config.keep_ratio))
        keep_mask = torch.zeros(B, self.grid_t, self.grid_h, self.grid_w, dtype=torch.bool, device=device)
        keep_indices = []

        # Restore history keep mask
        if history_keep_mask is not None:
            keep_mask = history_keep_mask.clone()
            # Zero out the new tubelet in history (will be recomputed)
            keep_mask[:, new_tubelet_id, :, :] = False

        for b in range(B):
            new_loss = loss_new[b]  # [256]
            if self.config.min_cell_keep:
                keep_local = self._topk_2d_with_cell_coverage(
                    new_loss.view(self.grid_h, self.grid_w), num_keep_per_tubelet
                )
            elif self.config.score_mode == "rank":
                _, top = torch.topk(new_loss, num_keep_per_tubelet)
                keep_local = top
            else:
                _, top = torch.topk(new_loss, num_keep_per_tubelet)
                keep_local = top

            # Set keep in new tubelet
            keep_h = keep_local // self.grid_w
            keep_w = keep_local % self.grid_w
            keep_mask[b, new_tubelet_id, keep_h, keep_w] = True

            # Build flat keep indices for entire window
            all_kept = torch.where(keep_mask[b].flatten())[0]
            keep_indices.append(all_kept.sort().values)

        # Fill loss map with full window loss if we want it (cache it)
        # For online, we only compute the new tubelet loss
        return PredictMemScore(
            loss_map=loss_map,
            keep_mask=keep_mask,
            keep_indices=keep_indices,
        )

    def _topk_2d_with_cell_coverage(
        self, loss_2d: torch.Tensor, total_keep: int
    ) -> torch.Tensor:
        """Per-tubelet top-k with spatial cell coverage on the 16x16 grid.

        Args:
            loss_2d: [H, W] = [16, 16] loss within one tubelet
            total_keep: number of patches to keep

        Returns:
            [K] flat indices within the 16x16 grid.
        """
        G = self.config.cell_grid_size  # 4
        H, W = loss_2d.shape  # 16, 16
        cells_h, cells_w = H // G, W // G  # 4x4

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
        """Convert full loss map to keep mask based on score_mode and keep_ratio.

        Args:
            loss_map: [B, T, H, W] per-patch prediction loss

        Returns:
            keep_mask: [B, T, H, W] bool tensor
            keep_indices: list of [K_b] flat index tensors per sample
        """
        B, T, H, W = loss_map.shape
        device = loss_map.device
        num_total = T * H * W  # 2048
        num_keep = max(1, int(num_total * self.config.keep_ratio))

        keep_mask = torch.zeros(B, T, H, W, dtype=torch.bool, device=device)
        keep_indices = []

        for b in range(B):
            flat_loss = loss_map[b].flatten()  # [2048]

            if self.config.min_cell_keep:
                keep_indices_b = self._topk_with_cell_coverage(flat_loss, num_keep)
            else:
                if self.config.score_mode == "rank":
                    _, top_indices = torch.topk(flat_loss, num_keep)
                    keep_indices_b = top_indices
                elif self.config.score_mode == "zscore":
                    mean, std = flat_loss.mean(), flat_loss.std()
                    z = (flat_loss - mean) / (std + 1e-8)
                    keep_indices_b = torch.where(z > 0)[0]
                    if len(keep_indices_b) > num_keep:
                        _, top = torch.topk(z[keep_indices_b], num_keep)
                        keep_indices_b = keep_indices_b[top]
                else:  # raw
                    threshold = flat_loss.quantile(1.0 - self.config.keep_ratio)
                    keep_indices_b = torch.where(flat_loss >= threshold)[0]

            keep_indices_b = keep_indices_b.sort().values
            keep_indices.append(keep_indices_b)

            thw = keep_indices_to_thw(keep_indices_b, H, W)
            keep_mask[b, thw[:, 0], thw[:, 1], thw[:, 2]] = True

        return keep_mask, keep_indices

    def _topk_with_cell_coverage(self, flat_loss: torch.Tensor, total_keep: int) -> torch.Tensor:
        """Top-k selection with minimum coverage per spatial cell.

        Divides the 16x16 spatial grid into cell_grid_size x cell_grid_size cells
        and ensures each cell keeps at least one patch across all tubelets.
        """
        G = self.config.cell_grid_size  # 4
        T = self.grid_t  # 8
        H = self.grid_h  # 16
        W = self.grid_w  # 16
        cells_h, cells_w = H // G, W // G  # 4x4

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


# ─── Checkpoint loader ────────────────────────────────────────────────────────

def _clean_backbone_keys(state_dict: dict) -> dict:
    """Strip module./backbone. prefixes from checkpoint keys."""
    new = {}
    for key, val in state_dict.items():
        k = key.replace("module.", "").replace("backbone.", "")
        new[k] = val
    return new


def load_vjepa_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    strict_encoder: bool = False,
    strict_predictor: bool = False,
) -> dict[str, dict]:
    """Load a V-JEPA checkpoint and extract all recognized sub-model states.

    Recognized top-level keys:
        encoder, target_encoder, ema_encoder, predictor

    Each undergoes module./backbone. key stripping.

    Returns:
        dict with keys found in the checkpoint, e.g.:
        {"encoder": {...}, "target_encoder": {...}, "predictor": {...}}
    """
    checkpoint_path = Path(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)

    recognized = {}
    for key in ["encoder", "target_encoder", "ema_encoder", "predictor"]:
        if key in state_dict:
            recognized[key] = _clean_backbone_keys(state_dict[key])

    if not recognized:
        # Maybe the whole state_dict is a raw encoder
        recognized["encoder"] = _clean_backbone_keys(state_dict)

    return recognized


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
    """Build V-JEPA context/target encoders + predictor, loading weights if available.

    Returns:
        dict with keys: context_encoder, target_encoder, predictor, degraded, keys_found
    """
    import sys
    vjepa_src = Path(__file__).parent.parent.parent / "site-packages" / "vjepa2"
    if str(vjepa_src) not in sys.path:
        sys.path.insert(0, str(vjepa_src))
    from src.models import vision_transformer as vit_encoder
    from src.models import predictor as vit_predictor

    # Build encoders
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

    context_encoder = vit_encoder.vit_large(**encoder_kwargs)
    target_encoder = vit_encoder.vit_large(**encoder_kwargs)

    # Build predictor
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
        num_mask_tokens=2,
        use_rope=True,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
    )
    predictor = vit_predictor.vit_predictor(**predictor_kwargs)

    degraded = False
    keys_found = []

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
            target_encoder.load_state_dict(ckpt[target_key], strict=strict_encoder)

        # Load context encoder (prefer encoder > target_encoder)
        context_key = None
        if "encoder" in ckpt:
            context_key = "encoder"
        elif "target_encoder" in ckpt:
            context_key = "target_encoder"
            degraded = True

        if context_key:
            context_encoder.load_state_dict(ckpt[context_key], strict=strict_encoder)

        # Load predictor
        if "predictor" in ckpt:
            predictor.load_state_dict(ckpt["predictor"], strict=strict_predictor)

    context_encoder.to(device)
    target_encoder.to(device)
    predictor.to(device)
    context_encoder.eval()
    target_encoder.eval()
    predictor.eval()

    return {
        "context_encoder": context_encoder,
        "target_encoder": target_encoder,
        "predictor": predictor,
        "degraded": degraded,
        "keys_found": keys_found,
    }
