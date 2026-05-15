"""V-JEPA prediction loss scorer for PredictMem.

Takes a 16-frame 256x256 video window, runs it through V-JEPA encoder + predictor,
and produces per-patch prediction loss. High-loss patches are harder to predict and
thus more likely to contain informative content worth keeping.
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

    Main entry point::

        scorer = VJEPAPredictLossScorer(config, encoder, predictor)
        score = scorer.score_window(frames_256)  # [B, 3, 16, 256, 256] -> PredictMemScore
    """

    def __init__(
        self,
        config: PredictMemConfig,
        encoder: nn.Module,
        predictor: nn.Module,
    ):
        self.config = config
        self.encoder = encoder
        self.predictor = predictor
        self.grid_t = config.jepa_grid_t  # 8
        self.grid_h = config.jepa_grid_h  # 16
        self.grid_w = config.jepa_grid_w  # 16
        self.num_patches = self.grid_t * self.grid_h * self.grid_w  # 2048

    @torch.no_grad()
    def score_window(self, frames_256: torch.Tensor) -> PredictMemScore:
        """Score a 16-frame video window.

        Args:
            frames_256: [B, 3, 16, 256, 256] float tensor in [0, 1] or normalized.

        Returns:
            PredictMemScore with loss_map, keep_mask, and keep_indices.
        """
        B = frames_256.shape[0]
        device = frames_256.device

        # 1. Run encoder on full input to get target latents for all patches
        full_embeddings = self.encoder(frames_256)  # [B, 2048, D]
        # The last tubelet (tubelet 7) corresponds to positions 1792..2047
        target_embeddings = full_embeddings[:, -256:, :]  # [B, 256, D]

        # 2. Run encoder on context-only input (mask out last tubelet)
        context_indices = torch.arange(0, 256 * 7, device=device)  # 0..1791
        context_mask = context_indices.unsqueeze(0).expand(B, -1)  # [B, 1792]
        context_embeddings = self._encode_masked(frames_256, context_mask)  # [B, 1792, D]

        # 3. Run predictor: context -> target
        target_indices = torch.arange(256 * 7, 256 * 8, device=device)  # 1792..2047
        target_mask = target_indices.unsqueeze(0).expand(B, -1)  # [B, 256]
        predictions = self._predict(context_embeddings, context_mask, target_mask)  # [B, 256, D]

        # 4. Compute per-patch prediction loss
        loss = (predictions - target_embeddings).abs().pow(2).mean(dim=-1)  # [B, 256]

        # 5. Assemble full loss_map (only target tubelet has scores; context set to 0)
        loss_map = torch.zeros(B, self.num_patches, device=device)
        loss_map[:, 256 * 7 : 256 * 8] = loss
        loss_map = loss_map.view(B, self.grid_t, self.grid_h, self.grid_w)

        # 6. Generate keep mask from loss map
        keep_mask, keep_indices = self._loss_to_keep_mask(loss_map)

        return PredictMemScore(
            loss_map=loss_map,
            keep_mask=keep_mask,
            keep_indices=keep_indices,
        )

    def _encode_masked(self, frames: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
        """Run encoder with only specified patches kept.

        Args:
            frames: [B, 3, 16, 256, 256]
            keep_indices: [B, K] flat indices of patches to keep

        Returns:
            [B, K, D] embeddings for kept patches.
        """
        # Run encoder on full input, then gather kept positions
        full_emb = self.encoder(frames)  # [B, 2048, D]
        idx = keep_indices.unsqueeze(-1).expand(-1, -1, full_emb.size(-1))
        return torch.gather(full_emb, dim=1, index=idx)

    def _predict(
        self,
        context_embeddings: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target latents from context.

        Args:
            context_embeddings: [B, N_ctxt, D] encoded context patches
            context_mask: [B, N_ctxt] indices of context patches in full grid
            target_mask: [B, N_tgt] indices of target patches in full grid

        Returns:
            [B, N_tgt, D] predicted target latents.
        """
        return self.predictor(context_embeddings, masks_x=context_mask, masks_y=target_mask)

    def _loss_to_keep_mask(self, loss_map: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Convert loss map to keep mask based on score_mode and keep_ratio.

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
        and ensures each cell keeps at least one patch.
        """
        G = self.config.cell_grid_size  # 4
        T = self.grid_t  # 8
        H = self.grid_h  # 16
        W = self.grid_w  # 16
        cells_h, cells_w = H // G, W // G  # 4x4

        # First, pick the best patch in each cell
        cell_kept = set()
        loss_2d = flat_loss.view(T * H, W)
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
            # Mask out already-kept, pick top-k from remaining
            all_indices = torch.arange(len(flat_loss), device=flat_loss.device)
            mask = torch.ones(len(flat_loss), dtype=torch.bool, device=flat_loss.device)
            mask[min_cell_keeps] = False
            remaining_loss = flat_loss[mask]
            remaining_indices = all_indices[mask]
            _, top_remaining = torch.topk(remaining_loss, min(remaining, len(remaining_loss)))
            extra_keeps = remaining_indices[top_remaining]
            return torch.cat([min_cell_keeps, extra_keeps]).sort().values

        return min_cell_keeps

    def compute_loss_only(self, frames_256: torch.Tensor, tubelet_id: int = 7) -> torch.Tensor:
        """Compute prediction loss for a specific tubelet without masking.

        Simplified path: one encoder pass, extract target tubelet and remaining
        context, predict, compute loss.

        Args:
            frames_256: [B, 3, 16, 256, 256]
            tubelet_id: Which tubelet to score (default 7 = last).

        Returns:
            loss_map: [B, 256] flat per-patch loss for the target tubelet.
        """
        B = frames_256.shape[0]
        device = frames_256.device
        H, W = self.grid_h, self.grid_w
        tubelet_start = tubelet_id * H * W
        tubelet_end = tubelet_start + H * W

        # Single encoder pass
        full_emb = self.encoder(frames_256)  # [B, 2048, D]
        target_emb = full_emb[:, tubelet_start:tubelet_end, :]

        # Context: all tokens except target tubelet
        context_indices = torch.cat([
            torch.arange(0, tubelet_start, device=device),
            torch.arange(tubelet_end, self.num_patches, device=device),
        ])
        context_mask = context_indices.unsqueeze(0).expand(B, -1)
        context_emb = full_emb[:, context_indices, :]

        # Predict target
        target_indices = torch.arange(tubelet_start, tubelet_end, device=device)
        target_mask = target_indices.unsqueeze(0).expand(B, -1)
        pred = self.predictor(context_emb, masks_x=context_mask, masks_y=target_mask)

        loss = (pred - target_emb).abs().pow(2).mean(dim=-1)  # [B, 256]
        return loss


def keep_indices_to_thw(indices: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert flat indices to (t, h, w) coordinates.

    Args:
        indices: [K] flat indices in [0, T*H*W)
        H: grid height
        W: grid width

    Returns:
        [K, 3] tensor of (t, h, w) coordinates.
    """
    area = H * W
    t = indices // area
    residual = indices % area
    h = residual // W
    w = residual % W
    return torch.stack([t, h, w], dim=-1)


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
) -> tuple[nn.Module, nn.Module]:
    """Build V-JEPA encoder + predictor, optionally loading pretrained weights.

    Args:
        img_size: Spatial input size (256 for standard V-JEPA).
        patch_size: Patch size (16).
        num_frames: Number of input frames (16).
        tubelet_size: Temporal patch size (2).
        embed_dim: Encoder embedding dimension.
        encoder_depth: Encoder depth.
        encoder_num_heads: Encoder number of heads.
        predictor_embed_dim: Predictor embedding dimension.
        predictor_depth: Predictor depth.
        predictor_num_heads: Predictor number of heads.
        checkpoint_path: Optional path to pretrained .pt checkpoint.
        device: Device to load models on.

    Returns:
        (encoder, predictor) tuple.
    """
    import sys
    vjepa_src = Path(__file__).parent.parent.parent / "site-packages" / "vjepa2"
    if str(vjepa_src) not in sys.path:
        sys.path.insert(0, str(vjepa_src))
    from src.models import vision_transformer as vit_encoder
    from src.models import predictor as vit_predictor

    encoder = vit_encoder.vit_large(
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

    predictor = vit_predictor.vit_predictor(
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

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(state_dict.get("encoder", state_dict), strict=False)
        predictor.load_state_dict(state_dict.get("predictor", {}), strict=False)

    encoder.to(device)
    predictor.to(device)
    return encoder, predictor
