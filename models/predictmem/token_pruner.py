"""Token pruning for PredictMem.

Prunes Qwen video placeholder tokens according to V-JEPA keep masks, producing
right-padded, shorter sequences for the language model.
"""

import torch

from .config import PredictMemConfig


class TokenPruner:
    """Prunes Qwen visual placeholder tokens by keep mask.

    Non-video tokens (text, timestamps, vision_start/vision_end) are always kept.
    Only video_token_id placeholders are selectively pruned.
    """

    def __init__(
        self,
        config: PredictMemConfig | None,
        video_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
    ):
        self.config = config
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

    def prune(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        video_keep_indices: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prune video tokens from each sample independently.

        Args:
            input_ids: [B, L] - token ids
            inputs_embeds: [B, L, D] - input embeddings (with video features already
                scattered into video placeholder positions)
            position_ids: [3, B, L] - 3D M-RoPE position ids
            attention_mask: [B, L] or None - attention mask
            video_keep_indices: list of [K_b] tensors, one per batch.
                Each tensor contains local indices (0..num_video_tokens-1) of
                video placeholder tokens to KEEP.

        Returns:
            (inputs_embeds, position_ids, attention_mask) pruned and padded.
            Shapes are [B, L_new, D], [P, B, L_new], [B, L_new].
        """
        if input_ids is None:
            raise ValueError("TokenPruner requires input_ids to locate video placeholders")
        if position_ids is None:
            raise ValueError("TokenPruner requires position_ids before pruning")

        B, L, D = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        keep_masks = self.build_sequence_keep_masks(
            input_ids=input_ids,
            video_keep_indices=video_keep_indices,
            attention_mask=attention_mask,
        )

        new_embeds_list = []
        new_pos_list = []
        max_len = 0

        for b in range(B):
            is_kept = keep_masks[b].to(device)
            new_embeds_list.append(inputs_embeds[b][is_kept])
            # position_ids: [P, B, L] -> select columns (P=3 for Qwen3.5, P=4 for Qwen3-VL)
            new_pos_list.append(position_ids[:, b, is_kept])
            max_len = max(max_len, is_kept.sum().item())

        # Pad to max length
        new_L = max_len
        P = position_ids.shape[0]  # 3 or 4
        padded_embeds = torch.zeros(B, new_L, D, device=device, dtype=dtype)
        padded_pos = torch.zeros(P, B, new_L, device=device, dtype=position_ids.dtype)
        mask_dtype = attention_mask.dtype if attention_mask is not None else torch.long
        new_attention_mask = torch.zeros(B, new_L, device=device, dtype=mask_dtype)

        for b in range(B):
            cur_len = new_embeds_list[b].shape[0]
            padded_embeds[b, :cur_len] = new_embeds_list[b]
            padded_pos[:, b, :cur_len] = new_pos_list[b]
            new_attention_mask[b, :cur_len] = 1

        return padded_embeds, padded_pos, new_attention_mask

    def build_sequence_keep_masks(
        self,
        input_ids: torch.LongTensor,
        video_keep_indices: list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Build per-sample boolean masks over the original sequence length.

        Non-video valid tokens are preserved. Padding tokens are preserved only
        when no ``attention_mask`` is supplied.
        """
        if input_ids is None:
            raise ValueError("TokenPruner requires input_ids to locate video placeholders")
        B, L = input_ids.shape
        keep_indices = self._normalize_keep_indices(video_keep_indices, B)
        device = input_ids.device

        sequence_keep_masks: list[torch.Tensor] = []
        for b in range(B):
            valid = torch.ones(L, dtype=torch.bool, device=device)
            if attention_mask is not None:
                valid = attention_mask[b].to(device=device).bool()

            is_video = (input_ids[b] == self.video_token_id) & valid
            video_positions = torch.where(is_video)[0]
            keep_local = torch.as_tensor(keep_indices[b], device=device, dtype=torch.long).flatten()
            keep_local = keep_local.unique(sorted=True)

            if keep_local.numel() > 0:
                if video_positions.numel() == 0:
                    raise ValueError(f"Batch {b}: keep indices were provided but no video tokens were found")
                invalid = (keep_local < 0) | (keep_local >= video_positions.numel())
                if invalid.any():
                    bad = keep_local[invalid][:8].detach().cpu().tolist()
                    raise IndexError(
                        f"Batch {b}: video keep indices out of range for {video_positions.numel()} "
                        f"video tokens; examples={bad}"
                    )

            is_kept = torch.zeros(L, dtype=torch.bool, device=device)
            is_kept[valid & ~is_video] = True
            if keep_local.numel() > 0:
                is_kept[video_positions[keep_local]] = True
            sequence_keep_masks.append(is_kept)

        return sequence_keep_masks

    @staticmethod
    def prune_sequence_tensor(
        sequence_tensor: torch.Tensor,
        sequence_keep_masks: list[torch.Tensor],
        pad_value: int | float | bool = 0,
    ) -> torch.Tensor:
        """Prune and right-pad a [B,L,...] tensor using sequence keep masks."""
        if sequence_tensor is None:
            raise ValueError("sequence_tensor cannot be None")
        B = sequence_tensor.shape[0]
        if len(sequence_keep_masks) != B:
            raise ValueError(f"Expected {B} sequence masks, got {len(sequence_keep_masks)}")

        pieces = []
        max_len = 0
        for b, keep_mask in enumerate(sequence_keep_masks):
            keep_mask = keep_mask.to(device=sequence_tensor.device, dtype=torch.bool)
            cur = sequence_tensor[b][keep_mask]
            pieces.append(cur)
            max_len = max(max_len, cur.shape[0])

        out_shape = (B, max_len, *sequence_tensor.shape[2:])
        out = sequence_tensor.new_full(out_shape, pad_value)
        for b, cur in enumerate(pieces):
            out[b, : cur.shape[0]] = cur
        return out

    @staticmethod
    def prune_token_mask(
        token_mask: torch.Tensor,
        sequence_keep_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Prune a boolean/0-1 token mask such as Qwen visual_pos_masks."""
        return TokenPruner.prune_sequence_tensor(token_mask, sequence_keep_masks, pad_value=False)

    @staticmethod
    def visual_flat_keep_indices(
        visual_pos_masks: torch.Tensor,
        sequence_keep_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Return flattened visual-feature indices retained after sequence pruning."""
        if visual_pos_masks.ndim != 2:
            raise ValueError(f"Expected visual_pos_masks [B,L], got {tuple(visual_pos_masks.shape)}")
        if len(sequence_keep_masks) != visual_pos_masks.shape[0]:
            raise ValueError(
                f"Expected {visual_pos_masks.shape[0]} sequence masks, got {len(sequence_keep_masks)}"
            )

        device = visual_pos_masks.device
        flat_indices = []
        offset = 0
        for b, keep_mask in enumerate(sequence_keep_masks):
            visual_mask = visual_pos_masks[b].to(device=device).bool()
            keep_mask = keep_mask.to(device=device).bool()
            n_visual = int(visual_mask.sum().item())
            ordinal = torch.full(
                (visual_mask.shape[0],),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            )
            ordinal[visual_mask] = torch.arange(n_visual, device=device, dtype=torch.long)
            kept_visual_ordinals = ordinal[visual_mask & keep_mask]
            flat_indices.append(kept_visual_ordinals + offset)
            offset += n_visual

        if not flat_indices:
            return torch.empty(0, device=device, dtype=torch.long)
        return torch.cat(flat_indices)

    @staticmethod
    def prune_deepstack_visual_embeds(
        deepstack_visual_embeds: list[torch.Tensor] | None,
        visual_pos_masks: torch.Tensor | None,
        sequence_keep_masks: list[torch.Tensor],
    ) -> list[torch.Tensor] | None:
        """Prune Qwen deepstack visual features to match pruned visual positions."""
        if deepstack_visual_embeds is None or visual_pos_masks is None:
            return deepstack_visual_embeds
        keep_indices = TokenPruner.visual_flat_keep_indices(visual_pos_masks, sequence_keep_masks)
        return [embed[keep_indices.to(embed.device)] for embed in deepstack_visual_embeds]

    @staticmethod
    def _normalize_keep_indices(video_keep_indices: list[torch.Tensor], batch_size: int) -> list[torch.Tensor]:
        if video_keep_indices is None:
            raise ValueError("video_keep_indices cannot be None")
        if len(video_keep_indices) == batch_size:
            return video_keep_indices
        if len(video_keep_indices) == 1 and batch_size > 1:
            return [video_keep_indices[0] for _ in range(batch_size)]
        raise ValueError(f"Expected {batch_size} keep-index tensors, got {len(video_keep_indices)}")

    def should_skip_pruning(self, pixel_values_videos, inputs_embeds) -> bool:
        """Return True if we should skip pruning (decode step or no video input)."""
        if pixel_values_videos is None:
            return True
        if inputs_embeds.shape[1] == 1:
            # Decode step: only one token, no pruning needed
            return True
        return False

    @staticmethod
    def make_random_keep_indices(
        num_video_tokens: int,
        keep_ratio: float,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> list[torch.Tensor]:
        """Generate random keep indices for testing/baseline.

        Args:
            num_video_tokens: Number of video placeholder tokens per sample.
            keep_ratio: Fraction of video tokens to keep.
            batch_size: Batch size.

        Returns:
            List of [K] int64 tensors, one per batch sample.
        """
        k = max(1, int(num_video_tokens * keep_ratio))
        indices = []
        for _ in range(batch_size):
            perm = torch.randperm(num_video_tokens, device=device)
            indices.append(perm[:k].sort().values)
        return indices
