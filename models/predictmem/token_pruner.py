"""Token pruning for PredictMem.

Prunes Qwen video placeholder tokens according to V-JEPA keep masks, producing
right-padded, shorter sequences for the language model.
"""

import torch
import torch.nn.functional as F

from .config import PredictMemConfig


class TokenPruner:
    """Prunes Qwen visual placeholder tokens by keep mask.

    Non-video tokens (text, timestamps, vision_start/vision_end) are always kept.
    Only video_token_id placeholders are selectively pruned.
    """

    def __init__(
        self,
        config: PredictMemConfig,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            All have shape [B, L_new, D], [3, B, L_new], [B, L_new].
        """
        B, L, D = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        new_embeds_list = []
        new_pos_list = []
        max_len = 0

        for b in range(B):
            # Identify video token positions in this sample
            is_video = input_ids[b] == self.video_token_id
            video_positions = torch.where(is_video)[0]

            keep_local = video_keep_indices[b].to(device)
            # Convert local video token indices to absolute sequence positions
            kept_video_pos = video_positions[keep_local]

            # All non-video positions are kept
            is_kept = torch.zeros(L, dtype=torch.bool, device=device)
            is_kept[kept_video_pos] = True
            is_kept[~is_video] = True  # keep all non-video tokens

            new_embeds_list.append(inputs_embeds[b][is_kept])
            # position_ids: [3, B, L] -> select columns
            new_pos_list.append(position_ids[:, b, is_kept])
            max_len = max(max_len, is_kept.sum().item())

        # Pad to max length
        new_L = max_len
        padded_embeds = torch.zeros(B, new_L, D, device=device, dtype=dtype)
        padded_pos = torch.zeros(3, B, new_L, device=device, dtype=position_ids.dtype)
        new_attention_mask = torch.zeros(B, new_L, device=device, dtype=torch.long)

        for b in range(B):
            cur_len = new_embeds_list[b].shape[0]
            padded_embeds[b, :cur_len] = new_embeds_list[b]
            padded_pos[:, b, :cur_len] = new_pos_list[b]
            new_attention_mask[b, :cur_len] = 1

        return padded_embeds, padded_pos, new_attention_mask

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
