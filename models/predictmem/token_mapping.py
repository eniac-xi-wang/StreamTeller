"""Token mapping between V-JEPA tokens and Qwen LLM video tokens.

Both grids are 8 x 16 x 16 = 2048 tokens for a 16-frame window, so they map
one-to-one via the linearized index:

    token_id = temporal_id * 16 * 16 + h_id * 16 + w_id
"""

import torch

from .config import PredictMemConfig


class TokenMapper:
    """Maintains the 1:1 mapping between JEPA and Qwen LLM video tokens."""

    def __init__(self, config: PredictMemConfig):
        self.config = config
        self._validate_config()
        self.num_tokens = config.num_jepa_tokens  # == num_qwen_video_tokens == 2048

    def _validate_config(self):
        cfg = self.config
        assert cfg.qwen_grid_t == 8, f"Expected qwen_grid_t=8, got {cfg.qwen_grid_t}"
        assert cfg.qwen_grid_h == 32, f"Expected qwen_grid_h=32, got {cfg.qwen_grid_h}"
        assert cfg.qwen_grid_w == 32, f"Expected qwen_grid_w=32, got {cfg.qwen_grid_w}"
        assert cfg.qwen_merge_size == 2, f"Expected qwen_merge_size=2, got {cfg.qwen_merge_size}"
        assert cfg.qwen_llm_h == 16, f"Expected qwen_llm_h=16, got {cfg.qwen_llm_h}"
        assert cfg.qwen_llm_w == 16, f"Expected qwen_llm_w=16, got {cfg.qwen_llm_w}"
        assert cfg.jepa_grid_t == 8, f"Expected jepa_grid_t=8, got {cfg.jepa_grid_t}"
        assert cfg.jepa_grid_h == 16, f"Expected jepa_grid_h=16, got {cfg.jepa_grid_h}"
        assert cfg.jepa_grid_w == 16, f"Expected jepa_grid_w=16, got {cfg.jepa_grid_w}"
        assert cfg.num_jepa_tokens == cfg.num_qwen_video_tokens, (
            f"Token count mismatch: JEPA={cfg.num_jepa_tokens}, Qwen={cfg.num_qwen_video_tokens}"
        )

    def jepa_to_qwen_index(self, jepa_token_id: int) -> int:
        """Map a single JEPA token local index to Qwen video token local index.

        Since both use the identical ordering (t * 256 + h * 16 + w),
        this is an identity mapping.
        """
        return jepa_token_id

    def jepa_to_qwen_indices(self, jepa_token_ids: torch.Tensor) -> torch.Tensor:
        """Map JEPA token local indices to Qwen video token local indices.

        Args:
            jepa_token_ids: [N] int64 tensor of JEPA token local indices.

        Returns:
            [N] int64 tensor of Qwen video token local indices (same values).
        """
        return jepa_token_ids

    def qwen_video_to_jepa_index(self, qwen_video_local_index: int) -> int:
        """Reverse mapping: Qwen video token local index -> JEPA token index."""
        return qwen_video_local_index

    def get_tubelet_indices(self, tubelet_id: int) -> torch.Tensor:
        """Return flat token indices for a given temporal tubelet (0..7).

        Each tubelet covers 2 frames -> 16*16 = 256 tokens.
        """
        offset = tubelet_id * self.config.jepa_grid_h * self.config.jepa_grid_w
        return torch.arange(offset, offset + 256)

    def assert_video_grid_thw(self, video_grid_thw: torch.Tensor):
        """Assert that video_grid_thw is the expected dense [8, 32, 32] grid."""
        assert video_grid_thw.ndim == 2, f"Expected 2D grid_thw, got shape {video_grid_thw.shape}"
        for i, row in enumerate(video_grid_thw):
            assert row[0].item() == 8, f"Row {i}: expected t=8, got {row[0].item()}"
            assert row[1].item() == 32, f"Row {i}: expected h=32, got {row[1].item()}"
            assert row[2].item() == 32, f"Row {i}: expected w=32, got {row[2].item()}"

    def compute_num_video_tokens(self, video_grid_thw: torch.Tensor) -> int:
        """Compute number of LLM video tokens from video_grid_thw.

        num_video_tokens = prod(video_grid_thw) // merge_size**2
        """
        merge = self.config.qwen_merge_size
        return int(video_grid_thw.prod().item() // (merge * merge))

    def tubelet_id_to_token_range(self, tubelet_id: int) -> tuple[int, int]:
        """Return (start, end) local indices for a temporal tubelet."""
        n = self.config.jepa_grid_h * self.config.jepa_grid_w
        start = tubelet_id * n
        end = start + n
        return start, end
