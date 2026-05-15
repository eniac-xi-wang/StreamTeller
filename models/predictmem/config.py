from dataclasses import dataclass, field


@dataclass
class PredictMemConfig:
    """Configuration for PredictMem visual token pruning via V-JEPA prediction loss."""

    enabled: bool = False
    window_frames: int = 16
    target_frames: int = 2
    temporal_stride: int = 2
    fps: float = 1.0
    jepa_size: int = 256
    qwen_size: int = 512
    patch_size: int = 16
    qwen_merge_size: int = 2
    keep_ratio: float = 0.5
    min_cell_keep: bool = True
    cell_grid_size: int = 4
    keep_recent_full_frames: int = 0
    score_mode: str = "rank"  # rank | zscore | raw
    runtime_mode: str = "offline"  # offline | online
    score_cache_path: str | None = None
    loss_exp: float = 1.0  # exponent for V-JEPA prediction loss

    # Derived properties (computed on access, not stored)
    jepa_grid_t: int = field(init=False, default=8)
    jepa_grid_h: int = field(init=False, default=16)
    jepa_grid_w: int = field(init=False, default=16)
    qwen_grid_t: int = field(init=False, default=8)
    qwen_grid_h: int = field(init=False, default=32)
    qwen_grid_w: int = field(init=False, default=32)
    qwen_llm_h: int = field(init=False, default=16)
    qwen_llm_w: int = field(init=False, default=16)
    num_jepa_tokens: int = field(init=False, default=2048)
    num_qwen_video_tokens: int = field(init=False, default=2048)

    def __post_init__(self):
        self.jepa_grid_t = self.window_frames // self.temporal_stride
        self.jepa_grid_h = self.jepa_size // self.patch_size
        self.jepa_grid_w = self.jepa_size // self.patch_size
        self.qwen_grid_t = self.window_frames // self.temporal_stride
        self.qwen_grid_h = self.qwen_size // self.patch_size
        self.qwen_grid_w = self.qwen_size // self.patch_size
        self.qwen_llm_h = self.qwen_grid_h // self.qwen_merge_size
        self.qwen_llm_w = self.qwen_grid_w // self.qwen_merge_size
        self.num_jepa_tokens = self.jepa_grid_t * self.jepa_grid_h * self.jepa_grid_w
        self.num_qwen_video_tokens = self.qwen_grid_t * self.qwen_llm_h * self.qwen_llm_w
