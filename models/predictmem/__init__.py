from .config import PredictMemConfig
from .token_pruner import TokenPruner
from .streaming_memory import PredictMemStreamingMemory
from .vision_inputs import build_predictmem_video_inputs
from .vjepa_scorer import (
    VJEPAPredictLossScorer,
    PredictMemScore,
    keep_indices_to_thw,
    make_vjepa_encoder_predictor,
    make_vjepa_analyzer_scorer,
    load_vjepa_checkpoint,
)

# Optional / legacy modules (not part of the plugin main path)
try:
    from .token_mapping import TokenMapper
except ImportError:
    TokenMapper = None
try:
    from .cache import ScoreCache
except ImportError:
    ScoreCache = None
try:
    from .video_sampling import DecordVideoSample, sample_video_1fps_decord
except ImportError:
    DecordVideoSample = None
    sample_video_1fps_decord = None
try:
    from .frame_plan import FramePlan, build_frame_plan
except ImportError:
    FramePlan = None
    build_frame_plan = None

__all__ = [
    # Mainline (plugin path)
    "PredictMemConfig",
    "TokenPruner",
    "PredictMemStreamingMemory",
    "build_predictmem_video_inputs",
    "VJEPAPredictLossScorer",
    "PredictMemScore",
    "keep_indices_to_thw",
    "make_vjepa_encoder_predictor",
    "make_vjepa_analyzer_scorer",
    "load_vjepa_checkpoint",
    # Legacy (compatibility, may be removed)
    "TokenMapper",
    "ScoreCache",
    "DecordVideoSample",
    "sample_video_1fps_decord",
    "FramePlan",
    "build_frame_plan",
]
