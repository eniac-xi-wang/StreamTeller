"""FramePlan: unified sampling plan shared by Qwen and V-JEPA.

Ensures both pipelines use the same frames, time range, and source indices.
All window definitions use integer frame counts — no closed-interval second
ambiguities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FramePlan:
    """A pre-computed sampling plan over a video chunk at target_fps=1.

    Qwen and V-JEPA must read frames exclusively from this plan to guarantee
    time alignment between visual tokens and prediction-loss keep indices.

    Window semantics (all half-open):
        [local_start_frame, local_start_frame + window_frames)
    """

    video_path: str
    video_duration_s: float
    source_fps: float
    target_fps: float = 1.0

    # 1FPS-aligned frame times and corresponding source indices
    frame_times_s: list[float] = field(default_factory=list)
    source_indices: list[int] = field(default_factory=list)

    # Pre-sampled frames at Qwen resolution [N, H, W, C] uint8 RGB
    qwen_frames_uint8: np.ndarray | None = None
    qwen_size: int = 512

    # V-JEPA frames are lazily sampled per window to save memory
    vjepa_size: int = 256

    # Stream metadata
    stream_mode: str = "full"
    frame_budget: int = 0
    truncated: bool = False

    @property
    def num_frames(self) -> int:
        return len(self.source_indices)

    @property
    def num_tubelets(self) -> int:
        """Number of 2-frame tubelets: ceil(num_frames / 2)."""
        return max(1, (self.num_frames + 1) // 2)

    @property
    def frame_plan_start_s(self) -> float:
        return self.frame_times_s[0] if self.frame_times_s else 0.0

    @property
    def frame_plan_end_s_exclusive(self) -> float:
        if not self.frame_times_s:
            return 0.0
        return self.frame_times_s[-1] + 1.0 / self.target_fps

    @property
    def coverage_ratio(self) -> float:
        if self.video_duration_s <= 0:
            return 1.0
        return self.num_frames / (self.video_duration_s * self.target_fps)

    def window_bounds(self, local_start: int, window_frames: int = 16) -> dict:
        """Return debug metadata for a window starting at local_start."""
        end_exclusive = local_start + window_frames
        return {
            "window_start_frame": local_start,
            "window_end_frame_exclusive": end_exclusive,
            "window_num_frames": window_frames,
            "window_start_s": self.frame_times_s[local_start] if local_start < self.num_frames else 0.0,
            "window_end_s_exclusive": (
                self.frame_times_s[min(end_exclusive, self.num_frames) - 1] + 1.0 / self.target_fps
                if end_exclusive <= self.num_frames and self.num_frames > 0
                else 0.0
            ),
            "window_last_frame_s": (
                self.frame_times_s[min(end_exclusive, self.num_frames) - 1]
                if end_exclusive <= self.num_frames and self.num_frames > 0
                else 0.0
            ),
            "source_indices": self.source_indices[local_start:end_exclusive],
        }

    def get_qwen_window(self, local_start: int, window_frames: int = 16) -> np.ndarray:
        """Return qwen_frames_uint8 slice [window_frames, H, W, C]."""
        if self.qwen_frames_uint8 is None:
            raise ValueError("qwen_frames_uint8 not populated — call build() first")
        end = local_start + window_frames
        return self.qwen_frames_uint8[local_start:end]

    def get_vjepa_tensor(self, local_start: int, window_frames: int = 16) -> torch.Tensor:
        """Sample and return V-JEPA tensor [1, 3, window_frames, 256, 256]."""
        from .video_sampling import sample_video_1fps_decord

        start_s = self.frame_times_s[local_start] if local_start < self.num_frames else 0.0
        sample = sample_video_1fps_decord(
            self.video_path,
            num_frames=window_frames,
            size=self.vjepa_size,
            target_fps=self.target_fps,
            start_time_s=start_s,
        )
        return sample.vjepa_tensor()

    def window_starts(self, window_frames: int = 16, stride_frames: int = 2) -> range:
        """Return range of local_start indices for sliding windows."""
        end_limit = max(0, self.num_frames - window_frames + 1)
        return range(0, end_limit, stride_frames)

    def to_dict(self) -> dict:
        return {
            "video_duration_s": self.video_duration_s,
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "frame_plan_num_frames": self.num_frames,
            "frame_plan_start_s": self.frame_plan_start_s,
            "frame_plan_end_s_exclusive": self.frame_plan_end_s_exclusive,
            "frame_plan_coverage_ratio": round(self.coverage_ratio, 4),
            "stream_mode": self.stream_mode,
            "frame_budget": self.frame_budget,
            "full_stream_truncated": self.truncated,
            "num_tubelets": self.num_tubelets,
        }


def build_frame_plan(
    video_path: str,
    *,
    stream_mode: str = "full",
    frame_budget: int = 0,
    target_fps: float = 1.0,
    qwen_size: int = 512,
    vjepa_size: int = 256,
) -> FramePlan:
    """Build a FramePlan over a video chunk at ``target_fps``.

    Args:
        video_path: path to the mp4 file
        stream_mode: ``full`` (entire chunk), ``tail_budget`` (last N frames),
            ``uniform_budget`` (N frames uniformly spaced), ``first_budget`` (first N)
        frame_budget: max frames when mode is not ``full`` (0 = no limit)
        target_fps: output frame rate (default 1.0)
        qwen_size: Qwen input frame size in pixels
        vjepa_size: V-JEPA input frame size in pixels
    """
    import decord
    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)
    source_fps = float(vr.get_avg_fps() or target_fps)
    duration = total_frames / source_fps if source_fps > 0 else 0.0

    # All available 1FPS times and source indices
    total_1fps_frames = max(1, int(duration * target_fps))
    all_times = [i / target_fps for i in range(total_1fps_frames)]
    all_source = [
        min(total_frames - 1, max(0, int(round(t * source_fps))))
        for t in all_times
    ]

    # Select frames per stream mode
    if stream_mode == "full" or (stream_mode in ("tail_budget", "first_budget", "uniform_budget") and frame_budget <= 0):
        selected_times = all_times
        selected_source = all_source
        truncated = False
    elif stream_mode == "tail_budget":
        n = min(frame_budget, total_1fps_frames)
        selected_times = all_times[-n:]
        selected_source = all_source[-n:]
        truncated = len(all_times) > n
    elif stream_mode == "first_budget":
        n = min(frame_budget, total_1fps_frames)
        selected_times = all_times[:n]
        selected_source = all_source[:n]
        truncated = len(all_times) > n
    elif stream_mode == "uniform_budget":
        n = min(frame_budget, total_1fps_frames)
        if n >= total_1fps_frames:
            selected_times = all_times
            selected_source = all_source
            truncated = False
        else:
            step = total_1fps_frames / n
            indices = [min(total_1fps_frames - 1, int(i * step)) for i in range(n)]
            # Deduplicate
            seen = set()
            dedup_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    dedup_indices.append(idx)
            selected_times = [all_times[i] for i in dedup_indices]
            selected_source = [all_source[i] for i in dedup_indices]
            truncated = True
    else:
        raise ValueError(f"Unknown stream_mode: {stream_mode}")

    # Pre-sample Qwen frames
    qwen_frames = _sample_qwen_frames(
        vr, selected_source, qwen_size, target_fps, selected_times[0] if selected_times else 0.0,
    )

    return FramePlan(
        video_path=str(video_path),
        video_duration_s=duration,
        source_fps=source_fps,
        target_fps=target_fps,
        frame_times_s=selected_times,
        source_indices=selected_source,
        qwen_frames_uint8=qwen_frames,
        qwen_size=qwen_size,
        vjepa_size=vjepa_size,
        stream_mode=stream_mode,
        frame_budget=frame_budget,
        truncated=truncated,
    )


def _sample_qwen_frames(
    vr,
    source_indices: list[int],
    size: int,
    target_fps: float,
    start_time_s: float,
) -> np.ndarray:
    """Pre-sample all Qwen-resolution frames at once."""
    import decord
    decord.bridge.set_bridge("torch")

    if not source_indices:
        raise ValueError("No source indices to sample")

    frames = vr.get_batch(source_indices)
    if hasattr(frames, "asnumpy"):
        frames = torch.from_numpy(frames.asnumpy())
    elif not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(np.asarray(frames))
    frames = frames.to(dtype=torch.uint8)

    # Convert to [N, C, H, W] for resize
    frames_chw = frames.permute(0, 3, 1, 2).float()
    if frames_chw.shape[-2:] != (size, size):
        frames_chw = F.interpolate(frames_chw, size=(size, size), mode="bilinear", align_corners=False)
    frames_chw = frames_chw.clamp(0, 255)

    # Back to [N, H, W, C] uint8
    frames_uint8 = frames_chw.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return frames_uint8
