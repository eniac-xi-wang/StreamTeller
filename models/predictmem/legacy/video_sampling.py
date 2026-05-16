"""Decord-based video sampling shared by PredictMem scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DecordVideoSample:
    """A fixed-FPS video window sampled once and reused by Qwen and V-JEPA."""

    frames_uint8: np.ndarray  # [T, H, W, C], RGB uint8
    frames_chw_float: torch.Tensor  # [T, C, H, W], float32 in [0, 1]
    source_indices: list[int]
    source_fps: float
    target_fps: float

    @property
    def num_frames(self) -> int:
        return int(self.frames_uint8.shape[0])

    def qwen_metadata(self) -> dict:
        """Metadata for pre-sampled Qwen processor input.

        The sampled window is intentionally represented as a 1FPS video, so
        Qwen timestamp prompts align with the PredictMem streaming window.
        """
        return {
            "total_num_frames": self.num_frames,
            "fps": float(self.target_fps),
            "duration": self.num_frames / float(self.target_fps),
            "frames_indices": list(range(self.num_frames)),
            "height": int(self.frames_uint8.shape[1]),
            "width": int(self.frames_uint8.shape[2]),
            "video_backend": "decord",
        }

    def vjepa_tensor(self) -> torch.Tensor:
        """Return [1, 3, T, H, W] float tensor for V-JEPA."""
        return self.frames_chw_float.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def sample_video_1fps_decord(
    video_path: str | Path,
    *,
    num_frames: int = 16,
    size: int = 512,
    target_fps: float = 1.0,
    start_time_s: float = 0.0,
) -> DecordVideoSample:
    """Sample a fixed-size 1FPS window with decord.

    Frames beyond the end of short videos are padded by repeating the last valid
    frame, preserving the requested window length and temporal token count.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")

    import decord

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    source_fps = float(vr.get_avg_fps() or target_fps)
    start_index = max(0, int(round(start_time_s * source_fps)))
    source_indices = [
        min(total - 1, max(0, int(round(start_index + i * source_fps / target_fps))))
        for i in range(num_frames)
    ]

    frames = vr.get_batch(source_indices)
    if hasattr(frames, "asnumpy"):
        frames = torch.from_numpy(frames.asnumpy())
    elif not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(np.asarray(frames))
    frames = frames.to(dtype=torch.uint8)

    frames_chw = frames.permute(0, 3, 1, 2).float()
    if frames_chw.shape[-2:] != (size, size):
        frames_chw = F.interpolate(frames_chw, size=(size, size), mode="bilinear", align_corners=False)
    frames_chw = frames_chw.clamp(0, 255)
    frames_uint8 = frames_chw.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
    frames_float = (frames_chw / 255.0).contiguous().cpu()

    return DecordVideoSample(
        frames_uint8=frames_uint8,
        frames_chw_float=frames_float,
        source_indices=source_indices,
        source_fps=source_fps,
        target_fps=float(target_fps),
    )
