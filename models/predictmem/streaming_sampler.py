"""Streaming video sampler that avoids loading entire video at once.

Instead of ``decord.VideoReader.get_batch(all_indices)`` which loads all
sampled frames into GPU/CPU memory simultaneously, this sampler yields
one tubelet (2 frames) at a time.

Usage::

    sampler = StreamingVideoSampler(video_path, fps=1.0, qwen_size=512, jepa_size=256)
    for tubelet in sampler:
        qwen_frames = tubelet["qwen"]   # [2, H, W, 3] uint8
        jepa_frames = tubelet["jepa"]   # [2, 3, H/2, W/2] ImageNet-normalized
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .resize_utils import compute_aligned_resize


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class StreamingVideoSampler:
    """Stream video frames tubelet-by-tubelet at a fixed FPS.

    Uses decord for frame access. Each call to ``__iter__`` / ``__next__``
    yields a dict with Qwen (uint8 RGB) and V-JEPA (float32 normalized) tensors
    for one 2-frame tubelet, plus metadata.
    """

    def __init__(
        self,
        video_path: str,
        fps: float = 1.0,
        qwen_size: int = 512,
        jepa_size: int = 256,
        frame_budget: int = 0,
        stream_mode: str = "full",
        start_time: float = 0.0,
        end_time: float | None = None,
    ):
        import decord
        decord.bridge.set_bridge("torch")

        self.vr = decord.VideoReader(str(video_path))
        self.fps = fps
        self.qwen_size = qwen_size
        self.jepa_size = jepa_size
        self.frame_budget = frame_budget
        self.stream_mode = stream_mode

        total_frames = len(self.vr)
        source_fps = float(self.vr.get_avg_fps() or fps)
        self.duration = total_frames / source_fps if source_fps > 0 else 0.0
        sample_frame = self.vr[0]
        if hasattr(sample_frame, "asnumpy"):
            sample_frame = torch.from_numpy(sample_frame.asnumpy())
        source_h, source_w = int(sample_frame.shape[0]), int(sample_frame.shape[1])
        (self.qwen_h, self.qwen_w), (self.jepa_h, self.jepa_w) = compute_aligned_resize(
            source_height=source_h,
            source_width=source_w,
            qwen_size=qwen_size,
            jepa_size=jepa_size,
        )
        self.source_height = source_h
        self.source_width = source_w

        clip_end = min(self.duration, end_time) if end_time is not None else self.duration
        clip_start = max(0.0, start_time)
        clip_duration = clip_end - clip_start

        total_1fps = max(1, int(clip_duration * fps))
        if frame_budget and frame_budget > 0:
            total_1fps = min(frame_budget, total_1fps)

        if stream_mode == "recent":
            self.times_s = [clip_end - (total_1fps - i) / fps for i in range(total_1fps)]
        elif stream_mode == "uniform":
            self.times_s = [clip_start + (i + 0.5) * clip_duration / total_1fps for i in range(total_1fps)]
        else:
            self.times_s = [clip_start + i / fps for i in range(total_1fps)]
        self.source_indices = [
            min(total_frames - 1, max(0, int(round(t * source_fps))))
            for t in self.times_s
        ]
        self.num_frames = total_1fps
        self.num_tubelets = (total_1fps + 1) // 2
        self.clip_start = clip_start
        self.clip_end = clip_end
        self.source_fps = source_fps

        self._cursor = 0

    def __len__(self) -> int:
        return self.num_tubelets

    def __iter__(self) -> Iterator[dict]:
        self._cursor = 0
        return self

    def __next__(self) -> dict:
        t = self._cursor
        if t >= self.num_tubelets:
            raise StopIteration

        f0 = t * 2
        f1 = min(t * 2 + 1, self.num_frames - 1)
        indices = self.source_indices[f0:f1 + 1]
        frames_raw = self.vr.get_batch(indices)
        if hasattr(frames_raw, "asnumpy"):
            frames_raw = torch.from_numpy(frames_raw.asnumpy())
        elif not isinstance(frames_raw, torch.Tensor):
            frames_raw = torch.from_numpy(np.asarray(frames_raw))

        n = frames_raw.shape[0]
        frames_chw = frames_raw.permute(0, 3, 1, 2).float()  # [n, 3, H, W]

        # Qwen frames: smart resize, preserve aspect ratio
        if frames_chw.shape[-2:] != (self.qwen_h, self.qwen_w):
            qwen_chw = F.interpolate(frames_chw, size=(self.qwen_h, self.qwen_w),
                                     mode="bilinear", align_corners=False)
        else:
            qwen_chw = frames_chw
        qwen_chw = qwen_chw.clamp(0, 255)
        qwen_uint8 = qwen_chw.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()

        # JEPA frames: same aspect ratio, exactly half Qwen resolution
        if frames_chw.shape[-2:] != (self.jepa_h, self.jepa_w):
            jepa_chw = F.interpolate(frames_chw, size=(self.jepa_h, self.jepa_w),
                                     mode="bilinear", align_corners=False)
        else:
            jepa_chw = frames_chw
        jepa_01 = jepa_chw / 255.0
        mean = torch.tensor(IMAGENET_MEAN, device=jepa_01.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=jepa_01.device).view(1, 3, 1, 1)
        jepa_norm = ((jepa_01 - mean) / std).contiguous().cpu()

        result = {
            "tubelet_id": t,
            "qwen": qwen_uint8,            # [n, qwen_h, qwen_w, 3] uint8 numpy
            "jepa": jepa_norm,             # [n, 3, jepa_h, jepa_w] float32
            "frame_indices": indices,
            "times_s": self.times_s[f0:f1 + 1],
            "num_frames_in_tubelet": n,
        }
        self._cursor += 1
        return result

    @property
    def metadata(self) -> dict:
        return {
            "total_num_frames": self.num_frames,
            "fps": float(self.fps),
            "duration": self.clip_end - self.clip_start,
            "frames_indices": self.source_indices,
            "height": self.qwen_h,
            "width": self.qwen_w,
            "video_backend": "decord",
        }

    @property
    def extra_meta(self) -> dict:
        return {
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "source_fps": self.source_fps,
            "stream_mode": self.stream_mode,
            "source_height": self.source_height,
            "source_width": self.source_width,
            "qwen_height": self.qwen_h,
            "qwen_width": self.qwen_w,
            "jepa_height": self.jepa_h,
            "jepa_width": self.jepa_w,
            "resize_mode": "qwen_smart_aspect_ratio",
            "num_tubelets": self.num_tubelets,
        }
