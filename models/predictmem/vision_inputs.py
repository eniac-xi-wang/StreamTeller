"""In-memory video sampling for Qwen + V-JEPA — no disk cache, no JSONL.

Samples a video once and derives aspect-aligned Qwen frames and a V-JEPA
tensor from the same sampling plan.  The resulting tensors are passed
directly to the model — ``predictmem_frames_256`` never touches disk.

V-JEPA tensor receives ImageNet normalization (aligned with Survey analyzer).
Qwen frames stay uint8 [0,255] for the Qwen processor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .resize_utils import compute_aligned_resize

# ImageNet normalization (same as Survey analyzer)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_predictmem_video_inputs(
    video_path: str,
    fps: float = 1.0,
    qwen_size: int = 512,
    jepa_size: int = 256,
    num_frames: int | None = None,
) -> tuple[np.ndarray, torch.Tensor, dict]:
    """Sample a video once; return Qwen frames, V-JEPA tensor, and metadata.

    Args:
        video_path: path to mp4
        fps: target frame rate (default 1.0)
        qwen_size: Qwen input resolution
        jepa_size: V-JEPA input resolution
        num_frames: number of 1FPS frames (None = all frames)

    Returns:
        qwen_frames_uint8: [N, qwen_h, qwen_w, 3] uint8 RGB
        predictmem_frames_256: [N, 3, jepa_h, jepa_w] ImageNet-normalized
        video_metadata: dict for Qwen processor
    """
    import decord
    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)
    source_fps = float(vr.get_avg_fps() or fps)
    duration = total_frames / source_fps if source_fps > 0 else 0.0

    total_1fps = max(1, int(duration * fps))
    if num_frames is not None:
        total_1fps = min(num_frames, total_1fps)

    times_s = [i / fps for i in range(total_1fps)]
    source_indices = [
        min(total_frames - 1, max(0, int(round(t * source_fps))))
        for t in times_s
    ]

    # Read all frames at Qwen resolution
    frames_raw = vr.get_batch(source_indices)
    if hasattr(frames_raw, "asnumpy"):
        frames_raw = torch.from_numpy(frames_raw.asnumpy())
    elif not isinstance(frames_raw, torch.Tensor):
        frames_raw = torch.from_numpy(np.asarray(frames_raw))
    frames_raw = frames_raw.to(dtype=torch.uint8)  # [N, H, W, C]

    source_h, source_w = int(frames_raw.shape[1]), int(frames_raw.shape[2])
    (qwen_h, qwen_w), (jepa_h, jepa_w) = compute_aligned_resize(
        source_height=source_h,
        source_width=source_w,
        qwen_size=qwen_size,
        jepa_size=jepa_size,
    )

    # Qwen frames: smart resize, preserve aspect ratio
    frames_chw = frames_raw.permute(0, 3, 1, 2).float()  # [N, C, H, W]
    if frames_chw.shape[-2:] != (qwen_h, qwen_w):
        qwen_chw = F.interpolate(frames_chw, size=(qwen_h, qwen_w),
                                  mode="bilinear", align_corners=False)
    else:
        qwen_chw = frames_chw
    qwen_chw = qwen_chw.clamp(0, 255)
    qwen_frames_uint8 = qwen_chw.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()

    # V-JEPA frames: same aspect ratio, exactly half Qwen resolution
    if frames_chw.shape[-2:] != (jepa_h, jepa_w):
        jepa_chw = F.interpolate(frames_chw, size=(jepa_h, jepa_w),
                                  mode="bilinear", align_corners=False)
    else:
        jepa_chw = frames_chw
    # Normalize to [0,1] then apply ImageNet stats (aligned with Survey analyzer)
    jepa_01 = jepa_chw / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=jepa_01.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=jepa_01.device).view(1, 3, 1, 1)
    predictmem_frames_256 = ((jepa_01 - mean) / std).contiguous().cpu()

    video_metadata = {
        "total_num_frames": total_1fps,
        "fps": float(fps),
        "duration": total_1fps / float(fps),
        "frames_indices": list(range(total_1fps)),
        "height": qwen_h,
        "width": qwen_w,
        "video_backend": "decord",
    }

    return qwen_frames_uint8, predictmem_frames_256, video_metadata
