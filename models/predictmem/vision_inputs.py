"""In-memory video sampling for Qwen + V-JEPA — no disk cache, no JSONL.

Samples a video once and derives both Qwen 512px frames and V-JEPA 256px
tensor from the same sampling plan.  The resulting tensors are passed
directly to the model — ``predictmem_frames_256`` never touches disk.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


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
        qwen_frames_uint8: [N, qwen_size, qwen_size, 3] uint8 RGB
        predictmem_frames_256: [N, 3, jepa_size, jepa_size] float in [0,1]
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

    # Qwen frames: resize to qwen_size
    frames_chw = frames_raw.permute(0, 3, 1, 2).float()  # [N, C, H, W]
    if frames_chw.shape[-2:] != (qwen_size, qwen_size):
        qwen_chw = F.interpolate(frames_chw, size=(qwen_size, qwen_size),
                                  mode="bilinear", align_corners=False)
    else:
        qwen_chw = frames_chw
    qwen_chw = qwen_chw.clamp(0, 255)
    qwen_frames_uint8 = qwen_chw.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()

    # V-JEPA frames: resize to jepa_size, normalize to [0,1]
    if frames_chw.shape[-2:] != (jepa_size, jepa_size):
        jepa_chw = F.interpolate(frames_chw, size=(jepa_size, jepa_size),
                                  mode="bilinear", align_corners=False)
    else:
        jepa_chw = frames_chw
    predictmem_frames_256 = (jepa_chw / 255.0).contiguous().cpu()  # [N, 3, 256, 256]

    video_metadata = {
        "total_num_frames": total_1fps,
        "fps": float(fps),
        "duration": total_1fps / float(fps),
        "frames_indices": list(range(total_1fps)),
        "height": qwen_size,
        "width": qwen_size,
        "video_backend": "decord",
    }

    return qwen_frames_uint8, predictmem_frames_256, video_metadata
