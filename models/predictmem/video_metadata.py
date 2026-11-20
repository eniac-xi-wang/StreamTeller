"""Qwen metadata for frames selected from a source video."""


def source_video_metadata(
    source_indices: list[int],
    source_fps: float,
    total_frames: int,
    height: int,
    width: int,
) -> dict:
    """Keep frame indices and FPS in the source video's time coordinates."""
    return {
        "frames_indices": list(source_indices),
        "fps": source_fps,
        "total_num_frames": total_frames,
        "duration": total_frames / source_fps,
        "height": height,
        "width": width,
        "video_backend": "decord",
    }
