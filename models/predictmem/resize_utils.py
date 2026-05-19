"""Aspect-ratio preserving resize helpers for aligned Qwen/V-JEPA inputs."""

from __future__ import annotations

import math


MAX_RATIO = 200


def round_by_factor(number: float, factor: int) -> int:
    """Return the closest integer to ``number`` that is divisible by ``factor``."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Return the smallest integer >= ``number`` that is divisible by ``factor``."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Return the largest integer <= ``number`` that is divisible by ``factor``."""
    return math.floor(number / factor) * factor


def smart_resize_keep_aspect(
    height: int,
    width: int,
    factor: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """Qwen-VL style smart resize.

    This mirrors ``qwen_vl_utils.vision_process.smart_resize``: preserve aspect
    ratio as closely as possible, make both dimensions divisible by ``factor``,
    and constrain the resized pixel count.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: height={height}, width={width}")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )

    min_pixels = min_pixels if min_pixels is not None else factor * factor
    max_pixels = max_pixels if max_pixels is not None else max(height * width, min_pixels)
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be greater than or equal to min_pixels")

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return int(h_bar), int(w_bar)


def compute_aligned_resize(
    source_height: int,
    source_width: int,
    qwen_size: int = 512,
    jepa_size: int = 256,
    qwen_factor: int = 32,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return aspect-preserving Qwen and V-JEPA sizes with a 2:1 ratio.

    ``qwen_size`` is treated as the square-area budget used by the previous
    512x512 path.  The returned Qwen size is divisible by 32, so the JEPA size
    obtained by halving it is divisible by 16 and aligns one-to-one with Qwen's
    post-merge visual token grid.
    """
    if qwen_size != 2 * jepa_size:
        raise ValueError(f"Expected qwen_size == 2 * jepa_size, got {qwen_size=} and {jepa_size=}")

    qwen_pixels = qwen_size * qwen_size
    qwen_h, qwen_w = smart_resize_keep_aspect(
        source_height,
        source_width,
        factor=qwen_factor,
        min_pixels=qwen_pixels,
        max_pixels=qwen_pixels,
    )
    jepa_h, jepa_w = qwen_h // 2, qwen_w // 2
    return (qwen_h, qwen_w), (jepa_h, jepa_w)
