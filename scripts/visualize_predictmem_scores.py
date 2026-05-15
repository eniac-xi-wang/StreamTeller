#!/usr/bin/env python3
"""Visualize PredictMem prediction loss as overlay heatmaps.

Outputs high-loss (keep) and low-loss (drop) patch overlays for a video sample.

Usage:
    python scripts/visualize_predictmem_scores.py \
        --video /data/qinian_workspace/OVO-Bench/chunked_videos/0.mp4 \
        --cache_path results/predictmem_scores.jsonl \
        --sample_id 0 \
        --output_dir results/overlays
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

_models_dir = Path(__file__).parent.parent / "models"
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from predictmem.cache import ScoreCache


def make_overlay(frames_512: np.ndarray, keep_mask: torch.Tensor, tubelet_id: int) -> np.ndarray:
    """Create a side-by-side: original frame + keep/drop overlay.

    Args:
        frames_512: numpy array [16, 512, 512, 3] in uint8
        keep_mask: [8, 16, 16] bool tensor
        tubelet_id: which tubelet to visualize (0..7)

    Returns:
        overlay image as numpy array
    """
    import cv2

    H, W = 512, 512
    grid_h, grid_w = 16, 16
    patch_h, patch_w = H // grid_h, W // grid_w  # 32x32

    # Frames for this tubelet (2 frames)
    f_start = tubelet_id * 2
    f_end = f_start + 2

    rows = []
    for f_idx in range(f_start, min(f_end, len(frames_512))):
        frame = frames_512[f_idx].copy()
        overlay = frame.copy()

        for h in range(grid_h):
            for w in range(grid_w):
                y1, y2 = h * patch_h, (h + 1) * patch_h
                x1, x2 = w * patch_w, (w + 1) * patch_w
                kept = bool(keep_mask[tubelet_id, h, w].item())

                if kept:
                    # Green tint for kept patches
                    overlay[y1:y2, x1:x2, 1] = np.clip(overlay[y1:y2, x1:x2, 1].astype(int) + 60, 0, 255).astype(np.uint8)
                    # Green border
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                else:
                    # Red tint for dropped patches
                    overlay[y1:y2, x1:x2, 2] = np.clip(overlay[y1:y2, x1:x2, 2].astype(int) + 60, 0, 255).astype(np.uint8)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)

        combined = np.hstack([frame, overlay])
        label = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(label, f"Frame {f_idx+1} (tubelet {tubelet_id})  |  Left: Original  |  Right: Green=Kept Red=Dropped",
                   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        rows.append(np.vstack([label, combined]))

    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--output_dir", default="results/overlays")
    args = parser.parse_args()

    cache = ScoreCache(args.cache_path)
    if not cache.has(args.sample_id):
        print(f"ERROR: sample {args.sample_id} not in cache")
        sys.exit(1)

    keep_mask = cache.get_keep_mask(args.sample_id)
    loss_map = cache.get_loss_map(args.sample_id)
    # Squeeze batch dim if present
    if keep_mask.ndim == 4:
        keep_mask = keep_mask[0]
    if loss_map.ndim == 4:
        loss_map = loss_map[0]
    print(f"Loaded: keep_mask shape={keep_mask.shape}, kept={keep_mask.sum().item()}/{keep_mask.numel()}")

    # Read video at 512 resolution for Qwen
    import cv2
    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (512, 512))
        frames.append(frame)
    cap.release()

    if len(frames) < 16:
        print(f"WARNING: only {len(frames)} frames available")
        while len(frames) < 16:
            frames.append(frames[-1])

    frames = np.stack(frames)  # [16, 512, 512, 3]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate overlays for each tubelet
    for t in range(8):
        img = make_overlay(frames, keep_mask, t)
        out_path = output_dir / f"{args.sample_id}_tubelet_{t:02d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        n_kept = keep_mask[t].sum().item()
        print(f"  Tubelet {t}: {n_kept}/256 kept -> {out_path}")

    # Generate a summary heatmap image
    # Per-tubelet mean loss
    loss_np = loss_map.numpy()  # [8, 16, 16]
    heatmap_rows = []
    for t in range(8):
        tubelet_loss = loss_np[t]  # [16, 16]
        # Normalize to [0, 255]
        vmin, vmax = loss_np.min(), loss_np.max()
        if vmax - vmin > 0:
            loss_img = ((tubelet_loss - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            loss_img = np.zeros_like(tubelet_loss, dtype=np.uint8)
        loss_img = cv2.applyColorMap(loss_img, cv2.COLORMAP_JET)
        loss_img = cv2.resize(loss_img, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.putText(loss_img, f"T{t} loss", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        heatmap_rows.append(loss_img)

    heatmap = np.vstack([np.hstack(heatmap_rows[:4]), np.hstack(heatmap_rows[4:])])
    heatmap_path = output_dir / f"{args.sample_id}_loss_heatmap.png"
    cv2.imwrite(str(heatmap_path), heatmap)
    print(f"  Loss heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
