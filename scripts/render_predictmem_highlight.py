#!/usr/bin/env python3
"""Generate highlight MP4 video from PredictMem keep/drop masks.

Reads a PredictMem plugin JSONL result entry (or standalone keepmask JSON)
and renders a video where:
  - kept patches: original brightness
  - dropped patches: darkened to 30%
  - bootstrap_drop: all darkened, annotated
  - protected_tail_full_keep: all original brightness, annotated

Usage:
    python scripts/render_predictmem_highlight.py \
        --jsonl results/predictmem_plugin_10.jsonl \
        --sample 0 \
        --output results/visualizations/
"""

import argparse
import json
from pathlib import Path

import cv2
import decord
import numpy as np


def load_keep_masks_from_entry(entry: dict) -> dict:
    """Extract keep masks from a JSONL result entry."""
    pm_stats = entry.get("predictmem_stats", {})
    masks = pm_stats.get("predictmem_keep_masks", None)
    if masks is None:
        raise ValueError("Entry does not contain predictmem_keep_masks")
    return masks


def render_highlight_video(
    video_path: str,
    keep_masks: dict,
    num_frames: int,
    sample_id: str,
    output_path: str,
    fps: float = 1.0,
):
    """Render highlight MP4 from keep masks.

    Args:
        video_path: path to original mp4
        keep_masks: dict with grid_h, grid_w, tubelets list
        num_frames: total 1FPS frames
        sample_id: OVO sample id
        output_path: output mp4 path
        fps: target FPS for frame sampling
    """
    grid_h = keep_masks["grid_h"]
    grid_w = keep_masks["grid_w"]
    tubelets = keep_masks["tubelets"]

    # Build per-frame mask from tubelets
    per_frame_mask = {}  # frame_idx -> np.ndarray [grid_h, grid_w] bool
    per_frame_mode = {}  # frame_idx -> str

    for tinfo in tubelets:
        t = tinfo["tubelet"]
        mode = tinfo["mode"]
        mask = np.array(tinfo["keep_mask"], dtype=bool).reshape(grid_h, grid_w)
        for f_idx in tinfo["frames"]:
            if f_idx < num_frames:
                per_frame_mask[f_idx] = mask
                per_frame_mode[f_idx] = mode

    # Load original frames
    vr = decord.VideoReader(str(video_path))
    native_fps = vr.get_avg_fps()
    step = max(1, int(native_fps / fps))
    frame_indices = list(range(0, len(vr), step))[:num_frames]
    original_frames = vr.get_batch(frame_indices).asnumpy()  # RGB

    # Generate highlight frames
    highlight_frames = []

    for f_idx in range(num_frames):
        original = original_frames[f_idx].copy()
        h, w = original.shape[:2]

        if f_idx in per_frame_mask:
            mask_16x16 = per_frame_mask[f_idx]
            mode = per_frame_mode[f_idx]

            if mode == "bootstrap_drop":
                # Fully darkened
                composite = (original * 0.3).astype(np.uint8)
                n_kept = 0
            elif mode == "protected_tail_full_keep":
                # Fully original
                composite = original.copy()
                n_kept = grid_h * grid_w
            else:
                # scored: keep high-loss, darken low-loss
                mask_u8 = mask_16x16.astype(np.uint8)
                mask_full = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
                darkened = (original * 0.3).astype(np.uint8)
                mask_3ch = np.stack([mask_full] * 3, axis=-1)
                composite = original * mask_3ch + darkened * (1 - mask_3ch)
                composite = composite.astype(np.uint8)
                n_kept = mask_16x16.sum()

            # Determine per-patch mode for a representative patch
            if "local_quantile" in mode:
                mode_label = "scored_local_quantile"
            elif "digest" in mode:
                mode_label = "scored_digest"
            else:
                mode_label = mode
        else:
            # Frame has no mask (shouldn't happen for scored frames)
            composite = (original * 0.3).astype(np.uint8)
            n_kept = 0
            mode_label = "unscored"

        # Overlay text
        tubelet_idx = f_idx // 2
        text_lines = [
            f"sample={sample_id} frame={f_idx} tubelet={tubelet_idx}",
            f"mode={mode_label} kept={n_kept}/{grid_h * grid_w}",
        ]
        for li, line in enumerate(text_lines):
            cv2.putText(composite, line, (10, 25 + li * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)

        highlight_frames.append(composite)

    # Write MP4 using OpenCV VideoWriter (more reliable)
    frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in highlight_frames]
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 1.0, (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  Wrote: {output_path} ({size_mb:.1f} MB, {len(highlight_frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True,
                        help="Plugin JSONL results file")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample index within JSONL (0-based)")
    parser.add_argument("--keepmask_json", default=None,
                        help="Standalone keepmask JSON (overrides JSONL)")
    parser.add_argument("--output", default="results/visualizations/",
                        help="Output directory")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames to render (0=all)")
    args = parser.parse_args()

    # Load keep masks
    if args.keepmask_json:
        with open(args.keepmask_json) as f:
            keep_masks = json.load(f)
        sample_id = Path(args.keepmask_json).stem.replace("_keepmask", "")
        video_path = None
        for vid_dir in ["/data/qinian_workspace/OVO-Bench/chunked_videos"]:
            candidate = Path(vid_dir) / f"{sample_id}.mp4"
            if candidate.exists():
                video_path = str(candidate)
                break
        if video_path is None:
            raise ValueError(f"Cannot find video for sample_id={sample_id}")
        entry = {}
    else:
        with open(args.jsonl) as f:
            lines = f.readlines()
        if args.sample >= len(lines):
            raise ValueError(f"Sample index {args.sample} out of range (file has {len(lines)} lines)")
        entry = json.loads(lines[args.sample])
        keep_masks = load_keep_masks_from_entry(entry)
        sample_id = entry.get("sample_id", str(args.sample))
        video_path = entry.get("video", "")
        if not video_path or not Path(video_path).exists():
            # Try to find video
            for vid_dir in ["/data/qinian_workspace/OVO-Bench/chunked_videos"]:
                candidate = Path(vid_dir) / f"{sample_id}.mp4"
                if candidate.exists():
                    video_path = str(candidate)
                    break
        if not video_path:
            raise ValueError(f"Cannot find video for sample_id={sample_id}")

    num_frames = entry.get("num_frames", keep_masks.get("num_frames", 0))
    if num_frames == 0:
        num_frames = max(tinfo["frames"][-1] + 1 if tinfo["frames"] else 0
                         for tinfo in keep_masks["tubelets"])

    if args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)

    print(f"Rendering sample {sample_id}: {num_frames} frames, "
          f"{len(keep_masks['tubelets'])} tubelets")

    video_out = str(Path(args.output) / f"sample_{sample_id}_predictmem_highlight.mp4")
    render_highlight_video(video_path, keep_masks, num_frames, sample_id, video_out)

    # Also save standalone keepmask JSON
    keepmask_out = str(Path(args.output) / f"sample_{sample_id}_keepmask.json")
    Path(keepmask_out).parent.mkdir(parents=True, exist_ok=True)
    with open(keepmask_out, "w") as f:
        json.dump(keep_masks, f, indent=2)
    print(f"  Wrote: {keepmask_out}")


if __name__ == "__main__":
    main()
