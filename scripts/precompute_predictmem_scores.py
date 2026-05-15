#!/usr/bin/env python3
"""Precompute V-JEPA prediction loss scores for OVO-Bench videos.

Usage:
    python scripts/precompute_predictmem_scores.py \
        --checkpoint /data/model_weights_public/jepa/jeap_vitl_16_256.pt \
        --video_dir /data/qinian_workspace/OVO-Bench/chunked_videos \
        --cache_path results/predictmem_scores.jsonl \
        --max_videos 5 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

_models_dir = Path(__file__).parent.parent / "models"
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from predictmem.config import PredictMemConfig
from predictmem.vjepa_scorer import VJEPAPredictLossScorer, make_vjepa_encoder_predictor
from predictmem.cache import ScoreCache


def sample_frames(video_path: str, num_frames: int = 16, size: int = 256) -> torch.Tensor | None:
    """Sample num_frames from video, resize to (size, size). Returns [1, 3, 16, size, size]."""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(video_path))
        total = len(vr)
        if total < num_frames:
            indices = list(range(total)) + [total - 1] * (num_frames - total)
        else:
            indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        frames = vr.get_batch(indices)  # [T, H, W, C]
        frames = frames.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
        frames = frames.unsqueeze(0)  # [1, C, T, H, W]
        B, C, T, H, W = frames.shape
        frames = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        frames = torch.nn.functional.interpolate(
            frames, size=(size, size), mode="bilinear", align_corners=False,
        )
        frames = frames.view(B, T, C, size, size).permute(0, 2, 1, 3, 4)
        return frames
    except Exception as e:
        print(f"  decord error: {e}, trying cv2 fallback...")
        return _sample_frames_cv2(video_path, num_frames, size)


def _sample_frames_cv2(video_path, num_frames, size):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        indices = list(range(total)) + [total - 1] * (num_frames - total)
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames_list = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = frames_list[-1] if frames_list else np.zeros((size, size, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (size, size))
        frames_list.append(frame)
    cap.release()

    frames = np.stack(frames_list)  # [T, H, W, C]
    frames = torch.from_numpy(frames).float() / 255.0
    return frames.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--cache_path", default="results/predictmem_scores.jsonl")
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    args = parser.parse_args()

    config = PredictMemConfig()
    config.keep_ratio = args.keep_ratio

    print(f"Loading checkpoint: {args.checkpoint}")
    models = make_vjepa_encoder_predictor(checkpoint_path=args.checkpoint, device=args.device)
    print(f"  Keys: {models['keys_found']}, degraded: {models['degraded']}")

    scorer = VJEPAPredictLossScorer(
        config, models["context_encoder"], models["target_encoder"],
        models["predictor"], degraded=models["degraded"],
    )

    cache = ScoreCache(args.cache_path)
    video_files = sorted(Path(args.video_dir).glob("*.mp4"), key=lambda p: int(p.stem))[: args.max_videos]
    print(f"Processing {len(video_files)} videos...")

    for i, vf in enumerate(video_files):
        sid = vf.stem
        if cache.has(sid):
            print(f"  [{i+1}/{len(video_files)}] {sid}: cached")
            continue
        frames = sample_frames(str(vf))
        if frames is None:
            print(f"  [{i+1}/{len(video_files)}] {sid}: READ FAILED")
            continue
        frames = frames.to(args.device)
        score = scorer.score_window(frames)
        cache.put(sid, score.loss_map.cpu(), score.keep_mask.cpu(), score.keep_indices[0].cpu())
        print(f"  [{i+1}/{len(video_files)}] {sid}: {score.keep_indices[0].shape[0]}/2048 kept")

    cache.flush()
    print(f"Done: {len(cache)} entries -> {args.cache_path}")


if __name__ == "__main__":
    main()
