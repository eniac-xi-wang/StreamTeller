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

_models_dir = Path(__file__).parent.parent / "models"
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from predictmem.config import PredictMemConfig
from predictmem.vjepa_scorer import VJEPAPredictLossScorer, make_vjepa_encoder_predictor
from predictmem.cache import ScoreCache
from predictmem.video_sampling import sample_video_1fps_decord


def sample_frames(
    video_path: str,
    num_frames: int = 16,
    size: int = 256,
    fps: float = 1.0,
) -> torch.Tensor | None:
    """Sample a 1FPS decord window. Returns [1, 3, 16, size, size]."""
    try:
        sample = sample_video_1fps_decord(
            video_path,
            num_frames=num_frames,
            size=size,
            target_fps=fps,
        )
        return sample.vjepa_tensor()
    except Exception as e:
        print(f"  decord 1FPS sampling error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--cache_path", default="results/predictmem_scores.jsonl")
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    config = PredictMemConfig()
    config.keep_ratio = args.keep_ratio
    config.window_frames = args.num_frames
    config.fps = args.fps
    config.__post_init__()

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
        frames = sample_frames(str(vf), num_frames=args.num_frames, fps=args.fps)
        if frames is None:
            print(f"  [{i+1}/{len(video_files)}] {sid}: READ FAILED")
            continue
        frames = frames.to(args.device)
        score = scorer.score_window(frames)
        cache.put(sid, score.loss_map.cpu(), score.keep_mask.cpu(), score.keep_indices[0].cpu())
        print(f"  [{i+1}/{len(video_files)}] {sid}: {score.keep_indices[0].shape[0]}/{config.num_jepa_tokens} kept")

    cache.flush()
    print(f"Done: {len(cache)} entries -> {args.cache_path}")


if __name__ == "__main__":
    main()
