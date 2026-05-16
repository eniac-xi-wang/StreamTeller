#!/usr/bin/env python3
"""Compare PredictMem V-JEPA scorer loss values against Survey analyzer output.

Reads the analyzer's losses.json and runs our scorer on the same video/checkpoint
with the same window schedule. Reports per-window loss differences.

Usage:
    python scripts/check_analyzer_parity.py
    python scripts/check_analyzer_parity.py --num_sample_windows 3
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).parent.parent
for _p in (_repo_root, _repo_root / "models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Survey analyzer paths (read-only reference data, no runtime imports)
ANALYZER_JSON = Path("/root/stream/Survey/survey/vjepa_loss_analyzer/losses.json")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_analyzer_reference(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def preprocess_frames_imagenet(frames_np: np.ndarray) -> torch.Tensor:
    """Resize to 256 and apply ImageNet normalization (same as analyzer)."""
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import functional as TF

    processed = []
    for f in frames_np:
        pil = Image.fromarray(f)
        resized = TF.resize(pil, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(resized)
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        processed.append(tensor)
    return torch.stack(processed)


def run_predictmem_scorer(frames_tensor: torch.Tensor, device: str):
    """Run our analyzer-compatible scorer on the same frames."""
    from predictmem.vjepa_scorer import VJEPAPredictLossScorer, make_vjepa_analyzer_scorer
    from predictmem.config import PredictMemConfig

    config = PredictMemConfig()
    config.__post_init__()

    checkpoint = "/data/model_weights_public/jepa/jeap_vitl_16_256.pt"
    models = make_vjepa_analyzer_scorer(checkpoint_path=checkpoint, device=device)
    scorer = VJEPAPredictLossScorer(
        config,
        models["context_encoder"],
        models["target_encoder"],
        models["predictor"],
        degraded=models["degraded"],
    )

    n_frames = frames_tensor.shape[0]
    window_frames = 16
    stride = 2
    patch_size = 16
    grid_size = 256 // patch_size  # 16

    our_losses_per_window = []

    # Phase 1: expanding windows (matching analyzer schedule)
    for target_end in range(3, min(window_frames, n_frames), 2):
        wlen = target_end + 1
        clip = frames_tensor[:wlen].permute(1, 0, 2, 3).unsqueeze(0).to(device)
        loss = scorer.score_latest_tubelet_variable(clip, window_frames=wlen)
        loss_2d = loss.squeeze(0).reshape(grid_size, grid_size).cpu().numpy()
        our_losses_per_window.append({
            "start_frame": 0,
            "target_frames": [target_end - 1, target_end],
            "window_frames": wlen,
            "patch_loss_16x16": loss_2d.flatten().tolist(),
            "total_loss": float(loss_2d.sum()),
        })

    # Phase 2: standard sliding windows
    for start_f in range(0, n_frames - window_frames + 1, stride):
        target_frames = [start_f + window_frames - 2, start_f + window_frames - 1]
        if target_frames[0] < window_frames - 1:
            continue  # already covered by expanding
        clip = frames_tensor[start_f:start_f + window_frames].permute(1, 0, 2, 3).unsqueeze(0).to(device)
        loss = scorer.score_latest_tubelet_variable(clip, window_frames=window_frames)
        loss_2d = loss.squeeze(0).reshape(grid_size, grid_size).cpu().numpy()
        our_losses_per_window.append({
            "start_frame": int(start_f),
            "target_frames": target_frames,
            "window_frames": window_frames,
            "patch_loss_16x16": loss_2d.flatten().tolist(),
            "total_loss": float(loss_2d.sum()),
        })

    return our_losses_per_window


def compare_windows(analyzer_windows: list, our_windows: list, num_sample: int = 3):
    """Compare analyzer vs PredictMem windows."""
    print(f"\nAnalyzer: {len(analyzer_windows)} windows, PredictMem: {len(our_windows)} windows")

    # Ensure counts match (minus any tail_keep windows our scorer would skip)
    diff = len(analyzer_windows) - len(our_windows)
    if diff != 0:
        print(f"  Window count difference: {diff}")
        print(f"  (PredictMem may skip tail-keep tubelets; analyzer scored everything)")

    # Match by target_frames
    our_by_target = {}
    for w in our_windows:
        key = tuple(w["target_frames"])
        our_by_target[key] = w

    analyzer_by_target = {}
    for w in analyzer_windows:
        key = tuple(w["target_frames"])
        analyzer_by_target[key] = w

    common_targets = sorted(set(our_by_target.keys()) & set(analyzer_by_target.keys()))
    print(f"  Common windows: {len(common_targets)}")

    if num_sample > 0 and common_targets:
        indices = np.linspace(0, len(common_targets) - 1, min(num_sample, len(common_targets)), dtype=int)
    else:
        indices = range(len(common_targets))

    results = []
    for idx in indices:
        tf = common_targets[idx]
        a_win = analyzer_by_target[tf]
        o_win = our_by_target[tf]

        a_loss = np.array(a_win["patch_loss_24x24"]).reshape(16, 16) if "patch_loss_24x24" in a_win else np.array(a_win.get("patch_loss_16x16", [0])).reshape(16, 16)
        o_loss = np.array(o_win["patch_loss_16x16"]).reshape(16, 16)

        abs_diff = np.abs(a_loss - o_loss)
        max_abs_diff = float(abs_diff.max())
        mean_abs_diff = float(abs_diff.mean())
        a_mean = float(a_loss.mean())
        o_mean = float(o_loss.mean())
        rel_diff = mean_abs_diff / (a_mean + 1e-8)

        results.append({
            "target_frames": list(tf),
            "analyzer_mean_loss": round(a_mean, 6),
            "predictmem_mean_loss": round(o_mean, 6),
            "max_abs_diff": round(max_abs_diff, 6),
            "mean_abs_diff": round(mean_abs_diff, 6),
            "relative_diff": round(rel_diff, 6),
        })
        print(f"  Window target={list(tf)}: analyzer_mean={a_mean:.6f}, ours_mean={o_mean:.6f}, "
              f"max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}")

    # Aggregate
    all_max_diff = max(r["max_abs_diff"] for r in results) if results else 0
    all_mean_diff = np.mean([r["mean_abs_diff"] for r in results]) if results else 0
    all_rel_diff = np.mean([r["relative_diff"] for r in results]) if results else 0
    num_windows = len(common_targets)

    print(f"\nAggregate: max_abs_diff={all_max_diff:.6f}, mean_abs_diff={all_mean_diff:.6f}, "
          f"relative_diff={all_rel_diff:.6f}, num_common_windows={num_windows}")

    return {
        "num_analyzer_windows": len(analyzer_windows),
        "num_predictmem_windows": len(our_windows),
        "num_common_windows": num_windows,
        "max_abs_diff": round(all_max_diff, 6),
        "mean_abs_diff": round(all_mean_diff, 6),
        "relative_diff": round(all_rel_diff, 6),
        "sample_details": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sample_windows", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/analyzer_parity.json")
    args = parser.parse_args()

    # Load analyzer reference
    ref = load_analyzer_reference(str(ANALYZER_JSON))
    print(f"Loaded analyzer: {ref['video_path']}, {ref['num_windows']} windows, "
          f"{ref['total_frames']} frames")

    # Load video frames
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(ref["video_path"])
    native_fps = vr.get_avg_fps()
    step = int(native_fps / ref["fps"])
    frame_indices = list(range(0, len(vr), step))[:ref["total_frames"]]
    frames_raw = vr.get_batch(frame_indices)
    if hasattr(frames_raw, "asnumpy"):
        frames_np = frames_raw.asnumpy()
    elif isinstance(frames_raw, np.ndarray):
        frames_np = frames_raw
    else:
        frames_np = frames_raw.numpy()
    print(f"Loaded {len(frames_np)} frames from {ref['video_path']}")

    # Preprocess with ImageNet normalization
    frames_tensor = preprocess_frames_imagenet(frames_np)
    print(f"Preprocessed: {frames_tensor.shape}")

    # Run our scorer
    our_windows = run_predictmem_scorer(frames_tensor, args.device)

    # Compare
    result = compare_windows(ref["windows"], our_windows, args.num_sample_windows)

    # Write result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
