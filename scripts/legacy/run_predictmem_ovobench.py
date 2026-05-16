#!/usr/bin/env python3
"""Real Qwen3.5 evaluator for PredictMem on OVO-Bench.

Uses a shared FramePlan so Qwen and V-JEPA always operate on the same frames.
Supports four stream modes and one-shot single-window policies.

Stream modes:
  - full:              Read entire chunk (default for experiments)
  - tail_budget:       Last N frames (ablation only)
  - uniform_budget:    N frames uniformly spaced (low-cost approximation)
  - first_budget:      First N frames (debug only)

Window policies (single-window, for prompt/scoring smoke):
  - anchor_end16:      Last 16 frames of target section
  - first16:           First 16 frames (debug only)

Usage:
    # Full-stream baseline on sample 0
    python scripts/run_predictmem_ovobench.py \
        --model_path /data/model_weights_public/Qwen/Qwen3.5-9B \
        --method baseline --stream_mode full \
        --max_samples 1 --disable_thinking --max_new_tokens 16 \
        --output results/debug_full_baseline_sample0.jsonl --device cuda

    # Full-stream PredictMem on sample 0
    python scripts/run_predictmem_ovobench.py \
        --model_path /data/model_weights_public/Qwen/Qwen3.5-9B \
        --method predictmem --stream_mode full \
        --cache_path results/predictmem_scores_full_sample0.jsonl \
        --max_samples 1 --keep_ratio 0.5 --disable_thinking \
        --max_new_tokens 16 \
        --output results/debug_full_predictmem_sample0.jsonl --device cuda
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _path in (_repo_root, _models_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from predictmem.config import PredictMemConfig
from predictmem.token_mapping import TokenMapper
from predictmem.cache import ScoreCache
from predictmem.frame_plan import FramePlan, build_frame_plan
from predictmem.video_sampling import sample_video_1fps_decord


# ─── OVO Prompt Builder ─────────────────────────────────────────────────────

_MC_TASKS = {"EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"}
_YESNO_TASKS = {"SSR", "CRR"}
_NUMBER_TASKS = {"REC"}


def _option_letter(idx: int) -> str:
    return chr(ord("A") + idx)


def build_ovo_prompt(sample: dict) -> tuple[str, str, str]:
    """Build OVO-format prompt. Returns (prompt_text, task, ground_truth_letter)."""
    task = sample["task"]
    question = sample.get("question", "")
    options = sample.get("options", [])
    answer = sample.get("answer", "")
    gt_idx = sample.get("gt")

    if task in _MC_TASKS and options:
        letters = [_option_letter(i) for i in range(len(options))]
        options_str = "; ".join(f"{l}. {opt}" for l, opt in zip(letters, options))
        prompt = (
            f"Question: {question}\n"
            f"Options:\n"
            f"{options_str}\n\n"
            f"Respond only with the letter corresponding to your chosen option "
            f"(e.g., {', '.join(letters[:3])}).\n"
            f"Do not include any additional text or explanation in your response."
        )
        gt_letter = _option_letter(gt_idx) if gt_idx is not None else ""
        return prompt, task, gt_letter

    elif task in _YESNO_TASKS:
        if question:
            prompt = (
                f"Question: {question}\n\n"
                f"Respond only with Yes or No.\n"
                f"Do not include any additional text or explanation in your response."
            )
        else:
            test_info = sample.get("test_info", [])
            if test_info:
                ti = test_info[0]
                step = ti.get("step", "this step")
                prompt = (
                    f"Is the following step being performed at the indicated time?\n"
                    f"Step: {step}\n\n"
                    f"Respond only with Yes or No.\n"
                    f"Do not include any additional text or explanation in your response."
                )
            else:
                prompt = "Respond only with Yes or No."
        gt_letter = ""
        if answer:
            gt_letter = "Yes" if "yes" in answer.lower() else answer
        elif sample.get("test_info"):
            ti_type = sample["test_info"][0].get("type")
            gt_letter = "Yes" if ti_type == 1 else "No"
        return prompt, task, gt_letter

    elif task in _NUMBER_TASKS:
        activity = sample.get("activity", "the action")
        prompt = (
            f"How many times was {activity} performed in the video?\n\n"
            f"Respond only with a number.\n"
            f"Do not include any additional text or explanation in your response."
        )
        gt_letter = ""
        test_info = sample.get("test_info", [])
        if test_info:
            gt_letter = str(test_info[0].get("count", ""))
        return prompt, task, gt_letter

    else:
        if question:
            prompt = f"Question: {question}\n\nAnswer concisely in a few words."
        else:
            prompt = "Describe what is happening in the video."
        gt_letter = answer
        return prompt, task, gt_letter


# ─── Response Parser ─────────────────────────────────────────────────────────

def parse_response(raw_response: str, task: str, num_options: int = 0) -> str:
    """Parse model output into canonical format: letter, Yes/No, or number."""
    cleaned = raw_response.strip()

    if task in _MC_TASKS and num_options > 0:
        letter_match = re.search(r'\b([A-D])\b', cleaned, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
        return ""

    elif task in _YESNO_TASKS:
        yes_match = re.search(r'\b(Yes|Y)\b', cleaned, re.IGNORECASE)
        no_match = re.search(r'\b(No|N)\b', cleaned, re.IGNORECASE)
        if yes_match and not no_match:
            return "Yes"
        elif no_match and not yes_match:
            return "No"
        elif yes_match and no_match:
            return "Yes" if yes_match.start() < no_match.start() else "No"
        return ""

    elif task in _NUMBER_TASKS:
        num_match = re.search(r'\b(\d+)\b', cleaned)
        return num_match.group(1) if num_match else ""

    else:
        return cleaned


def score_parsed_response(parsed: str, ground_truth_letter: str, task: str) -> int:
    """Score parsed response against ground truth. Returns 0 or 1."""
    if not parsed or not ground_truth_letter:
        return 0

    if task in _MC_TASKS:
        return 1 if parsed.upper() == ground_truth_letter.upper().strip() else 0
    elif task in _YESNO_TASKS:
        p = parsed.strip().lower()
        g = ground_truth_letter.strip().lower()
        return 1 if p == g else 0
    elif task in _NUMBER_TASKS:
        return 1 if parsed.strip() == ground_truth_letter.strip() else 0
    else:
        npred = " ".join(parsed.lower().split())
        ngt = " ".join(ground_truth_letter.lower().split())
        return 1 if ngt and npred and ngt in npred else 0


# ─── Model / Processor Loading ───────────────────────────────────────────────

def load_qwen35_model(model_path: str, device: str):
    from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else device,
    )


def load_qwen35_processor(model_path: str, fps: float):
    from transformers import AutoProcessor, AutoTokenizer
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        from models.qwen3_5.processing_qwen3_vl import Qwen3_5Processor
        from models.qwen3_5.video_processing_qwen3_vl import Qwen3_5VideoProcessor
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            video_processor = Qwen3_5VideoProcessor.from_pretrained(model_path, fps=fps)
        except Exception:
            video_processor = Qwen3_5VideoProcessor(fps=fps)
        processor = Qwen3_5Processor(
            image_processor=None, tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=getattr(tokenizer, "chat_template", None),
        )
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = fps
        processor.video_processor.do_resize = False
    return processor


def move_inputs_to_device(inputs: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def apply_chat_template_for_generation(processor, messages, disable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def decode_new_tokens(processor, generated_ids: torch.Tensor, input_len: int) -> str:
    output_ids = generated_ids[0, input_len:]
    if hasattr(processor, "decode"):
        return processor.decode(output_ids, skip_special_tokens=True)
    return processor.tokenizer.decode(output_ids, skip_special_tokens=True)


# ─── V-JEPA Scoring (FramePlan-based) ────────────────────────────────────────

def score_vjepa_windows(
    frame_plan: FramePlan,
    config: PredictMemConfig,
    cache: ScoreCache,
    sample_id: str,
    device: str,
    score_mode: str = "online_latest_tubelet",
) -> tuple[torch.Tensor, float, list[dict]]:
    """Run V-JEPA sliding-window scoring over a FramePlan.

    Args:
        frame_plan: shared sampling plan
        config: PredictMemConfig
        cache: ScoreCache for window-level results
        sample_id: OVO sample id
        device: cuda device
        score_mode: ``online_latest_tubelet`` (14→2) or ``offline_all_tubelets``

    Returns:
        (global_scores, cache_build_latency_s, window_debug_list)
        global_scores: [num_tubelets, 16, 16] mean-aggregated losses
    """
    from predictmem.vjepa_scorer import VJEPAPredictLossScorer, make_vjepa_encoder_predictor

    checkpoint = "/data/model_weights_public/jepa/jeap_vitl_16_256.pt"
    models = make_vjepa_encoder_predictor(checkpoint_path=checkpoint, device=device)
    scorer = VJEPAPredictLossScorer(
        config, models["context_encoder"], models["target_encoder"],
        models["predictor"], degraded=models["degraded"],
    )

    window_frames = config.window_frames  # 16
    stride_frames = config.temporal_stride  # 2

    num_tubelets = frame_plan.num_tubelets
    global_loss_sum = torch.zeros(num_tubelets, 16, 16)
    global_loss_count = torch.zeros(num_tubelets, 16, 16)

    t0 = time.perf_counter()
    window_debug: list[dict] = []

    for local_start in frame_plan.window_starts(window_frames, stride_frames):
        bounds = frame_plan.window_bounds(local_start, window_frames)
        if len(bounds["source_indices"]) != window_frames:
            continue

        # Use cache if available
        cache_key = _make_cache_key(
            sample_id, frame_plan.stream_mode, frame_plan.frame_budget,
            frame_plan.frame_plan_start_s, local_start, window_frames,
            stride_frames, score_mode,
        )
        if cache.has(cache_key):
            loss_map = cache.get_loss_map(cache_key)
            if loss_map is not None:
                _accumulate_loss(global_loss_sum, global_loss_count, loss_map, local_start, score_mode)
                window_debug.append({"local_start": local_start, "cached": True, **bounds})
                continue

        # Sample and score
        try:
            frames_256 = frame_plan.get_vjepa_tensor(local_start, window_frames).to(device)
        except Exception:
            continue

        if score_mode == "online_latest_tubelet":
            # Only score the latest (last) 2-frame tubelet for this window
            # Context = first 14 frames, target = last 2 frames
            new_tubelet_id = window_frames // 2 - 1  # 7 for 16-frame window
            score = scorer.score_window_online(frames_256, new_tubelet_id=new_tubelet_id)
        else:
            score = scorer.score_window(frames_256)

        loss_map = score.loss_map.cpu()
        if loss_map.dim() == 4:
            loss_map = loss_map.squeeze(0)

        keep_mask = score.keep_mask.cpu()
        if keep_mask.dim() == 4:
            keep_mask = keep_mask.squeeze(0)

        # Cache this window
        ki = score.keep_indices[0].cpu() if score.keep_indices else None
        cache.put(
            cache_key, loss_map, keep_mask, ki,
            window_id=local_start,
            window_start_s=bounds["window_start_s"],
            window_end_s=bounds["window_end_s_exclusive"],
            source_indices=bounds["source_indices"],
        )

        _accumulate_loss(global_loss_sum, global_loss_count, loss_map, local_start, score_mode)
        window_debug.append({
            "local_start": local_start, "cached": False,
            "loss_map_shape": list(loss_map.shape),
            **bounds,
        })

    cache.flush()
    t1 = time.perf_counter()

    # Average overlapping windows
    global_loss_count = global_loss_count.clamp(min=1)
    global_scores = global_loss_sum / global_loss_count

    return global_scores, t1 - t0, window_debug


def _accumulate_loss(
    global_loss_sum: torch.Tensor,
    global_loss_count: torch.Tensor,
    loss_map: torch.Tensor,
    local_start: int,
    score_mode: str,
):
    """Accumulate window loss into global tubelet scores."""
    if score_mode == "online_latest_tubelet":
        # loss_map is [1, 16, 16] for only the latest tubelet
        # or [8, 16, 16] from score_window_online which returns full keep_mask
        if loss_map.shape[0] == 1:
            # Only the latest tubelet
            global_t = local_start // 2 + (16 // 2 - 1)
            if global_t < global_loss_sum.shape[0]:
                global_loss_sum[global_t] += loss_map.squeeze(0) if loss_map.dim() == 4 else loss_map[0]
                global_loss_count[global_t] += 1
        else:
            # Full 8-tubelet map from score_window_online; only use last tubelet
            global_t = local_start // 2 + (loss_map.shape[0] - 1)
            if global_t < global_loss_sum.shape[0]:
                global_loss_sum[global_t] += loss_map[-1]
                global_loss_count[global_t] += 1
    else:
        # offline_all_tubelets: accumulate all tubelets
        for local_t in range(loss_map.shape[0]):
            global_t = local_start // 2 + local_t
            if global_t < global_loss_sum.shape[0]:
                global_loss_sum[global_t] += loss_map[local_t]
                global_loss_count[global_t] += 1


def _make_cache_key(
    sample_id: str, stream_mode: str, frame_budget: int,
    frame_plan_start_s: float, local_window_start: int,
    window_frames: int, stride_frames: int, score_mode: str,
) -> str:
    """Build a reproducible cache key from sampling parameters."""
    return (
        f"{sample_id}|{stream_mode}|b{frame_budget}|"
        f"s{frame_plan_start_s:.1f}|w{local_window_start}|"
        f"wf{window_frames}|sf{stride_frames}|{score_mode}"
    )


# ─── Keep Index Generators ───────────────────────────────────────────────────

def generate_keep_indices_for_method(
    method: str, config: PredictMemConfig, sample_id: str,
    cache: ScoreCache | None, seed: int, video_grid_thw: torch.Tensor,
    mapper: TokenMapper, global_scores: torch.Tensor | None = None,
) -> tuple[list, dict]:
    """Generate keep_indices. Uses global_scores for predictmem full-stream."""
    num_tokens = mapper.compute_num_video_tokens(video_grid_thw)
    stats = {"original_video_tokens": num_tokens}

    if method == "baseline":
        keep = [torch.arange(num_tokens)]
        stats["keep_ratio_actual"] = 1.0
        stats["kept_video_tokens"] = num_tokens
    elif method == "random":
        torch.manual_seed(seed)
        n_keep = max(1, int(num_tokens * config.keep_ratio))
        perm = torch.randperm(num_tokens)
        keep = [perm[:n_keep].sort().values]
        stats["keep_ratio_actual"] = n_keep / num_tokens
        stats["kept_video_tokens"] = n_keep
    elif method == "uniform":
        n_keep = max(1, int(num_tokens * config.keep_ratio))
        step = num_tokens / n_keep
        keep_local = torch.round(torch.arange(n_keep).float() * step).long().clamp(0, num_tokens - 1).unique()
        keep = [keep_local.sort().values]
        stats["keep_ratio_actual"] = len(keep[0]) / num_tokens
        stats["kept_video_tokens"] = len(keep[0])
    elif method == "predictmem":
        if global_scores is not None:
            keep = mapper.map_scores_to_qwen_keep_indices(
                video_grid_thw=video_grid_thw, loss_map=global_scores,
                keep_mask=None, keep_ratio=config.keep_ratio,
            )
        elif cache is not None:
            # Legacy single-window path
            loss_map = cache.get_loss_map(sample_id)
            keep_mask = None if loss_map is not None else cache.get_keep_mask(sample_id)
            keep = mapper.map_scores_to_qwen_keep_indices(
                video_grid_thw=video_grid_thw, loss_map=loss_map,
                keep_mask=keep_mask, keep_ratio=config.keep_ratio,
            )
        else:
            raise ValueError("predictmem requires cache or global_scores")
        stats["keep_ratio_actual"] = len(keep[0]) / num_tokens
        stats["kept_video_tokens"] = len(keep[0])
    else:
        raise ValueError(f"Unknown method: {method}")

    return keep, stats


# ─── Sample Loading ──────────────────────────────────────────────────────────

def load_ovo_samples(bench_path: str, max_samples: int, video_base: str) -> list[dict]:
    with open(bench_path) as f:
        data = json.load(f)
    samples = []
    for item in data[:max_samples]:
        samples.append({
            **item,
            "sample_id": str(item["id"]),
            "video_path": str(Path(video_base) / f"{item['id']}.mp4"),
        })
    return samples


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--method", required=True,
                        choices=["baseline", "random", "uniform", "predictmem"])
    parser.add_argument("--cache_path", default="results/predictmem_scores.jsonl")
    parser.add_argument("--bench_path", default="evaluate/ovobench/ovo_bench_new.json")
    parser.add_argument("--video_dir", default="/data/qinian_workspace/OVO-Bench/chunked_videos")
    parser.add_argument("--output", default="results/real_eval.jsonl")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--qwen_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    # Window / Stream mode
    parser.add_argument("--window_policy", default="anchor_end16",
                        choices=["first16", "anchor_end16"],
                        help="Single-window policy (for prompt smoke).")
    parser.add_argument("--stream_mode", default=None,
                        choices=["full", "tail_budget", "uniform_budget", "first_budget"],
                        help="Multi-window stream mode. When set, overrides --window_policy "
                             "for the full evaluation.")
    parser.add_argument("--frame_budget", type=int, default=0,
                        help="Max frames for budgeted stream modes (0=no limit).")
    # Other flags
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--visual_ablation", default="normal",
                        choices=["normal", "black", "shuffle", "text_only"])
    parser.add_argument("--score_mode", default="online_latest_tubelet",
                        choices=["online_latest_tubelet", "offline_all_tubelets"])
    parser.add_argument("--allow_dynamic_grid", action="store_true")
    args = parser.parse_args()

    config = PredictMemConfig()
    config.keep_ratio = args.keep_ratio
    config.fps = args.fps
    config.window_frames = args.num_frames
    config.qwen_size = args.qwen_size
    config.__post_init__()

    # Resolve mode: stream_mode takes precedence
    is_stream = args.stream_mode is not None
    stream_mode = args.stream_mode or "full"

    # Load model
    print(f"Loading Qwen3.5 from {args.model_path}...")
    model = load_qwen35_model(args.model_path, args.device)
    model.eval()
    processor = load_qwen35_processor(args.model_path, args.fps)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.1f}B params, video_token_id={model.config.video_token_id}")

    mapper = TokenMapper(config)

    # Load samples
    samples = load_ovo_samples(args.bench_path, args.max_samples, args.video_dir)
    mode_label = f"stream_mode={stream_mode}" if is_stream else f"window_policy={args.window_policy}"
    print(f"Running {len(samples)} samples, method={args.method}, "
          f"{mode_label}, max_new_tokens={args.max_new_tokens}")

    cache = ScoreCache(args.cache_path) if args.method == "predictmem" else None

    results = []
    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        task = sample["task"]
        seed = args.seed + i

        print(f"\n[{i+1}/{len(samples)}] sample_id={sid} task={task} method={args.method}")

        try:
            # Build OVO prompt
            prompt_text, task_type, gt_letter = build_ovo_prompt(sample)

            cache_build_latency = 0.0
            global_scores = None
            window_debug = []
            frame_plan = None

            if is_stream:
                # ── Stream mode: build FramePlan ──
                frame_plan = build_frame_plan(
                    sample["video_path"],
                    stream_mode=stream_mode,
                    frame_budget=args.frame_budget,
                    target_fps=config.fps,
                    qwen_size=config.qwen_size,
                    vjepa_size=config.jepa_size,
                )
                plan_info = frame_plan.to_dict()
                print(f"  FramePlan: {plan_info['frame_plan_num_frames']} frames, "
                      f"[{plan_info['frame_plan_start_s']:.0f}s, {plan_info['frame_plan_end_s_exclusive']:.0f}s), "
                      f"tubelets={plan_info['num_tubelets']}, "
                      f"truncated={plan_info['full_stream_truncated']}")

                if args.method == "predictmem":
                    print(f"  V-JEPA scoring: mode={args.score_mode}, "
                          f"windows={len(list(frame_plan.window_starts(16, 2)))}")
                    global_scores, cache_build_latency, window_debug = score_vjepa_windows(
                        frame_plan, config, cache, sid, args.device,
                        score_mode=args.score_mode,
                    )
                    print(f"  Global scores: {list(global_scores.shape)}, "
                          f"cache_lat={cache_build_latency:.1f}s")

                # Qwen gets all frames from the FramePlan
                qwen_frames = frame_plan.qwen_frames_uint8
                window_start_s = frame_plan.frame_plan_start_s
                window_end_s = frame_plan.frame_plan_end_s_exclusive
                video_duration_s = frame_plan.video_duration_s
                source_indices = frame_plan.source_indices
                num_frames_np = qwen_frames.shape[0]

            else:
                # ── Single-window mode ──
                import decord
                decord.bridge.set_bridge("torch")
                vr = decord.VideoReader(sample["video_path"])
                duration = len(vr) / (vr.get_avg_fps() or config.fps)

                window_size_s = config.window_frames / config.fps
                if args.window_policy == "anchor_end16":
                    start_s = max(0.0, duration - window_size_s)
                else:
                    start_s = 0.0

                ws = sample_video_1fps_decord(
                    sample["video_path"], num_frames=config.window_frames,
                    size=config.qwen_size, target_fps=config.fps,
                    start_time_s=start_s,
                )
                qwen_frames = ws.frames_uint8
                window_start_s = start_s
                window_end_s = start_s + config.window_frames / config.fps
                video_duration_s = duration
                source_indices = ws.source_indices
                num_frames_np = qwen_frames.shape[0]

            print(f"  Frames: {num_frames_np}, "
                  f"indices=[{source_indices[0]}...{source_indices[-1]}]")

            # ── Visual ablation ──
            frames_for_model = qwen_frames.copy()
            if args.visual_ablation == "black":
                frames_for_model = np.zeros_like(frames_for_model)
            elif args.visual_ablation == "shuffle":
                perm = np.random.permutation(len(frames_for_model))
                frames_for_model = frames_for_model[perm]

            # ── Build messages ──
            content_blocks = []
            if args.visual_ablation != "text_only":
                content_blocks.append({"type": "video", "video": frames_for_model, "fps": config.fps})
            content_blocks.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content_blocks}]

            text = apply_chat_template_for_generation(processor, messages, args.disable_thinking)

            if args.visual_ablation == "text_only":
                inputs = processor(text=[text], return_tensors="pt")
            else:
                # Use the original metadata from pre-sampled frames
                video_metadata = {
                    "total_num_frames": num_frames_np,
                    "fps": float(config.fps),
                    "duration": num_frames_np / float(config.fps),
                    "frames_indices": list(range(num_frames_np)),
                    "height": int(frames_for_model.shape[1]),
                    "width": int(frames_for_model.shape[2]),
                    "video_backend": "decord",
                }
                inputs = processor(
                    text=[text], videos=[frames_for_model],
                    video_metadata=[video_metadata],
                    do_sample_frames=False, do_resize=False, fps=config.fps,
                    return_tensors="pt",
                )
            inputs = move_inputs_to_device(inputs, args.device)

            video_grid_thw = inputs["video_grid_thw"].detach().cpu()
            num_qwen_video_tokens = mapper.compute_num_video_tokens(video_grid_thw)
            print(f"  video_grid_thw: {video_grid_thw.tolist()} -> {num_qwen_video_tokens} Qwen video tokens")

            # Visual debug fields
            tokenizer = processor.tokenizer
            video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
            input_ids_cpu = inputs["input_ids"][0].detach().cpu()
            video_token_count = int((input_ids_cpu == video_token_id).sum().item())
            mm_video_token_count = None
            if "mm_token_type_ids" in inputs:
                mm_video_token_count = int((inputs["mm_token_type_ids"][0].detach().cpu() == 2).sum().item())
            pvv_shape = (
                list(inputs["pixel_values_videos"].shape) if "pixel_values_videos" in inputs else None
            )
            token_count_match = (
                video_token_count == num_qwen_video_tokens
                and (mm_video_token_count is None or mm_video_token_count == num_qwen_video_tokens)
                and pvv_shape is not None
            )

            # ── Generate keep indices ──
            keep, _ = generate_keep_indices_for_method(
                method=args.method, config=config, sample_id=sid,
                cache=cache, seed=seed, video_grid_thw=video_grid_thw,
                mapper=mapper, global_scores=global_scores,
            )
            kept = keep[0].shape[0]
            keep_ratio_actual = kept / num_qwen_video_tokens
            print(f"  Keep: {kept}/{num_qwen_video_tokens} ({keep_ratio_actual:.1%})")

            # ── Generate (single call per sample) ──
            generate_kwargs = {"use_predictmem": False}
            if args.method != "baseline":
                generate_kwargs["use_predictmem"] = True
                generate_kwargs["predictmem_keep_indices"] = keep

            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=None, top_p=None,
                    **generate_kwargs,
                )

            t1 = time.perf_counter()
            total_latency = t1 - t0
            raw_response = decode_new_tokens(processor, generated_ids, inputs["input_ids"].shape[1])
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if args.device == "cuda" else 0.0

            # ── Parse and score ──
            num_options = len(sample.get("options", []))
            parsed_response = parse_response(raw_response, task, num_options)
            score = score_parsed_response(parsed_response, gt_letter, task)

            # ── Build output entry ──
            plan_dict = frame_plan.to_dict() if frame_plan else {}
            entry = {
                "sample_id": sid,
                "task": task,
                "video": sample["video_path"],
                "question": sample.get("question", ""),
                "answer_text": sample.get("answer", ""),
                "ground_truth_letter": gt_letter,
                "options": sample.get("options", []),
                "response": parsed_response,
                "raw_response": raw_response,
                "score": score,
                "method": args.method,
                # Window / stream
                "stream_mode": stream_mode if is_stream else "single_window",
                "window_policy": args.window_policy if not is_stream else None,
                "frame_budget": args.frame_budget,
                "score_mode": args.score_mode if is_stream else None,
                "window_start_s": round(window_start_s, 2),
                "window_end_s_exclusive": round(window_end_s, 2),
                "video_duration_s": round(video_duration_s, 2),
                "source_indices": source_indices,
                **plan_dict,
                # Frames
                "fps": config.fps,
                "num_frames": num_frames_np,
                "qwen_size": config.qwen_size,
                "video_grid_thw": video_grid_thw.tolist(),
                # Visual debug
                "chat_contains_video_pad": "<|video_pad|>" in text,
                "chat_opens_thinking": text.rstrip().endswith("<think>"),
                "chat_has_empty_think_block": "<think>\n\n</think>" in text,
                "video_token_count": video_token_count,
                "mm_video_token_count": mm_video_token_count,
                "expected_video_token_count": num_qwen_video_tokens,
                "pixel_values_videos_shape": pvv_shape,
                "token_count_match": token_count_match,
                # Ablation
                "visual_ablation": args.visual_ablation,
                "disable_thinking": args.disable_thinking,
                # Keep
                "keep_ratio_target": config.keep_ratio,
                "keep_ratio_actual": round(keep_ratio_actual, 3),
                "original_video_tokens": num_qwen_video_tokens,
                "kept_video_tokens": kept,
                # Latency
                "cache_build_latency_s": round(cache_build_latency, 3),
                "score_latency_s": 0.0,
                "vision_latency_s": 0.0,
                "prefill_latency_s": round(total_latency, 4),
                "decode_latency_s": 0.0,
                "total_latency_s": round(total_latency, 4),
                "peak_memory_mb": round(peak_memory, 1),
                # Window debug (P3: per-window metadata, not answers)
                "num_sliding_windows": len(window_debug),
            }

            results.append(entry)
            print(f"  Raw: '{raw_response.strip()}'")
            print(f"  Parsed: '{parsed_response}' | GT: '{gt_letter}' | Score: {score}")
            print(f"  Latency: {total_latency:.3f}s, Peak mem: {peak_memory:.0f}MB")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            prompt_text, task_type, gt_letter = build_ovo_prompt(sample)
            results.append({
                "sample_id": sid, "task": task, "video": sample["video_path"],
                "question": sample.get("question", ""),
                "answer_text": sample.get("answer", ""),
                "ground_truth_letter": gt_letter,
                "options": sample.get("options", []),
                "response": f"[error: {str(e)[:100]}]", "raw_response": "", "score": 0,
                "method": args.method,
                "stream_mode": stream_mode if is_stream else "single_window",
                "window_policy": args.window_policy if not is_stream else None,
                "frame_budget": args.frame_budget,
                "score_mode": args.score_mode if is_stream else None,
                "full_stream_truncated": False,
                "visual_ablation": args.visual_ablation,
                "disable_thinking": args.disable_thinking,
            })

    # Write output — one line per sample (P3)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults: {len(results)} entries -> {args.output}")


if __name__ == "__main__":
    main()
