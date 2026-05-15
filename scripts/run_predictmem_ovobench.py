#!/usr/bin/env python3
"""Real Qwen3.5 evaluator for PredictMem on OVO-Bench.

Loads Qwen3.5-9B, samples videos per window policy, and runs four methods
(baseline, random, uniform, predictmem) with real model.generate().

Window policies:
  - first16:      First 16s window (debug only)
  - anchor_end16: Last 16s window (default smoke — question likely near end)
  - full_stream:  Sliding V-JEPA windows over full chunk, one final answer

Usage:
    # Prompt/scoring smoke
    python scripts/run_predictmem_ovobench.py \
        --model_path /data/model_weights_public/Qwen/Qwen3.5-9B \
        --method baseline --window_policy anchor_end16 \
        --max_samples 5 --max_new_tokens 8 \
        --output results/prompt_smoke_baseline_5.jsonl --device cuda

    # Full-stream PredictMem smoke
    python scripts/run_predictmem_ovobench.py \
        --model_path /data/model_weights_public/Qwen/Qwen3.5-9B \
        --method predictmem --window_policy full_stream \
        --cache_path results/predictmem_scores_stream_5.jsonl \
        --max_samples 5 --max_stream_frames 64 --keep_ratio 0.5 \
        --max_new_tokens 8 --output results/fullstream_predictmem_5.jsonl --device cuda
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

_repo_root = Path(__file__).parent.parent
_models_dir = _repo_root / "models"
for _path in (_repo_root, _models_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from predictmem.config import PredictMemConfig
from predictmem.token_mapping import TokenMapper
from predictmem.cache import ScoreCache
from predictmem.video_sampling import sample_video_1fps_decord


# ─── OVO Prompt Builder ─────────────────────────────────────────────────────

# Task types that use multiple-choice prompt
_MC_TASKS = {"EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"}

# Task types with special output format
_YESNO_TASKS = {"SSR", "CRR"}
_NUMBER_TASKS = {"REC"}


def _option_letter(idx: int) -> str:
    return chr(ord("A") + idx)


def build_ovo_prompt(sample: dict) -> tuple[str, str, str]:
    """Build OVO-format prompt. Returns (prompt_text, task, ground_truth_letter).

    ground_truth_letter is the expected answer format:
      - "A"/"B"/"C"/"D" for multiple-choice
      - "Yes"/"No" for SSR/CRR
      - number string for REC
    """
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
            # SSR without question: construct from test_info
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
        # Determine ground truth: for SSR type=1 -> Yes, type=0 -> No
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
        # Ground truth: extract count from test_info
        gt_letter = ""
        test_info = sample.get("test_info", [])
        if test_info:
            gt_letter = str(test_info[0].get("count", ""))
        return prompt, task, gt_letter

    else:
        # Fallback: direct question
        if question:
            prompt = (
                f"Question: {question}\n\n"
                f"Answer concisely in a few words."
            )
        else:
            prompt = "Describe what is happening in the video."
        gt_letter = answer
        return prompt, task, gt_letter


# ─── Response Parser ─────────────────────────────────────────────────────────

def parse_response(raw_response: str, task: str, num_options: int = 0) -> str:
    """Parse model output into a canonical format: letter, Yes/No, or number."""
    cleaned = raw_response.strip()

    if task in _MC_TASKS and num_options > 0:
        # Try to extract a single letter
        letter_match = re.search(r'\b([A-D])\b', cleaned, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

        # Try to match option text to a letter
        return ""

    elif task in _YESNO_TASKS:
        yes_match = re.search(r'\b(Yes|Y)\b', cleaned, re.IGNORECASE)
        no_match = re.search(r'\b(No|N)\b', cleaned, re.IGNORECASE)
        if yes_match and not no_match:
            return "Yes"
        elif no_match and not yes_match:
            return "No"
        elif yes_match and no_match:
            # Both found, pick the first
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
        # Substring match for open-ended
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


def decode_new_tokens(processor, generated_ids: torch.Tensor, input_len: int) -> str:
    output_ids = generated_ids[0, input_len:]
    if hasattr(processor, "decode"):
        return processor.decode(output_ids, skip_special_tokens=True)
    return processor.tokenizer.decode(output_ids, skip_special_tokens=True)


# ─── Window Policy ───────────────────────────────────────────────────────────

def get_window_bounds(video_path: str, policy: str, window_size_s: float,
                       fps: float, max_stream_frames: int) -> tuple[float, float, float]:
    """Return (start_time_s, end_time_s, video_duration_s) for the given policy."""
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)
    source_fps = float(vr.get_avg_fps() or fps)
    duration = total_frames / source_fps if source_fps > 0 else 0

    if policy == "first16":
        start = 0.0
        end = min(window_size_s, duration)
    elif policy == "anchor_end16":
        end = duration
        start = max(0.0, duration - window_size_s)
    elif policy == "full_stream":
        start = 0.0
        end = min(duration, max_stream_frames / fps if max_stream_frames else duration)
    else:
        raise ValueError(f"Unknown window_policy: {policy}")

    return start, end, duration


def sample_window(video_path: str, start_s: float, end_s: float, config: PredictMemConfig):
    """Sample frames from [start_s, end_s] at 1FPS.

    Returns (DecordVideoSample, actual_start_s, actual_end_s).
    The number of frames is determined by the time window: ceil((end-start) * fps).
    For fixed 16-frame policies, uses config.window_frames.
    """
    num_frames = max(1, int((end_s - start_s) * config.fps))
    sample = sample_video_1fps_decord(
        video_path, num_frames=num_frames, size=config.qwen_size,
        target_fps=config.fps, start_time_s=start_s,
    )
    actual_end = start_s + num_frames / config.fps
    return sample, start_s, actual_end


# ─── Keep Index Generators ───────────────────────────────────────────────────

def generate_keep_indices_for_method(
    method: str, config: PredictMemConfig, sample_id: str,
    cache: ScoreCache | None, seed: int, video_grid_thw: torch.Tensor,
    mapper: TokenMapper, window_id: int | None = None,
) -> tuple[list, dict]:
    """Generate keep_indices for one sample/window. Returns (keep_indices_list, stats)."""
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
        if cache is None:
            raise ValueError("Cache required for predictmem method")
        cache_key = f"{sample_id}:{window_id}" if window_id is not None else sample_id
        if cache.has(cache_key):
            loss_map = cache.get_loss_map(cache_key)
            keep_mask = None if loss_map is not None else cache.get_keep_mask(cache_key)
            keep = mapper.map_scores_to_qwen_keep_indices(
                video_grid_thw=video_grid_thw, loss_map=loss_map,
                keep_mask=keep_mask, keep_ratio=config.keep_ratio,
            )
            stats["keep_ratio_actual"] = len(keep[0]) / num_tokens
            stats["kept_video_tokens"] = len(keep[0])
        else:
            raise ValueError(f"Cache key {cache_key} not found")
    else:
        raise ValueError(f"Unknown method: {method}")

    return keep, stats


# ─── Full-stream V-JEPA Scoring ──────────────────────────────────────────────

def run_full_stream_scoring(
    video_path: str, config: PredictMemConfig, cache: ScoreCache,
    sample_id: str, device: str, max_stream_frames: int,
) -> tuple[torch.Tensor, float]:
    """Run sliding-window V-JEPA scoring over the full stream.

    Returns (global_scores, cache_build_latency_s).
    global_scores shape: [global_t, 16, 16] where global_t = ceil(num_stream_frames / 2).
    """
    from predictmem.vjepa_scorer import VJEPAPredictLossScorer, make_vjepa_encoder_predictor

    # Load V-JEPA once
    checkpoint = "/data/model_weights_public/jepa/jeap_vitl_16_256.pt"
    models = make_vjepa_encoder_predictor(checkpoint_path=checkpoint, device=device)
    scorer = VJEPAPredictLossScorer(
        config, models["context_encoder"], models["target_encoder"],
        models["predictor"], degraded=models["degraded"],
    )

    # Read full video and determine number of 1FPS frames within limit
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(video_path))
    source_fps = float(vr.get_avg_fps() or config.fps)
    total_duration = len(vr) / source_fps

    num_stream_frames = min(max_stream_frames, int(total_duration * config.fps))
    num_stream_frames = max(num_stream_frames, 16)  # at least one window

    # Window params
    window_size = config.window_frames  # 16
    stride = config.temporal_stride  # 2 frames = 2 seconds

    # Accumulate per-tubelet losses
    # Each 2-second tubelet can appear in up to 8 windows
    num_tubelets = (num_stream_frames + 1) // 2  # ceil(num_frames / 2)
    global_loss_sum = torch.zeros(num_tubelets, 16, 16)
    global_loss_count = torch.zeros(num_tubelets, 16, 16)

    t0 = time.perf_counter()
    num_windows = 0

    for win_start_frame in range(0, num_stream_frames - window_size + 1, stride):
        win_end_frame = win_start_frame + window_size
        win_start_s = win_start_frame / config.fps
        win_end_s = win_end_frame / config.fps

        # Sample this window's frames at 256px for V-JEPA
        try:
            ws = sample_video_1fps_decord(
                video_path, num_frames=window_size, size=config.jepa_size,
                target_fps=config.fps, start_time_s=win_start_s,
            )
        except Exception:
            continue

        frames_256 = ws.vjepa_tensor().to(device)
        score = scorer.score_window(frames_256)
        loss_map = score.loss_map.cpu()  # [1, 8, 16, 16] or [8, 16, 16]

        if loss_map.dim() == 4:
            loss_map = loss_map.squeeze(0)
        # loss_map: [8, 16, 16] for this window's 8 tubelets

        # Map window tubelets to global tubelets
        # Each window of 16 frames = 8 tubelets (2 frames each)
        # Global tubelet index = (win_start_frame // 2) + local_tubelet_idx
        base_tubelet = win_start_frame // 2
        for local_t in range(loss_map.shape[0]):
            global_t = base_tubelet + local_t
            if global_t < num_tubelets:
                global_loss_sum[global_t] += loss_map[local_t]
                global_loss_count[global_t] += 1
        num_windows += 1

    t1 = time.perf_counter()
    cache_build_latency = t1 - t0

    # Average overlapping windows
    global_loss_count = global_loss_count.clamp(min=1)
    global_scores = global_loss_sum / global_loss_count

    # Cache each window's scores
    for win_start_frame in range(0, num_stream_frames - window_size + 1, stride):
        win_id = win_start_frame // stride
        win_start_s = win_start_frame / config.fps
        win_end_s = (win_start_frame + window_size) / config.fps
        cache_key = f"{sample_id}:{win_id}"
        if not cache.has(cache_key):
            # Re-sample and score for caching
            try:
                ws = sample_video_1fps_decord(
                    video_path, num_frames=window_size, size=config.jepa_size,
                    target_fps=config.fps, start_time_s=win_start_s,
                )
            except Exception:
                continue
            frames_256 = ws.vjepa_tensor().to(device)
            score = scorer.score_window(frames_256)
            lm = score.loss_map.cpu()
            if lm.dim() == 4:
                lm = lm.squeeze(0)
            km = score.keep_mask.cpu()
            if km.dim() == 4:
                km = km.squeeze(0)
            ki = score.keep_indices[0].cpu()
            cache.put(cache_key, lm, km, ki)
    cache.flush()

    return global_scores, cache_build_latency, num_windows


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
    parser.add_argument("--window_policy", default="anchor_end16",
                        choices=["first16", "anchor_end16", "full_stream"])
    parser.add_argument("--max_stream_frames", type=int, default=64)
    parser.add_argument("--allow_dynamic_grid", action="store_true")
    args = parser.parse_args()

    config = PredictMemConfig()
    config.keep_ratio = args.keep_ratio
    config.fps = args.fps
    config.window_frames = args.num_frames
    config.qwen_size = args.qwen_size
    config.__post_init__()

    window_size_s = config.window_frames / config.fps  # 16s

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
    print(f"Running {len(samples)} samples, method={args.method}, "
          f"window_policy={args.window_policy}, max_new_tokens={args.max_new_tokens}")

    # Cache for predictmem
    cache = None
    if args.method == "predictmem":
        cache = ScoreCache(args.cache_path)

    results = []
    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        task = sample["task"]
        print(f"\n[{i+1}/{len(samples)}] sample_id={sid} task={task} method={args.method}")

        seed = args.seed + i
        try:
            # Build OVO prompt
            prompt_text, task_type, gt_letter = build_ovo_prompt(sample)
            print(f"  Prompt task={task_type}, gt_letter='{gt_letter}'")

            # Get window bounds
            window_start_s, window_end_s, video_duration_s = get_window_bounds(
                sample["video_path"], args.window_policy, window_size_s,
                args.fps, args.max_stream_frames,
            )
            print(f"  Window: [{window_start_s:.1f}s, {window_end_s:.1f}s] "
                  f"/ {video_duration_s:.1f}s (policy={args.window_policy})")

            cache_build_latency = 0.0
            num_windows = 1
            global_scores = None

            if args.window_policy == "full_stream" and args.method == "predictmem":
                # P4: Full-stream V-JEPA scoring with sliding windows
                print(f"  Computing full-stream V-JEPA scores (max_frames={args.max_stream_frames})...")
                global_scores, cache_build_latency, num_windows = run_full_stream_scoring(
                    sample["video_path"], config, cache, sid, args.device,
                    args.max_stream_frames,
                )
                print(f"  Full-stream: {num_windows} windows, "
                      f"global_scores={list(global_scores.shape)}, "
                      f"cache_lat={cache_build_latency:.2f}s")

            # Sample video frames for Qwen
            video_sample, actual_start, actual_end = sample_window(
                sample["video_path"], window_start_s, window_end_s, config,
            )
            frames_np = video_sample.frames_uint8
            print(f"  Frames: {frames_np.shape[0]} frames, "
                  f"indices=[{video_sample.source_indices[0]}...{video_sample.source_indices[-1]}]")

            # Build messages: video + OVO prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames_np, "fps": config.fps},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text], videos=[frames_np],
                video_metadata=[video_sample.qwen_metadata()],
                do_sample_frames=False, do_resize=False, fps=config.fps,
                return_tensors="pt",
            )
            inputs = move_inputs_to_device(inputs, args.device)

            video_grid_thw = inputs["video_grid_thw"].detach().cpu()
            num_qwen_video_tokens = mapper.compute_num_video_tokens(video_grid_thw)
            print(f"  video_grid_thw: {video_grid_thw.tolist()} -> {num_qwen_video_tokens} Qwen video tokens")

            # Generate keep indices
            if args.window_policy == "full_stream" and args.method == "predictmem" and global_scores is not None:
                # Use global aggregated scores for full-stream
                keep = mapper.map_scores_to_qwen_keep_indices(
                    video_grid_thw=video_grid_thw,
                    loss_map=global_scores,
                    keep_mask=None,
                    keep_ratio=config.keep_ratio,
                )
            else:
                keep, _ = generate_keep_indices_for_method(
                    method=args.method, config=config, sample_id=sid,
                    cache=cache, seed=seed, video_grid_thw=video_grid_thw,
                    mapper=mapper,
                )

            kept = keep[0].shape[0]
            keep_ratio_actual = kept / num_qwen_video_tokens
            print(f"  Keep: {kept}/{num_qwen_video_tokens} ({keep_ratio_actual:.1%})")

            # Generate — single call per sample (P3)
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

            if args.device == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                peak_memory = 0.0

            # Parse response
            num_options = len(sample.get("options", []))
            parsed_response = parse_response(raw_response, task, num_options)
            score = score_parsed_response(parsed_response, gt_letter, task)

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
                "window_policy": args.window_policy,
                "window_start_s": round(window_start_s, 2),
                "window_end_s": round(window_end_s, 2),
                "video_duration_s": round(video_duration_s, 2),
                "source_indices": video_sample.source_indices,
                "fps": config.fps,
                "num_frames": frames_np.shape[0],
                "qwen_size": config.qwen_size,
                "video_grid_thw": video_grid_thw.tolist(),
                "keep_ratio_target": config.keep_ratio,
                "keep_ratio_actual": round(keep_ratio_actual, 3),
                "original_video_tokens": num_qwen_video_tokens,
                "kept_video_tokens": kept,
                "cache_build_latency_s": round(cache_build_latency, 3),
                "score_latency_s": 0.0,
                "vision_latency_s": 0.0,
                "prefill_latency_s": round(total_latency, 4),
                "decode_latency_s": 0.0,
                "total_latency_s": round(total_latency, 4),
                "peak_memory_mb": round(peak_memory, 1),
                # Window-level debug (P3)
                "num_sliding_windows": num_windows,
            }

            results.append(entry)
            print(f"  Raw: '{raw_response[:100]}'")
            print(f"  Parsed: '{parsed_response}' | GT: '{gt_letter}' | Score: {score}")
            print(f"  Latency: {total_latency:.3f}s, Peak mem: {peak_memory:.0f}MB")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Build minimal error entry with OVO fields
            prompt_text, task_type, gt_letter = build_ovo_prompt(sample)
            results.append({
                "sample_id": sid,
                "task": task,
                "video": sample["video_path"],
                "question": sample.get("question", ""),
                "answer_text": sample.get("answer", ""),
                "ground_truth_letter": gt_letter,
                "options": sample.get("options", []),
                "response": f"[error: {str(e)[:100]}]",
                "raw_response": "",
                "score": 0,
                "method": args.method,
                "window_policy": args.window_policy,
                "window_start_s": 0.0,
                "window_end_s": 0.0,
                "video_duration_s": 0.0,
                "source_indices": [],
                "fps": config.fps,
                "num_frames": 0,
                "qwen_size": config.qwen_size,
                "video_grid_thw": [],
                "keep_ratio_target": config.keep_ratio,
                "keep_ratio_actual": 0.0,
                "original_video_tokens": 0,
                "kept_video_tokens": 0,
                "cache_build_latency_s": 0.0,
                "score_latency_s": 0.0,
                "vision_latency_s": 0.0,
                "prefill_latency_s": 0.0,
                "decode_latency_s": 0.0,
                "total_latency_s": 0.0,
                "peak_memory_mb": 0.0,
                "num_sliding_windows": 0,
            })

    # Write output — one line per sample (P3)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults: {len(results)} entries -> {args.output}")
    print(f"Window policy: {args.window_policy}, per-sample answers: {len(results)}")


if __name__ == "__main__":
    main()
