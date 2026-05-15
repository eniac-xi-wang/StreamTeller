#!/usr/bin/env python3
"""Single-sample smoke test for PredictMem with real Qwen3-VL.

Monkey-patches Qwen3VLModel.forward to insert PredictMem pruning between
position_ids computation and language_model call. Also prunes
visual_pos_masks and deepstack_visual_embeds.

Usage:
    python scripts/smoke_qwen_predictmem.py \
      --model_path /data/model_weights_public/Qwen/Qwen3-VL-2B-Instruct \
      --video /data/qinian_workspace/OVO-Bench/chunked_videos/0.mp4 \
      --question "Who did I communicate to when chopping egg plants?" \
      --cache_path results/predictmem_scores.jsonl \
      --sample_id 0 \
      --method predictmem \
      --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

_models_dir = Path(__file__).parent.parent / "models"
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from predictmem.config import PredictMemConfig
from predictmem.token_mapping import TokenMapper
from predictmem.token_pruner import TokenPruner
from predictmem.cache import ScoreCache


class PredictMemPatcher:
    """Patches Qwen3VLModel.forward to insert PredictMem token pruning."""

    def __init__(self, model, config: PredictMemConfig, keep_indices: list):
        self.model = model
        self.config = config
        self.keep_indices = keep_indices
        self.video_token_id = model.config.video_token_id
        self.vision_start_token_id = model.config.vision_start_token_id
        self.vision_end_token_id = model.config.vision_end_token_id

        self.pruner = TokenPruner(
            config=config,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
        )

        self._original_forward = None

    def patch(self):
        self._original_forward = self.model.model.forward
        patcher = self

        def patched_forward(
            input_ids=None, attention_mask=None, position_ids=None,
            past_key_values=None, inputs_embeds=None,
            pixel_values=None, pixel_values_videos=None,
            image_grid_thw=None, video_grid_thw=None,
            mm_token_type_ids=None, **kwargs,
        ):
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = patcher.model.model.get_input_embeddings()(input_ids)

            image_mask = None
            video_mask = None
            deepstack_image_embeds = None
            deepstack_video_embeds = None

            if pixel_values is not None:
                image_outputs = patcher.model.model.get_image_features(
                    pixel_values, image_grid_thw, return_dict=True)
                image_embeds_list = image_outputs.pooler_output
                deepstack_image_embeds = image_outputs.deepstack_features
                image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = patcher.model.model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_outputs = patcher.model.model.get_video_features(
                    pixel_values_videos, video_grid_thw, return_dict=True)
                video_embeds_list = video_outputs.pooler_output
                deepstack_video_embeds = video_outputs.deepstack_features
                video_embeds = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = patcher.model.model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Build visual_pos_masks and deepstack_visual_embeds
            visual_pos_masks = None
            deepstack_visual_embeds = None
            if image_mask is not None and video_mask is not None:
                image_mask = image_mask[..., 0]
                video_mask = video_mask[..., 0]
                visual_pos_masks = image_mask | video_mask
                deepstack_visual_embeds = []
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
            elif image_mask is not None:
                image_mask = image_mask[..., 0]
                visual_pos_masks = image_mask
                deepstack_visual_embeds = deepstack_image_embeds
            elif video_mask is not None:
                video_mask = video_mask[..., 0]
                visual_pos_masks = video_mask
                deepstack_visual_embeds = deepstack_video_embeds

            if position_ids is None:
                position_ids = patcher.model.model.compute_3d_position_ids(
                    input_ids=input_ids, image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw, inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask, past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )

            # ---- PredictMem pruning ----
            should_prune = (
                patcher.keep_indices is not None
                and pixel_values_videos is not None
                and inputs_embeds.shape[1] > 1  # skip decode
            )
            if should_prune:
                ki_device = [ki.to(inputs_embeds.device) for ki in patcher.keep_indices]
                sequence_keep_masks = patcher.pruner.build_sequence_keep_masks(
                    input_ids=input_ids,
                    video_keep_indices=ki_device,
                    attention_mask=attention_mask,
                )
                if visual_pos_masks is not None:
                    deepstack_visual_embeds = TokenPruner.prune_deepstack_visual_embeds(
                        deepstack_visual_embeds,
                        visual_pos_masks,
                        sequence_keep_masks,
                    )
                    visual_pos_masks = TokenPruner.prune_token_mask(visual_pos_masks, sequence_keep_masks)
                inputs_embeds, position_ids, new_attention_mask = patcher.pruner.prune(
                    input_ids=input_ids, inputs_embeds=inputs_embeds,
                    position_ids=position_ids, attention_mask=attention_mask,
                    video_keep_indices=ki_device,
                )
                attention_mask = new_attention_mask

            outputs = patcher.model.model.language_model(
                input_ids=None, position_ids=position_ids,
                attention_mask=attention_mask, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds, **kwargs,
            )

            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
            return Qwen3VLModelOutputWithPast(**outputs, rope_deltas=patcher.model.model.rope_deltas)

        self.model.model.forward = patched_forward

    def unpatch(self):
        if self._original_forward is not None:
            self.model.model.forward = self._original_forward
            self._original_forward = None


# ─── Video sampling ───────────────────────────────────────────────────────────

def sample_frames_qwen(video_path: str, num_frames: int = 16, size: int = 512) -> np.ndarray | None:
    """Sample frames for Qwen input at 512x512. Returns [T, C, H, W] numpy float32 in [0,1]."""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(video_path))
        total = len(vr)
        if total < num_frames:
            indices = list(range(total)) + [total - 1] * (num_frames - total)
        else:
            indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        frames = vr.get_batch(indices)  # [T, H, W, C] as torch tensor or NDArray
        if hasattr(frames, 'asnumpy'):
            frames = torch.from_numpy(frames.asnumpy()).float() / 255.0
        elif isinstance(frames, torch.Tensor):
            frames = frames.float() / 255.0
        else:
            frames = torch.from_numpy(np.array(frames)).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = torch.nn.functional.interpolate(
            frames, size=(size, size), mode="bilinear", align_corners=False,
        )
        return frames.numpy()  # [T, C, 512, 512]
    except Exception as e:
        print(f"  decord error: {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--cache_path", default="results/predictmem_scores.jsonl")
    parser.add_argument("--sample_id", default="0")
    parser.add_argument("--method", default="predictmem", choices=["baseline", "predictmem"])
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    config = PredictMemConfig()
    config.keep_ratio = args.keep_ratio

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading Qwen3-VL from {args.model_path}...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.1f}B params, video_token_id={model.config.video_token_id}")

    # ── Sample frames ───────────────────────────────────────────────────────
    print(f"Sampling frames from {args.video}...")
    frames_np = sample_frames_qwen(args.video)
    if frames_np is None:
        print("ERROR: failed to read video")
        sys.exit(1)
    print(f"  Frames shape: {frames_np.shape}")

    # Keep indices are computed after processing, because Qwen token count comes
    # from the actual video_grid_thw produced by the processor.
    keep_indices = None
    num_qwen_video_tokens = None

    # ── Build messages ──────────────────────────────────────────────────────
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames_np, "fps": 1.0},
                {"type": "text", "text": args.question},
            ],
        }
    ]

    # ── Process inputs ──────────────────────────────────────────────────────
    print("Processing inputs...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=[frames_np], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    print(f"  input_ids shape: {inputs['input_ids'].shape}")
    if "pixel_values_videos" in inputs:
        pvv = inputs["pixel_values_videos"]
        print(f"  pixel_values_videos shape: {pvv.shape if isinstance(pvv, torch.Tensor) else 'N/A'}")
    if "video_grid_thw" not in inputs:
        print("ERROR: processor did not return video_grid_thw")
        sys.exit(1)

    mapper = TokenMapper(config)
    video_grid_thw = inputs["video_grid_thw"].detach().cpu()
    num_qwen_video_tokens = mapper.compute_num_video_tokens(video_grid_thw)
    print(f"  video_grid_thw: {video_grid_thw.tolist()} -> {num_qwen_video_tokens} Qwen video tokens")

    if args.method == "predictmem":
        cache = ScoreCache(args.cache_path)
        if not cache.has(args.sample_id):
            print(f"ERROR: sample {args.sample_id} not in cache")
            sys.exit(1)
        loss_map = cache.get_loss_map(args.sample_id)
        keep_mask = None if loss_map is not None else cache.get_keep_mask(args.sample_id)
        keep_indices = mapper.map_scores_to_qwen_keep_indices(
            video_grid_thw=video_grid_thw,
            loss_map=loss_map,
            keep_mask=keep_mask,
            keep_ratio=args.keep_ratio,
        )
        n_kept = keep_indices[0].shape[0]
        print(f"  PredictMem: {n_kept}/{num_qwen_video_tokens} tokens kept ({n_kept/num_qwen_video_tokens:.1%})")
    else:
        keep_indices = [torch.arange(num_qwen_video_tokens)]
        print(f"  Baseline: {num_qwen_video_tokens}/{num_qwen_video_tokens} tokens kept")

    # ── Generate ────────────────────────────────────────────────────────────
    print(f"\nGenerating ({args.method})...")
    t0 = time.perf_counter()

    if args.method == "predictmem":
        patcher = PredictMemPatcher(model, config, keep_indices)
        patcher.patch()
        print("  Patched forward for PredictMem pruning")

    with torch.no_grad():
        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        finally:
            if args.method == "predictmem":
                patcher.unpatch()

    t1 = time.perf_counter()
    total_latency = t1 - t0

    # Decode output
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0, input_len:]
    output_text = processor.decode(output_ids, skip_special_tokens=True)

    # ── Memory ──────────────────────────────────────────────────────────────
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_memory = 0.0

    # ── Results ─────────────────────────────────────────────────────────────
    kept = keep_indices[0].shape[0] if keep_indices else num_qwen_video_tokens
    result = {
        "sample_id": args.sample_id,
        "video": args.video,
        "question": args.question,
        "prediction": output_text,
        "method": args.method,
        "keep_ratio_target": args.keep_ratio,
        "keep_ratio_actual": kept / num_qwen_video_tokens,
        "original_video_tokens": num_qwen_video_tokens,
        "kept_video_tokens": kept,
        "score_latency_s": 0.0,
        "vision_latency_s": 0.0,
        "prefill_latency_s": round(total_latency, 4),
        "decode_latency_s": 0.0,
        "total_latency_s": round(total_latency, 4),
        "peak_memory_mb": round(peak_memory, 1),
    }

    print(f"\n{'='*60}")
    print(f"Method:      {args.method}")
    print(f"Prediction:  {output_text[:200]}")
    print(f"Kept tokens: {kept}/{num_qwen_video_tokens} ({kept/num_qwen_video_tokens:.1%})")
    print(f"Total latency: {total_latency:.3f}s")
    print(f"Peak mem:      {peak_memory:.0f}MB")
    print(f"{'='*60}")

    # Save
    output_path = f"results/smoke_{args.method}_{args.sample_id}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
