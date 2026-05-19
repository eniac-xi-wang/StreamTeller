"""Per-chunk Qwen visual processing for streaming compact memory.

Instead of calling the Qwen visual tower on the entire video at once, this
module processes one tubelet (or small chunk) at a time and returns the
visual embeddings for only that chunk, ready to be pruned and appended to
the compact memory.
"""

from __future__ import annotations

import torch
from typing import Tuple

from .config import PredictMemConfig


class QwenVisualChunkProcessor:
    """Process a small temporal chunk of video through Qwen's visual tower.

    Each call processes exactly one tubelet (2 frames) or a small chunk of
    tubelets, returning the visual embeddings for those frames only.
    """

    def __init__(self, model, processor, config: PredictMemConfig):
        self.model = model
        self.processor = processor
        self.config = config
        self._qwen_model = getattr(model, "model", model)
        self._visual = self._qwen_model.visual

    @torch.no_grad()
    def process_tubelet(
        self,
        qwen_frames_uint8,       # [n, H, W, 3] uint8 numpy, n=1 or 2
        tubelet_id: int,
        device: torch.device | str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen visual tower on a single tubelet's frames.

        Args:
            qwen_frames_uint8: uint8 numpy array [n, H, W, 3]
            tubelet_id: which tubelet this is (for position tracking)
            device: target device

        Returns:
            (embeddings, position_ids) where:
              embeddings: [n_tokens, hidden_dim] — visual embeddings for this chunk
              position_ids: [3, n_tokens] — 3D M-RoPE position ids
        """
        n_frames = qwen_frames_uint8.shape[0]
        height, width = int(qwen_frames_uint8.shape[1]), int(qwen_frames_uint8.shape[2])
        patch = self.config.patch_size
        merge = self.config.qwen_merge_size
        if height % (patch * merge) != 0 or width % (patch * merge) != 0:
            raise ValueError(
                "Qwen chunk size must be divisible by patch_size * merge_size, "
                f"got {(height, width)} with patch={patch}, merge={merge}"
            )

        if not hasattr(self.processor, "video_processor"):
            raise RuntimeError("Qwen processor missing video_processor")

        video_inputs = self.processor.video_processor(
            videos=[qwen_frames_uint8],
            do_sample_frames=False,
            do_resize=False,
            return_tensors="pt",
        )
        pixel_values_videos = video_inputs["pixel_values_videos"].to(device)
        video_grid_thw = video_inputs["video_grid_thw"].to(device)

        # The video processor pads a single trailing frame to temporal_patch_size=2,
        # so every sampler item still produces one tubelet grid.
        if int(video_grid_thw[0, 0].item()) != 1:
            raise ValueError(f"Expected one Qwen tubelet per compact chunk, got video_grid_thw={video_grid_thw.tolist()}")

        vision_output = self._qwen_model.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True
        )
        embeds = vision_output.pooler_output
        chunk_embeds = torch.cat([e.to(device) for e in embeds], dim=0) if isinstance(embeds, (list, tuple)) else embeds.to(device)

        chunk_position_ids = self._qwen_model.get_vision_position_ids(
            start_position=0,
            grid_thw=video_grid_thw[0],
            temp_merge_size=1,
            spatial_merge_size=merge,
            device=device,
        )
        chunk_position_ids[0] += tubelet_id

        expected_tokens = (int(video_grid_thw[0, 1].item()) // merge) * (int(video_grid_thw[0, 2].item()) // merge)
        if chunk_embeds.shape[0] != expected_tokens or chunk_position_ids.shape[1] != expected_tokens:
            raise ValueError(
                "Qwen compact chunk token count mismatch: "
                f"embeds={tuple(chunk_embeds.shape)}, positions={tuple(chunk_position_ids.shape)}, "
                f"expected={expected_tokens}, n_frames={n_frames}, grid={video_grid_thw.tolist()}"
            )

        return chunk_embeds, chunk_position_ids

    @torch.no_grad()
    def process_tubelet_via_processor(
        self,
        qwen_frames_uint8,       # [n, H, W, 3] uint8 numpy
        tubelet_id: int,
        global_frame_start: int,
        device: torch.device | str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alternative: use the full processor pipeline for a tubelet.

        This is more accurate for position_id computation (uses M-RoPE from
        Qwen's get_rope_index) but requires dummy text to form a valid input.

        Returns:
            (embeddings, position_ids, video_grid_thw)
        """
        n_frames = qwen_frames_uint8.shape[0]
        height, width = int(qwen_frames_uint8.shape[1]), int(qwen_frames_uint8.shape[2])
        video_token_id = self.model.config.video_token_id
        vision_start_id = self.model.config.vision_start_token_id
        vision_end_id = self.model.config.vision_end_token_id

        # Process
        inputs = self.processor(
            text=[""],
            videos=[qwen_frames_uint8],
            video_metadata=[{"total_num_frames": n_frames, "fps": 1.0,
                            "duration": n_frames, "frames_indices": list(range(n_frames)),
                            "height": height, "width": width,
                            "video_backend": "decord"}],
            do_sample_frames=False, do_resize=False, return_tensors="pt",
        )
        # We only need the pixel_values_videos and video_grid_thw
        pixel_values_videos = inputs["pixel_values_videos"].to(device)
        video_grid_thw = inputs["video_grid_thw"].to(device)
        num_video_tokens = int(video_grid_thw[0].prod().item()) // (self.config.qwen_merge_size ** 2)

        # Minimal input_ids: <vision_start> + video_tokens + <vision_end>
        input_ids = torch.tensor(
            [[vision_start_id] + [video_token_id] * num_video_tokens + [vision_end_id]],
            dtype=torch.long, device=device,
        )

        vision_output = self._qwen_model.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True
        )
        embeds = vision_output.pooler_output
        if isinstance(embeds, list):
            embeds = torch.cat([e.to(device) for e in embeds], dim=0)

        # Compute position_ids via model's own method
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        mm_token_type_ids = inputs.get("mm_token_type_ids", None)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(device)
        position_ids = self._qwen_model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            attention_mask=None,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )

        # Extract only video token positions (excluding vision_start/end)
        is_video = (input_ids[0] == video_token_id)
        vid_pos = position_ids[:, 0, is_video]  # [3, n_tokens]

        return embeds, vid_pos, video_grid_thw
