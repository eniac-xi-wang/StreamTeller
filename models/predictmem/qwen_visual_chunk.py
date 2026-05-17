"""Per-chunk Qwen visual processing for streaming compact memory.

Instead of calling the Qwen visual tower on the entire video at once, this
module processes one tubelet (or small chunk) at a time and returns the
visual embeddings for only that chunk, ready to be pruned and appended to
the compact memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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
        qwen_frames_uint8,       # [n, 512, 512, 3] uint8 numpy, n=1 or 2
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
        grid_h = self.config.qwen_grid_h  # 32
        grid_w = self.config.qwen_grid_w  # 32
        merge = self.config.qwen_merge_size  # 2

        if not hasattr(self._visual, "spatial_merge_size"):
            raise RuntimeError("Qwen visual tower missing spatial_merge_size attribute")

        # Build per-frame grid_thw for this tubelet: (n, 1, 32, 32) → (n, 32*32)
        chunk_grid_thw = torch.tensor(
            [[1, grid_h, grid_w] for _ in range(n_frames)],
            dtype=torch.long, device=device,
        )

        # Build pixel_values: [n_frames * 1 * 3 * 512 * 512]
        frames_tensor = torch.from_numpy(qwen_frames_uint8).to(device)  # [n, 512, 512, 3]
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [n, 3, 512, 512]
        # Resize to Qwen expected size if needed
        expected_size = getattr(self._visual.config, "image_size", 512)
        if frames_tensor.shape[-2:] != (expected_size, expected_size):
            frames_tensor = F.interpolate(
                frames_tensor.float(), size=(expected_size, expected_size),
                mode="bilinear", align_corners=False
            )
        pixel_values = frames_tensor.to(self._visual.dtype)  # [n, 3, H, W]

        # Run visual tower on this chunk
        vision_output = self._visual(pixel_values, grid_thw=chunk_grid_thw, return_dict=True)
        embeds = vision_output.pooler_output  # list of tensors, one per frame

        # Build 3D position_ids for this chunk
        # Each frame at temporal position (tubelet_id * 2 + frame_offset)
        pos_ids_chunks = []
        for f_idx in range(n_frames):
            global_frame = tubelet_id * 2 + f_idx
            # M-RoPE: [3, H*W/merge^2] for one frame
            # t dimension: 1 temporal patch → pos=global_frame // 2
            t_pos = global_frame // self.config.temporal_stride  # tubelet-level temporal id
            hh = grid_h // merge  # 16
            ww = grid_w // merge  # 16
            num_tokens = hh * ww  # 256

            # 3D position: [temporal, height, width]
            t_ids = torch.full((num_tokens,), t_pos, dtype=torch.long, device=device)
            h_ids = torch.arange(hh, device=device).view(-1, 1).expand(hh, ww).reshape(-1)
            w_ids = torch.arange(ww, device=device).view(1, -1).expand(hh, ww).reshape(-1)
            frame_pos = torch.stack([t_ids, h_ids, w_ids], dim=0)  # [3, 256]
            pos_ids_chunks.append(frame_pos)

        chunk_position_ids = torch.cat(pos_ids_chunks, dim=1)  # [3, n * 256]

        # Concatenate frame embeddings
        if isinstance(embeds, list):
            chunk_embeds = torch.cat([e.to(device) for e in embeds], dim=0)
        else:
            chunk_embeds = embeds.to(device)

        return chunk_embeds, chunk_position_ids

    @torch.no_grad()
    def process_tubelet_via_processor(
        self,
        qwen_frames_uint8,       # [n, 512, 512, 3] uint8 numpy
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
        video_grid_thw = torch.tensor(
            [[n_frames, self.config.qwen_grid_h, self.config.qwen_grid_w]],
            dtype=torch.long, device=device,
        )

        # Build a minimal input: just video placeholders
        num_video_tokens = (n_frames * self.config.qwen_grid_h * self.config.qwen_grid_w) \
            // (self.config.qwen_merge_size ** 2)
        video_token_id = self.model.config.video_token_id
        vision_start_id = self.model.config.vision_start_token_id
        vision_end_id = self.model.config.vision_end_token_id

        # Minimal input_ids: <vision_start> + video_tokens + <vision_end>
        input_ids = torch.tensor(
            [[vision_start_id] + [video_token_id] * num_video_tokens + [vision_end_id]],
            dtype=torch.long, device=device,
        )

        # Process
        frames_tensor = torch.from_numpy(qwen_frames_uint8).to(device).permute(0, 3, 1, 2)
        inputs = self.processor(
            text=[""],
            videos=[frames_tensor.cpu().numpy()],
            video_metadata=[{"total_num_frames": n_frames, "fps": 1.0,
                            "duration": n_frames, "frames_indices": list(range(n_frames)),
                            "height": self.config.qwen_size, "width": self.config.qwen_size,
                            "video_backend": "decord"}],
            do_sample_frames=False, do_resize=False, return_tensors="pt",
        )
        # We only need the pixel_values_videos and video_grid_thw
        pixel_values_videos = inputs["pixel_values_videos"].to(device)
        video_grid_thw = inputs["video_grid_thw"].to(device)

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
