import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from evaluate.common.qwen35_predictmem import build_video_inputs_for_eval
from evaluate.ovobench.run_fluxmem_style_hld import fluxmem_style_sample
from models.predictmem.streaming_sampler import StreamingVideoSampler
from models.qwen3_5.processing_qwen3_vl import Qwen3VLProcessor


class FakeVideoReader:
    """Ten seconds of 30 FPS video, with a 32x32 image at every index."""

    def __len__(self):
        return 300

    def get_avg_fps(self):
        return 30.0

    def __getitem__(self, index):
        return torch.zeros(32, 32, 3, dtype=torch.uint8)

    def get_batch(self, indices):
        return torch.stack([self[index] for index in indices])


class VideoTimestampTest(unittest.TestCase):
    def setUp(self):
        decoder = SimpleNamespace(VideoReader=lambda path: FakeVideoReader(), bridge=Mock())
        modules = patch.dict(sys.modules, decord=decoder)
        modules.start()
        self.addCleanup(modules.stop)

    def check_timestamps(self, metadata, indices, seconds):
        self.assertEqual(metadata["frames_indices"], indices)
        self.assertEqual(metadata["total_num_frames"], 300)
        self.assertEqual(metadata["duration"], 10.0)
        actual = Qwen3VLProcessor._calculate_timestamps(
            None, metadata["frames_indices"], metadata["fps"]
        )
        self.assertEqual(len(actual), len(seconds))
        for got, expected in zip(actual, seconds):
            self.assertAlmostEqual(got, expected)
        self.assertEqual(metadata["frames_indices"], indices)

    def test_shared_sampler_and_compact(self):
        cases = [
            ("full", 0, 3, 0, [0, 30, 60], [0.5, 2.0]),
            ("full", 3, 6, 0, [90, 120, 150], [3.5, 5.0]),
            ("recent", 0, 10, 2, [240, 270], [8.5]),
            ("uniform", 0, 10, 4, [38, 112, 188, 262], [2.5, 7.5]),
        ]
        for mode, start, end, budget, indices, seconds in cases:
            with self.subTest(mode=mode, start=start):
                kwargs = dict(fps=1.0, stream_mode=mode, frame_budget=budget, start_time=start,
                              end_time=end, qwen_size=32, jepa_size=16)
                _, _, metadata, _ = build_video_inputs_for_eval("video.mp4", **kwargs)
                self.check_timestamps(metadata, indices, seconds)

                sampler = StreamingVideoSampler("video.mp4", **kwargs)
                self.check_timestamps(sampler.metadata, indices, seconds)
                actual = [sum(group["times_s"]) / len(group["times_s"]) for group in sampler]
                self.assertEqual(actual, seconds)

    def test_fluxmem_style_baseline(self):
        _, metadata, _ = fluxmem_style_sample("video.mp4", min_frames=4, max_frames=4)
        self.check_timestamps(metadata, [0, 100, 199, 299], [50 / 30, 249 / 30])


if __name__ == "__main__":
    unittest.main()
