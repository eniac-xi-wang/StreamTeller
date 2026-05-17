"""GPU memory trace instrumentation for PredictMem evaluation.

Records GPU memory (allocated, reserved, peak, free, NVML, RSS) at key
checkpoints during video sampling → visual encoding → pruning → generation.

Usage as context manager:
    with MemoryTracer(enabled=True, log_path="mem.jsonl") as tracer:
        tracer.checkpoint("before_video_sampling", num_frames=N)
        ...
        tracer.checkpoint("after_generate", kept_video_tokens=K)

Usage as explicit recorder:
    tracer = MemoryTracer(enabled=True)
    tracer.checkpoint("sample_begin")
    ...
    tracer.write_summary()
"""

from __future__ import annotations

import json
import os
import time
from contextlib import AbstractContextManager
from typing import Any

import torch


def _rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


def _nvml_mb(device: int = 0) -> float:
    """NVML used memory in MB, or -1 if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        return -1.0


def snapshot(device: int = 0) -> dict[str, float]:
    """Take a single memory snapshot on the given CUDA device."""
    free_total, total = (-1.0, -1.0)
    try:
        free_total, total = torch.cuda.mem_get_info(device)
    except Exception:
        pass
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 1),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 1),
        "free_mb": round(free_total / (1024 * 1024), 1),
        "total_mb": round(total / (1024 * 1024), 1) if total > 0 else -1.0,
        "nvml_mb": round(_nvml_mb(device), 1),
        "rss_mb": round(_rss_mb(), 1),
    }


class MemoryTracer(AbstractContextManager):
    """Records GPU memory snapshots at named checkpoints for one sample.

    Each checkpoint stores the snapshot plus optional contextual keys
    (num_frames, video_grid_thw, full_video_tokens, kept_video_tokens, etc.).
    """

    def __init__(self, enabled: bool = True, device: int = 0, log_path: str | None = None):
        self.enabled = enabled
        self.device = device
        self.log_path = log_path
        self.records: list[dict[str, Any]] = []
        self._sample_start = 0.0

    def __enter__(self):
        self._sample_start = time.perf_counter()
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.checkpoint("sample_begin")
        return self

    def __exit__(self, *args):
        self.checkpoint("sample_end")
        if self.log_path:
            self.write_log()

    def checkpoint(
        self,
        name: str,
        num_frames: int = 0,
        video_grid_thw: list | None = None,
        full_video_tokens: int = 0,
        kept_video_tokens: int = 0,
        **extra,
    ):
        if not self.enabled:
            return
        rec = {
            "ts": round(time.perf_counter() - self._sample_start, 4),
            "checkpoint": name,
            "memory": snapshot(self.device),
        }
        if num_frames:
            rec["num_frames"] = num_frames
        if video_grid_thw is not None:
            rec["video_grid_thw"] = video_grid_thw
        if full_video_tokens:
            rec["full_video_tokens"] = full_video_tokens
        if kept_video_tokens:
            rec["kept_video_tokens"] = kept_video_tokens
        if extra:
            rec["extra"] = extra
        self.records.append(rec)

    def peak_stage(self) -> dict[str, Any] | None:
        """Return the checkpoint record with the highest allocated_mb."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r["memory"]["allocated_mb"])

    def summary(self) -> dict[str, Any]:
        peak = self.peak_stage()
        return {
            "num_checkpoints": len(self.records),
            "peak_allocated_mb": peak["memory"]["allocated_mb"] if peak else -1,
            "peak_checkpoint": peak["checkpoint"] if peak else "unknown",
            "peak_reserved_mb": peak["memory"]["reserved_mb"] if peak else -1,
            "peak_nvml_mb": peak["memory"]["nvml_mb"] if peak else -1,
            "baseline_allocated_mb": self.records[0]["memory"]["allocated_mb"] if self.records else -1,
        }

    def write_log(self):
        path = self.log_path
        if path is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")

    def write_summary(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
