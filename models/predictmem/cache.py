"""Score cache for offline PredictMem evaluation.

Reads/writes V-JEPA prediction loss scores as JSONL, avoiding recomputation
across multiple evaluation runs.
"""

import json
import os
from pathlib import Path

import torch


class ScoreCache:
    """File-backed cache for V-JEPA prediction loss scores.

    Stores scores as JSONL where each line is:
        {"sample_id": str, "keep_mask": [int, ...], "loss_map": [float, ...],
         "grid_shape": [8, 16, 16]}
    """

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self._data: dict[str, dict] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._data[entry["sample_id"]] = entry

    def has(self, sample_id: str) -> bool:
        self._ensure_loaded()
        return sample_id in self._data

    def get(self, sample_id: str) -> dict | None:
        self._ensure_loaded()
        return self._data.get(sample_id)

    def get_keep_mask(self, sample_id: str) -> torch.Tensor | None:
        entry = self.get(sample_id)
        if entry is None or "keep_mask" not in entry:
            return None
        keep_mask = torch.tensor(entry["keep_mask"], dtype=torch.bool)
        grid_shape = entry.get("grid_shape", [8, 16, 16])
        return keep_mask.reshape(grid_shape)

    def get_loss_map(self, sample_id: str) -> torch.Tensor | None:
        entry = self.get(sample_id)
        if entry is None or "loss_map" not in entry:
            return None
        loss_map = torch.tensor(entry["loss_map"], dtype=torch.float32)
        grid_shape = entry.get("grid_shape", [8, 16, 16])
        return loss_map.reshape(grid_shape)

    def get_keep_indices(self, sample_id: str) -> torch.Tensor | None:
        """Get flat keep indices (local video token positions)."""
        keep_mask = self.get_keep_mask(sample_id)
        if keep_mask is None:
            return None
        flat_mask = keep_mask.flatten()
        return torch.where(flat_mask)[0]

    def put(
        self,
        sample_id: str,
        loss_map: torch.Tensor,
        keep_mask: torch.Tensor,
        keep_indices: torch.Tensor | None = None,
    ):
        self._ensure_loaded()
        grid_shape = list(keep_mask.shape)
        self._data[sample_id] = {
            "sample_id": sample_id,
            "keep_mask": keep_mask.flatten().tolist(),
            "loss_map": loss_map.flatten().tolist(),
            "grid_shape": grid_shape,
            "keep_indices": keep_indices.flatten().tolist() if keep_indices is not None else [],
        }

    def flush(self):
        """Write all cached data to disk."""
        self._ensure_loaded()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            for entry in self._data.values():
                f.write(json.dumps(entry) + "\n")

    def __len__(self):
        self._ensure_loaded()
        return len(self._data)

    def __contains__(self, sample_id: str) -> bool:
        return self.has(sample_id)
