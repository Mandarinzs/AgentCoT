from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


class DatasetCache:
    """Cache processed datasets to local disk with pickle."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, payload: Any) -> Path:
        target = self.cache_dir / f"{key}.pkl"
        with target.open("wb") as f:
            pickle.dump(payload, f)
        return target

    def load(self, key: str) -> Any:
        target = self.cache_dir / f"{key}.pkl"
        if not target.exists():
            raise FileNotFoundError(f"Cache entry not found: {target}")
        with target.open("rb") as f:
            return pickle.load(f)
