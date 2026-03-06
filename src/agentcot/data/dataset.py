from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RecommendationQADataset:
    """Simple in-memory dataset wrapper for recommendation QA examples."""

    examples: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def load_jsonl_dataset(path: str | Path) -> RecommendationQADataset:
    """Load line-delimited JSON file into dataset wrapper."""
    file_path = Path(path)
    examples: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    return RecommendationQADataset(examples=examples)
