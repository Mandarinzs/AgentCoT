from __future__ import annotations

from collections import Counter


def compute_group_weights(groups: list[str]) -> dict[str, float]:
    """Inverse-frequency group weights for fairer optimization."""
    counts = Counter(groups)
    if not counts:
        return {}
    total = sum(counts.values())
    n_groups = len(counts)
    return {g: total / (n_groups * c) for g, c in counts.items()}
