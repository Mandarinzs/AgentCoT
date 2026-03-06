from __future__ import annotations

from collections import defaultdict


def demographic_parity_gap(preds: list[int], groups: list[str]) -> float:
    """Absolute difference between max/min group positive rates."""
    counts = defaultdict(int)
    positives = defaultdict(int)

    for p, g in zip(preds, groups, strict=True):
        counts[g] += 1
        positives[g] += int(p == 1)

    rates = [positives[g] / counts[g] for g in counts if counts[g] > 0]
    if not rates:
        return 0.0
    return max(rates) - min(rates)


def equal_opportunity_gap(preds: list[int], labels: list[int], groups: list[str]) -> float:
    """Absolute difference between max/min group true positive rates."""
    tp = defaultdict(int)
    pos = defaultdict(int)

    for p, y, g in zip(preds, labels, groups, strict=True):
        if y == 1:
            pos[g] += 1
            if p == 1:
                tp[g] += 1

    rates = [tp[g] / pos[g] for g in pos if pos[g] > 0]
    if not rates:
        return 0.0
    return max(rates) - min(rates)
