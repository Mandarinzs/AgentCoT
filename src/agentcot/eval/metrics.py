from __future__ import annotations

from agentcot.fairness.metrics import demographic_parity_gap, equal_opportunity_gap


def hit_rate_at_k(labels: list[list[str]], preds: list[list[str]], k: int = 5) -> float:
    """Recommendation-style hit-rate@k for QA candidate ranking."""
    hits = 0
    for gold, predicted in zip(labels, preds, strict=True):
        top_k = set(predicted[:k])
        hits += int(any(item in top_k for item in gold))
    return hits / max(1, len(labels))


def evaluate_joint_metrics(
    qa_labels: list[list[str]],
    qa_preds: list[list[str]],
    fairness_preds: list[int],
    fairness_labels: list[int],
    groups: list[str],
) -> dict[str, float]:
    """Joint evaluation for recommendation QA quality and fairness."""
    return {
        "hit_rate@5": hit_rate_at_k(qa_labels, qa_preds, k=5),
        "demographic_parity_gap": demographic_parity_gap(fairness_preds, groups),
        "equal_opportunity_gap": equal_opportunity_gap(
            fairness_preds, fairness_labels, groups
        ),
    }
