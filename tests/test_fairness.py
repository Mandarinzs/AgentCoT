from agentcot.fairness.metrics import demographic_parity_gap
from agentcot.fairness.reweighting import compute_group_weights


def test_demographic_parity_gap_basic() -> None:
    preds = [1, 0, 1, 1]
    groups = ["A", "A", "B", "B"]
    assert demographic_parity_gap(preds, groups) == 0.5


def test_compute_group_weights() -> None:
    weights = compute_group_weights(["A", "A", "B"])
    assert set(weights) == {"A", "B"}
    assert weights["B"] > weights["A"]
