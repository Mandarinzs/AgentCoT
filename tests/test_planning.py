from pathlib import Path

from agentcot.planning.loader import load_research_bundle


def test_load_research_bundle_has_expected_sections() -> None:
    bundle = load_research_bundle(Path(__file__).resolve().parents[1])
    assert set(bundle) == {
        "experiment_matrix",
        "ablation_plan",
        "phased_rollout",
        "post_network_research_checklist",
    }


def test_experiment_matrix_has_axes() -> None:
    bundle = load_research_bundle(Path(__file__).resolve().parents[1])
    axes = bundle["experiment_matrix"]["axes"]
    assert "model" in axes
    assert "fairness_strategy" in axes
