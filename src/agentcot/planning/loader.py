from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PLAN_PATHS = {
    "experiment_matrix": "configs/research/experiment_matrix.yaml",
    "ablation_plan": "configs/research/ablation_plan.yaml",
    "phased_rollout": "configs/research/phased_rollout.yaml",
    "post_network_research_checklist": "configs/research/post_network_research_checklist.yaml",
}


def load_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    with plan_path.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"Plan file must be a mapping: {plan_path}")
    return content


def load_research_bundle(repo_root: str | Path = ".") -> dict[str, dict[str, Any]]:
    root = Path(repo_root)
    return {
        name: load_plan(root / relative_path)
        for name, relative_path in DEFAULT_PLAN_PATHS.items()
    }
