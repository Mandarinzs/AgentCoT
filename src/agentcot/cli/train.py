from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agentcot.trainer import AutoTuneConfig, TrainerConfig, run_autotune


def _load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _simulated_train_once(config: TrainerConfig) -> dict[str, float]:
    """A deterministic placeholder objective used until full training wiring is added."""
    # Lower is better. Minimum around lr=2e-5, num_epochs=3.
    loss = abs(config.learning_rate - 2e-5) * 1e4 + abs(config.num_epochs - 3) * 0.15 + 0.2
    return {"val_loss": float(loss)}


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentCoT train entrypoint")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--train-config", required=True)
    args = parser.parse_args()

    train_cfg = _load_yaml(args.train_config)
    base = TrainerConfig(
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        num_epochs=int(train_cfg.get("num_epochs", 1)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
    )

    auto_tune_raw = train_cfg.get("auto_tune", {}) or {}
    auto_tune = AutoTuneConfig(
        enabled=bool(auto_tune_raw.get("enabled", False)),
        metric_name=str(auto_tune_raw.get("metric_name", "val_loss")),
        mode=str(auto_tune_raw.get("mode", "min")),
        sota_target=float(auto_tune_raw.get("sota_target", 0.0)),
        max_trials=int(auto_tune_raw.get("max_trials", 10)),
        learning_rates=tuple(float(x) for x in auto_tune_raw.get("learning_rates", [1e-5, 2e-5, 5e-5])),
        epoch_candidates=tuple(int(x) for x in auto_tune_raw.get("epoch_candidates", [1, 2, 3])),
    )

    if auto_tune.enabled:
        result = run_autotune(_simulated_train_once, base, auto_tune)
        print(f"[train] auto-tune enabled, reached_sota={result['reached_sota']}")
        print(f"[train] best_metric={result['best_metric']} best_config={result['best_config']}")
        print(f"[train] trials={len(result['trials'])}")
        return

    print(f"[train] model={args.model_config} data={args.data_config} train={args.train_config}")


if __name__ == "__main__":
    main()
