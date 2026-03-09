from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Sequence

import torch
from accelerate import Accelerator
from torch.optim import AdamW


@dataclass
class TrainerConfig:
    learning_rate: float = 2e-5
    num_epochs: int = 1
    max_grad_norm: float = 1.0


@dataclass
class AutoTuneConfig:
    """Config for automatic hyper-parameter tuning against a SOTA target."""

    enabled: bool = False
    metric_name: str = "val_loss"
    mode: str = "min"  # "min" for loss-like metrics, "max" for score-like metrics
    sota_target: float = 0.0
    max_trials: int = 10
    learning_rates: Sequence[float] = (1e-5, 2e-5, 5e-5)
    epoch_candidates: Sequence[int] = (1, 2, 3)


def run_training(
    model,
    train_dataloader: Iterable,
    val_dataloader: Iterable,
    config: TrainerConfig,
) -> dict[str, float]:
    """Minimal accelerate-based training/validation loop."""
    accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    for _ in range(config.num_epochs):
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            outputs = model(**batch)
            losses.append(accelerator.gather(outputs.loss.detach()).mean().item())

    val_loss = float(sum(losses) / max(1, len(losses)))
    return {"val_loss": val_loss}


def run_autotune(
    train_once: Callable[[TrainerConfig], dict[str, float]],
    base_config: TrainerConfig,
    tune_config: AutoTuneConfig,
) -> dict[str, object]:
    """Run simple grid-search tuning and stop early if SOTA target is reached."""
    if tune_config.mode not in {"min", "max"}:
        msg = f"Unsupported tune mode: {tune_config.mode}. Expected 'min' or 'max'."
        raise ValueError(msg)

    trial_grid = list(product(tune_config.learning_rates, tune_config.epoch_candidates))
    trial_grid = trial_grid[: tune_config.max_trials]
    if not trial_grid:
        return {
            "reached_sota": False,
            "best_metric": None,
            "best_config": None,
            "trials": [],
        }

    def is_better(candidate: float, incumbent: float) -> bool:
        if tune_config.mode == "min":
            return candidate < incumbent
        return candidate > incumbent

    def hit_target(metric: float) -> bool:
        if tune_config.mode == "min":
            return metric <= tune_config.sota_target
        return metric >= tune_config.sota_target

    best_metric = None
    best_config = None
    trials: list[dict[str, float | int]] = []

    for learning_rate, num_epochs in trial_grid:
        trial_config = TrainerConfig(
            learning_rate=float(learning_rate),
            num_epochs=int(num_epochs),
            max_grad_norm=base_config.max_grad_norm,
        )
        metrics = train_once(trial_config)
        trial_metric = float(metrics[tune_config.metric_name])

        trial_record = {
            "learning_rate": trial_config.learning_rate,
            "num_epochs": trial_config.num_epochs,
            "metric": trial_metric,
        }
        trials.append(trial_record)

        if best_metric is None or is_better(trial_metric, float(best_metric)):
            best_metric = trial_metric
            best_config = trial_config

        if hit_target(trial_metric):
            return {
                "reached_sota": True,
                "best_metric": best_metric,
                "best_config": best_config,
                "trials": trials,
            }

    return {
        "reached_sota": bool(best_metric is not None and hit_target(float(best_metric))),
        "best_metric": best_metric,
        "best_config": best_config,
        "trials": trials,
    }
