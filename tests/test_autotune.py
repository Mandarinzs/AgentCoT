from agentcot.trainer import AutoTuneConfig, TrainerConfig, run_autotune


def test_run_autotune_reaches_target() -> None:
    def train_once(config: TrainerConfig) -> dict[str, float]:
        return {"val_loss": abs(config.learning_rate - 2e-5) * 1e4 + abs(config.num_epochs - 3) * 0.1 + 0.2}

    result = run_autotune(
        train_once=train_once,
        base_config=TrainerConfig(),
        tune_config=AutoTuneConfig(
            enabled=True,
            metric_name="val_loss",
            mode="min",
            sota_target=0.25,
            max_trials=20,
            learning_rates=(1e-5, 2e-5, 5e-5),
            epoch_candidates=(1, 2, 3),
        ),
    )

    assert result["reached_sota"] is True
    assert result["best_metric"] <= 0.25


def test_run_autotune_invalid_mode() -> None:
    def train_once(_: TrainerConfig) -> dict[str, float]:
        return {"val_loss": 1.0}

    try:
        run_autotune(
            train_once=train_once,
            base_config=TrainerConfig(),
            tune_config=AutoTuneConfig(mode="median"),
        )
    except ValueError as exc:
        assert "Unsupported tune mode" in str(exc)
    else:
        assert False, "Expected ValueError"
