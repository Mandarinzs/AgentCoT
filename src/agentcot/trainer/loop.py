from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from accelerate import Accelerator
from torch.optim import AdamW


@dataclass
class TrainerConfig:
    learning_rate: float = 2e-5
    num_epochs: int = 1
    max_grad_norm: float = 1.0


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
