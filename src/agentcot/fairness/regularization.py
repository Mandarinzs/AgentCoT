from __future__ import annotations

import torch


def fairness_regularization_loss(
    logits: torch.Tensor,
    group_ids: torch.Tensor,
    lambda_fair: float = 0.1,
) -> torch.Tensor:
    """Penalty that reduces inter-group mean score disparities."""
    probs = torch.sigmoid(logits)
    unique_groups = torch.unique(group_ids)
    group_means = []

    for g in unique_groups:
        mask = group_ids == g
        if mask.any():
            group_means.append(probs[mask].mean())

    if len(group_means) <= 1:
        return torch.tensor(0.0, device=logits.device)

    spread = torch.stack(group_means).max() - torch.stack(group_means).min()
    return lambda_fair * spread
