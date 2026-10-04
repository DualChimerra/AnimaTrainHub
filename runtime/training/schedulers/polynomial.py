"""Polynomial decay scheduler with linear warmup.

Matches transformers' get_polynomial_decay_schedule_with_warmup: after warmup,
lr = min_lr + (base_lr - min_lr) * (1 - progress) ** power. power=1 is linear
decay; power>1 drops faster early and flattens out near the end.
"""

from __future__ import annotations

import logging
from typing import Optional


logger = logging.getLogger(__name__)


def build(args, optimizer, total_steps: Optional[int]):
    if total_steps is None or int(total_steps) <= 0:
        logger.warning("polynomial scheduler needs a known total_steps, falling back to none")
        return None

    import torch

    total_steps = int(total_steps)
    warmup_steps = int(getattr(args, "lr_scheduler_warmup_steps", 0) or 0)
    warmup_steps = max(0, min(warmup_steps, total_steps))
    power = float(getattr(args, "lr_scheduler_power", 1.0) or 1.0)
    eta_min = max(0.0, float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0))
    if power <= 0.0:
        raise ValueError("polynomial: lr_scheduler_power must be > 0")

    def make_lambda(base_lr: float):
        min_factor = min(1.0, eta_min / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(warmup_steps)
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
            return min_factor + (1.0 - min_factor) * (1.0 - progress) ** power

        return lr_lambda

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[make_lambda(group["lr"]) for group in optimizer.param_groups],
    )
    logger.info(
        "LR schedule: polynomial (total_steps=%s, warmup_steps=%s, power=%s, eta_min=%s)",
        total_steps, warmup_steps, power, eta_min,
    )
    return scheduler
