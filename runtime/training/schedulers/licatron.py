"""Licatron: linear warmup, constant peak, then a REX decay (the staged
warmup -> constant -> REX recipe, 5% / 55% / 40% by default).

Warmup and constant shares are fractions of the whole run; REX takes the rest.
The REX curve uses its own shape d (0.5, the paper curve) and decays to
lr_scheduler_eta_min.
"""

from __future__ import annotations

import logging
from typing import Optional

from training.schedulers.rex_annealing_warm_restarts import rex_factor


logger = logging.getLogger(__name__)


def build(args, optimizer, total_steps: Optional[int]):
    if total_steps is None or int(total_steps) <= 0:
        logger.warning("licatron scheduler needs a known total_steps, falling back to none")
        return None

    import torch

    total_steps = int(total_steps)
    warmup_ratio = float(getattr(args, "lr_scheduler_licatron_warmup_ratio", 0.05))
    constant_ratio = float(getattr(args, "lr_scheduler_licatron_constant_ratio", 0.55))
    if warmup_ratio < 0.0 or constant_ratio < 0.0 or warmup_ratio + constant_ratio >= 1.0:
        raise ValueError(
            "licatron: warmup and constant ratios must be >= 0 and sum to less than 1"
        )
    eta_min = max(0.0, float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0))
    d = float(getattr(args, "lr_scheduler_licatron_rex_d", 0.5))
    if not 0.0 < d <= 1.0:
        raise ValueError("licatron: lr_scheduler_licatron_rex_d must be in (0, 1]")

    warmup_steps = int(round(warmup_ratio * total_steps))
    constant_end = min(total_steps - 1, warmup_steps + int(round(constant_ratio * total_steps)))
    decay_steps = max(1, total_steps - constant_end)

    def make_lambda(base_lr: float):
        min_factor = min(1.0, eta_min / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(warmup_steps)
            if step < constant_end:
                return 1.0
            progress = min(1.0, max(0.0, float(step - constant_end) / float(decay_steps)))
            return min_factor + (1.0 - min_factor) * rex_factor(progress, d)

        return lr_lambda

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[make_lambda(group["lr"]) for group in optimizer.param_groups],
    )
    logger.info(
        "LR schedule: licatron (total_steps=%s, warmup_steps=%s, constant until step %s, d=%s, eta_min=%s)",
        total_steps, warmup_steps, constant_end, d, eta_min,
    )
    return scheduler
