"""REX scheduler (Chen et al., 2021, "REX: Revisiting Budgeted Training with an
Improved Schedule", arxiv 2107.04197) with optional linear warmup.

After warmup, with z = 1 - progress:
    factor = z / ((1 - d) + d * z)
d = 0.5 is the paper's curve; larger d (LoRA Easy Training Scripts uses 0.9) holds
the peak lr longer and drops more sharply at the end.
"""

from __future__ import annotations

import logging
from typing import Optional

from training.schedulers.rex_annealing_warm_restarts import rex_factor


logger = logging.getLogger(__name__)


def build(args, optimizer, total_steps: Optional[int]):
    if total_steps is None or int(total_steps) <= 0:
        logger.warning("rex scheduler needs a known total_steps, falling back to none")
        return None

    import torch

    total_steps = int(total_steps)
    warmup_steps = int(getattr(args, "lr_scheduler_warmup_steps", 0) or 0)
    warmup_steps = max(0, min(warmup_steps, total_steps))
    eta_min = max(0.0, float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0))
    d = float(getattr(args, "lr_scheduler_rex_d", 0.9))
    if not 0.0 < d <= 1.0:
        raise ValueError("rex: lr_scheduler_rex_d must be in (0, 1]")

    def make_lambda(base_lr: float):
        min_factor = min(1.0, eta_min / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(warmup_steps)
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
            return min_factor + (1.0 - min_factor) * rex_factor(progress, d)

        return lr_lambda

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[make_lambda(group["lr"]) for group in optimizer.param_groups],
    )
    logger.info(
        "LR schedule: rex (total_steps=%s, warmup_steps=%s, d=%s, eta_min=%s)",
        total_steps, warmup_steps, d, eta_min,
    )
    return scheduler
