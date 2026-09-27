"""Cosine scheduler with linear warmup build wrapper."""

from __future__ import annotations

import logging
import math
from typing import Optional


logger = logging.getLogger(__name__)


def build(args, optimizer, total_steps: Optional[int]):
    if total_steps is None:
        logger.warning("cosine_with_warmup scheduler needs a known total_steps, falling back to none")
        return None

    import torch

    total_steps = int(total_steps)
    warmup_steps = int(getattr(args, "lr_scheduler_warmup_steps", 100) or 0)
    eta_min = max(0.0, float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0))
    if total_steps <= 0:
        logger.warning("cosine_with_warmup scheduler has total_steps<=0, falling back to none")
        return None
    warmup_steps = max(0, min(warmup_steps, total_steps))
    if total_steps <= warmup_steps:
        logger.warning(
            "cosine_with_warmup: total_steps(%s) <= warmup_steps(%s), "
            "the whole run will just be warmup with no cosine decay phase",
            total_steps, warmup_steps,
        )

    def make_lambda(base_lr: float):
        min_factor = eta_min / base_lr if base_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            # warmup: step / warmup_steps (matches the transformers /
            # diffusers / PEFT / sd-scripts community convention, step 0-indexed)
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(warmup_steps)
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine_factor

        return lr_lambda

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[make_lambda(group["lr"]) for group in optimizer.param_groups],
    )
    logger.info(
        "LR schedule: cosine_with_warmup (total_steps=%s, warmup_steps=%s, eta_min=%s)",
        total_steps,
        warmup_steps,
        eta_min,
    )
    return scheduler
