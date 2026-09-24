"""Constant learning rate followed by one final cosine decay."""

from __future__ import annotations

import logging
import math
from typing import Optional


logger = logging.getLogger(__name__)


def build(args, optimizer, total_steps: Optional[int]):
    if total_steps is None or int(total_steps) <= 0:
        logger.warning("constant_then_cosine requires total_steps; falling back to none")
        return None

    import torch

    total_steps = int(total_steps)
    decay_start = float(
        getattr(args, "lr_scheduler_decay_start_ratio", 2.0 / 3.0)
    )
    min_lr = float(getattr(args, "lr_scheduler_decay_min_lr", 0.0) or 0.0)
    if not 0.0 <= decay_start < 1.0:
        raise ValueError(
            "constant_then_cosine: lr_scheduler_decay_start_ratio must be in [0, 1)"
        )
    if min_lr < 0.0:
        raise ValueError("constant_then_cosine: lr_scheduler_decay_min_lr must be >= 0")

    lambdas = []
    for group in optimizer.param_groups:
        base_lr = float(group["lr"])
        if min_lr > base_lr:
            raise ValueError(
                "constant_then_cosine: lr_scheduler_decay_min_lr "
                f"({min_lr}) exceeds parameter-group LR ({base_lr})"
            )
        min_factor = min_lr / base_lr if base_lr > 0.0 else 0.0

        def lr_lambda(step: int, *, _min_factor=min_factor) -> float:
            progress = min(1.0, max(0.0, float(step) / float(total_steps)))
            if progress <= decay_start:
                return 1.0
            decay_progress = (progress - decay_start) / (1.0 - decay_start)
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return _min_factor + (1.0 - _min_factor) * cosine

        lambdas.append(lr_lambda)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    logger.info(
        "LR scheduler: constant_then_cosine "
        "(total_steps=%s, decay_start_ratio=%s, min_lr=%s)",
        total_steps,
        decay_start,
        min_lr,
    )
    return scheduler
