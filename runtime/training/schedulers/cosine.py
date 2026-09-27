"""CosineAnnealingLR scheduler build wrapper (ADR 0003 PR-C)."""

from __future__ import annotations

import logging
from typing import Optional


logger = logging.getLogger(__name__)


def build(args, optimizer, total_steps: Optional[int]):
    """Build a CosineAnnealingLR. Warns and returns None (falls back to
    no-scheduler) when total_steps is unknown -- matches the old main() logic."""
    if total_steps is None:
        logger.warning("cosine scheduler needs a known total_steps, falling back to none")
        return None
    import torch

    eta_min = float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=eta_min,
    )
    logger.info(f"LR schedule: cosine (T_max={total_steps}, eta_min={eta_min})")
    return scheduler
