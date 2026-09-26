"""CosineAnnealingWarmRestarts scheduler build wrapper (ADR 0003 PR-C)."""

from __future__ import annotations

import logging
from typing import Optional


logger = logging.getLogger(__name__)


def build(args, optimizer, total_steps: Optional[int]):
    """Build a CosineAnnealingWarmRestarts. Not passing total_steps doesn't
    block it, matching the old main() logic."""
    import torch

    t0 = int(getattr(args, "lr_scheduler_t0", 500) or 500)
    t_mult = int(getattr(args, "lr_scheduler_t_mult", 2) or 2)
    eta_min = float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min,
    )
    logger.info(f"LR schedule: cosine_with_restart (T_0={t0}, T_mult={t_mult}, eta_min={eta_min})")
    return scheduler
