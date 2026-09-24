"""Equal-length cosine cycles with configurable LR bounds per cycle."""

from __future__ import annotations

import logging
import math
from typing import Optional


logger = logging.getLogger(__name__)


def _float_list(args, name: str) -> list[float]:
    value = getattr(args, name, None)
    if value is None:
        return []
    return [float(item) for item in value]


def _cycle_values(
    *,
    count: int,
    common: float,
    overrides: list[float],
    name: str,
) -> list[float]:
    if not overrides:
        return [common] * count
    if len(overrides) != count:
        raise ValueError(
            f"cosine_cycles: {name} must contain exactly {count} values; "
            f"got {len(overrides)}"
        )
    return overrides


def build(args, optimizer, total_steps: Optional[int]):
    """Build hard-restart cosine cycles of equal length.

    ``cycle_max_lrs`` / ``cycle_min_lrs`` are absolute learning rates.  Empty
    lists fall back to the common values.  A missing common maximum preserves
    each optimizer parameter group's own base LR.
    """
    if total_steps is None or int(total_steps) <= 0:
        logger.warning("cosine_cycles requires total_steps; falling back to none")
        return None

    import torch

    total_steps = int(total_steps)
    cycle_count = int(getattr(args, "lr_scheduler_cycle_count", 3) or 3)
    if cycle_count < 1:
        raise ValueError("cosine_cycles: lr_scheduler_cycle_count must be >= 1")

    common_max = getattr(args, "lr_scheduler_cycle_max_lr", None)
    common_max = None if common_max is None else float(common_max)
    common_min = float(getattr(args, "lr_scheduler_cycle_min_lr", 0.0) or 0.0)
    max_overrides = _float_list(args, "lr_scheduler_cycle_max_lrs")
    min_overrides = _float_list(args, "lr_scheduler_cycle_min_lrs")

    lambdas = []
    resolved_for_log: list[tuple[list[float], list[float]]] = []
    for group in optimizer.param_groups:
        base_lr = float(group["lr"])
        max_lrs = _cycle_values(
            count=cycle_count,
            common=base_lr if common_max is None else common_max,
            overrides=max_overrides,
            name="lr_scheduler_cycle_max_lrs",
        )
        min_lrs = _cycle_values(
            count=cycle_count,
            common=common_min,
            overrides=min_overrides,
            name="lr_scheduler_cycle_min_lrs",
        )
        for index, (min_lr, max_lr) in enumerate(zip(min_lrs, max_lrs), start=1):
            if max_lr <= 0.0:
                raise ValueError(f"cosine_cycles: cycle {index} max LR must be > 0")
            if min_lr < 0.0:
                raise ValueError(f"cosine_cycles: cycle {index} min LR must be >= 0")
            if min_lr > max_lr:
                raise ValueError(
                    f"cosine_cycles: cycle {index} min LR ({min_lr}) exceeds "
                    f"max LR ({max_lr})"
                )

        def lr_lambda(step: int, *, _base=base_lr, _max=max_lrs, _min=min_lrs) -> float:
            progress = min(1.0, max(0.0, float(step) / float(total_steps)))
            scaled = progress * cycle_count
            cycle_index = min(int(scaled), cycle_count - 1)
            phase = min(1.0, max(0.0, scaled - cycle_index))
            cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
            lr = _min[cycle_index] + (_max[cycle_index] - _min[cycle_index]) * cosine
            return lr / _base if _base > 0.0 else 0.0

        lambdas.append(lr_lambda)
        resolved_for_log.append((max_lrs, min_lrs))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    logger.info(
        "LR scheduler: cosine_cycles (total_steps=%s, cycles=%s, group_bounds=%s)",
        total_steps,
        cycle_count,
        resolved_for_log,
    )
    return scheduler
