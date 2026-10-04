"""REX annealing with warm restarts (RAWR).

Port of RexAnnealingWarmRestarts from derrian-distro/LoRA_Easy_Training_scripts_Backend,
rewritten as a stateless LambdaLR (cycle position derived from the step index, so
resume needs no extra scheduler state). Each cycle:
  - warmup: linear ramp from min_lr to the cycle peak over lr_scheduler_warmup_steps;
  - decay: REX curve min_lr + (peak - min_lr) * z / ((1 - d) + d * z), z = 1 - progress.
After each restart the peak is multiplied by gamma and the decay part of the cycle
by cycle_multiplier. The first cycle length is chosen so lr_scheduler_cycle_count
cycles fill total_steps.
"""

from __future__ import annotations

import logging
from typing import Optional


logger = logging.getLogger(__name__)


def rex_factor(progress: float, d: float) -> float:
    z = 1.0 - min(1.0, max(0.0, progress))
    return z / ((1.0 - d) + d * z)


def cycle_lengths(total_steps: int, cycles: int, multiplier: float, warmup: int) -> list[int]:
    """Lengths (warmup included) of `cycles` cycles covering ~total_steps."""
    geometric = sum(multiplier ** k for k in range(cycles))
    decay0 = max(1, round((total_steps - cycles * warmup) / geometric))
    lengths = [decay0 + warmup]
    while len(lengths) < cycles:
        lengths.append(max(1, round((lengths[-1] - warmup) * multiplier)) + warmup)
    return lengths


def build(args, optimizer, total_steps: Optional[int]):
    if total_steps is None or int(total_steps) <= 0:
        logger.warning("rex_annealing_warm_restarts needs a known total_steps, falling back to none")
        return None

    import torch

    total_steps = int(total_steps)
    cycles = max(1, int(getattr(args, "lr_scheduler_cycle_count", 3) or 1))
    multiplier = float(getattr(args, "lr_scheduler_cycle_multiplier", 1.0) or 1.0)
    gamma = float(getattr(args, "lr_scheduler_gamma", 0.9))
    d = float(getattr(args, "lr_scheduler_rex_d", 0.9))
    eta_min = max(0.0, float(getattr(args, "lr_scheduler_eta_min", 0.0) or 0.0))
    warmup = max(0, int(getattr(args, "lr_scheduler_warmup_steps", 0) or 0))
    if multiplier <= 0.0:
        raise ValueError("rex_annealing_warm_restarts: lr_scheduler_cycle_multiplier must be > 0")
    if not 0.0 < d <= 1.0:
        raise ValueError("rex_annealing_warm_restarts: lr_scheduler_rex_d must be in (0, 1]")
    if gamma <= 0.0:
        raise ValueError("rex_annealing_warm_restarts: lr_scheduler_gamma must be > 0")
    # Warmup has to leave room for decay in every cycle.
    max_warmup = max(0, total_steps // cycles - 1)
    if warmup > max_warmup:
        logger.warning(
            "rex_annealing_warm_restarts: warmup_steps=%s doesn't fit %s cycles in %s steps, "
            "clamping to %s", warmup, cycles, total_steps, max_warmup,
        )
        warmup = max_warmup
    lengths = cycle_lengths(total_steps, cycles, multiplier, warmup)

    def locate(step: int) -> tuple[int, int, int]:
        """(cycle index, step within cycle, cycle length); continues past the planned cycles."""
        cycle, start, length = 0, 0, lengths[0]
        while step >= start + length:
            start += length
            cycle += 1
            if cycle < len(lengths):
                length = lengths[cycle]
            else:
                length = max(1, round((length - warmup) * multiplier)) + warmup
        return cycle, step - start, length

    def make_lambda(base_lr: float):
        min_factor = min(1.0, eta_min / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            cycle, pos, length = locate(max(0, step))
            peak = gamma ** cycle
            if peak <= min_factor:
                return min_factor
            span = peak - min_factor
            if warmup > 0 and pos < warmup:
                return min_factor + span * pos / warmup
            progress = (pos - warmup) / max(1, length - warmup)
            return min_factor + span * rex_factor(progress, d)

        return lr_lambda

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[make_lambda(group["lr"]) for group in optimizer.param_groups],
    )
    logger.info(
        "LR schedule: rex_annealing_warm_restarts (total_steps=%s, cycles=%s, lengths=%s, "
        "multiplier=%s, gamma=%s, d=%s, warmup_steps=%s, eta_min=%s)",
        total_steps, cycles, lengths, multiplier, gamma, d, warmup, eta_min,
    )
    return scheduler
