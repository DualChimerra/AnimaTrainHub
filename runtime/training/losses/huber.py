"""Huber loss (constant delta).

Huber formula (per-element):
    L(x) = 0.5 * x^2                 if |x| < delta
    L(x) = delta * (|x| - 0.5*delta) otherwise

More robust to outliers than MSE, mitigating gradient explosions from
extreme samples. Mainstream trainers like EDM/Karras / kohya-ss all use a
constant delta.

**If a t-dependent delta schedule is introduced in the future, it must come
with (a) a paper DOI, (b) a link to an upstream trainer implementation, or
(c) this project's own ablation data** -- see
[[feedback_verify_paper_before_fixing_algo]] (same root cause as the P0-2 EMA
flip incident).
"""

from __future__ import annotations

import torch


class HuberLoss:
    """Huber loss with constant delta (per-element, reduction='none')."""

    def __init__(self, c: float = 0.15):
        self.c = float(c)

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,  # noqa: ARG002 -- kept for LossProtocol consistency; unused by constant-delta huber
    ) -> torch.Tensor:
        diff = pred - target
        abs_diff = diff.abs()
        quad = 0.5 * diff * diff
        lin = self.c * (abs_diff - 0.5 * self.c)
        return torch.where(abs_diff < self.c, quad, lin)


def build(args) -> HuberLoss:
    return HuberLoss(
        c=float(getattr(args, "huber_c", 0.15) or 0.15),
    )
