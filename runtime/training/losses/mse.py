"""MSE loss: wraps F.mse_loss(reduction='none'), exposing the LossProtocol interface.

Byte-for-byte equivalent to the pre-PR-A/B/C loop.py behavior (given the same
pred/target/t inputs, mse.compute() matches F.mse_loss(reduction='none')).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MseLoss:
    """Stateless MSE: doesn't depend on t, just forwards to F.mse_loss."""

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction="none")


def build(args) -> MseLoss:
    """MSE has no dedicated fields, so args is unused (kept for a consistent dispatch signature)."""
    return MseLoss()
