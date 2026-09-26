"""LossProtocol: the unified interface for all training losses (ADR 0003 plugin registry).

Modeled on training/timestep_samplers/protocol.py and training/adapters/protocol.py:
- 1 required method: compute(pred, target, t) -> Tensor

To add a new loss (follows the ADR 0003 PR-C registry pattern):
1. Write training/losses/{name}.py with `build(args) -> LossProtocol`.
2. Add a line to the BUILDERS dict in losses/__init__.py.
3. Add the enum value (and any dedicated fields) to the
   `loss_type: Literal[...]` in studio/schema.py.
4. Done -- zero changes needed in phases/optimizer.py / loop.py / TrainingContext.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LossProtocol(Protocol):
    """Unified interface for training losses.

    Uses Protocol instead of ABC: mse is a plain function wrapper, huber is a
    class, and we don't want to force inheritance. runtime_checkable keeps
    unit-test `isinstance` checks working.

    **Signature stability contract**: compute(pred, target, t) is the current
    v1 signature. If future losses need extra context (LPIPS / SNR-aware /
    classifier-free guidance, etc.), it will be appended as a keyword-only
    `*, ctx=None` (backward compatible; old implementations keep working when
    it's not passed). New loss implementations are encouraged to pre-declare
    `def compute(self, pred, target, t, *, ctx=None)` to avoid a future break;
    the caller (loop.py) doesn't pass ctx yet, and when it starts will be
    announced separately.
    """

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return a per-element loss tensor, same shape as pred/target (no reduction applied).

        pred / target -- Flow Matching velocity prediction vs. target, shape (B, C, *spatial)
        t              -- the current batch's timestep (B,), only needed by t-dependent losses;
                          mse / constant-delta huber can ignore it
        """
        ...
