"""TimestepSamplerProtocol: the unified interface for all timestep samplers (ADR 0003 plugin registry).

Modeled on training/adapters/protocol.py:
- 1 required method: sample(bs, device, *, token_counts=None) -> Tensor
- 3 optional hooks: record / maybe_refresh / status -- default no-op;
  adaptive samplers (InfoNoise etc.) override as needed, pure-distribution
  samplers (logit_normal etc.) keep them as no-op.

To add a new sampler (follows the ADR 0003 PR-C registry pattern):
1. Write training/timestep_samplers/{name}.py with
   `build(args, total_steps) -> TimestepSamplerProtocol`.
2. Add a line to the BUILDERS dict in timestep_samplers/__init__.py.
3. Done -- zero changes needed in phases/optimizer.py / loop.py / TrainingContext.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class TimestepSamplerProtocol(Protocol):
    """Unified interface for timestep samplers.

    Uses Protocol instead of ABC: baseline is a dataclass, InfoNoise is a
    plain class, and we don't want to force inheritance. runtime_checkable
    keeps unit-test `isinstance` checks working.
    """

    def sample(self, bs: int, device, *, token_counts=None) -> torch.Tensor:
        """Sample bs values of t in (0, 1).

        ``token_counts`` is optional batch context; resolution-aware samplers
        that need it declare ``requires_token_counts = True`` so the loop
        computes and passes it. Ordinary samplers don't take on extra
        per-batch computation.
        """
        ...

    # --- Optional hooks: default no-op; adaptive samplers override as needed ---

    def record(self, t: torch.Tensor, raw_mse: torch.Tensor) -> None:
        """Record the per-sample raw MSE after each micro-batch, so adaptive
        samplers can update their distribution.

        Non-adaptive samplers (baseline) keep this as a no-op.
        """
        return None

    def maybe_refresh(self, global_step: int) -> None:
        """Called after each optimizer step, letting the sampler decide
        whether to refresh internal state (e.g. the CDF).

        Non-adaptive samplers keep this as a no-op.
        """
        return None

    def status(self) -> dict:
        """Expose internal state for wandb monitoring / debugging; adaptive
        samplers override to provide meaningful info."""
        return {}

    # --- Pause/resume support (ADR 0006 Addendum 1): adaptive samplers must
    # override; stateless samplers keep the default no-op. ---

    def state_dict(self) -> dict:
        """Serialize internal state for resume.

        Stateless samplers (baseline / pure distribution) return {} --
        save_training_state skips persisting it. Adaptive samplers (InfoNoise
        etc.) must override this to export the EMA / CDF / FIFO buffer,
        otherwise resume falls back to a cold start and any learned schedule
        is lost.
        """
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore internal state from state_dict; stateless samplers keep the
        default no-op.

        Implementations should log a warning and fall back to a cold start on
        a shape mismatch (e.g. K / B changed) rather than raising -- training
        may already have run for hours, and resume shouldn't crash just
        because some hyperparameter changed.
        """
        return None
