"""AdapterProtocol: the unified interface for all LoRA variants (ADR 0003 PR-C).

Design principles:
- 4 required methods (inject / get_param_groups / save / load) -- every adapter must implement these
- 3 optional hooks (on_step_begin / regularization_loss / excludes_weight_decay) --
  no-op by default; dynamic/per-step variants (T-LoRA / AdaLoRA / OFT etc.) override as needed
- runtime_checkable -- tests can verify directly with `isinstance(adapter, AdapterProtocol)`

The hook design follows the real-world needs of T-LoRA / OFT / Ortho-Hydra described in the
"Case 3-5" section of ADR 0003.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StepContext:
    """The minimal context passed to adapter.on_step_begin at the start of each micro-batch forward pass.

    Fields are kept minimal: only what hooks actually use (to avoid dataclass field
    bloat and to keep hook implementations independent of TrainingContext's full
    internal state).
    """
    global_step: int
    total_steps: Optional[int]
    epoch: int
    sigma_t: Tensor          # shape [B], sigma for this micro-batch
    args: object             # the Namespace from parse_args; read fields as needed


@runtime_checkable
class AdapterProtocol(Protocol):
    """Shared interface for LoRA / LoKr / LoHa / paper-level variants.

    Uses Protocol rather than ABC: the existing AnimaLycorisAdapter already implements
    the first 4 required methods (duck-typed), and we don't want to force users to
    inherit. runtime_checkable keeps `isinstance` checks usable in unit tests.
    """

    # --- Required ---
    def inject(self, model: nn.Module) -> None:
        """Inject LoRA layers into the model (replacing Linear etc.)."""
        ...

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        """Return the optimizer param_groups. One or more groups (LoRA+ can split A/B
        into different lrs, Ortho-Hydra can give the router its own lr)."""
        ...

    def save(self, path: Path) -> None:
        """Save to safetensors on disk."""
        ...

    def load(self, path: Path) -> None:
        """Load from safetensors."""
        ...

    # --- Optional hooks: no-op by default; override as needed ---

    def on_step_begin(self, ctx: StepContext) -> None:
        """Called before each micro-batch forward pass.

        T-LoRA / AdaLoRA / B-LoRA use this to adjust the rank mask, active subset,
        column dropout, etc. ("runtime structural adjustment") based on sigma_t / step.
        No-op by default.
        """
        return None

    def regularization_loss(self, ctx: StepContext) -> Optional[Tensor]:
        """Return a regularization term to add to the main loss; None = none.

        OFT returns an orthogonality penalty; Ortho-Hydra returns an expert balance loss;
        None by default. When train_loop receives None it does nothing extra.
        """
        return None

    def excludes_weight_decay(self, param_name: str) -> bool:
        """Whether this param should be excluded from weight_decay.

        Replaces the old hardcoded `injector.use_lokr` check. In the LoKr implementation:
        return "w1" in param_name. False by default.
        """
        return False
