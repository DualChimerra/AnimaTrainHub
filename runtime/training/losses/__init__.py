"""Training loss plugin registry (ADR 0003 PR-C pattern).

`build_loss(args) -> LossProtocol` dispatches to a specific loss based on
args.loss_type:
- mse    -- wraps F.mse_loss (default, byte-identical to legacy behavior)
- huber  -- Huber loss with constant delta (standard EDM/Karras approach)

Steps to add a new loss:
1. Write training/losses/{name}.py containing `build(args) -> LossProtocol`
2. Add one line to the BUILDERS dict in this file
3. Add the value to `loss_type: Literal[...]` in studio/schema.py, plus any
   fields specific to that loss
4. Done. No changes needed to phases/optimizer.py / loop.py / TrainingContext.

Removing a loss: reverse the above, 3 steps and done;
`validate_schema_consistency()` guards against missing one.
"""

from __future__ import annotations

from typing import Callable

from training.losses import huber, mse
from training.losses.protocol import LossProtocol

__all__ = ["LossProtocol", "BUILDERS", "build_loss", "validate_schema_consistency"]


# Single source of truth: the registry of all loss factories
BUILDERS: dict[str, Callable[..., LossProtocol]] = {
    "mse": mse.build,
    "huber": huber.build,
}


def build_loss(args) -> LossProtocol:
    """Dispatch to the corresponding build() based on args.loss_type."""
    loss_type = (getattr(args, "loss_type", "mse") or "mse").lower()
    if loss_type not in BUILDERS:
        raise ValueError(
            f"Unknown loss_type={loss_type!r}; registered: {sorted(BUILDERS)}"
        )
    return BUILDERS[loss_type](args)


def validate_schema_consistency() -> None:
    """Startup-time check: TrainingConfig.loss_type's Literal set == BUILDERS keys.

    A mismatch usually means a new loss was added but one spot (schema or
    registry) was missed. Fail early to fix it early.
    """
    from studio.schema import TrainingConfig

    field = TrainingConfig.model_fields["loss_type"]
    schema_options = set(field.annotation.__args__)
    registered = set(BUILDERS)
    if schema_options != registered:
        raise RuntimeError(
            f"loss registry out of sync with schema:\n"
            f"  in schema but not registered: {schema_options - registered}\n"
            f"  registered but not listed in schema: {registered - schema_options}"
        )
