"""Optimizer plugin registry (ADR 0003 PR-C).

To add a new optimizer (Lion / CAME / Schedule-Free AdamW):
1. Write training/optimizers/{variant}.py with
   `build(args, params, lr, weight_decay)`, plus an optional `validate(args)`
   startup check.
2. Add a line to the BUILDERS / VALIDATORS dicts in this file.
3. Add the enum value (and any variant-specific fields) to the
   optimizer_type Literal in studio/schema.py.
4. Add the dependency to requirements.txt (if any).

See ADR 0003 "Case 6: Lion / CAME / Schedule-Free AdamW" for details.
"""

from __future__ import annotations

from typing import Callable, Optional

from training.optimizers import (
    adamw,
    adamw8bit,
    automagic,
    came,
    lion,
    prodigy,
    prodigy_plus_schedulefree,
    simplified_ademamix,
    soap,
    soap_sf,
)

__all__ = ["BUILDERS", "VALIDATORS", "build_optimizer", "validate_optimizer",
           "validate_schema_consistency"]


BUILDERS: dict[str, Callable] = {
    "adamw": adamw.build,
    "adamw8bit": adamw8bit.build,
    "automagic": automagic.build,
    "came": came.build,
    "lion": lion.build,
    "prodigy": prodigy.build,
    "prodigy_plus_schedulefree": prodigy_plus_schedulefree.build,
    "simplified_ademamix": simplified_ademamix.build,
    "soap": soap.build,
    "soap_sf": soap_sf.build,
}

# Startup validation functions (None / unregistered = skipped)
VALIDATORS: dict[str, Callable[[object], None]] = {
    "adamw8bit": adamw8bit.validate,
    "automagic": automagic.validate,
    "prodigy_plus_schedulefree": prodigy_plus_schedulefree.validate,
    "soap_sf": soap_sf.validate,
}


def build_optimizer(args, params, lr: float, weight_decay: float):
    """Dispatch on args.optimizer_type."""
    optimizer_type = (getattr(args, "optimizer_type", "adamw") or "adamw").lower()
    if optimizer_type not in BUILDERS:
        raise ValueError(
            f"Unknown optimizer_type={optimizer_type!r}; registered: {sorted(BUILDERS)}"
        )
    return BUILDERS[optimizer_type](args, params, lr, weight_decay)


def validate_optimizer(args) -> None:
    """Run the optimizer-specific startup compatibility check (e.g. PPSF requires lr_scheduler=none)."""
    optimizer_type = (getattr(args, "optimizer_type", "adamw") or "adamw").lower()
    validator = VALIDATORS.get(optimizer_type)
    if validator:
        validator(args)


def validate_schema_consistency() -> None:
    from studio.schema import TrainingConfig

    field = TrainingConfig.model_fields["optimizer_type"]
    schema_options = set(field.annotation.__args__)
    registered = set(BUILDERS)
    if schema_options != registered:
        raise RuntimeError(
            f"Optimizer registration is out of sync with the schema (PR-C registry):\n"
            f"  in schema but not registered: {schema_options - registered}\n"
            f"  registered but not in schema: {registered - schema_options}"
        )
