"""LR scheduler plugin registry (ADR 0003 PR-C).

To add a new scheduler (warmup_cosine / one_cycle / polynomial):
1. Write training/schedulers/{variant}.py with `build(args, optimizer, total_steps)`.
2. Add a line to the BUILDERS dict in this file.
3. Add the enum value (and any variant-specific fields) to the lr_scheduler
   Literal in studio/schema.py.

See ADR 0003 "Case 7: warmup_cosine" for details.

The special key "none" has no file: build_scheduler returns None directly.
"""

from __future__ import annotations

from typing import Callable, Optional

from training.schedulers import (
    constant_then_cosine,
    cosine,
    cosine_cycles,
    cosine_with_restart,
    cosine_with_warmup,
)

__all__ = ["BUILDERS", "build_scheduler", "validate_schema_consistency"]


# "none" is not in BUILDERS -- build_scheduler special-cases it and returns None.
BUILDERS: dict[str, Callable] = {
    "constant_then_cosine": constant_then_cosine.build,
    "cosine": cosine.build,
    "cosine_cycles": cosine_cycles.build,
    "cosine_with_restart": cosine_with_restart.build,
    "cosine_with_warmup": cosine_with_warmup.build,
}

# schema allows "none" but BUILDERS doesn't list it; validate_schema_consistency exempts it
SCHEMA_ONLY_OPTIONS = {"none"}


def build_scheduler(args, optimizer, total_steps: Optional[int]):
    """Dispatch on args.lr_scheduler; "none" or unset returns None."""
    lr_sched = (getattr(args, "lr_scheduler", "none") or "none").lower()
    if lr_sched == "none":
        return None
    if lr_sched not in BUILDERS:
        raise ValueError(
            f"Unknown lr_scheduler={lr_sched!r}; registered: {sorted(BUILDERS)} + 'none'"
        )
    return BUILDERS[lr_sched](args, optimizer, total_steps)


def validate_schema_consistency() -> None:
    from studio.schema import TrainingConfig

    field = TrainingConfig.model_fields["lr_scheduler"]
    schema_options = set(field.annotation.__args__) - SCHEMA_ONLY_OPTIONS
    registered = set(BUILDERS)
    if schema_options != registered:
        raise RuntimeError(
            f"Scheduler registration is out of sync with the schema (PR-C registry):\n"
            f"  in schema but not registered: {schema_options - registered}\n"
            f"  registered but not in schema: {registered - schema_options}\n"
            f"  (schema-only, skipped: {SCHEMA_ONLY_OPTIONS})"
        )
