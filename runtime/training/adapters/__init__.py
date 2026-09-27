"""LoRA adapter plugin registry (ADR 0003 PR-C).

Steps to add a new variant (see ADR 0003 Case 2-5 for worked examples):
1. Write training/adapters/{variant}.py containing `build(args, *, preset) -> AdapterProtocol`
2. Add one line to the BUILDERS dict in this file
3. Add an enum value + that variant's dedicated fields to `lora_type: Literal[...]` in studio/schema.py
4. Done. Zero changes needed in phases/models.py / loop.py / main().

Removing a variant: reverse order, 3 steps and you're done.
"""

from __future__ import annotations

from typing import Any, Callable

from training.adapters import lycoris, ortho, tlora
from training.adapters.protocol import AdapterProtocol, StepContext

__all__ = ["AdapterProtocol", "StepContext", "BUILDERS", "build_adapter",
           "validate_schema_consistency"]


# Single source of truth: the registry of all adapter factories
BUILDERS: dict[str, Callable[..., AdapterProtocol]] = {
    "lokr": lycoris.build,
    "loha": lycoris.build,
    "lora": lycoris.build,
    "ortho": ortho.build,
    "tlora": tlora.build,
}


def build_adapter(args, *, preset: dict[str, Any]) -> AdapterProtocol:
    """Dispatch by args.lora_type, explicitly injecting the current model family's target preset."""
    lora_type = args.lora_type
    if lora_type not in BUILDERS:
        raise ValueError(
            f"Unknown lora_type={lora_type!r}; registered: {sorted(BUILDERS)}"
        )
    return BUILDERS[lora_type](args, preset=preset)


def validate_schema_consistency() -> None:
    """Startup-time check: TrainingConfig.lora_type Literal set == BUILDERS keys.

    A mismatch usually means a new variant was added but one spot (schema or registry)
    was missed. Fail early, fix early.
    """
    from studio.schema import TrainingConfig

    field = TrainingConfig.model_fields["lora_type"]
    schema_options = set(field.annotation.__args__)
    registered = set(BUILDERS)
    if schema_options != registered:
        raise RuntimeError(
            f"Adapter registry is out of sync with schema (PR-C registry):\n"
            f"  in schema but not registered: {schema_options - registered}\n"
            f"  registered but not listed in schema: {registered - schema_options}"
        )
