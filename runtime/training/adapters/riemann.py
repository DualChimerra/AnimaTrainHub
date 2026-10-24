"""Build the fixed-rank Riemann LoRA adapter for Riemannion."""

from __future__ import annotations

from typing import Any

from training.adapters.protocol import AdapterProtocol


def build(args, *, preset: dict[str, Any]) -> AdapterProtocol:
    from training.optimizers.riemannion import validate
    from utils.riemann_lora_adapter import RiemannLoRAAdapter

    validate(args)
    return RiemannLoRAAdapter(
        rank=int(args.lora_rank),
        preset=preset,
        seed=int(getattr(args, "seed", 0) or 0),
    )
