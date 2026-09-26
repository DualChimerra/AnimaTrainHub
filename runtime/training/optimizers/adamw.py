"""AdamW optimizer build wrapper (ADR 0003 PR-C)."""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    """Build an AdamW instance.

    AdamW needs no extra args beyond the standard ones; keeps the same
    signature as the other builders so the registry can dispatch uniformly.
    """
    from utils.optimizer_utils import create_optimizer
    return create_optimizer(
        optimizer_type="adamw",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
    )
