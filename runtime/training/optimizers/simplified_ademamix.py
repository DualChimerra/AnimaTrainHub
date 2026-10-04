"""Simplified AdEMAMix optimizer build wrapper.

Simplified AdEMAMix (Morwani et al., 2025, arxiv 2502.02431): one momentum
buffer that sums raw gradients plus alpha * current gradient. A regular
optimizer with a configurable lr_scheduler, but its lr is ~(1 - beta1) times
the AdamW lr (1e-6 at beta1=0.99 for AdamW 1e-4). Implemented in
utils/optimizer_utils.py `class SimplifiedAdEMAMix`.
"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="simplified_ademamix",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "ademamix_beta1", 0.99)),
            float(getattr(args, "ademamix_beta2", 0.999)),
        ),
        alpha=float(getattr(args, "ademamix_alpha", 0.0)),
        beta1_warmup=int(getattr(args, "ademamix_beta1_warmup", 0) or 0),
        min_beta1=float(getattr(args, "ademamix_min_beta1", 0.9)),
    )
