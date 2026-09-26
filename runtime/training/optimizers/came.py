"""CAME optimizer build wrapper (ADR 0003 PR-C).

CAME = Confidence-guided Adaptive Memory Efficient optimizer
(Luo et al., 2023, ACL 2023, arxiv 2307.02047). Adafactor-style factored second
moment to save VRAM, plus confidence-guided suppression of update noise. A
regular optimizer: real lr (AdamW scale), configurable lr_scheduler, no
train()/eval() switching. Implemented in utils/optimizer_utils.py `class CAME`.
"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    """Build a CAME instance; reads the came_* args."""
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="came",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "came_beta1", 0.9)),
            float(getattr(args, "came_beta2", 0.999)),
            float(getattr(args, "came_beta3", 0.9999)),
        ),
        eps=(
            float(getattr(args, "came_eps1", 1e-30)),
            float(getattr(args, "came_eps2", 1e-16)),
        ),
        clip_threshold=float(getattr(args, "came_clip_threshold", 1.0)),
    )
