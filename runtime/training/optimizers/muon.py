"""Muon with auxiliary AdamW for vector parameters."""

from __future__ import annotations


def validate(args):
    if getattr(args, "use_riemannian", False):
        from .riemannion import validate as validate_riemannian
        validate_riemannian(args)


def build(args, params, lr: float, weight_decay: float):
    if getattr(args, "use_riemannian", False):
        from .riemannion import build as build_riemannian
        return build_riemannian(args, params, lr, weight_decay)
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="muon",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        momentum=float(getattr(args, "muon_momentum", 0.95)),
        ns_steps=int(getattr(args, "muon_ns_steps", 5)),
        adjust_lr_fn=str(getattr(args, "muon_adjust_lr", "match_rms_adamw")),
    )
