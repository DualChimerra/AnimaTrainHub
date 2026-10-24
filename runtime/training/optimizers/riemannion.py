"""Experimental Riemannion, paired with the Riemann LoRA adapter."""

from __future__ import annotations

import logging


def validate(args) -> None:
    if (getattr(args, "optimizer_type", None) != "muon"
            or not getattr(args, "use_riemannian", False)
            or getattr(args, "lora_type", None) != "lora"):
        raise ValueError("Riemannian updates require Muon, Use Riemannian and LoRA")
    # Plain parameter averaging loses the moving bases. Text/norm training
    # require LyCORIS adapter methods this adapter does not implement.
    for field in ("ema_enabled", "train_norm", "train_text_encoder", "lora_dora"):
        if getattr(args, field, False):
            raise ValueError(f"Riemann LoRA does not support {field}; disable it")


def build(args, params, lr: float, weight_decay: float):
    from training.optimizers.extra_args import with_optimizer_args
    from utils.riemannion_optimizer import RiemannionWithAuxAdam

    validate(args)
    kwargs = with_optimizer_args(args, {"momentum": float(getattr(args, "riemannion_momentum", 0.9))})
    optimizer = RiemannionWithAuxAdam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    logging.getLogger(__name__).info(
        "Created Riemannion: lr=%s momentum=%s weight_decay=%s", lr, kwargs["momentum"], weight_decay,
    )
    return optimizer
