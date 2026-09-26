"""SOAP optimizer build wrapper (ADR 0003 PR-C).

SOAP = Adam in the Shampoo eigenbasis (Vyas et al., 2024, arxiv 2409.11321).
This is the plain (non-schedule-free) variant: configurable lr_scheduler, no
train()/eval() switching. See soap_sf.py for the schedule-free variant.
"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    """Build a SOAP instance; reads the soap_* args."""
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="soap",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "soap_beta1", 0.95)),
            float(getattr(args, "soap_beta2", 0.95)),
        ),
        shampoo_beta=float(getattr(args, "soap_shampoo_beta", -1.0)),
        precondition_frequency=int(getattr(args, "soap_precondition_frequency", 10)),
        max_precond_dim=int(getattr(args, "soap_max_precond_dim", 10000)),
        precond_in_state=bool(getattr(args, "soap_precond_in_state", True)),
    )
