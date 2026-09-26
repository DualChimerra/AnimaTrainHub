"""Schedule-Free SOAP optimizer build wrapper (ADR 0003 PR-C).

SOAP preconditioning + Schedule-Free trajectory (Defazio et al., 2024, arxiv
2405.15682). Like PPSF, this belongs to the schedule-free family: it has a
built-in LR schedule that is mutually exclusive with an external lr_scheduler,
and exposes train()/eval(), handled uniformly by the trainer's
optimizer_eval_mode + state/resume guards.
This module also exposes validate(args) for the PR-C registry to run a
startup compatibility check.
"""

from __future__ import annotations


def validate(args) -> None:
    """Startup compatibility check: lr_scheduler must be none, otherwise SystemExit."""
    lr_sched_cfg = (getattr(args, "lr_scheduler", "none") or "none").lower()
    if lr_sched_cfg != "none":
        raise SystemExit(
            f"soap_sf (Schedule-Free SOAP) requires lr_scheduler=none "
            f"(Schedule-Free is scheduler-free by construction); got "
            f"lr_scheduler={lr_sched_cfg!r}. Set lr_scheduler=none or pick a "
            f"different optimizer."
        )


def build(args, params, lr: float, weight_decay: float):
    """Build a SOAPScheduleFree instance; reads the soap_* / soap_sf_* args."""
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="soap_sf",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "soap_beta1", 0.9)),
            float(getattr(args, "soap_beta2", 0.95)),
        ),
        shampoo_beta=float(getattr(args, "soap_shampoo_beta", -1.0)),
        precondition_frequency=int(getattr(args, "soap_precondition_frequency", 10)),
        max_precond_dim=int(getattr(args, "soap_max_precond_dim", 10000)),
        precond_in_state=bool(getattr(args, "soap_precond_in_state", True)),
        weight_lr_power=float(getattr(args, "soap_sf_weight_lr_power", 2.0)),
        r=float(getattr(args, "soap_sf_r", 0.0)),
        warmup_steps=int(getattr(args, "soap_sf_warmup_steps", 0)),
    )
