"""optimizer_phase: optimizer dispatch + grad_clip + total_steps + lr_scheduler.

Extracted from main() L344-437 (ADR 0003 PR-B).

Note: optimizer dispatch still keeps the old if-elif style for now (adamw /
prodigy / prodigy_plus_schedulefree); PR-C will replace it with a plugin registry.
"""

from __future__ import annotations

import logging

from training.context import TrainingContext


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """
    - injector.get_param_groups + build_optimizer(args, ...) via training.optimizers
    - validate_optimizer startup constraint checks (e.g. PPSF lr_scheduler=none)
    - grad_clip / trainable_params
    - compute total_steps (min(by_epochs, by_max_steps))
    - build_scheduler(args, optimizer, total_steps) via training.schedulers
    """
    args = ctx.args

    # Optimizer: PR-C will dispatch this through the optimizers/ plugin registry
    ctx.weight_decay = float(getattr(args, "weight_decay", 0.01) or 0.0)
    param_groups = ctx.injector.get_param_groups(ctx.weight_decay)
    ctx.optimizer_type = (getattr(args, "optimizer_type", "adamw") or "adamw").lower()

    # SRA v2: add the projection MLP's params to the optimizer (weight_decay=0, optional separate lr)
    if ctx.sra_aligner is not None:
        sra_groups = ctx.sra_aligner.get_param_groups(lr=None)
        param_groups = param_groups + sra_groups
        logger.info(f"SRA v2: added projection MLP params to optimizer ({sum(p.numel() for g in sra_groups for p in g['params'])/1e6:.1f}M params)")

    from training.optimizers import build_optimizer, validate_optimizer
    validate_optimizer(args)  # PPSF checks startup constraints like lr_scheduler=none
    ctx.optimizer = build_optimizer(args, param_groups, args.learning_rate, ctx.weight_decay)
    if ctx.weight_decay > 0:
        wd_info = f"{ctx.optimizer_type} weight_decay={ctx.weight_decay}"
        if ctx.injector.use_lokr:
            wd_info += " (w1 excluded from weight_decay)"
        logger.info(wd_info)
    ctx.grad_clip = float(getattr(args, "grad_clip_max_norm", 0) or 0)
    if ctx.grad_clip > 0:
        logger.info(f"gradient clipping max_norm={ctx.grad_clip}")
    ctx.trainable_params = [p for group in ctx.optimizer.param_groups for p in group["params"]]

    # Compute the total step count
    try:
        # ceil: a final partial grad_accum group still counts as one update step (the loop
        # steps on the last batch too), matching _accumulation_step's "a short tail group
        # still steps" so the scheduler step count lines up.
        # Note: the NaViT packer's __len__ is the packed-batch count sampled at epoch 0
        # (order-dependent on next-fit/windowed FFD, fluctuates slightly per epoch). The
        # scheduler horizon is built once here and can't be recomputed mid-run, so under
        # navit steps_per_epoch/total_steps are estimates and the LR curve's trough point
        # may be off by a few steps (acceptable); gradient-accumulation boundary accuracy
        # is guaranteed by loop.py refreshing dl_len every epoch.
        _dl_len = len(ctx.dataloader)
        ctx.steps_per_epoch = (_dl_len + args.grad_accum - 1) // args.grad_accum
    except Exception:
        ctx.steps_per_epoch = None

    # total_steps: the step count training will actually reach. The stop condition is
    # "whichever comes first, the epoch cap or max_steps" (see the max_steps break below
    # plus the for-epoch loop's natural exit), so we take the min of both candidates —
    # otherwise the progress bar could show "ran all 100 epochs but only 86%".
    by_epochs = (
        ctx.steps_per_epoch * args.epochs
        if ctx.steps_per_epoch is not None and args.epochs and args.epochs > 0
        else None
    )
    by_max_steps = (
        args.max_steps if (args.max_steps and args.max_steps > 0) else None
    )
    candidates = [c for c in (by_epochs, by_max_steps) if c is not None and c > 0]
    ctx.total_steps = min(candidates) if candidates else None

    logger.info(
        f"dataset size: {len(ctx.dataset)}, steps per epoch: {ctx.steps_per_epoch}, "
        f"total steps: {ctx.total_steps} (by_epochs={by_epochs}, by_max_steps={by_max_steps})"
    )

    # LR scheduler: PR-C will dispatch this through the schedulers/ plugin registry; "none" returns None automatically
    from training.schedulers import build_scheduler
    ctx.scheduler = build_scheduler(args, ctx.optimizer, ctx.total_steps)

    # Timestep sampler (baseline or InfoNoise; N_warm can only be computed once total_steps is known)
    from training.timestep_samplers import build_timestep_sampler
    ctx.timestep_sampler = build_timestep_sampler(args, ctx.total_steps)

    # Adapter weight EMA (ema_start_ratio needs to be scaled by total_steps, hence placed after it)
    from training.ema import build_ema
    ctx.ema = build_ema(args, ctx.trainable_params, ctx.total_steps)
