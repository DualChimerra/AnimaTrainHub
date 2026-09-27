"""finalize_phase: final save + cleanup + final curve + wandb finish after the
training loop ends.

Extracted from main() L886-905 (ADR 0003 PR-B).
"""

from __future__ import annotations

import logging

from training.context import TrainingContext
from training.observability import render_loss_curve
from training.snapshot import emit_event
from utils.optimizer_utils import optimizer_eval_mode


logger = logging.getLogger(__name__)


def _release_block_swap(ctx: TrainingContext) -> None:
    """Training wind-down: remove the block swap hooks and return pinned memory to the system.

    The process is about to exit and the OS would reclaim it anyway, so why release
    it proactively? Because **there's still a tail after exit**: wandb uploads the
    final LoRA (can take tens of seconds to a few minutes), and by then the
    supervisor has already received `eval_training_finished` and queued a
    post-training eval job -- the eval process needs to load a model, but runs
    into this process still holding 11GB+ of page-locked memory (can't even be
    paged out). On memory-constrained machines (exactly the ones block swap
    exists for), this can tank the eval outright.

    Why here and not earlier: during the training loop, pinned memory holds the
    model weights, so releasing it mid-loop would be self-sabotage.
    """
    swap = getattr(ctx, "block_swap", None)
    if swap is None:
        return
    from training.block_swap import release_pinned_host_cache

    freed = getattr(swap, "pinned_bytes", 0)
    try:
        # close() rather than detach(): the pinned master copy is referenced by
        # param.data, and ctx.model isn't the only holder of blocks (injector holds
        # org_module, optimizer holds params) -- hunting down every holder one by
        # one is unreliable, so let each component repoint its own params instead.
        swap.close()
        ctx.block_swap = None
        import gc

        gc.collect()
        release_pinned_host_cache()
        logger.info("block swap released: freed %.2fGB of pinned memory", freed / 1024**3)
    except Exception:  # noqa: BLE001
        logger.debug("block swap release failed (does not affect training results)", exc_info=True)


def run(ctx: TrainingContext) -> None:
    """
    - Write the final LoRA safetensors to disk (PPSF uses averaged weights)
    - Clean up Rich Live / Progress
    - Print the final loss curve
    - wandb finish
    """
    args = ctx.args

    # Final save
    final_path = ctx.output_dir / f"{args.output_name}.safetensors"
    # PPSF: the final output uses averaged weights
    with optimizer_eval_mode(ctx.optimizer):
        ctx.injector.save(final_path)
    # Training is fully done -> supervisor queues the post-training eval (inline /
    # checkpoint-trigger eval has been removed; eval now always runs as an
    # independent job after training).
    emit_event("eval_training_finished", {
        "epoch": int(ctx.current_epoch or 0),
        "step": int(ctx.global_step or 0),
    })

    # Clean up the SRA v2 hook (the MLP isn't saved to the LoRA safetensors, discard once training ends)
    if ctx.sra_aligner is not None:
        ctx.sra_aligner.remove_hooks()
        ctx.sra_aligner = None

    _release_block_swap(ctx)

    # Clean up the progress display
    if ctx.live:
        ctx.live.stop()
    elif ctx.use_rich:
        ctx.progress.stop()

    # Show the final loss curve
    if args.loss_curve_steps and ctx.loss_history:
        chart = render_loss_curve(ctx.loss_history, width=min(80, len(ctx.loss_history)), height=10)
        ctx.emit(f"Loss curve (first {len(ctx.loss_history)} steps):\n{chart}")

    ctx.emit(f"Saved final LoRA: {final_path}")
    ctx.wandb_monitor.upload_model(final_path)
    ctx.wandb_monitor.finish()
    logger.info("Training complete!")
