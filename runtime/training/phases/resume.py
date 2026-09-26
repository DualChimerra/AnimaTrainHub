"""resume_phase: progress init + state recovery + signal registration + step 0 baseline sampling.

Extracted from main() L439-594 (ADR 0003 PR-B).
"""

from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path

from training.bootstrap import init_progress
from training.context import TrainingContext
from training.observability import render_curve_panel
from training.sample_runner import run_sample
from training.snapshot import emit_event
from training.state import load_training_state


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """
    - init_progress + optional Rich Live (incl. loss curve panel)
    - if --resume-state is set: load_training_state + restore monitor's historical loss
    - register SIGINT -> ctx.handle_interrupt
    - prepare the sample_prompts list (multi-character rotation)
    - run baseline sampling when global_step==0 (up to 3 prompts)
    """
    args = ctx.args

    # Initialize the progress display
    ctx.progress, ctx.task_id, progress_kind = init_progress(not args.no_progress, ctx.total_steps)
    ctx.use_rich = progress_kind == "rich"
    ctx.use_plain = ctx.progress == "plain"
    ctx.live = None
    ctx.loss_history = []
    ctx.speed_ema = None

    if ctx.use_rich:
        try:
            from rich.console import Group
            from rich.live import Live
            curve_panel = None
            if args.loss_curve_steps > 0 and not args.no_live_curve:
                curve_panel = render_curve_panel([], width=min(60, args.loss_curve_steps), height=10)
            group = Group(ctx.progress, curve_panel) if curve_panel is not None else Group(ctx.progress)
            ctx.live = Live(group, refresh_per_second=10)
            ctx.live.start()
        except Exception:
            ctx.live = None
            ctx.progress.start()

    # Initial training loop state
    ctx.global_step = 0
    ctx.start_epoch = 0

    # Restore from training state (resume from checkpoint)
    if getattr(args, "resume_state", "") and Path(args.resume_state).exists():
        ctx.start_epoch, ctx.global_step, ctx.loss_history, saved_monitor_state = load_training_state(
            args.resume_state, ctx.injector, ctx.optimizer, ctx.scheduler,
            timestep_sampler=ctx.timestep_sampler,
            sra_aligner=ctx.sra_aligner,
            scaler=ctx.scaler,
            expected_family=ctx.family.spec.family_id,
            ema=ctx.ema,
        )
        ctx.emit(f"Resumed training from checkpoint: epoch={ctx.start_epoch}, step={ctx.global_step}")

        # Restore the monitor panel's historical data (loss curve etc.)
        if ctx.monitor_server and saved_monitor_state:
            try:
                from train_monitor import restore_monitor_state
                restore_monitor_state(
                    losses=saved_monitor_state.get("losses"),
                    lr_history=saved_monitor_state.get("lr_history"),
                    optimizer_metrics_history=saved_monitor_state.get("optimizer_metrics_history"),
                    epoch=ctx.start_epoch,
                    step=ctx.global_step,
                    total_steps=ctx.total_steps,
                )
                ctx.emit(f"Monitor panel history restored: {len(saved_monitor_state.get('losses', []))} loss points")
            except Exception as e:
                ctx.emit(f"Monitor data restore failed: {e}")

        # ADR §`_on_line` recognizes this event and cleans up the previous pause file pair (wired up via PR-3 cmd_builder).
        emit_event("resume_state_loaded", {"path": str(args.resume_state)})

    # Signal handling: handle_interrupt comes from TrainingContext itself, bound
    # cross-platform in two ways (ADR §`runtime/training/phases/resume.py`):
    #   POSIX: SIGINT (CLI Ctrl+C / supervisor `os.kill(pid, SIGINT)`)
    #   Windows: SIGINT is left for CLI Ctrl+C; SIGBREAK catches the supervisor's
    #     CTRL_BREAK_EVENT (a CREATE_NEW_PROCESS_GROUP child process group doesn't
    #     receive CTRL_C_EVENT)
    signal.signal(signal.SIGINT, ctx.handle_interrupt)
    if os.name == "nt":
        # SIGBREAK doesn't exist on POSIX; only register it on Windows
        signal.signal(signal.SIGBREAK, ctx.handle_interrupt)  # type: ignore[attr-defined]

    ctx.current_epoch = ctx.start_epoch
    ctx.model.train()
    # Schedule-Free-family optimizers (PPSF / soap_sf etc.) must start in
    # train_mode: their parameter tensors hold the gradient evaluation point y,
    # not the averaged x. We duck-type rather than hardcode optimizer_type, so
    # new schedule-free variants need zero changes here; AdamW / Prodigy have no
    # .train method and are silently skipped via hasattr.
    if hasattr(ctx.optimizer, "train") and callable(getattr(ctx.optimizer, "train")):
        ctx.optimizer.train()
    # step_start_time is reset inside train_loop itself; not needed here

    # Set up the sample prompt list (supports multi-character rotation)
    ctx.sample_prompts = getattr(args, "sample_prompts", []) or []
    if not ctx.sample_prompts and args.sample_prompt:
        ctx.sample_prompts = [args.sample_prompt]
    ctx.sample_prompt_idx = 0

    # Step 0 initial sampling (baseline result, tests all prompts)
    # Only runs for a fresh training run (global_step == 0); skipped on resume
    sampling_enabled = args.sample_steps > 0 or args.sample_every > 0
    if ctx.global_step == 0 and sampling_enabled:
        ctx.emit("Sampling (step 0, baseline)...")
        for i, prompt in enumerate(ctx.sample_prompts[:3]):  # test at most 3
            sample_path = ctx.sample_dir / f"step_0_baseline_{i}.png"
            run_sample(
                ctx,
                prompt=prompt,
                sample_path=sample_path,
                wandb_key="samples/baseline",
                wandb_caption=f"step 0 baseline {i}: {prompt}",
                wandb_step=0,
                seed_offset=i,
            )
    elif ctx.global_step > 0 and sampling_enabled:
        ctx.emit(f"Skipping startup baseline sampling (resumed from step {ctx.global_step}, not step 0)")

    # ADR §8.1 is_pausable signal: once the resume phase finishes -> training
    # enters the main loop -> the user may pause. When the supervisor's
    # `_on_line` receives this event it sets slot.train_loop_started = True,
    # which dispatches is_pausable=True over SSE to unlock the UI's pause button.
    emit_event("train_loop_started", {"global_step": ctx.global_step})
