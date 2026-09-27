"""TrainingContext: the shared state bundle used by all phases (ADR 0003 PR-B).

Collects every local variable from the original 793-line main() in
runtime/anima_train.py into one dataclass, so phase functions can follow a
pipeline style of take ctx -> mutate -> return None.

Design principles:
- Field types are explicit; late-populated fields use `Optional[X] = None`
- Logic with closures (progress display, signal handling, etc.) is collected
  into methods on this class (emit / handle_interrupt / get_next_sample_prompt),
  avoiding the nonlocal closures that main() used
- Holds no "input" beyond args -- any yaml / cli behavior is merged into args
  first, before any phase starts

Every phase function has the signature: `def run(ctx: TrainingContext) -> None`
(in-place mutation).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from training.losses.protocol import LossProtocol


@dataclass
class TrainingContext:
    # --- filled by bootstrap_phase ---
    args: Any  # argparse.Namespace
    family: Any = None  # ModelFamily (multi-model PR-2b; produced by resolve_family(args))
    config_path: Optional[Path] = None
    config_dir: Optional[Path] = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    # VAE working precision, independent of the training dtype. Even in fp16
    # training the VAE still runs in fp32: cards that need fp16 (V100/Turing
    # etc.) have no bf16, and fp16 VAE encode/decode overflows to NaN easily
    # (black image / all-NaN latent -> the whole training step gets skipped).
    # bootstrap derives it from dtype; kept aligned with ComfyUI's vae_dtype()
    # and the generation side's vae_precision.
    vae_dtype: torch.dtype = torch.float32
    output_dir: Optional[Path] = None
    sample_dir: Optional[Path] = None
    wandb_monitor: Any = None         # observability.WandBMonitor
    monitor_server: Optional[bool] = None  # legacy name kept for compat: True = monitor_state.json writes are active
    # The supervisor injects the queue task id via env LORA_TASK_ID when it
    # starts training; when run straight from the CLI the env var is absent
    # -> None -> state_dir() falls back to a task_unknown subdirectory.
    # Note: this is unrelated to the progress bar's task_id field (~line 80);
    # deliberately named differently to avoid confusion.
    lora_task_id: Optional[int] = None
    # Root of the task archive (studio_data/tasks/<id>/). bootstrap derives it
    # by going up two levels from --monitor-state-file; plain CLI runs that
    # don't pass it get None. samples/, state/ and the prompt text cache
    # (.text-cache/) all live under this root.
    task_archive_dir: Optional[Path] = None
    # ADR 0006 Addendum 2: auto_epoch_state.pt lives in the task archive
    # (studio_data/tasks/<id>/state/). bootstrap fills this in after deriving
    # the archive root from --monitor-state-file (same convention as
    # sample_dir); plain CLI runs without it get None -> auto_state_dir()
    # falls back to state_dir().
    task_archive_state_dir: Optional[Path] = None

    # --- filled by models_phase ---
    repo_root: Optional[Path] = None
    model: Any = None
    vae: Any = None
    # Text encoder holder (family-private structure; for Anima it's
    # (qwen_model, qwen_tok, t5_tok)); opaque to the loop -- multi-model
    # PR-2b D15, replaces the original three separate qwen_model/qwen_tok/t5_tok fields
    text_stack: Any = None
    injector: Any = None

    # --- filled by dataset_phase ---
    bucket_mgr: Any = None
    base_dataset: Any = None
    dataset: Any = None
    reg_dataset: Any = None
    use_cached: bool = False
    dataloader: Any = None

    # --- filled by optimizer_phase ---
    weight_decay: float = 0.0
    optimizer: Any = None
    optimizer_type: str = "adamw"
    grad_clip: float = 0.0
    trainable_params: list = field(default_factory=list)
    steps_per_epoch: Optional[int] = None
    total_steps: Optional[int] = None
    # For progress display only: total steps corrected per-epoch from the
    # actual number of packed batches (under navit packing, total_steps is an
    # epoch-0 snapshot that can drift). Only feeds monitor/CLI progress;
    # scheduler/adapter still use total_steps.
    total_steps_display: Optional[int] = None
    scheduler: Any = None
    timestep_sampler: Any = None    # training.timestep_samplers.TimestepSamplerProtocol
    loss_fn: Optional["LossProtocol"] = None
    sra_aligner: Any = None         # training.families.anima.sra_align.SRAAligner (optional)
    block_swap: Any = None          # training.block_swap.PinnedBlockSwap (optional)
    ema: Any = None                 # training.ema.AdapterEma (optional)

    # --- filled by resume_phase ---
    global_step: int = 0
    start_epoch: int = 0
    current_epoch: int = 0
    loss_history: list = field(default_factory=list)
    speed_ema: Optional[float] = None
    progress: Any = None
    task_id: Any = None
    use_rich: bool = False
    use_plain: bool = False
    live: Any = None
    sample_prompts: list = field(default_factory=list)
    sample_prompt_idx: int = 0
    interrupted: bool = False

    # --- loop.py epoch backup (ADR 0006 Addendum 1, option Delta) ---
    # These two fields are filled after auto_epoch_state.pt is overwritten at
    # the end of each epoch. handle_interrupt reads them to emit the
    # pause_state event; None means the first epoch hasn't finished yet ->
    # the supervisor marks the task canceled rather than paused (no
    # resumable progress).
    last_auto_epoch_state_path: Optional[Path] = None
    last_auto_epoch_config_path: Optional[Path] = None
    scaler: Any = None  # torch.cuda.amp.GradScaler, non-None only in fp16

    # --- shared methods ---

    def state_dir(self) -> Path:
        """Directory where the user's periodic saves (save_state_every*)
        write state, isolated per task.

        ADR 0006 SS5.3: multiple tasks running under the same version
        overwriting each other's state files was a latent bug; a task_id
        subdirectory was added to isolate them. When env LORA_TASK_ID isn't
        set (plain CLI run), falls back to task_unknown/.
        """
        assert self.output_dir is not None, "state_dir() called before bootstrap_phase"
        tid = self.lora_task_id if self.lora_task_id is not None else "unknown"
        d = self.output_dir / "state" / f"task_{tid}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def auto_state_dir(self) -> Path:
        """Directory where auto_epoch_state.pt (the system-level recovery
        point) is written (ADR 0006 Addendum 2).

        Kept separate from the user's periodic saves: the auto backup is part
        of the task archive (alongside run.log / monitor/ / samples/), living
        under `studio_data/tasks/<id>/state/` -- its lifecycle is tied to the
        task row (deleting the task cleans it up too), and the server can
        derive the path directly from the task id. The user's periodic saves
        are user output and stay in state_dir() (the version output tree,
        which ResumeFieldPicker scans by version).

        Plain CLI runs (without --monitor-state-file) fall back to
        state_dir(), unchanged behavior.
        """
        if self.task_archive_state_dir is not None:
            self.task_archive_state_dir.mkdir(parents=True, exist_ok=True)
            return self.task_archive_state_dir
        return self.state_dir()

    def emit(self, msg: str) -> None:
        """Print a user-facing message, routed by the current progress
        display mode.

        Ported from the emit closure in the original main(); behavior is
        identical.
        """
        if self.use_plain:
            print()
        if self.live:
            self.live.console.print(msg)
        elif self.use_rich:
            self.progress.console.print(msg)
        else:
            print(msg)

    def get_next_sample_prompt(self) -> str:
        """Get the next sampling prompt (round-robin; returns a default if
        sample_prompts is empty)."""
        if not self.sample_prompts:
            return "1girl, masterpiece"
        prompt = self.sample_prompts[self.sample_prompt_idx % len(self.sample_prompts)]
        self.sample_prompt_idx += 1
        return prompt

    def handle_interrupt(self, sig, frame) -> None:
        """Pause / Ctrl+C signal handler (ADR 0006 Addendum 1, option Delta):
        Pause = Cancel + release the GPU immediately.

        Signal sources:
          - CLI Ctrl+C: POSIX SIGINT / Windows SIGBREAK (registered by the resume phase)
          - Supervisor pause: Windows CTRL_BREAK_EVENT / POSIX SIGINT

        New flow (no more mid-epoch save):
          1. wandb finish (so all I/O is done by the time the supervisor reads the event)
          2. emit __EVENT__:pause_state, with state_path pointing at the **most
             recent end-of-epoch** auto_epoch_state.pt (overwritten at the end
             of every epoch by loop.py, tracked via ctx.last_auto_epoch_state_path)
          3. During the first epoch (last_auto_epoch_state_path is None) ->
             emit state_path=None, so the supervisor takes the cancel branch
             (ADR 0006 Addendum 1, decision point 3)
          4. sys.exit(0)

        Why mid-epoch save was dropped (see the three-way audit in the ADR
        Addendum 1 for details):
          - grad_accum boundaries aren't respected -> partial backward grads left dangling
          - dataloader progress isn't saved -> resuming double-trains the first 5% (skews the Prodigy d estimate)
          - current_epoch semantics become ambiguous (mid-epoch path keeps epoch / epoch-end path keeps epoch+1)
          - InfoNoise / cosine restart T_cur drifts
          - This is what actually matches the product semantics of "pause = release the GPU immediately"

        Triggering it again while already interrupted = hard exit.
        """
        # Deferred import to avoid a circular dependency
        from training.snapshot import emit_event

        if self.interrupted:
            self.emit("Force quitting...")
            sys.exit(1)
        self.interrupted = True
        self.emit("\nPause signal detected, exiting (keeping the most recent epoch backup for resume)...")
        try:
            self.wandb_monitor.finish()
        except Exception:
            pass
        # emit happens after wandb finish -- so all I/O is done by the time the supervisor reads the event.
        emit_event("pause_state", {
            "state_path": str(self.last_auto_epoch_state_path) if self.last_auto_epoch_state_path else None,
            "config_path": str(self.last_auto_epoch_config_path) if self.last_auto_epoch_config_path else None,
            "step": self.global_step,
        })
        if self.last_auto_epoch_state_path:
            self.emit(f"Paused! Resume point: {self.last_auto_epoch_state_path}")
        else:
            self.emit("The first epoch never finished, no auto backup to resume from -> the task will be marked canceled.")
        sys.exit(0)
