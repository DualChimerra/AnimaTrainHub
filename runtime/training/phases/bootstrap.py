"""bootstrap_phase: args + yaml + interactive mode + seed + device/dtype + output dir + wandb + monitor_state.

Extracted from main() L113-185 (ADR 0003 PR-B).
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import torch

from training.bootstrap import apply_yaml_config, ensure_dependencies, load_yaml_config
from training.cli import prompt_for_args
from training.context import TrainingContext
from training.observability import init_wandb_monitor


logger = logging.getLogger(__name__)


def _maybe_apply_pause_snapshot(args, resume_state_path: Path) -> None:
    """Read a pause snapshot and override args with it (ADR 0006 PR-3 / §5.7).

    args.resume_state = `.../pause_step_<N>.pt` -> snapshot = `.../pause_step_<N>.config.json`.
    Snapshot doesn't exist -> silently skip (the old path where the user picks a
    periodic save file via ResumeFieldPicker to start a new task).

    Override rules:
    - Every field in snapshot["args"] is written to the args namespace, **except**:
      - `resume_state` is not overridden (the snapshot recorded args from before the
        pause, when resume_state was empty; we're only using it now to resume)
      - `config` is not overridden (the snapshot recorded the user's yaml path at the
        time, which may have since been deleted/renamed)
    - snapshot["sample_prompts"] -> args.sample_prompts (resume_phase reads this)
    """
    snapshot_path = resume_state_path.with_suffix(".config.json")
    if not snapshot_path.exists():
        return  # not a pause state, keep the existing args
    try:
        raw = snapshot_path.read_text(encoding="utf-8")
        snapshot = json.loads(raw)
    except Exception as exc:
        logger.warning(
            f"failed to read pause snapshot, keeping existing args: {snapshot_path} ({exc})"
        )
        return
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("args"), dict):
        logger.warning(f"unrecognized pause snapshot schema, keeping existing args: {snapshot_path}")
        return
    logger.info(f"loaded pause snapshot, overriding training params: {snapshot_path}")
    snap_args: dict = snapshot["args"]
    skipped = {"resume_state", "config"}
    for k, v in snap_args.items():
        if k in skipped:
            continue
        setattr(args, k, v)
    sp = snapshot.get("sample_prompts")
    if isinstance(sp, list):
        args.sample_prompts = sp


def _resolve_sample_seed(args) -> None:
    """sample_seed=0 -> draw one random value at training start, write it back to args, and log it.

    Why: when sample_seed=0, sample_runner never calls torch.manual_seed, so the
    whole batch's sampling drifts with the global RNG -- the same prompt produces
    a different image across epochs, making it impossible to tell whether the
    model converged or the noise just changed. Drawing it once and fixing it
    means the same prompt gets the same seed for the whole training run.

    Interaction with the pause snapshot: the snapshot writes the full args.dict(),
    so the resolved value gets frozen; on resume, _maybe_apply_pause_snapshot
    restores it, so the same seed carries across a pause. If the user starts a
    fresh task and the yaml still has 0, a new random value is drawn at startup.
    """
    if int(getattr(args, "sample_seed", 0) or 0):
        return
    args.sample_seed = random.randint(1, 2**31 - 1)
    logger.info(f"sample_seed=0 -> using random seed for training: {args.sample_seed}")


def _prepend_trigger_to_sample_prompts(args) -> None:
    """trigger_word non-empty -> prepend it to every sample_prompt / sample_prompts entry.

    Matches the caption side's behavior (tag_worker also writes the trigger as the
    first tag): training sample images always carry the trigger, giving a direct
    visual check of whether the LoRA is active. "Already contains the trigger" is
    determined via token-level matching (split on comma, compare case-insensitively),
    avoiding false positives from substring matches. Empty prompts are not injected,
    to avoid producing a broken ``"trigger, "`` string.
    """
    trigger = (getattr(args, "trigger_word", "") or "").strip()
    if not trigger:
        return
    lower = trigger.lower()

    def _contains(prompt: str) -> bool:
        return any(t.strip().lower() == lower for t in prompt.split(",") if t.strip())

    sp = getattr(args, "sample_prompt", "") or ""
    if sp and not _contains(sp):
        args.sample_prompt = f"{trigger}, {sp}"

    sps = getattr(args, "sample_prompts", None) or []
    if sps:
        args.sample_prompts = [
            (f"{trigger}, {p}" if (p and not _contains(p)) else p) for p in sps
        ]


def run(ctx: TrainingContext) -> None:
    """Finish all pre-training prep that isn't model/data related:

    - Load the yaml config (if any) + fill missing fields in interactive mode
    - ensure_dependencies
    - Set the seed / pick device / dtype
    - Create output_dir + sample_dir
    - Initialize wandb_monitor + the monitor_state.json writer
    """
    args = ctx.args

    # PR-C: validate schema consistency across all plugin subpackages at startup, so a misconfig doesn't surface halfway through a run
    from training.adapters import validate_schema_consistency as _validate_adapters
    from training.losses import validate_schema_consistency as _validate_losses
    from training.optimizers import validate_schema_consistency as _validate_optimizers
    from training.schedulers import validate_schema_consistency as _validate_schedulers
    _validate_adapters()
    _validate_optimizers()
    _validate_schedulers()
    _validate_losses()

    # Load the YAML config file + normalize into TrainingConfig (cut 1 / R1). The
    # plain-CLI path with no yaml also goes through this: parse_args's sparse
    # namespace is missing the schema defaults, which apply_yaml_config fills in
    # uniformly via pydantic construction (migration / family overlay / validation all apply at once).
    config = {}
    if args.config:
        logger.info(f"loading config file: {args.config}")
        ctx.config_path = Path(args.config).resolve()
        ctx.config_dir = ctx.config_path.parent
        config = load_yaml_config(args.config)
    ctx.args = apply_yaml_config(args, config)
    args = ctx.args

    # The bridge already auto-generates --prefer-json / --no-prefer-json for the
    # prefer_json bool; no extra compat handling needed here.

    # ADR 0006 PR-3: a .config.json snapshot next to the pause file overrides args.
    # Trigger condition: the .pt pointed to by args.resume_state has a same-prefix
    # .config.json next to it. Only pause-triggered state carries a snapshot
    # (written by PR-2's handle_interrupt); a periodic save has no snapshot, and
    # starting a new task via ResumeFieldPicker takes the original path (the
    # user's current yaml config). Snapshot freezing is the core of ADR §5.7 --
    # on resume, the task's training params strictly use the values from the
    # moment it was paused, fully decoupled from any later version/preset/yaml changes by the user.
    if getattr(args, "resume_state", None):
        _maybe_apply_pause_snapshot(args, Path(args.resume_state))
        ctx.args = args

    # Interactive mode check
    required = [args.data_dir, args.transformer_path, args.vae_path, args.text_encoder_path]
    if args.interactive or any(not x for x in required):
        ctx.args = prompt_for_args(args)
        args = ctx.args

    # Multi-model PR-2b: fail-fast family resolution (after args are finalized,
    # before any weight loading; an unknown model_family is fatal here. Since the
    # pause snapshot already froze args, family consistency across a pause comes for free.)
    # Capability validation isn't done separately anymore (cut 1 / R1):
    # apply_yaml_config's TrainingConfig construction already runs
    # _validate_family_capabilities, so the direct-CLI path shares the same guard as Studio.
    from training.families import resolve_family

    ctx.family = resolve_family(args)

    # Audit #2 (design doc §10.1): T-LoRA's rank mask is generated from the
    # batch-mean timestep, so at batch>1 the per-sample "high noise -> low rank"
    # behavior degrades into a batch-mean approximation -- not blocked (a hard
    # block would unfairly hit users who want small batches), just an explicit startup warning.
    if getattr(args, "lora_type", "") == "tlora" and int(getattr(args, "batch_size", 1)) > 1:
        logger.warning(
            "T-LoRA combined with batch_size=%s: the rank mask is generated from the batch-mean "
            "timestep, so the per-sample mask degrades into a batch-mean approximation; "
            "use batch_size=1 to get the paper's exact behavior",
            args.batch_size,
        )

    # Trigger word injection: the caption side's tag_worker writes the trigger as
    # the first tag; here we inject it into sample_prompt(s) the same way, so
    # sample images naturally carry the trigger. The pause snapshot already froze
    # trigger_word (stored in args), and resume normalizes it again idempotently the same way.
    _prepend_trigger_to_sample_prompts(args)
    ctx.args = args

    # Dependency check
    ensure_dependencies(auto_install=args.auto_install)

    # Deferred import: preserves the original main() order -- numpy/PIL can only be imported after ensure_dependencies
    import numpy as np

    # Set the random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    _resolve_sample_seed(args)

    ctx.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mixed_precision == "bf16":
        ctx.dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        ctx.dtype = torch.float16
        ctx.scaler = torch.cuda.amp.GradScaler()
    else:
        ctx.dtype = torch.float32
    # VAE precision is decoupled from training precision: under fp16, the VAE still uses
    # fp32 (see TrainingContext.vae_dtype); under bf16/fp32, the VAE follows the main precision unchanged.
    ctx.vae_dtype = torch.float32 if ctx.dtype == torch.float16 else ctx.dtype

    # Create the output directory
    ctx.output_dir = Path(args.output_dir)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    # Sample images land under the task archive root's samples/. The supervisor
    # injects `--monitor-state-file <studio_data>/tasks/<id>/monitor/state.json`
    # per task; sample_dir takes its `samples/` sibling one level up --
    # `tasks/<id>/samples/`, alongside monitor/ -- so the whole set
    # (snapshot/ monitor/ samples/ run.log) forms the task's complete archive.
    # If --monitor-state-file wasn't passed (plain CLI training / compat with an
    # older injection path), falls back to output_dir/samples; samples.py can
    # still search several candidate locations around monitor_dir.
    _msf = getattr(args, "monitor_state_file", None)
    ctx.task_archive_dir = Path(_msf).parent.parent if _msf else None
    ctx.sample_dir = (ctx.task_archive_dir / "samples") if ctx.task_archive_dir else (ctx.output_dir / "samples")
    ctx.sample_dir.mkdir(parents=True, exist_ok=True)
    # ADR 0006 Addendum 2: auto_epoch_state.pt is likewise part of the task
    # archive -- tasks/<id>/state/, sharing a root with samples/. If
    # --monitor-state-file wasn't passed (plain CLI) -> None, and
    # ctx.auto_state_dir() falls back to output_dir/state/task_<id>/ (unchanged behavior).
    ctx.task_archive_state_dir = (ctx.task_archive_dir / "state") if ctx.task_archive_dir else None
    # The supervisor injects the queue task id via the LORA_TASK_ID env var when
    # starting training (ADR 0006). Used by ctx.state_dir() to compute the
    # per-task state subdirectory; falls back to "unknown" if the env var is absent.
    _env_tid = os.environ.get("LORA_TASK_ID")
    if _env_tid:
        try:
            ctx.lora_task_id = int(_env_tid)
        except ValueError:
            logger.warning(f"LORA_TASK_ID={_env_tid!r} is not an int, treating as unknown")
    ctx.wandb_monitor = init_wandb_monitor(args, ctx.output_dir, ctx.config_path)

    # Loss function (mse / huber; dispatched through the losses/ plugin registry)
    # Doesn't depend on total_steps, unlike timestep_sampler/scheduler; placed in
    # bootstrap rather than the optimizer phase to avoid an architectural mismatch.
    from training.losses import build_loss
    ctx.loss_fn = build_loss(args)

    # Training monitor state writer (PP6.1): always on; the file path comes
    # preferentially from --monitor-state-file, otherwise falls back to
    # output_dir/monitor_state.json. The Studio frontend reads this file via
    # /api/state?task_id=, so the training side no longer starts an HTTP server (Studio itself is the monitor).
    ctx.monitor_server = True  # kept for the branch check below; actually means "write the state file"
    try:
        from train_monitor import set_state_file, update_monitor
        state_path = (
            Path(args.monitor_state_file)
            if getattr(args, "monitor_state_file", None)
            else ctx.output_dir / "monitor_state.json"
        )
        set_state_file(state_path)
        update_monitor(
            total_epochs=int(args.epochs or 0),
            config={
                "model": {"lokr": "Anima LoKr"}.get(args.lora_type, "Anima LoRA"),
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.learning_rate,
                "resolution": args.resolution,
                "data_dir": str(args.data_dir),
            },
        )
        logger.info(f"📊 training monitor state file: {state_path}")
    except Exception as e:
        logger.warning(f"monitor state writer init failed: {e}")
        ctx.monitor_server = None
