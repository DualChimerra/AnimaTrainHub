"""Training state save/restore (resume from checkpoint).

Extracted from the original runtime/anima_train.py L1073-1142 (ADR 0003 PR-A).
Imported directly by tests/test_lycoris_resume.py.

Public:
- save_training_state -- saves a one-shot ckpt of LoRA / optimizer / scheduler / rng / monitor
- load_training_state -- restores it, returns (epoch, global_step, loss_history, monitor_state)
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


def save_training_state(
    path, injector, optimizer, epoch, global_step,
    loss_history=None, rng_state=None, monitor_state=None,
    scheduler=None, timestep_sampler=None, sra_aligner=None,
    scaler=None, model_family=None, ema=None,
):
    """Save the full training state, to support resuming from checkpoint.

    timestep_sampler (ADR 0006 Addendum 1): EMA / CDF / FIFO buffer for
    adaptive samplers (InfoNoise). A stateless sampler's (baseline)
    state_dict() is {}, and is skipped so the ckpt file doesn't grow for
    nothing.
    """
    state = {
        "lora_state_dict": injector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss_history": loss_history or [],
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "random": random.getstate(),
        },
        "monitor_state": monitor_state,  # save the monitor panel data (used to restore the loss curve)
        # Multi-model D13: family marker; the load side fail-fasts across families (strict=False would silently cold-start)
        "model_family": str(model_family or "anima"),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if sra_aligner is not None and hasattr(sra_aligner, "state_dict"):
        try:
            state["sra_aligner_state"] = sra_aligner.state_dict()
        except Exception as e:
            logger.warning(f"sra_aligner.state_dict() failed (skipped): {e}")
    if timestep_sampler is not None and hasattr(timestep_sampler, "state_dict"):
        # hasattr defensiveness: the Protocol provides no default dispatch,
        # so if a future sampler forgets to implement these two hooks, skip
        # silently rather than crash (8 hours of training must not be lost
        # over a missing resume hook)
        try:
            sampler_state = timestep_sampler.state_dict()
        except Exception as e:
            logger.warning(f"timestep_sampler.state_dict() failed (skipped): {e}")
            sampler_state = None
        if sampler_state:  # an empty dict (baseline) is not stored
            state["timestep_sampler_state"] = sampler_state
    if ema is not None and hasattr(ema, "state_dict"):
        # The shadow weights must be saved along with the rest of the state:
        # otherwise EMA restarts from the current weights after resume, the
        # averaging done so far is wasted, and the meaning of the produced
        # *_ema file quietly changes.
        try:
            state["ema_state"] = ema.state_dict()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ema.state_dict() failed (skipped): {e}")
    if scaler is not None:
        # fp16 GradScaler's scale factor / growth tracker. Without it on
        # resume, it resets to the default 2^16, and the first few steps
        # overflow and get skipped again until it reconverges. For bf16/fp32,
        # ctx.scaler is None and is skipped.
        state["scaler_state"] = scaler.state_dict()
    # ADR 0006 Addendum 2: atomic write via tmp + os.replace.
    # auto_epoch_state.pt is a single overwrite-in-place recovery point; a
    # direct torch.save that gets hit by a power loss / hard kill during the
    # write window would leave the only recovery point half-written. Writing
    # a sibling tmp file first and then renaming means the old file either
    # stays fully intact or is fully replaced by the new one.
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info(f"Training state saved: {path} (epoch={epoch}, step={global_step})")


def load_training_state(path, injector, optimizer, scheduler=None, timestep_sampler=None, sra_aligner=None, scaler=None, expected_family=None, ema=None):
    """Load training state, returns (epoch, global_step, loss_history, monitor_state).

    timestep_sampler (ADR 0006 Addendum 1): if the ckpt contains
    timestep_sampler_state and the sampler implements load_state_dict, pour
    the EMA / CDF / FIFO back in; otherwise stay cold-started (with a warning).
    """
    logger.info(f"Loading training state: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)

    # Multi-model D13: fail-fast on cross-family resume. Existing state with no marker is grandfathered in as anima.
    saved_family = str(state.get("model_family") or "anima")
    if expected_family is not None and saved_family != str(expected_family):
        raise RuntimeError(
            f"Cross-model-family resume rejected: the recovery point belongs to '{saved_family}', "
            f"current model_family='{expected_family}'. "
            f"(a strict=False load would silently turn into an all-missing cold start, which is worse than crashing)"
        )

    # Load LoRA weights (lycoris-lora backend) -- import the state_dict in one shot.
    # Old self-implemented ckpts are **not migrated**, per the Stage 4 plan
    # decision; strict=False lets missing keys take the default
    # initialization path instead of crashing; users should train a new
    # ckpt in the new format from scratch.
    lora_sd = state["lora_state_dict"]
    result = injector.load_state_dict(lora_sd, strict=False)
    missing = len(getattr(result, "missing_keys", [])) if hasattr(result, "missing_keys") else 0
    unexpected = len(getattr(result, "unexpected_keys", [])) if hasattr(result, "unexpected_keys") else 0
    if missing or unexpected:
        logger.warning(
            f"resume LoRA: missing={missing}, unexpected={unexpected} (old-format ckpt?)"
        )

    # Restore the SRA v2 projection MLP (if enabled). Must happen before the
    # optimizer state is restored, to guarantee subsequent training
    # continues from the same projection weights rather than a fresh random
    # MLP paired with old optimizer moments.
    if sra_aligner is not None:
        if "sra_aligner_state" in state and hasattr(sra_aligner, "load_state_dict"):
            try:
                sra_aligner.load_state_dict(state["sra_aligner_state"])
                logger.info("SRA v2 projection MLP state restored")
            except Exception as e:
                logger.warning(f"SRA v2 projection MLP state restore failed (cold start): {e}")
        else:
            logger.warning("checkpoint has no SRA v2 state; projection MLP will cold-start")

    # Load optimizer state
    optimizer.load_state_dict(state["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in state:
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"Scheduler state restore failed (will start over): {e}")

    # Restore GradScaler (fp16). Old ckpts / bf16-fp32 runs have no such key -> stays at the default cold-start scale.
    if scaler is not None and "scaler_state" in state:
        try:
            scaler.load_state_dict(state["scaler_state"])
            logger.info("GradScaler state restored")
        except Exception as e:
            logger.warning(f"GradScaler state restore failed (cold-starting at the default scale): {e}")

    # Restore EMA shadow (old ckpt / EMA only just enabled this run -> no such key, the shadow is rebuilt on the next update)
    if ema is not None and hasattr(ema, "load_state_dict") and "ema_state" in state:
        try:
            ema.load_state_dict(state["ema_state"])
            logger.info("EMA shadow weights restored")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"EMA shadow restore failed (rebuilding it): {e}")

    # Restore RNG state
    if "rng_state" in state:
        rng = state["rng_state"]
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
        if rng.get("random") is not None:
            random.setstate(rng["random"])

    # Restore the timestep sampler's internal state (InfoNoise CDF / EMA / FIFO etc.; a no-op for baseline)
    if (
        timestep_sampler is not None
        and "timestep_sampler_state" in state
        and hasattr(timestep_sampler, "load_state_dict")
    ):
        try:
            timestep_sampler.load_state_dict(state["timestep_sampler_state"])
            logger.info("timestep_sampler state restored (adaptive schedule continuing)")
        except Exception as e:
            logger.warning(f"timestep_sampler state restore failed (cold-starting, will re-warmup): {e}")

    # ADR 0006 Addendum 1, item 7: resume guard for Schedule-Free optimizers (PPSF etc.).
    # PPSF internally keeps a group['train_mode'] flag + Polyak-averaged x/y/z
    # weight sets; load_state_dict restores train_mode to the value it had
    # at save time (save happens inside `optimizer_eval_mode`, i.e.
    # train_mode False) -> the first step() after resume raises
    # "Not in train mode!". Explicitly calling .train() once:
    # set_train_mode(True) lerps p.data back from averaged x to y and sets
    # train_mode=True, aligning it with the dev training loop's starting
    # state. Spike-tested to be bit-exact against ground truth over 2000
    # steps (no drift). AdamW / Prodigy have no .train method, so hasattr
    # silently skips them.
    if hasattr(optimizer, "train") and callable(getattr(optimizer, "train")):
        try:
            optimizer.train()
        except Exception as e:
            logger.warning(f"optimizer.train() call failed (PPSF may be broken): {e}")

    epoch = state.get("epoch", 0)
    global_step = state.get("global_step", 0)
    loss_history = state.get("loss_history", [])
    monitor_state = state.get("monitor_state", None)  # restore monitor data

    logger.info(f"Training state restored: epoch={epoch}, step={global_step}")
    return epoch, global_step, loss_history, monitor_state
