"""Periodic sampling helper -- eliminates 3 near-line-for-line duplicated sample
blocks from the original main().

Extracted from main() L550-594 / L757-795 / L840-872 (ADR 0003 PR-B + memory P0).

Public:
- run_sample -- a single sample: reads args.sample_* parameters + calls
  sample_image + saves + wandb + monitor. All callers share the PPSF
  averaged-weights switch, the model.eval/train bracketing, and the
  exception fallback logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from training.context import TrainingContext
from utils.optimizer_utils import optimizer_eval_mode


logger = logging.getLogger(__name__)


def run_sample(
    ctx: TrainingContext,
    *,
    prompt: str,
    sample_path: Path,
    wandb_key: Optional[str] = None,
    wandb_caption: Optional[str] = None,
    wandb_step: Optional[int] = None,
    seed_offset: int = 0,
) -> None:
    """Run one sample and save it to sample_path.

    - PPSF: samples using the averaged weights during training, then switches back to training weights afterward
    - Exception fallback: a sampling error must not interrupt training, only logs a warning
    - wandb: log_image is called if wandb_key is given; caption / step are passed along too
    - monitor_state.json: always tries to push sample_path to the frontend preview
    - seed_offset: in baseline mode, offset by i so multiple test prompts produce different images

    sample_path must be decided by the caller (baseline numbering / step /
    epoch is not this function's concern).
    """
    args = ctx.args
    # resolution may be a list (multi-resolution); the sample preview uses the first (base) resolution.
    _res = args.resolution
    base_reso = int(_res[0]) if isinstance(_res, (list, tuple)) and _res else int(_res)
    s_w = int(getattr(args, "sample_width", 0) or 0) or base_reso
    s_h = int(getattr(args, "sample_height", 0) or 0) or base_reso
    # Must be a multiple of align_px (VAE stride 8 x patch_spatial 2 = 16),
    # otherwise cosmos_predict2's spatial_patch assertion fails
    spec = ctx.family.spec
    _align = spec.latent.align_px
    s_w = max(_align, (s_w // _align) * _align)
    s_h = max(_align, (s_h // _align) * _align)
    s_cfg_value = getattr(args, "sample_cfg_scale", spec.sampling.default_cfg)
    s_cfg = float(spec.sampling.default_cfg if s_cfg_value is None else s_cfg_value)
    s_neg = str(getattr(args, "sample_negative_prompt", "") or "")
    s_seed = int(getattr(args, "sample_seed", 0) or 0)
    s_steps = int(
        getattr(args, "sample_infer_steps", spec.sampling.default_steps)
        or spec.sampling.default_steps
    )
    s_sampler = str(
        getattr(args, "sample_sampler_name", spec.sampling.default_sampler)
        or spec.sampling.default_sampler
    )
    s_sched = str(
        getattr(args, "sample_scheduler", spec.sampling.default_scheduler)
        or spec.sampling.default_scheduler
    )

    # T-LoRA: matches ControlGenAI/T-LoRA's official inference -- the sample
    # stage should not apply the timestep mask. The official inferencer
    # doesn't pass sigma_mask, and the forward falls back to an all-ones
    # mask internally = full-rank inference. This explicitly clears the
    # training-state PR's mask; the next on_step_begin will write it again
    # from sigma_t, so no restore is needed afterward.
    # Non-tlora adapters (lokr / loha / lora) don't have this method; getattr returns None and it's skipped safely.
    clear_fn = getattr(ctx.injector, "clear_timestep_mask", None)
    if callable(clear_fn):
        clear_fn()

    was_training = bool(getattr(ctx.model, "training", True))
    # Return allocator fragments accumulated during training (reserved
    # segments from multiple bucket shapes) before sampling. The sample
    # resolution != the training bucket shape -> sampling activations are a
    # fresh allocation; without cleanup, the momentary commit of "training
    # reserved + new sampling segment" can blow past the dedicated VRAM
    # limit, which makes WDDM demote training tensors to shared memory
    # (= system RAM) -- after which every training step goes over PCIe,
    # showing up as sustained stutter for the whole machine after sampling
    # (while the VRAM/RAM numbers look stable). Settling and resetting the
    # peak of "the previous training segment" here, right before sampling,
    # means the number reported after sampling is the pure sampling-period
    # peak -- the two numbers together answer "is this card enough".
    _log_vram_watermark("before sampling", peak_label="training step")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        with optimizer_eval_mode(ctx.optimizer):
            ctx.model.eval()
            if s_seed:
                torch.manual_seed(s_seed + seed_offset)
            img = ctx.family.sample_image(
                ctx.model, ctx.vae, ctx.text_stack,
                prompt, height=s_h, width=s_w, steps=s_steps, cfg_scale=s_cfg,
                negative_prompt=s_neg,
                sampler_name=s_sampler,
                scheduler=s_sched,
                device=ctx.device, dtype=ctx.dtype,
                seed=(s_seed + seed_offset) if s_seed else None,
            )
            img.save(sample_path)
            ctx.emit(f"Sample saved: {sample_path.name}")
            if wandb_key and ctx.wandb_monitor.log_samples:
                ctx.wandb_monitor.log_image(
                    wandb_key,
                    sample_path,
                    caption=wandb_caption or prompt,
                    step=wandb_step,
                )
            if ctx.monitor_server:
                try:
                    from train_monitor import update_monitor
                    update_monitor(sample_path=sample_path)
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("Sampling failed, skipped this preview, training not interrupted: %s", exc, exc_info=True)
    finally:
        if was_training:
            ctx.model.train()
        else:
            ctx.model.eval()
        # Return the new shape's allocation from the sampling period, so the allocator is clean when training resumes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_vram_watermark("after sampling", peak_label="sampling period")


def _log_vram_watermark(stage: str, *, peak_label: str = "") -> None:
    """One log line for VRAM watermarks before/after sampling:
    alloc/reserved (torch's view) + whole-card (NVML).

    The change in the gap between reserved and whole-card is used to spot a
    WDDM demote (when the shared-memory curve jumps, torch's own numbers
    still look stable).

    ``peak_label``: also reports **the previous segment's peak**
    (the `max_memory_allocated` high watermark) and resets the counter.
    An instantaneous reading can't reveal a spike within a segment -- the
    training-step steady state and the sampling-period peak can differ by
    several GB, and it's the peak, not the steady state, that decides
    "is this card enough" (especially important with block swap: once the
    resident amount drops, the sampling period becomes the new ceiling).
    Fails silently -- logging must never block training."""
    try:
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        line = f"[{stage}] VRAM alloc={alloc:.1f}GB reserved={reserved:.1f}GB"
        if peak_label:
            peak = torch.cuda.max_memory_allocated() / 1e9
            line += f" | {peak_label} peak={peak:.1f}GB"
            torch.cuda.reset_peak_memory_stats()
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(0))
                line += f" whole-card={info.used / 1e9:.1f}GB"
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            pass
        logger.info(line)
    except Exception:
        pass
