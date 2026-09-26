"""Main training loop: for epoch / for batch / accumulation / forward / loss / periodic IO.

Extracted from main() L596-884 (ADR 0003 PR-B).

Lives at the top level of training/ (not under phases/) -- it's not a
one-shot setup, it's the iteration body; but run(ctx) has the same signature
as a phase, to make it easy for main() to orchestrate.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from training.context import TrainingContext
from training.loss_weighting import compute_loss_weight
from training.noise import make_noise, noise_params_from_args
from training.observability import render_curve_panel
from training.sample_runner import run_sample
from training.snapshot import (
    build_auto_epoch_config_path,
    build_auto_epoch_state_path,
    emit_event,
    write_config_snapshot,
)
from training.state import save_training_state
from training.timestep_sampling import apply_resolution_shift, latent_token_counts
from training import dop
from utils.optimizer_utils import get_optimizer_monitor_metrics, optimizer_eval_mode


logger = logging.getLogger(__name__)


def _save_adapter(ctx, path) -> None:
    """Save adapter weights to disk; when EMA is enabled and the shadow copy
    has been built, **additionally** saves a ``*_ema.safetensors``.

    The normal file is always saved -- EMA is a supplement, not a
    replacement; users should be able to try both. The EMA version reuses
    the same ``injector.save`` by temporarily swapping in the shadow weights
    via ``ema.applied()``, so the alpha rewrite / ss_* metadata / family tag
    all stay consistent.
    """
    ctx.injector.save(path)
    ema = getattr(ctx, "ema", None)
    if ema is None:
        return
    with ema.applied() as swapped:
        if not swapped:
            return
        ema_path = path.with_name(f"{path.stem}_ema{path.suffix}")
        ctx.injector.save(ema_path)
    ctx.emit(f"Saved EMA LoRA: {ema_path}")
    ctx.wandb_monitor.upload_model(ema_path)


def _resolve_sra_weight(args: Any) -> float:
    """Read sra_weight without treating explicit 0 as a missing value."""
    raw = getattr(args, "sra_weight", 0.2)
    return 0.2 if raw is None else float(raw)


def _resolve_sra_effective_weight(args: Any, global_step: int, total_steps: int | None) -> float:
    """Apply the configured SRA schedule to the base SRA weight."""
    base = _resolve_sra_weight(args)
    if base == 0.0:
        return 0.0

    decay_type = str(getattr(args, "sra_decay_type", "none") or "none").lower()
    if decay_type == "none" or not total_steps or total_steps <= 0:
        return base

    start = float(getattr(args, "sra_decay_start_ratio", 0.2) or 0.0)
    end = float(getattr(args, "sra_decay_end_ratio", 0.3) or 0.0)
    progress = max(0.0, min(1.0, float(global_step) / float(total_steps)))

    if decay_type == "jump":
        return base if progress < start else 0.0

    if progress <= start:
        return base
    if progress >= end:
        return 0.0

    x = (progress - start) / max(end - start, 1e-8)
    if decay_type == "linear":
        scale = 1.0 - x
    elif decay_type == "cosine":
        scale = 0.5 * (1.0 + math.cos(math.pi * x))
    else:
        scale = 1.0
    return base * max(0.0, min(1.0, scale))


def _compute_loss_weight_from_args(args, t, device):
    """Compute the timestep-dependent weight from args' loss_weighting config;
    scheme=none returns None.

    The navit (per-image t) and standard (per-sample t) paths share the same
    parameter assembly, to avoid the two drifting apart when a new scheme
    parameter is added."""
    scheme = str(getattr(args, "loss_weighting", "none") or "none")
    if scheme == "none":
        return None
    return compute_loss_weight(
        t,
        scheme=scheme,
        min_snr_gamma=float(getattr(args, "min_snr_gamma", 5.0) or 5.0),
        weight_cap_ratio=float(getattr(args, "weight_cap_ratio", 0.0) or 0.0),
        detail_inv_t_min=float(getattr(args, "detail_inv_t_min", 1.0) or 1.0),
        detail_inv_t_max=float(getattr(args, "detail_inv_t_max", 5.0) or 5.0),
    ).to(device=device, dtype=torch.float32)


def _masked_mean(per_element, spatial_mask):
    """Weighted-mean reduction for masked loss: `(loss*mask).sum() / mask.sum()`.

    The denominator is the sum of mask elements rather than the total
    element count -- samples with different mask areas get consistent weight
    within a batch, avoiding the implicit down-weighting of large-mask
    images that kohya's plain `mean()` causes (design doc SS8).
    Per-sample weights (reg / timestep weighting) are already multiplied
    into per_element, and the denominator excludes them (symmetric with how
    the no-mask `mean()`'s denominator excludes the weights too).
    """
    m = spatial_mask.expand_as(per_element)
    return (per_element * m).sum() / m.sum().clamp_min(1e-6)


def _masked_mean_per_sample(per_element, spatial_mask):
    """Per-sample version of masked mean (used for InfoNoise's `_raw_mse`
    recording): for each sample, computes the mask-weighted mean over the
    non-batch dims, returning (B,). Error in masked-out regions doesn't enter
    the I-MMSE statistics, or noise from unsupervised regions would
    contaminate the CDF."""
    m = spatial_mask.expand_as(per_element)
    dims = list(range(1, per_element.dim()))
    return (per_element * m).sum(dim=dims) / m.sum(dim=dims).clamp_min(1e-6)


def _accumulation_step(batch_idx, dl_len, grad_accum):
    """Returns (group_size, is_group_end): the actual size of the gradient
    accumulation group this micro-batch belongs to, and whether it's the
    last one in that group (triggering optimizer.step / zero_grad).

    Matches kohya-ss / HF Trainer:
    - The last batch of an epoch steps even if it doesn't fill grad_accum --
      otherwise the gradients of the trailing `len % ga` batches would be
      dropped (single epoch) or leak into the next epoch's first step
      (multi-epoch, since zero_grad is only called after step).
    - An incomplete tail group normalizes by its **actual** micro-batch
      count (group_size), rather than always dividing by grad_accum, so the
      final step's gradient isn't weakened.

    When dl_len=None (the dataloader has no __len__), falls back to the old
    behavior (always grad_accum, steps only at exact multiples).
    """
    if dl_len is None or grad_accum <= 1:
        return grad_accum, (batch_idx + 1) % grad_accum == 0
    remainder = dl_len % grad_accum
    in_tail = remainder != 0 and batch_idx >= dl_len - remainder
    group_size = remainder if in_tail else grad_accum
    is_group_end = (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == dl_len
    return group_size, is_group_end


def _display_total_steps(global_step, dl_len, grad_accum, total_epochs, epoch, max_steps):
    """Dynamic correction of the total step count used for
    monitor/progress display (doesn't touch the scheduler horizon).

    After the NaViT packer reshuffles each epoch, the pack count changes;
    ``ctx.total_steps``, computed when optimizer_phase starts, is an epoch-0
    snapshot -> under multiple epochs, the monitor's progress-bar
    denominator stays stale for the whole run (step != total_steps at the
    end). This function re-estimates it at the start of every epoch as
    "steps already taken + this epoch's step count x remaining epochs
    (including this one)" -- the estimate gets more accurate the later the
    epoch, and is exact on the last epoch.

    On the non-navit path, dl_len is constant, so the result always equals
    the total_steps computed at startup (a pure identity, zero behavior
    change). The scheduler's T_max is built once at construction and can't
    be recomputed mid-run, so it still uses the startup-time estimate (a
    few steps of drift at the LR floor is acceptable, see the
    optimizer_phase comment).

    dl_len=None (the dataloader has no __len__) or total_epochs<=0 -> None,
    and the caller falls back to ``ctx.total_steps``.
    """
    if dl_len is None or not total_epochs or total_epochs <= 0:
        return None
    ga = max(1, int(grad_accum or 1))
    steps_this_epoch = (int(dl_len) + ga - 1) // ga
    remaining_epochs = max(0, int(total_epochs) - int(epoch))
    est = int(global_step) + steps_this_epoch * remaining_epochs
    if max_steps and int(max_steps) > 0:
        est = min(est, int(max_steps))
    return est


def _sample_timesteps(timestep_sampler, bs: int, device, latents) -> torch.Tensor:
    """Inject batch context on demand based on the sampler's declared
    capabilities, so family branching doesn't leak into the shared loop."""
    if getattr(timestep_sampler, "requires_token_counts", False):
        return timestep_sampler.sample(
            bs,
            device,
            token_counts=latent_token_counts(latents),
        )
    return timestep_sampler.sample(bs, device)


def _cuda_reserved_gb(device) -> float | None:
    """The torch allocator's reserved amount on device (GB); returns None for a non-CUDA device."""
    try:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        return torch.cuda.memory_reserved(dev) / 1024**3
    except Exception:  # noqa: BLE001
        return None


class _BucketSwitchCacheRelease:
    """When ARB switches buckets, returns the allocator cache left over from
    the previous bucket to the driver (issue #505).

    BucketBatchSampler produces buckets one after another contiguously;
    within a bucket, every step's activation shape is the same and cached
    blocks are fully reused. After switching to a new bucket, the new
    shape's large tensors don't fit into the old bucket's cached blocks, so
    the allocator has to cudaMalloc again -> reserved ~= old bucket's peak +
    new bucket's peak. On Linux, when cudaMalloc hits the VRAM ceiling it
    fails, and the allocator's built-in "fail -> release cache -> retry"
    self-heals; under Windows WDDM, cudaMalloc doesn't fail but instead
    spills into shared memory, so the self-heal never triggers, and every
    subsequent step goes over PCIe, permanently cutting speed by more than
    half (measured on a 5090 32GB: 0.78 -> 0.3 it/s). This patches in that
    release at the bucket-switch point: at this moment the previous step's
    backward/optimizer step has already finished and all activations are
    freed, so empty_cache returns the entire old-peak segment cleanly. A
    bucket only switches a few times per epoch, so the overhead is
    negligible; the same call runs on Linux too, to avoid a platform
    branch.

    Only used on the ARB grid path: every NaViT pack has a different shape,
    so cleaning by shape would turn into empty_cache + re-cudaMalloc every
    step, which would clearly slow things down; and packs have a token
    ceiling, so reserved naturally converges to the largest pack's peak
    without intervention.
    """

    def __init__(self, device):
        self._device = device
        self._prev_hw: tuple[int, ...] | None = None

    def observe(self, latents) -> None:
        hw = tuple(int(x) for x in latents.shape[-2:])
        prev, self._prev_hw = self._prev_hw, hw
        if prev is None or prev == hw:
            return
        debug = logger.isEnabledFor(logging.DEBUG)
        before = _cuda_reserved_gb(self._device) if debug else None
        torch.cuda.empty_cache()
        if not debug:
            return
        after = _cuda_reserved_gb(self._device)
        detail = (
            f" (torch reserved {before:.2f}GB->{after:.2f}GB)"
            if before is not None and after is not None
            else ""
        )
        logger.debug(
            "[vram] ARB bucket switch %sx%s->%sx%s: released the previous bucket's allocator cache%s",
            prev[0], prev[1], hw[0], hw[1], detail,
        )


def run(ctx: TrainingContext) -> None:
    """Run training until the args.epochs or args.max_steps limit."""
    args = ctx.args

    step_start_time = time.perf_counter()

    # The dataloader's batch count per epoch (used for "the tail group steps
    # even if incomplete" + normalizing by actual size). The rare dataloader
    # without __len__ falls back to the old behavior (see _accumulation_step).
    try:
        dl_len = len(ctx.dataloader)
        _has_len = True
    except TypeError:
        dl_len = None
        _has_len = False

    # timestep shift resolution correction (timestep_shift_resolution_aware,
    # opt-in): under multi-resolution / NaViT native resolution, pushes t to
    # the noise level equivalent to the base entry, based on each image's
    # token count (SD3 SS5.3.2, s_i = sqrt(n_i/n_base)). Base entry =
    # resolution's first entry: that entry always has s=1, so the global
    # timestep_shift is still "the base entry's calibrated value"; this
    # switch is a no-op under single-resolution training.
    # DOP differential output preservation: settle at startup whether it can
    # even run (the adapter must be able to temporarily disable itself, and
    # a trigger word must exist), so we don't discover an invalid config
    # halfway through training.
    _dop_enabled = bool(getattr(args, "dop_enabled", False))
    _dop_trigger = str(getattr(args, "trigger_word", "") or "").strip()
    if _dop_enabled:
        dop.assert_adapter_supports_dop(ctx.injector)
        if not _dop_trigger:
            raise ValueError(
                "dop_enabled=true requires a non-empty trigger_word: "
                "DOP's preservation branch is exactly \"strip the trigger word out of the "
                "caption\"; without a trigger word the two branches are identical and the "
                "constraint degenerates to 0."
                " / DOP needs a trigger word: set it on the Train page"
                " (Trigger word card) -- it must appear in the captions."
            )
        logger.info(
            "[DOP] differential output preservation enabled: trigger=%r, weight=%.3g, ratio=%.2f"
            " (two extra forward passes on every enabled step)",
            _dop_trigger,
            float(getattr(args, "dop_weight", 1.0) or 0.0),
            float(getattr(args, "dop_ratio", 1.0) or 0.0),
        )

    res_shift_base_tokens = 0
    if bool(getattr(args, "timestep_shift_resolution_aware", False)):
        _r = getattr(args, "resolution", 1024)
        _base_reso = int(_r[0] if isinstance(_r, (list, tuple)) else _r)
        res_shift_base_tokens = max(1, (_base_reso // 16) ** 2)
        logger.info(
            "[res-shift] timestep shift resolution correction enabled: s_i=sqrt(token_i/%d)"
            " (base entry %dpx), applied to t after sampling.",
            res_shift_base_tokens, _base_reso,
        )
        if getattr(ctx.timestep_sampler, "applies_resolution_shift", False):
            raise ValueError(
                "timestep_shift_resolution_aware cannot be enabled together with a timestep "
                "sampler that already applies its own resolution shift"
            )

    bucket_switch = _BucketSwitchCacheRelease(ctx.device)

    for epoch in range(ctx.start_epoch, args.epochs):
        ctx.current_epoch = epoch
        epoch_loss_sum = 0.0
        epoch_step_count = 0
        if ctx.use_cached and hasattr(ctx.dataloader, "batch_sampler") and hasattr(ctx.dataloader.batch_sampler, "set_epoch"):
            ctx.dataloader.batch_sampler.set_epoch(epoch)
            # After the NaViT packer reshuffles each epoch, the pack count
            # changes (next-fit/windowed FFD is order-dependent), so dl_len
            # must be refreshed after set_epoch, or _accumulation_step's
            # tail-group check would keep using epoch-0's stale pack count
            # -> the gradient accumulation tail group would be dropped or
            # leak across epochs.
            # ARB BucketBatchSampler's pack count is independent of
            # shuffle, so refreshing it is idempotent.
            if _has_len:
                dl_len = len(ctx.dataloader)
        # The total step count for progress display is progressively
        # corrected based on the current epoch's actual pack count (an
        # identity for non-navit, see the function's docstring)
        _display_total = _display_total_steps(
            ctx.global_step, dl_len, args.grad_accum, args.epochs, epoch,
            getattr(args, "max_steps", 0),
        )
        if _display_total is not None:
            ctx.total_steps_display = _display_total
        for batch_idx, batch in enumerate(ctx.dataloader):
            # Record the time at the start of the accumulation cycle
            if batch_idx % args.grad_accum == 0:
                step_start_time = time.perf_counter()

            captions = batch["captions"]

            # Get latents (cached mode or real-time encoding)
            navit_latents = None
            if bool(getattr(args, "navit_packing", False)):
                # NaViT pack: per-image cached latents (shapes differ → kept as a list).
                navit_latents = [
                    l.to(ctx.device, dtype=ctx.dtype) for l in batch["navit_latents"]
                ]
                bs = len(navit_latents)
            elif ctx.use_cached:
                latents = batch["latents"].to(ctx.device, dtype=ctx.dtype)
                bs = latents.shape[0]
            else:
                pixels = batch["pixel_values"].to(ctx.device, dtype=ctx.dtype)
                with torch.no_grad():
                    pixels_5d = pixels.unsqueeze(2)  # [B,C,1,H,W]
                    latents = ctx.vae.model.encode(pixels_5d, ctx.vae.scale)
                bs = latents.shape[0]

            # ARB bucket switch -> return the previous bucket's allocator
            # cache (doesn't apply to the navit path, see the class docstring)
            if navit_latents is None:
                bucket_switch.observe(latents)

            # Text encoding: the whole block is delegated to the family
            # (cond is opaque to the loop, 03 SS2.7-4; pad-to-512 / kv_trim /
            # LLMAdapter fusion are all Anima-specific)
            _enc_kwargs = dict(
                comfy_encoding=bool(getattr(args, "caption_comfy_encoding", True)),
                kv_trim=bool(getattr(args, "kv_trim", False)),
            )
            if navit_latents is not None:
                # NaViT packing needs a per-image attention mask for
                # cross-attn padding truncation; navit is an Anima
                # capability bit (capability check fail-fast), so this
                # family-private kwarg doesn't go into the protocol.
                cross, t5_attn = ctx.family.encode_text_for_batch(
                    ctx.text_stack, ctx.model, captions,
                    ctx.device, ctx.dtype,
                    return_t5_attn=True, **_enc_kwargs,
                )
            else:
                cross = ctx.family.encode_text_for_batch(
                    ctx.text_stack, ctx.model, captions,
                    ctx.device, ctx.dtype,
                    **_enc_kwargs,
                )

            # Flow Matching: always sampled through the timestep_sampler
            # plugin interface (baseline = 4 modes; adaptive = InfoNoise
            # etc.; the interface lives in the ADR 0003 plugin registry)
            t = _sample_timesteps(
                ctx.timestep_sampler,
                bs,
                ctx.device,
                navit_latents if navit_latents is not None else latents,
            )

            # Resolution-dependent shift correction (see the setup at the
            # top of run()): navit computes each image's own latent token
            # count individually; on the grid-batch path, same-size images
            # in a batch yield an equal-valued vector. All downstream
            # record / sigma_t / noising see the corrected t (what gets
            # recorded is the actual noise level trained on).
            if res_shift_base_tokens:
                t = apply_resolution_shift(
                    t,
                    latent_token_counts(
                        navit_latents if navit_latents is not None else latents
                    ),
                    res_shift_base_tokens,
                )

            # PR-C: adapter hook -- lets variants adjust their runtime
            # structure based on sigma_t / step (T-LoRA / AdaLoRA / B-LoRA
            # etc.). LyCORIS uses the default no-op.
            from training.adapters.protocol import StepContext
            step_ctx = StepContext(
                global_step=ctx.global_step,
                total_steps=ctx.total_steps,
                epoch=epoch,
                sigma_t=t,
                args=args,
            )
            ctx.injector.on_step_begin(step_ctx)

            # NaViT adds noise per-image inside its own packed forward pass
            # (each image with its own t / shape), bypassing grid-batch noising.
            t_exp = noise = pad_mask = None
            use_leap_this_step = False
            if navit_latents is None:
                t_exp = t.view(-1, 1, 1, 1, 1)
                # Noise-augmentation parameters are dispatched by
                # noise_enhancement_type (audit #3: the other, unused
                # parameter set doesn't participate, preventing offset+pyramid
                # from silently stacking)
                _no, _pi, _pd = noise_params_from_args(args)
                noise = make_noise(
                    latents,
                    noise_offset=_no,
                    pyramid_iters=_pi,
                    pyramid_discount=_pd,
                )

                leap_enabled = bool(getattr(args, "leap_enabled", False))
                # Mixed training scheme A: for every micro-batch, roll the
                # dice based on leap_ratio to decide which objective to use.
                # leap governs global structure, the traditional path governs
                # fine detail sharpness; the two gradients stack on the same
                # set of LoRA weights, each contributing its strength.
                # ratio=1.0 pure leap; 0.0 pure traditional; 0.6 mostly leap
                # with some detail left (default).
                # Uses Python's random (bootstrap already sets random.seed)
                # rather than torch.rand, to avoid consuming torch's global
                # RNG state on every step -- otherwise, in a "same seed,
                # different leap_ratio" control experiment, the standard
                # path's noise / timestep would drift with leap_ratio.
                # Note: only default/None falls back to 0.6; leap_ratio=0.0
                # (pure traditional) is a valid value and must not be
                # swallowed into the default by `or`.
                _leap_ratio = getattr(args, "leap_ratio", 0.6)
                use_leap_this_step = leap_enabled and (
                    random.random() < (0.6 if _leap_ratio is None else float(_leap_ratio))
                )

                # pad_mask: on the standard path this has already moved
                # down into forward_train under the family (03-3); kept
                # here in the loop only for leap (Anima-only gated code)
                if use_leap_this_step:
                    pad_mask = torch.zeros(bs, 1, latents.shape[-2], latents.shape[-1], device=ctx.device, dtype=ctx.dtype)
            denoise_loss_log = None
            dop_loss_log = None
            sra_align_loss_log = None
            sra_weighted_loss_log = None
            sra_effective_weight_log = None
            with torch.autocast("cuda", dtype=ctx.dtype):
                if navit_latents is not None:
                    # -- NaViT / Patch-n-Pack block-diagonal packing path --
                    # per-image noise + one packed forward (block-diagonal self/cross).
                    # The standard path's leap / SRA / InfoNoise assume a
                    # grid batch + a single t per batch, incompatible with
                    # per-image packing -- the mutual-exclusion check
                    # already fail-fast disables them.
                    # loss_weighting / regularization-set loss_weight are not
                    # in that list: navit's per-image t maps exactly onto the
                    # per-sample SNR-weight semantics, hooked up per-image
                    # (see per_image_weights below).
                    from training.families.anima.navit import (  # noqa: PLC0415  capability gate (D5), lazy import
                        navit_packed_forward_and_loss,
                        pack_cross_embeddings,
                    )

                    cross_packed, text_seqlens = pack_cross_embeddings(
                        cross, t5_attn,
                        bool(getattr(args, "navit_text_trim_padding", False)),
                    )
                    # per-image weight: regularization-set loss_weight x
                    # timestep-dependent loss_weighting. Symmetric with the
                    # standard path -- navit's per-image t corresponds to
                    # per-sample SNR weights (min_snr / cosmap /
                    # detail_inv_t compute weights from per-image t,
                    # semantically consistent; unlike leap, which has
                    # multiple timesteps).
                    _piw = None
                    if "loss_weight" in batch:
                        _piw = batch["loss_weight"].to(ctx.device, dtype=torch.float32)
                    _lw = _compute_loss_weight_from_args(args, t, ctx.device)
                    if _lw is not None:
                        _piw = _lw if _piw is None else _piw * _lw
                    _no, _pi, _pd = noise_params_from_args(args)
                    loss, pred, _navit_info = navit_packed_forward_and_loss(
                        ctx.model, navit_latents, t, cross_packed, text_seqlens,
                        ctx.loss_fn,
                        noise_offset=_no,
                        pyramid_iters=_pi,
                        pyramid_discount=_pd,
                        use_checkpoint=bool(args.grad_checkpoint),
                        per_image_weights=_piw,
                    )
                    denoise_loss_log = loss.detach()
                elif use_leap_this_step:
                    # -- LeapAlign / FlowBP trajectory self-distillation path (four variants) --
                    # Uses the real latent as x0, integrates along an
                    # analytically constructed proxy trajectory to get an
                    # estimate x_hat0, and self-distills with
                    # loss = MSE(x_hat0, real x0). The variant determines
                    # the trajectory structure, see training/leap.py for details:
                    #   original  two-step jump + straight-through connector (K=2, same behavior as the historical version)
                    #   sparse    K-point Euler replay, pure direct-term sum (zero connector / zero Jacobian)
                    #   bridge    two-step jump + Euler-reconstructed connector (no straight-through bias)
                    #   lagrange  two-segment jump + 3-point Simpson integration per segment (6x forward passes)
                    leap_variant = str(getattr(args, "leap_variant", "original") or "original")
                    from training.families.anima.leap import (  # noqa: PLC0415  capability gate (D5), lazy import
                        bridge_training_step,
                        lagrange_training_step,
                        leap_training_step,
                        sample_activation_timesteps,
                        sample_two_timesteps,
                        sparse_training_step,
                    )

                    _leap_min_gap = float(getattr(args, "leap_min_gap", 0.1) or 0.1)
                    _leap_tsw = bool(getattr(args, "leap_traj_sim_weighting", False))
                    _leap_tsm = float(getattr(args, "leap_traj_sim_min", 0.1) or 0.1)
                    _leap_ngc = float(getattr(args, "leap_nested_grad_coe", 0.3))
                    if leap_variant == "sparse":
                        # K-point activation set (K forward passes + K activations of VRAM)
                        t_steps = sample_activation_timesteps(
                            bs, ctx.device,
                            k=int(getattr(args, "leap_activation_k", 3) or 3),
                            dtype=torch.float32,
                        )
                        loss_per_sample = sparse_training_step(
                            ctx.model, latents, noise, cross, pad_mask, t_steps,
                            traj_sim_weighting=_leap_tsw, traj_sim_min=_leap_tsm,
                            use_checkpoint=args.grad_checkpoint,
                        )
                    else:
                        # original / bridge / lagrange share the two-timestep (k,j) topology
                        t_k, t_j = sample_two_timesteps(
                            bs, ctx.device, min_gap=_leap_min_gap, dtype=torch.float32,
                        )
                        if leap_variant == "bridge":
                            _step_fn = bridge_training_step
                        elif leap_variant == "lagrange":
                            _step_fn = lagrange_training_step
                        else:  # original (default, zero behavior change)
                            _step_fn = leap_training_step
                        loss_per_sample = _step_fn(
                            ctx.model, latents, noise, cross, pad_mask, t_k, t_j,
                            nested_grad_coe=_leap_ngc,
                            traj_sim_weighting=_leap_tsw, traj_sim_min=_leap_tsm,
                            use_checkpoint=args.grad_checkpoint,
                        )
                    # The leap path deliberately skips two standard
                    # mechanisms (the mutual-exclusion check already
                    # disables them in TrainingConfig):
                    #   - InfoNoise record: the I-MMSE semantics don't match a dual timestep the way they match a single t
                    #   - loss_weighting: relies on a single t to compute the SNR weight; leap has its own traj_sim weighting
                    # Still respects the batch's loss_weight (regularization-set downweighting), consistent with the standard path.
                    if "loss_weight" in batch:
                        w = batch["loss_weight"].to(ctx.device).view(-1, *([1] * (loss_per_sample.dim() - 1)))
                        loss_per_sample = loss_per_sample * w
                    loss = loss_per_sample.mean()
                    denoise_loss_log = loss.detach()
                else:
                    # -- Standard rectified flow path (zero behavior change) --
                    noisy = (1 - t_exp) * latents + t_exp * noise
                    target = noise - latents
                    pred = ctx.family.forward_train(
                        ctx.model, noisy, t, cross,
                        use_checkpoint=args.grad_checkpoint,
                    )
                    # masked loss (B2): the dataset has already downsampled
                    # the mask to latent resolution, (B,h,w) -> (B,1,1,h,w)
                    # broadcasting to the loss's (B,C,T,H,W).
                    # None when masked_loss is off or this batch has no mask at all (zero overhead).
                    spatial_mask = None
                    if bool(getattr(args, "masked_loss", False)) and "masks" in batch:
                        _m = batch["masks"].to(ctx.device, dtype=torch.float32)
                        spatial_mask = _m.view(bs, 1, 1, *_m.shape[-2:])
                    # Training loss is dispatched through the losses/ plugin registry (mse / huber / ...)
                    loss_per_sample = ctx.loss_fn.compute(pred.float(), target.float(), t)
                    # The adaptive sampler (e.g. InfoNoise) records the raw
                    # per-sample MSE (unaffected by huber/loss_weighting
                    # etc. post-processing); decoupled from the training
                    # loss to preserve consistency with the InfoNoise paper.
                    # The baseline sampler is a no-op, so no if-guard is needed.
                    # Uses no_grad to avoid building autograd metadata
                    # (cheaper than .detach(), one less grad_fn).
                    with torch.no_grad():
                        _raw_mse_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none")
                        if spatial_mask is not None:
                            # Masked-out regions are unsupervised, their error doesn't enter the I-MMSE statistics
                            _raw_mse = _masked_mean_per_sample(_raw_mse_per_sample, spatial_mask)
                        else:
                            _raw_mse = _raw_mse_per_sample.mean(
                                dim=list(range(1, _raw_mse_per_sample.dim()))
                            )
                    # Only main-set samples feed the InfoNoise schedule
                    # learning: I-MMSE assumes a single data distribution;
                    # the reg set is typically generic images (booru) vs
                    # the main set being a single subject, so mixing them
                    # into the record would learn a mixture MMSE, not
                    # mmse_main(t). Uses the is_reg flag rather than a
                    # loss_weight threshold, because distribution identity
                    # and gradient weight are two independent axes
                    # (when reg_weight=1.0, loss_weight=1.0 but reg is
                    # still a different distribution).
                    # See docs/todo/infonoise-reg-policy-reeval.md for future re-evaluation conditions.
                    if "is_reg" in batch:
                        _main_mask = ~batch["is_reg"].to(t.device)
                        if _main_mask.any():
                            ctx.timestep_sampler.record(t.detach()[_main_mask], _raw_mse[_main_mask])
                    else:
                        ctx.timestep_sampler.record(t.detach(), _raw_mse)
                    # Weight by sample (a regularization set can lower its weight)
                    if "loss_weight" in batch:
                        w = batch["loss_weight"].to(ctx.device).view(-1, *([1] * (loss_per_sample.dim() - 1)))
                        loss_per_sample = loss_per_sample * w
                    # timestep-dependent loss weight
                    lw = _compute_loss_weight_from_args(args, t, ctx.device)
                    if lw is not None:
                        loss_per_sample = loss_per_sample * lw.view(-1, *([1] * (loss_per_sample.dim() - 1)))
                    if spatial_mask is not None:
                        loss = _masked_mean(loss_per_sample, spatial_mask)
                    else:
                        loss = loss_per_sample.mean()
                    denoise_loss_log = loss.detach()

                # SRA v2 representation-alignment loss (standard path only; not applicable to leap / navit)
                if ctx.sra_aligner is not None and not use_leap_this_step and navit_latents is None:
                    sra_weight = _resolve_sra_effective_weight(args, ctx.global_step, ctx.total_steps)
                    sra_effective_weight_log = sra_weight
                    if sra_weight != 0.0:
                        align_loss = ctx.sra_aligner.compute(
                            latents,
                            sample_weight=batch.get("loss_weight"),
                        )
                        weighted_align_loss = sra_weight * align_loss
                        loss = loss + weighted_align_loss
                        sra_align_loss_log = align_loss.detach()
                        sra_weighted_loss_log = weighted_align_loss.detach()
                    else:
                        sra_weighted_loss_log = loss.new_tensor(0.0).detach()

                # DOP differential output preservation (standard path only;
                # mutually exclusive with leap / navit at the schema level).
                # Compares "adapter on" vs "adapter off" predictions on the
                # same batch of images, with the trigger word stripped from
                # the caption: this teaches the LoRA "change nothing when
                # there's no trigger word", and as a side effect blocks off
                # the "copy the dataset's content" shortcut -- the two
                # branches have identical content, so copying it earns no
                # reward. See training/dop.py for details.
                if (
                    _dop_enabled
                    and not use_leap_this_step
                    and navit_latents is None
                    and dop.should_apply(getattr(args, "dop_ratio", 1.0), random)
                ):
                    _cross_wo = ctx.family.encode_text_for_batch(
                        ctx.text_stack, ctx.model,
                        dop.preservation_captions(captions, _dop_trigger),
                        ctx.device, ctx.dtype, **_enc_kwargs,
                    )
                    dop_loss_raw = dop.compute_dop_loss(
                        family=ctx.family,
                        model=ctx.model,
                        injector=ctx.injector,
                        noisy=noisy,
                        t=t,
                        cross_wo_trigger=_cross_wo,
                        use_checkpoint=bool(args.grad_checkpoint),
                    )
                    _dop_w = float(getattr(args, "dop_weight", 1.0) or 0.0)
                    loss = loss + _dop_w * dop_loss_raw
                    dop_loss_log = dop_loss_raw.detach()

                # PR-C: adapter hook -- variants can add a regularization
                # term (OFT orth penalty / Ortho-Hydra balance loss, etc.).
                # LyCORIS returns None, a no-op.
                reg = ctx.injector.regularization_loss(step_ctx)
                if reg is not None:
                    loss = loss + reg

            # NaN detection: skip this micro-batch when the forward pass produces NaN
            if not torch.isfinite(loss):
                logger.warning(f"step {ctx.global_step} micro-batch {batch_idx}: loss={loss.item():.4g}, skipping")
                ctx.optimizer.zero_grad()
                continue

            # Backward pass. An incomplete tail group (len % grad_accum)
            # normalizes by its actual micro-batch count, and the last
            # batch of an epoch also steps even if incomplete -- this is
            # the fix for tail-batch dropping + cross-epoch gradient
            # leakage (see _accumulation_step).
            group_size, is_group_end = _accumulation_step(batch_idx, dl_len, args.grad_accum)
            loss = loss / group_size
            if ctx.scaler is not None:
                ctx.scaler.scale(loss).backward()
            else:
                loss.backward()

            if is_group_end:
                if ctx.scaler is not None:
                    ctx.scaler.unscale_(ctx.optimizer)
                # NaN gradient detection: skip this update, zero and continue
                has_nan_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in ctx.trainable_params
                )
                if has_nan_grad:
                    logger.warning(f"step {ctx.global_step}: gradient contains NaN/Inf, skipping optimizer.step()")
                    ctx.optimizer.zero_grad()
                    if ctx.scaler is not None:
                        ctx.scaler.update()
                    continue

                if ctx.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ctx.trainable_params, max_norm=ctx.grad_clip)
                if ctx.scaler is not None:
                    ctx.scaler.step(ctx.optimizer)
                    ctx.scaler.update()
                else:
                    ctx.optimizer.step()
                if ctx.scheduler is not None and ctx.optimizer_type != "prodigy_plus_schedulefree":
                    ctx.scheduler.step()
                ctx.optimizer.zero_grad()
                ctx.global_step += 1

                # Weight EMA: only accumulates after a real update step (not
                # every micro-batch), or grad_accum would make the
                # "effective window" drift with the accumulation step count
                if ctx.ema is not None:
                    ctx.ema.update(ctx.global_step)

                # Adaptive sampler: refresh the sampling distribution; baseline is a no-op
                ctx.timestep_sampler.maybe_refresh(ctx.global_step)

                # Record loss history
                loss_val = float(loss.item() * group_size)
                denoise_loss_val = (
                    float(denoise_loss_log.item())
                    if denoise_loss_log is not None else loss_val
                )
                sra_align_loss_val = (
                    float(sra_align_loss_log.item())
                    if sra_align_loss_log is not None else None
                )
                dop_loss_val = (
                    float(dop_loss_log.item()) if dop_loss_log is not None else None
                )
                sra_weighted_loss_val = (
                    float(sra_weighted_loss_log.item())
                    if sra_weighted_loss_log is not None else None
                )
                epoch_loss_sum += loss_val
                epoch_step_count += 1
                if args.loss_curve_steps and len(ctx.loss_history) < args.loss_curve_steps:
                    ctx.loss_history.append(loss_val)

                # Update the progress display
                now = time.perf_counter()
                optimizer_metrics = get_optimizer_monitor_metrics(ctx.optimizer)
                lr = optimizer_metrics["lr"]

                # Update the training monitor panel
                if ctx.monitor_server:
                    try:
                        from train_monitor import update_monitor
                        monitor_metrics = dict(optimizer_metrics)
                        monitor_metrics["denoise_loss"] = denoise_loss_val
                        if sra_align_loss_val is not None:
                            monitor_metrics["sra_align_loss"] = sra_align_loss_val
                        if dop_loss_val is not None:
                            monitor_metrics["dop_loss"] = dop_loss_val
                        if sra_weighted_loss_val is not None:
                            monitor_metrics["sra_weighted_loss"] = sra_weighted_loss_val
                        if sra_effective_weight_log is not None:
                            monitor_metrics["sra_effective_weight"] = float(sra_effective_weight_log)
                        update_monitor(
                            loss=loss_val, lr=lr, epoch=epoch + 1,
                            total_epochs=int(args.epochs or 0),
                            step=ctx.global_step,
                            total_steps=ctx.total_steps_display or ctx.total_steps,
                            speed=ctx.speed_ema or 0,
                            optimizer_metrics=monitor_metrics,
                        )
                    except Exception:
                        pass
                dt_step = now - step_start_time
                steps_per_sec = (1.0 / dt_step) if dt_step > 0 else 0.0
                ctx.speed_ema = steps_per_sec if ctx.speed_ema is None else (0.9 * ctx.speed_ema + 0.1 * steps_per_sec)
                log_payload: dict[str, Any] = {
                    "train/loss": loss_val,
                    "train/denoise_loss": denoise_loss_val,
                    "train/lr": float(lr),
                    "train/speed_it_s": float(ctx.speed_ema or 0),
                }
                if sra_align_loss_val is not None:
                    log_payload["train/sra_align_loss"] = sra_align_loss_val
                if dop_loss_val is not None:
                    log_payload["train/dop_loss"] = dop_loss_val
                if sra_weighted_loss_val is not None:
                    log_payload["train/sra_weighted_loss"] = sra_weighted_loss_val
                if sra_effective_weight_log is not None:
                    log_payload["train/sra_effective_weight"] = float(sra_effective_weight_log)
                if "d" in optimizer_metrics:
                    log_payload["train/optimizer_d"] = float(optimizer_metrics["d"])
                if "base_lr" in optimizer_metrics:
                    log_payload["train/base_lr"] = float(optimizer_metrics["base_lr"])
                if "effective_lr" in optimizer_metrics:
                    log_payload["train/effective_lr"] = float(optimizer_metrics["effective_lr"])
                # Adaptive-sampler observability (P1-1): whether the CDF is ready + degradation count
                if (
                    ctx.global_step % args.log_every == 0
                    and ctx.timestep_sampler.status().get("kind") == "infonoise"
                ):
                    status = ctx.timestep_sampler.status()
                    log_payload["infonoise/cdf_ready"] = float(status["cdf_ready"])
                    log_payload["infonoise/refresh_degraded_count"] = status["refresh_degraded_count"]
                ctx.wandb_monitor.log(log_payload, step=ctx.global_step)

                if ctx.use_rich:
                    desc = f"epoch {epoch+1}/{args.epochs} step {ctx.global_step}/{ctx.total_steps_display or ctx.total_steps or '?'}"
                    ctx.progress.update(
                        ctx.task_id, advance=1, description=desc,
                        loss=loss_val, lr=float(lr), speed=float(ctx.speed_ema or 0),
                    )
                    if ctx.live and args.loss_curve_steps > 0 and not args.no_live_curve:
                        panel = render_curve_panel(ctx.loss_history, width=min(60, args.loss_curve_steps), height=10)
                        if panel is not None:
                            from rich.console import Group
                            ctx.live.update(Group(ctx.progress, panel))
                elif ctx.use_plain:
                    sra_suffix = (
                        f" denoise={denoise_loss_val:.6f}"
                        f" sra={sra_align_loss_val:.6f}"
                        f" sra_w={sra_weighted_loss_val:.6f}"
                        if sra_align_loss_val is not None and sra_weighted_loss_val is not None
                        else f" denoise={denoise_loss_val:.6f}"
                    )
                    print(f"epoch {epoch+1}/{args.epochs} step {ctx.global_step} loss={loss_val:.6f}{sra_suffix} lr={lr:.2e} speed={ctx.speed_ema:.2f} it/s", end="\r", flush=True)
                elif args.log_every and ctx.global_step % args.log_every == 0:
                    sra_suffix = (
                        f" denoise={denoise_loss_val:.6f}"
                        f" sra={sra_align_loss_val:.6f}"
                        f" sra_w={sra_weighted_loss_val:.6f}"
                        if sra_align_loss_val is not None and sra_weighted_loss_val is not None
                        else f" denoise={denoise_loss_val:.6f}"
                    )
                    # flush must be on: studio-spawned stdout is a pipe
                    # (fully buffered, 8KB), and without flush the step
                    # line would sit in the buffer -- for a short training
                    # run or an aborted one, the task log would show not a
                    # single line (once mistaken for krea2 not logging at all)
                    print(
                        f"epoch={epoch} step={ctx.global_step} "
                        f"loss={loss_val:.6f}{sra_suffix} lr={lr:.2e} "
                        f"speed={steps_per_sec:.2f} it/s",
                        flush=True,
                    )

                # Sample by step (rotating prompts)
                if args.sample_steps > 0 and ctx.global_step % args.sample_steps == 0:
                    prompt = ctx.get_next_sample_prompt()
                    prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    ctx.emit(f"Sampling (step {ctx.global_step}): {prompt_short}")
                    run_sample(
                        ctx,
                        prompt=prompt,
                        sample_path=ctx.sample_dir / f"step_{ctx.global_step}.png",
                        wandb_key="samples/step",
                        wandb_caption=f"step {ctx.global_step}: {prompt}",
                        wandb_step=ctx.global_step,
                    )

                # Periodically save LoRA weights (by step)
                save_every_steps = getattr(args, "save_every_steps", 0)
                if save_every_steps > 0 and ctx.global_step % save_every_steps == 0:
                    lora_path = ctx.output_dir / f"{args.output_name}_step{ctx.global_step}.safetensors"
                    # PPSF: save the LoRA of the averaged weights
                    with optimizer_eval_mode(ctx.optimizer):
                        _save_adapter(ctx, lora_path)
                    ctx.emit(f"Saved LoRA: {lora_path}")
                    ctx.wandb_monitor.upload_model(lora_path)

                # Periodically save training state (for resuming from checkpoint)
                save_state_every_steps = getattr(args, "save_state_every_steps", 0)
                if save_state_every_steps > 0 and ctx.global_step % save_state_every_steps == 0:
                    state_path = ctx.state_dir() / f"training_state_step{ctx.global_step}.pt"
                    # Get the monitor panel data, used to restore the loss curve
                    monitor_data = None
                    if ctx.monitor_server:
                        try:
                            from train_monitor import get_state
                            monitor_data = get_state()
                        except Exception:
                            pass
                    # PPSF: both state + LoRA use the averaged weights
                    with optimizer_eval_mode(ctx.optimizer):
                        save_training_state(
                            state_path, ctx.injector, ctx.optimizer, epoch, ctx.global_step,
                            ctx.loss_history, monitor_state=monitor_data, scheduler=ctx.scheduler,
                            timestep_sampler=ctx.timestep_sampler,
                            sra_aligner=ctx.sra_aligner,
                            scaler=ctx.scaler,
                            model_family=ctx.family.spec.family_id,
                        ema=ctx.ema,
                        )
                        # Also save LoRA weights
                        lora_path = ctx.output_dir / f"{args.output_name}_step{ctx.global_step}.safetensors"
                        _save_adapter(ctx, lora_path)
                    ctx.emit(f"Saved training state (step {ctx.global_step}): {state_path.name}")
                    ctx.wandb_monitor.upload_state_manual(state_path)

                # Check max_steps
                if args.max_steps and ctx.global_step >= args.max_steps:
                    break

        # Actions performed after an epoch ends
        ctx.current_epoch = epoch + 1
        if epoch_step_count > 0:
            ctx.wandb_monitor.log(
                {
                    "train/loss_epoch": epoch_loss_sum / epoch_step_count,
                    "train/epoch": ctx.current_epoch,
                },
                step=ctx.global_step,
            )
        if not args.max_steps or ctx.global_step < args.max_steps:
            # Save checkpoint
            if args.save_every_epochs > 0 and ctx.current_epoch % args.save_every_epochs == 0:
                save_path = ctx.output_dir / f"{args.output_name}_epoch{ctx.current_epoch}.safetensors"
                # PPSF: save the LoRA of the averaged weights
                with optimizer_eval_mode(ctx.optimizer):
                    _save_adapter(ctx, save_path)
                ctx.emit(f"Saved LoRA: {save_path}")
                ctx.wandb_monitor.upload_model(save_path)

            # Sample (rotating prompts)
            if args.sample_every > 0 and ctx.current_epoch % args.sample_every == 0:
                prompt = ctx.get_next_sample_prompt()
                prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                ctx.emit(f"Sampling (epoch {ctx.current_epoch}): {prompt_short}")
                run_sample(
                    ctx,
                    prompt=prompt,
                    sample_path=ctx.sample_dir / f"epoch_{ctx.current_epoch}.png",
                    wandb_key="samples/epoch",
                    wandb_caption=f"epoch {ctx.current_epoch}: {prompt}",
                    wandb_step=ctx.global_step,
                )

            # Periodically save training state (epoch version)
            # ADR 0006 Addendum 1: while touching the epoch field, also fix
            # an off-by-one (dev current_epoch was already advanced at L297
            # = epoch+1; pass ctx.current_epoch here rather than epoch, so
            # resuming's `for epoch in range(start, N)` inclusive doesn't
            # retrain the whole epoch).
            save_state_every_epochs = int(getattr(args, "save_state_every_epochs", 0) or 0)
            if save_state_every_epochs > 0 and ctx.current_epoch % save_state_every_epochs == 0:
                state_path = ctx.state_dir() / f"training_state_epoch{ctx.current_epoch}.pt"
                monitor_data = None
                if ctx.monitor_server:
                    try:
                        from train_monitor import get_state
                        monitor_data = get_state()
                    except Exception:
                        pass
                with optimizer_eval_mode(ctx.optimizer):
                    save_training_state(
                        state_path, ctx.injector, ctx.optimizer, ctx.current_epoch, ctx.global_step,
                        ctx.loss_history, monitor_state=monitor_data, scheduler=ctx.scheduler,
                        timestep_sampler=ctx.timestep_sampler,
                        sra_aligner=ctx.sra_aligner,
                        scaler=ctx.scaler,
                        model_family=ctx.family.spec.family_id,
                        ema=ctx.ema,
                    )
                    lora_path = ctx.output_dir / f"{args.output_name}_epoch{ctx.current_epoch}.safetensors"
                    if not lora_path.exists():
                        _save_adapter(ctx, lora_path)
                ctx.emit(f"Saved training state (epoch {ctx.current_epoch}): {state_path.name}")
                ctx.wandb_monitor.upload_state_manual(state_path)

            # ADR 0006 Addendum 1 scheme delta: at the end of every epoch,
            # **unconditionally** write auto_epoch_state.pt (overwriting).
            # Independent of the user's own save_state_every_epochs /
            # save_state_every_steps (multiple archived copies over time) --
            # no args gate -- this is the system-level pause safety net,
            # referenced by handle_interrupt when pausing.
            # Timing: placed after the user-opt epoch save, to ensure a
            # backup exists even when the user-opt setting is off.
            # Addendum 2: writes to auto_state_dir() (the task's own
            # archive, tasks/<id>/state/; CLI falls back to state_dir()) --
            # the user's periodic save still goes through state_dir() above, unchanged.
            auto_state_path = build_auto_epoch_state_path(ctx.auto_state_dir())
            auto_config_path = build_auto_epoch_config_path(ctx.auto_state_dir())
            monitor_data = None
            if ctx.monitor_server:
                try:
                    from train_monitor import get_state
                    monitor_data = get_state()
                except Exception:
                    pass
            # Write the config snapshot first -- small and low failure probability
            write_config_snapshot(auto_config_path, args, ctx.sample_prompts)
            with optimizer_eval_mode(ctx.optimizer):
                save_training_state(
                    auto_state_path, ctx.injector, ctx.optimizer,
                    ctx.current_epoch, ctx.global_step, ctx.loss_history,
                    monitor_state=monitor_data, scheduler=ctx.scheduler,
                    timestep_sampler=ctx.timestep_sampler,
                    sra_aligner=ctx.sra_aligner,
                    scaler=ctx.scaler,
                    model_family=ctx.family.spec.family_id,
                    ema=ctx.ema,
                )
            ctx.wandb_monitor.upload_state_auto(auto_state_path)
            # Update ctx fields for handle_interrupt to use when emitting pause_state
            ctx.last_auto_epoch_state_path = auto_state_path
            ctx.last_auto_epoch_config_path = auto_config_path
            # The supervisor's `_on_line` catches this event -> marks
            # slot.last_auto_epoch_state_path -> the is_pausable upgrade
            # condition is met -> SSE unlocks the UI pause button (ADR
            # Addendum 1 SSUI)
            emit_event("auto_epoch_backup_written", {
                "state_path": str(auto_state_path),
                "config_path": str(auto_config_path),
                "epoch": ctx.current_epoch,
                "step": ctx.global_step,
            })

        # Check max_steps
        if args.max_steps and ctx.global_step >= args.max_steps:
            break
