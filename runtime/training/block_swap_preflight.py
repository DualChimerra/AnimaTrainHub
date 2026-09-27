"""Block swap preflight: decide *before any weight loading* whether the current
blocks_to_swap can actually run, and fail fast with a recommended value if not.

Why this exists: ``blocks_to_swap`` defaults to 0, and the Krea 2 DiT is 13GB
even in fp8 -- on a 12GB card, "load the selected model and go" is guaranteed
to OOM, and that OOM happens after dataset scanning, latent caching and text
encoding have all finished, once the user has already waited several minutes,
with an error that's just "CUDA out of memory" and no hint which number to
change.

The two existing guardrails don't solve this:
- ``check_load_budget``: budgets by file size only **at load time**, and is
  disabled by default on the training side (too many false rejects, see
  secrets.TrainingSecretsConfig.ram_guard);
- ``check_pinned_budget``: only covers the **RAM** side, and only fires inside
  the loader, at the moment a swapped-out layer is about to be pinned.

This module combines both sides of the arithmetic, moves it ahead of loading,
and **produces a recommended value** instead of just rejecting. Pure
arithmetic, no CUDA access, no weight reads (just file size stat); on failure
it silently lets the run proceed.

Family-agnostic: any family that implements ``swapped_param_ratio`` /
``swappable_blocks`` (currently krea2 and anima, i.e. families with the
block_swap capability flag) takes part; everything else is skipped entirely.
Both methods accept a ``checkpoint_path`` keyword: for anima, the layer count
and parameter distribution are only known from the weight file itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional


logger = logging.getLogger(__name__)

_GIB = 1024 ** 3

#: VRAM headroom to reserve for the recommended value. Wider than
#: ``_VRAM_BASE_BYTES`` (3GB, the figure used by the "do the weights even
#: fit" guardrail) because training-time VRAM is more than just weights:
#: LoRA params + gradients + optimizer state, activations kept by gradient
#: checkpointing, temporary weights from per-layer dequant of an fp8 base
#: model, allocator fragmentation. Recommending a value that "just barely
#: fits the weights" and then letting the user OOM three minutes in is worse
#: than not recommending anything, so this is deliberately conservative.
_RECOMMEND_HEADROOM_BYTES = 5 * _GIB


@dataclass(frozen=True)
class SwapVerdict:
    """Budget verdict for one candidate blocks_to_swap value."""

    blocks: int
    vram_need: int
    pinned_need: int
    vram_ok: bool
    ram_ok: bool

    @property
    def ok(self) -> bool:
        return self.vram_ok and self.ram_ok


@dataclass(frozen=True)
class PreflightResult:
    """Preflight verdict. ``checked=False`` means not enough information to
    decide this time."""

    checked: bool
    ok: bool
    current: Optional[SwapVerdict] = None
    recommended: Optional[int] = None
    message: str = ""


def evaluate(
    *,
    file_bytes: int,
    blocks_to_swap: int,
    total_blocks: int,
    ratio_fn: Callable[[int], float],
    free_vram_bytes: Optional[int],
    avail_ram_bytes: Optional[int],
    vram_base_bytes: int,
    pinned_limit_bytes: Optional[int],
) -> PreflightResult:
    """Pure arithmetic core: given budget inputs, judge the current setting
    and search for a recommendation.

    The caller is responsible for gathering system queries (free VRAM /
    available RAM / file size) and passing them in -- that keeps this
    function side-effect free and directly testable with plain numbers.

    ``free_vram_bytes is None`` (no CUDA / query failed) -> no verdict.
    ``pinned_limit_bytes is None`` (RAM query failed) -> only check the VRAM
    side, the RAM side always passes.
    """
    if file_bytes <= 0 or total_blocks <= 0 or free_vram_bytes is None:
        return PreflightResult(checked=False, ok=True)

    def verdict(blocks: int) -> SwapVerdict:
        ratio = min(max(float(ratio_fn(blocks)), 0.0), 1.0)
        # Same arithmetic as check_load_budget on the VRAM side
        # (need x (1-ratio) + base), so the two guardrails never disagree
        # ("preflight says fine, load rejects it").
        vram_need = int(file_bytes * (1.0 - ratio)) + vram_base_bytes
        # The RAM side is an approximation: the loader's check_pinned_budget
        # reads the header to count the **actual** bytes of swapped-out
        # layers; here we estimate via param ratio x file size. For a
        # single-dtype checkpoint the two are nearly equal (the extra F32
        # scale in fp8_scaled is negligible). The loader remains the
        # authoritative check; this preflight is only responsible for an
        # early warning and a recommended value.
        pinned_need = int(file_bytes * ratio)
        return SwapVerdict(
            blocks=blocks,
            vram_need=vram_need,
            pinned_need=pinned_need,
            vram_ok=free_vram_bytes >= vram_need,
            ram_ok=pinned_limit_bytes is None or pinned_need <= pinned_limit_bytes,
        )

    current = verdict(min(max(int(blocks_to_swap), 0), total_blocks))
    if current.ok:
        return PreflightResult(checked=True, ok=True, current=current)

    # Recommend: the smallest swap count that satisfies "a wider training
    # headroom" (fewer swapped layers = faster). If no such value exists,
    # fall back to the minimum that "at least fits the weights" and flag it
    # as tight in the message.
    candidates = [verdict(n) for n in range(total_blocks + 1)]

    def has_training_headroom(v: SwapVerdict) -> bool:
        weights_only = v.vram_need - vram_base_bytes
        return free_vram_bytes >= weights_only + _RECOMMEND_HEADROOM_BYTES

    roomy = [v for v in candidates if v.ram_ok and has_training_headroom(v)]
    workable = [v for v in candidates if v.ok]
    if roomy:
        recommended, tight = roomy[0].blocks, False
    elif workable:
        # There's a value that fits the weights, but none leaves enough
        # training headroom. Still recommend it (it's the best this machine
        # can offer), but the message must say it's tight -- OOMing three
        # minutes into training with no warning is worse than not
        # recommending at all.
        recommended, tight = workable[0].blocks, True
    else:
        recommended, tight = None, False

    return PreflightResult(
        checked=True,
        ok=False,
        current=current,
        recommended=recommended,
        message=_message(
            current=current,
            recommended=recommended,
            tight=tight,
            candidates=candidates,
            total_blocks=total_blocks,
            free_vram_bytes=free_vram_bytes,
            avail_ram_bytes=avail_ram_bytes,
            pinned_limit_bytes=pinned_limit_bytes,
        ),
    )


def _gb(value: Optional[int]) -> str:
    return "unknown" if value is None else f"{value / _GIB:.1f}GB"


def _message(
    *,
    current: SwapVerdict,
    recommended: Optional[int],
    tight: bool,
    candidates: list,
    total_blocks: int,
    free_vram_bytes: int,
    avail_ram_bytes: Optional[int],
    pinned_limit_bytes: Optional[int],
) -> str:
    head = (
        f"block swap preflight failed: at blocks_to_swap={current.blocks} the "
        f"base model needs approx {_gb(current.vram_need)} VRAM, "
        f"{_gb(current.pinned_need)} pinned RAM; "
        f"currently free VRAM {_gb(free_vram_bytes)}"
    )
    if pinned_limit_bytes is not None:
        head += f", available RAM {_gb(avail_ram_bytes)} (pinnable cap {_gb(pinned_limit_bytes)})"
    head += "."

    if recommended is not None:
        body = (
            f"{head}\n"
            f"  Suggestion: set blocks_to_swap to {recommended} (out of {total_blocks} layers). "
            f"Swapped-out layers stay resident in RAM and are moved to VRAM only when needed; "
            f"this does not affect training results, just makes it a bit slower."
        )
        if tight:
            body += (
                f"\n  Note: this value only **just barely fits the weights** -- this "
                f"machine has no setting that also leaves enough training headroom "
                f"(LoRA + optimizer state + activations + dequant temp weights). "
                f"You may still OOM mid-training -- a more reliable fix is switching "
                f"to an fp8 base model (half the size), or lowering the training resolution."
            )
        return (
            f"{body}\n"
            f"  If you're confident this estimate is wrong, you can disable this check "
            f"at the \"block swap preflight\" toggle."
        )

    # No candidate works at all: distinguish whether VRAM or RAM is the
    # blocker and give a different fix for each.
    vram_reachable = any(v.vram_ok for v in candidates)
    if not vram_reachable:
        why = (
            f"Even swapping out all {total_blocks} layers, the non-swappable parts "
            f"(embedding / output layers, etc.) still need approx "
            f"{_gb(candidates[total_blocks].vram_need)} VRAM."
        )
        fix = "Switch to an fp8 base model (half the size), or close other programs using VRAM (ComfyUI, image generation jobs)."
    else:
        why = (
            f"VRAM is fine, but the RAM that swapped-out layers need to pin exceeds "
            f"the cap (needs at least {_gb(min(v.pinned_need for v in candidates if v.vram_ok))} pinned). "
            f"Pinned RAM can't be paged out, and going over the cap can bring down the whole machine."
        )
        fix = "Switch to an fp8 base model (halves pinned RAM too), or close other programs using RAM and retry."
    return f"{head}\n  {why}\n  Fix: {fix}"


def run(ctx) -> None:
    """Called from models_phase: raises RuntimeError (with a recommended
    value) if the check fails.

    Silently skipped in these cases (better to skip than to falsely reject):
    the toggle is off, the family has no block_swap capability, the family
    doesn't implement the ratio/block-count estimators, the weight file size
    can't be read, or the VRAM query fails.
    """
    args = ctx.args
    if not bool(getattr(args, "block_swap_preflight", True)):
        logger.info("block swap preflight disabled (block_swap_preflight=false)")
        return
    family = ctx.family
    if family is None or "block_swap" not in family.spec.capabilities:
        return
    ratio_fn = getattr(family, "swapped_param_ratio", None)
    blocks_fn = getattr(family, "swappable_blocks", None)
    if ratio_fn is None or blocks_fn is None:
        return

    from training import sysmem

    # checkpoint_path is a cross-family protocol parameter: for anima, the
    # layer count and parameter distribution are determined by the
    # checkpoint (2B=28 layers / 14B=36 layers); without it we can only
    # return 0, which would make the preflight misjudge "VRAM is fine once
    # swapped" as "nothing can be saved at all" and reject a configuration
    # that would actually run. krea2's structure is fixed, so it accepts
    # and ignores this.
    transformer_path = str(getattr(args, "transformer_path", "") or "")

    def ratio_at(blocks: int) -> float:
        return float(ratio_fn(blocks, checkpoint_path=transformer_path))

    try:
        total_blocks = int(blocks_fn(checkpoint_path=transformer_path))
        file_bytes = sysmem._file_bytes([getattr(args, "transformer_path", "")])
        avail_ram = sysmem.available_ram_bytes()
        result = evaluate(
            file_bytes=file_bytes,
            blocks_to_swap=int(getattr(args, "blocks_to_swap", 0) or 0),
            total_blocks=total_blocks,
            ratio_fn=ratio_at,
            free_vram_bytes=sysmem.gpu_free_bytes_global(),
            avail_ram_bytes=avail_ram,
            vram_base_bytes=sysmem._VRAM_BASE_BYTES,
            pinned_limit_bytes=(
                None if avail_ram is None else sysmem.pinned_safe_limit(avail_ram)
            ),
        )
    except Exception:  # noqa: BLE001
        # The preflight is a helper facility; an error in it must never
        # block training itself.
        logger.warning("block swap preflight failed, skipping", exc_info=True)
        return

    if not result.checked:
        return
    if not result.ok:
        raise RuntimeError(result.message)
    if result.current is not None and result.current.blocks > 0:
        logger.info(
            "block swap preflight passed: swapping out %d layers, base model uses approx %s VRAM, %s pinned RAM",
            result.current.blocks,
            _gb(result.current.vram_need),
            _gb(result.current.pinned_need),
        )
