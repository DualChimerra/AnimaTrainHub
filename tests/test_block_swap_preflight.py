"""Block swap preflight (training/block_swap_preflight.py).

The core ``evaluate`` is pure arithmetic: all budget inputs are passed in as
parameters, it never queries the system, touches CUDA, or reads weights, so
we can assert directly on numbers. ``run``'s test cases only cover the
"when does it decline to judge" anti-false-rejection main line.

Scenario numbers are taken from a real-machine target config: a 12GB card +
32GB RAM + Krea 2 (fp8 13.1GB / bf16 26.3GB, 28 layers, the backbone is
~94.5% of the full model's parameters).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training import block_swap_preflight as preflight  # noqa: E402
from training import sysmem  # noqa: E402


_GIB = 1024 ** 3
_TOTAL_BLOCKS = 28
#: Fraction of the full model's parameters held by Krea 2's 28-layer backbone (the rest is embedding / output layers, which can't be swapped out)
_BLOCK_SHARE = 0.945

_FP8_BYTES = int(13.1 * _GIB)
_BF16_BYTES = int(26.3 * _GIB)
#: Actual free VRAM on a 12GB card (driver + desktop take a bit)
_FREE_VRAM_12G = int(11.5 * _GIB)
#: Typical available RAM on a 32GB machine when training starts
_AVAIL_RAM_32G = int(24 * _GIB)


def _ratio(blocks: int) -> float:
    """Fraction of the full model's parameters covered by swapping N layers (layers are equal-sized, matching krea2's real structure)."""
    return min(max(blocks, 0), _TOTAL_BLOCKS) / _TOTAL_BLOCKS * _BLOCK_SHARE


def _evaluate(
    *,
    file_bytes=_FP8_BYTES,
    blocks_to_swap=0,
    free_vram=_FREE_VRAM_12G,
    avail_ram=_AVAIL_RAM_32G,
):
    return preflight.evaluate(
        file_bytes=file_bytes,
        blocks_to_swap=blocks_to_swap,
        total_blocks=_TOTAL_BLOCKS,
        ratio_fn=_ratio,
        free_vram_bytes=free_vram,
        avail_ram_bytes=avail_ram,
        vram_base_bytes=sysmem._VRAM_BASE_BYTES,
        pinned_limit_bytes=(
            None if avail_ram is None else sysmem.pinned_safe_limit(avail_ram)
        ),
    )


# ---------------------------------------------------------------------------
# Main line: the default blocks_to_swap=0 must be blocked on 12GB, with a recommendation given
# ---------------------------------------------------------------------------


def test_default_zero_swap_is_rejected_with_recommendation() -> None:
    """A 13.1GB fp8 base model without swap doesn't fit in 12GB -- exactly the "select and run" trap."""
    result = _evaluate(blocks_to_swap=0)

    assert result.checked and not result.ok
    assert result.recommended is not None
    assert 0 < result.recommended <= _TOTAL_BLOCKS
    assert "blocks_to_swap" in result.message


def test_recommendation_leaves_training_headroom() -> None:
    """The recommendation can't be just enough to fit the weights: LoRA / optimizer state / activations / dequant still need room.

    Otherwise the user follows the recommendation and OOMs 3 minutes in --
    worse than not giving a recommendation at all.
    """
    result = _evaluate(blocks_to_swap=0)
    weights_only = int(_FP8_BYTES * (1.0 - _ratio(result.recommended)))

    assert _FREE_VRAM_12G - weights_only >= preflight._RECOMMEND_HEADROOM_BYTES


def test_recommended_value_itself_passes_preflight() -> None:
    """The recommendation must be self-consistent: applying it as-is should pass (never recommend a value that's still rejected)."""
    recommended = _evaluate(blocks_to_swap=0).recommended

    assert _evaluate(blocks_to_swap=recommended).ok


def test_configured_swap_that_fits_passes() -> None:
    """fp8 + swapping 26 layers is the 12GB/32GB target config; it must pass straight through."""
    result = _evaluate(blocks_to_swap=26)

    assert result.checked and result.ok
    assert result.message == ""


# ---------------------------------------------------------------------------
# bf16 base model: a runnable setting exists on 12GB/32GB, but none of them leave training headroom
# ---------------------------------------------------------------------------


def test_bf16_on_32g_ram_rejects_26_blocks() -> None:
    """Swapping 26 layers puts pinned memory over the cap (24.8GB > 19.2GB) -- exactly the boundary for a 32GB machine."""
    result = _evaluate(file_bytes=_BF16_BYTES, blocks_to_swap=26)

    assert result.checked and not result.ok
    assert "锁定" in result.message


def test_bf16_on_32g_ram_recommends_a_tight_value_and_says_so() -> None:
    """bf16 on 12GB/32GB isn't entirely impossible, but headroom is zero -- the message must say so.

    Giving a bare number without mentioning "this only fits the weights"
    would make the user OOM mid-run after following it, thinking preflight
    lied to them.
    """
    result = _evaluate(file_bytes=_BF16_BYTES, blocks_to_swap=26)

    assert result.recommended is not None
    assert "刚好装下权重" in result.message
    assert "fp8" in result.message


def test_fp8_recommendation_is_not_flagged_tight() -> None:
    """Control: fp8 base model has a setting with enough headroom, and shouldn't carry a "tight" warning."""
    result = _evaluate(blocks_to_swap=0)

    assert "刚好装下权重" not in result.message


def test_no_workable_swap_count_when_ram_is_small() -> None:
    """bf16 + 16GB RAM: the pinned cap is squeezed to 8GB, so no setting works."""
    result = _evaluate(
        file_bytes=_BF16_BYTES, blocks_to_swap=26, avail_ram=int(12 * _GIB),
    )

    assert result.checked and not result.ok
    assert result.recommended is None
    assert "fp8" in result.message


def test_bf16_fits_comfortably_when_ram_is_large_enough() -> None:
    """Same bf16 base model, but on a large-RAM machine swapping enough layers makes it runnable."""
    result = _evaluate(
        file_bytes=_BF16_BYTES, blocks_to_swap=28, avail_ram=int(56 * _GIB),
    )

    assert result.ok


# ---------------------------------------------------------------------------
# Anti-false-rejection: never judge when information is incomplete
# ---------------------------------------------------------------------------


def test_unknown_free_vram_skips_judgement() -> None:
    result = _evaluate(blocks_to_swap=0, free_vram=None)

    assert not result.checked and result.ok


def test_missing_checkpoint_size_skips_judgement() -> None:
    result = _evaluate(file_bytes=0, blocks_to_swap=0)

    assert not result.checked


def test_unknown_ram_still_checks_vram_side() -> None:
    """Being unable to query RAM doesn't affect the VRAM-side judgement (each signal degrades independently)."""
    result = _evaluate(blocks_to_swap=0, avail_ram=None)

    assert result.checked and not result.ok


def test_unknown_ram_never_blames_pinned_memory() -> None:
    """When RAM is unknown, it must not be rejected for "pinned memory over the cap" -- that would be a fabricated criterion."""
    result = _evaluate(
        file_bytes=_BF16_BYTES, blocks_to_swap=28, avail_ram=None,
    )

    assert result.ok


def test_blocks_to_swap_above_total_is_clamped() -> None:
    """An old yaml / bare CLI might write a value above the layer count; evaluate against the cap instead of blowing up."""
    result = _evaluate(blocks_to_swap=999)

    assert result.checked and result.ok


# ---------------------------------------------------------------------------
# Shares the same watermark as check_pinned_budget (drift between the two = preflight says OK but the guard rejects on the spot)
# ---------------------------------------------------------------------------


def test_pinned_limit_agrees_with_guard(monkeypatch) -> None:
    avail = int(24 * _GIB)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: avail)
    limit = sysmem.pinned_safe_limit(avail)

    sysmem.check_pinned_budget(limit, blocks=26)  # right on the line: passes
    with pytest.raises(RuntimeError):
        sysmem.check_pinned_budget(limit + 1, blocks=26)


# ---------------------------------------------------------------------------
# run(ctx): skip conditions
# ---------------------------------------------------------------------------


#: Family protocol: both estimator methods accept a ``checkpoint_path``
#: keyword (only the weight file knows anima's layer count and parameter
#: distribution; krea2 accepts it but ignores it). The fake family follows
#: the same shape.
def _ctx(*, capabilities=frozenset({"block_swap"}), preflight_on=True):
    family = types.SimpleNamespace(
        spec=types.SimpleNamespace(capabilities=capabilities),
        swapped_param_ratio=lambda blocks, *, checkpoint_path=None: _ratio(blocks),
        swappable_blocks=lambda *, checkpoint_path=None: _TOTAL_BLOCKS,
    )
    args = types.SimpleNamespace(
        block_swap_preflight=preflight_on,
        blocks_to_swap=0,
        transformer_path="/nonexistent/model.safetensors",
    )
    return types.SimpleNamespace(args=args, family=family)


def test_run_is_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(sysmem, "_file_bytes", lambda _p: _FP8_BYTES)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: _FREE_VRAM_12G)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: _AVAIL_RAM_32G)

    preflight.run(_ctx(preflight_on=False))  # turned off, so it shouldn't raise


def test_run_is_noop_for_families_without_block_swap(monkeypatch) -> None:
    monkeypatch.setattr(sysmem, "_file_bytes", lambda _p: _FP8_BYTES)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: _FREE_VRAM_12G)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: _AVAIL_RAM_32G)

    preflight.run(_ctx(capabilities=frozenset({"masked_loss"})))


def test_run_raises_with_recommendation(monkeypatch) -> None:
    monkeypatch.setattr(sysmem, "_file_bytes", lambda _p: _FP8_BYTES)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: _FREE_VRAM_12G)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: _AVAIL_RAM_32G)

    with pytest.raises(RuntimeError, match="建议"):
        preflight.run(_ctx())


def test_run_survives_broken_family_estimate(monkeypatch) -> None:
    """Preflight is an auxiliary facility: if it errors internally it must let training through, not block it."""
    monkeypatch.setattr(sysmem, "_file_bytes", lambda _p: _FP8_BYTES)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: _FREE_VRAM_12G)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: _AVAIL_RAM_32G)

    def _boom(_blocks, *, checkpoint_path=None):
        raise RuntimeError("failed to construct the meta model")

    ctx = _ctx()
    ctx.family.swapped_param_ratio = _boom

    preflight.run(ctx)


def test_run_forwards_checkpoint_path_to_family(monkeypatch) -> None:
    """Only the checkpoint itself knows anima's layer count/ratio -- failing to pass it through rejects a runnable config.

    This regression-tests a specific failure mode: when ``checkpoint_path``
    isn't passed through, anima's ``swapped_param_ratio`` returns 0, from
    which preflight computes "swapping any number of layers saves no VRAM",
    judging an otherwise runnable 6GB config as failing.
    """
    monkeypatch.setattr(sysmem, "_file_bytes", lambda _p: _FP8_BYTES)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: _FREE_VRAM_12G)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: _AVAIL_RAM_32G)

    seen: list[str | None] = []

    def _ratio_spy(blocks, *, checkpoint_path=None):
        seen.append(checkpoint_path)
        return _ratio(blocks)

    def _blocks_spy(*, checkpoint_path=None):
        seen.append(checkpoint_path)
        return _TOTAL_BLOCKS

    ctx = _ctx()
    ctx.family.swapped_param_ratio = _ratio_spy
    ctx.family.swappable_blocks = _blocks_spy

    with pytest.raises(RuntimeError):
        preflight.run(ctx)

    assert seen, "preflight never asked the family for any estimate"
    assert all(p == "/nonexistent/model.safetensors" for p in seen)
