"""Budget guardrails for block swap (docs/design/block-swap.md §3.2 ① / knife 3).

Two independent budgets with different semantics that cannot substitute for each other:
- ``check_load_budget``'s VRAM side: swapped-out layers **never go on the GPU**, so they
  must be discounted -- otherwise a small-VRAM card with swap fully enabled would be
  wrongly rejected as if "the full model doesn't fit."
- ``check_pinned_budget``: swapped-out layers are locked in RAM, where
  ``trim_working_set`` has no effect on them; gated by a safe fraction of available
  physical memory.

Doesn't need CUDA (pure budget arithmetic + monkeypatched query functions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training import sysmem  # noqa: E402

_GIB = 1024 ** 3


@pytest.fixture
def fake_env(monkeypatch):
    """Replace the RAM / VRAM queries and file size with controllable values."""

    def _apply(*, ram_gb: float, vram_gb: float, file_gb: float):
        monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: int(ram_gb * _GIB))
        monkeypatch.setattr(
            sysmem, "gpu_free_bytes_global", lambda: int(vram_gb * _GIB)
        )
        monkeypatch.setattr(sysmem, "_file_bytes", lambda _paths: int(file_gb * _GIB))

    return _apply


def test_vram_discount_lets_small_card_pass(fake_env):
    """16GB card + 25.8GB bf16 model + 94.8% swapped out -> should pass (this is exactly B12's scenario).

    Without the discount, the full-model calculation needs 25.8+3=28.8GB and would be wrongly rejected.
    """
    fake_env(ram_gb=64, vram_gb=15.0, file_gb=25.8)

    # no discount: rejected
    with pytest.raises(RuntimeError, match="Not enough free GPU VRAM"):
        sysmem.check_load_budget(True, weight_paths=["x"], stage="test")

    # discount the 94.8% swapped out (all 28 layers swapped): resident 1.3GB + base 3GB < 15GB, passes
    sysmem.check_load_budget(
        True, weight_paths=["x"], stage="test", vram_discount_ratio=0.9482,
    )


def test_vram_discount_is_ratio_so_fp8_is_not_over_discounted(fake_env):
    """**Regression**: the discount must be proportional, not based on the compute dtype's byte count.

    An fp8 checkpoint file is only half the size of bf16 (13GB). If the discount
    subtracted the bf16-estimated byte count (22.64GB), vram_need would be pushed to 0 --
    the guardrail would be completely ineffective for fp8. Proportionally, fp8 resident
    = 13 x (1-0.948) = 0.7GB, matching reality.
    """
    fake_env(ram_gb=64, vram_gb=3.0, file_gb=13.0)  # fp8 file is 13GB, the card only has 3.0GB left

    # proportional discount: needs 0.7+3=3.7GB > 3.0GB available -> correctly rejected
    with pytest.raises(RuntimeError, match="Not enough free GPU VRAM"):
        sysmem.check_load_budget(
            True, weight_paths=["x"], stage="test", vram_discount_ratio=0.9482,
        )
    # the ratio is clamped to [0,1], so a bogus input value can't blow the guardrail's discount through
    with pytest.raises(RuntimeError, match="Not enough free GPU VRAM"):
        sysmem.check_load_budget(
            True, weight_paths=["x"], stage="test", vram_discount_ratio=-5.0,
        )


def test_vram_discount_still_rejects_when_genuinely_short(fake_env):
    """The discount isn't a free pass: it still rejects when the resident part alone doesn't fit."""
    fake_env(ram_gb=64, vram_gb=4.0, file_gb=25.8)
    with pytest.raises(RuntimeError, match="Not enough free GPU VRAM"):
        sysmem.check_load_budget(
            True, weight_paths=["x"], stage="test", vram_discount_ratio=0.5,
        )


def test_vram_discount_does_not_relax_ram_side(fake_env):
    """The discount only applies to the VRAM side -- swapped-out layers still occupy RAM, so the RAM budget is computed as usual."""
    fake_env(ram_gb=8, vram_gb=80, file_gb=25.8)
    with pytest.raises(RuntimeError, match="Not enough available system RAM"):
        sysmem.check_load_budget(
            True, weight_paths=["x"], stage="test", vram_discount_ratio=0.9482,
        )


def test_swapped_bytes_use_checkpoint_dtype_not_compute_dtype(tmp_path):
    """**Regression**: the pinned budget must be computed from the checkpoint's
    actual dtype, not the compute dtype.

    An fp8 checkpoint is only half the size of bf16. Estimating from bf16
    would count 28 layers as double, hitting the 60% safety line on a
    machine with plenty of RAM and being **wrongly rejected** -- exactly
    blocking B12's target config. Overestimating here isn't conservative,
    it's a false negative.
    """
    import torch
    from safetensors.torch import save_file

    from training.families.krea2 import loader as L

    # build one fp8 and one bf16 mini checkpoint whose keys carry a blocks.N. prefix
    n2s = {}
    fp8_t, bf16_t = {}, {}
    for i in range(4):
        key = f"blocks.{i}.w"
        n2s[key] = key
        fp8_t[key] = torch.zeros(256, 256, dtype=torch.float8_e4m3fn)
        bf16_t[key] = torch.zeros(256, 256, dtype=torch.bfloat16)
    fp8_path = tmp_path / "fp8.safetensors"
    bf16_path = tmp_path / "bf16.safetensors"
    save_file(fp8_t, str(fp8_path))
    save_file(bf16_t, str(bf16_path))

    prefixes = ("blocks.2.", "blocks.3.")  # trailing 2 layers
    fp8_bytes = L._swapped_bytes_from_checkpoint(fp8_path, prefixes, n2s)
    bf16_bytes = L._swapped_bytes_from_checkpoint(bf16_path, prefixes, n2s)

    assert fp8_bytes == 2 * 256 * 256 * 1
    assert bf16_bytes == 2 * 256 * 256 * 2
    assert bf16_bytes == 2 * fp8_bytes  # exactly the factor that was being miscalculated


def test_swapped_bytes_falls_back_when_header_unreadable(tmp_path):
    """When the header can't be read, returns 0 so the caller falls back to estimating from the compute dtype (never silently passes)."""
    from training.families.krea2 import loader as L

    bogus = tmp_path / "not-a-safetensors.bin"
    bogus.write_bytes(b"garbage")
    assert L._swapped_bytes_from_checkpoint(bogus, ("blocks.0.",), {"blocks.0.w": "w"}) == 0


def test_pinned_budget_rejects_over_safe_fraction(monkeypatch):
    """Large-RAM machine: the 80% ratio is the side that governs."""
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 40 * _GIB)
    # safe cap = min(40 x 0.8, 40 - 4) = min(32, 36) = 32GB
    sysmem.check_pinned_budget(int(31 * _GIB), blocks=28)
    with pytest.raises(RuntimeError, match="Not enough memory to swap out"):
        sysmem.check_pinned_budget(int(33 * _GIB), blocks=28)


def test_pinned_budget_absolute_floor_protects_small_ram(monkeypatch):
    """**Small-RAM machine**: the 4GB absolute floor is the side that governs; a pure ratio would crush the machine.

    With 10GB available, a pure 80% would allow pinning 8GB, leaving only
    2GB for the training process's own non-pinned part (base ~4GB) ->
    paging. Taking the min gives a 6GB cap.
    """
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 10 * _GIB)
    # safe cap = min(10 x 0.8, 10 - 4) = min(8, 6) = 6GB
    sysmem.check_pinned_budget(int(5.5 * _GIB), blocks=14)
    with pytest.raises(RuntimeError, match="Not enough memory to swap out"):
        sysmem.check_pinned_budget(int(7 * _GIB), blocks=14)


def test_pinned_budget_real_scenario_fp8_28_layers(monkeypatch):
    """Regression from a real user machine: 37.5GB available + fp8 28 layers (11.3GB) should pass.

    The old rule (60% + a bf16-estimated 22.6GB) wrongly rejected this; only passes once both are fixed.
    """
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: int(37.5 * _GIB))
    sysmem.check_pinned_budget(int(11.32 * _GIB), blocks=28)
    # bf16 28 layers at 22.65GB now also passes on the same machine (cap 30GB)
    sysmem.check_pinned_budget(int(22.65 * _GIB), blocks=28)


def test_pinned_budget_message_is_actionable(monkeypatch):
    """B6: the error doesn't silently degrade, and the message should tell the user what to do."""
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 8 * _GIB)
    with pytest.raises(RuntimeError) as exc:
        sysmem.check_pinned_budget(int(20 * _GIB), blocks=28)
    msg = str(exc.value)
    assert "28" in msg
    assert "blocks_to_swap" in msg
    assert "pinned" in msg


def test_pinned_budget_silent_when_query_fails(monkeypatch):
    """A failed query silently passes (consistent with the other guards' rule -- never block training just because probing failed)."""
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: None)
    sysmem.check_pinned_budget(int(999 * _GIB), blocks=28)


def test_pinned_budget_noop_for_zero():
    sysmem.check_pinned_budget(0, blocks=0)
