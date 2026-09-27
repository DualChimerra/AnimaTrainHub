
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training import block_swap as bs  # noqa: E402
from training.block_swap import PinnedAllocationError, PinnedPacker  # noqa: E402

_MIB = 1024 ** 2


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class _FakeAlloc:

    def __init__(self, fail_at_total: int | None = None) -> None:
        self.sizes: list[int] = []
        self.fail_at_total = fail_at_total

    def __call__(self, nbytes: int) -> torch.Tensor:
        if self.fail_at_total is not None and sum(self.sizes) + nbytes > self.fail_at_total:
            raise RuntimeError("CUDA error: out of memory")
        self.sizes.append(nbytes)
        return torch.empty(nbytes, dtype=torch.uint8)


def test_pow2_helpers():
    assert [bs._pow2_ceil(n) for n in (0, 1, 2, 3, 4, 5, 1000)] == [1, 1, 2, 4, 4, 8, 1024]
    assert bs._pow2_decomposition(0) == []
    assert bs._pow2_decomposition(11) == [8, 2, 1]
    assert bs._pow2_decomposition(64 * _MIB) == [64 * _MIB]


def test_preallocates_binary_decomposition_of_total():
    alloc = _FakeAlloc()
    total = 100 * _MIB
    packer = PinnedPacker(total, granularity=64 * _MIB, allocate=alloc)
    assert alloc.sizes == [128 * _MIB]
    assert packer.num_chunks == 1 and packer.allocated_bytes == 128 * _MIB

    alloc2 = _FakeAlloc()
    total2 = 11 * 1024 * _MIB + 300 * _MIB
    packer2 = PinnedPacker(total2, granularity=64 * _MIB, allocate=alloc2)
    assert alloc2.sizes == [8192 * _MIB, 2048 * _MIB, 1024 * _MIB, 256 * _MIB, 64 * _MIB]
    assert all(_is_pow2(s) for s in alloc2.sizes)
    assert packer2.allocated_bytes - total2 < 64 * _MIB


def test_pin_returns_aligned_views_with_same_content_dtype_shape():
    alloc = _FakeAlloc()
    tensors = [
        torch.randn(37, 53, dtype=torch.float32),
        torch.randn(5, dtype=torch.bfloat16),
        torch.tensor(3.5, dtype=torch.float32),
        torch.randn(0, 8, dtype=torch.float16),
        torch.randn(16, 16).to(torch.float8_e4m3fn),
        torch.randn(9, 7, dtype=torch.float32).t(),
    ]
    total = sum(t.numel() * t.element_size() for t in tensors)
    packer = PinnedPacker(total, allocate=alloc)
    outs = [packer.pin(t) for t in tensors]

    assert packer.num_chunks == 1 and packer.overflow_chunks == 0
    chunk_ptr = packer._chunks[0][0].data_ptr()
    for src, out in zip(tensors, outs):
        assert out.shape == src.shape and out.dtype == src.dtype
        assert out.is_contiguous()
        if src.dtype == torch.float8_e4m3fn:
            assert torch.equal(out.view(torch.uint8), src.contiguous().view(torch.uint8))
        else:
            assert torch.equal(out, src)
        storage_ptr = out.untyped_storage().data_ptr()
        assert storage_ptr == chunk_ptr
        if out.numel():
            assert (out.data_ptr() - chunk_ptr) % PinnedPacker.ALIGN == 0
    assert packer.packed_bytes == total

    outs[0].fill_(1.0)
    assert torch.equal(outs[0], torch.ones_like(tensors[0]))


def test_pin_with_dtype_casts_on_the_fly():
    alloc = _FakeAlloc()
    src = torch.randn(8, 8, dtype=torch.float32)
    packer = PinnedPacker(src.numel() * 2, allocate=alloc)
    out = packer.pin(src, dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, src.to(torch.bfloat16))
    assert packer.packed_bytes == src.numel() * 2


def test_overflow_opens_pow2_chunk_never_worse_than_per_tensor_pin():
    alloc = _FakeAlloc()
    packer = PinnedPacker(64 * _MIB, allocate=alloc)
    big = torch.empty(100 * _MIB, dtype=torch.uint8)
    out = packer.pin(big)
    assert out.shape == big.shape
    assert packer.overflow_chunks == 1
    assert alloc.sizes == [64 * _MIB, 128 * _MIB]
    small = torch.empty(10 * _MIB, dtype=torch.uint8)
    out2 = packer.pin(small)
    ptr2 = out2.untyped_storage().data_ptr()
    assert ptr2 == packer._chunks[1][0].data_ptr()
    assert alloc.sizes == [64 * _MIB, 128 * _MIB]


def test_zero_total_means_lazy_allocation():
    alloc = _FakeAlloc()
    packer = PinnedPacker(0, allocate=alloc)
    assert alloc.sizes == [] and packer.num_chunks == 0
    packer.pin(torch.empty(3 * _MIB, dtype=torch.uint8))
    assert alloc.sizes == [4 * _MIB]


def test_allocation_failure_is_actionable_and_fail_fast():
    alloc = _FakeAlloc(fail_at_total=1024 * _MIB)
    with pytest.raises(PinnedAllocationError) as info:
        PinnedPacker(4096 * _MIB, allocate=alloc)
    msg = str(info.value)
    assert "blocks_to_swap" in msg and "not insufficient VRAM" in msg and "GB" in msg
    assert isinstance(info.value, RuntimeError)


def _krea2_swapped_sizes(blocks_to_swap: int, *, fp8: bool) -> list[int]:
    from training.families.krea2.loader import (
        KREA2_CONFIG, SingleStreamDiT, _swapped_block_prefixes,
    )

    prefixes = _swapped_block_prefixes(KREA2_CONFIG, blocks_to_swap)
    with torch.device("meta"):
        probe = SingleStreamDiT(KREA2_CONFIG)
    sizes = []
    for name, p in probe.named_parameters():
        if not name.startswith(prefixes):
            continue
        per_elem = 1 if (fp8 and p.dim() == 2) else 2
        sizes.append(p.numel() * per_elem)
    return sizes


@pytest.mark.parametrize("blocks,fp8", [(28, True), (18, True), (14, True), (28, False)])
def test_real_krea2_layout_packs_within_three_percent(blocks: int, fp8: bool):
    sizes = _krea2_swapped_sizes(blocks, fp8=fp8)
    raw = sum(sizes)
    per_tensor_pow2 = sum(bs._pow2_ceil(n) for n in sizes)
    assert per_tensor_pow2 / raw > 1.4

    alloc = _ZeroStorageAlloc()
    packer = PinnedPacker(raw, allocate=alloc)
    for n in sizes:
        packer._reserve(n)
    assert all(_is_pow2(s) for s in alloc.sizes)
    assert packer.packed_bytes == raw
    assert packer.allocated_bytes == sum(alloc.sizes)
    assert packer.allocated_bytes / raw <= 1.03, (
        f"{blocks} layers fp8={fp8}: actually locked {packer.allocated_bytes / raw:.3f}x the weights"
    )


class _ZeroStorageAlloc(_FakeAlloc):
    def __call__(self, nbytes: int) -> torch.Tensor:
        self.sizes.append(nbytes)
        return torch.empty(1, dtype=torch.uint8).expand(nbytes)
