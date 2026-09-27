"""Core unit tests for the block swap mechanism (docs/design/block-swap.md §9 knife 1).

Covers:
- Numerical correctness: swap forward/backward matches fully-resident bit-for-bit
- In-place swap semantics: module identity is preserved (a precondition for LoRA
  compatibility), param.data points into a slot
- fp8 case: the non-persistent weight_scale buffer always stays paired with its weight
- Boundaries: num_swap validation, no-op for non-swapped layers
- Allocation failure: BlockSwapAllocationError carries context

The whole file is skipped without CUDA (block swap is a CUDA-only mechanism).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="block swap is a CUDA-only mechanism"
)


def _import():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "runtime"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from training.block_swap import BlockSwapAllocationError, PinnedBlockSwap

    return PinnedBlockSwap, BlockSwapAllocationError


class _Tiny(nn.Module):
    """A structurally-identical, stackable small block (avoids depending on real krea2 weights)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(torch.relu(self.lin1(x)))


def _make_blocks(n: int, dim: int, device) -> nn.ModuleList:
    """Base model frozen -- matches the real flow (loader calls model.requires_grad_(False)).

    The component only manages frozen base weights; trainable params (LoRA) aren't
    its concern, so the test baseline must be frozen, otherwise nothing gets swapped out.
    """
    torch.manual_seed(0)
    blocks = nn.ModuleList([_Tiny(dim) for _ in range(n)])
    blocks.requires_grad_(False)
    return blocks.to(device)


def _run_resident(blocks, x):
    h = x
    for block in blocks:
        h = block(h)
    return h


def _run_swap(swap, blocks, x):
    h = x
    for i, block in swap.iter_forward():
        h = block(h)
    return h


def test_forward_matches_resident():
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    dim, n = 32, 6
    blocks = _make_blocks(n, dim, device)
    x = torch.randn(2, dim, device=device)

    expected = _run_resident(blocks, x)

    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)
    got = _run_swap(swap, blocks, x)

    torch.testing.assert_close(got, expected)


def test_module_identity_preserved():
    """The core guarantee of in-place swap: block/Linear objects don't change -- a precondition for LoRA compatibility."""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)

    ids_before = [id(b) for b in blocks]
    lin_ids_before = [id(b.lin1) for b in blocks]

    x = torch.randn(1, 16, device=device)
    _run_swap(swap, blocks, x)

    assert [id(b) for b in blocks] == ids_before
    assert [id(b.lin1) for b in blocks] == lin_ids_before


def test_lora_style_forward_hook_not_bypassed():
    """Simulate a LyCORIS bypass: wrap Linear with an extra addition. swap must still go through this layer.

    If swap rotated buffers (swapping module instances), this hook would be bypassed --
    this test pins down exactly the "silently learns nothing" trap from doc §9.1.
    """
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)

    marker = {"calls": 0}

    def hook(_module, _inp, out):
        marker["calls"] += 1
        return out + 1.0  # LoRA-like extra contribution

    handles = [b.lin1.register_forward_hook(hook) for b in blocks]

    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)
    x = torch.randn(1, 16, device=device)
    _run_swap(swap, blocks, x)

    for h in handles:
        h.remove()
    # each of the 4 blocks' lin1 should be hit by the hook exactly once (including the 2 swapped-out layers)
    assert marker["calls"] == 4


def test_backward_grad_flows_through_swapped_blocks():
    """With the base model frozen, gradients must still flow through swapped-out layers to the input (LoRA training depends on this path)."""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(6, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)
    swap.attach()

    x = torch.randn(3, 16, device=device, requires_grad=True)
    h = x
    for block in blocks:
        h = block(h)
    h.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_num_swap_validation():
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 8, device)

    with pytest.raises(ValueError, match="num_swap must be positive"):
        PinnedBlockSwap(blocks, num_swap=0, device=device)
    with pytest.raises(ValueError, match="exceeds the total block count"):
        PinnedBlockSwap(blocks, num_swap=5, device=device)
    with pytest.raises(ValueError, match="num_slots must be at least 2"):
        PinnedBlockSwap(blocks, num_swap=2, device=device, num_slots=1)


def test_non_swapped_layers_noop():
    """The first N-num_swap layers' ensure_resident/release are no-ops, leaving their params untouched."""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)

    # blocks 0/1/2 are resident (first_swapped == 3)
    assert swap.first_swapped == 3
    resident_ptr = blocks[0].lin1.weight.data_ptr()
    swap.ensure_resident(0)
    swap.release(0)
    assert blocks[0].lin1.weight.data_ptr() == resident_ptr


def test_pinned_bytes_accounting():
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    dim, n, num_swap = 16, 5, 3
    blocks = _make_blocks(n, dim, device)
    swap = PinnedBlockSwap(blocks, num_swap=num_swap, device=device)

    # each _Tiny: 2 Linears, each dim*dim weights + dim bias, fp32
    per_block = num_swap * 2 * (dim * dim + dim) * 4
    assert swap.pinned_bytes == per_block


def test_fp8_scale_stays_paired():
    """fp8 case: weight_scale is a non-persistent buffer bound to the module; after
    swapping weights in place, the scale must still be paired with that module's weight
    (module unchanged -> automatically correct)."""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")

    class _ScaledBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.lin = nn.Linear(dim, dim)
            # simulate patch_fp8_linears: non-persistent buffer
            self.lin.register_buffer(
                "weight_scale", torch.tensor(2.0, device=device), persistent=False
            )

        def forward(self, x):
            w = self.lin.weight * self.lin.weight_scale
            return torch.nn.functional.linear(x, w, self.lin.bias)

    torch.manual_seed(1)
    blocks = nn.ModuleList([_ScaledBlock(16) for _ in range(4)]).to(device)
    x = torch.randn(2, 16, device=device)
    expected = _run_resident(blocks, x)

    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    got = _run_swap(swap, blocks, x)

    # scale wasn't corrupted by the move, forward still matches bit-for-bit
    torch.testing.assert_close(got, expected)
    for b in blocks:
        assert b.lin.weight_scale.item() == 2.0


def test_shape_readable_after_construct():
    """After construction (weights already moved to CPU), shape/dtype are still readable
    correctly -- LyCORIS reads the base weight shape when injecting after swap; pointing
    to empty(0) would make it read the wrong shape (doc §9.1)."""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    dim = 24
    blocks = _make_blocks(4, dim, device)
    shapes_before = [tuple(b.lin1.weight.shape) for b in blocks]

    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)

    for i, block in enumerate(blocks):
        assert tuple(block.lin1.weight.shape) == shapes_before[i]
        assert block.lin1.weight.dtype == torch.float32
    # the swapped-out layer's weight should now be on CPU (VRAM freed), resident layers still on GPU
    assert blocks[swap.first_swapped].lin1.weight.device.type == "cpu"
    assert blocks[0].lin1.weight.device.type == "cuda"


def test_attach_hooks_forward_matches_resident():
    """attach()'s hook path: forward values match fully-resident, without needing to change the model loop."""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(6, 32, device)
    x = torch.randn(2, 32, device=device)
    expected = _run_resident(blocks, x)

    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)
    swap.attach()
    got = _run_resident(blocks, x)  # plain loop, the hook takes over automatically

    torch.testing.assert_close(got, expected)


def test_attach_with_gradient_checkpointing_backward():
    """**Core claim verification**: with gradient checkpointing on, the backward recompute
    triggers block forward again, and the pre-hook swaps weights back in in reverse
    order -- so backward needs no separate orchestration.

    The control group is a fully-resident model with the same weights, comparing
    gradients of LoRA-style trainable params.
    """
    from torch.utils.checkpoint import checkpoint

    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    dim, n = 32, 6
    blocks = _make_blocks(n, dim, device)

    # simulate LoRA: attach a small trainable param to each block, base model frozen
    for b in blocks:
        b.lora = nn.Parameter(torch.ones(dim, device=device) * 0.1)
    for b in blocks:
        b.lin1.requires_grad_(False)
        b.lin2.requires_grad_(False)

    def run(use_checkpoint: bool):
        h = torch.randn(2, dim, device=device, generator=torch.Generator(device).manual_seed(7))
        for b in blocks:
            def fwd(t, blk=b):
                return blk(t) * blk.lora
            h = checkpoint(fwd, h, use_reentrant=False) if use_checkpoint else fwd(h)
        return h.sum()

    # baseline: fully resident + checkpoint
    run(True).backward()
    expected = [b.lora.grad.clone() for b in blocks]
    for b in blocks:
        b.lora.grad = None

    # swap + attach + checkpoint
    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)
    swap.attach()
    run(True).backward()

    for i, b in enumerate(blocks):
        assert b.lora.grad is not None, f"block {i} has no gradient"
        torch.testing.assert_close(b.lora.grad, expected[i], msg=f"block {i} gradient mismatch")


def test_trainable_params_are_not_managed():
    """Trainable params (LoRA) must stay in place, resident on GPU -- not swapped out as base weights."""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)
    for b in blocks:
        b.lin1.requires_grad_(False)
        b.lin2.requires_grad_(False)
        b.lora = nn.Parameter(torch.ones(16, device=device))  # trainable

    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)

    for b in list(blocks)[swap.first_swapped:]:
        assert b.lora.device.type == "cuda", "the LoRA param was incorrectly swapped to CPU"
        assert b.lin1.weight.device.type == "cpu", "the frozen base weight should already be on CPU"
    # lora should not appear in the registered spec
    for rel in range(swap.num_swap):
        names = [n for n, _s, _d in swap._param_specs[rel]]
        assert "lora" not in names


def test_params_added_after_construct_do_not_break_rebind():
    """Params added after construction (the case where LoRA builds a submodule inside
    the block) should not crash the swap-in.

    Regression: _rebind used to iterate named_parameters() and look up buf[name]
    directly -> KeyError.
    """
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)
    for b in blocks:
        b.requires_grad_(False)

    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)
    # attach the param after construction
    for b in blocks:
        b.late = nn.Parameter(torch.ones(16, device=device))
    swap.attach()

    x = torch.randn(1, 16, device=device)
    h = x
    for b in blocks:
        h = b(h)
    assert torch.isfinite(h).all()


def test_detach_removes_hooks():
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)
    swap.attach()
    swap.attach()  # idempotent
    # 4 hooks per swapped-out block: forward pre/post + backward pre/post. The backward
    # two can't be skipped -- missing them silently miscomputes gradients (see test_block_swap_grad_fidelity.py)
    assert len(swap._handles) == 2 * 4
    swap.detach()
    assert swap._handles == []


def test_adopts_cpu_weights_without_recopy():
    """When the loader has already placed the tail layers on pinned CPU memory, the component takes over in place without recopying."""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)
    # simulate the loader: put the last 2 layers on pinned CPU memory
    for b in list(blocks)[2:]:
        for p in b.parameters():
            p.data = p.detach().to("cpu").pin_memory()
    ptrs = {id(p): p.data_ptr() for b in list(blocks)[2:] for p in b.parameters()}

    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)

    # the master copy should be the same pinned tensors as before (not reallocated)
    for rel in range(swap.num_swap):
        for _name, t in swap._cpu_weights[rel].items():
            assert t.is_pinned()
    for b in list(blocks)[2:]:
        for p in b.parameters():
            assert p.data_ptr() == ptrs[id(p)]


def test_fp8_base_with_swap_forward_matches_resident():
    """fp8 base model + block swap (B7's core combination): goes through the real patch_fp8_linears.

    Pins down two things:
    - fp8 weights can be pinned / H2D-transferred (dtype unchanged, no cast)
    - weight_scale must stay resident on the compute device. If it followed
      module.weight.device, patching a swapped-out layer with the weight on CPU would
      leave scale on CPU -> mismatched device once the weight is back on GPU for forward.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "runtime"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from training.families.krea2.quant_fp8 import patch_fp8_linears

    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    dim, n = 32, 6

    class _Fp8Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return x + self.lin(x)

    torch.manual_seed(3)
    blocks = nn.ModuleList([_Fp8Block() for _ in range(n)]).to(device)
    blocks.requires_grad_(False)
    # convert to fp8 storage + per-layer scale (simulate an fp8_scaled checkpoint)
    scales = {}
    for i, b in enumerate(blocks):
        b.lin.weight.data = b.lin.weight.data.to(torch.float8_e4m3fn)
        scales[f"{i}.lin"] = torch.tensor(0.5, device=device)
    patch_fp8_linears(blocks, scales, device=device)

    x = torch.randn(2, dim, device=device, dtype=torch.bfloat16)
    expected = _run_resident(blocks, x)

    swap = PinnedBlockSwap(blocks, num_swap=4, device=device)
    swap.attach()
    got = _run_resident(blocks, x)

    torch.testing.assert_close(got, expected)
    # scale stays on GPU throughout (doesn't follow the weight to CPU)
    for b in blocks:
        assert b.lin.weight_scale.device.type == "cuda"
    # the swapped-out layer's fp8 weight master copy is indeed fp8 and pinned
    for rel in range(swap.num_swap):
        for _name, t in swap._cpu_weights[rel].items():
            assert t.dtype == torch.float8_e4m3fn
            assert t.is_pinned()


def test_restore_masters_points_params_back_to_cpu():
    """restore_masters points params back to the CPU master copy -- must be done before an fp8 merge."""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    swap.attach()

    # run one forward: the swapped-out layer's .data now points into a GPU slot
    _run_resident(blocks, torch.randn(1, 16, device=device))
    assert blocks[swap.first_swapped].lin1.weight.device.type == "cuda"

    swap.restore_masters()
    for b in list(blocks)[swap.first_swapped:]:
        assert b.lin1.weight.device.type == "cpu"
        assert b.lin1.weight.is_pinned()


def test_write_after_restore_masters_takes_effect_in_forward():
    """**Core guarantee**: changes written to weights after restore (= the fp8 merge delta)
    get carried onto the GPU by the next swap-in and genuinely affect the forward output --
    they aren't swallowed by GPU slot rotation.

    Note the assertion is on the **forward output**, not the swapped-out layer's `.data` --
    after a pass ends, a swapped-out layer's `.data` still points to whichever slot it
    used at the time, and that slot has long since been overwritten by a later layer
    (double buffering rotation). A swapped-out layer's weight is only valid within its
    own forward window; reading the weight outside that window requires calling
    restore_masters() first.
    """
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    swap.attach()

    x = torch.randn(1, 16, device=device)
    before = _run_resident(blocks, x).clone()

    # simulate an fp8 merge: modify the master copy in place after restore
    swap.restore_masters()
    blocks[swap.first_swapped].lin1.weight.data.add_(1.0)

    after = _run_resident(blocks, x)
    assert not torch.allclose(before, after), "the merge's changes did not take effect (swallowed by slot rotation)"

    # the master copy is the persistent one: after restore it should still carry the change
    swap.restore_masters()
    w = blocks[swap.first_swapped].lin1.weight
    assert w.device.type == "cpu" and w.is_pinned()


def test_swapped_weight_outside_forward_window_is_stale():
    """Pins down the semantics above: after a pass ends, a swapped-out layer's .data is an
    overwritten slot and cannot be trusted.

    This isn't a bug but an inevitable result of double buffering -- recorded here to
    stop future readers from reading weights via `.data`.
    """
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    swap.attach()
    _run_resident(blocks, torch.randn(1, 16, device=device))

    first, last = swap.first_swapped, swap.total - 1
    # rel=0 and rel=2 share slot 0 (rel % 2) -> after the pass, rel=0's .data actually holds rel=2's data
    torch.testing.assert_close(
        blocks[first].lin1.weight.data, blocks[last].lin1.weight.data,
    )
    # after restore, everything is back where it belongs
    swap.restore_masters()
    assert not torch.allclose(
        blocks[first].lin1.weight.data, blocks[last].lin1.weight.data,
    )


def test_move_module_excluding_keeps_swapped_on_cpu():
    """A blanket module.to(device) would move the master copy onto the GPU, wasting the swap; this helper must skip it."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "runtime"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from training.block_swap import move_module_excluding

    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 16, device)
    model = nn.Sequential(blocks)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    swap.restore_masters()

    # first move a resident layer to CPU, simulating the case of needing to move it back after offload
    blocks[0].lin1.weight.data = blocks[0].lin1.weight.data.cpu()

    move_module_excluding(model, device, swap)

    assert blocks[0].lin1.weight.device.type == "cuda", "the resident layer should be moved back to GPU"
    for b in list(blocks)[swap.first_swapped:]:
        assert b.lin1.weight.device.type == "cpu", "the swapped-out layer must stay on CPU"
        assert b.lin1.weight.is_pinned()


def test_move_module_excluding_without_swap_is_plain_move():
    """With swap=None, this degrades to a plain .to() with zero behavior change."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "runtime"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from training.block_swap import move_module_excluding

    device = torch.device("cuda")
    blocks = _make_blocks(3, 16, torch.device("cpu"))
    model = nn.Sequential(blocks)
    move_module_excluding(model, device, None)
    for b in blocks:
        assert b.lin1.weight.device.type == "cuda"


def test_close_drops_masters_even_with_other_holders():
    """close() must be able to release the pinned master copy even **while others still hold the block**.

    Real-machine testing: just dropping the swap object frees 0 bytes -- pinned memory
    is referenced by param.data, and more than ctx.model holds the block (LyCORIS
    injector holds org_module, the optimizer holds params). Hunting down holders one by
    one isn't reliable, so the component redirects the params itself.
    """
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(5, 32, device)
    swap = PinnedBlockSwap(blocks, num_swap=3, device=device)
    swap.attach()

    holder = [b.lin1 for b in blocks]        # simulate LyCORIS holding org_module
    masters = [swap._cpu_weights[r]["lin1.weight"] for r in range(swap.num_swap)]
    assert all(m.is_pinned() for m in masters)
    assert swap.pinned_bytes > 0

    swap.close()

    # internal storage cleared, accounting zeroed, hooks removed
    assert swap._cpu_weights == [] and swap._slot_buffers == []
    assert swap.pinned_bytes == 0
    assert swap._handles == []
    # the managed params now point at empty tensors -> the master copy is no longer referenced by the model
    for b in list(blocks)[swap.first_swapped:]:
        assert b.lin1.weight.numel() == 0
    assert holder  # the holder still exists, but no longer pins the memory


def test_close_is_idempotent():
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)
    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)
    swap.attach()
    swap.close()
    swap.close()  # should not raise


def test_release_pinned_host_cache_is_silent_without_api(monkeypatch):
    """A missing/failing internal API should be silent -- a cleanup failure shouldn't crash training teardown."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "runtime"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from training.block_swap import release_pinned_host_cache

    def _boom():
        raise RuntimeError("no such API")

    monkeypatch.setattr(torch._C, "_host_emptyCache", _boom, raising=False)
    release_pinned_host_cache()


def test_allocation_error_carries_context():
    """BlockSwapAllocationError carries num_swap/first_swapped/detail (for the caller's error text)."""
    _, BlockSwapAllocationError = _import()
    err = BlockSwapAllocationError(num_swap=14, first_swapped=14, detail="out of memory")
    assert err.num_swap == 14
    assert err.first_swapped == 14
    assert "out of memory" in str(err)
    assert "14" in str(err)


def test_build_packs_unpinned_weights_into_pow2_chunks():
    """Unpinned base weights (the anima placement path / the GPU-resident path), once
    packed by PinnedPacker: master copies are all pinned, and land in a small number
    of shared large chunks (not pinned tensor-by-tensor into its own block each)."""
    PinnedBlockSwap, _ = _import()
    device = torch.device("cuda")
    blocks = _make_blocks(4, 16, device)
    # the last 2 layers simulate anima placement: pageable CPU; the first 2 layers stay on GPU (verifies GPU->pinned also goes through packing)
    for b in list(blocks)[3:]:
        for p in b.parameters():
            p.data = p.detach().to("cpu")
    ref = {n: p.detach().clone().cpu() for n, p in blocks.named_parameters()}
    x = torch.randn(3, 16, device=device)
    expected = _run_resident(_make_blocks(4, 16, device), x)  # resident baseline with the same seed

    swap = PinnedBlockSwap(blocks, num_swap=2, device=device)

    storages = set()
    for rel in range(swap.num_swap):
        for _name, t in swap._cpu_weights[rel].items():
            assert t.is_pinned() and t.device.type == "cpu"
            storages.add(t.untyped_storage().data_ptr())
    # 2 layers x 4 params = 8 tensors, but there should only be 1 shared large chunk (total < the 64MB granularity)
    assert len(storages) == 1
    for n, p in blocks.named_parameters():
        if n.startswith(("2.", "3.")):
            assert torch.equal(p.detach().cpu(), ref[n])
    # behavior unchanged: forward still matches resident
    torch.testing.assert_close(_run_swap(swap, blocks, x), expected)
