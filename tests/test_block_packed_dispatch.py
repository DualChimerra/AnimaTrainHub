"""Dispatch contract for ``Block.forward(packed_tokens=True)`` (the hook precondition
for the NaViT packed path).

Block swap hangs weight fetch/release off nn.Module's four hooks, and those hooks
**only fire on ``__call__``**. So the packing loop must go through
``blk(..., packed_tokens=True)`` and must never call ``blk.forward_tokens(...)``
directly -- the latter would let a swapped-out layer silently compute with another
layer's leftover weights still in the slot (fully on-device, no error, no NaN).

This file pins down two things, neither needing CUDA / xformers:
1. Dispatch itself: the output of ``packed_tokens=True`` == calling ``forward_tokens``
   directly, and the hooks fire;
2. A source invariant: the block loop inside ``forward_packed_navit`` never calls
   ``forward_tokens`` directly.

Numerical equivalence (swap on/off matching value-for-value) is guarded on GPU by
``test_navit_packed_forward_backward_matches_without_swap`` in
tests/test_block_swap_anima.py.
"""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from modeling.anima.cosmos_predict2_modeling import Block, MiniTrainDIT

D = 32
CTX = 24
COUNTS = [3, 5]


def _block() -> Block:
    torch.manual_seed(0)
    return Block(
        x_dim=D, context_dim=CTX, num_heads=2,
        use_adaln_lora=True, adaln_lora_dim=8,
        # matches MiniTrainDIT's atten_backend; the transformer_engine backend isn't available on CPU/CI
        self_attention_backend="torch", cross_attention_backend="torch",
    ).eval()


def _packed_inputs():
    torch.manual_seed(1)
    sn = sum(COUNTS)
    g = len(COUNTS)
    x = torch.randn(1, sn, D)
    cross = torch.randn(1, sum([4] * g), CTX)
    emb = torch.randn(1, g, D)
    lora = torch.randn(1, g, 3 * D)
    mod_index = torch.repeat_interleave(torch.arange(g), torch.tensor(COUNTS))
    return x, emb, cross, lora, mod_index


def test_packed_dispatch_matches_direct_forward_tokens():
    """``blk(..., packed_tokens=True)`` is value-for-value equivalent to ``blk.forward_tokens(...)``."""
    blk = _block()
    x, emb, cross, lora, mod_index = _packed_inputs()
    kwargs = dict(token_wise_mod=True, mod_index=mod_index)

    with torch.no_grad():
        direct = blk.forward_tokens(x, emb, cross, adaln_lora_B_T_3D=lora, **kwargs)
        dispatched = blk(x, emb, cross, adaln_lora_B_T_3D=lora, packed_tokens=True, **kwargs)

    torch.testing.assert_close(direct, dispatched)


def test_packed_dispatch_fires_module_hooks():
    """The dispatch path fires the forward pre/post hooks -- this is exactly where block swap's fetch/release hangs."""
    blk = _block()
    x, emb, cross, lora, mod_index = _packed_inputs()
    fired: list[str] = []
    blk.register_forward_pre_hook(lambda *_a: fired.append("pre"))
    blk.register_forward_hook(lambda *_a: fired.append("post"))

    with torch.no_grad():
        blk(
            x, emb, cross, adaln_lora_B_T_3D=lora, packed_tokens=True,
            token_wise_mod=True, mod_index=mod_index,
        )

    assert fired == ["pre", "post"]


def test_packed_dispatch_backward_hooks_fire():
    """Same for the backward hooks: without them gradients get silently computed wrong
    (see attach()'s docstring; measured error was ~300x the noise floor)."""
    blk = _block()
    x, emb, cross, lora, mod_index = _packed_inputs()
    fired: list[str] = []
    blk.register_full_backward_pre_hook(lambda *_a: fired.append("bwd_pre"))
    blk.register_full_backward_hook(lambda *_a: fired.append("bwd_post"))

    out = blk(
        x.requires_grad_(True), emb, cross, adaln_lora_B_T_3D=lora,
        packed_tokens=True, token_wise_mod=True, mod_index=mod_index,
    )
    out.square().mean().backward()

    assert "bwd_pre" in fired and "bwd_post" in fired


def test_dense_path_still_rejects_unknown_kwargs():
    """``**packed_kwargs`` must not swallow a typo'd kwarg on the dense path."""
    blk = _block()
    with pytest.raises(TypeError, match="unexpected keyword"):
        blk(torch.randn(1, 1, 2, 2, D), torch.randn(1, 1, D), torch.randn(1, 4, CTX),
            typo_kwarg=1)


def test_packed_loop_routes_through_call():
    """Source invariant: the packing loop must not call blk.forward_tokens directly
    (the hooks would silently stop firing)."""
    src = inspect.getsource(MiniTrainDIT.forward_packed_navit)
    # only look at code lines: the comment mentioning blk.forward_tokens(...) is explaining why it must not be written that way
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "blk.forward_tokens(" not in code, (
        "forward_packed_navit's block loop must go through blk(..., packed_tokens=True): "
        "calling forward_tokens directly skips the nn.Module hooks, so block swap silently uses the wrong weights"
    )
    assert "packed_tokens=True" in code
