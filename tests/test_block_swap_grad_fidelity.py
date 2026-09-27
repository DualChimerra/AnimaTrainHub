"""Gradient fidelity for block swap (real sizes + noise-floor calibration).

**Why a separate file, and why it must use real sizes**: the small-tensor
checkpoint backward test in `test_block_swap.py` **passes cleanly even when
there's a race condition** in weight swap-in/swap-out -- a small block has
no attention, and the compute finishes too fast for the race window to show
up. Only a real machine's 6144-dim x 28 layers exposes it: the gradient
deviation reaches 300x the noise floor, and PPSF's `d` estimate blows up
outright (reported from a user's real machine).

**Why not compare element-wise with assert_close**: SDPA's backward pass is
nondeterministic on CUDA/bf16 -- running the same weights twice already
differs by about 5e-3. So the criterion is "the same order of magnitude as
the **control group's own repeatability**": first measure the difference
between two no-swap runs (the noise floor), then require the swap
difference to not exceed some multiple of it.

Historical readings (RTX 5090, 8 layers x features 6144):
    before the fix  noise floor 4.4e-3  swap diff 1.31    -> 298x fail
    after the fix   noise floor 4.4e-3  swap diff 4.7e-3  ->   1x pass
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="block swap is a CUDA-only mechanism"
)

_LAYERS = 6
_SWAP = 4
_SEQ = 512 + 128
#: Tolerance multiplier for swap's gradient deviation relative to the noise
#: floor. The real bug is on the order of 300x, and noise itself jitters
#: 1-2x, so 10x gives enough margin while still firmly catching a regression.
_TOLERANCE = 10.0


def _make_model(device, dtype):
    """A real-size krea2 block (with attention -- needed for the race to show up) + a mock LoRA."""
    from modeling.krea2 import KREA2_CONFIG
    from modeling.krea2.krea2_modeling import SingleStreamBlock

    cfg = KREA2_CONFIG
    torch.manual_seed(0)
    blocks = nn.ModuleList([
        SingleStreamBlock(
            cfg.features, cfg.heads, cfg.multiplier, cfg.bias, cfg.kvheads,
        )
        for _ in range(_LAYERS)
    ]).to(device, dtype)
    blocks.requires_grad_(False)          # base model frozen
    torch.manual_seed(1)
    for b in blocks:                      # LoRA-style trainable params (always resident, never swapped)
        b.lora = nn.Parameter(torch.ones(cfg.features, device=device, dtype=dtype) * 0.01)
    return blocks, cfg


def _inputs(cfg, device, dtype):
    from modeling.krea2.krea2_modeling import PositionalEncoding

    head = cfg.features // cfg.heads
    axes = (head - 12 * (head // 16), 6 * (head // 16), 6 * (head // 16))
    freqs = PositionalEncoding(axes, theta=cfg.theta)(
        torch.zeros(1, _SEQ, 3, device=device)
    )
    torch.manual_seed(9)
    x = torch.randn(1, _SEQ, cfg.features, device=device, dtype=dtype)
    vec = torch.randn(1, 6 * cfg.features, device=device, dtype=dtype)
    return x, vec, freqs


def _grads(blocks, inputs, *, use_checkpoint: bool):
    x, vec, freqs = inputs
    h = x
    for b in blocks:
        def fwd(t, blk=b):
            return blk(t, vec, freqs) * blk.lora
        h = checkpoint(fwd, h, use_reentrant=False) if use_checkpoint else fwd(h)
    h.sum().backward()
    return [b.lora.grad.clone() for b in blocks]


def _max_rel(a_list, b_list) -> float:
    return max(
        ((a.float() - b.float()).abs().max()
         / max(b.float().abs().max().item(), 1e-9)).item()
        for a, b in zip(a_list, b_list)
    )


@pytest.mark.parametrize("use_checkpoint", [True, False])
def test_swap_gradients_stay_within_nondeterminism_noise(use_checkpoint):
    """Swap's gradient deviation must be the same order of magnitude as "running no-swap twice".

    What this regression-tests: the slot gets released as soon as the
    forward pass ends, while the backward pass still needs to read those
    weights -> the next swap-in overwrites the weights currently being read
    -> the gradient is silently corrupted (no error, no NaN, just wrong
    numbers).
    """
    from training.block_swap import PinnedBlockSwap

    device = torch.device("cuda")
    dtype = torch.bfloat16

    blocks_a, cfg = _make_model(device, dtype)
    inputs = _inputs(cfg, device, dtype)
    grads_a = _grads(blocks_a, inputs, use_checkpoint=use_checkpoint)

    blocks_b, _ = _make_model(device, dtype)
    grads_b = _grads(blocks_b, inputs, use_checkpoint=use_checkpoint)
    noise = _max_rel(grads_a, grads_b)      # inherent jitter from running the same weights twice

    blocks_swap, _ = _make_model(device, dtype)
    swap = PinnedBlockSwap(blocks_swap, num_swap=_SWAP, device=device)
    swap.attach()
    grads_swap = _grads(blocks_swap, inputs, use_checkpoint=use_checkpoint)
    observed = _max_rel(grads_swap, grads_a)

    assert observed <= max(noise, 1e-4) * _TOLERANCE, (
        f"block swap's gradient deviation {observed:.2e} exceeds {_TOLERANCE}x "
        f"the noise floor {noise:.2e} -- weights were overwritten by a "
        f"swap-in during the backward pass"
    )
