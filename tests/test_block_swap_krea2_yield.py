"""Block swap x krea2 per-sample "DiT yields to the TE" (real CUDA).

Krea2Family.sample_image moves the DiT to CPU before encoding when the prompt
misses the online LRU and the TE won't fit, then moves it back. Under block swap a
blanket ``model.to()`` there would push the swapped layers' CPU pinned master copy
onto the card on the way back -- the swap is wasted and the peak is the full model
(OOM on 12/16GB cards).

Pins down: the master copy stays the same CPU pinned storage, the peak on the way
back excludes the swapped layers, and the forward after the yield still matches the
fully resident one bit for bit. Skipped without CUDA.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="block swap is CUDA-only"
)

_DIM = 1024          # 4 MiB per Linear (fp32), 8 MiB per block: peak difference far above noise
_N_BLOCKS = 6
_N_SWAP = 4


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(torch.relu(self.lin1(x)))


class _DiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.blocks = nn.ModuleList([_Block(_DIM) for _ in range(_N_BLOCKS)])
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _Text:
    """Duck-typed Krea2TextStack: LRU miss, TE not on the card -> triggers the yield check."""

    is_model_on_device = False

    def online_conditions_cached(self, captions):
        return False

    def offload_model(self):
        pass


def _block_bytes(block: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in block.parameters())


def _build():
    from training.block_swap import PinnedBlockSwap

    device = torch.device("cuda")
    model = _DiT().to(device)
    x = torch.randn(4, _DIM, device=device)
    with torch.no_grad():
        reference = model(x).clone()
    swap = PinnedBlockSwap(model.blocks, _N_SWAP, device)
    swap.attach()
    with torch.no_grad():
        model(x)  # one pass binds the swapped layers' .data to GPU slots (state after a real sample)
    return model, swap, x, reference


def _swapped_params(model, swap):
    return [
        (name, p)
        for block in list(model.blocks)[swap.first_swapped:]
        for name, p in block.named_parameters()
    ]


def test_find_block_swap_registered_and_cleared_on_close():
    from training.block_swap import find_block_swap

    model, swap, _x, _ref = _build()
    assert find_block_swap(model) is swap
    swap.close()
    assert find_block_swap(model) is None


def test_krea2_yield_keeps_swapped_layers_pinned_on_cpu(monkeypatch):
    import training.families.krea2 as krea2_module
    from training.families.krea2 import Krea2Family

    model, swap, x, reference = _build()
    device = torch.device("cuda")
    master_ptrs = [{n: t.data_ptr() for n, t in w.items()}
                   for w in swap._cpu_weights]
    resident_bytes = sum(
        _block_bytes(b) for b in list(model.blocks)[:swap.first_swapped]
    )
    swapped_bytes = sum(
        _block_bytes(b) for b in list(model.blocks)[swap.first_swapped:]
    )
    seen = {}

    def _fake_sample(m, *a, **k):
        # after the move back, before sampling: every swapped tensor is still swap-managed (master or GPU slot)
        managed = swap.managed_data_ptrs()
        seen["swapped_managed"] = all(
            p.data.data_ptr() in managed for _n, p in _swapped_params(m, swap)
        )
        seen["resident_on_gpu"] = all(
            p.device.type == "cuda"
            for b in list(m.blocks)[:swap.first_swapped] for p in b.parameters()
        )
        seen["peak"] = torch.cuda.max_memory_allocated(device)
        with torch.no_grad():
            seen["out"] = m(x).clone()
        return "image"

    monkeypatch.setattr(
        krea2_module, "prepare_sampling_condition", lambda *a, **k: "cond",
    )
    monkeypatch.setattr(krea2_module, "sample_image", _fake_sample)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = Krea2Family().sample_image(
        model, object(), _Text(), "a prompt",
        distilled=True, device="cuda", vram_policy="save_vram",
    )
    assert result == "image"

    # master copy: same storage, still on CPU and pinned
    for rel, weights in enumerate(swap._cpu_weights):
        for name, tensor in weights.items():
            assert tensor.device.type == "cpu" and tensor.is_pinned()
            assert tensor.data_ptr() == master_ptrs[rel][name]
    assert seen["swapped_managed"], "swapped layers were moved out of swap management by .to()"
    assert seen["resident_on_gpu"], "resident layers should be back on the GPU"
    # the peak on the way back holds only resident layers (base already has GPU
    # slots + input); a blanket .to() would add all swapped layers (_N_SWAP blocks)
    assert seen["peak"] - base < resident_bytes + swapped_bytes // 2, (
        f"peak {(seen['peak'] - base) / 2**20:.1f} MiB includes swapped layers"
        f" (resident {resident_bytes / 2**20:.0f} MiB / swapped {swapped_bytes / 2**20:.0f} MiB)"
    )
    # after the yield round trip the forward still matches the fully resident one bit for bit
    assert torch.equal(seen["out"], reference)
    swap.close()
