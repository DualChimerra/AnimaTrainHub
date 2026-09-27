from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from utils.lycoris_adapter import AnimaLycorisAdapter
from training.families.anima.preset import ANIMA_PRESET
from training.families.krea2.preset import KREA2_PRESET
from training.families.krea2.quant_fp8 import patch_fp8_linears

pytest.importorskip("lycoris")


class MockDiT(nn.Module):

    def __init__(self, d: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.output_proj = nn.Linear(d, d, bias=False)


def _bypass_modes(adapter: AnimaLycorisAdapter) -> list[bool]:
    return [bool(getattr(m, "bypass_mode", False)) for m in adapter.network.loras]




def _build_lora_module(bypass: bool, seed: int = 0):
    from lycoris.modules.locon import LoConModule

    torch.manual_seed(seed)
    linear = nn.Linear(64, 64, bias=False)
    mod = LoConModule(
        lora_name="test",
        org_module=linear,
        multiplier=1.0,
        lora_dim=8,
        alpha=8,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        bypass_mode=bypass,
    )
    mod.apply_to()
    return linear, mod


def _copy_lora_weights(src, dst) -> None:
    dst.lora_up.weight.data.copy_(src.lora_up.weight.data)
    dst.lora_down.weight.data.copy_(src.lora_down.weight.data)


def test_locon_bypass_vs_rebuild_forward_equivalent() -> None:
    linear_a, mod_bypass = _build_lora_module(bypass=True, seed=0)
    linear_b, mod_rebuild = _build_lora_module(bypass=False, seed=0)
    _copy_lora_weights(mod_bypass, mod_rebuild)
    with torch.no_grad():
        mod_bypass.lora_up.weight.normal_(std=0.1)
        mod_rebuild.lora_up.weight.copy_(mod_bypass.lora_up.weight)

    with torch.no_grad():
        linear_b.weight.copy_(linear_a.weight)

    mod_bypass.eval()
    mod_rebuild.eval()
    x = torch.randn(2, 16, 64)
    out_bypass = linear_a(x)
    out_rebuild = linear_b(x)
    assert torch.allclose(out_bypass, out_rebuild, atol=1e-5, rtol=1e-5)


def test_locon_bypass_vs_rebuild_backward_equivalent() -> None:
    linear_a, mod_bypass = _build_lora_module(bypass=True, seed=1)
    linear_b, mod_rebuild = _build_lora_module(bypass=False, seed=1)
    _copy_lora_weights(mod_bypass, mod_rebuild)
    with torch.no_grad():
        mod_bypass.lora_up.weight.normal_(std=0.1)
        mod_rebuild.lora_up.weight.copy_(mod_bypass.lora_up.weight)
        linear_b.weight.copy_(linear_a.weight)

    mod_bypass.train()
    mod_rebuild.train()
    x = torch.randn(2, 16, 64, requires_grad=False)
    target = torch.randn(2, 16, 64)

    loss_bypass = (linear_a(x) - target).pow(2).mean()
    loss_rebuild = (linear_b(x) - target).pow(2).mean()
    assert torch.allclose(loss_bypass, loss_rebuild, atol=1e-5)

    loss_bypass.backward()
    loss_rebuild.backward()

    assert torch.allclose(
        mod_bypass.lora_up.weight.grad,
        mod_rebuild.lora_up.weight.grad,
        atol=1e-5, rtol=1e-5,
    )
    assert torch.allclose(
        mod_bypass.lora_down.weight.grad,
        mod_rebuild.lora_down.weight.grad,
        atol=1e-5, rtol=1e-5,
    )




def test_adapter_lora_defaults_to_bypass_mode() -> None:
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lora", rank=8, alpha=8)
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes, "preset should match at least one of q/k/v/output_proj"
    assert all(modes), f"all lora-algo modules should go through bypass, but got {modes}"


def test_adapter_lora_with_dora_forces_rebuild() -> None:
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, 
        algo="lora", rank=8, alpha=8, weight_decompose=True,
    )
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert not any(modes), f"DoRA must go through rebuild, but bypass_mode={modes}"


def test_adapter_lokr_keeps_rebuild() -> None:
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr", rank=8, alpha=8, factor=8)
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert not any(modes), f"lokr should stay on rebuild, but bypass_mode={modes}"


def test_adapter_lokr_fp8_base_forces_bypass_and_trains() -> None:
    """Monkeypatched FP8 nn.Linear uses bypass and full-precision LoKr params."""
    torch.manual_seed(0)
    model = MockDiT(d=16)
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.requires_grad_(False)
            module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
            scales[name] = torch.tensor(0.5)
    patch_fp8_linears(model, scales)

    adapter = AnimaLycorisAdapter(
        preset=KREA2_PRESET, algo="lokr", rank=4, alpha=4, factor=4,
    )
    adapter.inject(model)

    modes = _bypass_modes(adapter)
    assert modes and all(modes)
    params = adapter.get_params()
    assert params and all(p.dtype == torch.float32 for p in params)

    output = model.q_proj(torch.randn(2, 3, 16))
    output.square().mean().backward()
    grads = [p.grad for p in params if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_adapter_loha_keeps_rebuild() -> None:
    torch.manual_seed(0)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="loha", rank=8, alpha=8)
    adapter.inject(model)
    modes = _bypass_modes(adapter)
    assert modes
    assert not any(modes), f"loha should stay on rebuild, but bypass_mode={modes}"
