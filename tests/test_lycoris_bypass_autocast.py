"""Exercise real LyCORIS fused bypass precision through Anima attention."""
from __future__ import annotations

import logging

import pytest
import torch

pytest.importorskip("lycoris")

from modeling.anima import cosmos_predict2_modeling as cosmos
from training.families.anima.preset import ANIMA_PRESET
from utils.lycoris_adapter import LycorisAdapter


@pytest.fixture
def cuda_fused(monkeypatch: pytest.MonkeyPatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dispatch = pytest.importorskip("lycoris.kernels.dispatch")
    backends = dispatch.fused_backends()
    if not backends:
        pytest.skip("LyCORIS fused kernels are unavailable")
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", backends[0])
    monkeypatch.setattr(cosmos, "_USE_FLASH_ATTN", False)
    monkeypatch.setattr(cosmos, "_USE_XFORMERS", False)
    return torch.device("cuda")


def _inject(model, algo: str, device: torch.device):
    adapter = LycorisAdapter(
        preset=ANIMA_PRESET,
        algo=algo,
        rank=8,
        alpha=12.0,
        tlora_min_rank=2,
    )
    adapter.inject(model)
    with torch.no_grad():
        for layer in adapter.network.loras:
            layer.lora_up.weight.normal_(std=0.03)
    if algo == "tlora":
        adapter._set_tlora_mask(torch.tensor([1.0], device=device))
        assert int(adapter._tlora_mask.sum()) == 2
    return adapter


def _assert_grad(tensor: torch.Tensor) -> None:
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert tensor.grad.abs().sum() > 0


def _assert_adapter_grads(adapter: LycorisAdapter) -> None:
    for layer in adapter.network.loras:
        _assert_grad(layer.lora_down.weight)
        _assert_grad(layer.lora_up.weight)
        if adapter.use_timestep_mask:
            assert torch.count_nonzero(layer.lora_down.weight.grad[2:]) == 0
            assert torch.count_nonzero(layer.lora_up.weight.grad[:, 2:]) == 0


@pytest.mark.parametrize("algo", ["lora", "tlora"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fused_attention_bypass_honors_cuda_autocast(cuda_fused, algo, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unavailable")
    torch.manual_seed(41)
    model = cosmos.Attention(64, n_heads=2, head_dim=32, backend="torch")
    model.to(device=cuda_fused, dtype=dtype).requires_grad_(False)
    adapter = _inject(model, algo, cuda_fused)
    # CUDA LayerNorm produces this FP32 input in a real Anima block.
    x = torch.randn(2, 8, 64, device=cuda_fused, requires_grad=True)

    with torch.autocast("cuda", dtype=dtype):
        assert model.q_proj(x).dtype == dtype
        q, k, v = model.compute_qkv(x)
        assert (q.dtype, k.dtype, v.dtype) == (dtype, dtype, dtype)
        out = model.compute_attention(q, k, v)
        assert out.dtype == dtype
        loss = out.float().square().mean()
    loss.backward()

    _assert_grad(x)
    _assert_adapter_grads(adapter)


@pytest.mark.parametrize("algo", ["lora", "tlora"])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_bypass_preserves_fp32_without_autocast(request, monkeypatch, algo, device_name):
    device = (
        request.getfixturevalue("cuda_fused")
        if device_name == "cuda"
        else torch.device("cpu")
    )
    monkeypatch.setattr(cosmos, "_USE_FLASH_ATTN", False)
    monkeypatch.setattr(cosmos, "_USE_XFORMERS", False)
    torch.manual_seed(42)
    model = cosmos.Attention(64, n_heads=2, head_dim=32, backend="torch")
    model.to(device=device, dtype=torch.float32).requires_grad_(False)
    adapter = _inject(model, algo, device)
    x = torch.randn(1, 8, 64, device=device, requires_grad=True)

    q, k, v = model.compute_qkv(x)
    assert (q.dtype, k.dtype, v.dtype) == (torch.float32,) * 3
    out = model.compute_attention(q, k, v)
    assert out.dtype == torch.float32
    out.square().mean().backward()

    _assert_grad(x)
    _assert_adapter_grads(adapter)


@pytest.mark.parametrize("algo", ["lora", "tlora"])
def test_anima_block_flash_training_and_sampling(cuda_fused, monkeypatch, caplog, algo):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unavailable")
    if not cosmos._FLASH_ATTN_AVAILABLE or cosmos._flash_attn_func is None:
        pytest.skip("FlashAttention is unavailable")
    real_flash = cosmos._flash_attn_func
    calls = []

    def flash_spy(q, k, v):
        calls.append((q.dtype, k.dtype, v.dtype))
        return real_flash(q, k, v)

    monkeypatch.setattr(cosmos, "_USE_FLASH_ATTN", True)
    monkeypatch.setattr(cosmos, "_flash_attn_func", flash_spy)
    monkeypatch.setattr(cosmos, "_FLASH_FALLBACK_WARNED", set())
    caplog.set_level(logging.WARNING, logger=cosmos.__name__)
    torch.manual_seed(43)
    block = cosmos.Block(
        x_dim=64,
        context_dim=64,
        num_heads=2,
        self_attention_backend="torch",
        cross_attention_backend="torch",
    ).to(device=cuda_fused, dtype=torch.bfloat16).requires_grad_(False)
    adapter = _inject(block, algo, cuda_fused)
    x = torch.randn(
        1, 1, 2, 4, 64, device=cuda_fused, dtype=torch.bfloat16, requires_grad=True,
    )
    emb = torch.randn(1, 1, 64, device=cuda_fused, dtype=torch.bfloat16)
    context = torch.randn(1, 5, 64, device=cuda_fused, dtype=torch.bfloat16)
    rope = torch.randn(8, 1, 1, 32, device=cuda_fused)
    norm_dtypes = []
    hook = block.layer_norm_self_attn.register_forward_hook(
        lambda _module, _args, output: norm_dtypes.append(output.dtype)
    )

    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = block(x, emb, context, rope_emb_L_1_1_D=rope)
            loss = out.float().square().mean()
        loss.backward()
        _assert_grad(x)
        _assert_adapter_grads(adapter)

        block.eval()
        adapter.clear_timestep_mask()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            sample = block(x.detach(), emb, context, rope_emb_L_1_1_D=rope)
        assert torch.isfinite(sample).all()
        assert not adapter.network.training
    finally:
        hook.remove()

    assert norm_dtypes == [torch.float32, torch.float32]
    assert calls == [(torch.bfloat16,) * 3] * 4  # self/cross, train/eval
    assert not cosmos._FLASH_FALLBACK_WARNED
    assert not any("flash_attn fallback" in record.message for record in caplog.records)
