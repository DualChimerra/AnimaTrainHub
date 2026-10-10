from __future__ import annotations

import copy

import pytest
import torch
from safetensors import safe_open
from torch import nn

from studio.schema import TrainingConfig
from studio.services.inference.core import LoRASpec, apply_loras, read_lora_meta
from training.families.anima.family import AnimaFamily
from training.families.anima.preset import ANIMA_PRESET, ANIMA_TEXT_PRESET
from training.state import load_training_state, save_training_state
from utils.lycoris_adapter import LycorisAdapter


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.q_norm = nn.RMSNorm(16, eps=1e-6)
        self.k_norm = nn.RMSNorm(16, eps=1e-6)

    def forward(self, x):
        return self.k_norm(self.q_norm(self.q_proj(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.cross_attn = Attention()

    def forward(self, x):
        return self.cross_attn(self.self_attn(x))


class TinyDiT(nn.Module):
    def __init__(self, blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(Block())
        self.llm_adapter = Attention()
        self.requires_grad_(False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def preprocess_text_embeds(self, embeds, _ids, **_kwargs):
        return self.llm_adapter(embeds)


def adapter(model, *, train_norm=False, algo="lokr"):
    result = LycorisAdapter(preset=ANIMA_PRESET, algo=algo, rank=4, alpha=4,
                            factor=4, train_norm=train_norm, text_encoder_lr=1e-5)
    result.inject(model)
    return result


def test_norm_training_export_resume_and_disable(tmp_path):
    torch.manual_seed(42)
    model = TinyDiT()
    clean = copy.deepcopy(model)
    x = torch.randn(2, 3, 16)
    before = model(x).detach()
    a = adapter(model, train_norm=True)
    groups = a.get_param_groups(0.01)
    params = [p for group in groups for p in group["params"]]
    assert len({id(p) for p in params}) == len(params)
    assert all(any(p is q for q in params) for p in a.norms.parameters())
    optimizer = torch.optim.AdamW(groups, lr=0.01)
    model(x).square().mean().backward()
    assert all(p.grad is not None for p in a.norms.parameters())
    assert all(p.grad is None for p in model.llm_adapter.parameters())
    optimizer.step()
    expected = model(x).detach()
    assert not torch.allclose(expected, before)
    with a.disabled():
        torch.testing.assert_close(model(x), before)
    torch.testing.assert_close(model(x), expected)

    path = tmp_path / "norm.safetensors"
    a.save(path)
    with safe_open(path, framework="pt") as f:
        assert f.metadata()["ss_train_norm_exported_count"] == "4"
        assert f.metadata()["ss_train_norm_export_format"] == "comfyui_diff"
        assert sum(k.endswith(".diff") for k in f.keys()) == 4
        assert not any(k.startswith("train_norm_") for k in f.keys())
    loaded = apply_loras(clean, [LoRASpec(str(path), 1)], "cpu", torch.float32)[0]
    torch.testing.assert_close(clean(x), expected)
    loaded.set_multiplier(0)
    torch.testing.assert_close(clean(x), before)
    loaded.detach()
    torch.testing.assert_close(clean(x), before)

    state = tmp_path / "state.pt"
    save_training_state(state, a, optimizer, epoch=2, global_step=3)
    resumed = adapter(copy.deepcopy(clean), train_norm=True)
    optimizer2 = torch.optim.AdamW(resumed.get_param_groups(0.01), lr=0.01)
    load_training_state(state, resumed, optimizer2)
    for key, value in a.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)


def test_norm_count_matches_28_block_anima(tmp_path):
    a = adapter(TinyDiT(28), train_norm=True)
    path = tmp_path / "28blocks.safetensors"
    a.save(path)
    with safe_open(path, framework="pt") as f:
        assert f.metadata()["ss_train_norm_exported_count"] == "112"


def test_norm_inference_stacks_scales_and_hot_reload():
    model = TinyDiT()
    a = adapter(model)
    b = adapter(model)
    key = "lora_unet_blocks_0_self_attn_q_norm.diff"
    a.load_state_dict({key: torch.full((16,), 0.25)}, strict=False)
    b.load_state_dict({key: torch.full((16,), -0.125)}, strict=False)
    a.set_multiplier(0.5)
    b.set_multiplier(2)
    norm = model.blocks[0].self_attn.q_norm
    torch.testing.assert_close(norm.weight, torch.full((16,), 0.875))
    a.load_state_dict({}, strict=False)
    torch.testing.assert_close(norm.weight, torch.full((16,), 0.75))
    a.detach()
    torch.testing.assert_close(norm.weight, torch.full((16,), 0.75))
    b.detach()
    torch.testing.assert_close(norm.weight, torch.ones(16))


def test_invalid_norm_patch_fails_instead_of_silently_skipping():
    a = adapter(TinyDiT())
    with pytest.raises(ValueError, match="size mismatch"):
        a.load_state_dict({"lora_unet_blocks_0_self_attn_q_norm.diff": torch.zeros(2)}, strict=False)
    with pytest.raises(ValueError, match="Unsupported normalization"):
        a.load_state_dict({"lora_unet_llm_adapter_q_norm.diff": torch.zeros(16)}, strict=False)


def test_norm_fp32_master_and_ema(tmp_path):
    from training.ema import AdapterEma

    a = adapter(TinyDiT().to(dtype=torch.bfloat16), train_norm=True)
    parameter = a.norms.parameters()[0]
    assert parameter.dtype == torch.float32
    ema = AdapterEma(a.get_params(), decay=0.5, warmup=False)
    ema.update(0)
    with torch.no_grad():
        parameter.add_(0.0001)
    ema.update(1)
    live = parameter.detach().clone()
    with ema.applied():
        path = tmp_path / "ema.safetensors"
        a.save(path)
        with safe_open(path, framework="pt") as f:
            delta = f.get_tensor(next(iter(a.norms.patches)))
        assert delta.abs().max().item() == pytest.approx(0.00005, abs=1e-7)
    torch.testing.assert_close(parameter, live, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA block swap")
def test_norm_training_with_block_swap_and_checkpoint():
    from torch.utils.checkpoint import checkpoint
    from training.block_swap import PinnedBlockSwap

    torch.manual_seed(19)
    model = TinyDiT(3).to(device="cuda", dtype=torch.bfloat16)
    reference = copy.deepcopy(model)
    a = adapter(model, train_norm=True)
    b = adapter(reference, train_norm=True)
    b.load_state_dict(a.state_dict())
    swap = PinnedBlockSwap(model.blocks, 2, "cuda")
    swap.attach()
    x = torch.randn(2, 4, 16, device="cuda", dtype=torch.bfloat16)
    target = torch.randn_like(x)
    try:
        outputs = []
        for net in (model, reference):
            value = x
            for block in net.blocks:
                value = checkpoint(block, value, use_reentrant=False)
            (value.float() - target.float()).square().mean().backward()
            outputs.append(value.detach())
        torch.testing.assert_close(*outputs)
        for actual, expected in zip(a.norms.parameters(), b.norms.parameters()):
            assert actual.device.type == "cuda" and actual.dtype == torch.float32
            assert actual.grad is not None and actual.grad.isfinite().all()
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-3, rtol=0.02)
        torch.nn.utils.clip_grad_norm_(a.get_params(), 1.0)
    finally:
        swap.close()


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, **_kwargs):
        from transformers import BatchEncoding

        return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]] * len(texts)),
                              "attention_mask": torch.ones(len(texts), 3, dtype=torch.long)})


def text_model(backend):
    if backend == "hf":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        ))
    else:
        from training.families.anima.comfy_qwen import (
            ComfyQwen3Encoder, Qwen3_06BBackbone, Qwen3_06BConfig,
        )

        model = ComfyQwen3Encoder.__new__(ComfyQwen3Encoder)
        nn.Module.__init__(model)
        model.model = Qwen3_06BBackbone(Qwen3_06BConfig(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        ))
        # The production loader fills these empty parameters from safetensors.
        for module in model.modules():
            if module.__class__.__name__ == "RMSNorm":
                nn.init.ones_(module.weight)
    return model.eval().requires_grad_(False)


@pytest.mark.parametrize("backend", ["hf", "comfy"])
@pytest.mark.parametrize("algo", ["lora", "lokr", "loha"])
def test_text_encoder_gradient_lr_export_and_inference(backend, algo, tmp_path, monkeypatch):
    from training.families.anima import text_encoding
    from training.optimizers.came import build as build_came
    from types import SimpleNamespace

    def tokenize(_tokenizer, captions, **_kwargs):
        return (torch.ones(len(captions), 3, dtype=torch.long),
                torch.ones(len(captions), 3, dtype=torch.long), torch.ones(len(captions), 3))

    monkeypatch.setattr(text_encoding, "tokenize_t5_comfy_literal", tokenize)
    torch.manual_seed(13)
    model = TinyDiT()
    text = text_model(backend)
    clean_model, clean_text = copy.deepcopy(model), copy.deepcopy(text)
    original_text = {k: v.clone() for k, v in text.state_dict().items()}
    a = adapter(model, train_norm=True, algo=algo)
    a.inject_text(text, preset=ANIMA_TEXT_PRESET, training=True)
    stack = (text, Tokenizer(), object())
    family = AnimaFamily()
    family.prepare_text_training(stack, checkpoint=True)
    groups = a.get_param_groups(0)
    optimizer = build_came(SimpleNamespace(), groups, 1e-3, 0)
    te_params = a.text_adapter.get_params()
    assert all(group["lr"] == 1e-5 for group in optimizer.param_groups
               if any(any(p is te for te in te_params) for p in group["params"]))
    cross = family.encode_text_for_batch(stack, model, ["test"], "cpu", torch.float32)
    assert cross.requires_grad
    model(cross[:, :3]).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in te_params)
    before = [p.clone() for p in te_params]
    optimizer.step()
    assert any(not torch.equal(x, y) for x, y in zip(before, te_params))
    for key, value in text.state_dict().items():
        torch.testing.assert_close(value, original_text[key], rtol=0, atol=0)

    model.eval()
    assert not text.training and not a.text_adapter.network.training
    with torch.no_grad():
        expected = family.encode_text_for_batch(stack, model, ["test"], "cpu", torch.float32)
    path = tmp_path / "text.safetensors"
    a.save(path)
    assert read_lora_meta(str(path)).has_text_encoder
    with safe_open(path, framework="pt") as f:
        assert f.metadata()["ss_text_encoder_lr"] == "1e-05"
        assert any(key.startswith("lora_te_") for key in f.keys())
        assert not any(key.startswith("lora_te_model_") for key in f.keys())
    clean_stack = (clean_text, Tokenizer(), object())
    loaded = apply_loras(clean_model, [LoRASpec(str(path), 1)], "cpu", torch.float32,
                         text_stack=clean_stack)[0]
    actual = family.encode_text_for_batch(clean_stack, clean_model, ["test"], "cpu", torch.float32)
    torch.testing.assert_close(actual, expected)
    loaded.set_multiplier(0)
    assert loaded.text_adapter.network.multiplier == 0
    loaded.detach()
    assert not clean_text._train_text_encoder

    state_path = tmp_path / "text-state.pt"
    save_training_state(state_path, a, optimizer, epoch=1, global_step=1)
    resumed = adapter(clean_model, train_norm=True, algo=algo)
    resumed.inject_text(clean_text, preset=ANIMA_TEXT_PRESET, training=True)
    opt2 = build_came(SimpleNamespace(), resumed.get_param_groups(0), 1e-3, 0)
    load_training_state(state_path, resumed, opt2)
    for key, value in a.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)


def test_training_options_schema_cli_and_preset():
    from pydantic import ValidationError
    from studio.infrastructure.argparse_bridge import build_parser, namespace_from_config
    from studio.services.presets.io import _tolerant_validate

    assert not TrainingConfig().train_norm
    assert not TrainingConfig().train_text_encoder
    raw = {"train_norm": True, "train_text_encoder": True, "ss_text_encoder_lr": 2e-5}
    config, dropped, fixed = _tolerant_validate(raw)
    assert config.text_encoder_lr == 2e-5 and not dropped and not fixed
    parser = build_parser(TrainingConfig, suppress_defaults=True)
    ns = namespace_from_config(parser.parse_args(["--ss-text-encoder-lr", "3e-5"]), raw, TrainingConfig)
    assert ns.text_encoder_lr == 3e-5
    for change in ({"model_family": "krea2"}, {"lora_type": "ortho"}, {"dop_enabled": True}):
        with pytest.raises(ValidationError):
            TrainingConfig(train_text_encoder=True, **change)
    with pytest.raises(ValidationError):
        TrainingConfig(train_norm=True, model_family="krea2")
