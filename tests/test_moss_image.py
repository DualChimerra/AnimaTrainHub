"""Small CPU-only checks; no production weights, downloads, or CUDA context."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from modeling.moss_image.model import MossImage
from modeling.moss_image.vae import patchify_latents, unpatchify_latents
from training.families import get_family
from training.families.moss_image.loader import load_dit
from training.families.moss_image.text_encoding import TextCondition, MossTextStack
from studio.domain.training import TrainingConfig
from studio.domain.family_switch import switch_family

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)


def tiny_model():
    return MossImage(**TINY, attn_mode="torch")


def test_packing_is_exact_and_forward_checkpoint_gradients_agree():
    torch.manual_seed(17)
    x = torch.randn(2, 32, 4, 4)
    assert torch.equal(unpatchify_latents(patchify_latents(x)), x)
    model = tiny_model().train()
    cond = TextCondition(torch.randn(2, 3, 32), torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool))
    family = get_family("moss_image")
    results = []
    for checkpoint in (False, True):
        latent = x.unsqueeze(2).clone().requires_grad_()
        model.zero_grad(set_to_none=True)
        y = family.forward_train(model, latent, torch.tensor([.2, .7]), cond, use_checkpoint=checkpoint)
        y.square().mean().backward()
        results.append((y.detach(), latent.grad.clone(), model.layers[0].proj.q.weight.grad.clone()))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("quantization", ["none", "convrot_int8"])
def test_stream_loader_roundtrip_and_input_gradient(tmp_path, quantization):
    torch.manual_seed(3)
    original = tiny_model().eval().requires_grad_(False)
    path = tmp_path / "weights.safetensors"
    save_file(original.state_dict(), str(path))
    loaded = load_dit(path, "cpu", torch.float32, config=TINY, quantization=quantization)
    x = torch.randn(1, 128, 2, 2, requires_grad=True)
    text = torch.randn(1, 3, 32)
    expected = original(x, torch.tensor([500.]), text)
    actual = loaded(x, torch.tensor([500.]), text)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    if quantization == "none":
        torch.testing.assert_close(actual, expected)
    else:
        assert loaded.layers[0].proj.q.weight.dtype == torch.int8
        assert loaded.is_convrot_int8
        assert (actual - expected).square().mean().sqrt() < 0.1


def test_fused_qkv_checkpoint_loads_identically(tmp_path):
    model = tiny_model()
    sd = dict(model.state_dict())
    for i in range(2):
        sd[f'layers.{i}.proj.qkv.weight'] = torch.cat([sd.pop(f'layers.{i}.proj.{k}.weight') for k in 'qkv'])
    file = tmp_path / 'fused.safetensors'
    save_file(sd, str(file))
    result = load_dit(file, 'cpu', torch.float32, config=TINY)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(result.state_dict()[k], v)


@pytest.mark.parametrize('algo', ['lora', 'lokr', 'loha', 'tlora'])
def test_convrot_adapter_can_train_and_save_on_cpu(tmp_path, algo):
    from utils.lycoris_adapter import LycorisAdapter
    from training.families.moss_image.preset import MOSS_PRESET
    model = tiny_model().requires_grad_(False)
    path = tmp_path / 'base.safetensors'
    save_file(model.state_dict(), str(path))
    model = load_dit(path, 'cpu', torch.bfloat16, config=TINY, quantization='convrot_int8')
    adapter = LycorisAdapter(preset=MOSS_PRESET, algo=algo, rank=4, alpha=4)
    adapter.inject(model)
    assert adapter.network.loras
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(torch.randn(1, 128, 2, 2), torch.tensor([500.]), torch.randn(1, 3, 32))
        output.float().square().mean().backward()
    gradients = [p.grad for p in adapter.network.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    adapter.metadata_extra = get_family('moss_image').lora_metadata()
    adapter.save(tmp_path / 'adapter.safetensors')


def test_family_defaults_and_switch_preserve_optimizer():
    cfg = TrainingConfig(model_family='moss_image')
    assert (cfg.timestep_sampling, cfg.timestep_shift) == ('logit_normal', 3.5)
    assert cfg.sample_sampler_name == 'euler'
    assert not cfg.shuffle_caption
    old = TrainingConfig(optimizer_type='adamw', learning_rate=0.0002).model_dump()
    switched, _ = switch_family(old, 'moss_image', {'vae_path': 'flux2.safetensors'})
    assert TrainingConfig(**switched).learning_rate == 0.0002
    assert switched['optimizer_type'] == 'adamw'


@pytest.mark.parametrize('options', [
    {'mixed_precision': 'fp16'}, {'train_norm': True}, {'train_text_encoder': True},
    {'edit_mode': True}, {'shuffle_caption': True},
    {'infonoise_enabled': True}, {'moss_uncond_skip_layers': [42]},
    {'bucket_step': 24}, {'blocks_to_swap': 43},
    {'transformer_quantization': 'convrot_int8', 'lora_dora': True},
])
def test_moss_rejects_incompatible_options(options):
    with pytest.raises(ValueError):
        TrainingConfig(model_family='moss_image', **options)


def test_text_encoding_is_unpadded_and_cache_is_reused(tmp_path):
    weight = tmp_path / 'text.safetensors'
    save_file({'dummy': torch.ones(1)}, str(weight))
    calls = []
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs.get('padding', False) is False
            n = 2 if 'short' in text else 4
            return {'input_ids': torch.arange(n).reshape(1, n)}
    class Encoder:
        def __call__(self, **kwargs):
            ids = kwargs['input_ids']
            calls.append(ids.shape)
            return SimpleNamespace(last_hidden_state=torch.ones(1, ids.shape[1], 4096))
    stack = MossTextStack(weight, 'cpu', tokenizer=Tokenizer(), model_loader=lambda *a, **k: Encoder())
    result = stack.encode_text_for_batch(['short', 'long'], device='cpu', dtype=torch.float32)
    assert calls == [torch.Size([1, 2]), torch.Size([1, 4])]
    assert result.context.shape == (2, 4, 4096)
    assert result.attention_mask.tolist() == [[True, True, False, False], [True] * 4]
    stack.encode_text_for_batch(['short'], device='cpu', dtype=torch.float32)
    assert len(calls) == 2

    # A new process can reuse prompt files without loading the encoder again.
    stack.prepare_text_cache([], ['short', 'long'], cache_root=tmp_path)
    count = len(calls)
    restored = MossTextStack(weight, 'cpu', tokenizer=Tokenizer(), model_loader=lambda *a, **k: Encoder())
    restored.prepare_text_cache([], ['short', 'long'], cache_root=tmp_path)
    result = restored.encode_text_for_batch(['short', 'long'], device='cpu', dtype=torch.float32)
    assert len(calls) == count and not restored.is_model_loaded
    assert result.attention_mask.tolist() == [[True, True, False, False], [True] * 4]


def test_discovery_reads_headers_and_keeps_families_separate(tmp_path):
    from studio.services.models.weight_library import identify
    file = tmp_path / 'renamed.safetensors'
    save_file(tiny_model().state_dict(), str(file))
    assert identify(file, 'checkpoints') == ['moss_image']
    assert identify(file, 'vae') == []
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
    assert identify(file, 'text_encoders') == ['moss_image']


def test_prequantized_checkpoint_preserves_boundary_compute_dtype(tmp_path):
    from modeling.convrot import quantize_weight_convrot
    model = tiny_model().eval().requires_grad_(False)
    state = dict(model.state_dict())
    for key in ('video_embedder.weight', 'layers.0.proj.q.weight'):
        weight, scale, group = quantize_weight_convrot(key, state[key], (64,))
        owner = key.removesuffix('.weight')
        state[key] = weight
        state[owner + '.weight_scale'] = scale
        spec = json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': group})
        state[owner + '.comfy_quant'] = torch.tensor(list(spec.encode()), dtype=torch.uint8)
    path = tmp_path / 'prequant.safetensors'
    save_file(state, str(path))
    result = load_dit(path, 'cpu', torch.float32, config=TINY)
    x, context = torch.randn(1, 128, 2, 2), torch.randn(1, 3, 32)
    expected = model(x, torch.tensor([500.]), context)
    actual = result(x, torch.tensor([500.]), context)
    assert result.dtype == torch.float32
    assert (actual - expected).square().mean().sqrt() < 0.1
    del state['video_embedder.weight_scale']
    save_file(state, str(path))
    with pytest.raises(ValueError, match='Incomplete ConvRot'):
        load_dit(path, 'cpu', torch.float32, config=TINY)


def test_frozen_encoder_hooks_restore_parameters_and_reject_grad():
    from training.layer_offload import FrozenLayerOffload
    layers = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)]).requires_grad_(False)
    originals = [p.detach().clone() for p in layers.parameters()]
    offload = FrozenLayerOffload(layers, 2, 'cpu')
    with torch.no_grad():
        assert layers[1](layers[0](torch.ones(1, 4))).shape == (1, 2)
    with pytest.raises(RuntimeError, match='no_grad'):
        layers[0](torch.ones(1, 4))
    offload.close()
    assert not offload.handles and not offload.masters
    for actual, expected in zip(layers.parameters(), originals):
        torch.testing.assert_close(actual, expected)


def test_weight_selection_isolated_from_other_families(tmp_path, monkeypatch):
    from studio.services.models import weight_library
    from studio.services.models.families import FAMILY_ASSETS
    from studio.infrastructure.secrets import ModelsConfig
    from studio import secrets
    # Only the models root is scanned; external weight folders were retired.
    folder = tmp_path / 'diffusion_models'
    folder.mkdir()
    file = folder / 'moss.safetensors'
    save_file(tiny_model().state_dict(), str(file))
    cfg = ModelsConfig()
    sections = {}
    for assets in FAMILY_ASSETS.values():
        sections.update(assets.catalog_sections(tmp_path, cfg))
    library = weight_library.build_library(tmp_path, cfg, sections, {})
    found = next(r for r in library['checkpoints']['rows'] if r['path'] == str(file))
    assert found['families'] == ['moss_image'] and found['exists']
    writes = []
    monkeypatch.setattr(secrets, 'update', writes.append)
    monkeypatch.setattr(secrets, 'load', lambda: secrets.Secrets(models=cfg))
    weight_library.select_weight('checkpoints', str(file), {'weight_library': library})
    assert writes == [{'models': {'selected': {'moss_image': str(file)}, 'active_family': 'moss_image'}}]
    with pytest.raises(ValueError):
        weight_library.select_weight('vae', str(file), {'weight_library': library})


def test_missing_shards_are_not_selectable(tmp_path):
    from modeling.moss_image.weights import checkpoint_files
    from studio.services.models.weight_library import shard_files
    first = tmp_path / 'model-00001-of-00002.safetensors'
    save_file({'x': torch.ones(1)}, str(first))
    assert not all(f.is_file() for f in shard_files(first))
    with pytest.raises(FileNotFoundError, match='00002'):
        checkpoint_files(first)


def test_sampling_schedule_and_seed_are_reproducible():
    from training.families.moss_image.sampling import get_sampling_sigmas, sample_image
    class Text:
        def encode_text_for_batch(self, captions, **kwargs):
            return TextCondition(torch.zeros(len(captions), 2, 32), torch.ones(len(captions), 2, dtype=torch.bool))
        def release_model(self):
            pass
    class Vae:
        def decode(self, latent):
            return torch.nn.functional.interpolate(latent[:, :3, 0], scale_factor=8).unsqueeze(2)
    sigmas = get_sampling_sigmas(50)
    assert sigmas[0] == 1 and sigmas[-1] == 0
    assert (sigmas[:-1] > sigmas[1:]).all()
    model = tiny_model().eval()
    options = dict(height=32, width=32, steps=2, cfg_scale=3, device='cpu', dtype=torch.float32, seed=4)
    a = sample_image(model, Vae(), Text(), 'test', **options)
    b = sample_image(model, Vae(), Text(), 'test', **options)
    assert a.size == (32, 32) and a.tobytes() == b.tobytes()
