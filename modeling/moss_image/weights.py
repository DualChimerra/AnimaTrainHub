"""Strict, streaming safetensors loading for single-file and sharded checkpoints."""
from pathlib import Path
import re

import torch
from safetensors import safe_open


def checkpoint_files(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob('*.safetensors'))
        if not files and (path / 'transformer').is_dir():
            return checkpoint_files(path / 'transformer')
        if not files:
            raise FileNotFoundError(f'No safetensors weights in {path}')
        shards = [p for p in files if re.search(r'-\d{5}-of-\d{5}\.safetensors$', p.name)]
        if shards:
            groups = {re.sub(r'-\d{5}-of-', '-SHARD-of-', p.name) for p in shards}
            if len(groups) != 1 or len(shards) != len(files):
                raise ValueError(f'Select a specific checkpoint file; multiple models found in {path}')
            return checkpoint_files(shards[0])
        if len(files) != 1:
            raise ValueError(f'Select one checkpoint file in {path}')
        path = files[0]
    match = re.match(r'^(.*)-(\d{5})-of-(\d{5})\.safetensors$', path.name)
    files = ([path.with_name(f'{match[1]}-{i:05d}-of-{match[3]}.safetensors')
              for i in range(1, int(match[3]) + 1)] if match else [path])
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f'Missing model file: {file}')
    return files


def load_split_weights(path, device='cpu', disable_mmap=False):
    result = {}
    for file in checkpoint_files(path):
        with safe_open(str(file), framework='pt', device=str(device)) as reader:
            for key in reader.keys():
                if key in result:
                    raise ValueError(f'Duplicate weight: {key}')
                result[key] = reader.get_tensor(key)
    return result


def load_frozen_weights(model, path, *, rename, split=None, quantize=False,
                        quant_target=lambda key: False, device='cpu', layer_prefix='layers.',
                        offload_layers=0, dtype=torch.bfloat16, qfloat8=False):
    """Assign one tensor at a time; quantization never materializes the full bf16 model."""
    from .convrot import quantize_weight_convrot, parse_comfy_quant_spec
    from .convrot import _patch_convrot_int8_linear, _patch_convrot_int8_embedding
    device = torch.device(device)
    model.requires_grad_(False)
    expected = set(model.state_dict())
    seen, scales, specs = set(), {}, {}
    modules = dict(model.named_modules())
    layers = getattr(model, 'layers', ())
    first_offloaded = len(layers) - min(int(offload_layers), len(layers))
    quantized = set()
    fp8_scales = {}
    for file in checkpoint_files(path):
        with safe_open(str(file), framework='pt', device='cpu') as reader:
            for raw_key in reader.keys():
                key = rename(raw_key)
                if key is None or key.endswith('.comfy_attention.config'):
                    continue
                tensor = reader.get_tensor(raw_key)
                items = split(key, tensor) if split else [(key, tensor)]
                for key, tensor in items:
                    if key.endswith('.comfy_quant'):
                        specs[key.removesuffix('.comfy_quant')] = parse_comfy_quant_spec(key, tensor)
                        continue
                    if key.endswith(('.weight_scale', '.scale_weight')):
                        scales[key.rsplit('.', 1)[0]] = tensor.float().to(device)
                        continue
                    if key not in expected or key in seen:
                        raise ValueError(f'Unexpected or duplicate MOSS weight: {key}')
                    owner, attr = key.rsplit('.', 1) if '.' in key else ('', key)
                    module = modules[owner]
                    reference = getattr(module, attr)
                    if tensor.shape != reference.shape:
                        raise ValueError(f'Incompatible checkpoint: {key} shape {tuple(tensor.shape)}, expected {tuple(reference.shape)}')
                    if attr in module._buffers:
                        # RoPE bands are persistent FP32 buffers, never parameters or INT8 weights.
                        module._buffers[attr] = tensor.to(device=device, dtype=reference.dtype)
                        seen.add(key)
                        continue
                    destination = device
                    if key.startswith(layer_prefix):
                        index = int(key[len(layer_prefix):].split('.')[0])
                        if index >= first_offloaded:
                            destination = torch.device('cpu')
                    if tensor.dtype == torch.int8:
                        if qfloat8:
                            raise ValueError('qfloat8 requires BF16/FP32 source weights; select none for an existing ConvRot INT8 checkpoint')
                        quantized.add(owner)
                    elif not tensor.is_floating_point():
                        raise ValueError(f'Unsupported weight format: {key} ({tensor.dtype})')
                    elif tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        raise ValueError('Pre-quantized MOSS FP8 checkpoints are not supported; select BF16 weights and qfloat8 to quantize on load')
                    elif qfloat8 and attr == 'weight' and quant_target(key) and isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                        from .quant_fp8 import quantize_weight
                        tensor, scale = quantize_weight(tensor.to(device))
                        fp8_scales[owner] = scale
                    elif quantize and key.endswith('.weight') and quant_target(key):
                        converted = quantize_weight_convrot(key, tensor.to(device), (256, 64))
                        if converted is not None:
                            tensor, scale, group = converted
                            scales[owner] = scale.to(device)
                            specs[owner] = {'convrot_groupsize': group}
                            quantized.add(owner)
                    if tensor.is_floating_point() and tensor.dtype != torch.float8_e4m3fn:
                        tensor = tensor.to(dtype)
                    module._parameters[attr] = torch.nn.Parameter(tensor.to(destination), requires_grad=False)
                    seen.add(key)
    if seen != expected:
        raise ValueError(f'Incomplete MOSS checkpoint, missing: {sorted(expected - seen)[:8]}')
    if set(scales) != quantized or set(specs) != quantized:
        raise ValueError('Incomplete ConvRot weights: every INT8 matrix needs its scale and rotation metadata')
    for name in quantized:
        module = modules[name]
        scale = scales[name]
        if scale.numel() != module.weight.shape[0]:
            raise ValueError(f'Invalid ConvRot scale shape for {name}')
        group = specs[name]['convrot_groupsize']
        if module.weight.shape[1] % group:
            raise ValueError(f'Invalid ConvRot group size for {name}')
        module.register_buffer('scale_weight', scale.reshape(-1, 1))
        module.compute_dtype = dtype
        if isinstance(module, torch.nn.Embedding):
            _patch_convrot_int8_embedding(module, group, dtype)
        elif isinstance(module, torch.nn.Linear):
            _patch_convrot_int8_linear(module, group, 'bf16')
        else:
            raise ValueError(f'Unsupported ConvRot layer: {name}')
    if fp8_scales:
        from .quant_fp8 import patch_module
        for name, scale in fp8_scales.items():
            patch_module(modules[name], scale, dtype)
    # Rotary buffers are not checkpoint weights and retain their original float32 precision.
    for module in modules.values():
        for key, value in module._buffers.items():
            if value is not None:
                module._buffers[key] = value.to(device)
    model.is_convrot_int8 = bool(quantized)
    model.is_qfloat8 = bool(fp8_scales)
    return model.eval()
