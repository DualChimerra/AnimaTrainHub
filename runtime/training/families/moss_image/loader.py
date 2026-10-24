"""MOSS-Image loading, using the pinned OpenMOSS/kohya checkpoint conventions."""
from pathlib import Path
import json
from typing import Optional
import torch
from accelerate import init_empty_weights
from modeling.moss_image import model as moss_image_models
from modeling.moss_image.weights import load_frozen_weights

DIT_KEY_PREFIXES = ["model.diffusion_model.", "diffusion_model."]

def _strip_dit_prefix(key: str) -> str:
    for prefix in DIT_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def split_fused_qkv(key: str, tensor: Optional[torch.Tensor], dim: int = moss_image_models.MOSS_IMAGE_CONFIG["dim"]):
    """Split hook for the ComfyUI format: `layers.N.proj.qkv.weight` [3*dim, dim] -> q / k / v weights.

    The ConvRot INT8 files of ComfyUI quantize the fused `qkv`: the int8 `.weight` [3*dim, dim] and the per-row
    `.weight_scale` [3*dim, 1] are split along the rows as well (the rotation acts on the input features, so the rows are
    independent), and the `.comfy_quant` spec is duplicated for q / k / v.

    Returns (new_keys, new_tensors). (None, None) if the key is not a fused qkv tensor. If tensor is None, only the keys
    are returned (the loader calls the hook without a tensor first).
    """
    stripped = _strip_dit_prefix(key)
    if not stripped.startswith("layers."):
        return None, None
    suffix = None
    for candidate in (".proj.qkv.weight", ".proj.qkv.weight_scale", ".proj.qkv.comfy_quant"):
        if stripped.endswith(candidate):
            suffix = candidate[len(".proj.qkv") :]  # ".weight", ".weight_scale" or ".comfy_quant"
            break
    if suffix is None:
        return None, None
    base = stripped[: -len("qkv" + suffix)]
    new_keys = [base + "q" + suffix, base + "k" + suffix, base + "v" + suffix]
    if tensor is None:
        return new_keys, None
    if suffix == ".comfy_quant":
        return new_keys, [tensor, tensor.clone(), tensor.clone()]
    assert tensor.shape[0] == 3 * dim, f"Unexpected fused qkv tensor shape {tuple(tensor.shape)} for {key}"
    q, k, v = torch.split(tensor, dim, dim=0)
    return new_keys, [q.contiguous(), k.contiguous(), v.contiguous()]

def load_dit(path, device, dtype, *, quantization="none", blocks_to_swap=0, attention_backend="none", config=None):
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("MOSS Base requires BF16 or FP32 compute")
    if quantization not in ("none", "convrot_int8", "qfloat8"):
        raise ValueError("MOSS checkpoint quantization must be none, convrot_int8 or qfloat8")
    with init_empty_weights():
        model = moss_image_models.MossImage(**(config or moss_image_models.MOSS_IMAGE_CONFIG),
            attn_mode={"none": "torch", "flash_attn": "flash", "xformers": "xformers"}[attention_backend])
    model.moss_fused_qkv = False
    def split(key, tensor):
        if ".proj.qkv." in key:
            model.moss_fused_qkv = True
        keys, tensors = split_fused_qkv(key, tensor, model.dim)
        return list(zip(keys, tensors)) if keys else [(key, tensor)]
    return load_frozen_weights(model, path, rename=_strip_dit_prefix, split=split,
        quantize=quantization == "convrot_int8", quant_target=lambda k: k.startswith("layers.") and ".proj." in k,
        device=device, offload_layers=blocks_to_swap, dtype=dtype, qfloat8=quantization == "qfloat8")


def load_text_model(path, device, *, quantization="none", offload_layers=0):
    if quantization not in ("none", "convrot_int8", "qfloat8"):
        raise ValueError("MOSS text encoder quantization must be none, convrot_int8 or qfloat8")
    try:
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    except ImportError as exc:
        raise RuntimeError("MOSS requires transformers>=5.3 for Qwen3.5. Update the environment dependencies.") from exc
    config_path = Path(moss_image_models.__file__).with_name("qwen35_config.json")
    config = Qwen3_5TextConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    config.use_cache = False
    with init_empty_weights():
        model = Qwen3_5TextModel(config)
    def rename(key):
        if key.startswith(("visual.", "model.visual.", "mtp.", "lm_head.")):
            return None
        for prefix in ("model.language_model.", "language_model.", "model."):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key
    targets = ("self_attn.", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
               "linear_attn.out_proj", "mlp.", "embed_tokens.")
    model = load_frozen_weights(model, path, rename=rename, device=device,
        offload_layers=offload_layers, quantize=quantization == "convrot_int8",
        quant_target=lambda key: any(part in key for part in targets), qfloat8=quantization == "qfloat8")
    if getattr(model, 'is_convrot_int8', False):
        from .quant_int8 import patch_int8_linear
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.int8:
                patch_int8_linear(module, module.scale_weight, module._convrot_groupsize, forward="dequant")
    if offload_layers:
        from ...layer_offload import FrozenLayerOffload
        model._layer_offload = FrozenLayerOffload(model.layers, offload_layers, device)
    return model
