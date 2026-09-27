"""LoRA merge writeback for Krea2 fp8 base models (bit-for-bit ComfyUI parity).

ComfyUI never computes LoRA on top of quantized storage: at load time it
merges the LoRA into the weights (model_patcher.patch_weight_to_device), and
the sampling-time forward pass follows exactly the same path as with no
LoRA. This module bit-for-bit replicates that chain (oracle = local ComfyUI
v0.27.1 + comfy_kitchen 0.2.16 eager path, docs/design/krea2-fp8-inference.md
SS1.5):

    W16    = W_fp8.to(fp16) [* scale.to(fp16)]            # dequant to fp16
    W16   += ((strength * alpha/dim) * dW_fp32).to(fp16)   # per LoRA, fp32 intermediate
    scale' = amax(|W16|).to(fp32) / 448 (fp16 underflow-prevention clamp)
    W16   *= (1/scale').to(fp16)
    W_fp8' = stochastic_round(W16, seed=CRC32(key))        # ck eager SR

Three layer shapes (mixed within the same model; the user's official Turbo
fp8 file has 256 quantized out of 264 total Linear layers):
- scaled fp8: dequant multiplies by scale; writeback recalculates scale + SR
- plain cast fp8: dequant has no scale; writeback is a direct SR (no scale concept)
- non-quantized Linear: cast to fp16 to merge; writeback casts straight back
  to the original dtype (comfy's set_weight layout_type=None branch, no SR, no seed)

Numerical basis (all verified line-by-line against the local oracle):
- dequant lands directly in the fp16 domain: QuantizedTensor.to(fp16) only
  flips the orig_dtype flag (ck tensor/base.py _handle_to); dequantize is
  qdata.to(fp16)*scale.to(fp16), no bf16 in between; fp16 = comfy's
  lora_compute_dtype (should_use_fp16 on modern cards)
- delta domain: comfy's weight_adapter lokr/lora/loha cast the factors to
  fp32 to do mm/kron/Hadamard, then
  ``weight += ((strength * alpha) * diff).type(fp16)``, i.e. the addition
  happens in the fp16 domain
- requantize: the scale="recalculate" + stochastic_rounding branch of
  quant_ops._TensorCoreFP8LayoutBase.quantize
- SR: comfy/float.py (Generator(device)+manual_seed -> randint(0,256,uint8))
  -> ck eager stochastic_rounding_fp8/calc_mantissa. CUDA and CPU generator
  sequences differ -- bit-for-bit parity holds on GPU (comfy also merges on
  GPU); the CPU unit test only checks determinism and formula properties
- seed key = "diffusion_model.{layer}.weight" (comfy ModelPatcher's model
  key prefix; layer names match checkpoint keys since the loader reads
  comfy files directly)

See THIRD_PARTY_NOTICES for derivation attribution (ComfyUI GPL-3.0 +
comfy_kitchen). Used only on the inference path; merged weights keep
requires_grad=False.
"""

from __future__ import annotations

import logging
import zlib

import torch
from torch import Tensor

from training.families.krea2.quant_fp8 import _FP8_TORCH_DTYPES


logger = logging.getLogger(__name__)

_LORA_PREFIX = "lora_unet_"
_COMFY_PREFIX = "diffusion_model."
# PEFT/comfy key suffix -> kohya/lycoris naming (a common shape for krea2
# LoRAs in the civitai ecosystem: ``diffusion_model.{dotted layer name}.lora_A/lora_B``,
# lora_A=down, lora_B=up, usually with no alpha key -- comfy treats a
# missing alpha as a scale of 1.0, matching _apply_lora_delta's default
# branch for "alpha")
_PEFT_SUFFIXES = (
    ("lora_A.weight", "lora_down.weight"),
    ("lora_B.weight", "lora_up.weight"),
    ("lora_down.weight", "lora_down.weight"),
    ("lora_up.weight", "lora_up.weight"),
    ("alpha", "alpha"),
    ("dora_scale", "dora_scale"),
)


def string_to_seed(key: str) -> int:
    """Equivalent to comfy.utils.string_to_seed: standard CRC-32 (0xEDB88320).

    comfy's hand-rolled implementation XORs ord() of each character of the
    string -- layer names are all ASCII, which matches the utf-8 byte
    sequence, so zlib.crc32 is bit-identical (unit test checks against a
    hand-written reference implementation).
    """
    if not key.isascii():
        raise ValueError(f"seed key must be ASCII (comfy layer-name domain): {key!r}")
    return zlib.crc32(key.encode("utf-8"))


def _calc_mantissa(
    abs_x: Tensor,
    exponent: Tensor,
    normal_mask: Tensor,
    mantissa_bits: int,
    exponent_bias: int,
    rng: Tensor,
) -> Tensor:
    # Line-by-line copy of ck eager calc_mantissa
    # (backends/eager/quantization.py:66-74): everything stays in the fp16
    # domain; rng/256 supplies a random carry amount in [0,1), and floor()
    # performs the stochastic rounding.
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - exponent_bias)) - 1.0) * (2 ** mantissa_bits),
        (abs_x / (2.0 ** (-exponent_bias + 1 - mantissa_bits))),
    )
    mantissa_scaled += rng.to(dtype=mantissa_scaled.dtype) * (1.0 / 256.0)
    return mantissa_scaled.floor() / (2 ** mantissa_bits)


def stochastic_round_to_fp8(value: Tensor, fp8_dtype: torch.dtype, seed: int) -> Tensor:
    """Bit-for-bit copy of comfy.float.stochastic_rounding's fp8 branch + ck eager SR."""
    if fp8_dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif fp8_dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15
    else:
        raise ValueError(f"stochastic_round_to_fp8 only supports fp8 dtypes: {fp8_dtype}")

    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    rng = torch.randint(
        0, 256, value.size(),
        dtype=torch.uint8, layout=value.layout, device=value.device,
        generator=generator,
    )

    x = value.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + exponent_bias,
        0,
        2 ** exponent_bits - 1,
    )
    normal_mask = ~(exponent == 0)

    abs_x[:] = _calc_mantissa(abs_x, exponent, normal_mask, mantissa_bits, exponent_bias, rng)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - exponent_bias)) * (1.0 + abs_x),
        (2.0 ** (-exponent_bias + 1)) * abs_x,
    )

    info = torch.finfo(fp8_dtype)
    torch.clamp(sign, min=info.min, max=info.max, out=sign)
    return sign.to(fp8_dtype)


def _requantize_scaled(w16: Tensor, fp8_dtype: torch.dtype, seed: int) -> tuple[Tensor, Tensor]:
    """The recalculate+SR branch of quant_ops._TensorCoreFP8LayoutBase.quantize.

    Returns (fp8 qdata, new F32 scalar scale).
    """
    scale = torch.amax(w16.abs()).to(dtype=torch.float32) / torch.finfo(fp8_dtype).max
    # Guard against too-small a scale for fp16 input (comfy's original
    # comment: Prevent scale from being too small)
    if w16.dtype not in (torch.float32, torch.bfloat16):
        tensor_info = torch.finfo(w16.dtype)
        scale = 1.0 / torch.clamp(1.0 / scale, min=tensor_info.min, max=tensor_info.max)
    w16 = w16 * (1.0 / scale).to(w16.dtype)
    return stochastic_round_to_fp8(w16, fp8_dtype, seed), scale


def _group_lora_layers(sd: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    """Group by layer, normalizing both key formats into
    {layer_underscored: {kohya_suffix: tensor}}:

    - kohya/lycoris: ``lora_unet_{layer name with underscores}.{suffix}``
      (this app's own training output)
    - PEFT/comfy: ``diffusion_model.{dotted layer name}.{lora_A|lora_B|alpha}``
      (civitai / musubi / comfy ecosystem)
    """
    layers: dict[str, dict[str, Tensor]] = {}
    for key, tensor in sd.items():
        if key.startswith(_LORA_PREFIX):
            name, _, suffix = key.partition(".")
            layers.setdefault(name[len(_LORA_PREFIX):], {})[suffix] = tensor
        elif key.startswith(_COMFY_PREFIX):
            rest = key[len(_COMFY_PREFIX):]
            for peft_suffix, kohya_suffix in _PEFT_SUFFIXES:
                if rest.endswith("." + peft_suffix):
                    layer = rest[: -len(peft_suffix) - 1]
                    layers.setdefault(layer.replace(".", "_"), {})[kohya_suffix] = tensor
                    break
            else:
                raise ValueError(
                    f"fp8 merge could not recognize the suffix of a comfy-style LoRA key: {key}"
                )
    return layers


def _apply_lora_delta(
    w16: Tensor,
    tensors: dict[str, Tensor],
    strength: float,
    layer: str,
    source: str,
) -> Tensor:
    """Merge a single LoRA into a single layer, in comfy's order: compute the diff in fp32, add in fp16."""
    device = w16.device
    if "dora_scale" in tensors:
        raise ValueError(
            f"fp8 base model merge does not support DoRA (weight_decompose) LoRA: {source} layer {layer}. "
            f"Please mount a bf16 base model instead."
        )

    if "lora_down.weight" in tensors:  # plain LoRA / LoCon Linear
        if "lora_mid.weight" in tensors:
            raise ValueError(f"fp8 merge does not support LoCon mid (tucker) form: {source} layer {layer}")
        mat1 = tensors["lora_up.weight"].to(device=device, dtype=torch.float32)
        mat2 = tensors["lora_down.weight"].to(device=device, dtype=torch.float32)
        alpha = float(tensors["alpha"]) / mat2.shape[0] if "alpha" in tensors else 1.0
        lora_diff = torch.mm(
            mat1.flatten(start_dim=1), mat2.flatten(start_dim=1),
        ).reshape(w16.shape)
        w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    if "hada_w1_a" in tensors:  # LoHa
        if "hada_t1" in tensors or "hada_t2" in tensors:
            raise ValueError(f"fp8 merge does not support LoHa tucker (t1/t2) form: {source} layer {layer}")
        m1 = torch.mm(
            tensors["hada_w1_a"].to(device=device, dtype=torch.float32),
            tensors["hada_w1_b"].to(device=device, dtype=torch.float32),
        )
        m2 = torch.mm(
            tensors["hada_w2_a"].to(device=device, dtype=torch.float32),
            tensors["hada_w2_b"].to(device=device, dtype=torch.float32),
        )
        # dim semantics follow comfy weight_adapter/loha.py: the divisor is
        # w1_b's rank dimension
        alpha = float(tensors["alpha"]) / tensors["hada_w1_b"].shape[0] if "alpha" in tensors else 1.0
        lora_diff = (m1 * m2).reshape(w16.shape)
        w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    if "lokr_w1" in tensors or "lokr_w1_a" in tensors:  # LoKr
        if "lokr_t2" in tensors:
            raise ValueError(f"fp8 merge does not support LoKr tucker (t2) form: {source} layer {layer}")
        # dim semantics follow comfy weight_adapter/lokr.py: whichever of
        # w1/w2 is factorized assigns dim; if both are factorized, w2_b's
        # assignment comes later and wins (overwrites); if both are full
        # matrices, dim stays None -> the alpha coefficient is 1.0
        dim = None
        if "lokr_w1" in tensors:
            w1 = tensors["lokr_w1"].to(device=device, dtype=torch.float32)
        else:
            dim = tensors["lokr_w1_b"].shape[0]
            w1 = torch.mm(
                tensors["lokr_w1_a"].to(device=device, dtype=torch.float32),
                tensors["lokr_w1_b"].to(device=device, dtype=torch.float32),
            )
        if "lokr_w2" in tensors:
            w2 = tensors["lokr_w2"].to(device=device, dtype=torch.float32)
        else:
            dim = tensors["lokr_w2_b"].shape[0]
            w2 = torch.mm(
                tensors["lokr_w2_a"].to(device=device, dtype=torch.float32),
                tensors["lokr_w2_b"].to(device=device, dtype=torch.float32),
            )
        if "alpha" in tensors and dim is not None:
            alpha = float(tensors["alpha"]) / dim
        else:
            alpha = 1.0
        lora_diff = torch.kron(w1, w2).reshape(w16.shape)
        w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    raise ValueError(
        f"fp8 merge could not recognize the LoRA layer shape: {source} layer {layer} ({sorted(tensors)})"
    )


class Fp8LoraMergeAdapter:
    """Lifecycle handle for the merge writeback -- a duck-typed member of the
    daemon's adapters list.

    - ``detach()``: bit-for-bit restores the original weight and scale from
      the CPU backup (when switching LoRA / unloading)
    - ``network = None``: keeps the lycoris side's multiplier-setting path a
      safe no-op
    - ``supports_hot_reload = False``: a merge has no resident network, so
      the daemon's hot weight-swap path is disabled (must go through
      detach -> re-merge)
    """

    network = None
    supports_hot_reload = False

    def __init__(self, model: torch.nn.Module,
                 backup: dict[str, tuple[Tensor, Tensor | None]]) -> None:
        self._model = model
        self._backup = backup

    def detach(self) -> bool:
        if not self._backup:
            return True
        modules = dict(self._model.named_modules())
        for layer, (weight_cpu, scale_cpu) in self._backup.items():
            module = modules[layer]
            module.weight.data.copy_(weight_cpu.to(module.weight.device))
            if scale_cpu is not None:
                module.weight_scale.copy_(scale_cpu.to(module.weight_scale.device))
        self._backup = {}
        # The model reference must be dropped too -- this handle is also
        # held by the local `adapters` variable in _run_generate / _run_xy;
        # without clearing it, the old model stays pinned in VRAM in full
        # during a LoRA switch/reload (with XY grid LoRA switching, up to
        # three model copies can be resident at once -- upstream #499 showed
        # an actual OOM on the 3rd cell on a 32GB card).
        self._model = None
        return True


def merge_loras_into_fp8_model(
    model: torch.nn.Module,
    sources: list[tuple[dict[str, Tensor], float, str]],
) -> Fp8LoraMergeAdapter:
    """Bake multiple LoRAs into the Linear weights of a (partially) fp8 model,
    following comfy's merge semantics.

    ``sources``: [(state_dict, strength, source_name), ...], in mount order =
    comfy patch order. Returns a restore handle holding a CPU backup of the
    original weights.
    """
    module_index = {
        name.replace(".", "_"): (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    # layer -> [(tensors, strength, source), ...], preserving mount order
    per_layer: dict[str, list[tuple[dict[str, Tensor], float, str]]] = {}
    missing: list[str] = []
    for sd, strength, source in sources:
        grouped = _group_lora_layers(sd)
        if not grouped:
            raise ValueError(f"LoRA file has no {_LORA_PREFIX}* layers at all: {source}")
        for layer_key, tensors in grouped.items():
            if layer_key not in module_index:
                missing.append(f"{source}:{layer_key}")
                continue
            per_layer.setdefault(layer_key, []).append((tensors, strength, source))

    if missing and not per_layer:
        raise ValueError(f"None of the LoRA's layers matched the current model: {missing[:5]} ...")
    if missing:
        logger.warning(
            "fp8 merge: %d LoRA layers have no matching Linear in the model, skipped (same as comfy's behavior): %s%s",
            len(missing), missing[:5], " ..." if len(missing) > 5 else "",
        )

    backup: dict[str, tuple[Tensor, Tensor | None]] = {}
    for layer_key, layer_sources in per_layer.items():
        name, module = module_index[layer_key]
        weight = module.weight
        scale = getattr(module, "weight_scale", None)
        is_fp8 = weight.dtype in _FP8_TORCH_DTYPES

        backup[name] = (
            weight.detach().to("cpu", copy=True),
            None if scale is None else scale.detach().to("cpu", copy=True),
        )

        # Dequant to fp16 (comfy's lora_compute_dtype domain;
        # QuantizedTensor never passes through bf16); a non-quantized Linear
        # is likewise cast to fp16 to take part in the merge
        w16 = weight.detach().to(torch.float16)
        if scale is not None:
            w16 = w16 * scale.to(torch.float16)

        for tensors, strength, source in layer_sources:
            w16 = _apply_lora_delta(w16, tensors, strength, name, source)

        if is_fp8 and scale is not None:
            seed = string_to_seed(f"diffusion_model.{name}.weight")
            qdata, new_scale = _requantize_scaled(w16, weight.dtype, seed)
            weight.data.copy_(qdata)
            module.weight_scale.copy_(new_scale)
        elif is_fp8:
            # Plain-cast form: comfy has no set_func -> SR straight back to
            # fp8, no scale recalculation
            seed = string_to_seed(f"diffusion_model.{name}.weight")
            weight.data.copy_(stochastic_round_to_fp8(w16, weight.dtype, seed))
        else:
            # Non-quantized Linear: comfy set_weight's layout_type=None
            # branch -- cast straight back to the storage dtype, no SR
            weight.data.copy_(w16.to(weight.dtype))

    logger.info(
        "Krea2 fp8 merge: baked %d LoRA(s) into %d Linear layer(s) (backup kept, can detach to restore)",
        len(sources), len(per_layer),
    )
    return Fp8LoraMergeAdapter(model, backup)
