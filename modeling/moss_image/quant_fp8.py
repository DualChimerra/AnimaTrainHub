"""Scaled FP8 E4M3 storage for frozen MOSS weights (qfloat8).

Keep ordinary Linear/Embedding modules so adapter targeting and layer offload
retain their existing behavior. Computation uses BF16/FP32, not FP8 activations.
"""
from types import MethodType

import torch
import torch.nn.functional as F


@torch.no_grad()
def quantize_weight(weight):
    weight = weight.float()
    bound = torch.finfo(torch.float8_e4m3fn).max
    scale = weight.abs().amax(dim=1, keepdim=True) / bound
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (weight / scale).clamp(-bound, bound).to(torch.float8_e4m3fn), scale


def _dequantize(weight, scale, dtype):
    return weight.to(dtype) * scale.to(device=weight.device, dtype=dtype)


class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, scale, bias):
        # Save compact storage, not a full BF16 copy per layer for backward.
        ctx.save_for_backward(weight, scale)
        ctx.input_dtype = inputs.dtype
        with torch.autocast(inputs.device.type, enabled=False):
            return F.linear(inputs, _dequantize(weight, scale, inputs.dtype), bias)

    @staticmethod
    def backward(ctx, grad_output):
        weight, scale = ctx.saved_tensors
        with torch.autocast(grad_output.device.type, enabled=False):
            grad_input = grad_output.to(ctx.input_dtype) @ _dequantize(weight, scale, ctx.input_dtype)
        return grad_input, None, None, None


def _linear_forward(self, inputs):
    inputs = inputs.to(self.compute_dtype)
    bias = self.bias.to(inputs.dtype) if self.bias is not None else None
    return _FrozenLinear.apply(inputs, self.weight, self.scale_weight, bias)


def _embedding_forward(self, indices):
    flat = indices.reshape(-1)
    rows = self.weight.index_select(0, flat)
    scales = self.scale_weight.index_select(0, flat)
    values = _dequantize(rows, scales, self.compute_dtype)
    return values.reshape(*indices.shape, self.embedding_dim)


def patch_module(module, scale, dtype):
    module.register_buffer("scale_weight", scale, persistent=False)
    module.compute_dtype = dtype
    forward = _embedding_forward if isinstance(module, torch.nn.Embedding) else _linear_forward
    module.forward = MethodType(forward, module)
