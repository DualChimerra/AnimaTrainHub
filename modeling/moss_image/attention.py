"""Masked attention for MOSS. Text padding never participates in image attention."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class AttentionParams:
    mode: str
    mask: torch.Tensor | None = None
    seqlens: tuple[int, ...] | None = None

    @classmethod
    def create_attention_params_from_mask(cls, mode, split, image_length, mask):
        image = torch.ones(mask.shape[0], image_length, device=mask.device, dtype=torch.bool)
        return cls(mode, torch.cat((image, mask.bool()), dim=1))

def attention(qkv, *, attn_params):
    if attn_params.seqlens is not None:
        lengths = attn_params.seqlens
        if attn_params.mode == "xformers":
            from xformers.ops import memory_efficient_attention
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask
            bias = BlockDiagonalMask.from_seqlens(list(lengths), device=qkv[0].device)
            return memory_efficient_attention(*(x.contiguous() for x in qkv), attn_bias=bias).flatten(2)
        # Standard batches also use packing without requiring xformers. Linear
        # layers still run once for the whole batch and swapped blocks move once.
        pieces = [x.split(lengths, dim=1) for x in qkv]
        outputs = []
        for q, k, v in zip(*pieces):
            if attn_params.mode == "flash":
                from flash_attn import flash_attn_func
                out = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous())
            else:
                out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            outputs.append(out.flatten(2))
        return torch.cat(outputs, dim=1)
    q, k, v = (x.transpose(1, 2) for x in qkv)
    mask = attn_params.mask[:, None, None, :]
    if attn_params.mode == "xformers":
        from xformers.ops import memory_efficient_attention
        # Per-sample attention avoids allocating a quadratic padding mask.
        outputs = []
        for i, valid in enumerate(attn_params.mask):
            length = int(valid.sum())
            out = memory_efficient_attention(
                q[i:i+1, :, :length].transpose(1, 2).contiguous(),
                k[i:i+1, :, :length].transpose(1, 2).contiguous(),
                v[i:i+1, :, :length].transpose(1, 2).contiguous(),
            )
            outputs.append(F.pad(out, (0, 0, 0, 0, 0, q.shape[2] - length)))
        return torch.cat(outputs).flatten(2)
    if attn_params.mode == "flash":
        from flash_attn import flash_attn_func
        outputs = []
        for i, valid in enumerate(attn_params.mask):
            length = int(valid.sum())
            out = flash_attn_func(
                q[i:i+1, :, :length].transpose(1, 2).contiguous(),
                k[i:i+1, :, :length].transpose(1, 2).contiguous(),
                v[i:i+1, :, :length].transpose(1, 2).contiguous(),
            )
            outputs.append(F.pad(out, (0, 0, 0, 0, 0, q.shape[2] - length)))
        return torch.cat(outputs).flatten(2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).flatten(2)
