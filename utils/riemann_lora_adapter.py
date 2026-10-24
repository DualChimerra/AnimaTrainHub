"""Experimental fixed-rank LoRA adapter for Riemannion (arxiv:2507.12142).

Adapted from WalkingMeatAxolotl/AnimaLoraStudio, commit
42f510faa6e0a74676541393ca44fb506d8c2093 (GPL-3.0).

The update is A_L @ B.T. A zero-valued Z1 branch supplies the second
gradient needed by the optimizer; only Riemannion may update this adapter.
Random orthonormal bases and B=0 initialize an unchanged base model.
This differs from the paper's LOI initialization.
Exports ordinary lora_up/lora_down weights; training snapshots keep bases.
"""

from __future__ import annotations

from contextlib import contextmanager
from fnmatch import fnmatch
import json
import logging
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from utils.ortho_adapter import _lora_name, _matches_preset, _split_parent

logger = logging.getLogger(__name__)


def _random_orthonormal(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(rows, cols, generator=generator))
    return q


class RiemannLoRALinear(nn.Module):
    """nn.Linear wrapper: ΔW = A_L·Bᵀ; trainable lora_b (B) and lora_z1 (always 0)."""

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Linear,
        *,
        rank: int,
        generator: torch.Generator,
        multiplier: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, nn.Linear):
            raise TypeError(f"RiemannLoRALinear only supports nn.Linear, got {type(org_module)!r}")
        self.lora_name = lora_name
        self.org_module = org_module
        self.org_module.requires_grad_(False)
        self.multiplier = float(multiplier)

        out_dim, in_dim = org_module.weight.shape
        self.lora_dim = max(1, min(int(rank), int(out_dim), int(in_dim)))
        # Under block swap the offloaded base weights sit on the CPU (pinned); adapter
        # parameters and bases must live on the training device (swap moves frozen weights only)
        device = device if device is not None else org_module.weight.device
        r = self.lora_dim
        self.register_buffer("a_l", _random_orthonormal(out_dim, r, generator).to(device))
        self.register_buffer("b_r", _random_orthonormal(in_dim, r, generator).to(device))
        self.lora_b = nn.Parameter(torch.zeros(in_dim, r, device=device))
        self.lora_z1 = nn.Parameter(torch.zeros(out_dim, r, device=device))
        # The optimizer finds the owning layer through these tags (weakref avoids a param <-> module cycle)
        self.lora_b._riemann_role = "b"
        self.lora_z1._riemann_role = "z1"
        self.lora_b._riemann_layer = weakref.ref(self)
        self.lora_z1._riemann_layer = weakref.ref(self)

    @property
    def weight(self):
        return self.org_module.weight

    @property
    def bias(self):
        return self.org_module.bias

    @property
    def compute_dtype(self):
        return getattr(self.org_module, "compute_dtype", self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_forwarded = self.org_module(x)
        if self.multiplier == 0:
            return org_forwarded
        work = x.dtype
        h_b = F.linear(x, self.lora_b.mT.to(work))
        h_r = F.linear(x, self.b_r.mT.to(work))
        out = F.linear(h_b, self.a_l.to(work)) + F.linear(h_r, self.lora_z1.to(work))
        return org_forwarded + (out * self.multiplier).to(org_forwarded.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        return self.a_l.float() @ self.lora_b.float().mT

    @torch.no_grad()
    def distilled_lora_state(self, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
        """ΔW = A_L·Bᵀ = A_L·Rᵀ·Qᵀ (B=QR); Rᵀ=U S Vᵀ -> up=A_L U √S, down=√S Vᵀ Qᵀ."""
        q, r = torch.linalg.qr(self.lora_b.float(), mode="reduced")
        u, s, vh = torch.linalg.svd(r.mT)
        root = s.sqrt()
        lora_up = (self.a_l.float() @ u) * root.unsqueeze(0)
        lora_down = root.unsqueeze(1) * (vh @ q.mT)
        return {
            f"{self.lora_name}.lora_down.weight": lora_down.to(dtype).cpu().contiguous(),
            f"{self.lora_name}.lora_up.weight": lora_up.to(dtype).cpu().contiguous(),
            f"{self.lora_name}.alpha": torch.tensor(float(self.lora_dim)),
        }

    @torch.no_grad()
    def load_plain_lora_(self, lora_down: torch.Tensor, lora_up: torch.Tensor, scale: float) -> None:
        """Exact restore from a standard LoRA: ΔW = s·up·down, up=QR -> A_L=Q, B=(s·R·down)ᵀ."""
        expected_up = tuple(self.a_l.shape)
        expected_down = (self.lora_dim, self.lora_b.shape[0])
        if tuple(lora_up.shape) != expected_up or tuple(lora_down.shape) != expected_down:
            raise ValueError(
                f"{self.lora_name}: resume_lora must have the same rank and layer sizes. "
                f"Expected up={expected_up}, down={expected_down}; "
                f"got up={tuple(lora_up.shape)}, down={tuple(lora_down.shape)}"
            )
        q, r = torch.linalg.qr(lora_up.float(), mode="reduced")
        b = (scale * r @ lora_down.float()).mT
        b_r, _ = torch.linalg.qr(b, mode="reduced")
        self.a_l.copy_(q.to(self.a_l))
        self.lora_b.copy_(b.to(self.lora_b))
        self.b_r.copy_(b_r.to(self.b_r))
        self.lora_z1.zero_()


class RiemannLoRAAdapter:
    """AdapterProtocol for fixed-rank LoRA; used only by Muon with Use Riemannian."""

    def __init__(self, *, rank: int = 32, preset: dict | None = None, seed: int = 0) -> None:
        self.rank = int(rank)
        self._preset = preset
        self._seed = int(seed)
        self.loras = nn.ModuleList()
        self.use_lokr = False

    @contextmanager
    def disabled(self):
        """Temporarily disable the adapter for DOP's base-model reference."""
        previous = [layer.multiplier for layer in self.loras]
        try:
            for layer in self.loras:
                layer.multiplier = 0.0
            yield
        finally:
            for layer, value in zip(self.loras, previous):
                layer.multiplier = value

    def inject(self, model: nn.Module) -> dict[str, nn.Module]:
        generator = torch.Generator().manual_seed(self._seed)
        # Same as LycorisAdapter: take the training device from the model's first parameter
        # (block swap leaves the tail blocks on the CPU, so per-layer devices are wrong)
        device = next(model.parameters()).device
        preset = self._preset or {}
        # MOSS selects block classes; Anima/Krea select layer names.
        target_classes = set(preset.get("target_module", []))
        target_roots = [
            name for name, module in model.named_modules()
            if type(module).__name__ in target_classes
        ]
        replacements = []
        for name, module in model.named_modules():
            if any(fnmatch(name, pattern) for pattern in preset.get("exclude_name", [])):
                continue
            selected = _matches_preset(name, preset) or any(
                not root or name == root or name.startswith(root + ".") for root in target_roots
            )
            if isinstance(module, nn.Linear) and selected:
                parent, child_name = _split_parent(model, name)
                replacements.append((name, parent, child_name, module))
        if not replacements:
            raise ValueError("Riemann LoRA preset did not match any linear layers")
        for name, parent, child_name, module in replacements:
            layer = RiemannLoRALinear(
                _lora_name(name), module, rank=self.rank, generator=generator, device=device,
            )
            setattr(parent, child_name, layer)
            self.loras.append(layer)
        logger.info("Injected Riemann LoRA into %d layers; exporting standard LoRA", len(self.loras))
        return {layer.lora_name: layer for layer in self.loras}

    def on_step_begin(self, ctx) -> None:
        return None

    def regularization_loss(self, ctx) -> None:
        return None

    def excludes_weight_decay(self, param_name: str) -> bool:
        return False

    def get_params(self) -> list[nn.Parameter]:
        return [p for layer in self.loras for p in (layer.lora_b, layer.lora_z1)]

    def get_param_groups(self, weight_decay: float) -> list[dict]:
        return [{"params": self.get_params(), "weight_decay": weight_decay}]

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd: dict[str, torch.Tensor] = {}
        for layer in self.loras:
            sd[f"{layer.lora_name}.a_l"] = layer.a_l.detach().cpu()
            sd[f"{layer.lora_name}.b_r"] = layer.b_r.detach().cpu()
            sd[f"{layer.lora_name}.lora_b"] = layer.lora_b.detach().cpu()
        return sd

    def load_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True) -> Any:
        missing: list[str] = []
        unexpected = set(sd)
        for layer in self.loras:
            prefix = layer.lora_name
            if f"{prefix}.a_l" in sd:
                with torch.no_grad():
                    for key, target in (("a_l", layer.a_l), ("b_r", layer.b_r), ("lora_b", layer.lora_b)):
                        target.copy_(sd[f"{prefix}.{key}"].to(target))
                        unexpected.discard(f"{prefix}.{key}")
                    layer.lora_z1.zero_()
            elif f"{prefix}.lora_up.weight" in sd:
                up = sd[f"{prefix}.lora_up.weight"]
                down = sd[f"{prefix}.lora_down.weight"]
                alpha = float(sd.get(f"{prefix}.alpha", torch.tensor(float(up.shape[1]))))
                layer.load_plain_lora_(down.to(layer.a_l.device), up.to(layer.a_l.device), alpha / up.shape[1])
                for suffix in ("lora_up.weight", "lora_down.weight", "alpha"):
                    unexpected.discard(f"{prefix}.{suffix}")
            else:
                missing.append(prefix)
        if strict and (missing or unexpected):
            raise RuntimeError(f"RiemannLoRA load_state_dict mismatch: missing={missing}, unexpected={sorted(unexpected)}")
        return SimpleNamespace(missing_keys=missing, unexpected_keys=sorted(unexpected))

    def save(self, path: str | Path) -> None:
        sd: dict[str, torch.Tensor] = {}
        for layer in self.loras:
            sd.update(layer.distilled_lora_state())
        ss_args = {"algo": "lora", "source_algo": "riemannion", "weight_decompose": False, "rs_lora": False}
        ss_args.update(getattr(self, "metadata_extra", None) or {})
        meta = {
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(float(self.rank)),
            "ss_network_module": "lycoris.kohya",
            "ss_network_args": json.dumps(ss_args),
        }
        save_file(sd, str(path), metadata=meta)
        logger.info("Saved Riemann LoRA as standard LoRA: %s", path)

    def load(self, path: str | Path) -> None:
        """Warm start (resume_lora): accepts this adapter's snapshot keys or standard LoRA keys (lora_up/down).

        Raises when no layer matches (e.g. a LoKr / LoHa file) instead of silently training from zero.
        """
        with safe_open(str(path), framework="pt", device="cpu") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
        result = self.load_state_dict(sd, strict=False)
        if self.loras and len(result.missing_keys) == len(self.loras):
            raise RuntimeError(
                f"resume_lora does not match Riemann LoRA: {path}. "
                "Use a standard LoRA with matching layers or a Riemann training snapshot."
            )
        if result.missing_keys:
            logger.warning(
                "resume_lora matched %d/%d layers; unmatched layers start from zero: %s",
                len(self.loras) - len(result.missing_keys), len(self.loras),
                result.missing_keys[:5],
            )
