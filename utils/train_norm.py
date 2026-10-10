"""Train norm weights and export their difference from the frozen base.

The absolute-weight optimization and ComfyUI .diff/.diff_b format follow
WhitecrowAurora/lora-rescripts, networks/lora_anima.py (0e3afaed).
Parametrizations keep the base intact for DOP, weighted inference and unload.
Family code supplies the target names; this module contains no Anima targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Callable

import torch
from torch import nn
from torch.nn.utils import parametrize


class NormWeight(nn.Module):
    def __init__(self, original: torch.Tensor, scale: Callable[[], float], *, trainable: bool):
        super().__init__()
        # Optimize absolute weights, as upstream does (also preserves weight decay).
        # FP32 masters retain updates smaller than a bf16 weight's rounding step.
        self.register_buffer("base", original.detach().float().clone(), persistent=False)
        self.value = nn.Parameter(self.base.clone(), requires_grad=trainable)
        self._scale = scale

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        scale = self._scale()
        if scale == 0:
            return original
        delta = (self.value - self.base) * scale
        return (original.float() + delta.to(original.device)).to(original.dtype)


@dataclass
class NormTarget:
    module: nn.Module
    name: str
    patch: NormWeight

    def detach(self) -> None:
        patches = self.module.parametrizations[self.name]
        index = next(i for i, patch in enumerate(patches) if patch is self.patch)
        if len(patches) == 1:
            parametrize.remove_parametrizations(self.module, self.name, leave_parametrized=False)
        else:
            del patches[index]


class TrainNorm:
    def __init__(self, model: nn.Module, patterns: list[str], prefix: str,
                 scale: Callable[[], float], *, trainable: bool):
        self.trainable = trainable
        self.scale = scale
        self.targets: dict[str, tuple[nn.Module, str]] = {}
        self.patches: dict[str, NormTarget] = {}
        for path, module in list(model.named_modules()):
            if not any(fnmatchcase(path, pattern) for pattern in patterns):
                continue
            for name, suffix in (("weight", "diff"), ("bias", "diff_b")):
                if getattr(module, name, None) is not None:
                    key = f"{prefix}_{path.replace('.', '_')}.{suffix}"
                    self.targets[key] = (module, name)
        if trainable:
            if not self.targets:
                raise ValueError("train_norm found no supported normalization weights in this model")
            for key in self.targets:
                self._attach(key)

    def _attach(self, key: str) -> NormTarget:
        if key not in self.patches:
            module, name = self.targets[key]
            original = (module.parametrizations[name].original
                        if parametrize.is_parametrized(module, name) else getattr(module, name))
            patch = NormWeight(original, self.scale, trainable=self.trainable)
            parametrize.register_parametrization(module, name, patch)
            self.patches[key] = NormTarget(module, name, patch)
        return self.patches[key]

    def parameters(self) -> list[nn.Parameter]:
        return [target.patch.value for target in self.patches.values()
                if target.patch.value.requires_grad]

    @staticmethod
    def state_key(diff_key: str) -> str:
        name, suffix = diff_key.rsplit(".", 1)
        return name.replace("lora_", "train_norm_", 1) + (".weight" if suffix == "diff" else ".bias")

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Absolute weights avoid subtract/add rounding in pause/resume checkpoints.
        return {self.state_key(key): target.patch.value.detach()
                for key, target in self.patches.items()}

    def export(self) -> dict[str, torch.Tensor]:
        return {key: (target.patch.value.detach().float() - target.patch.base.float()).cpu().contiguous()
                for key, target in self.patches.items()}

    def load(self, state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
        norm_keys = {key for key in state if key.endswith((".diff", ".diff_b"))
                     or key.startswith("train_norm_")}
        lookup = {self.state_key(key): key for key in self.targets}
        lookup.update({key: key for key in self.targets})
        unknown = norm_keys - lookup.keys()
        if unknown:
            raise ValueError(f"Unsupported normalization weights: {sorted(unknown)[:3]}")
        # Validate the entire patch before applying any of it.
        seen = set()
        for key in norm_keys:
            diff_key = lookup[key]
            if diff_key in seen:
                raise ValueError(f"Duplicate absolute and diff normalization weights: {diff_key}")
            seen.add(diff_key)
            module, name = self.targets[lookup[key]]
            value = state[key]
            if value.shape != getattr(module, name).shape:
                raise ValueError(f"Normalization size mismatch for {key}: {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite normalization weights: {key}")
        loaded = set()
        with torch.no_grad():
            for key in sorted(norm_keys):
                diff_key = lookup[key]
                patch = self._attach(diff_key).patch
                value = state[key].to(device=patch.value.device, dtype=torch.float32)
                patch.value.copy_(patch.base + value if key == diff_key else value)
                loaded.add(diff_key)
            # Daemon hot reload: a checkpoint without norm weights must clear the
            # previous checkpoint's changes, including when strict=False.
            for key, target in self.patches.items():
                if key not in loaded:
                    target.patch.value.copy_(target.patch.base)
        missing = [self.state_key(key) for key in self.targets if key not in loaded] if self.trainable else []
        return {key: value for key, value in state.items() if key not in norm_keys}, missing

    def detach(self) -> None:
        for target in self.patches.values():
            target.detach()
        self.patches.clear()
