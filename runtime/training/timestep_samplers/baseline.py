"""Baseline timestep 采样器：包装 training.timestep_sampling.sample_t。

非自适应；record / maybe_refresh 是 no-op。
覆盖 8 种 mode：logit_normal / uniform / logit_normal_low / mode /
mixed_uniform_low / mixed_uniform_logit / style_friendly / dual_peak。
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from dual_peak_sampling import DualPeakConfig
from training.timestep_samplers.dual_peak import config_from_args
from training.timestep_sampling import sample_t


class BaselineTimestepSampler:
    """sample_t 的 thin wrapper，使它符合 TimestepSamplerProtocol。"""

    def __init__(
        self,
        mode: str = "logit_normal",
        shift: float = 3.0,
        mix_low_prob: float = 0.0,
        timestep_schedule_shift: float = 1.0,
        style_snr_mean: float = -6.0,
        style_snr_sigma: float = 2.0,
        dual_peak_config: DualPeakConfig | None = None,
    ):
        self.mode = mode
        self.shift = shift
        self.mix_low_prob = mix_low_prob
        self.timestep_schedule_shift = timestep_schedule_shift
        self.style_snr_mean = style_snr_mean
        self.style_snr_sigma = style_snr_sigma
        self.dual_peak_config = (
            (dual_peak_config or DualPeakConfig())
            if (mode or "logit_normal").lower() == "dual_peak" else None
        )

    def sample(self, bs: int, device, *, token_counts=None) -> torch.Tensor:
        return sample_t(
            bs,
            device,
            mode=self.mode,
            shift=self.shift,
            mix_low_prob=self.mix_low_prob,
            timestep_schedule_shift=self.timestep_schedule_shift,
            style_snr_mean=self.style_snr_mean,
            style_snr_sigma=self.style_snr_sigma,
            dual_peak_config=self.dual_peak_config,
        )

    def record(self, t: torch.Tensor, raw_mse: torch.Tensor) -> None:
        return None

    def maybe_refresh(self, global_step: int) -> None:
        return None

    def status(self) -> dict:
        return {
            "kind": "baseline",
            "mode": self.mode,
            "shift": self.shift,
            "mix_low_prob": self.mix_low_prob,
            "timestep_schedule_shift": self.timestep_schedule_shift,
            "style_snr_mean": self.style_snr_mean,
            "style_snr_sigma": self.style_snr_sigma,
            **({"dual_peak": asdict(self.dual_peak_config)} if self.dual_peak_config is not None else {}),
        }

    # ─── pause/resume（ADR 0006 Addendum 1）：无状态采样器是真的无状态，no-op ───
    # Protocol body 里的 default 仅给 type checker；运行时不分发，必须显式实现。
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


def build(args, total_steps) -> BaselineTimestepSampler:
    """按 args 构建 BaselineTimestepSampler。total_steps 此采样器用不到。"""
    return BaselineTimestepSampler(
        mode=str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal"),
        shift=float(getattr(args, "timestep_shift", 3.0) or 3.0),
        mix_low_prob=float(getattr(args, "timestep_mix_low_prob", 0.0) or 0.0),
        timestep_schedule_shift=float(getattr(args, "timestep_schedule_shift", 1.0) or 1.0),
        # 显式 None 检查而非 `or`：mean 的合法值里有 0.0，sigma 没有但保持同一写法
        style_snr_mean=_float_or(getattr(args, "style_snr_mean", None), -6.0),
        style_snr_sigma=_float_or(getattr(args, "style_snr_sigma", None), 2.0),
        dual_peak_config=config_from_args(args),
    )


def _float_or(value, default: float) -> float:
    """None/缺省 → default；其余按 float 解析（0.0 是合法值，不能被 `or` 吞掉）。"""
    return default if value is None else float(value)
