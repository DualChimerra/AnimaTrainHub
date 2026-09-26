"""
Optimizer Utils Module - Optimizer creation
============================================
Supports multiple optimizers:
1. Standard AdamW - PyTorch built-in
2. 8-bit AdamW (bitsandbytes) - Memory-efficient
3. Prodigy (prodigyopt) - Adaptive optimizer that needs no lr tuning
4. ProdigyPlusScheduleFree (prodigy-plus-schedule-free) - Schedule-Free + Prodigy,
   addressing the mutation-episode / style-shift problem Prodigy has in diffusion LoRA training.
5. Lion - EvoLved Sign Momentum (Chen et al., 2023, arxiv 2302.06675)
6. Automagic - Per-parameter adaptive lr via sign-agreement tracking
   Original author: Ostris (https://github.com/ostris/ai-toolkit, MIT license, Copyright (c)
   2024 Ostris, LLC). The bf16 Kahan summation path is borrowed from tdrussell/diffusion-pipe.
7. SOAP - Adam in the Shampoo eigenbasis (Vyas et al., 2024, arxiv 2409.11321)
8. SOAP-SF - Schedule-Free SOAP, SOAP preconditioning + Schedule-Free trajectory
   (Defazio et al., 2024, "The Road Less Scheduled", arxiv 2405.15682). The SOAP class
   is implemented in utils/soap_optimizer.py (MIT, Copyright (c) 2024 Nikhil Vyas).
9. CAME - Confidence-guided Adaptive Memory Efficient optimizer
   (Luo et al., 2023, ACL 2023, arxiv 2307.02047). Adafactor-style factored second-moment
   to save memory + confidence guidance (instability EMA) to suppress the update noise
   introduced by the factorization approximation.
   Derived from the official implementation yangluo7/CAME (MIT, Copyright (c) 2023 Yang Luo).
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
import logging
from typing import List, Dict, Any, Optional, Iterator

import torch
from torch import nn
from torch.optim import Optimizer, AdamW

logger = logging.getLogger(__name__)

# Try importing bitsandbytes
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    from optimum.quanto import QBytesTensor
except ImportError:
    QBytesTensor = ()


def _is_param_groups(params: Any) -> bool:
    if isinstance(params, (list, tuple)) and len(params) > 0:
        return isinstance(params[0], dict) and "params" in params[0]
    return False


def _as_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().item())
        return float(value)
    except (RuntimeError, TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# kwargs the user explicitly configures via the schema (ppsf_* fields). If the upstream
# version doesn't accept these, a silent drop would make the user's toggles/values silently
# stop working, possibly not noticed until 8 hours later when training results look off --
# so these must fail loud rather than just log a warning.
_USER_EXPOSED_PPSF_KWARGS = frozenset({
    "d_coef", "prodigy_steps",
    "split_groups", "split_groups_mean",
    "use_speed", "fused_back_pass",
    "use_stableadamw",
})


def _filter_kwargs_by_signature(cls_or_fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(cls_or_fn)
    except (TypeError, ValueError):
        return dict(kwargs)

    accepted, has_var_keyword = set(), False
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            break
        accepted.add(name)

    if has_var_keyword:
        return dict(kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = [k for k in kwargs if k not in accepted]
    if dropped:
        exposed_dropped = [k for k in dropped if k in _USER_EXPOSED_PPSF_KWARGS]
        if exposed_dropped:
            cls_name = getattr(cls_or_fn, "__name__", str(cls_or_fn))
            raise RuntimeError(
                f"[optimizer] {cls_name} does not support the following user-configured kwargs: "
                f"{exposed_dropped}. This is likely a prodigy-plus-schedule-free library version "
                f"mismatch (check the version with pip show prodigy-plus-schedule-free). "
                f"Upgrade/downgrade the dependency, or disable the corresponding field in the yaml."
            )
        logger.warning(
            f"[optimizer] Dropped unsupported kwargs for "
            f"{getattr(cls_or_fn, '__name__', cls_or_fn)}: {dropped}"
        )
    return filtered


def create_optimizer(
    optimizer_type: str,
    params: Iterator[nn.Parameter],
    learning_rate: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    **kwargs
) -> Optimizer:
    """
    Create an optimizer

    Creates different optimizer types based on config. This is a factory-pattern
    application that centralizes optimizer creation logic for easier maintenance and extension.

    Args:
        optimizer_type: optimizer type ("adamw", "adamw8bit", "prodigy")
        params: iterator of model parameters
        learning_rate: learning rate
        betas: Adam beta parameters (beta1, beta2)
        weight_decay: weight decay coefficient
        eps: numerical stability epsilon
        **kwargs: other optimizer-specific parameters

    Returns:
        Optimizer: the created optimizer instance

    Raises:
        ValueError: if the optimizer type is not supported
        ImportError: if a required library is not installed
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adamw8bit":
        return create_8bit_adamw(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs
        )

    elif optimizer_type == "adamw":
        return create_standard_adamw(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs
        )

    elif optimizer_type == "prodigy":
        return create_prodigy(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs
        )

    elif optimizer_type == "lion":
        return create_lion(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs,
        )

    elif optimizer_type == "automagic":
        return create_automagic(
            params=params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

    elif optimizer_type == "automagic_v2":
        return create_automagic_v2(
            params=params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

    elif optimizer_type == "prodigy_plus_schedulefree":
        return create_prodigy_plus_schedulefree(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )

    elif optimizer_type == "soap":
        return create_soap(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )

    elif optimizer_type == "soap_sf":
        return create_soap_sf(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )

    elif optimizer_type == "came":
        return create_came(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: adamw, automagic, automagic_v2, came, lion, prodigy, "
            f"prodigy_plus_schedulefree, soap, soap_sf"
        )


def iter_optimizer_params(params) -> Iterator[nn.Parameter]:
    """Flatten a "parameter list" or a "param group list" into individual parameters.

    The trainer works with the param group shape (`injector.get_param_groups()` returns
    `[{"params": [...], "weight_decay": ...}, ...]`, and LoKr also groups w1 separately),
    while a plain parameter list is also valid input. torch.optim accepts both, so **don't
    flatten when constructing the optimizer**; flattening is only needed for side logic
    like counting parameters -- calling `p.numel()` directly would hit
    `'dict' object has no attribute 'numel'` on the group shape.
    """
    for item in params:
        if isinstance(item, dict):
            yield from item.get("params", ())
        else:
            yield item


def create_8bit_adamw(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    min_8bit_size: int = 4096,
    **kwargs
) -> Optimizer:
    """
    Create an 8-bit AdamW optimizer

    8-bit AdamW is a memory-efficient optimizer provided by the bitsandbytes library.
    It quantizes the optimizer state (momentum, second moment) to 8-bit, which can
    reduce optimizer VRAM usage by roughly 50%.

    Principle:
    - Most deep learning parameters don't need full 32-bit precision to store optimizer state
    - Through block-wise quantization and dynamic range adjustment, 8-bit can maintain good optimization performance

    Use cases:
    - VRAM-constrained training (e.g. training a large model on a single RTX 3090)
    - LoRA training (although LoRA has few parameters, 8-bit can save memory further)

    Parameter notes:
    - min_8bit_size: tensors smaller than this size stay 32-bit
      This is because small tensors gain little from 8-bit quantization, and may actually lose precision

    Args:
        params: model parameters
        lr: learning rate
        betas: Adam beta parameters
        weight_decay: weight decay
        eps: epsilon
        min_8bit_size: minimum tensor size for 8-bit quantization

    Returns:
        bnb.optim.AdamW8bit: 8-bit AdamW optimizer
    """
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError(
            "bitsandbytes is required for 8-bit AdamW. "
            "Install with: pip install bitsandbytes"
        )
    
    print(f"Creating 8-bit AdamW optimizer (lr={lr}, weight_decay={weight_decay})")
    print(f"  min_8bit_size: {min_8bit_size}")

    # Convert parameters to a list (bitsandbytes needs indexable parameters)
    param_list = list(params)

    # Create the optimizer (the param group shape is passed through as-is; the torch.optim protocol already accepts it)
    optimizer = bnb.optim.AdamW8bit(
        param_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        min_8bit_size=min_8bit_size,
        **kwargs
    )
    
    # Calculate memory savings
    total_params = sum(p.numel() for p in iter_optimizer_params(param_list))
    # 8-bit optimizer state: roughly 2 bytes per parameter (vs 8 bytes for 32-bit)
    # Saves roughly 75% of optimizer state memory
    estimated_savings_gb = (total_params * 6) / (1024 ** 3)  # saves 6 bytes per param
    print(f"  [OK] 8-bit AdamW created (estimated memory savings: {estimated_savings_gb:.2f} GB)")
    
    return optimizer


def create_standard_adamw(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    **kwargs
) -> Optimizer:
    """
    Create a standard AdamW optimizer

    The standard PyTorch AdamW implementation. Used as a fallback option when other
    optimizers are unavailable.

    AdamW characteristics:
    - Decouples weight decay from the gradient update (decoupled weight decay)
    - Performs better than Adam + L2 regularization
    - The de facto standard optimizer in modern deep learning

    Args:
        params: model parameters
        lr: learning rate
        betas: Adam beta parameters
        weight_decay: weight decay
        eps: epsilon

    Returns:
        AdamW: standard AdamW optimizer
    """
    print(f"Creating standard AdamW optimizer (lr={lr}, weight_decay={weight_decay})")

    # Convert parameters to a list
    param_list = list(params)
    
    optimizer = AdamW(
        param_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        **kwargs
    )
    
    print("  [OK] AdamW optimizer created")

    return optimizer


# ============================================================================
# Automagic optimizer + Auto8bitTensor helper
#
# Adapted from ostris/ai-toolkit (https://github.com/ostris/ai-toolkit), which
# also flowed through tdrussell/diffusion-pipe (added Kahan summation for
# bfloat16). The core sign-agreement schedule, 8-bit lr_mask, Adafactor-style
# factored 2nd moment, and stochastic-rounding helpers below are Ostris's
# design. We re-implement on top of the upstream structure and align with
# diffusion-pipe's bf16 Kahan path.
#
# MIT License — Copyright (c) 2024 Ostris, LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================


class Auto8bitTensor:
    def __init__(self, data: torch.Tensor | dict[str, Any]) -> None:
        if isinstance(data, dict):
            self.quantized = data["quantized"]
            self.scale = data["scale"]
            self.orig_dtype = data["orig_dtype"]
            return
        abs_max = data.abs().max().item()
        self.scale = abs_max / 127.0 if abs_max > 0 else 1.0
        self.quantized = (data / self.scale).round().clamp(-127, 127).to(torch.int8)
        self.orig_dtype = data.dtype

    def dequantize(self) -> torch.Tensor:
        return self.quantized.to(dtype=torch.float32) * self.scale

    def to(self, *args, **kwargs):
        return self.dequantize().to(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "quantized": self.quantized,
            "scale": self.scale,
            "orig_dtype": self.orig_dtype,
        }


def _copy_stochastic_bf16(target: torch.Tensor, source: torch.Tensor) -> None:
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))


def _copy_stochastic(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.dtype == torch.float32:
        target.copy_(source)
        return
    if target.dtype == torch.bfloat16:
        _copy_stochastic_bf16(target, source)
        return
    target.copy_(source.to(target.dtype))


# Note on stochastic-rounding grad accumulation:
# Upstream (ostris/ai-toolkit, tdrussell/diffusion-pipe) registers a
# post-accumulate-grad hook to do fp32 grad accumulation with stochastic
# rounding. Automagic (v1) deliberately does NOT enable that path because it
# silently breaks two common training paths (Automagic2 / v2 below is the
# explicit experimental opt-in for the fused-backward design — gated off in
# the UI by default, see SystemConfig.enable_automagic_v2):
#   1. torch.cuda.amp.GradScaler.unscale_ skips params where p.grad is None
#      — the hook deletes p.grad after each backward, so unscale never runs
#      and the optimizer would step on still-scaled gradients.
#   2. torch.nn.utils.clip_grad_norm_ skips p.grad is None, so grad clipping
#      becomes a no-op.
# bf16 numerical stability is instead handled inside Automagic.step via
# Kahan compensated summation (see `shift` buffer below). See upstream
# diffusion-pipe optimizers/automagic.py:72-79 (commented out by author).


class Automagic(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-6,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if min_lr < 0.0:
            raise ValueError(f"Invalid min_lr: {min_lr}")
        if max_lr <= 0.0 or max_lr < min_lr:
            raise ValueError(f"Invalid max_lr: {max_lr}")
        if lr_bump < 0.0:
            raise ValueError(f"Invalid lr_bump: {lr_bump}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if clip_threshold <= 0.0:
            raise ValueError(f"Invalid clip_threshold: {clip_threshold}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.lr = min(lr, max_lr)
        if lr > 1e-3:
            logger.warning("Automagic start lr %s is high; clamping to 1e-6", lr)
            self.lr = 1e-6
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        super().__init__(
            params,
            dict(
                lr=self.lr,
                eps=eps,
                clip_threshold=clip_threshold,
                beta2=beta2,
                weight_decay=weight_decay,
            ),
        )
        self.base_lrs = [self.lr for _ in self.param_groups]

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @staticmethod
    def _get_lr(param_state: dict[str, Any]) -> float:
        avg_lr = param_state.get("avg_lr")
        if avg_lr is None:
            return 0.0
        if isinstance(avg_lr, torch.Tensor):
            return float(avg_lr.detach().float().item())
        return float(avg_lr)

    def _get_group_lr(self, group: dict[str, Any]) -> float:
        lrs = [self._get_lr(self.state[p]) for p in group["params"] if p in self.state]
        if not lrs:
            return self.lr
        return sum(lrs) / len(lrs)

    def get_learning_rates(self) -> list[float]:
        lrs = [self._get_group_lr(group) for group in self.param_groups]
        return lrs or self.base_lrs

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs)

    def initialize_state(self, p: nn.Parameter) -> None:
        state = self.state[p]
        state["step"] = 0
        if "lr_mask" not in state:
            state["lr_mask"] = Auto8bitTensor(
                torch.ones(p.shape, device=p.device, dtype=torch.float32) * self.lr
            )
        state["avg_lr"] = torch.mean(state["lr_mask"].to(torch.float32))
        if "last_polarity" not in state:
            state["last_polarity"] = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
        if len(p.shape) >= 2:
            state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1]).to(p)
            state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:]).to(p)
        else:
            state["exp_avg_sq"] = torch.zeros_like(p)
        state["RMS"] = 0
        # Kahan compensated summation for bf16 — keeps the rounded-off portion
        # of each update so it can be applied on the next step. Aligns with
        # upstream diffusion-pipe (optimizers/automagic.py:354-356).
        if p.dtype == torch.bfloat16 and "shift" not in state:
            state["shift"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad
                if grad.dtype != torch.float32:
                    grad = grad.to(torch.float32)
                if grad.is_sparse:
                    raise RuntimeError("Automagic does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    self.initialize_state(p)
                factored = len(grad.shape) >= 2
                if factored:
                    state.setdefault("exp_avg_sq_row", torch.zeros(p.shape[:-1]).to(grad))
                    state.setdefault("exp_avg_sq_col", torch.zeros(p.shape[:-2] + p.shape[-1:]).to(grad))
                    state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                    state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                else:
                    state.setdefault("exp_avg_sq", torch.zeros_like(grad))
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.dequantize() if QBytesTensor and isinstance(p, QBytesTensor) else p
                if p.dtype != torch.float32:
                    p_data_fp32 = p_data_fp32.clone().float()

                state["step"] = state.get("step", 0) + 1
                state["RMS"] = self._rms(p_data_fp32)
                beta2 = group["beta2"]
                eps = group["eps"]
                update = (grad ** 2) + eps
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=(1.0 - beta2))
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=(1.0 - beta2))
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=(1.0 - beta2))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                if "last_polarity" not in state or "lr_mask" not in state:
                    self.initialize_state(p)
                current_polarity = (update > 0).to(torch.bool)
                sign_agreement = torch.where(state["last_polarity"] == current_polarity, 1, -1)
                state["last_polarity"] = current_polarity
                lr_mask = state["lr_mask"].to(torch.float32)
                new_lr = torch.where(
                    sign_agreement > 0,
                    lr_mask + self.lr_bump,
                    lr_mask - self.lr_bump,
                )
                new_lr = torch.clamp(new_lr, min=self.min_lr, max=self.max_lr)
                update.mul_(new_lr)
                state["lr_mask"] = Auto8bitTensor(new_lr)
                state["avg_lr"] = torch.mean(new_lr)

                if group["weight_decay"] != 0:
                    weight_decay_update = p_data_fp32 * (-group["weight_decay"]) * new_lr
                else:
                    weight_decay_update = None

                if p.dtype == torch.bfloat16:
                    # Kahan compensated summation — matches upstream
                    # diffusion-pipe (optimizers/automagic.py:308-318). Trades
                    # one extra bf16 buffer (`state['shift']`) for unbiased,
                    # zero-variance accumulation over long bf16 training; the
                    # alternative stochastic-rounding path injects ~scale/2
                    # noise per step into the lr_mask sign-agreement signal.
                    update.mul_(-1)
                    if weight_decay_update is not None:
                        update.add_(weight_decay_update)
                    shift = state.setdefault("shift", torch.zeros_like(p))
                    shift.add_(update)
                    # Use grad tensor as scratch buffer for the pre-update p,
                    # so shift carries forward only the bf16 rounding error.
                    grad.copy_(p.detach())
                    p.add_(shift)
                    shift.add_(grad.sub_(p))
                else:
                    if weight_decay_update is not None:
                        p_data_fp32.add_(weight_decay_update)
                    p_data_fp32.add_(-update)
                    if p.dtype != torch.float32:
                        _copy_stochastic(p, p_data_fp32)

        return loss

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["state"] = {
            p: {
                **{k: v for k, v in state.items() if k != "lr_mask"},
                **({"lr_mask": state["lr_mask"].state_dict()} if "lr_mask" in state else {}),
            }
            for p, state in state_dict["state"].items()
        }
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        converted = {
            "state": {},
            "param_groups": state_dict.get("param_groups", []),
        }
        for param_id, state in state_dict.get("state", {}).items():
            converted_state = dict(state)
            if isinstance(converted_state.get("lr_mask"), dict):
                converted_state["lr_mask"] = Auto8bitTensor(converted_state["lr_mask"])
            converted["state"][param_id] = converted_state
        result = super().load_state_dict(converted)
        # PyTorch only moves float state to the param's dtype/device; bool
        # tensors and the int8 inside Auto8bitTensor are not covered, and are
        # left on CPU after resume. Move them all back explicitly.
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                device = p.device
                if isinstance(st.get("last_polarity"), torch.Tensor):
                    st["last_polarity"] = st["last_polarity"].to(device=device)
                if isinstance(st.get("lr_mask"), Auto8bitTensor):
                    st["lr_mask"].quantized = st["lr_mask"].quantized.to(device=device)
        return result


def create_automagic(
    params: Iterator[nn.Parameter],
    lr: float,
    min_lr: float = 1e-7,
    max_lr: float = 1e-3,
    lr_bump: float = 1e-6,
    eps: float = 1e-30,
    clip_threshold: float = 1.0,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
    **kwargs,
) -> Optimizer:
    # Automagic's upstream recommends init lr=1e-6 (a per-param adaptive
    # starting point); anything in the >1e-5 range is an AdamW-style lr used
    # by mistake, and the sign-agreement schedule takes many steps to
    # converge back to the working range from too high a start. The UI
    # rewrites lr=1e-6 automatically when switching optimizer_type; this is a
    # fallback for the saved-config / CLI path.
    if lr > 1e-5:
        logger.warning(
            "Automagic initial lr=%.2e is far above the recommended 1e-6; sign-agreement adaptation "
            "converges slowly from too high a start, recommend setting it to 1e-6 (per-param lr is auto-tuned within [min_lr, max_lr])",
            lr,
        )
    param_list = params if _is_param_groups(params) else list(params)
    optimizer = Automagic(
        param_list,
        lr=lr,
        min_lr=min_lr,
        max_lr=max_lr,
        lr_bump=lr_bump,
        eps=eps,
        clip_threshold=clip_threshold,
        beta2=beta2,
        weight_decay=weight_decay,
        **kwargs,
    )
    print(
        f"Creating Automagic optimizer (lr={lr}, min_lr={min_lr}, max_lr={max_lr}, "
        f"lr_bump={lr_bump}, beta2={beta2}, weight_decay={weight_decay})"
    )
    print("  [OK] Automagic optimizer created")
    return optimizer


class Automagic2(Optimizer):
    """Automagic v2 — per-parameter scalar LR, fused into backward pass.

    Instead of a per-element lr_mask, keeps one scalar lr per parameter tensor.
    The optimizer step is fused into backward via register_post_accumulate_grad_hook:
    each parameter is updated and its grad freed as soon as autograd finishes
    accumulating into it. .step() is therefore a no-op and peak VRAM stays low.

    Incompatible with GradScaler, clip_grad_norm_ and gradient accumulation
    (params update during backward) — see the design note above class Automagic.
    Experimental: hidden by default in the UI (SystemConfig.enable_automagic_v2).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-6,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        agreement_threshold: float = 0.5,
    ) -> None:
        if lr > 1e-3:
            logger.warning("Automagic2 start lr %s is high; forcing to 1e-6.", lr)
            lr = 1e-6
        defaults = dict(
            lr=lr, min_lr=min_lr, max_lr=max_lr, lr_bump=lr_bump,
            beta2=beta2, eps=eps, clip_threshold=clip_threshold,
            weight_decay=weight_decay, agreement_threshold=agreement_threshold,
        )
        super().__init__(params, defaults)

        self._hook_handles: list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_backward_hook(group)
                    )
                    self._hook_handles.append(handle)

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        logger.info(f"Automagic2 total training params: {total:,}")

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr"] = torch.full((), float(group["lr"]), dtype=torch.float32, device=p.device)
        state["last_polarity"] = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
        # Second moment is fixed at fp32: under bf16 storage, the per-step
        # relative increment of (1-beta2)=1e-3 is smaller than bf16's
        # relative resolution (~3.9e-3), so the EMA would get swallowed by
        # rounding on write-back and stall.
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1], dtype=torch.float32, device=p.device)
            state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:], dtype=torch.float32, device=p.device)
        else:
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)

    def _make_backward_hook(self, group):
        def _hook(p: torch.Tensor):
            self._update_param(p, group)
        return _hook

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        if len(state) == 0:
            self._init_state(p, group)

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Automagic2 does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)

        # The fused path bypasses the training loop's step-boundary NaN
        # gradient check (by the time the hook finishes, p.grad is already
        # None), so this has to defend itself here -- otherwise one bad
        # micro-batch poisons the weights directly.
        if not torch.isfinite(grad).all():
            logger.warning(
                "Automagic2: param %s gradient contains NaN/Inf, skipping this fused update",
                tuple(p.shape),
            )
            p.grad = None
            return

        beta2 = group["beta2"]
        eps = group["eps"]
        sq = (grad * grad).add_(eps)

        if p.dim() >= 2:
            row_state = state["exp_avg_sq_row"]
            col_state = state["exp_avg_sq_col"]
            if row_state.dtype == torch.float32:
                row, col = row_state, col_state
            else:
                row = row_state.to(torch.float32)
                col = col_state.to(torch.float32)
            row.mul_(beta2).add_(sq.mean(dim=-1), alpha=1.0 - beta2)
            col.mul_(beta2).add_(sq.mean(dim=-2), alpha=1.0 - beta2)
            if row_state.dtype != torch.float32:
                row_state.copy_(row.to(row_state.dtype))
                col_state.copy_(col.to(col_state.dtype))
            update = self._approx_sq_grad(row, col).mul_(grad)
        else:
            v_state = state["exp_avg_sq"]
            if v_state.dtype == torch.float32:
                v = v_state
            else:
                v = v_state.to(torch.float32)
            v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
            if v_state.dtype != torch.float32:
                v_state.copy_(v.to(v_state.dtype))
            update = v.rsqrt().mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        # Per-element sign agreement collapsed to a single scalar lr decision
        cur_polarity = update > 0
        last_polarity = state["last_polarity"]
        agreement = (cur_polarity == last_polarity).to(torch.float32).mean()
        state["last_polarity"] = cur_polarity

        lr_t = state["lr"]
        if state["step"] > 0:
            direction = (agreement >= group["agreement_threshold"]).to(lr_t.dtype) * 2.0 - 1.0
            lr_t.add_(direction, alpha=group["lr_bump"]).clamp_(
                min=group["min_lr"], max=group["max_lr"]
            )
        state["step"] += 1

        update.mul_(lr_t)
        wd = group["weight_decay"]

        if p.dtype == torch.bfloat16:
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.addcmul_(new_p_fp32, lr_t, value=wd)
            new_p_fp32.sub_(update)
            # Stochastic rounding fp32 → bf16
            as_int = new_p_fp32.view(torch.int32)
            as_int.add_(torch.randint_like(as_int, 1 << 16)).bitwise_and_(-65536)
            p.copy_(new_p_fp32)
        else:
            if wd != 0.0:
                p_fp32 = p if p.dtype == torch.float32 else p.to(torch.float32)
                update.addcmul_(p_fp32, lr_t, value=wd)
            p.add_(update.to(p.dtype), alpha=-1.0)

        p.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss

    def get_learning_rates(self) -> "list[float]":
        out = []
        for group in self.param_groups:
            lrs = [
                float(self.state[p]["lr"])
                for p in group["params"]
                if p in self.state and "lr" in self.state[p]
            ]
            out.append(sum(lrs) / len(lrs) if lrs else float(group["lr"]))
        return out

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else float(self.defaults["lr"])

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                # PyTorch load casts float state to the param dtype (with bf16
                # training this would downgrade the fp32 scalar/second moment);
                # restore them to fp32 uniformly here.
                for key in ("lr", "exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq"):
                    v = st.get(key)
                    if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                        st[key] = v.to(torch.float32)


def create_automagic_v2(
    params: Iterator[nn.Parameter],
    lr: float,
    min_lr: float = 1e-7,
    max_lr: float = 1e-3,
    lr_bump: float = 1e-6,
    eps: float = 1e-30,
    clip_threshold: float = 1.0,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
    agreement_threshold: float = 0.5,
    **kwargs,
) -> Optimizer:
    if lr > 1e-5:
        logger.warning(
            "Automagic2 initial lr=%.2e is far above the recommended 1e-6; sign-agreement adaptation "
            "converges slowly from too high a start, recommend setting it to 1e-6", lr,
        )
    param_list = params if _is_param_groups(params) else list(params)
    optimizer = Automagic2(
        param_list, lr=lr, min_lr=min_lr, max_lr=max_lr, lr_bump=lr_bump,
        eps=eps, clip_threshold=clip_threshold, beta2=beta2, weight_decay=weight_decay,
        agreement_threshold=agreement_threshold,
    )
    logger.info(
        f"Creating Automagic v2 (lr={lr}, min_lr={min_lr}, max_lr={max_lr}, "
        f"lr_bump={lr_bump}, beta2={beta2}, wd={weight_decay}, "
        f"agreement_threshold={agreement_threshold})"
    )
    return optimizer


class Lion(Optimizer):
    """EvoLved Sign Momentum optimizer (Chen et al., 2023).

    Paper: "Symbolic Discovery of Optimization Algorithms"
        https://arxiv.org/abs/2302.06675  (Google Brain)

    Reference implementations:
    - Google official:    https://github.com/google/automl/tree/master/lion
    - Community PyTorch:  https://github.com/lucidrains/lion-pytorch

    This is a minimal re-implementation aligning with the paper's Algorithm 1
    and Google's reference; no dependency on `lion-pytorch`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def create_lion(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.99),
    weight_decay: float = 0.0,
    **kwargs,
) -> Optimizer:
    # Lion's paper (Chen et al. 2023, arxiv 2302.06675 SS4.3) rule of thumb:
    # lr ~= AdamW lr / 3, weight_decay 3-10x AdamW. Switching directly from
    # AdamW's default lr=1e-4 to Lion easily diverges; this warns when lr is
    # in AdamW's range (1e-4 and above). See docs/user-guide/optimizers.md for details.
    if lr >= 1e-4:
        logger.warning(
            "Lion lr=%.2e is near/above AdamW's range; the paper recommends lr ~= AdamW lr / 3 "
            "(e.g. AdamW 1e-4 -> Lion ~3e-5). Training will continue but may diverge, see "
            "docs/user-guide/optimizers.md",
            lr,
        )
    param_list = params if _is_param_groups(params) else list(params)
    optimizer = Lion(param_list, lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)
    print(f"Creating Lion optimizer (lr={lr}, betas={betas}, weight_decay={weight_decay})")
    print("  [OK] Lion optimizer created")
    return optimizer


class CAME(Optimizer):
    """Confidence-guided Adaptive Memory Efficient optimizer (Luo et al., 2023).

    Paper: "CAME: Confidence-guided Adaptive Memory Efficient Optimization"
        https://arxiv.org/abs/2307.02047  (ACL 2023 Outstanding Paper)

    Adafactor-style row/column-factored second moment to save VRAM +
    confidence guidance: uses the EMA of the squared residual
    (instability) between the update and its momentum exp_avg as an
    approximate per-element confidence, and applies inverse-variance
    weighting to the momentum update to suppress the update noise
    introduced by the factorization approximation.

    Derived from the official implementation yangluo7/CAME (MIT License,
    Copyright (c) 2023 Yang Luo), with engineering adjustments unified
    across this repo (the algorithm formulas line up with the official
    step, unchanged):
    - optimizer state is fixed at fp32 (numerically stable for bf16
      LoRA/LoKr training, same as SOAP)
    - bf16 parameter write-back uses stochastic rounding (the
      _copy_stochastic bit-trick only applies to bf16; fp16 write-back is
      a plain cast, but compute and state stay fp32 throughout)
    - fp32 state is restored after load_state_dict (PyTorch load casts to the param dtype)

    Args:
        params: parameters or a list of param groups
        lr: learning rate (a real lr, AdamW-scale; not filled with 1.0 like Prodigy)
        eps: (eps1, eps2) -- for the second-moment regularizer and the instability regularizer, respectively
        clip_threshold: RMS clipping threshold for the update
        betas: (beta1, beta2, beta3) -- decay for momentum / second moment / instability EMA
        weight_decay: decoupled weight decay (L2)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # beta=1.0 would leave the corresponding EMA stuck at its initial
        # zero forever, and _approx_sq_grad computing 0/0=NaN poisons the
        # params on the very first step, so, same as Lion, this uses an open upper bound
        if len(betas) != 3 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Invalid betas: {betas} (CAME needs 0 <= β1, β2, β3 < 1)")
        # eps=0 would likewise produce 0/0=NaN whenever the whole gradient is
        # zero (e.g. the first step of a zero-initialized LoRA)
        if len(eps) != 2 or not all(e > 0.0 for e in eps):
            raise ValueError(f"Invalid eps: {eps} (CAME needs positive (eps1, eps2))")
        if clip_threshold <= 0.0:
            raise ValueError(f"Invalid clip_threshold: {clip_threshold}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            eps=tuple(eps),
            clip_threshold=clip_threshold,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        # PyTorch load casts float state to the param dtype; CAME state is
        # fixed at fp32 (under bf16, the EMA increment of (1-beta2)=1e-3 would
        # get swallowed by bf16 rounding), restored uniformly here.
        # No key whitelist needed: every tensor state CAME creates is fp32
        # ("step" is an int), so restoring every floating-point tensor is
        # enough, and a future new state key won't be missed.
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point() \
                        and value.dtype != torch.float32:
                    state[key] = value.float()
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            eps1, eps2 = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients")
                if grad.dtype != torch.float32:
                    grad = grad.float()

                state = self.state[p]
                factored = grad.dim() >= 2

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad.shape[:-1], device=grad.device, dtype=torch.float32)
                        state["exp_avg_sq_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device, dtype=torch.float32)
                        state["exp_avg_res_row"] = torch.zeros(grad.shape[:-1], device=grad.device, dtype=torch.float32)
                        state["exp_avg_res_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device, dtype=torch.float32)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1

                update = (grad ** 2) + eps1
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

                # Confidence-guided strategy: instability is the EMA of
                # (update - exp_avg)^2; after row/column factorization, its
                # rsqrt is used as an inverse-variance weight multiplied into the momentum
                if factored:
                    res = (update - exp_avg) ** 2 + eps2
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]
                    exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
                    exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                # .float() already returns a new copy for a non-fp32 tensor, no need to clone first
                p_data_fp32 = p if p.dtype == torch.float32 else p.detach().float()

                if weight_decay != 0.0:
                    p_data_fp32.add_(p_data_fp32, alpha=-weight_decay * lr)

                update.mul_(lr)
                p_data_fp32.add_(-update)

                if p.dtype != torch.float32:
                    _copy_stochastic(p, p_data_fp32)

        return loss


def create_came(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999, 0.9999),
    weight_decay: float = 0.0,
    eps: tuple = (1e-30, 1e-16),
    clip_threshold: float = 1.0,
    **kwargs,
) -> Optimizer:
    """Create the CAME optimizer (Luo et al. 2023, arxiv 2307.02047).

    lr uses a real AdamW-scale value (LoRA commonly uses ~1e-4), not filled
    with 1.0 like Prodigy. A regular lr_scheduler can be configured.

    Args:
        betas: (beta1, beta2, beta3) -- decay for momentum / factored second
            moment / instability EMA. When the upper create_optimizer passes
            its Adam default (0.9, 0.999), the paper's default beta3 is
            appended; any other 2-tuple is treated as a config error and left
            for CAME.__init__ to raise Invalid betas.
        eps: (eps1, eps2) -- second-moment regularizer / instability
            regularizer. When the upper create_optimizer passes its Adam
            default scalar 1e-8, it falls back to the paper's default
            (1e-30, 1e-16); an explicit other scalar raises an error (a
            scalar eps is meaningless for CAME, and silently substituting it
            would swallow the user's config).
        clip_threshold: RMS clipping threshold for the update (same as Adafactor's d parameter).
    """
    betas = tuple(betas)
    if betas == (0.9, 0.999):
        # create_optimizer's generic Adam default; CAME needs (beta1, beta2,
        # beta3). Same default-sentinel detection as PPSF: only the default
        # value is mapped, an explicit 2-tuple is rejected by CAME.__init__
        betas = (0.9, 0.999, 0.9999)
    if isinstance(eps, (int, float)):
        if eps == 1e-8:
            # create_optimizer's generic Adam default scalar; fall back to the paper's default
            eps = (1e-30, 1e-16)
        else:
            raise ValueError(
                f"CAME needs eps as an (eps1, eps2) pair, got scalar {eps}"
            )

    param_list = params if _is_param_groups(params) else list(params)
    optimizer = CAME(
        param_list,
        lr=lr,
        betas=betas,
        eps=tuple(eps),
        clip_threshold=clip_threshold,
        weight_decay=weight_decay,
        **kwargs,
    )
    logger.info(
        f"Creating CAME optimizer (lr={lr}, betas={betas}, eps={tuple(eps)}, "
        f"clip_threshold={clip_threshold}, weight_decay={weight_decay})"
    )
    return optimizer


def create_prodigy(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    d_coef: float = 1.0,
    safeguard_warmup: bool = True,
    use_bias_correction: bool = True,
    **kwargs,
) -> Optimizer:
    """
    Create the Prodigy optimizer (https://github.com/konstmish/prodigy)

    Prodigy adaptively estimates the learning rate; lr should be set to 1.0
    (the lr here is a scaling coefficient on d). If the passed lr != 1.0, it
    is forcibly overridden to 1.0 with a printed warning.

    Recommended defaults: safeguard_warmup=True, use_bias_correction=True, d_coef=1.0.
    A constant or cosine scheduler is enough; stacking restarts is not recommended.

    Args:
        params: model parameters
        lr: learning rate (Prodigy requires 1.0)
        betas: Adam beta parameters
        weight_decay: weight decay
        eps: epsilon
        d_coef: initial scaling coefficient for d
        safeguard_warmup: protects d from growing too fast during warmup
        use_bias_correction: whether to use bias correction

    Returns:
        Prodigy: the Prodigy optimizer
    """
    try:
        from prodigyopt import Prodigy
    except ImportError as e:
        raise ImportError(
            "prodigyopt is required for Prodigy optimizer. "
            "Install with: pip install prodigyopt"
        ) from e

    if abs(lr - 1.0) > 1e-9:
        print(
            f"[WARN] Prodigy requires lr=1.0 (received {lr}); forcing lr=1.0. "
            f"Tune d_coef/weight_decay instead of lr."
        )
        lr = 1.0

    print(
        f"Creating Prodigy optimizer (lr={lr}, weight_decay={weight_decay}, "
        f"d_coef={d_coef}, safeguard_warmup={safeguard_warmup})"
    )

    param_list = list(params)

    optimizer = Prodigy(
        param_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        d_coef=d_coef,
        safeguard_warmup=safeguard_warmup,
        use_bias_correction=use_bias_correction,
        **kwargs,
    )

    print("  [OK] Prodigy optimizer created")

    return optimizer


def create_prodigy_plus_schedulefree(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.0,
    eps: Optional[float] = None,
    d_coef: float = 1.0,
    prodigy_steps: int = 0,
    split_groups: bool = True,
    split_groups_mean: bool = False,
    use_speed: bool = False,
    fused_back_pass: bool = False,
    use_stableadamw: bool = True,
    **kwargs,
) -> Optimizer:
    """
    Create the ProdigyPlusScheduleFree optimizer
    (https://github.com/LoganBooker/prodigy-plus-schedule-free)

    A combination of Prodigy + Schedule-Free. Core problems it solves
    relative to plain Prodigy:
    - **Schedule-Free's averaged weights**: maintains a training weight y
      and an averaged weight x; sample/save use x to generate -> the style
      mutation-episode phenomenon largely disappears.
      *Usage requirement*: optimizer.eval() must be called before
      sample/eval/save, and optimizer.train() afterward. Wrapping it with
      the optimizer_eval_mode(optimizer) context manager is safest.
    - **prodigy_steps freezes d**: stops updating d after a certain step, avoiding late-run jumps.
    - **split_groups fine-grained estimation**: estimates d separately per param group.

    *Learning rate*: must be fixed at 1.0 (same as plain Prodigy).
    Schedule-Free needs no scheduler; the caller should force
    lr_scheduler=none and validate it at startup.

    *betas default*: PPSF's upstream default is (0.9, 0.99), not PyTorch
    AdamW's (0.9, 0.999). This factory auto-overrides to the PPSF-recommended
    value when the passed betas is the PyTorch default; an explicit value from the caller is respected.

    Args:
        params: model parameters
        lr: learning rate (PPSF requires 1.0)
        betas: Adam beta parameters (PPSF recommends (0.9, 0.99))
        weight_decay: weight decay
        eps: epsilon
        d_coef: initial scaling coefficient for d (0.5 recommended for small datasets)
        prodigy_steps: freeze d after step N (0 = never freeze, keeps updating throughout training)
        split_groups: estimate d separately per param group
        split_groups_mean: whether to average d across groups when split_groups=True
            (PPSF default is False; SimpleTuner changed it to True, but for a
             reason specific to transformer-only training -- doesn't fit our
             LoRA + LoKr multi-param-group setup, kept False here)
        use_speed: enable speed mode (experimental)
        fused_back_pass: integrate with PyTorch's fused-backward path (enable when VRAM is tight)
        use_stableadamw: use the stable AdamW normalization strategy

    Returns:
        ProdigyPlusScheduleFree: the optimizer instance
    """
    try:
        # pip package name `prodigy-plus-schedule-free`, import name `prodigyplus`
        from prodigyplus import ProdigyPlusScheduleFree
    except ImportError as e:
        raise ImportError(
            "prodigy-plus-schedule-free is required for ProdigyPlusScheduleFree "
            "optimizer. Install with: pip install 'prodigy-plus-schedule-free>=2.0.0'"
        ) from e

    if abs(lr - 1.0) > 1e-9:
        logger.warning(
            f"[ProdigyPlus] Forcing lr=1.0 (got {lr}); "
            f"Prodigy adapts step size internally via d."
        )
    lr = 1.0

    if isinstance(eps, (int, float)) and eps <= 0:
        logger.warning(f"[ProdigyPlus] eps={eps} non-positive, falling back to None (Adam-atan2).")
        eps = None

    # The upper create_optimizer defaults to betas=(0.9, 0.999) (suited to
    # AdamW), but PPSF recommends (0.9, 0.99). If the caller didn't change
    # the default, override to PPSF's recommended value.
    if tuple(betas) == (0.9, 0.999):
        betas = (0.9, 0.99)

    candidate = dict(
        lr=lr,
        betas=tuple(betas),
        eps=eps,
        weight_decay=weight_decay,
        d_coef=d_coef,
        prodigy_steps=prodigy_steps,
        split_groups=split_groups,
        split_groups_mean=split_groups_mean,
        use_speed=use_speed,
        fused_back_pass=fused_back_pass,
        use_stableadamw=use_stableadamw,
        **kwargs,
    )
    safe_kwargs = _filter_kwargs_by_signature(ProdigyPlusScheduleFree, candidate)

    param_list = params if _is_param_groups(params) else list(params)

    logger.info(
        f"Creating ProdigyPlusScheduleFree "
        f"(d_coef={d_coef}, betas={tuple(betas)}, wd={weight_decay}, "
        f"eps={eps}, stableadamw={use_stableadamw})"
    )
    logger.info(f"[ProdigyPlus] Effective kwargs: {list(safe_kwargs.keys())}")

    optimizer = ProdigyPlusScheduleFree(param_list, **safe_kwargs)

    total = sum(p.numel() for g in optimizer.param_groups for p in g["params"] if p.requires_grad)
    logger.info(f"[ProdigyPlus] Trainable params: {total:,}")
    return optimizer


def create_soap(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.95, 0.95),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    shampoo_beta: float = -1.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precond_in_state: bool = True,
    **kwargs,
) -> Optimizer:
    """Create the SOAP optimizer (Vyas et al. 2024, arxiv 2409.11321).

    SOAP = Adam running in Shampoo's eigenbasis: rotates the gradient into
    the eigenbasis of the gradient covariance, runs standard Adam there, then
    rotates back. Compared to plain Adam, fits matrix-shaped parameters
    (LoRA/LoKr's low-rank factors) faster; compared to Shampoo, the
    preconditioner refresh is cheaper (precondition_frequency).

    See the implementation in utils/soap_optimizer.py (MIT, Copyright (c) 2024 Nikhil Vyas).

    Args:
        betas: (Adam beta1, beta2), the SOAP paper's default is (0.95, 0.95).
        shampoo_beta: Shampoo covariance EMA decay; reuses beta2 when < 0.
        precondition_frequency: refreshes the eigenbasis every N steps (larger = cheaper compute, staler).
        max_precond_dim: a per-axis threshold -- an axis builds a full-rank
            preconditioner only if its dimension is <= this value; above it,
            that axis degrades to Adam. Setting it large (e.g. 10000) lets
            large feature dims also get second-order treatment = the source
            of the speedup; setting it small = SOAP-lite.
        precond_in_state: when False, strips the recomputable Shampoo
            matrices (GG/Q) out of state_dict, making the ckpt smaller and
            requiring a cold rebuild on resume (zero cost when training from scratch without resume).
    """
    from utils.soap_optimizer import SOAP

    param_list = params if _is_param_groups(params) else list(params)
    optimizer = SOAP(
        param_list,
        lr=lr,
        betas=tuple(betas),
        shampoo_beta=shampoo_beta,
        eps=eps,
        weight_decay=weight_decay,
        precondition_frequency=precondition_frequency,
        max_precond_dim=max_precond_dim,
        precond_in_state=precond_in_state,
        **kwargs,
    )
    logger.info(
        f"Creating SOAP optimizer (lr={lr}, betas={tuple(betas)}, wd={weight_decay}, "
        f"precond_freq={precondition_frequency}, max_precond_dim={max_precond_dim})"
    )
    return optimizer


def create_soap_sf(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.95),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    shampoo_beta: float = -1.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precond_in_state: bool = True,
    weight_lr_power: float = 2.0,
    r: float = 0.0,
    warmup_steps: int = 0,
    **kwargs,
) -> Optimizer:
    """Create Schedule-Free SOAP (SOAP preconditioning + Schedule-Free trajectory).

    Wraps SOAP's Adam-in-Shampoo-eigenbasis update with the Schedule-Free
    mechanism (Defazio et al. 2024, "The Road Less Scheduled", arxiv
    2405.15682): drops the first-moment momentum buffer, replacing the LR
    schedule with an interpolation between a base sequence z and the
    Polyak-Ruppert average x. This means it **needs no lr_scheduler** (the
    caller should force lr_scheduler=none and validate at startup), and, like
    other schedule-free optimizers, exposes train()/eval(): eval() switches
    to the averaged weight x before sample/save, and train() switches back
    to y afterward (the trainer's optimizer_eval_mode already handles this uniformly).

    See the implementation in utils/soap_optimizer.py, `SOAPScheduleFree`.

    Args:
        betas: (SF interpolation weight beta1, second-moment decay beta2).
            Under SF, beta1 is not momentum but the z<->x interpolation coefficient.
        weight_lr_power / r / warmup_steps: Schedule-Free-specific -- the
            power of lr / the power of the step index (0=uniform averaging) /
            the number of linear lr warmup steps within the Polyak weight.
        Everything else is the same as create_soap.

    Note: SF's Polyak averaging lags badly on very short training runs
    (<= ~100 steps) (x ~= the trajectory centroid = underfit); use plain
    ``soap`` rather than ``soap_sf`` for that regime; SF is fine for
    thousand-step-scale training.
    """
    from utils.soap_optimizer import SOAPScheduleFree

    param_list = params if _is_param_groups(params) else list(params)
    optimizer = SOAPScheduleFree(
        param_list,
        lr=lr,
        betas=tuple(betas),
        shampoo_beta=shampoo_beta,
        eps=eps,
        weight_decay=weight_decay,
        precondition_frequency=precondition_frequency,
        max_precond_dim=max_precond_dim,
        precond_in_state=precond_in_state,
        weight_lr_power=weight_lr_power,
        r=r,
        warmup_steps=warmup_steps,
        **kwargs,
    )
    logger.info(
        f"Creating Schedule-Free SOAP optimizer (lr={lr}, betas={tuple(betas)}, "
        f"wd={weight_decay}, precond_freq={precondition_frequency}, "
        f"max_precond_dim={max_precond_dim}, weight_lr_power={weight_lr_power}, r={r})"
    )
    return optimizer


@contextmanager
def optimizer_eval_mode(optimizer: Optimizer):
    """Context manager that switches a Schedule-Free-family optimizer (PPSF
    etc.) to eval mode.

    Schedule-Free optimizers internally keep two sets of weights: the
    training weight y and the averaged weight x. sample / validation / save
    should use x (averaged), while training steps use y. PPSF's
    optimizer.eval() / optimizer.train() switch which set the parameter
    tensors point to via an in-place p.lerp_(); forgetting to switch back
    with train() leaves training continuing on the averaged weights, corrupting results.

    Usage:
        with optimizer_eval_mode(optimizer):
            model.eval()
            img = sample_image(...)
            model.train()

    Non-Schedule-Free optimizers (AdamW / Prodigy etc.) have no .train/.eval
    method -- this context manager is a silent no-op for them, so the caller
    doesn't need to branch on optimizer_type.
    """
    has_eval = hasattr(optimizer, "eval") and callable(getattr(optimizer, "eval"))
    has_train = hasattr(optimizer, "train") and callable(getattr(optimizer, "train"))
    if has_eval and has_train:
        optimizer.eval()
        try:
            yield
        finally:
            optimizer.train()
    else:
        yield


def create_optimizer_grouped_parameters(
    model: nn.Module,
    weight_decay: float,
    no_decay_modules: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Create grouped optimizer parameters

    Certain parameters (such as biases and LayerNorm weights) usually
    shouldn't have weight decay applied. This function splits parameters
    into two groups:
    1. Parameters that need weight decay (weight matrices)
    2. Parameters that don't need weight decay (biases, LayerNorm)

    This is standard best practice for Transformer training.

    Args:
        model: the model
        weight_decay: weight decay coefficient
        no_decay_modules: list of module names to exclude from weight decay

    Returns:
        List[Dict]: the grouped parameter list
    """
    if no_decay_modules is None:
        # Default: biases and LayerNorm parameters don't get weight decay
        no_decay_modules = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"]
    
    # Group parameters
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check whether weight decay is needed
        needs_decay = True
        for no_decay_pattern in no_decay_modules:
            if no_decay_pattern in name:
                needs_decay = False
                break
        
        if needs_decay:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    # Build the parameter groups
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    
    # Print stats
    num_decay_params = sum(p.numel() for p in decay_params)
    num_no_decay_params = sum(p.numel() for p in no_decay_params)
    print(f"Parameter groups:")
    print(f"  With weight decay: {len(decay_params)} params, {num_decay_params:,} elements")
    print(f"  Without weight decay: {len(no_decay_params)} params, {num_no_decay_params:,} elements")
    
    return optimizer_grouped_parameters


def get_optimizer_info(optimizer: Optimizer) -> Dict[str, Any]:
    """
    Get optimizer info

    Used for logging and debugging

    Args:
        optimizer: the optimizer instance

    Returns:
        Dict: a dict of optimizer info
    """
    info = {
        "type": type(optimizer).__name__,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "num_param_groups": len(optimizer.param_groups),
    }
    
    # Get the total parameter count
    total_params = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            total_params += p.numel()
    
    info["total_trainable_params"] = total_params

    # Add optimizer-specific info (duck typing: AdamW / Prodigy / PPSF / bnb AdamW8bit all have these fields)
    pg0 = optimizer.param_groups[0] if optimizer.param_groups else {}
    if "betas" in pg0:
        info["betas"] = pg0["betas"]
    if "weight_decay" in pg0:
        info["weight_decay"] = pg0["weight_decay"]
    if "eps" in pg0:
        info["eps"] = pg0["eps"]
    # PPSF's internal d estimate -- useful for debugging
    if "d" in pg0:
        info["d"] = pg0["d"]
    if hasattr(optimizer, "get_avg_learning_rate") and callable(getattr(optimizer, "get_avg_learning_rate")):
        info["learning_rate"] = optimizer.get_avg_learning_rate()

    return info


def get_optimizer_monitor_metrics(optimizer: Optimizer) -> Dict[str, float]:
    """Return LR metrics suitable for train_monitor / logs.

    AdamW-style optimizers expose a real `lr` in the param group. Prodigy-style
    optimizers keep the UI-facing base lr at 1.0 and adapt the step size through
    `d`; PPSF v2 additionally exposes `effective_lr` for logging. This helper
    normalizes those shapes into one monitor point while preserving the raw
    ingredients for debugging.
    """
    if hasattr(optimizer, "get_avg_learning_rate") and callable(getattr(optimizer, "get_avg_learning_rate")):
        avg_lr = _as_float(optimizer.get_avg_learning_rate())
        if avg_lr is not None:
            return {"lr": avg_lr, "actual_lr": avg_lr}

    groups = list(getattr(optimizer, "param_groups", []) or [])
    if not groups:
        return {"lr": 0.0}

    base_lrs: list[float] = []
    d_values: list[float] = []
    effective_lrs: list[float] = []
    actual_lrs: list[float] = []

    for group in groups:
        base_lr = _as_float(group.get("lr"))
        if base_lr is not None:
            base_lrs.append(base_lr)

        d_source = group.get("d")
        if (
            group.get("split_groups")
            and group.get("split_groups_mean")
            and group.get("shared_d") is not None
        ):
            d_source = group.get("shared_d")
        d_value = _as_float(d_source)
        if d_value is None:
            continue
        d_values.append(d_value)

        # PPSF v2 recommends logging d * effective_lr. Older Prodigy/PPSF
        # versions do not expose it, so fall back to the base group lr.
        effective_lr = _as_float(group.get("effective_lr"))
        if effective_lr is None:
            effective_lr = base_lr
        if effective_lr is None:
            continue
        effective_lrs.append(effective_lr)
        actual_lrs.append(d_value * effective_lr)

    if not actual_lrs:
        return {"lr": base_lrs[0] if base_lrs else 0.0}

    metrics: Dict[str, float] = {
        "lr": _mean(actual_lrs),
        "actual_lr": _mean(actual_lrs),
        "base_lr": _mean(base_lrs),
        "d": _mean(d_values),
    }
    if effective_lrs:
        metrics["effective_lr"] = _mean(effective_lrs)
    if len(d_values) > 1:
        metrics["d_min"] = min(d_values)
        metrics["d_max"] = max(d_values)
    if len(actual_lrs) > 1:
        metrics["actual_lr_min"] = min(actual_lrs)
        metrics["actual_lr_max"] = max(actual_lrs)
    return metrics
