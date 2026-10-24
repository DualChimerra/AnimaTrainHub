"""optimizer_args: parsing, config validation and delivery to every optimizer builder."""
from __future__ import annotations

import types

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from studio.domain.optimizer_args import parse_optimizer_args
from studio.schema import TrainingConfig


def test_parses_literals_and_bare_words() -> None:
    assert parse_optimizer_args([
        "betas=(0.9, 0.99)", " eps = 1e-6 ", "nesterov=false", "foreach=True",
        "adjust_lr_fn=original", "", "d0=None",
    ]) == {
        "betas": (0.9, 0.99), "eps": 1e-6, "nesterov": False, "foreach": True,
        "adjust_lr_fn": "original", "d0": None,
    }


@pytest.mark.parametrize("line", ["betas", "=1", "1x=2", "lr=1e-4", "weight_decay=0.1", "eps="])
def test_rejects_bad_lines(line) -> None:
    with pytest.raises(ValueError, match="optimizer_args"):
        parse_optimizer_args([line])


def test_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="more than once"):
        parse_optimizer_args(["eps=1e-6", "eps=1e-7"])


def test_config_validates_and_strips_lines() -> None:
    assert TrainingConfig(optimizer_args=[" eps=1e-6 ", ""]).optimizer_args == ["eps=1e-6"]
    with pytest.raises(ValidationError, match="optimizer_args"):
        TrainingConfig(optimizer_args=["learning_rate=1"])


def test_field_is_last_group_of_first_tab() -> None:
    from studio.domain.common import GROUP_ORDER

    keys = [key for key, _, _ in GROUP_ORDER]
    assert keys.index("optimizer_args") == keys.index("lora") + 1
    extra = TrainingConfig.model_fields["optimizer_args"].json_schema_extra
    assert extra["group"] == "optimizer_args" and extra["control"] == "string-list"


def _args(optimizer_type, *lines, **fields):
    return types.SimpleNamespace(optimizer_type=optimizer_type, optimizer_args=list(lines), **fields)


@pytest.mark.parametrize(("optimizer_type", "line", "key", "expected"), [
    ("adamw", "betas=(0.8, 0.9)", "betas", (0.8, 0.9)),
    ("lion", "betas=(0.8, 0.9)", "betas", (0.8, 0.9)),
    ("came", "clip_threshold=2.0", "clip_threshold", 2.0),
    ("muon", "nesterov=False", "nesterov", False),
    ("muon", "momentum=0.8", "momentum", 0.8),
    ("simplified_ademamix", "min_beta1=0.5", "min_beta1", 0.5),
    ("soap", "precondition_frequency=3", "precondition_frequency", 3),
])
def test_builders_apply_optimizer_args(optimizer_type, line, key, expected) -> None:
    from training.optimizers import build_optimizer

    w = nn.Parameter(torch.randn(4, 8))
    opt = build_optimizer(_args(optimizer_type, line), [w], 1e-3, 0.0)
    assert opt.param_groups[0][key] == expected


def test_automagic_applies_optimizer_args() -> None:
    from training.optimizers import build_optimizer

    w = nn.Parameter(torch.randn(4, 8))
    assert build_optimizer(_args("automagic", "lr_bump=2e-6"), [w], 1e-6, 0.0).lr_bump == 2e-6


def test_automagic_v2_applies_optimizer_args() -> None:
    from training.optimizers import build_optimizer

    w = nn.Parameter(torch.randn(4, 8))
    opt = build_optimizer(_args("automagic", "agreement_threshold=0.25", automagic_variant="v2"), [w], 1e-6, 0.0)
    assert opt.param_groups[0]["agreement_threshold"] == 0.25


def test_riemannion_applies_optimizer_args() -> None:
    from training.optimizers import build_optimizer
    from utils.riemann_lora_adapter import RiemannLoRALinear

    layer = RiemannLoRALinear("lora_unet_x", nn.Linear(6, 5, bias=False), rank=2,
                              generator=torch.Generator().manual_seed(0))
    args = _args("muon", "momentum=0.5", "adam_eps=1e-6", use_riemannian=True, lora_type="lora")
    opt = build_optimizer(args, [layer.lora_b, layer.lora_z1], 1e-2, 0.0)
    assert opt.param_groups[0]["momentum"] == 0.5 and opt.param_groups[0]["adam_eps"] == 1e-6


def test_unknown_name_fails_at_startup() -> None:
    from training.optimizers import build_optimizer

    w = nn.Parameter(torch.randn(4, 8))
    with pytest.raises(TypeError, match="no_such_arg"):
        build_optimizer(_args("adamw", "no_such_arg=1"), [w], 1e-3, 0.0)
