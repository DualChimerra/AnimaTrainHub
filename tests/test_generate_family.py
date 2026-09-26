
from __future__ import annotations

import pytest

from studio.domain.generate import GenerateConfig
from studio.domain.reg import RegAiConfig
from studio.services.models.families import get_assets


def test_generate_config_family_sampler_whitelist():
    GenerateConfig(model_family="krea2", sampler_name="euler",
                   scheduler="simple")
    with pytest.raises(ValueError, match="er_sde"):
        GenerateConfig(model_family="krea2", sampler_name="er_sde",
                       scheduler="simple")
    with pytest.raises(ValueError, match="euler"):
        GenerateConfig(model_family="anima", sampler_name="euler")
    with pytest.raises(ValueError, match="sgm_uniform"):
        GenerateConfig(model_family="krea2", sampler_name="euler",
                       scheduler="sgm_uniform")


def test_reg_config_family_sampler_whitelist():
    RegAiConfig(model_family="krea2", sampler_name="euler",
                scheduler="simple")
    with pytest.raises(ValueError, match="sgm_uniform"):
        RegAiConfig(model_family="krea2", sampler_name="euler",
                    scheduler="sgm_uniform")


def test_generate_config_carries_family_and_distilled_to_daemon():
    cfg = GenerateConfig(model_family="krea2", distilled=True,
                         sampler_name="euler", scheduler="simple")
    dumped = cfg.model_dump()
    assert dumped["model_family"] == "krea2"
    assert dumped["distilled"] is True


def test_is_distilled_path_by_official_variant():
    krea2 = get_assets("krea2")
    assert krea2.is_distilled_path("G:/models/diffusion_models/krea2-turbo-bf16.safetensors")
    assert krea2.is_distilled_path("G:/models/diffusion_models/krea2-turbo-fp8-scaled.safetensors")
    assert not krea2.is_distilled_path("G:/models/diffusion_models/krea2-raw-bf16.safetensors")
    assert not krea2.is_distilled_path("G:/models/diffusion_models/krea2-raw-fp8-scaled.safetensors")
    assert not krea2.is_distilled_path("G:/models/my-community-turbo-mix.safetensors")
    assert not krea2.is_distilled_path("")
    assert not get_assets("anima").is_distilled_path(
        "G:/models/diffusion_models/anima-base-v1.0.safetensors")
