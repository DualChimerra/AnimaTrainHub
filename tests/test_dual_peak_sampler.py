"""Schema -> CLI/YAML -> registry -> real sample_t integration."""

from dataclasses import asdict, fields
from types import SimpleNamespace

import pytest
import torch
import yaml
from pydantic import ValidationError

from dual_peak_sampling import DualPeakConfig, sample_timesteps
from studio.schema import TrainingConfig
from studio.infrastructure.argparse_bridge import build_parser
from training.timestep_sampling import sample_t, apply_resolution_shift
from training.timestep_samplers import build_timestep_sampler
from training.timestep_samplers.protocol import TimestepSamplerProtocol


def test_schema_defaults_match_portable_package_and_roundtrip():
    cfg = TrainingConfig(timestep_sampling="dual_peak")
    for field in fields(DualPeakConfig):
        key = f"dual_peak_{field.name}"
        assert getattr(cfg, key) == getattr(DualPeakConfig(), field.name)
        assert TrainingConfig.model_fields[key].json_schema_extra["show_when"] == "timestep_sampling==dual_peak"
    restored = TrainingConfig.model_validate(yaml.safe_load(yaml.safe_dump(cfg.model_dump())))
    assert restored == cfg
    assert "timestep_sampling!=dual_peak" in TrainingConfig.model_fields["timestep_shift"].json_schema_extra["show_when"]
    assert TrainingConfig().timestep_sampling == "logit_normal"


def test_cli_to_registry_preserves_all_custom_parameters():
    expected = DualPeakConfig(peak1_position=.48, peak1_width=.7, peak1_weight=.2,
                              peak2_position=.82, peak2_width=1.4, peak2_weight=.4,
                              background_mean=0, background_width=2.4,
                              background_weight=.3, uniform_weight=0)
    argv = ["--timestep-sampling", "dual_peak"]
    for name, value in asdict(expected).items():
        argv.extend(["--dual-peak-" + name.replace("_", "-"), str(value)])
    args = build_parser(TrainingConfig, add_config_arg=False).parse_args(argv)
    sampler = build_timestep_sampler(args, 100)
    assert isinstance(sampler, TimestepSamplerProtocol)
    assert sampler.status()["dual_peak"] == asdict(expected)
    torch.manual_seed(31)
    actual = sampler.sample(1024, "cpu")
    torch.manual_seed(31)
    torch.testing.assert_close(actual, sample_timesteps(1024, config=expected), rtol=0, atol=0)


def test_shift_is_ignored_but_schedule_and_resolution_shifts_compose():
    torch.manual_seed(7)
    plain = sample_t(512, "cpu", mode="dual_peak", shift=.5)
    torch.manual_seed(7)
    other_shift = sample_t(512, "cpu", mode="dual_peak", shift=9)
    torch.testing.assert_close(plain, other_shift, rtol=0, atol=0)
    torch.manual_seed(7)
    shifted = sample_t(512, "cpu", mode="dual_peak", timestep_schedule_shift=2)
    expected = (2 * plain / (1 + plain)).clamp(1e-4, 1-1e-4)
    torch.testing.assert_close(shifted, expected)
    torch.testing.assert_close(apply_resolution_shift(plain, [1024]*512, 1024), plain)


def test_infonoise_warmup_uses_custom_config():
    args = SimpleNamespace(timestep_sampling="dual_peak", infonoise_enabled=True,
                           dual_peak_peak1_weight=0, dual_peak_peak2_weight=0,
                           dual_peak_background_weight=0, dual_peak_uniform_weight=1)
    sampler = build_timestep_sampler(args, 100)
    assert sampler.baseline_dual_peak_config.uniform_weight == 1
    torch.manual_seed(4)
    actual = sampler.sample(512, "cpu")
    torch.manual_seed(4)
    expected = sample_timesteps(512, config=sampler.baseline_dual_peak_config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_stateless_resume_uses_trainer_global_rng():
    args = SimpleNamespace(timestep_sampling="dual_peak")
    first = build_timestep_sampler(args, 100)
    first.sample(5, "cpu")
    rng = torch.get_rng_state()
    state = first.state_dict()
    expected = first.sample(64, "cpu")
    resumed = build_timestep_sampler(args, 100)
    resumed.load_state_dict(state)
    torch.set_rng_state(rng)
    torch.testing.assert_close(resumed.sample(64, "cpu"), expected, rtol=0, atol=0)


def test_overrides_warn_without_changing_user_settings(caplog):
    args = SimpleNamespace(timestep_sampling="dual_peak", leap_enabled=True,
                           timestep_schedule_shift=2, timestep_shift_resolution_aware=True)
    sampler = build_timestep_sampler(args, 100)
    assert "non-Leap" in caplog.text
    assert "move the configured peaks" in caplog.text
    assert sampler.timestep_schedule_shift == 2
    assert args.leap_enabled and args.timestep_shift_resolution_aware


@pytest.mark.parametrize("changes", [
    {"dual_peak_peak1_position": 0}, {"dual_peak_peak2_position": 1},
    {"dual_peak_peak1_width": 0}, {"dual_peak_peak2_width": 3},
    {"dual_peak_background_mean": float("nan")}, {"dual_peak_uniform_weight": -1},
    {"dual_peak_peak1_weight": 0, "dual_peak_peak2_weight": 0,
     "dual_peak_background_weight": 0, "dual_peak_uniform_weight": 0},
])
def test_schema_and_runtime_reject_invalid_custom_config(changes):
    with pytest.raises(ValidationError):
        TrainingConfig(timestep_sampling="dual_peak", **changes)
    with pytest.raises(ValueError):
        build_timestep_sampler(SimpleNamespace(timestep_sampling="dual_peak", **changes), 100)
