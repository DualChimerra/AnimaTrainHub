from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from training.timestep_sampling import sample_t, sample_t_style_friendly
from training.timestep_samplers import build_timestep_sampler


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_maps_log_snr_to_flow_matching_t():
    for mean in (-6.0, -3.0, 0.0, 2.0):
        t = sample_t_style_friendly(64, "cpu", mean=mean, sigma=1e-4)
        expected = _sigmoid(-mean / 2)
        assert abs(float(t.mean()) - expected) < 1e-3, mean


def test_paper_default_sits_in_the_style_window():
    torch.manual_seed(0)
    t = sample_t_style_friendly(20000, "cpu", mean=-6.0, sigma=2.0)
    median = float(t.median())
    assert 0.93 <= median <= 0.97, median
    assert float((t > 0.75).float().mean()) > 0.85


def test_mean_is_monotone_in_noise_level():
    torch.manual_seed(0)
    medians = [
        float(sample_t_style_friendly(8000, "cpu", mean=m, sigma=2.0).median())
        for m in (-8.0, -6.0, -4.0, -2.0)
    ]
    assert medians == sorted(medians, reverse=True), medians


def test_sigma_controls_window_width():
    torch.manual_seed(0)
    narrow = sample_t_style_friendly(8000, "cpu", mean=-6.0, sigma=0.5)
    wide = sample_t_style_friendly(8000, "cpu", mean=-6.0, sigma=3.0)
    assert float(narrow.std()) < float(wide.std())


def test_stays_in_open_interval():
    torch.manual_seed(0)
    t = sample_t_style_friendly(5000, "cpu", mean=-11.0, sigma=6.0)
    assert float(t.min()) > 0.0 and float(t.max()) < 1.0


def test_sample_t_dispatches_and_ignores_timestep_shift():
    outs = []
    for shift in (0.5, 3.0, 9.0):
        torch.manual_seed(42)
        outs.append(sample_t(
            256, "cpu", mode="style_friendly", shift=shift,
            style_snr_mean=-6.0, style_snr_sigma=2.0,
        ))
    torch.testing.assert_close(outs[0], outs[1])
    torch.testing.assert_close(outs[1], outs[2])


def test_schedule_shift_still_composes():
    torch.manual_seed(7)
    plain = sample_t(512, "cpu", mode="style_friendly", timestep_schedule_shift=1.0)
    torch.manual_seed(7)
    shifted = sample_t(512, "cpu", mode="style_friendly", timestep_schedule_shift=2.0)
    assert float(shifted.mean()) > float(plain.mean())


def test_registry_builds_baseline_with_style_params():
    args = SimpleNamespace(
        timestep_sampling="style_friendly",
        timestep_shift=3.0,
        timestep_mix_low_prob=0.0,
        timestep_schedule_shift=1.0,
        style_snr_mean=-4.5,
        style_snr_sigma=2.5,
        infonoise_enabled=False,
    )
    sampler = build_timestep_sampler(args, total_steps=100)
    status = sampler.status()
    assert status["kind"] == "baseline"
    assert status["mode"] == "style_friendly"
    assert status["style_snr_mean"] == -4.5
    assert status["style_snr_sigma"] == 2.5
    t = sampler.sample(64, "cpu")
    assert t.shape == (64,)


def test_zero_mean_is_not_swallowed_by_falsy_default():
    args = SimpleNamespace(
        timestep_sampling="style_friendly",
        style_snr_mean=0.0,
        style_snr_sigma=1e-4,
        infonoise_enabled=False,
    )
    sampler = build_timestep_sampler(args, total_steps=10)
    assert sampler.style_snr_mean == 0.0
    assert abs(float(sampler.sample(256, "cpu").mean()) - 0.5) < 1e-3


def test_schema_exposes_mode_and_window_fields():
    from studio.schema import TrainingConfig

    literal = set(TrainingConfig.model_fields["timestep_sampling"].annotation.__args__)
    assert "style_friendly" in literal

    cfg = TrainingConfig(timestep_sampling="style_friendly")
    assert cfg.style_snr_mean == -6.0
    assert cfg.style_snr_sigma == 2.0

    extra = TrainingConfig.model_fields["timestep_shift"].json_schema_extra
    assert "timestep_sampling!=style_friendly" in extra["show_when"]
    for name in ("style_snr_mean", "style_snr_sigma"):
        meta = TrainingConfig.model_fields[name].json_schema_extra
        assert meta["show_when"] == "timestep_sampling==style_friendly", name


def test_is_a_strict_generalisation_of_logit_normal_shift():
    for shift in (2.0, 2.5, 3.0, 4.0):
        torch.manual_seed(0)
        legacy = sample_t(100000, "cpu", mode="logit_normal", shift=shift)
        torch.manual_seed(1)
        equiv = sample_t_style_friendly(
            100000, "cpu", mean=-2.0 * math.log(shift), sigma=2.0,
        )
        assert abs(float(legacy.median()) - float(equiv.median())) < 5e-3, shift
        assert abs(float(legacy.mean()) - float(equiv.mean())) < 5e-3, shift
        assert abs(float(legacy.std()) - float(equiv.std())) < 5e-3, shift
