"""Portable tests: run after installing this directory as a package."""

from dataclasses import replace
import math

import pytest
import torch

from dual_peak_sampling import DualPeakConfig, density, sample_timesteps


def test_default_distribution_matches_agreed_curve():
    t = sample_timesteps(200_000, generator=torch.Generator().manual_seed(314))
    for lo, hi, expected in (
        (0, .3, .02093), (.3, .6, .21464),
        (.6, .9, .58642), (.9, 1, .17800),
        (.45, .6, .16832), (.8, .9, .28897),
    ):
        measured = float(((t >= lo) & (t < hi)).float().mean())
        assert abs(measured - expected) < .005


def test_density_normalization_and_actual_mixture_peaks():
    t = torch.linspace(.00001, .99999, 100_000, dtype=torch.float64)
    p = density(t)
    assert abs(float(torch.trapezoid(p, t)) - 1.0) < 1e-4
    peaks = t[1:-1][(p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])]
    assert peaks.numel() == 2
    assert abs(float(peaks[0]) - .540) < .003
    assert abs(float(peaks[1]) - .870) < .003


@pytest.mark.parametrize("position,width", [(.2, .55), (.525, .55), (.85, 1.2), (.7, 2.8)])
def test_individual_peak_location_is_mode_not_median(position, width):
    config = DualPeakConfig(peak1_position=position, peak1_width=width,
                            peak1_weight=1, peak2_weight=0,
                            background_weight=0, uniform_weight=0)
    t = torch.linspace(.0001, .9999, 20000, dtype=torch.float64)
    assert abs(float(t[density(t, config).argmax()]) - position) < .0001


def test_independent_second_peak_and_width_controls():
    base = DualPeakConfig(peak1_weight=0, peak2_weight=1,
                          background_weight=0, uniform_weight=0)
    for position, width in ((.65, .5), (.85, 1.2), (.9, 2)):
        config = replace(base, peak2_position=position, peak2_width=width)
        t = torch.linspace(.0001, .9999, 20000, dtype=torch.float64)
        assert abs(float(t[density(t, config).argmax()]) - position) < .0001


def test_zero_weights_and_normalization():
    config = DualPeakConfig(peak1_weight=0, peak2_weight=0,
                            background_weight=0, uniform_weight=.1)
    t = sample_timesteps(30000, config=config, generator=torch.Generator().manual_seed(1))
    assert abs(float(t.mean()) - .5) < .01
    torch.testing.assert_close(density(torch.tensor([.1, .4, .9]), config), torch.ones(3))
    original = DualPeakConfig()
    scaled = replace(original, peak1_weight=.3, peak2_weight=.7,
                     background_weight=.9, uniform_weight=.1)
    torch.testing.assert_close(
        sample_timesteps(256, config=original, generator=torch.Generator().manual_seed(7)),
        sample_timesteps(256, config=scaled, generator=torch.Generator().manual_seed(7)),
        rtol=0, atol=0,
    )


def test_pure_background_matches_logsnr_normal():
    config = DualPeakConfig(peak1_weight=0, peak2_weight=0,
                            background_weight=1, uniform_weight=0)
    t = sample_timesteps(60000, config=config, generator=torch.Generator().manual_seed(8))
    lam = 2 * torch.log((1-t)/t)
    assert abs(float(lam.mean()) - config.background_mean) < .04
    assert abs(float(lam.std()) - config.background_width) < .04


@pytest.mark.parametrize("changes", [
    {"peak1_position": 0}, {"peak2_position": 1},
    {"peak1_width": 0}, {"peak2_width": 3}, {"background_width": -1},
    {"background_width": 7}, {"background_mean": -13},
    {"peak1_weight": -1}, {"uniform_weight": 2},
    {"peak2_position": math.nan}, {"background_mean": math.inf},
    {"uniform_weight": math.nan},
    {"peak1_weight": 0, "peak2_weight": 0, "background_weight": 0, "uniform_weight": 0},
])
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        DualPeakConfig(**changes)


def test_reproducible_with_external_rng_state_restore():
    generator = torch.Generator().manual_seed(22)
    sample_timesteps(13, generator=generator)
    state = generator.get_state()
    expected = sample_timesteps(512, generator=generator)
    generator.set_state(state)
    torch.testing.assert_close(expected, sample_timesteps(512, generator=generator), rtol=0, atol=0)


def test_empty_singleton_and_dtype_under_autocast():
    assert sample_timesteps(0).shape == (0,)
    assert sample_timesteps(1).shape == (1,)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        t = sample_timesteps(1000, config=DualPeakConfig(background_mean=-12, background_width=6))
    assert t.dtype == torch.float32
    assert not t.requires_grad
    assert bool(torch.isfinite(t).all())
    assert bool(((t >= 1e-4) & (t <= 1-1e-4)).all())


def test_float32_even_with_changed_global_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert sample_timesteps(16).dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("batch_size", [-1, 1.5, True])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        sample_timesteps(batch_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = sample_timesteps(1024, "cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    assert t.is_cuda and t.dtype == torch.float32 and t.shape == (1024,)
    assert bool(((t > 0) & (t < 1)).all())
