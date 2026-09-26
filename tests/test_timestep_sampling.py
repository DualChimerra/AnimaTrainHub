from __future__ import annotations

import torch

from training.timestep_sampling import _apply_timestep_schedule_shift, sample_t
from training.timestep_samplers.baseline import BaselineTimestepSampler
from training.timestep_samplers.baseline import build as build_baseline
from training.timestep_samplers.infonoise import InfoNoiseScheduler
from training.timestep_samplers.infonoise import build as build_infonoise


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_timestep_schedule_shift_identity_when_one():
    t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    out = _apply_timestep_schedule_shift(t, 1.0)
    torch.testing.assert_close(out, t)


def test_timestep_schedule_shift_formula_matches_spec():
    """t' = (t * s) / (1 + (s-1)*t); s=2, t=0.5 → 1/1.5 = 2/3"""
    t = torch.tensor([0.5])
    out = _apply_timestep_schedule_shift(t, 2.0)
    torch.testing.assert_close(out, torch.tensor([2.0 / 3.0]), rtol=1e-5, atol=1e-5)


def test_timestep_schedule_shift_clamps_to_open_interval():
    t = torch.tensor([1e-5, 0.5, 1 - 1e-5])
    out = _apply_timestep_schedule_shift(t, 5.0)
    assert (out > 0).all() and (out < 1).all()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_legacy_logit_normal_shift_pushes_high():
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="logit_normal", shift=3.0)
    assert t.mean().item() > 0.55
    assert (t > 0).all() and (t < 1).all()


def test_legacy_uniform_mean_half():
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="uniform")
    assert abs(t.mean().item() - 0.5) < 0.05


def test_legacy_logit_normal_low_pushes_low():
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="logit_normal_low", shift=3.0)
    assert t.mean().item() < 0.45


def test_legacy_mode_outputs_in_range():
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="mode", shift=3.0)
    assert (t > 0).all() and (t < 1).all()


# ---------------------------------------------------------------------------
# mixed_uniform_low
# ---------------------------------------------------------------------------


def test_mixed_uniform_low_p_zero_is_pure_uniform():
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="mixed_uniform_low", mix_low_prob=0.0)
    assert abs(t.mean().item() - 0.5) < 0.05


def test_mixed_uniform_low_p_one_matches_logit_normal_low_distribution():
    torch.manual_seed(0)
    t_mixed = sample_t(8192, "cpu", mode="mixed_uniform_low", shift=3.0, mix_low_prob=1.0)
    torch.manual_seed(0)
    t_pure = sample_t(8192, "cpu", mode="logit_normal_low", shift=3.0)
    assert t_mixed.mean().item() < 0.45
    assert t_pure.mean().item() < 0.45
    assert abs(t_mixed.mean().item() - t_pure.mean().item()) < 0.03


def test_mixed_uniform_low_partial_interpolates():
    torch.manual_seed(0)
    t_mixed = sample_t(8192, "cpu", mode="mixed_uniform_low", shift=3.0, mix_low_prob=0.3)
    torch.manual_seed(0)
    t_pure_low = sample_t(8192, "cpu", mode="logit_normal_low", shift=3.0)
    low_mean = t_pure_low.mean().item()
    assert low_mean < t_mixed.mean().item() < 0.5


# ---------------------------------------------------------------------------
# mixed_uniform_logit
# ---------------------------------------------------------------------------


def test_mixed_uniform_logit_p_one_matches_logit_normal_distribution():
    torch.manual_seed(0)
    t_mixed = sample_t(8192, "cpu", mode="mixed_uniform_logit", shift=3.0, mix_low_prob=1.0)
    torch.manual_seed(0)
    t_pure = sample_t(8192, "cpu", mode="logit_normal", shift=3.0)
    assert t_mixed.mean().item() > 0.55
    assert abs(t_mixed.mean().item() - t_pure.mean().item()) < 0.03


def test_mixed_uniform_logit_partial_pulls_toward_high():
    torch.manual_seed(0)
    t = sample_t(8192, "cpu", mode="mixed_uniform_logit", shift=3.0, mix_low_prob=0.3)
    assert t.mean().item() > 0.5


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_timestep_schedule_shift_high_value_raises_mean():
    torch.manual_seed(0)
    t_base = sample_t(4096, "cpu", mode="uniform", timestep_schedule_shift=1.0)
    torch.manual_seed(0)
    t_high = sample_t(4096, "cpu", mode="uniform", timestep_schedule_shift=3.0)
    assert t_high.mean().item() > t_base.mean().item() + 0.05


def test_timestep_schedule_shift_applies_to_mixed_mode():
    torch.manual_seed(0)
    t = sample_t(4096, "cpu", mode="mixed_uniform_low", shift=3.0, mix_low_prob=1.0, timestep_schedule_shift=3.0)
    assert t.mean().item() > 0.4


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_baseline_sampler_accepts_new_params():
    s = BaselineTimestepSampler(
        mode="mixed_uniform_low",
        shift=3.0,
        mix_low_prob=0.5,
        timestep_schedule_shift=1.5,
    )
    t = s.sample(64, "cpu")
    assert t.shape == (64,)
    assert (t > 0).all() and (t < 1).all()
    status = s.status()
    assert status["mode"] == "mixed_uniform_low"
    assert status["mix_low_prob"] == 0.5
    assert status["timestep_schedule_shift"] == 1.5


def test_baseline_build_reads_new_args():
    class Args:
        timestep_sampling = "mixed_uniform_low"
        timestep_shift = 2.0
        timestep_mix_low_prob = 0.25
        timestep_schedule_shift = 1.12

    s = build_baseline(Args(), total_steps=1000)
    assert s.mode == "mixed_uniform_low"
    assert s.mix_low_prob == 0.25
    assert s.timestep_schedule_shift == 1.12


def test_baseline_build_backward_compatible_with_missing_new_args():
    class OldArgs:
        timestep_sampling = "logit_normal"
        timestep_shift = 3.0

    s = build_baseline(OldArgs(), total_steps=1000)
    assert s.mix_low_prob == 0.0
    assert s.timestep_schedule_shift == 1.0


def test_baseline_sampler_default_args_unchanged():
    s = BaselineTimestepSampler()
    assert s.mix_low_prob == 0.0
    assert s.timestep_schedule_shift == 1.0
    t = s.sample(128, "cpu")
    assert t.mean().item() > 0.5


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_infonoise_build_reads_new_args():
    class Args:
        timestep_sampling = "mixed_uniform_low"
        timestep_shift = 3.0
        timestep_mix_low_prob = 0.2
        timestep_schedule_shift = 1.5
        infonoise_K = 32
        infonoise_N_warm = 0  # auto
        infonoise_M = 50
        infonoise_B = 128
        infonoise_beta = 0.9
        infonoise_N_min = 10

    s = build_infonoise(Args(), total_steps=500)
    assert s.baseline_mode == "mixed_uniform_low"
    assert s.baseline_mix_low_prob == 0.2
    assert s.baseline_timestep_schedule_shift == 1.5


def test_infonoise_warmup_sample_uses_new_params():
    s = InfoNoiseScheduler(
        K=32, N_warm=100, M=10, B=10, N_min=1,
        baseline_mode="mixed_uniform_low",
        baseline_shift=3.0,
        baseline_mix_low_prob=0.5,
        baseline_timestep_schedule_shift=1.0,
    )
    assert s._cdf_values is None
    t = s.sample(128, "cpu")
    assert t.shape == (128,)
    assert (t > 0).all() and (t < 1).all()


def test_infonoise_default_baseline_params_unchanged():
    s = InfoNoiseScheduler(K=32, N_warm=100, M=10, B=10, N_min=1)
    assert s.baseline_mix_low_prob == 0.0
    assert s.baseline_timestep_schedule_shift == 1.0


def test_infonoise_cdf_path_ignores_baseline_timestep_schedule_shift():
    import numpy as np

    s1 = InfoNoiseScheduler(K=16, N_warm=100, M=5, B=10, N_min=1,
                            baseline_timestep_schedule_shift=1.0)
    s2 = InfoNoiseScheduler(K=16, N_warm=100, M=5, B=10, N_min=1,
                            baseline_timestep_schedule_shift=5.0)
    cdf = np.linspace(0.0, 1.0, 17)
    s1._cdf_values = cdf
    s2._cdf_values = cdf
    torch.manual_seed(42)
    t1 = s1.sample(2048, "cpu")
    torch.manual_seed(42)
    t2 = s2.sample(2048, "cpu")
    torch.testing.assert_close(t1, t2, rtol=0, atol=0)


def test_infonoise_warmup_path_respects_baseline_timestep_schedule_shift():
    s1 = InfoNoiseScheduler(K=16, N_warm=100, M=5, B=10, N_min=1,
                            baseline_mode="uniform",
                            baseline_timestep_schedule_shift=1.0)
    s2 = InfoNoiseScheduler(K=16, N_warm=100, M=5, B=10, N_min=1,
                            baseline_mode="uniform",
                            baseline_timestep_schedule_shift=5.0)
    assert s1._cdf_values is None and s2._cdf_values is None
    torch.manual_seed(0)
    t1 = s1.sample(4096, "cpu")
    torch.manual_seed(0)
    t2 = s2.sample(4096, "cpu")
    assert t2.mean().item() > t1.mean().item() + 0.1
