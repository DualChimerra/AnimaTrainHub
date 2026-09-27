from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from training.timestep_samplers import build_timestep_sampler
from training.timestep_samplers.infonoise import InfoNoiseScheduler
from training.timestep_samplers.infonoise import build as build_info_noise


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_ema_responsiveness_codifies_design_choice():
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1, beta=0.9)
    s._mse_ema = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    for k in range(4):
        s._fifo[k].append(10.0)
        s._n_count[k] = 1
    s._refresh()
    np.testing.assert_allclose(s._mse_ema, 9.1, rtol=1e-6)


def test_ema_beta_high_means_high_responsiveness():
    s_responsive = InfoNoiseScheduler(K=2, N_warm=1, M=1, B=2, N_min=1, beta=0.99)
    s_responsive._mse_ema = np.array([0.0, 0.0], dtype=np.float64)
    for k in range(2):
        s_responsive._fifo[k].append(100.0)
        s_responsive._n_count[k] = 1
    s_responsive._refresh()
    assert s_responsive._mse_ema[0] >= 99.0

    s_smooth = InfoNoiseScheduler(K=2, N_warm=1, M=1, B=2, N_min=1, beta=0.01)
    s_smooth._mse_ema = np.array([0.0, 0.0], dtype=np.float64)
    for k in range(2):
        s_smooth._fifo[k].append(100.0)
        s_smooth._n_count[k] = 1
    s_smooth._refresh()
    assert s_smooth._mse_ema[0] <= 2.0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_refresh_status_mse_collapsed():
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1, beta=0.5)
    for k in range(4):
        s._fifo[k].append(0.0)
        s._n_count[k] = 1
    s._refresh()
    assert s._last_refresh_status == "mse_collapsed"
    assert s._cdf_values is None
    assert s._refresh_degraded_count == 1


def test_refresh_status_ok_path():
    s = InfoNoiseScheduler(K=8, N_warm=1, M=1, B=2, N_min=1, beta=0.5)
    for k in range(8):
        val = 10.0 if 2 <= k <= 5 else 0.01
        s._fifo[k].append(val)
        s._n_count[k] = 1
    s._refresh()
    assert s._last_refresh_status == "ok"
    assert s._cdf_values is not None
    cdf = s._cdf_values
    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0
    assert np.all(np.diff(cdf) >= -1e-12)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_sample_falls_back_to_baseline_when_cdf_not_ready():
    s = InfoNoiseScheduler(K=4, N_warm=10, M=5, B=2, N_min=1)
    t = s.sample(4, "cpu")
    assert t.shape == (4,)
    assert (t > 0).all() and (t < 1).all()


def test_sample_after_refresh_in_unit_range():
    s = InfoNoiseScheduler(K=8, N_warm=1, M=1, B=2, N_min=1)
    for k in range(8):
        val = 10.0 if 2 <= k <= 5 else 0.01
        s._fifo[k].append(val)
        s._n_count[k] = 1
    s._refresh()
    assert s._cdf_values is not None
    t = s.sample(16, "cpu")
    assert (t > 0).all() and (t < 1).all()


@pytest.mark.parametrize("mode", ["logit_normal", "uniform", "logit_normal_low", "mode"])
def test_baseline_mode_works_for_all_options(mode):
    s = InfoNoiseScheduler(K=4, N_warm=10, M=5, B=2, N_min=1, baseline_mode=mode)
    t = s._sample_baseline(8, "cpu")
    assert t.shape == (8,)
    assert (t > 0).all() and (t < 1).all()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_cold_start_warning_emits_once(caplog):
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1)
    for k in range(4):
        for _ in range(2):
            s._fifo[k].append(0.0)
            s._n_count[k] = 2
    s._internal_step = 100

    with caplog.at_level(logging.WARNING, logger="training.timestep_samplers.infonoise"):
        s.maybe_refresh(global_step=1)
        s.maybe_refresh(global_step=2)
        s.maybe_refresh(global_step=3)
    warnings = [r for r in caplog.records if "InfoNoise" in r.message]
    assert len(warnings) == 1
    assert "mse_collapsed" in warnings[0].message


def test_status_dict_shape():
    s = InfoNoiseScheduler(K=4, N_warm=10, M=5, B=2, N_min=1)
    status = s.status()
    assert set(status.keys()) == {
        "kind",
        "cdf_ready",
        "last_refresh_status",
        "refresh_attempts",
        "refresh_degraded_count",
        "internal_step",
    }
    assert status["kind"] == "infonoise"
    assert status["cdf_ready"] is False
    assert status["last_refresh_status"] == "not_refreshed_yet"


# ---------------------------------------------------------------------------
# build_info_noise factory
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_build_timestep_sampler_disabled_returns_baseline():
    args = _FakeArgs(
        infonoise_enabled=False,
        timestep_sampling="logit_normal",
        timestep_shift=3.0,
    )
    sampler = build_timestep_sampler(args, total_steps=1000)
    assert sampler is not None
    assert sampler.status()["kind"] == "baseline"


def test_build_info_noise_n_warm_auto_20_pct():
    args = _FakeArgs(infonoise_enabled=True, infonoise_N_warm=0, timestep_sampling="logit_normal", timestep_shift=3.0)
    s = build_info_noise(args, total_steps=10_000)
    assert s is not None
    # 10000 × 20% = 2000
    assert s.N_warm == 2000


def test_build_info_noise_n_warm_min_200():
    args = _FakeArgs(infonoise_enabled=True, infonoise_N_warm=0, timestep_sampling="logit_normal", timestep_shift=3.0)
    s = build_info_noise(args, total_steps=100)
    assert s is not None
    assert s.N_warm == 200


def test_build_info_noise_passes_baseline_mode():
    args = _FakeArgs(
        infonoise_enabled=True, infonoise_N_warm=500,
        timestep_sampling="uniform", timestep_shift=3.0,
    )
    s = build_info_noise(args, total_steps=10_000)
    assert s is not None
    assert s.baseline_mode == "uniform"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_n_warm_uses_optimizer_step_not_micro_batch():
    s = InfoNoiseScheduler(K=4, N_warm=100, M=10, B=4, N_min=1, beta=0.9)
    for _ in range(200):
        s.record(torch.tensor([0.5] * 4), torch.tensor([1.0] * 4))
    s.maybe_refresh(global_step=50)
    assert s._cdf_values is None
    assert s._internal_step == 200
    assert s._internal_step > 100


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_default_gate_pivot_uses_paper_c_value():
    s = InfoNoiseScheduler(K=8, N_warm=1, M=1, B=2, N_min=1)
    assert s.gate_pivot_c == 0.15


def test_dynamic_gate_pivot_when_set_to_zero():
    s = InfoNoiseScheduler(K=8, N_warm=1, M=1, B=2, N_min=1, gate_pivot_c=0.0)
    for k in range(8):
        val = 10.0 if 2 <= k <= 5 else 0.01
        s._fifo[k].append(val)
        s._n_count[k] = 1
    s._refresh()
    assert s._last_refresh_status == "ok"
    assert s._cdf_values is not None


def test_p_onset_rejected_when_invalid():
    with pytest.raises(ValueError, match="p_onset"):
        InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1, p_onset=1.5)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_state_dict_version_mismatch_triggers_cold_start():
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1)
    old_state = {
        "K": 4, "B": 2,
        "fifo": [[1.0], [2.0], [3.0], [4.0]],
        "mse_ema": np.array([1.0, 2.0, 3.0, 4.0]),
        "n_count": np.array([1, 1, 1, 1], dtype=np.int32),
        "cdf_values": np.linspace(0, 1, 5),
        "internal_step": 100,
        "last_refresh_status": "ok",
        "refresh_attempts": 5,
        "refresh_degraded_count": 0,
        "warned_cold_start": False,
    }
    s.load_state_dict(old_state)
    np.testing.assert_array_equal(s._mse_ema, np.zeros(4))
    assert s._cdf_values is None
    assert s._internal_step == 0


def test_state_dict_round_trip_v2():
    s1 = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1)
    for k in range(4):
        s1._fifo[k].append(float(k))
    s1._mse_ema = np.array([1.0, 2.0, 3.0, 4.0])
    state = s1.state_dict()
    assert state["__version__"] == 2

    s2 = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=2, N_min=1)
    s2.load_state_dict(state)
    np.testing.assert_array_equal(s2._mse_ema, s1._mse_ema)
    assert list(s2._fifo[2]) == [2.0]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_record_accepts_partial_batch_after_reg_mask():
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=4, N_min=1, beta=0.5)
    t_full = torch.tensor([0.1, 0.5, 0.9, 0.5])
    mse_full = torch.tensor([10.0, 5.0, 1.0, 5.0])
    is_reg = torch.tensor([False, True, True, False])
    main_mask = ~is_reg
    s.record(t_full[main_mask], mse_full[main_mask])
    assert s._internal_step == 1
    assert int(s._n_count.sum()) == 2


def test_record_handles_empty_after_mask():
    s = InfoNoiseScheduler(K=4, N_warm=1, M=1, B=4, N_min=1, beta=0.5)
    empty_t = torch.tensor([], dtype=torch.float32)
    empty_mse = torch.tensor([], dtype=torch.float32)
    s.record(empty_t, empty_mse)
    assert int(s._n_count.sum()) == 0
    assert s._internal_step == 1
