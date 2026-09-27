from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from training.state import load_training_state, save_training_state
from training.timestep_samplers import baseline as baseline_mod
from training.timestep_samplers import infonoise as infonoise_mod


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _drive_to_cdf_ready(sched, n_steps=120, seed=0):
    rng = np.random.default_rng(seed)
    log_sigma_lo = float(np.log(0.001 / 0.999))   # ≈ -6.91
    log_sigma_hi = float(np.log(0.999 / 0.001))   # ≈ +6.91
    for step in range(n_steps):
        bs = 8
        log_sigma = rng.uniform(log_sigma_lo, log_sigma_hi, size=bs)
        sigma = np.exp(log_sigma)
        t = (sigma / (1.0 + sigma)).astype(np.float32)
        mse = (0.1 + 2.0 * np.log1p(sigma) + rng.normal(0, 0.05, size=bs)).astype(np.float32)
        sched.record(torch.from_numpy(t), torch.from_numpy(mse))
        sched.maybe_refresh(global_step=step)


def _new_info_sched(**kw):
    defaults = dict(K=8, N_warm=20, B=16, M=5, N_min=2)
    defaults.update(kw)
    return infonoise_mod.InfoNoiseScheduler(**defaults)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_baseline_state_dict_is_empty():
    s = baseline_mod.BaselineTimestepSampler(mode="logit_normal", shift=3.0)
    assert s.state_dict() == {}


def test_baseline_load_state_dict_accepts_anything():
    s = baseline_mod.BaselineTimestepSampler(mode="logit_normal", shift=3.0)
    s.load_state_dict({})
    s.load_state_dict({"garbage": [1, 2, 3], "unexpected": "field"})


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_infonoise_state_dict_roundtrip_cold():
    s1 = _new_info_sched()
    sd = s1.state_dict()
    assert sd["cdf_values"] is None
    assert sd["internal_step"] == 0

    s2 = _new_info_sched()
    s2.load_state_dict(sd)
    assert s2._internal_step == s1._internal_step
    assert s2._cdf_values is None
    assert np.array_equal(s2._mse_ema, s1._mse_ema)
    assert np.array_equal(s2._n_count, s1._n_count)
    for buf1, buf2 in zip(s1._fifo, s2._fifo):
        assert list(buf1) == list(buf2)
        assert buf2.maxlen == s2.B


def test_infonoise_state_dict_roundtrip_warm_cdf_ready():
    s1 = _new_info_sched()
    _drive_to_cdf_ready(s1, n_steps=80)
    assert s1._cdf_values is not None, "test setup did not push the CDF to ready"
    assert s1._last_refresh_status == "ok"

    sd = s1.state_dict()
    s2 = _new_info_sched()
    s2.load_state_dict(sd)

    assert s2._internal_step == s1._internal_step
    assert s2._last_refresh_status == s1._last_refresh_status
    assert s2._refresh_attempts == s1._refresh_attempts
    assert s2._refresh_degraded_count == s1._refresh_degraded_count
    assert s2._warned_cold_start == s1._warned_cold_start
    np.testing.assert_allclose(s2._mse_ema, s1._mse_ema, rtol=1e-12, atol=1e-12)
    assert np.array_equal(s2._n_count, s1._n_count)
    np.testing.assert_allclose(s2._cdf_values, s1._cdf_values, rtol=1e-12, atol=1e-12)

    for i, (buf1, buf2) in enumerate(zip(s1._fifo, s2._fifo)):
        assert list(buf1) == list(buf2), f"fifo[{i}] content mismatch"
        assert buf2.maxlen == s2.B, f"fifo[{i}] maxlen mismatch"


def test_infonoise_sample_deterministic_after_resume():
    s1 = _new_info_sched()
    _drive_to_cdf_ready(s1, n_steps=80)
    assert s1._cdf_values is not None

    torch.manual_seed(7)
    expected = s1.sample(8, device="cpu")

    sd = s1.state_dict()
    s2 = _new_info_sched()
    s2.load_state_dict(sd)

    torch.manual_seed(7)
    actual = s2.sample(8, device="cpu")
    assert torch.allclose(actual, expected, atol=1e-6), (
        f"sampling distribution drifted after resume: expected={expected}, actual={actual}"
    )


def test_infonoise_sample_cold_vs_loaded_differs():
    s1 = _new_info_sched()
    _drive_to_cdf_ready(s1, n_steps=80)
    assert s1._cdf_values is not None

    torch.manual_seed(7)
    with_cdf = s1.sample(8, device="cpu")

    s2 = _new_info_sched()
    assert s2._cdf_values is None
    torch.manual_seed(7)
    cold = s2.sample(8, device="cpu")

    assert not torch.allclose(with_cdf, cold, atol=1e-3), (
        "cold-start sample and CDF-ready sample produced the same output unexpectedly -- did test setup fail to push the CDF?"
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_infonoise_load_state_dict_K_mismatch_falls_back_cold(caplog):
    s_save = _new_info_sched(K=8)
    _drive_to_cdf_ready(s_save, n_steps=80)
    sd = s_save.state_dict()

    s_load = _new_info_sched(K=16)
    with caplog.at_level("WARNING"):
        s_load.load_state_dict(sd)
    assert any("shape mismatch" in r.message for r in caplog.records)
    assert s_load._internal_step == 0
    assert s_load._cdf_values is None


def test_infonoise_load_state_dict_B_mismatch_falls_back_cold():
    s_save = _new_info_sched(B=16)
    _drive_to_cdf_ready(s_save, n_steps=80)
    sd = s_save.state_dict()

    s_load = _new_info_sched(B=64)
    s_load.load_state_dict(sd)
    assert s_load._internal_step == 0
    assert s_load._cdf_values is None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_infonoise_continues_recording_after_resume():
    s1 = _new_info_sched()
    _drive_to_cdf_ready(s1, n_steps=60)
    sd = s1.state_dict()

    s2 = _new_info_sched()
    s2.load_state_dict(sd)
    pre_step = s2._internal_step
    pre_attempts = s2._refresh_attempts

    _drive_to_cdf_ready(s2, n_steps=60, seed=1)
    assert s2._internal_step > pre_step
    assert s2._refresh_attempts > pre_attempts


def test_infonoise_fifo_maxlen_preserved_after_resume():
    s1 = _new_info_sched(B=4)
    for _ in range(10):
        t = torch.tensor([0.5], dtype=torch.float32)
        mse = torch.tensor([0.1], dtype=torch.float32)
        s1.record(t, mse)
    full_bins = [i for i, buf in enumerate(s1._fifo) if len(buf) == 4]
    assert full_bins, "test setup: no bin got filled"

    sd = s1.state_dict()
    s2 = _new_info_sched(B=4)
    s2.load_state_dict(sd)
    for i in full_bins:
        assert s2._fifo[i].maxlen == 4
        s2._fifo[i].append(999.0)
        assert len(s2._fifo[i]) == 4
        assert 999.0 in s2._fifo[i]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class _TinyInjector(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.lin(x)


def _make_tiny_optimizer():
    m = _TinyInjector()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss = m(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return m, opt


def test_save_load_training_state_persists_timestep_sampler(tmp_path):
    injector, opt = _make_tiny_optimizer()
    sched = _new_info_sched()
    _drive_to_cdf_ready(sched, n_steps=80)
    assert sched._cdf_values is not None

    state_path = tmp_path / "state.pt"
    save_training_state(
        state_path, injector, opt, epoch=1, global_step=42,
        timestep_sampler=sched,
    )

    injector2 = _TinyInjector()
    opt2 = torch.optim.AdamW(injector2.parameters(), lr=1e-3)
    sched2 = _new_info_sched()
    assert sched2._cdf_values is None

    epoch, step, _, _ = load_training_state(
        state_path, injector2, opt2, timestep_sampler=sched2,
    )
    assert (epoch, step) == (1, 42)
    assert sched2._cdf_values is not None, "InfoNoise CDF was not restored from the checkpoint"
    np.testing.assert_allclose(sched2._cdf_values, sched._cdf_values, rtol=1e-12)


def test_save_skips_baseline_sampler_state(tmp_path):
    injector, opt = _make_tiny_optimizer()
    sched = baseline_mod.BaselineTimestepSampler(mode="logit_normal", shift=3.0)
    state_path = tmp_path / "state.pt"
    save_training_state(
        state_path, injector, opt, epoch=0, global_step=0,
        timestep_sampler=sched,
    )
    raw = torch.load(state_path, map_location="cpu", weights_only=False)
    assert "timestep_sampler_state" not in raw


def test_load_state_without_sampler_state_does_not_call_load(tmp_path):
    injector, opt = _make_tiny_optimizer()
    state_path = tmp_path / "state.pt"
    save_training_state(state_path, injector, opt, epoch=0, global_step=0)

    injector2 = _TinyInjector()
    opt2 = torch.optim.AdamW(injector2.parameters(), lr=1e-3)
    sched2 = _new_info_sched()
    _drive_to_cdf_ready(sched2, n_steps=50)
    cdf_before = sched2._cdf_values.copy() if sched2._cdf_values is not None else None
    internal_before = sched2._internal_step

    load_training_state(state_path, injector2, opt2, timestep_sampler=sched2)

    assert sched2._internal_step == internal_before
    if cdf_before is not None:
        np.testing.assert_array_equal(sched2._cdf_values, cdf_before)


def test_load_corrupted_sampler_state_logs_and_continues(tmp_path, caplog):
    injector, opt = _make_tiny_optimizer()
    sched = _new_info_sched()
    _drive_to_cdf_ready(sched, n_steps=80)
    state_path = tmp_path / "state.pt"
    save_training_state(
        state_path, injector, opt, epoch=0, global_step=0,
        timestep_sampler=sched,
    )

    raw = torch.load(state_path, map_location="cpu", weights_only=False)
    del raw["timestep_sampler_state"]["fifo"]
    torch.save(raw, state_path)

    injector2 = _TinyInjector()
    opt2 = torch.optim.AdamW(injector2.parameters(), lr=1e-3)
    sched2 = _new_info_sched()
    with caplog.at_level("WARNING"):
        load_training_state(state_path, injector2, opt2, timestep_sampler=sched2)
    assert any("timestep_sampler state restore failed" in r.message for r in caplog.records)
    assert sched2._cdf_values is None
