from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from training.ema import AdapterEma, build_ema


def _params(values, dtype=torch.float32):
    return [torch.nn.Parameter(torch.tensor(v, dtype=dtype)) for v in values]


def test_shadow_starts_from_current_weights_not_from_zero():
    p = _params([[1.0, 2.0]])
    ema = AdapterEma(p, decay=0.999)
    assert not ema.active
    ema.update(0)
    assert ema.active
    with ema.applied() as ok:
        assert ok
        assert torch.allclose(p[0].data, torch.tensor([1.0, 2.0]))


def test_tracks_a_moving_average():
    p = _params([[0.0]])
    ema = AdapterEma(p, decay=0.9, warmup=False)
    ema.update(0)
    p[0].data.fill_(1.0)
    ema.update(1)                      # 0.9*0 + 0.1*1 = 0.1
    with ema.applied():
        assert pytest.approx(float(p[0].data), abs=1e-6) == 0.1
    assert pytest.approx(float(p[0].data)) == 1.0


def test_warmup_lets_the_shadow_catch_up_early():
    def run(warmup: bool) -> float:
        p = _params([[0.0]])
        ema = AdapterEma(p, decay=0.999, warmup=warmup)
        ema.update(0)
        for step in range(1, 30):
            p[0].data.fill_(1.0)
            ema.update(step)
        with ema.applied():
            return float(p[0].data)

    assert run(True) > 0.5
    assert run(False) < 0.05


def test_start_step_skips_the_early_phase():
    p = _params([[5.0]])
    ema = AdapterEma(p, decay=0.9, start_step=10)
    for step in range(10):
        ema.update(step)
    assert not ema.active
    ema.update(10)
    assert ema.active
    with ema.applied():
        assert pytest.approx(float(p[0].data)) == 5.0


def test_shadow_is_fp32_even_for_bf16_weights():
    p = _params([[0.0]], dtype=torch.bfloat16)
    ema = AdapterEma(p, decay=0.999, warmup=False)
    ema.update(0)
    assert ema._shadow[0].dtype == torch.float32

    for step in range(1, 200):
        p[0].data.fill_(1.0)
        ema.update(step)
    assert float(ema._shadow[0]) > 0.15


def test_applied_restores_weights_even_on_exception():
    p = _params([[3.0]])
    ema = AdapterEma(p, decay=0.9, warmup=False)
    ema.update(0)
    p[0].data.fill_(7.0)
    with pytest.raises(RuntimeError):
        with ema.applied():
            assert pytest.approx(float(p[0].data)) == 3.0
            raise RuntimeError("boom")
    assert pytest.approx(float(p[0].data)) == 7.0


def test_applied_is_a_noop_before_the_shadow_exists():
    p = _params([[1.0]])
    ema = AdapterEma(p, decay=0.9, start_step=100)
    with ema.applied() as ok:
        assert ok is False


def test_state_roundtrip_preserves_the_average():
    p = _params([[0.0]])
    ema = AdapterEma(p, decay=0.9, warmup=False)
    ema.update(0)
    p[0].data.fill_(1.0)
    ema.update(1)
    state = ema.state_dict()

    fresh_p = _params([[99.0]])
    fresh = AdapterEma(fresh_p, decay=0.5)
    fresh.load_state_dict(state)
    assert fresh.decay == 0.9
    assert fresh.updates == ema.updates
    with fresh.applied() as ok:
        assert ok
        assert pytest.approx(float(fresh_p[0].data), abs=1e-6) == 0.1


def test_state_with_mismatched_shapes_is_rejected_not_crashed():
    ema = AdapterEma(_params([[0.0], [0.0]]), decay=0.9)
    ema.update(0)
    state = ema.state_dict()
    other = AdapterEma(_params([[0.0]]), decay=0.9)
    other.load_state_dict(state)
    assert not other.active


def test_build_ema_reads_args():
    params = _params([[0.0]])
    assert build_ema(SimpleNamespace(ema_enabled=False), params, 1000) is None

    args = SimpleNamespace(ema_enabled=True, ema_decay=0.995, ema_start_ratio=0.3)
    ema = build_ema(args, params, 1000)
    assert ema is not None
    assert ema.decay == 0.995
    assert ema.start_step == 300


def test_build_ema_without_total_steps_starts_from_zero():
    args = SimpleNamespace(ema_enabled=True, ema_decay=0.999, ema_start_ratio=0.5)
    ema = build_ema(args, _params([[0.0]]), None)
    assert ema is not None and ema.start_step == 0


def test_schema_fields():
    from studio.schema import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.ema_enabled is False
    assert cfg.ema_decay == 0.999
    assert cfg.ema_start_ratio == 0.0
    for name in ("ema_decay", "ema_start_ratio"):
        meta = TrainingConfig.model_fields[name].json_schema_extra
        assert meta["show_when"] == "ema_enabled==true", name
