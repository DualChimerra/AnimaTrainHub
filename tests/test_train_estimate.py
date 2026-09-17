"""Pre-run estimate service: measured speed + memory verdict.

The point of this module is that both halves are *honest*: the memory verdict
reuses the same arithmetic as the guard that aborts training, and the duration
comes from a real measurement rather than a guess. These tests pin exactly that,
plus the degrade-to-None behaviour that keeps the training page rendering when a
piece of the estimate is unavailable.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from studio.services import train_estimate


# ── measured speed ───────────────────────────────────────────────────────────

class _FakeConn:
    pass


def _task(tid: int, *, project_id: int = 1, status: str = "done",
          started: str = "2026-01-01T00:00:00", ttype: str = "train") -> dict:
    return {
        "id": tid, "project_id": project_id, "status": status,
        "type": ttype, "started_at": started, "config_name": f"run{tid}",
    }


def _patch(monkeypatch, tasks, speeds):
    monkeypatch.setattr(
        "studio.infrastructure.db.list_tasks",
        lambda conn: tasks,
    )
    monkeypatch.setattr(
        train_estimate, "_load_monitor_speed",
        lambda tid: speeds.get(tid),
    )


def test_picks_the_most_recent_task_with_a_speed(monkeypatch):
    _patch(monkeypatch,
           [_task(1, started="2026-01-01"), _task(2, started="2026-03-01")],
           {1: 0.10, 2: 0.25})
    out = train_estimate.measured_speed(_FakeConn(), 1)
    assert out == {"it_per_s": 0.25, "task_id": 2, "task_name": "run2"}


def test_skips_tasks_that_never_recorded_a_speed(monkeypatch):
    """A task that died before the first logged step has no usable number."""
    _patch(monkeypatch,
           [_task(1, started="2026-01-01"), _task(2, started="2026-03-01")],
           {1: 0.10})
    out = train_estimate.measured_speed(_FakeConn(), 1)
    assert out is not None and out["task_id"] == 1


def test_ignores_other_projects_and_non_training_tasks(monkeypatch):
    _patch(monkeypatch,
           [_task(1, project_id=2), _task(2, ttype="generate")],
           {1: 0.5, 2: 0.5})
    assert train_estimate.measured_speed(_FakeConn(), 1) is None


def test_no_history_returns_none(monkeypatch):
    """Before the first run there is nothing to measure — the UI shows nothing
    rather than a fabricated duration."""
    _patch(monkeypatch, [], {})
    assert train_estimate.measured_speed(_FakeConn(), 1) is None


def test_db_failure_does_not_break_the_page(monkeypatch):
    def boom(conn):
        raise RuntimeError("db is gone")
    monkeypatch.setattr("studio.infrastructure.db.list_tasks", boom)
    assert train_estimate.measured_speed(_FakeConn(), 1) is None


# ── monitor state reading ────────────────────────────────────────────────────

def test_reads_speed_from_monitor_state(monkeypatch, tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"speed": 0.42, "step": 100}), encoding="utf-8")
    monkeypatch.setattr(
        "studio.infrastructure.paths.task_monitor_state_path",
        lambda tid: state,
    )
    assert train_estimate._load_monitor_speed(7) == 0.42


@pytest.mark.parametrize("payload", ['{"speed": 0}', '{"speed": null}', '{}', 'not json'])
def test_unusable_monitor_state_yields_none(monkeypatch, tmp_path, payload):
    state = tmp_path / "state.json"
    state.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(
        "studio.infrastructure.paths.task_monitor_state_path",
        lambda tid: state,
    )
    assert train_estimate._load_monitor_speed(7) is None


def test_missing_monitor_state_yields_none(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "studio.infrastructure.paths.task_monitor_state_path",
        lambda tid: tmp_path / "nope.json",
    )
    assert train_estimate._load_monitor_speed(7) is None


# ── memory verdict ───────────────────────────────────────────────────────────

def test_memory_without_a_checkpoint_is_unknown():
    """No verdict is the correct answer for "we cannot tell" — not a false OK."""
    assert train_estimate.memory_fit({"transformer_path": "/nope/missing.safetensors"}) is None
    assert train_estimate.memory_fit({}) is None


def test_memory_verdict_shape(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"x" * 1024)

    fake_family = SimpleNamespace(
        spec=SimpleNamespace(capabilities={"block_swap"}),
        swapped_param_ratio=lambda n, checkpoint_path=None: n / 28,
        swappable_blocks=lambda checkpoint_path=None: 28,
    )
    monkeypatch.setattr("training.families.get_family", lambda fid: fake_family)
    monkeypatch.setattr("training.sysmem.gpu_free_bytes_global", lambda: 12 * 1024 ** 3)
    monkeypatch.setattr("training.sysmem.available_ram_bytes", lambda: 32 * 1024 ** 3)

    out = train_estimate.memory_fit({
        "transformer_path": str(ckpt),
        "model_family": "anima",
        "blocks_to_swap": 26,
    })
    assert out is not None
    assert out["blocks_to_swap"] == 26
    assert out["total_blocks"] == 28
    assert out["free_vram_bytes"] == 12 * 1024 ** 3
    assert isinstance(out["ok"], bool)


def test_memory_verdict_is_none_without_block_swap_capability(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(
        "training.families.get_family",
        lambda fid: SimpleNamespace(spec=SimpleNamespace(capabilities=set())),
    )
    assert train_estimate.memory_fit({"transformer_path": str(ckpt)}) is None


def test_estimate_bundles_both_halves(monkeypatch):
    monkeypatch.setattr(train_estimate, "measured_speed", lambda conn, pid: {"it_per_s": 1.0})
    monkeypatch.setattr(train_estimate, "memory_fit", lambda cfg: {"ok": True})
    out = train_estimate.estimate(_FakeConn(), 1, {})
    assert set(out) == {"speed", "memory"}
