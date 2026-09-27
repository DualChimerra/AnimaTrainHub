"""PP6.1 -- train_monitor: file writer (HTTP server removed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import train_monitor


@pytest.fixture(autouse=True)
def reset_state():
    train_monitor.reset_state()
    train_monitor.set_state_file(None)
    yield
    train_monitor.reset_state()
    train_monitor.set_state_file(None)


def test_set_state_file_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "deeply" / "nested" / "state.json"
    train_monitor.set_state_file(target)
    assert target.parent.exists()


def test_save_state_silent_when_no_file_set(tmp_path: Path) -> None:
    """save_state should not raise / should not write to disk when no path is set."""
    train_monitor.update_monitor(loss=0.5, step=1)
    # just needs to not raise (the disk write is guarded internally by try/except)


def test_update_monitor_writes_state_to_file(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    train_monitor.set_state_file(target)
    train_monitor.update_monitor(
        loss=0.5,
        lr=1e-4,
        step=10,
        total_steps=100,
        optimizer_metrics={"actual_lr": 1e-4, "d": 1e-4, "base_lr": 1.0},
    )
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["step"] == 10
    assert data["losses"][0]["loss"] == 0.5
    assert data["lr_history"][0]["lr"] == 1e-4
    assert data["optimizer_metrics_history"][0]["actual_lr"] == 1e-4
    assert data["optimizer_metrics_history"][0]["d"] == 1e-4
    assert data["start_time"] is not None  # auto-set


def test_update_monitor_appends_loss_history(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    train_monitor.set_state_file(target)
    for i in range(5):
        train_monitor.update_monitor(loss=0.5 - 0.01 * i, step=i)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert len(data["losses"]) == 5
    assert [p["step"] for p in data["losses"]] == list(range(5))


def test_restore_monitor_state(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    train_monitor.set_state_file(target)
    train_monitor.restore_monitor_state(
        losses=[{"step": 1, "loss": 0.5, "time": 100.0}],
        lr_history=[{"step": 1, "lr": 1e-4}],
        optimizer_metrics_history=[{"step": 1, "actual_lr": 1e-4, "d": 1e-4}],
        epoch=5, step=100, total_steps=1000, start_time=1700000000.0,
        config={"lora_rank": 64},
    )
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["epoch"] == 5
    assert data["step"] == 100
    assert data["config"]["lora_rank"] == 64
    assert data["optimizer_metrics_history"][0]["d"] == 1e-4


def test_get_state_returns_copy() -> None:
    train_monitor.update_monitor(loss=0.1, step=1)
    s1 = train_monitor.get_state()
    s2 = train_monitor.get_state()
    assert s1 == s2
    assert s1 is not s2


def test_downsample_uniform() -> None:
    pts = list(range(100))
    out = train_monitor._downsample_uniform(pts, 10)
    assert len(out) == 10
    assert out[0] == 0
    assert out[-1] == 99


def test_downsample_uniform_short_input_unchanged() -> None:
    pts = [1, 2, 3]
    assert train_monitor._downsample_uniform(pts, 10) == pts


def test_no_http_server_artifacts() -> None:
    """PP6.1 removal: start_monitor_server / MonitorHandler / HTML_TEMPLATE should no longer exist."""
    assert not hasattr(train_monitor, "start_monitor_server")
    assert not hasattr(train_monitor, "MonitorHandler")
    assert not hasattr(train_monitor, "HTML_TEMPLATE")
