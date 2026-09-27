from __future__ import annotations

import json
import time
from pathlib import Path

from studio.infrastructure.log_tail import LogTailer, MonitorStatePoller


def _wait_lines(received: list[str], n: int, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while len(received) < n and time.time() < deadline:
        time.sleep(0.05)


def test_tailer_picks_up_appended_lines(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.touch()
    received: list[str] = []
    tailer = LogTailer(log, received.append, poll_interval=0.05)
    tailer.start()
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write("line one\n")
            f.write("line two\n")
        _wait_lines(received, 2)
        with open(log, "a", encoding="utf-8") as f:
            f.write("line three\n")
        _wait_lines(received, 3)
    finally:
        tailer.stop()
    assert received[:3] == ["line one", "line two", "line three"]


def test_tailer_handles_missing_file(tmp_path: Path) -> None:
    log = tmp_path / "later.log"
    received: list[str] = []
    tailer = LogTailer(log, received.append, poll_interval=0.05)
    tailer.start()
    try:
        time.sleep(0.1)
        log.write_text("hello\n", encoding="utf-8")
        _wait_lines(received, 1)
    finally:
        tailer.stop()
    assert received == ["hello"]


def test_tailer_flushes_partial_line_on_stop(tmp_path: Path) -> None:
    log = tmp_path / "p.log"
    log.write_text("abc", encoding="utf-8")
    received: list[str] = []
    tailer = LogTailer(log, received.append, poll_interval=0.05)
    tailer.start()
    time.sleep(0.1)
    tailer.stop()
    assert received == ["abc"]




def _write_state(path: Path, state: dict) -> None:
    prev_mtime = path.stat().st_mtime if path.exists() else 0
    path.write_text(json.dumps(state), encoding="utf-8")
    import os
    new_mtime = max(prev_mtime + 1.0, time.time())
    os.utime(path, (new_mtime, new_mtime))


def _wait_for(n: int, deltas: list, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while len(deltas) < n and time.time() < deadline:
        time.sleep(0.05)


def test_poller_first_publish_is_full_delta(tmp_path: Path) -> None:
    sf = tmp_path / "state.json"
    _write_state(sf, {
        "step": 3, "total_steps": 100,
        "losses": [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.4}, {"step": 3, "loss": 0.3}],
        "lr_history": [{"step": 1, "lr": 1e-4}],
        "optimizer_metrics_history": [{"step": 1, "actual_lr": 1e-4, "d": 1e-4}],
        "samples": [{"path": "/x/y.png", "step": 1}],
        "config": {"model": "X"},
    })
    deltas: list[dict] = []
    poller = MonitorStatePoller(sf, deltas.append, poll_interval=0.05, min_publish_interval=0.0)
    poller.start()
    try:
        _wait_for(1, deltas)
    finally:
        poller.stop()
    assert len(deltas) >= 1
    d = deltas[0]
    assert d["step"] == 3
    assert len(d["appended_losses"]) == 3
    assert len(d["appended_lr"]) == 1
    assert len(d["appended_optimizer_metrics"]) == 1
    assert d["appended_optimizer_metrics"][0]["d"] == 1e-4
    assert len(d["appended_samples"]) == 1
    assert d.get("config") == {"model": "X"}


def test_poller_second_publish_only_new_entries(tmp_path: Path) -> None:
    sf = tmp_path / "state.json"
    _write_state(sf, {
        "step": 1, "losses": [{"step": 1, "loss": 0.5}], "lr_history": [],
        "optimizer_metrics_history": [], "samples": [],
    })
    deltas: list[dict] = []
    poller = MonitorStatePoller(sf, deltas.append, poll_interval=0.05, min_publish_interval=0.0)
    poller.start()
    try:
        _wait_for(1, deltas)
        _write_state(sf, {
            "step": 3,
            "losses": [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.4}, {"step": 3, "loss": 0.3}],
            "lr_history": [{"step": 3, "lr": 1e-5}],
            "optimizer_metrics_history": [{"step": 3, "actual_lr": 1e-5, "d": 1e-5}],
            "samples": [{"path": "/x/new.png", "step": 3}],
        })
        _wait_for(2, deltas)
    finally:
        poller.stop()
    assert len(deltas) >= 2
    d2 = deltas[1]
    assert d2["step"] == 3
    assert [l["step"] for l in d2["appended_losses"]] == [2, 3]
    assert len(d2["appended_lr"]) == 1
    assert len(d2["appended_optimizer_metrics"]) == 1
    assert d2["appended_optimizer_metrics"][0]["actual_lr"] == 1e-5
    assert len(d2["appended_samples"]) == 1
    assert "config" not in d2


def test_poller_min_publish_interval_throttles(tmp_path: Path) -> None:
    sf = tmp_path / "state.json"
    _write_state(sf, {"step": 1, "losses": [{"step": 1, "loss": 0.5}], "lr_history": [], "samples": []})
    deltas: list[dict] = []
    poller = MonitorStatePoller(sf, deltas.append, poll_interval=0.05, min_publish_interval=1.0)
    poller.start()
    try:
        _wait_for(1, deltas, timeout=1.0)
        _write_state(sf, {"step": 2, "losses": [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.4}],
                          "lr_history": [], "samples": []})
        time.sleep(0.3)
        assert len(deltas) == 1, f"expected throttled, got {len(deltas)} deltas"
    finally:
        poller.stop()


def test_poller_no_progress_no_publish(tmp_path: Path) -> None:
    sf = tmp_path / "state.json"
    state = {"step": 0, "losses": [], "lr_history": [], "samples": [], "config": {}}
    _write_state(sf, state)
    deltas: list[dict] = []
    poller = MonitorStatePoller(sf, deltas.append, poll_interval=0.05, min_publish_interval=0.0)
    poller.start()
    try:
        _wait_for(1, deltas)
        assert len(deltas) == 1
        _write_state(sf, state)
        time.sleep(0.3)
        assert len(deltas) == 1
    finally:
        poller.stop()


def test_tailer_strips_ansi_and_nul(tmp_path: Path) -> None:
    log = tmp_path / "ansi.log"
    raw = (
        b"\x1b[1;31m2026-05-06 [E:onnxruntime] FAIL\x1b[m\n"
        b"o\x00n\x00n\x00x\x00\n"
    )
    log.write_bytes(raw)
    received: list[str] = []
    tailer = LogTailer(log, received.append, poll_interval=0.05)
    tailer.start()
    _wait_lines(received, 2)
    tailer.stop()
    assert received[:2] == [
        "2026-05-06 [E:onnxruntime] FAIL",
        "onnx",
    ]
