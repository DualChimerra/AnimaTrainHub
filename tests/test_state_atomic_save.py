"""ADR 0006 Addendum 2 -- save_training_state / write_config_snapshot atomic disk writes.

auto_epoch_state.pt is an overwrite-in-place single-file recovery checkpoint; a direct torch.save hit by
a power loss / hard kill during the write window turns the only recovery point into a half-written file. Invariants after switching to tmp + os.replace:

1. Success path: the target file loads cleanly and no .tmp file is left in the directory.
2. Failure mid-write: the old file is preserved untouched (not corrupted by a half-written new file), and the tmp file is cleaned up.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from runtime.training.snapshot import write_config_snapshot
from runtime.training.state import save_training_state


class _FakeInjector:
    def state_dict(self):
        return {"w": torch.zeros(2)}


class _FakeOptimizer:
    def state_dict(self):
        return {"lr": 1e-4}


def _save(path: Path) -> None:
    save_training_state(
        path, _FakeInjector(), _FakeOptimizer(), epoch=1, global_step=100,
    )


# ---------------------------------------------------------------------------
# save_training_state
# ---------------------------------------------------------------------------


def test_save_success_no_tmp_leftover(tmp_path: Path) -> None:
    pt = tmp_path / "auto_epoch_state.pt"
    _save(pt)
    assert pt.exists()
    state = torch.load(pt, map_location="cpu", weights_only=False)
    assert state["global_step"] == 100
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_overwrites_previous_atomically(tmp_path: Path) -> None:
    pt = tmp_path / "auto_epoch_state.pt"
    _save(pt)
    save_training_state(
        pt, _FakeInjector(), _FakeOptimizer(), epoch=2, global_step=200,
    )
    state = torch.load(pt, map_location="cpu", weights_only=False)
    assert state["global_step"] == 200
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_failure_keeps_old_file(tmp_path: Path, monkeypatch) -> None:
    """torch.save blows up mid-write (simulating power loss / disk full) -> old recovery point preserved as-is + tmp cleaned up."""
    pt = tmp_path / "auto_epoch_state.pt"
    _save(pt)
    before = pt.read_bytes()

    import runtime.training.state as state_mod

    def _boom(obj, path):
        Path(path).write_bytes(b"half-written garbage")
        raise OSError("disk full")

    monkeypatch.setattr(state_mod.torch, "save", _boom)
    with pytest.raises(OSError):
        save_training_state(
            pt, _FakeInjector(), _FakeOptimizer(), epoch=2, global_step=200,
        )

    assert pt.read_bytes() == before  # old file wasn't corrupted
    assert list(tmp_path.glob("*.tmp")) == []  # tmp cleaned up


# ---------------------------------------------------------------------------
# write_config_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_success_no_tmp_leftover(tmp_path: Path) -> None:
    snap = tmp_path / "auto_epoch_state.config.json"
    write_config_snapshot(snap, {"lr": 1e-4}, ["prompt"])
    payload = json.loads(snap.read_text(encoding="utf-8"))
    assert payload["args"]["lr"] == 1e-4
    assert list(tmp_path.glob("*.tmp")) == []


def test_snapshot_failure_keeps_old_file(tmp_path: Path, monkeypatch) -> None:
    snap = tmp_path / "auto_epoch_state.config.json"
    write_config_snapshot(snap, {"lr": 1e-4})
    before = snap.read_text(encoding="utf-8")

    import runtime.training.snapshot as snap_mod

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(snap_mod.os, "replace", _boom)
    with pytest.raises(OSError):
        write_config_snapshot(snap, {"lr": 5e-5})

    assert snap.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob("*.tmp")) == []
