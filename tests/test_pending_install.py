"""PR-S2.1 — pending_install marker read/write + apply_pending scheduling.

Doesn't actually run pip: torch_setup.reinstall is monkeypatched with a fake
implementation to verify the flow.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.services.runtime import pending_install


@pytest.fixture
def isolated_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Each test gets its own marker path, to avoid cross-contamination."""
    marker = tmp_path / ".pending-pip-install.json"
    monkeypatch.setattr(pending_install, "STUDIO_DATA", tmp_path)
    monkeypatch.setattr(pending_install, "PENDING_MARKER", marker)
    return marker


def test_register_writes_marker(isolated_marker: Path) -> None:
    pending_install.register_torch_reinstall("cu128")
    assert isolated_marker.exists()
    data = pending_install.read_pending()
    assert data == {"kind": "torch", "target": "cu128"}


def test_register_overwrites_previous(isolated_marker: Path) -> None:
    pending_install.register_torch_reinstall("cu118")
    pending_install.register_torch_reinstall("cu128")  # overwrite
    assert pending_install.read_pending()["target"] == "cu128"


def test_read_pending_returns_none_when_missing(isolated_marker: Path) -> None:
    assert pending_install.read_pending() is None


def test_read_pending_returns_none_on_corrupt_marker(isolated_marker: Path) -> None:
    isolated_marker.write_text("not-json{", encoding="utf-8")
    assert pending_install.read_pending() is None


def test_clear_pending_removes_marker(isolated_marker: Path) -> None:
    pending_install.register_torch_reinstall("cu128")
    pending_install.clear_pending()
    assert not isolated_marker.exists()
    assert pending_install.read_pending() is None


def test_clear_pending_no_marker_no_error(isolated_marker: Path) -> None:
    """clear should also succeed silently when there's no marker."""
    pending_install.clear_pending()
    assert not isolated_marker.exists()


def test_apply_pending_no_marker_is_noop(
    isolated_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No pending → should not trigger torch_setup.reinstall."""
    from studio.services.runtime import torch as torch_setup
    called: list[str] = []
    monkeypatch.setattr(torch_setup, "reinstall", lambda t: called.append(t))
    pending_install.apply_pending()
    assert called == []


def test_apply_pending_runs_torch_reinstall_and_clears(
    isolated_marker: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """pending=torch → calls reinstall + clears the marker."""
    pending_install.register_torch_reinstall("cu128")
    from studio.services.runtime import torch as torch_setup
    captured: list[str] = []

    def fake_reinstall(target, *, stream=False):
        captured.append(target)
        return {
            "target": target, "tag": "cu128",
            "index_url": "https://x/cu128",
            "version": "2.5.0+cu128",
            "stdout_tail": "ok",
            "restart_required": True,
        }

    monkeypatch.setattr(torch_setup, "reinstall", fake_reinstall)
    pending_install.apply_pending()
    assert captured == ["cu128"]
    assert not isolated_marker.exists()  # cleared after success
    out = capsys.readouterr().out
    assert "torch reinstall complete" in out


def test_apply_pending_keeps_marker_on_failure(
    isolated_marker: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """reinstall raises RuntimeError → marker is kept, retried on next start."""
    pending_install.register_torch_reinstall("cu128")
    from studio.services.runtime import torch as torch_setup

    def fake_reinstall(_t, *, stream=False):
        raise RuntimeError("network failed")

    monkeypatch.setattr(torch_setup, "reinstall", fake_reinstall)
    pending_install.apply_pending()
    # marker is kept
    assert isolated_marker.exists()
    assert pending_install.read_pending()["target"] == "cu128"
    err = capsys.readouterr().err
    assert "torch reinstall failed" in err
    assert "network failed" in err


def test_apply_pending_unknown_kind_clears_marker(
    isolated_marker: Path, capsys
) -> None:
    """Unknown kind → warn + clear the marker (avoids getting stuck forever)."""
    isolated_marker.write_text(
        '{"kind": "modelscope", "target": "auto"}', encoding="utf-8"
    )
    pending_install.apply_pending()
    assert not isolated_marker.exists()
    err = capsys.readouterr().err
    assert "unknown" in err
