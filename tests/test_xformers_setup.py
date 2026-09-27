"""xformers_setup unit tests -- coverage for current_status / install / _torch_cuda_index paths.

Follows the style of test_flash_attention_setup.py but simpler (the xformers service is
much simpler than flash_attention).
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from studio.services.runtime import xformers as xs


# ---------------------------------------------------------------------------
# current_status
# ---------------------------------------------------------------------------


def test_current_status_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """importlib.metadata can't find xformers -> installed=False."""
    import importlib.metadata as md
    def boom(name: str) -> str:
        raise md.PackageNotFoundError(name)
    monkeypatch.setattr(xs.importlib.metadata, "version", boom)
    s = xs.current_status()
    assert s == {"installed": False, "version": None}


def test_current_status_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xs.importlib.metadata, "version", lambda _: "0.0.28")
    s = xs.current_status()
    assert s == {"installed": True, "version": "0.0.28"}


# ---------------------------------------------------------------------------
# _torch_cuda_index -- gets the ABI tag from torch the same way as flash_attention_setup.detect_env
# ---------------------------------------------------------------------------


def _patch_torch(monkeypatch: pytest.MonkeyPatch, version: str | None) -> None:
    """Inject / remove a fake torch module (version=None simulates torch not being installed)."""
    if version is None:
        monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]
        return
    fake = types.ModuleType("torch")
    fake.__version__ = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_torch_cuda_index_from_torch_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch.__version__='2.11.0+cu128' -> cu128 index URL."""
    _patch_torch(monkeypatch, "2.11.0+cu128")
    assert xs._torch_cuda_index() == "https://download.pytorch.org/whl/cu128"


def test_torch_cuda_index_no_cu_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU-only torch (no +cu) -> None; caller falls back to the PyPI default."""
    _patch_torch(monkeypatch, "2.11.0")
    assert xs._torch_cuda_index() is None


def test_torch_cuda_index_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_torch(monkeypatch, None)
    assert xs._torch_cuda_index() is None


# ---------------------------------------------------------------------------
# disable_triton_probe -- short-circuits triton probing via subprocess env injection
# ---------------------------------------------------------------------------


def test_triton_probe_disabled() -> None:
    """Unconditionally sets XFORMERS_FORCE_DISABLE_TRITON=1 (the app's xformers path doesn't use triton)."""
    env: dict[str, str] = {}
    xs.disable_triton_probe(env)
    assert env == {"XFORMERS_FORCE_DISABLE_TRITON": "1"}


def test_triton_probe_respects_explicit_value() -> None:
    """If the user has explicitly set it (e.g. =0 to force probing) -> setdefault doesn't override it."""
    env = {"XFORMERS_FORCE_DISABLE_TRITON": "0"}
    xs.disable_triton_probe(env)
    assert env == {"XFORMERS_FORCE_DISABLE_TRITON": "0"}


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def _make_pip_result(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_install_success_with_torch_cu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Success path: cmd contains --index-url cu_tag; returns status + restart_required."""
    _patch_torch(monkeypatch, "2.11.0+cu128")
    captured: list[list[str]] = []
    def fake_run(cmd, **_kw):
        captured.append(cmd)
        return _make_pip_result(0, stdout="Successfully installed xformers-0.0.28")
    monkeypatch.setattr(xs.subprocess, "run", fake_run)
    monkeypatch.setattr(xs.importlib.metadata, "version", lambda _: "0.0.28")

    result = xs.install()

    assert result["installed"] is True
    assert result["version"] == "0.0.28"
    assert result["restart_required"] is True
    assert "Successfully installed" in result["stdout_tail"]
    # cmd contains --index-url cu128
    assert "--index-url" in captured[0]
    idx = captured[0].index("--index-url")
    assert captured[0][idx + 1] == "https://download.pytorch.org/whl/cu128"


def test_install_no_torch_falls_back_to_pypi(monkeypatch: pytest.MonkeyPatch) -> None:
    """No torch / no cu suffix -> --index-url is not passed, falls back to the PyPI default."""
    _patch_torch(monkeypatch, None)
    captured: list[list[str]] = []
    def fake_run(cmd, **_kw):
        captured.append(cmd)
        return _make_pip_result(0, stdout="Successfully installed xformers-0.0.27")
    monkeypatch.setattr(xs.subprocess, "run", fake_run)
    monkeypatch.setattr(xs.importlib.metadata, "version", lambda _: "0.0.27")

    result = xs.install()

    assert result["installed"] is True
    assert "--index-url" not in captured[0]


def test_install_pip_failure_raises_with_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """pip exit != 0 -> RuntimeError contains the tail of stderr."""
    _patch_torch(monkeypatch, "2.11.0+cu128")
    monkeypatch.setattr(
        xs.subprocess, "run",
        lambda *a, **k: _make_pip_result(1, stderr="ERROR: No matching distribution found"),
    )
    with pytest.raises(RuntimeError) as exc_info:
        xs.install()
    assert "No matching distribution found" in str(exc_info.value)
    assert "exit 1" in str(exc_info.value)


def test_install_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_torch(monkeypatch, "2.11.0+cu128")
    def boom(*_a, **_k):
        raise xs.subprocess.TimeoutExpired(cmd="pip", timeout=600)
    monkeypatch.setattr(xs.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="timed out"):
        xs.install()
