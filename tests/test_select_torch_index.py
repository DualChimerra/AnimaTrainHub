"""PR-S1a -- tools/select_torch_index.py bootstrap helper.

Doesn't actually run nvidia-smi; uses monkeypatch to simulate it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HELPER_PATH = _REPO_ROOT / "tools" / "select_torch_index.py"


@pytest.fixture
def helper_module():
    """Manually load the helper by file path -- tools/ isn't a package (no __init__.py)."""
    spec = importlib.util.spec_from_file_location(
        "_select_torch_index_for_test", _HELPER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# select_index_url: driver major version -> URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("major,expected_tag", [
    (596, "cu128"),  # driver a user actually tested
    (570, "cu128"),
    (555, "cu128"),  # boundary
    (554, "cu126"),  # just below the boundary
    (550, "cu126"),
    (549, "cu124"),
    (545, "cu124"),
    (470, "cu118"),
    (469, None),     # too old
    (None, None),
])
def test_select_index_url(helper_module, major, expected_tag) -> None:
    res = helper_module.select_index_url(major)
    if expected_tag is None:
        assert res is None
    else:
        assert res == f"https://download.pytorch.org/whl/{expected_tag}"


# ---------------------------------------------------------------------------
# detect_driver_major: nvidia-smi parsing
# ---------------------------------------------------------------------------


def test_detect_driver_major_no_nvidia_smi(
    helper_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a, **_k):
        raise FileNotFoundError("no nvidia-smi")
    monkeypatch.setattr(helper_module.subprocess, "run", boom)
    assert helper_module.detect_driver_major() is None


def test_detect_driver_major_returncode_nonzero(
    helper_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        helper_module.subprocess, "run",
        lambda *a, **k: MagicMock(returncode=9, stdout="", stderr="error"),
    )
    assert helper_module.detect_driver_major() is None


def test_detect_driver_major_parses_output(
    helper_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        helper_module.subprocess, "run",
        lambda *a, **k: MagicMock(returncode=0, stdout="596.36\n", stderr=""),
    )
    assert helper_module.detect_driver_major() == 596


def test_detect_driver_major_garbage_output(
    helper_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        helper_module.subprocess, "run",
        lambda *a, **k: MagicMock(returncode=0, stdout="not a version\n", stderr=""),
    )
    assert helper_module.detect_driver_major() is None


def test_detect_driver_major_first_line_when_multi_gpu(
    helper_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """nvidia-smi prints one line per card on multi-GPU machines; using the first is fine (driver version matches across cards on one machine)."""
    monkeypatch.setattr(
        helper_module.subprocess, "run",
        lambda *a, **k: MagicMock(returncode=0, stdout="555.86\n555.86\n", stderr=""),
    )
    assert helper_module.detect_driver_major() == 555


# ---------------------------------------------------------------------------
# main(): end-to-end + output format
# ---------------------------------------------------------------------------


def test_main_outputs_url_no_trailing_newline(
    helper_module, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """shell captures output with $() / for /f; no trailing newline, to avoid ambiguity."""
    monkeypatch.setattr(helper_module, "detect_driver_major", lambda: 555)
    rc = helper_module.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert out == "https://download.pytorch.org/whl/cu128"
    assert not out.endswith("\n")


def test_main_outputs_nothing_when_no_driver(
    helper_module, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(helper_module, "detect_driver_major", lambda: None)
    rc = helper_module.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_outputs_nothing_when_driver_too_old(
    helper_module, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Old driver (< 470) -> silent -> caller falls back to the PyPI default (CPU torch)."""
    monkeypatch.setattr(helper_module, "detect_driver_major", lambda: 460)
    rc = helper_module.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# kept consistent with studio.services.torch_setup's mapping
# ---------------------------------------------------------------------------


def test_mapping_matches_torch_setup_canonical(helper_module) -> None:
    """The helper's _DRIVER_TO_CU must match torch_setup's source (to avoid drift)."""
    from studio.services.runtime.torch import _DRIVER_TO_BEST_CU
    canonical = [(int(thresh), tag) for thresh, tag in _DRIVER_TO_BEST_CU]
    assert helper_module._DRIVER_TO_CU == canonical
