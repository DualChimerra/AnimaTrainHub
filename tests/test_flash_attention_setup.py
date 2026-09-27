from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock

import pytest

from studio.services.runtime import flash_attention as fa


# ---------------------------------------------------------------------------
# _parse_wheel
# ---------------------------------------------------------------------------


def test_parse_wheel_canonical() -> None:
    tags = fa._parse_wheel("flash_attn-2.8.3+cu130torch2.11-cp312-cp312-win_amd64.whl")
    assert tags == {
        "version": "2.8.3",
        "cuda": "cu130",
        "torch": "torch2.11",
        "python": "cp312",
        "platform": "win_amd64",
    }


def test_parse_wheel_linux_with_minor_torch() -> None:
    tags = fa._parse_wheel(
        "flash_attn-2.7.0+cu124torch2.4.0-cp310-cp310-linux_x86_64.whl"
    )
    assert tags is not None
    assert tags["cuda"] == "cu124"
    assert tags["torch"] == "torch2.4.0"
    assert tags["platform"] == "linux_x86_64"


@pytest.mark.parametrize("name", [
    "flash_attn-2.8.3.whl",
    "flash_attn-2.8.3+cu130torch2.11-cp312-cp312-macosx_arm64.whl",
    "totally-not-a-wheel.txt",
    "",
])
def test_parse_wheel_invalid(name: str) -> None:
    res = fa._parse_wheel(name)
    if name == "":
        assert res is None
    elif "macosx" in name:
        assert res is not None
    else:
        assert res is None


# ---------------------------------------------------------------------------
# _cuda_major
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tag,expected", [
    ("cu130", 13),
    ("cu128", 12),
    ("cu124", 12),
    ("cu99", 9),
    ("invalid", -1),
    ("", -1),
])
def test_cuda_major(tag: str, expected: int) -> None:
    assert fa._cuda_major(tag) == expected


# ---------------------------------------------------------------------------
# detect_env
# ---------------------------------------------------------------------------


def _patch_torch(monkeypatch: pytest.MonkeyPatch, version: str | None) -> None:
    import sys
    import types
    if version is None:
        monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]
        return
    fake = types.ModuleType("torch")
    fake.__version__ = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_detect_env_no_nvidia_smi_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise FileNotFoundError("no nvidia-smi")
    monkeypatch.setattr(fa.subprocess, "run", boom)
    _patch_torch(monkeypatch, None)
    env = fa.detect_env()
    assert env["cuda_tag"] is None
    assert env["cuda_ver"] is None
    assert env["driver_cuda_ver"] is None
    assert env["torch_tag"] is None
    assert env["python_tag"].startswith("cp")


def test_detect_env_prefers_torch_cu_tag_over_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock(returncode=0, stdout="CUDA Version: 13.0 \n", stderr="")
    monkeypatch.setattr(fa.subprocess, "run", lambda *a, **k: fake)
    _patch_torch(monkeypatch, "2.11.0+cu128")
    env = fa.detect_env()
    assert env["cuda_tag"] == "cu128"
    assert env["cuda_ver"] == "12.8"
    assert env["driver_cuda_ver"] == "13.0"
    assert env["torch_tag"] == "torch2.11"
    assert env["torch_ver"] == "2.11.0+cu128"


def test_detect_env_falls_back_to_nvidia_smi_when_torch_has_no_cu_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock(returncode=0, stdout="CUDA Version: 12.8 \n", stderr="")
    monkeypatch.setattr(fa.subprocess, "run", lambda *a, **k: fake)
    _patch_torch(monkeypatch, "2.11.0")
    env = fa.detect_env()
    assert env["cuda_tag"] == "cu128"
    assert env["cuda_ver"] == "12.8"
    assert env["driver_cuda_ver"] == "12.8"
    assert env["torch_tag"] == "torch2.11"


def test_detect_env_no_torch_falls_back_to_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock(returncode=0, stdout="CUDA Version: 12.8 \n", stderr="")
    monkeypatch.setattr(fa.subprocess, "run", lambda *a, **k: fake)
    _patch_torch(monkeypatch, None)
    env = fa.detect_env()
    assert env["cuda_tag"] == "cu128"
    assert env["cuda_ver"] == "12.8"
    assert env["driver_cuda_ver"] == "12.8"
    assert env["torch_tag"] is None


# ---------------------------------------------------------------------------
# find_candidates
# ---------------------------------------------------------------------------


def _mock_releases(monkeypatch: pytest.MonkeyPatch, payload: Any) -> None:
    if isinstance(payload, BaseException):
        def _raise(*_a, **_k):
            raise payload
        monkeypatch.setattr(fa.urllib.request, "urlopen", _raise)
        return
    body = json.dumps(payload).encode("utf-8")
    monkeypatch.setattr(
        fa.urllib.request,
        "urlopen",
        lambda *_a, **_k: io.BytesIO(body),
    )


def test_find_candidates_no_platform_returns_empty() -> None:
    candidates, err = fa.find_candidates({"platform": None})
    assert candidates == []
    assert err is None


def test_find_candidates_rate_limited_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_releases(monkeypatch, {"message": "API rate limit exceeded for ..."})
    candidates, err = fa.find_candidates({
        "platform": "win_amd64", "torch_tag": "torch2.5", "cuda_tag": "cu128",
        "python_tag": "cp311",
    })
    assert candidates == []
    assert err is not None and "rate limit" in err


def test_find_candidates_network_error_returns_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_releases(monkeypatch, urllib.error.URLError("getaddrinfo failed"))
    candidates, err = fa.find_candidates({
        "platform": "linux_x86_64", "torch_tag": "torch2.5", "cuda_tag": "cu128",
        "python_tag": "cp311",
    })
    assert candidates == []
    assert err is not None and "getaddrinfo" in err


def test_find_candidates_filters_platform_and_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [{
        "assets": [
            {
                "name": "flash_attn-2.8+cu128torch2.5-cp311-cp311-linux_x86_64.whl",
                "browser_download_url": "https://x/linux.whl",
            },
            {
                "name": "flash_attn-2.8+cu128torch2.4-cp311-cp311-win_amd64.whl",
                "browser_download_url": "https://x/torch24.whl",
            },
            {
                "name": "flash_attn-2.8+cu128torch2.5-cp311-cp311-win_amd64.whl",
                "browser_download_url": "https://x/perfect.whl",
            },
        ],
    }]
    _mock_releases(monkeypatch, payload)
    candidates, err = fa.find_candidates({
        "platform": "win_amd64", "torch_tag": "torch2.5",
        "cuda_tag": "cu128", "python_tag": "cp311",
    })
    assert err is None
    assert len(candidates) == 1
    assert candidates[0]["name"].startswith("flash_attn-2.8+cu128torch2.5-cp311-cp311-win_amd64")
    assert candidates[0]["usable"] is True


def test_find_candidates_python_mismatch_marks_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [{
        "assets": [{
            "name": "flash_attn-2.8+cu128torch2.5-cp310-cp310-win_amd64.whl",
            "browser_download_url": "https://x/cp310.whl",
        }],
    }]
    _mock_releases(monkeypatch, payload)
    env = {
        "platform": "win_amd64", "torch_tag": "torch2.5",
        "cuda_tag": "cu128", "python_tag": "cp311",
    }
    candidates, err = fa.find_candidates(env)
    assert err is None
    assert len(candidates) == 1
    assert candidates[0]["usable"] is False
    assert any("Python incompatible" in n for n in candidates[0]["notes"])
    assert fa.find_best_wheel(env) is None


def test_find_candidates_cuda_minor_diff_scores_lower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [{
        "assets": [
            {
                "name": "flash_attn-2.8+cu124torch2.5-cp311-cp311-win_amd64.whl",
                "browser_download_url": "https://x/cu124.whl",
            },
            {
                "name": "flash_attn-2.8+cu128torch2.5-cp311-cp311-win_amd64.whl",
                "browser_download_url": "https://x/cu128.whl",
            },
        ],
    }]
    _mock_releases(monkeypatch, payload)
    env = {
        "platform": "win_amd64", "torch_tag": "torch2.5",
        "cuda_tag": "cu128", "python_tag": "cp311",
    }
    candidates, err = fa.find_candidates(env)
    assert err is None
    assert candidates[0]["name"].startswith("flash_attn-2.8+cu128")
    assert candidates[0]["score"] > candidates[1]["score"]
    assert fa.find_best_wheel(env) == "https://x/cu128.whl"


def test_find_candidates_cuda_major_diff_negative_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [{
        "assets": [{
            "name": "flash_attn-2.8+cu118torch2.5-cp311-cp311-win_amd64.whl",
            "browser_download_url": "https://x/cu118.whl",
        }],
    }]
    _mock_releases(monkeypatch, payload)
    env = {
        "platform": "win_amd64", "torch_tag": "torch2.5",
        "cuda_tag": "cu130", "python_tag": "cp311",
    }
    candidates, err = fa.find_candidates(env)
    assert err is None
    assert len(candidates) == 1
    assert candidates[0]["score"] == 15
    assert any("CUDA major version differs" in n for n in candidates[0]["notes"])


def test_find_candidates_prefers_newest_version_on_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [{
        "assets": [
            {"name": "flash_attn-2.6.3+cu128torch2.11-cp312-cp312-linux_x86_64.whl",
             "browser_download_url": "https://x/2.6.3.whl"},
            {"name": "flash_attn-2.7.4+cu128torch2.11-cp312-cp312-linux_x86_64.whl",
             "browser_download_url": "https://x/2.7.4.whl"},
            {"name": "flash_attn-2.8.3+cu128torch2.11-cp312-cp312-linux_x86_64.whl",
             "browser_download_url": "https://x/2.8.3.whl"},
        ],
    }]
    _mock_releases(monkeypatch, payload)
    env = {
        "platform": "linux_x86_64", "torch_tag": "torch2.11",
        "cuda_tag": "cu128", "python_tag": "cp312",
    }
    candidates, err = fa.find_candidates(env)
    assert err is None
    assert [c["version"] for c in candidates] == ["2.8.3", "2.7.4", "2.6.3"]
    assert fa.find_best_wheel(env) == "https://x/2.8.3.whl"


# ---------------------------------------------------------------------------
# current_status / install
# ---------------------------------------------------------------------------


def test_current_status_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_pkg):
        raise fa.importlib.metadata.PackageNotFoundError
    monkeypatch.setattr(fa.importlib.metadata, "version", _raise)
    s = fa.current_status()
    assert s == {"installed": False, "version": None}


def test_current_status_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fa.importlib.metadata, "version", lambda _: "2.8.3")
    s = fa.current_status()
    assert s == {"installed": True, "version": "2.8.3"}


def test_install_no_url_unsupported_platform_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fa, "detect_env",
        lambda: {"platform": None, "torch_tag": "torch2.5", "python_tag": "cp311"},
    )
    with pytest.raises(RuntimeError, match="Unsupported platform"):
        fa.install()


def test_install_no_url_no_torch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fa, "detect_env",
        lambda: {"platform": "win_amd64", "torch_tag": None, "python_tag": "cp311"},
    )
    with pytest.raises(RuntimeError, match="PyTorch was not detected"):
        fa.install()


def test_install_no_candidate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fa, "detect_env",
        lambda: {
            "platform": "win_amd64", "torch_tag": "torch2.5",
            "python_tag": "cp311", "cuda_tag": "cu128",
        },
    )
    monkeypatch.setattr(fa, "find_best_wheel", lambda _env: None)
    with pytest.raises(RuntimeError, match="No suitable wheel"):
        fa.install()


def test_install_pip_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock(returncode=1, stdout="", stderr="ERROR: some pip failure")
    monkeypatch.setattr(fa.subprocess, "run", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="pip install failed"):
        fa.install("https://x/wheel.whl")


def test_install_success_returns_status(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock(returncode=0, stdout="Successfully installed flash_attn-2.8.3\n", stderr="")
    monkeypatch.setattr(fa.subprocess, "run", lambda *a, **k: fake)
    monkeypatch.setattr(fa.importlib.metadata, "version", lambda _: "2.8.3")
    res = fa.install("https://x/wheel.whl")
    assert res["installed"] is True
    assert res["version"] == "2.8.3"
    assert res["url"] == "https://x/wheel.whl"
    assert res["restart_required"] is True
    assert "Successfully installed" in res["stdout_tail"]
