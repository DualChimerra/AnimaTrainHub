"""PP8 -- onnxruntime startup-time detection / install logic (mock subprocess).

Doesn't actually run pip / spawn nvidia-smi; uses monkeypatch to replace subprocess.run +
shutil.which and covers the install decision table.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from studio.services.runtime import onnxruntime as ors


# ---------------------------------------------------------------------------
# detect_cuda
# ---------------------------------------------------------------------------


def test_detect_cuda_no_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ors.shutil, "which", lambda _: None)
    res = ors.detect_cuda()
    assert res == {"available": False, "driver_version": None, "gpu_name": None}


def test_detect_cuda_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ors.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    fake = MagicMock(returncode=0, stdout="551.86, NVIDIA GeForce RTX 5090\n", stderr="")
    monkeypatch.setattr(ors.subprocess, "run", lambda *a, **k: fake)
    res = ors.detect_cuda()
    assert res == {
        "available": True,
        "driver_version": "551.86",
        "gpu_name": "NVIDIA GeForce RTX 5090",
    }


def test_detect_cuda_returncode_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ors.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    fake = MagicMock(returncode=9, stdout="", stderr="error")
    monkeypatch.setattr(ors.subprocess, "run", lambda *a, **k: fake)
    res = ors.detect_cuda()
    assert res["available"] is False


def test_detect_cuda_subprocess_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ors.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def _raise(*_a, **_k):
        raise OSError("permission denied")

    monkeypatch.setattr(ors.subprocess, "run", _raise)
    res = ors.detect_cuda()
    assert res["available"] is False


# ---------------------------------------------------------------------------
# _decide_target
# ---------------------------------------------------------------------------


def test_decide_target_explicit() -> None:
    assert ors._decide_target("gpu").startswith("onnxruntime-gpu")
    assert ors._decide_target("cpu").startswith("onnxruntime")
    assert "gpu" not in ors._decide_target("cpu")
    assert ors._decide_target("directml").startswith("onnxruntime-directml")


def test_decide_target_auto_with_gpu_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux + GPU present -> onnxruntime-gpu (native CUDA EP)."""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": True, "driver_version": "551.86", "gpu_name": "RTX 5090"},
    )
    assert ors._decide_target("auto").startswith("onnxruntime-gpu")


def test_decide_target_auto_with_gpu_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows + GPU present -> onnxruntime-directml (avoids CUDA dlopen compatibility issues)."""
    monkeypatch.setattr(ors.sys, "platform", "win32")
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": True, "driver_version": "551.86", "gpu_name": "RTX 5090"},
    )
    assert ors._decide_target("auto").startswith("onnxruntime-directml")


def test_decide_target_auto_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": False, "driver_version": None, "gpu_name": None},
    )
    res = ors._decide_target("auto")
    assert res.startswith("onnxruntime")
    # neither onnxruntime-gpu nor onnxruntime-directml
    assert "gpu" not in res
    assert "directml" not in res


def test_decide_target_invalid() -> None:
    with pytest.raises(ValueError):
        ors._decide_target("xpu")


# ---------------------------------------------------------------------------
# GPU version constraint branches by torch's CUDA major version (cu12 pins <1.26 / cu13 uses >=1.26)
# ---------------------------------------------------------------------------


def test_decide_target_gpu_cu12_when_torch_cu12(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch CUDA 12 -> onnxruntime-gpu>=1.20,<1.26 (ORT 1.26+ defaults to CUDA 13, must be pinned)."""
    monkeypatch.setattr(ors, "_resolve_cuda_major", lambda: 12)
    assert ors._decide_target("gpu") == "onnxruntime-gpu>=1.20,<1.26"


def test_decide_target_gpu_cu13_when_torch_cu13(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch CUDA 13 -> onnxruntime-gpu>=1.26 (CUDA 13 line)."""
    monkeypatch.setattr(ors, "_resolve_cuda_major", lambda: 13)
    assert ors._decide_target("gpu") == "onnxruntime-gpu>=1.26,<2.0"


def test_decide_target_gpu_defaults_cu12_when_major_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch's CUDA major can't be determined -> defaults to cu12 (= old behavior, zero regression)."""
    monkeypatch.setattr(ors, "_resolve_cuda_major", lambda: None)
    assert ors._decide_target("gpu") == "onnxruntime-gpu>=1.20,<1.26"


def test_decide_target_auto_linux_follows_torch_major(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Linux + GPU auto path also picks the spec by torch major (not just explicit gpu)."""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": True, "driver_version": "551.86", "gpu_name": "RTX 5090"},
    )
    monkeypatch.setattr(ors, "_resolve_cuda_major", lambda: 13)
    assert ors._decide_target("auto") == "onnxruntime-gpu>=1.26,<2.0"


# ---------------------------------------------------------------------------
# _resolve_cuda_major -- anchor for the ORT build's CUDA major version
# ---------------------------------------------------------------------------


def test_resolve_cuda_major_from_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch CUDA build's version.cuda determines the major version (the most authoritative anchor)."""
    import sys as _sys
    fake_torch = MagicMock()
    fake_torch.version.cuda = "12.8"
    monkeypatch.setitem(_sys.modules, "torch", fake_torch)
    assert ors._resolve_cuda_major() == 12
    fake_torch.version.cuda = "13.0"
    assert ors._resolve_cuda_major() == 13


def test_resolve_cuda_major_none_when_cpu_build_no_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch CPU build (version.cuda=None) + no GPU driver -> None (caller falls back to cu12)."""
    import sys as _sys
    fake_torch = MagicMock()
    fake_torch.version.cuda = None
    monkeypatch.setitem(_sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": False, "driver_version": None, "gpu_name": None},
    )
    assert ors._resolve_cuda_major() is None


def test_resolve_cuda_major_falls_back_to_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch CPU build but an NVIDIA driver is present -> recommend_cu_tag suggests cu126 -> major 12."""
    import sys as _sys
    fake_torch = MagicMock()
    fake_torch.version.cuda = None
    monkeypatch.setitem(_sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        ors, "detect_cuda",
        lambda: {"available": True, "driver_version": "551.86", "gpu_name": "RTX 5090"},
    )
    assert ors._resolve_cuda_major() == 12


# ---------------------------------------------------------------------------
# install_runtime -- mock pip
# ---------------------------------------------------------------------------


def test_install_runtime_runs_uninstall_then_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_pip(args):
        calls.append(args)
        return 0, "ok"

    monkeypatch.setattr(ors, "_pip", fake_pip)
    monkeypatch.setattr(
        ors, "_query_dist_info",
        lambda: ("onnxruntime-gpu", "1.20.0"),
    )
    # The GPU path chains into _install_cuda_runtime_wheels, whose internal _pip call count
    # varies by platform (Windows skips / Linux really installs). This test only verifies the
    # uninstall->install sequence, so we mock it out to stay platform-independent (same
    # treatment as the DirectML test).
    monkeypatch.setattr(
        ors, "_install_cuda_runtime_wheels",
        lambda: {"installed": [], "skipped": [], "platform_skip": True, "stdout": ""},
    )
    res = ors.install_runtime("gpu")
    assert len(calls) == 2
    assert calls[0][0] == "uninstall"
    # uninstall must cover all three mutually-exclusive packages, so no old package lingers
    assert "onnxruntime-gpu" in calls[0]
    assert "onnxruntime" in calls[0]
    assert "onnxruntime-directml" in calls[0]
    assert calls[1][0] == "install"
    assert any("onnxruntime-gpu" in a for a in calls[1])
    assert res["installed_pkg"] == "onnxruntime-gpu"
    assert res["installed_version"] == "1.20.0"
    # after install, restart_required must be returned to signal the frontend
    assert res["restart_required"] is True


def test_install_runtime_directml_skips_cuda_wheels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DirectML path should not trigger _install_cuda_runtime_wheels (DX12 doesn't need a CUDA runtime)."""
    calls: list[list[str]] = []

    def fake_pip(args, mirror=None):
        calls.append(args)
        return 0, "ok"

    monkeypatch.setattr(ors, "_pip", fake_pip)
    monkeypatch.setattr(
        ors, "_query_dist_info",
        lambda: ("onnxruntime-directml", "1.20.0"),
    )
    wheel_called = []
    monkeypatch.setattr(
        ors, "_install_cuda_runtime_wheels",
        lambda: wheel_called.append(True) or {"installed": [], "skipped": [], "platform_skip": True, "stdout": ""},
    )
    res = ors.install_runtime("directml")
    assert wheel_called == []  # should not be called at all
    assert res["cuda_runtime"] is None
    assert res["installed_pkg"] == "onnxruntime-directml"
    assert res["restart_required"] is True
    # uninstall must still cover all three
    assert "onnxruntime-directml" in calls[0]
    assert "onnxruntime-gpu" in calls[0]
    assert "onnxruntime" in calls[0]
    # the install command installs directml
    assert any("onnxruntime-directml" in a for a in calls[1])


def test_install_runtime_install_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # note: _pip supports a mirror= kwarg (retries on a mirror when the official source fails),
    # the mock must accept the same signature
    def fake_pip(args, mirror=None):
        if args[0] == "install":
            return 1, "ERROR: no matching distribution"
        return 0, ""

    monkeypatch.setattr(ors, "_pip", fake_pip)
    with pytest.raises(RuntimeError, match="Installing"):
        ors.install_runtime("gpu")


# ---------------------------------------------------------------------------
# current_runtime -- restart_required determination
# ---------------------------------------------------------------------------


def test_current_runtime_no_restart_when_directml_version_decoupled_from_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """regression: onnxruntime-directml's dist version (1.24.4) and the bundled onnxruntime
    core's ort.__version__ (1.27.0) are two independent version lines that naturally differ.
    As long as the Dml EP is loaded, it should not report needing a restart -- otherwise a
    DirectML user could restart any number of times and never clear the "restart Studio
    required" banner (which is exactly how the old logic, comparing version strings for
    staleness, produced false positives)."""
    monkeypatch.setattr(ors, "_query_dist_info", lambda: ("onnxruntime-directml", "1.24.4"))
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
    fake_ort.__version__ = "1.27.0"
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    rt = ors.current_runtime()
    assert rt["restart_required"] is False
    assert rt["directml_available"] is True


def test_current_runtime_flags_restart_when_directml_pkg_but_no_dml_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DirectML package installed but the process has no Dml EP (still running the old CPU package) -> restart required."""
    monkeypatch.setattr(ors, "_query_dist_info", lambda: ("onnxruntime-directml", "1.24.4"))
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    fake_ort.__version__ = "1.27.0"
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    rt = ors.current_runtime()
    assert rt["restart_required"] is True


def test_current_runtime_flags_restart_when_cpu_pkg_but_process_has_accel_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU package installed but the process still has a leftover accelerated EP (e.g. just switched from DirectML back to CPU without restarting) -> restart required."""
    monkeypatch.setattr(ors, "_query_dist_info", lambda: ("onnxruntime", "1.18.0"))
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
    fake_ort.__version__ = "1.27.0"
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    rt = ors.current_runtime()
    assert rt["restart_required"] is True


def test_current_runtime_flags_restart_when_gpu_pkg_but_no_cuda_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dist-info says onnxruntime-gpu but providers has no CUDA EP -> the process is still on the old CPU package."""
    monkeypatch.setattr(ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.20.0"))
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = ["AzureExecutionProvider", "CPUExecutionProvider"]
    fake_ort.__version__ = "1.20.0"
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    rt = ors.current_runtime()
    assert rt["restart_required"] is True


def test_current_runtime_no_restart_when_gpu_pkg_and_cuda_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.20.0"))
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    fake_ort.__version__ = "1.20.0"
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    rt = ors.current_runtime()
    assert rt["restart_required"] is False
    assert rt["cuda_available"] is True


# ---------------------------------------------------------------------------
# PP9.5 -- preload + cuda_load_error
# ---------------------------------------------------------------------------


def test_preload_skips_on_unsupported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not Linux / not Windows (e.g. macOS) -> skipped entirely."""
    monkeypatch.setattr(ors.sys, "platform", "darwin")
    res = ors._preload_torch_cuda_libs()
    assert res["platform_skip"] is True
    assert res["applied"] is False
    assert res["preloaded"] == []
    assert res["candidates"] == 0


def test_preload_windows_adds_torch_lib_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Windows + torch GPU build installed -> os.add_dll_directory(torch/lib) is called,
    and the returned preloaded list includes the directory path (so onnxruntime's dlopen can
    find cublasLt etc.)."""
    monkeypatch.setattr(ors.sys, "platform", "win32")

    # fake up torch.__file__ pointing at a directory that has a lib/ subdir
    torch_pkg = tmp_path / "torch"
    (torch_pkg / "lib").mkdir(parents=True)
    fake_torch = MagicMock()
    fake_torch.__file__ = str(torch_pkg / "__init__.py")

    def _import(name: str):
        if name == "torch":
            return fake_torch
        raise ImportError(name)

    monkeypatch.setattr(ors.importlib, "import_module", _import)
    # workaround: under venv/Scripts/python.exe, `import torch` goes through sys.modules,
    # but _add_torch_dll_dirs_windows uses a function-local `import torch` -- feed it
    # directly via monkeypatch sys.modules
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    called: list[str] = []
    monkeypatch.setattr(
        ors.os,
        "add_dll_directory",
        lambda d: called.append(d) or MagicMock(),
        raising=False,
    )

    res = ors._preload_torch_cuda_libs()
    assert res["applied"] is True
    assert res["platform_skip"] is False
    assert str(torch_pkg / "lib") in res["preloaded"]
    assert called == [str(torch_pkg / "lib")]


def test_preload_windows_noop_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows but the venv has no torch -> applied is still True (platform supported), candidates=0."""
    monkeypatch.setattr(ors.sys, "platform", "win32")
    monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)

    # make `import torch` fail: patching importlib.import_module isn't enough since the
    # function uses a literal import; a sys.modules sentinel + meta_path hack isn't clean
    # either. Simplest approach: construct an actual import error directly, since the
    # function body returns early as soon as the import fails anyway.
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    res = ors._preload_torch_cuda_libs()
    assert res["applied"] is True
    assert res["candidates"] == 0
    assert res["preloaded"] == []


def test_preload_noop_when_no_torch_nvidia_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No torch CUDA wheel in the venv -> preload doesn't error, candidates=0, preloaded is empty."""
    monkeypatch.setattr(ors.sys, "platform", "linux")

    def _no_pkg(_name: str):
        raise ImportError("not installed")

    monkeypatch.setattr(ors.importlib, "import_module", _no_pkg)
    res = ors._preload_torch_cuda_libs()
    assert res["applied"] is True
    assert res["candidates"] == 0
    assert res["preloaded"] == []


def test_preload_loads_present_libs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Simulate an nvidia.curand package: mod.__path__ has lib/libcurand.so.10 -> should be
    loaded via ctypes.CDLL, with RTLD_GLOBAL mode."""
    monkeypatch.setattr(ors.sys, "platform", "linux")

    # fake up nvidia.curand's __path__ + a lib/libcurand.so.10 file
    pkg_root = tmp_path / "nvidia_curand_pkg"
    (pkg_root / "lib").mkdir(parents=True)
    so = pkg_root / "lib" / "libcurand.so.10"
    so.write_bytes(b"")  # content doesn't matter, ctypes.CDLL is mocked

    fake_mod = MagicMock()
    fake_mod.__path__ = [str(pkg_root)]

    def _import(name: str):
        if name == "nvidia.curand":
            return fake_mod
        raise ImportError(name)

    monkeypatch.setattr(ors.importlib, "import_module", _import)

    cdll_calls: list[tuple[str, int]] = []

    def _fake_cdll(path, mode=0):
        cdll_calls.append((path, mode))
        return MagicMock()

    monkeypatch.setattr(ors.ctypes, "CDLL", _fake_cdll)

    res = ors._preload_torch_cuda_libs()
    assert str(so) in res["preloaded"]
    assert any(p == str(so) for p, _ in cdll_calls)
    # must use RTLD_GLOBAL, or a later onnxruntime dlopen won't see the symbols
    mode = next(m for p, m in cdll_calls if p == str(so))
    assert mode == ors.ctypes.RTLD_GLOBAL


def test_preload_loads_cu13_sonames_via_glob(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """glob doesn't hardcode the soname: libcurand.so.13 (cu13) is still loaded with
    RTLD_GLOBAL (verifies preload adapts to cu12 / cu13 without maintaining two soname tables)."""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    pkg_root = tmp_path / "nvidia_curand_pkg"
    (pkg_root / "lib").mkdir(parents=True)
    so = pkg_root / "lib" / "libcurand.so.13"
    so.write_bytes(b"")

    fake_mod = MagicMock()
    fake_mod.__path__ = [str(pkg_root)]

    def _import(name: str):
        if name == "nvidia.curand":
            return fake_mod
        raise ImportError(name)

    monkeypatch.setattr(ors.importlib, "import_module", _import)
    cdll_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        ors.ctypes, "CDLL",
        lambda path, mode=0: cdll_calls.append((path, mode)) or MagicMock(),
    )
    res = ors._preload_torch_cuda_libs()
    assert str(so) in res["preloaded"]
    mode = next(m for p, m in cdll_calls if p == str(so))
    assert mode == ors.ctypes.RTLD_GLOBAL


def test_record_cuda_load_error_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors, "_CUDA_LOAD_ERROR", None, raising=False)
    assert ors.get_cuda_load_error() is None
    ors.record_cuda_load_error("libcurand.so.10: cannot open shared object file")
    assert "libcurand" in (ors.get_cuda_load_error() or "")
    ors.record_cuda_load_error(None)
    assert ors.get_cuda_load_error() is None


# ---------------------------------------------------------------------------
# PP9.6 -- CUDA runtime wheels install / rollback
# ---------------------------------------------------------------------------


def test_install_cuda_runtime_wheels_skip_on_non_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors.sys, "platform", "win32")
    res = ors._install_cuda_runtime_wheels()
    assert res["platform_skip"] is True
    assert res["installed"] == []


def test_install_cuda_runtime_wheels_installs_missing_cu12(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux cu12: cuDNN already present -> skipped; the other 6 all install nvidia-*-cu12."""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    # cuDNN already installed (bundled with torch); nothing else is
    monkeypatch.setattr(
        ors,
        "_is_dist_installed",
        lambda p: p == ors._NVIDIA_CUDNN_WHEEL_CU12,
    )
    pip_calls: list[list[str]] = []

    def fake_pip(args):
        pip_calls.append(args)
        return 0, "Successfully installed nvidia-curand-cu12 ..."

    monkeypatch.setattr(ors, "_pip", fake_pip)
    res = ors._install_cuda_runtime_wheels(major=12)
    assert res["platform_skip"] is False
    assert res["cuda_major"] == 12
    assert ors._NVIDIA_CUDNN_WHEEL_CU12 in res["skipped"]
    assert ors._NVIDIA_CUDNN_WHEEL_CU12 not in res["installed"]
    # all 6 made it into the install args
    for pkg in ors._NVIDIA_CUDA_RUNTIME_WHEELS_CU12:
        assert pkg in res["installed"]
    # a single pip install call
    install_calls = [c for c in pip_calls if c[0] == "install"]
    assert len(install_calls) == 1


def test_install_cuda_runtime_wheels_cu13_uses_unversioned_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cu13: installs the unsuffixed nvidia-cuda-runtime / nvidia-cublas ... + nvidia-cudnn-cu13
    (the `-cu13`-suffixed packages on PyPI are deprecated empty placeholders and must not be
    installed)."""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(ors, "_is_dist_installed", lambda _p: False)
    monkeypatch.setattr(ors, "_pip", lambda args: (0, "ok"))
    res = ors._install_cuda_runtime_wheels(major=13)
    assert res["cuda_major"] == 13
    installed = res["installed"]
    assert "nvidia-cublas" in installed
    assert "nvidia-cuda-runtime" in installed
    assert ors._NVIDIA_CUDNN_WHEEL_CU13 in installed
    # cu13 must not use cu12-suffixed names, nor should the deprecated -cu13 runtime package appear
    assert not any(p.endswith("-cu12") for p in installed)
    assert "nvidia-cuda-runtime-cu13" not in installed


def test_install_cuda_runtime_wheels_noop_when_all_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(ors, "_is_dist_installed", lambda _p: True)
    pip_calls: list[list[str]] = []
    monkeypatch.setattr(
        ors, "_pip", lambda args: (pip_calls.append(args) or (0, "")),
    )
    res = ors._install_cuda_runtime_wheels()
    assert res["installed"] == []
    assert pip_calls == []  # everything already installed, pip isn't called


def test_install_cuda_runtime_wheels_rolls_back_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pip install fails -> everything queued for this attempt is uninstalled again, then raises. venv is not left polluted."""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(ors, "_is_dist_installed", lambda _p: False)
    pip_calls: list[list[str]] = []

    def fake_pip(args, mirror=None):
        pip_calls.append(args)
        if args[0] == "install":
            return 1, "ERROR: pip failed to resolve dependencies"
        return 0, "uninstalled"

    monkeypatch.setattr(ors, "_pip", fake_pip)
    with pytest.raises(RuntimeError, match="CUDA runtime wheels"):
        ors._install_cuda_runtime_wheels()
    # there must be one uninstall call to roll back
    assert any(c[0] == "uninstall" for c in pip_calls)


def test_install_runtime_gpu_path_calls_cuda_wheels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """install_runtime("gpu") must call _install_cuda_runtime_wheels after onnxruntime-gpu is installed."""
    monkeypatch.setattr(ors, "_pip", lambda _args: (0, "ok"))
    monkeypatch.setattr(
        ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.20.0"),
    )
    called: list[bool] = []

    def fake_install_wheels():
        called.append(True)
        return {"installed": ["nvidia-curand-cu12"], "skipped": [], "platform_skip": False, "stdout": ""}

    monkeypatch.setattr(ors, "_install_cuda_runtime_wheels", fake_install_wheels)
    res = ors.install_runtime("gpu")
    assert called == [True]
    assert res["cuda_runtime"]["installed"] == ["nvidia-curand-cu12"]


def test_install_runtime_cpu_path_skips_cuda_wheels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors, "_pip", lambda _args: (0, "ok"))
    monkeypatch.setattr(
        ors, "_query_dist_info", lambda: ("onnxruntime", "1.18.0"),
    )
    called: list[bool] = []
    monkeypatch.setattr(
        ors, "_install_cuda_runtime_wheels",
        lambda: (called.append(True), {"installed": []})[1],
    )
    res = ors.install_runtime("cpu")
    assert called == []  # the CPU path doesn't call it
    assert res["cuda_runtime"] is None


def test_install_runtime_does_not_fail_when_cuda_wheels_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA wheels install fails after onnxruntime-gpu succeeds -> doesn't raise, records an error for the UI to show."""
    monkeypatch.setattr(ors, "_pip", lambda _args: (0, "ok"))
    monkeypatch.setattr(
        ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.20.0"),
    )

    def boom():
        raise RuntimeError("pip resolver could not satisfy")

    monkeypatch.setattr(ors, "_install_cuda_runtime_wheels", boom)
    res = ors.install_runtime("gpu")
    # ort-gpu is already installed; doesn't raise
    assert res["installed_pkg"] == "onnxruntime-gpu"
    assert "error" in res["cuda_runtime"]
    assert "pip resolver" in res["cuda_runtime"]["error"]


def test_current_runtime_exposes_cuda_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ors, "_query_dist_info", lambda: ("onnxruntime-gpu", "1.20.0"))
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider", "CPUExecutionProvider"
    ]
    fake_ort.__version__ = "1.20.0"
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    ors.record_cuda_load_error("simulated dlopen failure")
    try:
        rt = ors.current_runtime()
        assert rt["cuda_load_error"] == "simulated dlopen failure"
        assert "preload" in rt
    finally:
        ors.record_cuda_load_error(None)


# ---------------------------------------------------------------------------
# PR-3 -- system CUDA detection: avoid overwriting the system cuBLAS and causing an ABI mismatch
# ---------------------------------------------------------------------------


def test_has_system_cuda_libs_via_cuda_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """CUDA_HOME + cuDNN also present on the system -> True."""
    fake_root = tmp_path / "fake-cuda"
    (fake_root / "lib64").mkdir(parents=True)
    monkeypatch.setenv("CUDA_HOME", str(fake_root))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(ors.os.path, "isdir", lambda p: p.endswith(str(fake_root / "lib64")))
    import ctypes.util as _cu  # noqa: PLC0415
    # cuDNN is on the system linker path
    monkeypatch.setattr(_cu, "find_library", lambda name: "libcudnn.so.9" if name == "cudnn" else None)
    assert ors._has_system_cuda_libs() is True


def test_has_system_cuda_libs_via_default_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/usr/local/cuda/lib64 + system cuDNN -> True."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(
        ors.os.path,
        "isdir",
        lambda p: p == "/usr/local/cuda/lib64",
    )
    import ctypes.util as _cu  # noqa: PLC0415
    monkeypatch.setattr(_cu, "find_library", lambda name: "libcudnn.so.9" if name == "cudnn" else None)
    assert ors._has_system_cuda_libs() is True


def test_has_system_cuda_libs_returns_false_when_no_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(ors.os.path, "isdir", lambda _p: False)
    import ctypes.util as _cu  # noqa: PLC0415
    monkeypatch.setattr(_cu, "find_library", lambda _name: None)
    assert ors._has_system_cuda_libs() is False


def test_has_system_cuda_libs_returns_false_when_cudnn_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Key fix regression: the system has a CUDA Toolkit (cuBLAS at /usr/local/cuda) but
    **no cuDNN installed** -- must return False so the torch wheel preload can fill in cuDNN
    as a fallback. Otherwise onnxruntime's dlopen of libcudnn.so.9 fails -> silently falls
    back to CPU (a user hit this in production on a cloud instance)."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(
        ors.os.path,
        "isdir",
        lambda p: p == "/usr/local/cuda/lib64",
    )
    import ctypes.util as _cu  # noqa: PLC0415
    # cuBLAS is on the system, cuDNN is not
    monkeypatch.setattr(_cu, "find_library", lambda name: "libcublas.so.12" if name == "cublas" else None)
    assert ors._has_system_cuda_libs() is False


def test_has_system_cuda_libs_returns_false_when_only_cublas_in_ld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No CUDA_HOME, no /usr/local/cuda, only cuBLAS on the linker path (installed via apt) +
    no cuDNN -> False (same as above: a partial system CUDA install still counts as incomplete)."""
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(ors.os.path, "isdir", lambda _p: False)
    import ctypes.util as _cu  # noqa: PLC0415
    monkeypatch.setattr(_cu, "find_library", lambda name: "libcublas.so.12" if name == "cublas" else None)
    assert ors._has_system_cuda_libs() is False


def test_preload_skips_when_system_cuda_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux + system CUDA -> skips preload, avoiding an ABI conflict between the torch wheel and the system cuBLAS."""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(ors, "_has_system_cuda_libs", lambda: True)
    res = ors._preload_torch_cuda_libs()
    assert res["system_cuda_skip"] is True
    assert res["applied"] is False
    assert res["preloaded"] == []
    assert res["candidates"] == 0


def test_preload_runs_when_system_cuda_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux + no system CUDA -> falls through to the original preload path (even with no torch wheel, applied is at least True)."""
    monkeypatch.setattr(ors.sys, "platform", "linux")
    monkeypatch.setattr(ors, "_has_system_cuda_libs", lambda: False)

    def _no_pkg(_name: str):
        raise ImportError("not installed")

    monkeypatch.setattr(ors.importlib, "import_module", _no_pkg)
    res = ors._preload_torch_cuda_libs()
    assert res["applied"] is True
    assert res["system_cuda_skip"] is False
