"""onnxruntime runtime detection / install (driven by the Settings page).

onnxruntime (CPU) and onnxruntime-gpu share the same import name and are
**mutually exclusive** - they can't be installed together. CUDA versions
differ across machines, so requirements.txt doesn't pin one. Installation is
triggered by the user from Settings -> ONNX Runtime, matching the xformers /
flash-attention pattern - users who don't tag aren't blocked by a pip install
at startup.

Main entry points:
    install_runtime(target) - called by the Settings page's "Install / reinstall as X" button; synchronous pip
    current_runtime()       - Settings page status display / cli.py startup check
    detect_cuda()           - nvidia-smi probe

Conventions:
- onnxruntime-gpu's version constraint is split by **CUDA major version**
  (cu12/cu13), anchored on _resolve_cuda_major() (= the major of
  torch.version.cuda): the ORT build must match torch's major version, or a
  mixed cu12/cu13 ABI in the same process breaks (ORT 1.26+ defaults to CUDA
  13, and the old `>=1.20` constraint would pull it in and crash on
  libcudart.so.13). cu12 is pinned to `>=1.20,<1.26`, cu13 to `>=1.26`; the
  preloaded soname set and the supplementary runtime wheels also pick the
  cu12 (`nvidia-*-cu12`) or cu13 (unsuffixed) set by the same major version.
- "Wrong CUDA version installed" shows up as `import onnxruntime` succeeding
  but `CUDAExecutionProvider` missing from providers; we don't auto-reinstall
  (may be intentional), the UI offers a manual button + warning instead.
- Package installs go through `subprocess.run([sys.executable, "-m", "pip", ...])`,
  never the internal pip API.

PP9.5 - CUDA shared library preload + session-creation fallback:
- onnxruntime-gpu doesn't bundle the CUDA runtime .so files (libcurand /
  libcublas / libcudnn ...). Common Linux failure: `get_available_providers()`
  reports the CUDA EP as available, but creating a session fails to dlopen
  `libcurand.so.10: cannot open shared object file`.
- Fix: at module top level, before `import onnxruntime`, preload torch's own
  `nvidia/*/lib/*.so` (PyTorch installs the nvidia-* wheels there) with
  ctypes RTLD_GLOBAL. Nobody does this in the ComfyUI / WD14 ecosystem, but
  it's the cheapest general fix.
- If it still fails, wd14_tagger falls back to CPU; this module stashes the
  reason via record_cuda_load_error for the UI to show.
"""
from __future__ import annotations

import ctypes
import importlib
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

GPU_PACKAGE = "onnxruntime-gpu"
CPU_PACKAGE = "onnxruntime"
# onnxruntime-directml gives Windows users a DX12 backend, sidestepping
# CUDA/cuDNN ABI compatibility issues (issue #231: RTX 5090 + onnxruntime-gpu
# silently falling back to CPU). Works on any DX12-capable GPU (NVIDIA / AMD /
# Intel). PyPI only ships a wheel for Windows.
DIRECTML_PACKAGE = "onnxruntime-directml"
CPU_VERSION_SPEC = ">=1.16"
DIRECTML_VERSION_SPEC = ">=1.20"
# All three packages share the import name `onnxruntime` and are mutually
# exclusive; the other two must be uninstalled before switching.
_MUTUALLY_EXCLUSIVE_PACKAGES: tuple[str, ...] = (GPU_PACKAGE, CPU_PACKAGE, DIRECTML_PACKAGE)

# onnxruntime-gpu's version constraint is split by **CUDA major version** -
# the ORT build's CUDA major must match torch's (mixing cu12/cu13 ABIs in the
# same process breaks), hence anchoring on _resolve_cuda_major().
# Starting with ORT 1.26.0 the default PyPI wheel switched to CUDA 13, and
# 1.27.0 dropped CUDA 12 builds entirely; the old `>=1.20` constraint now
# resolves to 1.27 (CUDA 13), which doesn't match this project's cu128 torch +
# cu12 preload/wheels -> `import onnxruntime` crashes on libcudart.so.13 (see
# _resolve_cuda_major).
GPU_VERSION_SPEC_CU12 = ">=1.20,<1.26"  # ORT 1.20-1.25 default to CUDA 12.x
GPU_VERSION_SPEC_CU13 = ">=1.26,<2.0"   # ORT 1.26+ CUDA 13
# Default when torch's CUDA major can't be determined: cu12 (= old behavior, zero regression).
DEFAULT_CUDA_MAJOR = 12

# PP9.6 - the onnxruntime-gpu wheel doesn't bundle the CUDA runtime .so files
# (libcurand / libcublas / ...), so dlopen fails outright on machines without
# a system CUDA install. We rely on the nvidia-* wheels on PyPI to install
# them into the venv's site-packages/nvidia/*/lib/, paired with this module's
# top-level RTLD_GLOBAL preload so onnxruntime's later dlopen finds the
# symbols.
#
# Notes:
# - **cuDNN wheel excluded by default**: torch's GPU build already installs
#   and pins it; `pip install nvidia-cudnn-cu12` without a version would
#   upgrade it to latest and break torch. Only install it if it's completely
#   missing.
# - These wheels are manylinux-only; unavailable on Windows / macOS, so the
#   install function returns early there. On Windows the correct path is
#   installing the CUDA Toolkit + cuDNN system-wide.
# CUDA 12 and CUDA 13 nvidia wheels have different names: the CUDA 13
# `nvidia-*-cu13` packages on PyPI are deprecated empty placeholders; the real
# packages are unsuffixed (`nvidia-cuda-runtime` / `nvidia-cublas` / ...);
# only cuDNN still uses the `-cu13` suffix. Both sets are picked by
# _resolve_cuda_major().
_NVIDIA_CUDA_RUNTIME_WHEELS_CU12: tuple[str, ...] = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusolver-cu12",
)
_NVIDIA_CUDNN_WHEEL_CU12 = "nvidia-cudnn-cu12"
_NVIDIA_CUDA_RUNTIME_WHEELS_CU13: tuple[str, ...] = (
    # cu13's runtime/math libs dropped the suffix (`-cu13` is an empty placeholder on PyPI)
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvrtc",
    "nvidia-cublas",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-cusparse",
    "nvidia-cusolver",
)
_NVIDIA_CUDNN_WHEEL_CU13 = "nvidia-cudnn-cu13"

# The nvidia-* wheels torch's GPU build pulls in install to
# site-packages/nvidia/<sub>/lib/. We preload every lib*.so* under each
# subpackage's lib/ with RTLD_GLOBAL - no hardcoded soname, so it adapts
# automatically to cu12 (libcudart.so.12 / libcudnn.so.9) and cu13 (.so.13 /
# libcudnn.so.10). The intent is simply "load the full set of CUDA .so files
# torch ships"; torch and ORT are always the same major (guaranteed by
# _decide_target), so they never cross major versions. cublasLt lives in the
# nvidia.cublas package, picked up by the same glob.
_TORCH_NVIDIA_LIB_PKGS_LINUX: tuple[str, ...] = (
    "nvidia.cuda_runtime",
    "nvidia.cuda_nvrtc",
    "nvidia.cublas",
    "nvidia.cufft",
    "nvidia.curand",
    "nvidia.cusparse",
    "nvidia.cusolver",
    "nvidia.cudnn",
)


def _cuda_wheels_for(major: Optional[int]) -> tuple[tuple[str, ...], str]:
    """Return (runtime_wheels, cudnn_wheel) for a CUDA major version. None / 12 -> cu12; 13 -> cu13."""
    if major == 13:
        return _NVIDIA_CUDA_RUNTIME_WHEELS_CU13, _NVIDIA_CUDNN_WHEEL_CU13
    return _NVIDIA_CUDA_RUNTIME_WHEELS_CU12, _NVIDIA_CUDNN_WHEEL_CU12


def _gpu_version_spec_for(major: Optional[int]) -> str:
    """Return the pip version constraint string for onnxruntime-gpu, given a CUDA major version."""
    if major == 13:
        return f"{GPU_PACKAGE}{GPU_VERSION_SPEC_CU13}"
    return f"{GPU_PACKAGE}{GPU_VERSION_SPEC_CU12}"


# ---------------------------------------------------------------------------
# dist-info probing (does not import the .pyd)
# ---------------------------------------------------------------------------


def _query_dist_info() -> tuple[Optional[str], Optional[str]]:
    """Read the install status of the three mutually-exclusive packages from dist-info. Returns (pkg_name, version)."""
    try:
        from importlib.metadata import PackageNotFoundError, version as _ver
        for pkg in _MUTUALLY_EXCLUSIVE_PACKAGES:
            try:
                return pkg, _ver(pkg)
            except PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return None, None


# ---------------------------------------------------------------------------
# CUDA shared library preload (PP9.5)
# ---------------------------------------------------------------------------


_PRELOAD_RESULT: Optional[dict[str, Any]] = None
_CUDA_LOAD_ERROR: Optional[str] = None


def _has_system_cuda_libs() -> bool:
    """Whether the Linux system ships a **complete** CUDA runtime (cuBLAS + cuDNN both on the ld path).

    When a **complete** system CUDA is present, skip the PP9.5 preload -
    torch's bundled CUDA .so files (cu128 -> cuBLAS 12.8) and the CUDA .so
    files onnxruntime-gpu's wheel was built against (typically some other
    12.x point release) can have mismatched ABIs; forcing torch's into the
    global symbol table with RTLD_GLOBAL makes onnxruntime's later cuBLAS
    dlopen resolve to the wrong version -> CUBLAS_STATUS_INVALID_VALUE at
    inference time. When system CUDA is complete, letting onnxruntime dlopen
    the system version directly is correct.

    **Important**: both cuBLAS and cuDNN must be present for this to count as
    complete. It's very common for cloud images to have the CUDA Toolkit
    (with cuBLAS) but not cuDNN (cuDNN requires a separate NVIDIA Developer
    account download); checking only cuBLAS would misclassify this ->
    preload skipped -> onnxruntime fails to dlopen libcudnn.so.9 -> silent
    fallback to CPU. In this "partial system CUDA" case, torch's wheel
    preload must still cover cuDNN (torch's GPU build bundles cuDNN 9.x in
    its nvidia.cudnn subpackage).

    Both checks below must pass:
    1. CUDA Toolkit: CUDA_HOME / CUDA_PATH pointing at a dir with lib64 /
       default /usr/local/cuda exists / libcublas found on the ld path - any one hit
    2. cuDNN: libcudnn found on the ld path - required
    """
    import ctypes.util  # noqa: PLC0415  Linux-only path, avoid this import at module load
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
    has_toolkit = False
    if cuda_home and os.path.isdir(os.path.join(cuda_home, "lib64")):
        has_toolkit = True
    elif os.path.isdir("/usr/local/cuda/lib64"):
        has_toolkit = True
    elif ctypes.util.find_library("cublas"):
        has_toolkit = True
    if not has_toolkit:
        return False
    # cuDNN must also be on the system ld path. Cloud images with the CUDA
    # Toolkit (incl. cuBLAS) but no cuDNN are common; checking cuBLAS alone
    # would misclassify -> preload skipped -> onnxruntime fails to dlopen
    # libcudnn.so.9 -> silent CPU fallback. In that partial-system-CUDA case
    # torch's wheel preload must still cover cuDNN (bundled in its
    # nvidia.cudnn subpackage).
    return bool(ctypes.util.find_library("cudnn"))


def _resolve_cuda_major() -> Optional[int]:
    """Decide which CUDA major version (12 / 13 / None) onnxruntime-gpu should target.

    Anchored on the CUDA version **torch was actually built against**: the
    preload relies on torch's bundled nvidia wheels, and the ORT build must
    match the same major, or a mixed cu12/cu13 process breaks - ABI
    mismatches at inference time (CUBLAS_STATUS_*) or a dlopen failure at
    import time (e.g. libcudart.so.13 not found - the exact regression this
    fix addresses: the old `>=1.20` pulling in ORT 1.27 / CUDA 13 while the
    project's torch is cu12).

    Resolution order:
    1. `torch.version.cuda` (e.g. "12.8" -> 12, "13.0" -> 13) - the most
       authoritative source, torch's own CUDA build major.
    2. torch is a CPU build / not installed -> fall back to the nvidia-smi
       driver version via torch.recommend_cu_tag (imported locally to avoid a
       circular import between torch.py and onnxruntime.py) to get a cu tag:
       `cu128`->12, `cu130`->13. The current recommend_cu_tag table tops out
       at cu128, so this fallback always returns 12 today.
    3. Neither available (no GPU, no torch) -> None; callers fall back to
       DEFAULT_CUDA_MAJOR (=12).
    """
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
        cuda_v = getattr(torch.version, "cuda", None)
        if cuda_v:
            return int(str(cuda_v).split(".")[0])
    except (ImportError, ValueError, TypeError):
        pass
    try:
        from .torch import recommend_cu_tag  # noqa: PLC0415  local import breaks a cycle
    except ImportError:
        return None
    tag = recommend_cu_tag(detect_cuda().get("driver_version"))
    # tag looks like "cu128" -> 128 // 10 = 12; "cu118" -> 11; "cu130" -> 13; "cpu" -> None
    if tag.startswith("cu") and tag[2:].isdigit():
        return int(tag[2:]) // 10
    return None


def _add_torch_dll_dirs_windows() -> dict[str, Any]:
    """Add torch's bundled CUDA DLL directory to the Python DLL search path (Windows).

    Since Python 3.8, Windows no longer dlopens native DLLs found via PATH
    for security reasons - `os.add_dll_directory()` must be called
    explicitly. Without it, onnxruntime fails to dlopen `cublasLt64_12.dll` /
    `cudnn_*.dll` at import time; `get_available_providers()` still lists
    CUDAExecutionProvider, but InferenceSession silently falls back to CPU
    (onnx_tagger_base._create_session already detects this fallback).

    torch's GPU build wheel puts the full set of CUDA DLLs under
    `site-packages/torch/lib/` (cublasLt64_12 / cudnn*_9 / curand64_10 /
    cufft64_11 / cusparse64_12 / cudart64_12 / nvrtc). Adding it to the DLL
    search path lets onnxruntime's dlopen find them.

    Returns `{"added", "errors", "candidates"}`:
    - `added`: directories successfully passed to add_dll_directory
    - `errors`: (dir, reason) pairs for attempts that failed
    - `candidates`: number of candidate directories found (0 means no torch GPU build in the venv)
    """
    added: list[str] = []
    errors: list[tuple[str, str]] = []
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return {"added": added, "errors": errors, "candidates": 0}
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(lib):
        return {"added": added, "errors": errors, "candidates": 0}
    try:
        # Windows-only API since Python 3.8; not present on other platforms
        os.add_dll_directory(lib)  # type: ignore[attr-defined]
        added.append(lib)
    except (OSError, AttributeError) as exc:
        errors.append((lib, str(exc)))
    return {"added": added, "errors": errors, "candidates": 1}


def _preload_torch_cuda_libs() -> dict[str, Any]:
    """Cross-platform preload of torch's bundled CUDA libraries so onnxruntime-gpu's dlopen can find them.

    Background: onnxruntime-gpu's wheel doesn't bundle the CUDA runtime; on
    machines without a system CUDA install, the CUDA EP looks available in
    `get_available_providers()`, but creating a session fails to dlopen
    (Linux: `libcurand.so.10`; Windows: `cublasLt64_12.dll`). onnxruntime
    doesn't raise - it **silently falls back to CPU** (onnx_tagger_base
    already detects this).

    PyTorch's GPU build bundles every CUDA library needed:
    - **Linux**: installed under `site-packages/nvidia/*/lib/` - preloaded
      into the global symbol table with `ctypes.CDLL(RTLD_GLOBAL)`
    - **Windows**: installed under `site-packages/torch/lib/` - added to the
      DLL search path with `os.add_dll_directory()` (required since Python 3.8)

    **Note**: on Linux, skip the preload if the system already has CUDA
    (e.g. an nvidia docker image). If torch's wheel CUDA version (cu128 =
    cuBLAS 12.8) doesn't match the CUDA sub-version onnxruntime-gpu was built
    against, forcing an RTLD_GLOBAL override causes
    CUBLAS_STATUS_INVALID_VALUE at inference time. On Windows,
    `add_dll_directory` only adds a search directory and doesn't override
    already-loaded symbols, so this isn't an issue there.

    Only affects the **current process**; server subprocesses must run this
    again themselves (this module does so automatically on import).

    Returns `{"applied", "platform_skip", "system_cuda_skip", "preloaded", "errors", "candidates"}`:
    - `platform_skip=True`: neither Linux nor Windows (e.g. macOS), skip entirely
    - `system_cuda_skip=True`: Linux with system CUDA present, skip preload
      and let onnxruntime dlopen the system version itself
    - `preloaded`: absolute paths successfully dlopen'd / add_dll_directory'd
    - `errors`: (path, reason) pairs for attempts that failed
    - `candidates`: number of candidates inspected (Linux: nvidia.* subpackages; Windows: torch/lib dir)
    """
    if sys.platform == "win32":
        wres = _add_torch_dll_dirs_windows()
        return {
            "applied": True,
            "platform_skip": False,
            "system_cuda_skip": False,
            "preloaded": wres["added"],
            "errors": wres["errors"],
            "candidates": wres["candidates"],
        }
    if not sys.platform.startswith("linux"):
        return {
            "applied": False,
            "platform_skip": True,
            "system_cuda_skip": False,
            "preloaded": [],
            "errors": [],
            "candidates": 0,
        }
    if _has_system_cuda_libs():
        return {
            "applied": False,
            "platform_skip": False,
            "system_cuda_skip": True,
            "preloaded": [],
            "errors": [],
            "candidates": 0,
        }
    preloaded: list[str] = []
    errors: list[tuple[str, str]] = []
    seen: set[str] = set()
    candidates = 0
    for pkg in _TORCH_NVIDIA_LIB_PKGS_LINUX:
        try:
            mod = importlib.import_module(pkg)
        except ImportError:
            continue
        candidates += 1
        for base in getattr(mod, "__path__", []):
            lib_dir = os.path.join(base, "lib")
            if not os.path.isdir(lib_dir):
                continue
            # Glob every lib*.so* this subpackage bundles: cu12 is .so.12 /
            # libcudnn.so.9, cu13 is .so.13 / libcudnn.so.10 - the glob adapts
            # automatically, no soname table to maintain.
            for so in sorted(os.listdir(lib_dir)):
                if not so.startswith("lib") or ".so" not in so:
                    continue
                candidate = os.path.join(lib_dir, so)
                if candidate in seen:
                    continue
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                except OSError as exc:
                    errors.append((candidate, str(exc)))
                    continue
                preloaded.append(candidate)
                seen.add(candidate)
    return {
        "applied": True,
        "platform_skip": False,
        "system_cuda_skip": False,
        "preloaded": preloaded,
        "errors": errors,
        "candidates": candidates,
    }


def _ensure_preload() -> dict[str, Any]:
    """Idempotently trigger the preload; runs once, cached on subsequent calls.

    Skipped entirely if onnxruntime isn't installed - the preload only exists
    to backstop the CUDA EP's dlopen during a later `import onnxruntime`, so
    it's pointless without it. After installing, Studio must be restarted (a
    C extension can't be hot-swapped); the new process re-triggers this on
    import and works correctly.
    """
    global _PRELOAD_RESULT
    if _PRELOAD_RESULT is not None:
        return _PRELOAD_RESULT
    if _query_dist_info()[0] is None:
        _PRELOAD_RESULT = {
            "applied": False,
            "platform_skip": False,
            "system_cuda_skip": False,
            "not_installed_skip": True,
            "preloaded": [],
            "errors": [],
            "candidates": 0,
        }
        return _PRELOAD_RESULT
    _PRELOAD_RESULT = _preload_torch_cuda_libs()
    if sys.platform == "win32" and _PRELOAD_RESULT["preloaded"]:
        logger.info(
            "[onnx_setup] Added torch/lib to the DLL search path (for onnxruntime-gpu's CUDA dlopen)"
        )
    elif _PRELOAD_RESULT["preloaded"]:
        logger.info(
            "[onnx_setup] Preloaded %d CUDA libraries bundled with torch: %s",
            len(_PRELOAD_RESULT["preloaded"]),
            ", ".join(
                os.path.basename(p) for p in _PRELOAD_RESULT["preloaded"]
            ),
        )
    elif _PRELOAD_RESULT.get("system_cuda_skip"):
        logger.info(
            "[onnx_setup] System CUDA detected, skipping the torch wheel preload (avoids cuBLAS version mismatch)"
        )
    elif _PRELOAD_RESULT["applied"] and _PRELOAD_RESULT["candidates"] == 0:
        logger.debug(
            "[onnx_setup] No torch-bundled CUDA wheels found; the GPU EP depends on system CUDA"
        )
    return _PRELOAD_RESULT


def record_cuda_load_error(msg: Optional[str]) -> None:
    """Called by wd14_tagger.prepare when creating an InferenceSession fails, to stash the reason.

    None means success / clear (a successful session creation clears any old error).
    """
    global _CUDA_LOAD_ERROR
    _CUDA_LOAD_ERROR = msg


def get_cuda_load_error() -> Optional[str]:
    return _CUDA_LOAD_ERROR


# Triggering the preload at module load time - must take effect before any
# `import onnxruntime` anywhere. server.py's top-level
# `from .services import onnxruntime_setup` already covers server
# subprocesses; cli.py also imports this module early in cmd_run.
_ensure_preload()


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def detect_cuda() -> dict[str, Any]:
    """Run the nvidia-smi probe. Returns {"available": bool, "driver_version": str|None, "gpu_name": str|None}.

    nvidia-smi doesn't need root and is the cheapest GPU detection available;
    if it's missing or fails, treat it as no GPU.
    """
    nv = shutil.which("nvidia-smi")
    if not nv:
        return {"available": False, "driver_version": None, "gpu_name": None}
    try:
        out = subprocess.run(
            [
                nv,
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("nvidia-smi exec failed: %s", exc)
        return {"available": False, "driver_version": None, "gpu_name": None}
    if out.returncode != 0:
        return {"available": False, "driver_version": None, "gpu_name": None}
    line = (out.stdout or "").strip().splitlines()
    if not line:
        return {"available": False, "driver_version": None, "gpu_name": None}
    parts = [p.strip() for p in line[0].split(",", 1)]
    driver = parts[0] if parts else None
    name = parts[1] if len(parts) > 1 else None
    return {"available": True, "driver_version": driver, "gpu_name": name}


def current_runtime() -> dict[str, Any]:
    """Return onnxruntime status from the current process's point of view.

    `installed` comes from dist-info (pip's view); `providers` are the EPs
    actually available after import (the loaded native module's view). The
    two **can disagree** - after installing a package without restarting,
    dist-info shows the new package while providers still reflect the old
    one. `restart_required` flags this state.
    """
    installed_pkg, installed_ver = _query_dist_info()
    process_version: Optional[str] = None
    providers: list[str] = []
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
        providers = list(ort.get_available_providers())
        process_version = getattr(ort, "__version__", None)
    except ImportError:
        pass

    # Detect a mismatch between "the package pip has installed" and "the
    # native module actually imported in this process" - onnxruntime is a C
    # extension, so pip uninstall/reinstall doesn't hot-swap an already
    # imported .pyd; a restart is required to switch EPs.
    #
    # The check only compares "installed package type" against "EPs actually
    # loaded in the process", **never version strings**: onnxruntime-directml's
    # dist version (e.g. 1.24.4) and the onnxruntime core version bundled
    # inside it (ort.__version__, e.g. 1.27.0) are two independent version
    # lines that are naturally unequal; comparing version strings would make
    # DirectML users see a permanent false "restart needed" that never clears
    # no matter how many times they restart. EP consistency is the reliable
    # signal, and it's exactly what this feature is meant to detect (that an
    # EP switch actually took effect).
    _ACCEL_EPS = ("CUDAExecutionProvider", "DmlExecutionProvider")
    restart_required = False
    if installed_pkg is not None and process_version is not None:
        if installed_pkg == GPU_PACKAGE and "CUDAExecutionProvider" not in providers:
            # GPU package installed but no CUDA EP in this process -> still running the old (CPU / DirectML) package
            restart_required = True
        elif installed_pkg == DIRECTML_PACKAGE and "DmlExecutionProvider" not in providers:
            # DirectML package installed but no Dml EP in this process -> still running the old (CPU / GPU) package
            restart_required = True
        elif installed_pkg == CPU_PACKAGE and any(ep in providers for ep in _ACCEL_EPS):
            # CPU package installed but an accelerated EP is still present -> still running the old (GPU / DirectML) package
            restart_required = True

    # torch's CUDA major version (the anchor onnxruntime-gpu build selection uses); for UI / diagnostics.
    torch_cuda_major = _resolve_cuda_major()
    # Mismatch heuristic: cuda_load_error mentions .so.13 while torch is major
    # 12 (or vice versa) -> the installed ORT build's major doesn't match
    # torch's (exactly the regression this fix avoids). Best-effort, just a hint.
    ort_cuda_major_mismatch = False
    load_err = _CUDA_LOAD_ERROR or ""
    if load_err and torch_cuda_major is not None:
        if ".so.13" in load_err and torch_cuda_major != 13:
            ort_cuda_major_mismatch = True
        elif ".so.12" in load_err and torch_cuda_major != 12:
            ort_cuda_major_mismatch = True

    return {
        "installed": installed_pkg,
        "version": installed_ver or process_version,
        "providers": providers,
        "cuda_available": "CUDAExecutionProvider" in providers,
        "directml_available": "DmlExecutionProvider" in providers,
        # Platform id: the frontend disables the DirectML/GPU buttons per
        # platform (DirectML is Windows-only; the CUDA runtime wheel is
        # Linux-only; CPU works everywhere)
        "platform": sys.platform,
        "restart_required": restart_required,
        # PP9.5 - the actual dlopen error hit while creating an
        # InferenceSession (e.g. missing `libcurand.so.10`); filled in after
        # wd14_tagger.prepare falls back to CPU. None = untouched / last attempt succeeded.
        "cuda_load_error": _CUDA_LOAD_ERROR,
        # PP9.5 - result of preloading torch's bundled CUDA .so files (Linux only); used for UI diagnostics
        "preload": _PRELOAD_RESULT,
        # torch's CUDA major version (the anchor for the onnxruntime-gpu build) + whether it mismatches the installed ORT
        "torch_cuda_major": torch_cuda_major,
        "ort_cuda_major_mismatch": ort_cuda_major_mismatch,
    }


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


_PIP_FALLBACK_MIRROR = "https://mirrors.cloud.tencent.com/pypi/simple/"


def _pip(args: list[str], *, mirror: str = "") -> tuple[int, str]:
    """Run `<sys.executable> -m pip <args>`; returns (rc, combined_output).

    When `mirror` is non-empty, appends `-i {mirror}` (used for a mirror fallback retry).
    """
    cmd = [sys.executable, "-m", "pip", *args]
    if mirror:
        cmd += ["-i", mirror]
    logger.info("[onnx_setup] %s", " ".join(cmd))
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # pip install can take several minutes
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return 1, f"pip timed out (10 minutes): {exc}"
    except Exception as exc:  # noqa: BLE001
        return 1, f"pip invocation failed: {exc}"
    text = (out.stdout or "") + (out.stderr or "")
    return out.returncode, text


def _decide_target(target: str) -> str:
    """auto/gpu/cpu/directml -> the actual package name (with version constraint).

    The GPU path's version constraint is split cu12/cu13 by
    _resolve_cuda_major(), keeping the ORT build on the same major as torch
    (otherwise a mixed cu12/cu13 process gets ABI mismatches).

    The auto path splits by platform:
    - Windows + GPU -> DirectML (sidesteps CUDA dlopen issues, vendor-agnostic)
    - Linux + GPU -> onnxruntime-gpu (best native CUDA EP), version constraint by torch's major
    - No GPU -> CPU package
    """
    if target == "gpu":
        return _gpu_version_spec_for(_resolve_cuda_major())
    if target == "cpu":
        return f"{CPU_PACKAGE}{CPU_VERSION_SPEC}"
    if target == "directml":
        return f"{DIRECTML_PACKAGE}{DIRECTML_VERSION_SPEC}"
    if target == "auto":
        cuda = detect_cuda()
        if cuda["available"]:
            if sys.platform == "win32":
                return f"{DIRECTML_PACKAGE}{DIRECTML_VERSION_SPEC}"
            return _gpu_version_spec_for(_resolve_cuda_major())
        return f"{CPU_PACKAGE}{CPU_VERSION_SPEC}"
    raise ValueError(f"Invalid target: {target!r} (expected auto/gpu/cpu/directml)")


def _is_dist_installed(pkg: str) -> bool:
    """Whether this package is present in dist-info; doesn't import it, to avoid loading the native module."""
    try:
        from importlib.metadata import PackageNotFoundError, version as _ver
        try:
            _ver(pkg)
            return True
        except PackageNotFoundError:
            return False
    except Exception:  # noqa: BLE001
        return False


def _install_cuda_runtime_wheels(major: Optional[int] = None) -> dict[str, Any]:
    """PP9.6 - install the CUDA runtime wheels onnxruntime-gpu needs to actually run, on Linux.

    `major`: CUDA major version (12/13); if None, inferred by
    _resolve_cuda_major() (falling back to DEFAULT_CUDA_MAJOR). Picks cu12
    (`nvidia-*-cu12`) or cu13 (unsuffixed) accordingly - the same major used
    for the ORT version constraint in _decide_target (both anchored on
    torch.version.cuda).

    Returns `{"installed": [...newly installed], "skipped": [...already present], "platform_skip": bool,
           "cuda_major": int|None, "stdout": str}`. Raises RuntimeError on
    failure, **rolling back whatever it just installed** first (keeps the
    venv clean).

    cuDNN is handled separately: left alone if already present (avoids
    clashing with torch's pinned version); only installed if missing.
    """
    if not sys.platform.startswith("linux"):
        # Windows / macOS: the nvidia CUDA runtime wheels aren't available;
        # users should rely on a system CUDA Toolkit install.
        return {
            "installed": [],
            "skipped": [],
            "platform_skip": True,
            "cuda_major": major,
            "stdout": "non-linux platform; skip nvidia cuda runtime wheels",
        }
    if major is None:
        major = _resolve_cuda_major() or DEFAULT_CUDA_MAJOR
    runtime_wheels, cudnn_wheel = _cuda_wheels_for(major)
    targets: list[str] = []
    skipped: list[str] = []
    # cuDNN: only install if missing (torch's GPU build usually already has it)
    if _is_dist_installed(cudnn_wheel):
        skipped.append(cudnn_wheel)
    else:
        targets.append(cudnn_wheel)
    # The other 6: install whichever are missing
    for pkg in runtime_wheels:
        if _is_dist_installed(pkg):
            skipped.append(pkg)
        else:
            targets.append(pkg)
    if not targets:
        return {
            "installed": [],
            "skipped": skipped,
            "platform_skip": False,
            "cuda_major": major,
            "stdout": "all CUDA runtime wheels already present",
        }
    rc, out = _pip(["install", *targets])
    if rc != 0:
        logger.warning("[onnx_setup] CUDA wheels failed from the official PyPI index, retrying via the Tencent mirror...")
        rc, out = _pip(["install", *targets], mirror=_PIP_FALLBACK_MIRROR)
    if rc != 0:
        # Roll back: uninstall whatever we intended to install, restoring the
        # pre-install state (pip may have partially installed some packages;
        # we don't distinguish, just uninstall all of them)
        rb_rc, rb_out = _pip(["uninstall", "-y", *targets])
        raise RuntimeError(
            f"Installing the CUDA runtime wheels failed (rc={rc}):\n{out}\n"
            f"--- rollback (rc={rb_rc}) ---\n{rb_out}"
        )
    return {
        "installed": targets,
        "skipped": skipped,
        "platform_skip": False,
        "cuda_major": major,
        "stdout": out,
    }


def install_runtime(target: str = "auto") -> dict[str, Any]:
    """Uninstall all three mutually-exclusive packages, then install the target.

    target: "auto" | "gpu" | "cpu" | "directml"
    Returns `{"target", "installed_pkg", "installed_version", "restart_required": True,
           "stdout", "cuda_runtime"}`; the last field is PP9.6's report on the
    nvidia CUDA runtime wheels installed (cu12 or cu13 by torch's CUDA major;
    GPU path only; None for CPU / DirectML). Raises RuntimeError on failure.

    **Important**: onnxruntime is a C extension, so after pip
    uninstall/reinstall the .pyd/.so already imported in **this process**
    isn't hot-swapped - Studio must be restarted to switch EPs. This function
    therefore doesn't attempt a reload; it returns `restart_required=True`
    for the UI to prompt the user.
    """
    spec = _decide_target(target)
    rc1, log1 = _pip(["uninstall", "-y", *_MUTUALLY_EXCLUSIVE_PACKAGES])
    rc2, log2 = _pip(["install", "--upgrade", spec])
    if rc2 != 0:
        logger.warning("[onnx_setup] pip failed from the official index, retrying via the Tencent mirror...")
        rc2, log2 = _pip(["install", "--upgrade", spec], mirror=_PIP_FALLBACK_MIRROR)
    if rc2 != 0:
        raise RuntimeError(f"Installing {spec} failed (rc={rc2}):\n{log2}")

    # PP9.6 - fill in the CUDA runtime wheels on the GPU path
    # (onnxruntime-gpu doesn't bundle them). Skipped on the CPU path or when
    # auto-detection resolves to CPU.
    cuda_runtime: Optional[dict[str, Any]] = None
    if GPU_PACKAGE in spec:
        try:
            cuda_runtime = _install_cuda_runtime_wheels()
        except RuntimeError as exc:
            # A failed CUDA wheels install isn't fatal: onnxruntime-gpu is
            # already installed, and the user can see cuda_load_error on the
            # Settings page and fix it manually. Log the reason so the UI can
            # surface it too.
            logger.error("[onnx_setup] Installing CUDA runtime wheels failed: %s", exc)
            cuda_runtime = {
                "installed": [],
                "skipped": [],
                "platform_skip": False,
                "cuda_major": None,
                "stdout": str(exc),
                "error": str(exc),
            }

    # Read dist-info directly for the newly installed version (not imported; the process still has the old native module)
    new_pkg, new_ver = _query_dist_info()
    return {
        "target": spec,
        "installed_pkg": new_pkg,
        "installed_version": new_ver,
        "restart_required": True,
        "stdout": log1 + log2,
        "cuda_runtime": cuda_runtime,
    }
