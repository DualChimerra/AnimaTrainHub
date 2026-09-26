"""PyTorch install detection + one-click reinstall service (PR-S2 / PR-S2.1).

Why this gets its own service:
- requirements.txt writes `torch>=2.0.0` without an `--index-url`, so pip installs the CPU wheel by default -- for a
  user with an NVIDIA GPU, the usual outcome is a PR-4 startup warning, but a user with an **existing venv** won't get auto-fixed.
- onnxruntime_setup already has a ready-made detect_cuda (nvidia-smi probe) + pip helper pattern; this
  service mirrors it: detect_torch / recommend_index_url / reinstall.
- The UI (Settings -> Training -> PyTorch section) and the CLI both go through this same API, a single source of truth.

Design notes:
- `pip uninstall torch torchvision -y && pip install torch torchvision --index-url <cu>`
  -- no `--upgrade`; an explicit reinstall always forces the given index-url
- Driver version -> cu wheel mapping conservatively takes "the highest cu that driver can run" (per NVIDIA's backward-compat docs)
- timeout of 30 minutes (torch + cuda deps are ~3 GB, common on slow connections)
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Any, Optional

from . import onnxruntime as onnxruntime_setup

logger = logging.getLogger(__name__)

# Wheel index URLs published by PyTorch. https://download.pytorch.org/whl/<tag>
# Order: newest to oldest; auto picks the first one the driver supports. New cu tags are inserted at the front.
SUPPORTED_INDEX_TAGS: tuple[str, ...] = ("cu128", "cu126", "cu124", "cu118", "cpu")

# Driver version -> the PyTorch CUDA wheel tag that driver can run. NVIDIA drivers are backward compatible (a newer driver can run an older cu).
# Thresholds taken from NVIDIA's official CUDA Toolkit Release Notes "Driver Required" column.
# Source: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id5
_DRIVER_TO_BEST_CU: tuple[tuple[float, str], ...] = (
    (555.0, "cu128"),  # CUDA 12.8 → driver R555+
    (550.0, "cu126"),  # CUDA 12.6 -> driver R550+ (12.6 is about the same tier)
    (545.0, "cu124"),  # CUDA 12.4 → driver R545+
    (470.0, "cu118"),  # CUDA 11.8 → driver R470+
)

PYPI_INDEX_BASE = "https://download.pytorch.org/whl"


def _index_url_for(tag: str) -> Optional[str]:
    """`cu128` -> `https://download.pytorch.org/whl/cu128`; `cpu` -> same; invalid -> None."""
    if tag not in SUPPORTED_INDEX_TAGS:
        return None
    return f"{PYPI_INDEX_BASE}/{tag}"


def recommend_cu_tag(driver_version: Optional[str]) -> str:
    """Returns the recommended cu tag based on the NVIDIA driver version; driver too old / no driver -> 'cpu'."""
    if not driver_version:
        return "cpu"
    try:
        major_minor = float(".".join(driver_version.split(".")[:2]))
    except (ValueError, AttributeError):
        return "cpu"
    for threshold, tag in _DRIVER_TO_BEST_CU:
        if major_minor >= threshold:
            return tag
    return "cpu"


def detect_torch() -> dict[str, Any]:
    """Reads dist-info + probes after import, returning the current torch status.

    `cuda_build`:
    - 'cu128' / 'cu126' / 'cu124' / 'cu118' —— PyTorch CUDA wheel
    - 'cpu' -- a CPU-only wheel (torch.version.cuda is None)
    - None -- torch is not installed

    `cuda_available` reflects `torch.cuda.is_available()` -- even with a CUDA wheel installed, this can still be
    False due to a driver / WSL issue.
    """
    try:
        installed_version = _pkg_version("torch")
    except PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
            "cuda_build": None,
            "cuda_available": False,
            "device_name": None,
        }

    cuda_build: Optional[str] = None
    cuda_available = False
    device_name: Optional[str] = None
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
        # torch.__version__ looks like "2.5.0+cu128" / "2.5.0+cpu" / "2.5.0"
        m = re.search(r"\+(cu\d+|cpu)$", torch.__version__)
        if m:
            cuda_build = m.group(1)
        else:
            # Compat for old builds without a + suffix, falls back to torch.version.cuda
            cuda_v = getattr(torch.version, "cuda", None)
            if cuda_v is None:
                cuda_build = "cpu"
            else:
                # cuda_v looks like "12.8"; map it to a wheel tag
                clean = cuda_v.replace(".", "")
                cuda_build = f"cu{clean}"
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                device_name = "?"
    except ImportError:
        pass

    return {
        "installed": True,
        "version": installed_version,
        "cuda_build": cuda_build,
        "cuda_available": cuda_available,
        "device_name": device_name,
    }


def current_status() -> dict[str, Any]:
    """Packaged for the UI: torch status + driver detection + recommended cu tag."""
    torch_state = detect_torch()
    cuda_detect = onnxruntime_setup.detect_cuda()
    recommended = recommend_cu_tag(cuda_detect.get("driver_version"))

    # Misinstall diagnosis: a CPU wheel installed but there's an NVIDIA GPU -> the UI should surface this prominently
    is_cpu_with_gpu = (
        torch_state["installed"]
        and torch_state["cuda_build"] == "cpu"
        and cuda_detect["available"]
    )
    # A CUDA wheel installed but cuda.is_available()=False -> a driver / WSL issue, not something pip can fix
    is_cuda_build_unavailable = (
        torch_state["installed"]
        and torch_state["cuda_build"] not in (None, "cpu")
        and not torch_state["cuda_available"]
    )

    return {
        **torch_state,
        "cuda_detect": cuda_detect,
        "recommended_cu_tag": recommended,
        "is_cpu_with_gpu": is_cpu_with_gpu,
        "is_cuda_build_unavailable": is_cuda_build_unavailable,
    }


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def _pip(args: list[str], timeout: int = 1800, stream: bool = False) -> tuple[int, str]:
    """Runs `<sys.executable> -m pip <args>`; returns (rc, combined_output).

    Default timeout is 30 minutes -- torch + cuda deps packaged together are ~3 GB, which can take up to an hour on a slow connection.
    stream=True: output is piped straight through to the terminal (used by the launch scenario), not captured, returns an empty string.
    stream=False (default): captures and returns the text (used by the API endpoint / logs).
    """
    cmd = [sys.executable, "-m", "pip", *args]
    logger.info("[torch_setup] %s", " ".join(cmd))
    try:
        if stream:
            rc = subprocess.call(cmd, timeout=timeout)
            return rc, ""
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return 1, f"pip timed out ({timeout}s): {exc}"
    except Exception as exc:  # noqa: BLE001
        return 1, f"pip call failed: {exc}"
    text = (out.stdout or "") + (out.stderr or "")
    return out.returncode, text


def _decide_target_tag(target: str) -> str:
    """auto / cu128 / cu126 / cu124 / cu118 / cpu -> the actual cu tag.

    'auto' -> uses the nvidia-smi recommendation; other values pass through as-is. An invalid value raises ValueError.
    """
    if target == "auto":
        return recommend_cu_tag(onnxruntime_setup.detect_cuda().get("driver_version"))
    if target in SUPPORTED_INDEX_TAGS:
        return target
    raise ValueError(
        f"Invalid target: {target!r} (expected auto / {' / '.join(SUPPORTED_INDEX_TAGS)})"
    )


def _cleanup_zombie_dirs() -> list[str]:
    """Cleans up `~*` zombie directories left in site-packages by a failed pip run.

    A failed pip uninstall / install can leave behind temp directories like `~orch-...dist-info/`,
    `~orchvision/` (the `~` prefix is pip's placeholder meaning "currently being renamed").  These directories
    can cause the next pip install of torch to report `Ignoring invalid distribution ~orch`, and worse,
    can make `import torch` see leftover dist-info while the actual .pyd is missing.

    Returns the list of cleaned-up paths, for logging. Linux site-packages can also have `~`-prefixed
    leftovers (rare, but pip behaves the same way there), so this isn't platform-skipped.
    """
    site_packages = Path(sysconfig.get_path("purelib"))
    cleaned: list[str] = []
    if not site_packages.is_dir():
        return cleaned
    for entry in site_packages.glob("~*"):
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            cleaned.append(entry.name)
        except OSError as exc:
            logger.warning("[torch_setup] failed to clean up zombie directory %s: %s", entry, exc)
    if cleaned:
        logger.info("[torch_setup] cleaned up pip zombie directories: %s", ", ".join(cleaned))
    return cleaned


def reinstall(target: str = "auto", stream: bool = False) -> dict[str, Any]:
    """Uninstalls torch + torchvision, then reinstalls per target.

    target: "auto" | "cu128" | "cu126" | "cu124" | "cu118" | "cpu"
    Returns `{"target", "tag", "index_url", "version", "stdout_tail",
            "restart_required": True, "cleaned_zombies": [...]}`.
    Raises RuntimeError on failure.

    **Important**: torch is a C extension, so after a pip uninstall/reinstall, the .so/.pyd already imported into the
    **current process** won't hot-swap. If the server process has already imported torch, pip uninstall will also hit [WinError 5].
    So this function should run in the launcher process (called via pending_install.apply_pending), not run synchronously
    in the server process's `/api/torch/reinstall` endpoint (that endpoint only writes the marker).

    Self-healing: always cleans up `~*` zombie directories in site-packages first (leftover state from a previous failure).
    """
    tag = _decide_target_tag(target)
    index_url = _index_url_for(tag)

    # Step 1: clean up any zombie directories a previous failure may have left behind. Even if there's no
    # leftover state this time, it's a cheap operation (listing site-packages + globbing '~*'), so the cost is negligible.
    cleaned = _cleanup_zombie_dirs()

    # Step 2: uninstall torch + torchvision (user-installed flash_attn / xformers etc. are left alone,
    # even though they're tightly ABI-bound to torch -- uninstalling torch doesn't automatically uninstall them; the user re-enables / reinstalls them after restarting)
    rc1, log1 = _pip(["uninstall", "-y", "torch", "torchvision"], stream=stream)

    # Clean zombie directories once more after uninstalling (a failed pip uninstall can also leave `~`-prefixed leftovers)
    cleaned += _cleanup_zombie_dirs()

    # Step 3: install. cu* goes through PyTorch's own index; cpu also has its own index (not the PyPI default, to avoid ambiguity)
    install_args = ["install", "torch", "torchvision"]
    if index_url:
        install_args += ["--index-url", index_url]
    rc2, log2 = _pip(install_args, stream=stream)
    if rc2 != 0:
        raise RuntimeError(f"Installing torch ({tag}) failed (rc={rc2}):\n{log2}")

    stdout = log1 + log2
    tail = "\n".join(stdout.splitlines()[-40:])

    # The new version from dist-info's perspective (the process still has the old .pyd / .so loaded)
    try:
        new_version = _pkg_version("torch")
    except PackageNotFoundError:
        new_version = None

    return {
        "target": target,
        "tag": tag,
        "index_url": index_url,
        "version": new_version,
        "stdout_tail": tail,
        "restart_required": True,
        "cleaned_zombies": cleaned,
    }
