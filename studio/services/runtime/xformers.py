"""xformers install service (simplified, mirrors flash_attention_setup).

xformers and flash_attn are both attention-acceleration C extensions, but xformers's install path is **notably simpler**:
  - flash_attn: depends on GitHub Releases prebuilt by dao-AILab + mjun0812,
    one wheel per torch+cuda+python combination, requiring wheel-name parsing + score-based matching
  - xformers: Facebook publishes wheels directly on official PyPI; tightly bound to torch+cuda, but
    the official PyTorch wheel index (download.pytorch.org/whl/cuXXX) already
    groups wheels by their cu_tag.

So this service only exposes:
  - current_status() → {installed, version}
  - install() → pip install xformers --index-url <torch-cuda-index>

It doesn't replicate flash_attention_setup's GitHub Releases parsing / candidate-list UI.
On install failure, stderr is passed straight through for the user to read (most failures are the upstream project
not having published a wheel for that torch+cu combination yet, requiring a torch version change or waiting for upstream to catch up).
"""
from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from typing import Any, Optional


def current_status() -> dict[str, Any]:
    """Current xformers install status: {installed: bool, version: str|None}."""
    try:
        version = importlib.metadata.version("xformers")
        return {"installed": True, "version": version}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def detect_attention_backend() -> str:
    """Decides the attention backend based on what's currently installed.
    Priority: flash_attn > xformers > none (PyTorch SDPA).
    Used when secrets.generate.attention_backend='auto'.
    """
    try:
        importlib.metadata.version("flash_attn")
        return "flash_attn"
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        importlib.metadata.version("xformers")
        return "xformers"
    except importlib.metadata.PackageNotFoundError:
        pass
    return "none"


def disable_triton_probe(env: dict[str, str]) -> None:
    """Short-circuits xformers's triton probing (used for subprocess env injection).

    Once xformers is enabled, its `_is_triton_available()` calls `import triton`. triton doesn't publish
    official Windows wheels, so when it's not installed this always ImportErrors; xformers uses
    `logger.warning(..., exc_info=True)` to dump the full traceback into the task log --
    which then gets mistakenly picked up as the failure reason by the failure-summary helper
    `_tail_log_for_error_msg` (which grabs the last Traceback) and shown to the user.

    Unconditional short-circuit: within xformers, triton only serves LLM-style kernels (fmha triton_splitk /
    rmsnorm / rope_padded / tiled_matmul); this app's memory_efficient_attention
    (including NaViT varlen) goes through the cutlass/flash C++ kernels, so whether triton is installed or working
    is completely irrelevant, and neither the probe result nor the warning has any value to the user. torch.compile's
    own triton usage doesn't read this variable, so it's unaffected. `XFORMERS_FORCE_DISABLE_TRITON=1`'s check
    in xformers's source sits before `import triton`, so setting it skips the probe entirely; setdefault ensures
    a value the user explicitly set takes priority, and xformers's own `XFORMERS_ENABLE_TRITON=1` still takes
    even higher priority, remaining a forced-on escape hatch.
    """
    env.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")


def _torch_cuda_index() -> Optional[str]:
    """Derives the PyTorch CUDA index URL from the `+cuXXX` suffix of `torch.__version__`.

    xformers wheels are tightly ABI-bound to torch (each xformers version is locked to a specific torch+cuda),
    so the wheel installed must match the current torch's CUDA build. The official PyTorch index groups by cu_tag:
        https://download.pytorch.org/whl/cu128
        https://download.pytorch.org/whl/cu130
        ...

    ABI detection follows the same principle as flash_attention_setup.detect_env(): taken from torch,
    not from nvidia-smi (nvidia-smi reports the driver-supported CUDA, not the one PyTorch was built against).
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    m = re.search(r"\+cu(\d+)", torch.__version__)
    if m:
        return f"https://download.pytorch.org/whl/cu{m.group(1)}"
    return None


def install() -> dict[str, Any]:
    """pip installs xformers, automatically picking the wheel via the current torch's CUDA index.

    Returns {installed, version, stdout_tail, restart_required}.
    Raises RuntimeError on install failure, with the message including the tail of stderr (when a wheel can't be found,
    pip usually prints "No matching distribution found for xformers").

    `restart_required=True` because xformers is a C extension -- after installing it, the
    Studio process must be restarted to import it (same as flash_attn).
    """
    cmd = [sys.executable, "-m", "pip", "install", "xformers"]
    index = _torch_cuda_index()
    if index:
        cmd += ["--index-url", index]

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("pip install xformers timed out (10 minutes)") from exc

    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-1500:]
        raise RuntimeError(
            f"pip install xformers failed (exit {r.returncode}):\n{tail}"
        )

    status = current_status()
    return {
        **status,
        "stdout_tail": (r.stdout or "")[-1500:],
        "restart_required": True,
    }
