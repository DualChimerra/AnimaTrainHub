"""onnxruntime-gpu silent-downgrade-to-CPU diagnostics -- after running on this PR's branch,
if tagging produces a "CUDA EP silently downgraded to CPU" warning, use this script to find the root cause.

Usage (from the studio venv):

    python tools/diagnose_onnx_gpu.py

Paste the whole stdout back into the PR comment / issue.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def section(title: str) -> None:
    print()
    print("=" * 60)
    print("==", title)
    print("=" * 60)


def main() -> None:
    section("platform")
    print("python:", sys.version.split()[0])
    print("platform:", sys.platform)
    print("executable:", sys.executable)

    section("onnxruntime_setup.current_runtime()")
    try:
        from studio.services.runtime import onnxruntime as o
    except ImportError as exc:
        print("studio import failed -- are you running this from the studio repo root?")
        print("reason:", exc)
        return
    rt = o.current_runtime()
    print(json.dumps(rt, indent=2, ensure_ascii=False, default=str))
    if rt.get("ort_cuda_major_mismatch"):
        print(">>> Warning: the installed ORT's CUDA major version doesn't match torch's -- "
              "when reinstalling, pick cu12(<1.26) or cu13(>=1.26) build based on torch's major version")

    section("System CUDA detection (decides whether preload is skipped and whether ORT uses cu12 or cu13)")
    print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
    print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
    print("/usr/local/cuda/lib64 exists:", os.path.isdir("/usr/local/cuda/lib64"))
    import ctypes.util
    print("cublas on ld path:", ctypes.util.find_library("cublas"))
    print("cudnn on ld path:", ctypes.util.find_library("cudnn"))
    print("_has_system_cuda_libs():", o._has_system_cuda_libs())
    # Anchor CUDA major version for the ORT build (= torch.version.cuda major); the installed
    # onnxruntime-gpu must match its major, otherwise dlopen fails on libcudart.so.13 at import
    # time (cu128 torch -> 12)
    print("_resolve_cuda_major():", o._resolve_cuda_major())

    section("torch")
    try:
        import torch
        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device_name:", torch.cuda.get_device_name(0))
            print("cudnn_version:", torch.backends.cudnn.version())
    except Exception as exc:  # noqa: BLE001
        print("torch import failed:", exc)

    section("nvidia CUDA wheels (source of torch wheel preload) + onnxruntime")
    out = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        capture_output=True, text=True,
    ).stdout
    keep = ("nvidia", "cudnn", "onnxruntime", "torch")
    for line in out.splitlines():
        low = line.lower()
        if any(k in low for k in keep):
            print(" ", line)

    section("nvidia-smi")
    try:
        nv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        print("rc:", nv.returncode)
        print("stdout:", nv.stdout.strip())
        print("stderr:", nv.stderr.strip())
    except FileNotFoundError:
        print("nvidia-smi not on PATH (cloud machines running in docker may not expose it)")
    except Exception as exc:  # noqa: BLE001
        print("nvidia-smi failed to run:", exc)

    section("Try actually creating an InferenceSession (using onnxruntime's built-in test model or an existing wd14 model)")
    try:
        import onnxruntime as ort
        print("ort.__version__:", ort.__version__)
        print("ort.get_available_providers():", ort.get_available_providers())
        import glob
        candidates = (
            glob.glob("models/wd14/**/model.onnx", recursive=True)
            + glob.glob("models/cltagger/**/*.onnx", recursive=True)
        )
        if not candidates:
            print("No local onnx model found, skipping session creation test")
            return
        path = candidates[0]
        print("using model:", path)
        sess = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        actual = sess.get_providers()
        print("actual session.get_providers():", actual)
        if "CUDAExecutionProvider" not in actual:
            print(">>> Confirmed silent downgrade: CUDA was requested but the session is using", actual)
        else:
            print(">>> CUDA EP is genuinely active")
    except Exception as exc:  # noqa: BLE001
        print("session creation raised an exception:", exc)


if __name__ == "__main__":
    main()
