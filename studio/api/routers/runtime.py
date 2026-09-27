"""Runtime mode (Colab / Local) — new in this fork.

2 routes:
    GET /api/runtime         current mode + detection results + environment facts for that mode
    PUT /api/runtime         persist the user's choice (the frontend's first-run selection dialog)

The frontend does one GET on startup: `mode` being an empty string means the
user hasn't chosen yet → show the selection dialog; PUT once they choose.
When `locked=true` (set once `ALS_RUNTIME_MODE` has been injected, which the
Colab notebook's startup cell does), the frontend neither shows the dialog
nor allows changing it — the environment has already answered on the user's
behalf.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ...domain.errors import DomainError
from ...infrastructure import runtime_mode
from ...infrastructure.paths import STUDIO_DATA
from ... import secrets

logger = logging.getLogger(__name__)
router = APIRouter()


class RuntimeModePatch(BaseModel):
    mode: str


def _environment() -> dict[str, Any]:
    """Environment facts shown in the selection dialog and settings area.

    Deliberately limited to things the user can use to judge whether they
    picked correctly: where the disk is, how much space is left, whether
    there's a GPU. The GPU name comes from nvidia-smi rather than `import
    torch` — this endpoint sits on the UI startup path, and it shouldn't pay
    torch's import cost (seconds on first import) just to show one line of text.
    """
    total = free = None
    try:
        usage = shutil.disk_usage(STUDIO_DATA if STUDIO_DATA.exists() else Path.cwd())
        total, free = usage.total, usage.free
    except OSError:
        logger.debug("disk_usage failed for studio_data", exc_info=True)

    gpu = ""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            import subprocess

            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                gpu = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
        except Exception:
            logger.debug("nvidia-smi probe failed", exc_info=True)

    return {
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "studio_data": str(STUDIO_DATA),
        "studio_data_env": os.environ.get("ALS_STUDIO_DATA", ""),
        "disk_total": total,
        "disk_free": free,
        "gpu": gpu,
    }


def _payload() -> dict[str, Any]:
    data = runtime_mode.describe()
    data["environment"] = _environment()
    return data


@router.get("/api/runtime")
def get_runtime() -> dict[str, Any]:
    return _payload()


@router.put("/api/runtime")
def put_runtime(body: RuntimeModePatch) -> dict[str, Any]:
    mode = runtime_mode.normalize(body.mode)
    if mode not in runtime_mode.MODES:
        raise DomainError(
            f"unknown runtime mode {body.mode!r}; expected one of {list(runtime_mode.MODES)}",
            code="runtime.invalid_mode",
            details={"mode": body.mode, "allowed": list(runtime_mode.MODES)},
            http_status=400,
        )
    override = runtime_mode.env_override()
    if override and override != mode:
        # The env var is authoritative and never persisted: writing to secrets
        # here would create a split-brain state ("settings show local, but
        # it's actually running colab"), so we just reject and explain why.
        raise DomainError(
            f"runtime mode is pinned to {override!r} by the "
            f"{runtime_mode.ENV_OVERRIDE} environment variable",
            code="runtime.mode_locked",
            details={"mode": override, "env": runtime_mode.ENV_OVERRIDE},
            http_status=409,
        )
    secrets.update({"runtime": {"mode": mode, "asked": True}})
    return _payload()
