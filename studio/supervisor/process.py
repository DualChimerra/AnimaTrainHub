"""Cross-platform process-tree killing utilities (extracted from supervisor.py in PR-4).

On Windows, `proc.kill()` only kills the immediate child -- DataLoader workers /
accelerate's sub-subprocesses would be left behind still holding the GPU; `taskkill /T /F`
recurses through the whole process tree. POSIX uses killpg.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess

logger = logging.getLogger(__name__)


def _kill_process_tree_psutil(pid: int) -> None:
    """Windows fallback for when taskkill is unavailable / denied by policy.

    psutil is a required project dependency. Kills the deepest children first, then the
    root, to approximate taskkill's tree semantics as closely as possible; NoSuchProcess
    is a normal race from a process exiting concurrently.
    """
    import psutil

    try:
        root = psutil.Process(pid)
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    except Exception:
        logger.exception("psutil enumerate process tree failed for pid %d", pid)
        descendants = []
        try:
            root = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return

    for proc in reversed(descendants):
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        except Exception:
            logger.exception("psutil kill failed for child pid %d", proc.pid)
    try:
        root.kill()
    except psutil.NoSuchProcess:
        pass
    except Exception:
        logger.exception("psutil kill failed for root pid %d", pid)


def _kill_process_tree_windows(pid: int) -> None:
    """Windows taskkill primary path; guaranteed to fall back to the psutil tree kill on failure."""
    try:
        result = subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            check=False, capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return
        detail = (result.stderr or result.stdout or b"").decode(
            errors="replace"
        ).strip()
        logger.warning(
            "taskkill failed for pid %d (rc=%d): %s; falling back to psutil",
            pid, result.returncode, detail or "no output",
        )
    except Exception:
        logger.exception("taskkill /T /F failed for pid %d", pid)
    _kill_process_tree_psutil(pid)


def _kill_process_tree(pid: int) -> None:
    """Kill the entire process tree rooted at pid.

    On Windows, `proc.kill()` only kills the immediate child -- DataLoader workers /
    accelerate's sub-subprocesses would be left behind still holding the GPU; `taskkill /T /F`
    recurses through the whole process tree. POSIX uses killpg.
    """
    if os.name == "nt":
        _kill_process_tree_windows(pid)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            logger.exception("killpg failed for pid %d", pid)
