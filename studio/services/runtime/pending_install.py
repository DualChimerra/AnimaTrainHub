"""Cross-process pip install queue: lets the launcher process take over an install the server process can't complete.

Why:
- Windows file locking: a C extension `.pyd` that's already been imported can't be replaced by pip
  (`[WinError 5] Access is denied: torch\\_C.cp311-win_amd64.pyd`)
- onnxruntime_setup already works around this with "pip uninstall + install + restart_required=True",
  but its install path is triggered by the user manually uninstalling/reinstalling; onnxruntime may not necessarily have been imported in the current process
- torch is different: the server imports it along the way at startup (flash_attention_setup.detect_env, various
  services indirectly importing it), so a same-process pip uninstall **inevitably hits the file lock**

Design:
- When the server receives a reinstall request -> instead of actually running pip, it writes the marker `studio_data/.pending-pip-install.json` ->
  returns `pending: true`, and the UI tells the user to restart
- cli.py cmd_run / cmd_dev at startup -> first `apply_pending()` -> installs, then starts the server
- Retry on failure: if pip fails, the marker isn't cleared, so it retries on the next startup
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional

from ...paths import STUDIO_DATA

logger = logging.getLogger(__name__)

# studio_data/ is gitignored, kept across restarts
PENDING_MARKER = STUDIO_DATA / ".pending-pip-install.json"


def register_torch_reinstall(target: str) -> None:
    """Registers a torch reinstall request; the marker is written to disk before returning."""
    STUDIO_DATA.mkdir(parents=True, exist_ok=True)
    PENDING_MARKER.write_text(
        json.dumps({"kind": "torch", "target": target}, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("[pending_install] registered torch reinstall: target=%s", target)


def read_pending() -> Optional[dict[str, Any]]:
    """Reads the marker; returns None if it doesn't exist / fails to parse."""
    if not PENDING_MARKER.exists():
        return None
    try:
        return json.loads(PENDING_MARKER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[pending_install] Failed to parse marker: %s", exc)
        return None


def clear_pending() -> None:
    if PENDING_MARKER.exists():
        try:
            PENDING_MARKER.unlink()
        except OSError as exc:
            logger.warning("[pending_install] Failed to delete marker: %s", exc)


def apply_pending() -> None:
    """Handles a pending request at startup; must be called before any `import torch`.

    On success -> clears the marker; on failure -> keeps the marker (retried on the next startup). Errors print to stderr,
    no exception is raised, so the launcher can still start the server (the user can see the old torch still in use in the UI).
    """
    pending = read_pending()
    if not pending:
        return

    kind = pending.get("kind")
    if kind == "torch":
        target = pending.get("target", "auto")
        print(f"[studio] Detected a pending torch reinstall request (target={target}), starting install...")
        print("[studio] Tip: press Ctrl+C to skip this install (the marker is kept, retried on next startup)")
        print(f"[studio] To skip this permanently, delete the marker file: {PENDING_MARKER}")
        # Deferred import: side effects triggered by the torch_setup -> onnxruntime_setup chain are all left until now
        from . import torch as torch_setup  # noqa: PLC0415
        try:
            res = torch_setup.reinstall(target, stream=True)
        except KeyboardInterrupt:
            print("\n[studio] torch reinstall interrupted by user, skipping. Marker kept, will retry on next startup.",
                  file=sys.stderr)
            print(f"[studio] To skip this permanently, delete the marker file: {PENDING_MARKER}",
                  file=sys.stderr)
            return  # doesn't clear_pending, keeps trying on the next startup
        except RuntimeError as exc:
            print(f"[studio] torch reinstall failed: {exc}", file=sys.stderr)
            print("[studio] Marker kept, will retry on next startup", file=sys.stderr)
            print(f"[studio] If the install keeps failing and you want to skip it permanently, delete the marker file: {PENDING_MARKER}",
                  file=sys.stderr)
            return
        print(f"[studio] torch reinstall complete: {res.get('version')} ({res.get('tag')})")
    else:
        print(
            f"[studio] Warning: unknown pending install kind {kind!r}, ignoring and clearing",
            file=sys.stderr,
        )

    clear_pending()
