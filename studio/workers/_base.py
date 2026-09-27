"""Common worker subprocess entry-point template (extracted from 4 workers in PR-8 commit 1).

Every worker subprocess is launched by `supervisor` following the same pattern:
    python -m studio.workers.<kind>_worker --job-id N
    -> read the project_jobs row -> do the work -> exit code reflects success/failure

Each worker module only needs to provide a `run(job_id: int) -> int` body; this template
wraps it with:
    - argparse parsing of `--job-id`
    - calling run + sys.exit

Optional helper `reconfigure_console_utf8()` -- the Windows console defaults to
a legacy ANSI codepage, so writing non-ASCII / emoji raises UnicodeEncodeError; calling this once
switches stdout/stderr to UTF-8 + replace mode. Currently only tag_worker needs it (other
workers don't write non-ASCII captions).
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable


def worker_main(run_fn: Callable[[int], int]) -> None:
    """The `if __name__ == "__main__"` entry point shared by all 4 workers.

    Each worker module's bottom reads:
        if __name__ == "__main__":
            from ._base import worker_main
            worker_main(run)

    PR-1 C4: sets up setup_logging (ADR-0009). file=False -- workers write to stdout,
    which supervisor redirects into a single jobs/<id>.log (C round2 §1.2 decision;
    changed to True after 0.13.x ADR-0009 §"pay off the debt" evolved to dual-write).
    process name format is "worker:<module>/<job_id>", convenient for jq filtering by
    process; module comes from sys.argv[0]'s basename (download_worker.py ->
    download_worker -> strip the _worker suffix -> download).

    PR-1 C6: bind_trace_id reads the ANIMA_TRACE_ID env var (injected by supervisor's
    _spawn_task from task.request_trace_id -- the trace_id from the moment of the HTTP
    request, carried all the way down). Falls back to new_trace_id if absent (for testing
    a worker standalone / running -m studio.workers.x_worker manually). bind is not reset
    -- the trace_id should follow the worker process for its whole lifetime.
    """
    import os
    from ..infrastructure.logging import (
        PROCESS_ENV, TRACE_ENV,
        bind_job_id, bind_trace_id, new_trace_id, setup_logging,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", type=int, required=True)
    args = p.parse_args()
    kind = _worker_kind_from_argv()
    process = os.environ.get(PROCESS_ENV) or f"worker:{kind}/{args.job_id}"
    setup_logging(process, file=False, console=True)
    bind_trace_id(os.environ.get(TRACE_ENV) or new_trace_id())
    bind_job_id(args.job_id)
    sys.exit(run_fn(args.job_id))


def _worker_kind_from_argv() -> str:
    """sys.argv[0] basename -> kind (download_worker.py -> download)."""
    from pathlib import Path as _Path
    name = _Path(sys.argv[0]).stem
    if name.endswith("_worker"):
        name = name[: -len("_worker")]
    return name or "unknown"


def reconfigure_console_utf8() -> None:
    """The Windows console defaults to a legacy ANSI codepage, so writing non-ASCII / emoji raises
    UnicodeEncodeError. Forces stdout/stderr to UTF-8 + replace-unencodable-chars, so
    progress output never throws.

    Currently only called at tag_worker's top level (other workers don't write non-ASCII
    captions). supervisor's `_popen` already injects `PYTHONIOENCODING=utf-8 /
    PYTHONUTF8=1` into every worker subprocess's env, but a handful of Windows console
    hosts still need this reconfigure as a second line of defense.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass
