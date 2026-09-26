"""Download worker subprocess entry point (pp2).

Launched by supervisor: `python -m studio.workers.download_worker --job-id N`.
Reads the `project_jobs` row + `secrets.gelbooru` -> calls
`studio.services.downloader.download()` -> writes logs -> exit code reflects success/failure.
Status fields (running / done / failed) are written back uniformly by supervisor when the
subprocess exits.

Logging goes through stdout only: supervisor's `subprocess.Popen(stdout=log_fp,
stderr=STDOUT)` redirects the whole subprocess's output into the task log file; the
worker itself must **not** open that same log and write to it directly -- otherwise each
line would be written to disk twice, LogTailer would read it twice, and the frontend
would show every log line duplicated.
"""
from __future__ import annotations

import logging
import threading

from studio import db, secrets

logger = logging.getLogger(__name__)
from studio.services.projects import jobs as project_jobs, projects
from studio.services.booru import downloader


def run(job_id: int) -> int:
    """Body: returns the exit code (0 success / 1 failure)."""
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != "download":
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params = job.get("params_decoded") or {}

    def progress(line: str) -> None:
        print(line, flush=True)

    try:
        with db.connection_for() as conn:
            project = projects.get_project(conn, job["project_id"])
        if not project:
            progress(f"[error] project {job['project_id']} missing")
            return 1
        dest = projects.project_dir(project["id"], project["slug"]) / "download"
        sec = secrets.load()
        api_source = params.get("api_source", "gelbooru")
        if api_source == "danbooru":
            user_id = ""
            username = sec.danbooru.username
            api_key = sec.danbooru.api_key
        else:
            user_id = sec.gelbooru.user_id
            username = ""
            api_key = sec.gelbooru.api_key
        opts = downloader.DownloadOptions(
            tag=params.get("tag", ""),
            count=int(params.get("count", 0)),
            api_source=api_source,
            save_tags=sec.download.save_tags,
            convert_to_png=sec.download.convert_to_png,
            remove_alpha_channel=sec.download.remove_alpha_channel,
            user_id=user_id,
            username=username,
            api_key=api_key,
            exclude_tags=list(sec.download.exclude_tags),
        )
        progress(
            f"[start] tag={opts.tag!r} count={opts.count} "
            f"source={opts.api_source} "
            f"exclude={','.join(opts.exclude_tags) or '(none)'}"
        )
        saved = downloader.download(
            opts,
            dest,
            on_progress=progress,
            cancel_event=threading.Event(),  # supervisor cancels via SIGTERM
        )
        progress(f"[done] saved={saved}")
        return 0
    except Exception as exc:
        # PR-1 C7: same as tag_worker -- logger.exception carries the trace_id into
        # stderr, progress gives the human-readable short summary.
        logger.exception("download worker crashed (job_id=%s)", job_id)
        progress(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
