"""Upload worker subprocess entry point (fix 524).

Launched by supervisor: `python -m studio.workers.upload_worker --job-id N`.

Why a worker is needed: `/upload` used to synchronously unzip + re-encode every image
through convert_to_png; a 184MB package easily takes >100s, and Cloudflare's 100s timeout
returns a straight 524. Moving it to a background job lets the endpoint respond instantly;
the actual unzip / transcode runs here, and the frontend polls `upload/status` for
progress + results.

job.params:
- ``staging_dir``: the endpoint drops the raw uploaded files into this temp directory;
  the whole directory is deleted once processing finishes.
- ``paths``: a directly given list of server-visible paths (used by upload-from-path),
  **not** deleted after processing.
At least one of the two must be present.

Results (added / skipped) are written to `jobs/{id}.result.json`, read back by the status
endpoint. Logging goes through stdout only (see the note in download_worker).
"""
from __future__ import annotations

import logging
from pathlib import Path

from studio import db, secrets

logger = logging.getLogger(__name__)
from studio.services.projects import jobs as project_jobs, projects
from studio.services.dataset import uploads as uploads_svc


def _gather_sources(params: dict) -> tuple[list[Path], Path | None]:
    """Resolve params into the list of files to process + the staging directory to clean up (None if there isn't one)."""
    staging = params.get("staging_dir")
    if staging:
        staging_dir = Path(staging)
        if not staging_dir.is_dir():
            return [], staging_dir
        sources = sorted(p for p in staging_dir.iterdir() if p.is_file())
        return sources, staging_dir
    paths = params.get("paths") or []
    return [Path(p) for p in paths], None


def run(job_id: int) -> int:
    """Body: returns the exit code (0 success / 1 failure)."""
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != "upload":
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params = job.get("params_decoded") or {}

    def progress(line: str) -> None:
        print(line, flush=True)

    staging_dir: Path | None = None
    try:
        with db.connection_for() as conn:
            project = projects.get_project(conn, job["project_id"])
        if not project:
            progress(f"[error] project {job['project_id']} missing")
            return 1
        dest = projects.project_dir(project["id"], project["slug"]) / "download"

        sources, staging_dir = _gather_sources(params)
        if not sources:
            progress("[error] no files to process")
            project_jobs.write_result(job_id, {"added": [], "skipped": []})
            return 1

        sec = secrets.load()
        progress(
            f"[start] files={len(sources)} convert_to_png={sec.download.convert_to_png}"
        )
        result = uploads_svc.ingest_paths(
            sources, dest,
            convert_to_png=sec.download.convert_to_png,
            remove_alpha_channel=sec.download.remove_alpha_channel,
            on_progress=progress,
        )
        project_jobs.write_result(job_id, result.as_dict())
        progress(
            f"[done] added={len(result.added)} skipped={len(result.skipped)}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001 -- same as download_worker
        logger.exception("upload worker crashed (job_id=%s)", job_id)
        progress(f"[error] {exc}")
        project_jobs.write_result(job_id, {"added": [], "skipped": []})
        return 1
    finally:
        # staging is the temp directory the endpoint created for this upload; clean it up
        # regardless of success or failure once processing is done.
        if staging_dir is not None and staging_dir.is_dir():
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
