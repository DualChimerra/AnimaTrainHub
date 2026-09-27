"""Version eval metric result endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ...schemas.projects import EvalClipStart, EvalDinoStart
from ._shared import _publish_job_state, _version_dir_or_404
from .... import db, secrets
from ....infrastructure.paths import task_eval_dir
from ....services import eval_clip, eval_dino, eval_metrics, eval_samples
from ....services.projects import jobs as project_jobs

router = APIRouter()

# Job kinds for post-training / manual evaluation (inline eval during training has no job, goes to the training log).
_EVAL_JOB_KINDS = ("eval_samples", "eval_clip", "eval_dino", "eval_tag", "eval_ccip")


@router.get("/api/projects/{pid}/versions/{vid}/eval/metrics")
def list_eval_metric_results_endpoint(
    pid: int,
    vid: int,
    task_id: int | None = None,
) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = task_eval_dir(task_id) if task_id else None
    try:
        results = eval_metrics.list_results(vdir, eval_root)
    except (eval_metrics.EvalMetricsError, eval_samples.EvalSamplesError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "metric_specs": eval_metrics.metric_specs(),
        "cache": eval_metrics.cache_layout(vdir, eval_root),
        "results": results,
    }


@router.get("/api/projects/{pid}/versions/{vid}/eval/jobs")
def list_task_eval_jobs_endpoint(
    pid: int, vid: int, task_id: int,
) -> dict[str, Any]:
    """List the post-training/manual evaluation jobs (eval_samples/clip/dino) for a task,
    letting the frontend associate a checkpoint row by run_id + fetch the raw log
    (job_log_appended / GET /api/jobs/{id}/log).

    The job_log_appended event carries no task_id, so a refresh loses the live
    association; this endpoint lets it be rediscovered. Inline eval during training has
    no job (it goes into the training log), so it's excluded here.

    Only returns jobs whose **run still exists**: each "run evaluation" deletes the
    previous run's files, so once an old job's run is gone it's filtered out, leaving the
    merged eval log with only this run's data. Doesn't change any job status or pollute history.
    """
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = task_eval_dir(task_id)
    with db.connection_for() as conn:
        rows = project_jobs.list_jobs(conn, project_id=pid, version_id=vid)
    out: list[dict[str, Any]] = []
    for j in rows:
        if j.get("kind") not in _EVAL_JOB_KINDS:
            continue
        params = j.get("params_decoded") or {}
        if int(params.get("task_id") or 0) != task_id:
            continue
        run_id = params.get("run_id")
        if not run_id or not eval_samples.run_path(vdir, str(run_id), eval_root).exists():
            continue  # run has been cleared -> don't show this historical job anymore
        out.append({
            "id": j.get("id"),
            "kind": j.get("kind"),
            "status": j.get("status"),
            "run_id": run_id,
            "checkpoint_path": params.get("checkpoint_path"),
        })
    return {"jobs": out}


@router.get("/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}/metrics")
def get_eval_metric_result_endpoint(
    pid: int,
    vid: int,
    run_id: str,
    task_id: int | None = None,
) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    eval_root = task_eval_dir(task_id) if task_id else None
    try:
        result = eval_metrics.load_result(vdir, run_id, eval_root)
    except (eval_metrics.EvalMetricsError, eval_samples.EvalSamplesError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if result is None:
        raise HTTPException(404, f"Eval sample run not found: {run_id}")
    return {"metric_specs": eval_metrics.metric_specs(), "result": result}


@router.post("/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}/metrics/clip")
def start_eval_clip_metrics_endpoint(
    pid: int, vid: int, run_id: str, body: EvalClipStart
) -> dict[str, Any]:
    p, v, vdir = _version_dir_or_404(pid, vid)
    cfg = secrets.load().eval_metrics
    try:
        with db.connection_for() as conn:
            job, result = eval_clip.start_job(
                conn,
                p,
                v,
                vdir,
                run_id,
                model_name=body.model_name or cfg.clip_model_name,
            )
    except (
        eval_clip.EvalClipError,
        eval_metrics.EvalMetricsError,
        eval_samples.EvalSamplesError,
    ) as exc:
        raise HTTPException(400, str(exc)) from exc
    _publish_job_state(job)
    return {"job": job, "result": result}


@router.post("/api/projects/{pid}/versions/{vid}/eval/samples/{run_id}/metrics/dino")
def start_eval_dino_metrics_endpoint(
    pid: int, vid: int, run_id: str, body: EvalDinoStart
) -> dict[str, Any]:
    p, v, vdir = _version_dir_or_404(pid, vid)
    cfg = secrets.load().eval_metrics
    try:
        with db.connection_for() as conn:
            job, result = eval_dino.start_job(
                conn,
                p,
                v,
                vdir,
                run_id,
                model_name=body.model_name or cfg.dino_model_name,
            )
    except (
        eval_dino.EvalDinoError,
        eval_metrics.EvalMetricsError,
        eval_samples.EvalSamplesError,
    ) as exc:
        raise HTTPException(400, str(exc)) from exc
    _publish_job_state(job)
    return {"job": job, "result": result}
