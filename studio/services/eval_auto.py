"""POC glue for automatically evaluating saved training checkpoints."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from studio import db
from studio import secrets
from studio.infrastructure.paths import task_eval_dir
from studio.services import (
    eval_ccip,
    eval_clip,
    eval_dino,
    eval_registry,
    eval_samples,
    eval_tag,
)
from studio.services.projects import jobs as project_jobs, projects, versions

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]

# Job kinds for post-training / manual evaluation (matches the router's _EVAL_JOB_KINDS).
EVAL_JOB_KINDS = ("eval_samples", "eval_clip", "eval_dino", "eval_tag", "eval_ccip")


def _clear_previous_task_eval(
    conn, project_id: int, version_id: int, task_id: int, vdir: Path, eval_root: Path
) -> None:
    """Clear this task's previous eval round before rerunning: cancel unfinished old
    eval jobs and delete old run files.

    Only "this round"'s data should show. Completed old jobs are **not** touched
    (their done/failed status stays as-is) — list_task_eval_jobs filters by whether
    the run still exists, so history stays clean and no cross-round merge is needed.
    """
    for j in project_jobs.list_jobs(conn, project_id=project_id, version_id=version_id):
        if j.get("kind") not in EVAL_JOB_KINDS:
            continue
        params = j.get("params_decoded") or {}
        if int(params.get("task_id") or 0) != task_id:
            continue
        if j.get("status") in ("pending", "running"):
            project_jobs.mark_canceled(conn, int(j["id"]))
    eval_samples.delete_all_runs(vdir, eval_root)


def _version_eval_enabled(project: dict[str, Any], version: dict[str, Any]) -> bool:
    """Per-version opt-in for post-training validation metrics (training config)."""
    from studio.services import version_config
    try:
        cfg = version_config.read_version_config(project, version)
    except Exception:
        return False
    return bool(cfg.get("eval_validation_enabled"))


def queue_training_finished_eval(
    conn,
    task: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Queue eval sample jobs for all saved LoRA checkpoints after training.

    Gated on the version's per-version opt-in (training config
    ``eval_validation_enabled``); the held-out validation set is the reference.
    Evaluation now always runs after training (inline / checkpoint-trigger were removed).
    """
    project_id = int(task.get("project_id") or 0)
    version_id = int(task.get("version_id") or 0)
    if not project_id or not version_id:
        return []

    project = projects.get_project(conn, project_id)
    version = versions.get_version(conn, version_id)
    if not project or not version or int(version["project_id"]) != project_id:
        return []

    if not _version_eval_enabled(project, version):
        return []

    vdir = versions.version_dir(project_id, str(project["slug"]), str(version["label"]))
    task_id = int(task.get("id") or 0)
    eval_root = task_eval_dir(task_id) if task_id else None
    queued: list[tuple[dict[str, Any], dict[str, Any]]] = []
    last_rel = ""
    for checkpoint in versions.list_lora_ckpts(vdir):
        rel = _checkpoint_relative_to_output(
            project_id,
            str(project.get("slug") or ""),
            version,
            str(checkpoint.get("path") or ""),
        )
        if not rel:
            continue
        try:
            job, run = eval_samples.start_job(
                conn,
                project,
                version,
                vdir,
                checkpoint_path=rel,
                auto_metrics=True,
                auto_source={
                    "task_id": int(task.get("id") or 0),
                    "epoch": (payload or {}).get("epoch"),
                    "step": (payload or {}).get("step"),
                    "trigger": "after_training",
                },
                eval_root=eval_root,
            )
        except Exception:
            logger.exception(
                "auto eval after-training enqueue failed for task=%s checkpoint=%s",
                task.get("id"),
                checkpoint.get("path"),
            )
            continue
        queued.append((job, run))
        last_rel = rel

    base = _ensure_baseline_queued(
        conn, project, version, vdir,
        task_id=task_id, eval_root=eval_root, ckpt_rel=last_rel, trigger="after_training",
    )
    if base is not None:
        queued.append(base)

    logger.info(
        "queued after-training eval sample jobs for task=%s count=%s",
        task.get("id"),
        len(queued),
    )
    return queued


def _has_baseline_run(vdir: Path, eval_root: Path | None) -> bool:
    """Whether this eval scope already has a baseline run (only generated once per task)."""
    for run in eval_samples.list_runs(vdir, eval_root):
        if run.get("baseline"):
            return True
    return False


def _ensure_baseline_queued(
    conn,
    project: dict[str, Any],
    version: dict[str, Any],
    vdir: Path,
    *,
    task_id: int,
    eval_root: Path | None,
    ckpt_rel: str,
    trigger: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Queue one baseline run per task (base model only, lora_scale=0, same
    prompt/seed) so each checkpoint's delta can be computed. Skipped if a baseline
    already exists, the feature is disabled, or there's no usable checkpoint. Shared
    by after-training and manual entry points — both need a baseline, otherwise a
    manual rerun would show no delta.
    """
    cfg = secrets.load().eval_metrics
    if not cfg.eval_baseline_enabled or not ckpt_rel:
        return None
    if _has_baseline_run(vdir, eval_root):
        return None
    try:
        return eval_samples.start_job(
            conn, project, version, vdir,
            checkpoint_path=ckpt_rel,
            auto_metrics=True,
            auto_source={"task_id": task_id, "trigger": trigger, "baseline": True},
            eval_root=eval_root,
            baseline=True,
        )
    except Exception:
        logger.exception("baseline eval enqueue failed for task=%s", task_id)
        return None


def queue_manual_task_eval(
    conn,
    task: dict[str, Any],
    checkpoints: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Queue task-scoped eval sample+metric jobs for an explicit checkpoint set.

    Unlike :func:`queue_training_finished_eval`, this is the *manual* entry point
    (a user clicking "Run evaluation" on a finished task), so it does NOT gate on the
    per-version opt-in.
    It evaluates the full validation set and reuses the Settings metric models,
    writing under ``tasks/<id>/eval/`` so results show up in that task's eval page.

    Every "Run evaluation" click **first automatically clears the previous round**
    (deletes old run files, cancels unfinished old eval jobs), so the eval page
    always shows only this round's data — no cross-round merge/dedup, since old
    values have no reference value. Completed old jobs keep their status (filtered
    by whether the run still exists, see list_task_eval_jobs), so history stays clean.
    """
    project_id = int(task.get("project_id") or 0)
    version_id = int(task.get("version_id") or 0)
    task_id = int(task.get("id") or 0)
    if not project_id or not version_id or not task_id:
        return []

    project = projects.get_project(conn, project_id)
    version = versions.get_version(conn, version_id)
    if not project or not version or int(version["project_id"]) != project_id:
        return []

    vdir = versions.version_dir(project_id, str(project["slug"]), str(version["label"]))
    eval_root = task_eval_dir(task_id)
    _clear_previous_task_eval(conn, project_id, version_id, task_id, vdir, eval_root)
    queued: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    baseline_rel = ""
    for raw in checkpoints:
        rel = _checkpoint_relative_to_output(
            project_id, str(project.get("slug") or ""), version, str(raw or "")
        )
        if not rel or rel in seen:
            continue
        seen.add(rel)
        if not baseline_rel:
            baseline_rel = rel
        try:
            job, run = eval_samples.start_job(
                conn,
                project,
                version,
                vdir,
                checkpoint_path=rel,
                auto_metrics=True,
                auto_source={"task_id": task_id, "trigger": "manual"},
                eval_root=eval_root,
            )
        except Exception:
            logger.exception(
                "manual eval enqueue failed for task=%s checkpoint=%s", task_id, raw
            )
            continue
        queued.append((job, run))

    base = _ensure_baseline_queued(
        conn, project, version, vdir,
        task_id=task_id, eval_root=eval_root, ckpt_rel=baseline_rel, trigger="manual",
    )
    if base is not None:
        queued.append(base)

    logger.info(
        "queued manual eval sample jobs for task=%s count=%s", task_id, len(queued)
    )
    return queued


def queue_metric_jobs_for_sample(
    conn,
    project: dict[str, Any],
    version: dict[str, Any],
    version_dir: Path,
    run_id: str,
    *,
    eval_root: Path | None = None,
    task_id: int | None = None,
) -> list[dict[str, Any]]:
    """Queue metric jobs (the runner matching each metric checked in Settings) after
    a sample run.

    Only queues the runners in the "enabled set"; reuses an existing active job
    instead of requeuing. clip/dino are wired into this table first — matches the
    original behavior when everything is enabled; ccip/tag runners were added to
    `queue_runners` later.
    """
    cfg = secrets.load().eval_metrics
    # runner key → (JOB_KIND, start_job, model_name)
    queue_runners = {
        "clip": (eval_clip.JOB_KIND, eval_clip.start_job, cfg.clip_model_name),
        "dino": (eval_dino.JOB_KIND, eval_dino.start_job, cfg.dino_model_name),
        "tag": (eval_tag.JOB_KIND, eval_tag.start_job, eval_tag.DEFAULT_MODEL_NAME),
        "ccip": (eval_ccip.JOB_KIND, eval_ccip.start_job, cfg.ccip_model_name),
    }
    jobs: list[dict[str, Any]] = []
    for runner_key in eval_registry.enabled_runners(cfg.enabled_metrics):
        spec = queue_runners.get(runner_key)
        if spec is None:
            continue
        kind, start_job, model_name = spec
        job = _active_metric_job(
            conn,
            project=project,
            version=version,
            kind=kind,
            run_id=run_id,
            task_id=task_id,
        )
        if job is None:
            job, _ = start_job(
                conn,
                project,
                version,
                version_dir,
                run_id,
                model_name=model_name,
                eval_root=eval_root,
                task_id=task_id,
            )
        jobs.append(job)
    logger.info(
        "queued auto eval metric jobs for run=%s: %s",
        run_id,
        [j.get("id") for j in jobs],
    )
    return jobs


def _resolve_task_checkpoint(
    conn,
    task: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, str] | None:
    project_id = int(task.get("project_id") or 0)
    version_id = int(task.get("version_id") or 0)
    if not project_id or not version_id:
        return None

    project = projects.get_project(conn, project_id)
    version = versions.get_version(conn, version_id)
    if not project or not version or int(version["project_id"]) != project_id:
        return None

    checkpoint = _checkpoint_relative_to_output(
        project_id,
        str(project.get("slug") or ""),
        version,
        str(payload.get("checkpoint_path") or ""),
    )
    if not checkpoint:
        return None

    vdir = versions.version_dir(project_id, str(project["slug"]), str(version["label"]))
    return project, version, vdir, checkpoint


def _active_metric_job(
    conn,
    *,
    project: dict[str, Any],
    version: dict[str, Any],
    kind: str,
    run_id: str,
    task_id: int | None = None,
) -> dict[str, Any] | None:
    for status in ("pending", "running"):
        for job in project_jobs.list_jobs(
            conn,
            project_id=int(project["id"]),
            version_id=int(version["id"]),
            kind=kind,
            status=status,
        ):
            params = job.get("params_decoded")
            if not isinstance(params, dict):
                continue
            if str(params.get("run_id") or "") == str(run_id):
                if task_id is not None and int(params.get("task_id") or 0) != int(task_id):
                    continue
                if task_id is None and int(params.get("task_id") or 0):
                    continue
                return job
    return None


def _checkpoint_relative_to_output(
    project_id: int,
    project_slug: str,
    version: dict[str, Any],
    raw_path: str,
) -> str | None:
    if not raw_path or not project_slug:
        return None
    vdir = versions.version_dir(project_id, project_slug, str(version["label"]))
    output_dir = (vdir / "output").resolve()
    path = Path(raw_path)
    if not path.is_absolute():
        path = output_dir / raw_path.replace("\\", "/")
    try:
        rel = path.resolve().relative_to(output_dir)
    except ValueError:
        logger.warning("auto eval checkpoint outside output dir: %s", raw_path)
        return None
    return rel.as_posix()
