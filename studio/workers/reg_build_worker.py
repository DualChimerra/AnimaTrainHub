"""Regularization set build worker (PP5 + PP5.1 + PP5.5).

`python -m studio.workers.reg_build_worker --job-id N`. Reads `project_jobs.params`:
    {
      "version_id": int,
      "target_count": int | null,        # null = use train's total image count
      "excluded_tags": [str, ...],
      "auto_tag": bool,
      "api_source": "gelbooru" | "danbooru",  # optional, default gelbooru
      "incremental": bool,                    # PP5.1, optional, default False
    }

Credentials are pulled from `secrets.gelbooru` / `secrets.danbooru`.

Workflow:
1. reg_builder.build(opts) downloads images + writes meta.json (auto_tagged=False, postprocessed_at=None)
2. PP5.5 -- reg_postprocess.postprocess(reg_dir) clusters by resolution + smart-resizes to a uniform resolution
   - failure / no K found that satisfies max_crop -> caught, meta.postprocessed_at stays None; reg set is kept
3. if auto_tag, calls WD14 inline to tag every image in reg/
4. failure caught -> meta.auto_tagged stays false; the reg set itself is kept

No subprocess spawned: WD14 / postprocess are imported directly, progress goes through the
same log_path.

Logging goes through stdout only: see the note at the top of `download_worker.py`.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

# PR-1 C4: setup_logging already calls reconfigure_console_utf8 uniformly.

logger = logging.getLogger(__name__)

# PP9.5 -- must be imported before any `import onnxruntime`, to trigger the top-level preload.
# The auto_tag path calls wd14_tagger inline (line ~105 `get_tagger("wd14")`); the worker
# is an independent subprocess and must import it itself -- otherwise the CUDA EP silently
# falls back to CPU with no signal visible to the user.
from studio.services.runtime import onnxruntime as onnxruntime_setup  # noqa: F401

from studio import db, secrets
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services.dataset.scan import IMAGE_EXTS
from studio.services.reg import (
    builder as reg_builder,
    dedup as reg_dedup,
    postprocess as reg_postprocess,
)
from studio.services.dataset import tagedit


def _collect_reg_images(reg_dir: Path) -> list[Path]:
    """Recursively collect all images under the reg directory (including mirrored subfolders)."""
    if not reg_dir.exists():
        return []
    out: list[Path] = []
    for f in reg_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            out.append(f)
    return sorted(out)


def _run_postprocess(
    reg_dir: Path, progress, cancel_event,
    *, method: str = "smart", max_crop_ratio: float = 0.1,
) -> None:
    """PP5.5 -- resolution clustering post-process. A failure isn't fatal; meta fields reflect the result."""
    import time as _time
    try:
        result = reg_postprocess.postprocess(
            reg_dir,
            method=method,
            max_crop_ratio=max_crop_ratio,
            on_progress=progress,
            cancel_event=cancel_event,
        )
    except Exception as exc:
        progress(f"[postprocess] failed: {exc}")
        progress(traceback.format_exc())
        reg_builder.update_meta_postprocess(
            reg_dir, when=None, clusters=None, method=None, max_crop_ratio=None
        )
        return
    if result.get("clusters") is None:
        # no K found that satisfies max_crop -> leave files untouched
        reg_builder.update_meta_postprocess(
            reg_dir, when=None, clusters=None,
            method=result.get("method"),
            max_crop_ratio=result.get("max_crop_ratio"),
        )
        return
    reg_builder.update_meta_postprocess(
        reg_dir,
        when=_time.time(),
        clusters=int(result["clusters"]),
        method=str(result.get("method") or method),
        max_crop_ratio=float(result.get("max_crop_ratio") or max_crop_ratio),
    )


def _run_auto_tag(reg_dir: Path, progress, kind: str = "wd14") -> bool:
    """Runs a tagger inline to tag the reg set, returns False on failure.

    A3 -- `kind` goes through `studio.services.tagging.base.get_tagger`; the UI currently
    exposes wd14 / cltagger, but the underlying layer supports the full VALID_TAGGER_NAMES
    set. LLM / JoyCaption will be added in a later PR; note their cost/latency against the
    reg image count (which can be larger than train) will feel slow/expensive.
    """
    images = _collect_reg_images(reg_dir)
    if not images:
        progress("[auto-tag] no images, skipping")
        return False
    progress(f"[auto-tag] starting {kind}, {len(images)} images")
    try:
        from studio.services.tagging.base import get_tagger
        tagger = get_tagger(kind)
        tagger.prepare()
        progress(f"[auto-tag] {kind} model ready")
        ok = 0
        errs = 0
        for r in tagger.tag(
            images,
            on_progress=lambda d, t: progress(f"[auto-tag] {d}/{t}"),
        ):
            if r.get("error"):
                progress(f"[auto-tag err] {r['image'].name}: {r['error']}")
                errs += 1
                continue
            tagedit.write_tags(r["image"], r.get("tags") or [])
            ok += 1
        progress(f"[auto-tag] done {ok}/{len(images)} (errors={errs})")
        return ok > 0
    except Exception as exc:
        progress(f"[auto-tag] failed: {exc}")
        progress(traceback.format_exc())
        return False


def run(job_id: int) -> int:
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != "reg_build":
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params: dict[str, Any] = job.get("params_decoded") or {}

    cancel_event = threading.Event()  # supervisor cancels via SIGTERM; kept here only for API completeness

    def progress(line: str) -> None:
        print(line, flush=True)

    try:
        version_id = int(params["version_id"])
        with db.connection_for() as conn:
            v = versions.get_version(conn, version_id)
            if not v or v["project_id"] != job["project_id"]:
                progress(f"[error] version {version_id} not in project {job['project_id']}")
                return 1
            p = projects.get_project(conn, v["project_id"])
        assert p is not None

        vdir = versions.version_dir(p["id"], p["slug"], v["label"])
        train_dir = vdir / "train"
        output_dir = vdir / "reg"  # consistent with the original script: mirrors train's subfolders directly

        sec = secrets.load()
        api_source = str(params.get("api_source", "gelbooru"))
        if api_source == "danbooru":
            user_id = ""
            username = sec.danbooru.username
            api_key = sec.danbooru.api_key
            # account_type affects the max_search_tags ceiling
            account_type = (sec.danbooru.account_type or "free").lower()
            max_search_tags = {
                "free": 2, "gold": 6, "platinum": 12,
            }.get(account_type, 2)
        else:
            user_id = sec.gelbooru.user_id
            username = ""
            api_key = sec.gelbooru.api_key
            max_search_tags = 20  # gelbooru default is 20

        opts = reg_builder.RegBuildOptions(
            train_dir=train_dir,
            output_dir=output_dir,
            api_source=api_source,
            user_id=user_id,
            api_key=api_key,
            username=username,
            target_count=params.get("target_count"),  # B1: None = use train's total count / ignored in mirror mode
            max_search_tags=max_search_tags,
            # batch_size = "how many images per batch before recomputing missing tags"
            # inside the search loop, unrelated to mirroring train's subfolders; uses the
            # original script's default of 5, not exposed in the UI
            skip_similar=bool(params.get("skip_similar", True)),
            aspect_ratio_filter_enabled=bool(
                params.get("aspect_ratio_filter_enabled", False)
            ),
            min_aspect_ratio=float(params.get("min_aspect_ratio", 0.5)),
            max_aspect_ratio=float(params.get("max_aspect_ratio", 2.0)),
            excluded_tags=list(params.get("excluded_tags") or []),
            blacklist_tags=list(sec.download.exclude_tags or []),
            auto_tag=bool(params.get("auto_tag", True)),
            auto_tag_kind=str(params.get("auto_tag_kind") or "wd14"),
            auto_dedup=bool(params.get("auto_dedup", True)),
            build_mode=str(params.get("build_mode") or "flat"),
            based_on_version=v["label"],
            save_tags=sec.download.save_tags,
            convert_to_png=sec.download.convert_to_png,
            remove_alpha_channel=sec.download.remove_alpha_channel,
        )
        incremental = bool(params.get("incremental", True))
        pp_method = str(params.get("postprocess_method", "smart"))
        pp_max_crop = float(params.get("postprocess_max_crop_ratio", 0.1))
        progress(
            f"[start] version={v['label']} api={api_source} "
            f"max_tags={max_search_tags} auto_tag={opts.auto_tag} "
            f"incremental={incremental} auto_dedup={opts.auto_dedup} "
            f"pp={pp_method}/{pp_max_crop}"
        )

        # full mode: clears reg/ first (images, subfolders, meta, .deleted_ids.json) --
        # the user's intent is "start from zero". incremental mode keeps all existing content.
        if not incremental and output_dir.exists():
            progress("[start] full mode: clearing existing contents of reg/")
            reg_builder.clear_reg_dir(output_dir)

        meta = reg_builder.build(
            opts,
            on_progress=progress,
            cancel_event=cancel_event,
            incremental=incremental,
        )
        progress(f"[reg-done] actual={meta.actual_count}/{meta.target_count}")

        # A4 -- auto_dedup: after build, scan for duplicates -> keep 1 per group, delete
        # the rest -> if that leaves a shortfall, top up incrementally. At most
        # MAX_DEDUP_ROUNDS rounds, exits early once a round deletes 0.
        if opts.auto_dedup and meta.actual_count > 0:
            MAX_DEDUP_ROUNDS = 3
            for r in range(MAX_DEDUP_ROUNDS):
                if cancel_event.is_set():
                    progress("[dedup] aborted by user")
                    break
                progress(f"[dedup r{r + 1}/{MAX_DEDUP_ROUNDS}] scanning for duplicates...")
                to_delete = reg_dedup.scan_for_dedup(output_dir)
                if not to_delete:
                    progress(f"[dedup r{r + 1}] nothing to delete, done")
                    break
                purged = reg_dedup.purge_paths(output_dir, to_delete)
                progress(f"[dedup r{r + 1}] deleted {purged['count']} images")
                if purged["count"] == 0:
                    break  # scan found candidates but every unlink failed / was already gone -> avoid an infinite loop
                meta = reg_builder.read_meta(output_dir) or meta
                shortfall = meta.target_count - meta.actual_count
                if shortfall <= 0:
                    progress(f"[dedup r{r + 1}] target already reached, done")
                    break
                progress(
                    f"[dedup r{r + 1}] short {shortfall} images, topping up incrementally"
                )
                meta = reg_builder.build(
                    opts,
                    on_progress=progress,
                    cancel_event=cancel_event,
                    incremental=True,
                )
                progress(
                    f"[dedup r{r + 1}] after top-up actual={meta.actual_count}/{meta.target_count}"
                )

        # PP5.5 -- resolution clustering post-process (before auto_tag, since tagging
        # works on the final images). A4 ordering: must run after dedup -- postprocess
        # resizes images, which changes their phash.
        if meta.actual_count > 0:
            _run_postprocess(
                output_dir, progress, cancel_event,
                method=pp_method, max_crop_ratio=pp_max_crop,
            )

        # auto_tag: run the selected tagger inline after the download + post-process are done
        auto_ok = False
        if opts.auto_tag and meta.actual_count > 0:
            auto_ok = _run_auto_tag(output_dir, progress, kind=opts.auto_tag_kind)
            reg_builder.update_meta_auto_tagged(
                output_dir, auto_ok, kind=opts.auto_tag_kind,
            )

        return 0 if meta.actual_count > 0 else 1
    except Exception as exc:
        # PR-1 C7: same as tag_worker -- logger.exception carries the trace_id into
        # stderr, progress gives the human-readable short summary.
        logger.exception("reg_build worker crashed (job_id=%s)", job_id)
        progress(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
