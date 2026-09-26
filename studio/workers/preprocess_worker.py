"""Preprocess worker subprocess entry point (upscale + crop).

Launched by supervisor: `python -m studio.workers.preprocess_worker --job-id N`.

Reads the project_jobs row -> dispatches by `params['stage']`:
  - stage='upscale' (default): calls `studio.services.upscaler.upscale_file()` serially
  - stage='crop': uses PIL to cut the images under preprocess/ into N output pieces
    according to normalized rects

Logging convention: stdout only (supervisor redirects it to the log file); don't open
that same log file again, to avoid LogTailer reading it twice.

Cancellation: the worker body checks for a SIGTERM/CTRL_BREAK signal before each image
(the Python interpreter's default behavior is to raise KeyboardInterrupt on the main
thread for SIGTERM); it exits cleanly after finishing the current image, keeping whatever
output has already been written to disk (incremental).
"""
from __future__ import annotations

import json
import logging
import math
import signal
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

logger = logging.getLogger(__name__)

from studio import db
from studio.domain.errors import DomainError
from studio.services.preprocess import core as preprocess
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services import models as model_downloader
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.preprocess import masks as train_masks
from studio.services.inference import upscaler


_stop_requested = False


def _on_signal(_signum, _frame) -> None:  # pragma: no cover - signal path
    global _stop_requested
    _stop_requested = True


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGBREAK"):  # Windows
        signal.signal(signal.SIGBREAK, _on_signal)  # type: ignore[attr-defined]


def _unlink_image_and_sidecars(path: Path, *, keep_sidecars: bool = False) -> None:
    """Remove an image and caption sidecars that are no longer part of train/."""
    if path.is_file():
        path.unlink(missing_ok=True)
    if keep_sidecars:
        return
    for ext in (".txt", ".json"):
        path.with_suffix(ext).unlink(missing_ok=True)


def run(job_id: int) -> int:  # noqa: PLR0912, PLR0915 - main flow is linear and readable
    _install_signal_handlers()

    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    if not job:
        print(f"[error] job {job_id} not found", flush=True)
        return 1
    if job["kind"] != preprocess.PREPROCESS_KIND:
        print(f"[error] wrong kind: {job['kind']}", flush=True)
        return 1

    params = job.get("params_decoded") or {}
    # missing stage field is treated as a legacy upscale job (backward compat)
    stage = params.get("stage", preprocess.STAGE_UPSCALE)

    def log(line: str) -> None:
        print(line, flush=True)

    def emit_event(evt_type: str, **payload) -> None:
        """Marked stdout line -> parsed by supervisor -> SSE. Used for real-time frontend
        updates, does not go into the job log. Supervisor-side constant is
        `studio/supervisor.py:_EVENT_MARKER`."""
        try:
            print(f"__EVENT__:{evt_type}:{json.dumps(payload, ensure_ascii=False)}", flush=True)
        except Exception:  # noqa: BLE001 -- a failed event emit shouldn't affect the main flow
            pass

    try:
        with db.connection_for() as conn:
            project = projects.get_project(conn, job["project_id"])
        if not project:
            log(f"[error] project {job['project_id']} missing")
            return 1

        version_id = job.get("version_id")
        if version_id is None:
            log("[error] preprocess job is missing version_id (ADR 0010 train scope)")
            return 1
        with db.connection_for() as conn:
            version = versions.get_version(conn, version_id)
        if not version:
            log(f"[error] version {version_id} missing")
            return 1
        if stage == preprocess.STAGE_CROP:
            return _run_crop_train(project, version, params, log, emit_event)
        if stage == preprocess.STAGE_UPSCALE:
            return _run_upscale_train(
                project, version, params, log, emit_event,
            )
        log(f"[error] unknown stage: {stage!r}")
        return 1
    except Exception as exc:  # noqa: BLE001
        # PR-1 C7: same as tag_worker -- logger.exception carries the trace_id into
        # stderr, log gives the human-readable short summary.
        logger.exception("preprocess worker crashed (job_id=%s)", job_id)
        log(f"[error] {exc}")
        return 1


def _run_upscale_train(
    project: dict[str, Any],
    version: dict[str, Any],
    params: dict[str, Any],
    log: Callable[[str], None],
    emit_event: Callable[..., None],
) -> int:
    """ADR 0010 train-scope upscale.

    Source + output both live in `versions/{label}/train/{folder}/`, the manifest is
    written to `versions/{label}/train/manifest.json`. ADR 0010 fixup (2026-06-04):
    **doesn't change the extension** -- overwrites in place under the same name
    (X.jpg -> X.jpg / X.png -> X.png), to avoid breaking caption correspondence /
    dataset_config extension globs. upscaler saves using the src extension (JPEG
    quality=95 / PNG uncompressed / WebP quality=95); the manifest entry gets a
    `processed=True` flag for the UI's badge inference.
    """
    mode = params.get("mode", "all")
    names = params.get("names") or None
    model_label = params.get("model", preprocess.DEFAULT_MODEL)
    tile_size = int(params.get("tile_size", preprocess.DEFAULT_TILE_SIZE))
    tile_pad = int(params.get("tile_pad", preprocess.DEFAULT_TILE_PAD))
    device = params.get("device", preprocess.DEFAULT_DEVICE)
    target_area_raw = params.get("target_area", preprocess.DEFAULT_TARGET_AREA)
    target_area = int(target_area_raw) if target_area_raw else None

    project_dir = projects.project_dir(project["id"], project["slug"])
    train_dir = preprocess.version_train_dir(project, version["label"])
    train_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_downloader.upscaler_target(model_label)
    if not model_path.exists():
        log(
            f"[error] model weights not found: {model_path} (download {model_label} on the settings page first)"
        )
        return 1

    try:
        sources = preprocess.resolve_targets_train(
            project, version["label"], mode=mode, names=names
        )
    except DomainError as exc:
        log(f"[error] failed to resolve targets: {exc}")
        return 1

    total = len(sources)
    if total == 0:
        log("[done] no images to process")
        return 0

    target_desc = (
        f"{int(math.sqrt(target_area))}²={target_area}px"
        if target_area else "off (direct 4x)"
    )
    log(
        f"[start] mode={mode} model={model_label} tile={tile_size}+{tile_pad} "
        f"device={device} target={target_desc} total={total} scope=train"
    )

    try:
        import torch
        resolved_dev = upscaler.resolve_device(device)
        resolved_dtype = upscaler.resolve_dtype("auto", resolved_dev)
        gpu_name = (
            torch.cuda.get_device_name(0)
            if resolved_dev.type == "cuda" and torch.cuda.is_available()
            else "-"
        )
        log(
            f"[device] resolved={resolved_dev} dtype={str(resolved_dtype).replace('torch.', '')} "
            f"gpu={gpu_name} cuda_available={torch.cuda.is_available()}"
        )
        upscaler.load_model(model_path, device=resolved_dev, dtype=resolved_dtype)
        log(f"[model] {model_label} loaded -> {resolved_dev}")
    except Exception as exc:  # noqa: BLE001
        log(f"[device] diagnostic failed: {exc} (continuing, but it may run on CPU)")

    succeeded = 0
    failed = 0
    skipped = 0

    for idx, src_rel in enumerate(sources, start=1):
        if _stop_requested:
            log(f"[cancel] cancellation signal received, processed {idx - 1}/{total}")
            break
        src_path = train_dir / src_rel
        if not src_path.exists():
            log(f"[skip] ({idx}/{total}) {src_rel}: source no longer exists")
            skipped += 1
            emit_event(
                "preprocess_progress",
                idx=idx, total=total, name=src_rel, status="skip",
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
            continue

        # origin follows the manifest's existing entry (multi-crop derived root),
        # otherwise use the last segment of the rel path (curate writes the copied
        # image's file name == origin)
        existing = preprocess_manifest.train_get_entry(
            project_dir, version["label"], src_rel
        )
        src_filename = src_rel.rsplit("/", 1)[-1]
        if existing is not None:
            origin_name = preprocess_manifest.entry_origin(existing, src_filename)
        else:
            origin_name = src_filename

        # ADR 0010 fixup: dst == src, overwritten in place. upscaler saves using the src
        # extension (JPEG 95 / WebP 95 / PNG uncompressed), preserving caption +
        # dataset_config's dependency on the extension; manifest entry gets a
        # processed=True flag.
        dst_path = src_path
        src_ext = Path(src_filename).suffix.lower()
        if src_ext in (".jpg", ".jpeg"):
            save_kwargs: dict[str, Any] = {"format": "JPEG", "quality": 95}
        elif src_ext == ".webp":
            save_kwargs = {"format": "WEBP", "quality": 95, "method": 6}
        else:
            save_kwargs = {"format": "PNG", "optimize": False}

        log(f"[upscale] ({idx}/{total}) {src_rel}")
        try:
            meta = upscaler.upscale_file(
                src_path,
                dst_path,
                model_path=model_path,
                label=model_label,
                tile_size=tile_size,
                tile_pad=tile_pad,
                device=device,
                target_area=target_area,
                on_log=log,
                prewarm_thumb_sizes=[256, 768],
                save_kwargs=save_kwargs,
            )
            meta["origin"] = origin_name
            meta["processed"] = True
            preprocess_manifest.train_add_processed(
                project_dir, version["label"], src_rel, meta,
            )
            # mask sidecar follows along: NEAREST resize to the upscaled size (no-op if no mask)
            try:
                with Image.open(dst_path) as up_img:
                    train_masks.resize_mask_like(train_dir, src_rel, up_img.size)
            except Exception as exc:  # noqa: BLE001
                log(f"   Warning: mask failed to follow upscale: {exc}")
            succeeded += 1
            emit_event(
                "preprocess_progress",
                idx=idx, total=total, name=src_rel, status="done",
                action=meta.get("action"),
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[fail] {src_rel}: {exc}")
            failed += 1
            emit_event(
                "preprocess_progress",
                idx=idx, total=total, name=src_rel, status="fail",
                error=str(exc)[:200],
                succeeded=succeeded, failed=failed, skipped=skipped,
            )

    log(f"[done] succeeded={succeeded} failed={failed} skipped={skipped}")
    return 0


def _run_crop_train(
    project: dict[str, Any],
    version: dict[str, Any],
    params: dict[str, Any],
    log: Callable[[str], None],
    emit_event: Callable[..., None],
) -> int:
    """ADR 0010 train-scope crop.

    `params['crops']` = `{rel_path: [rects]}`, rel_path looks like `1_data/X.png`.
    crop output goes into the same folder: N=1 produces `folder/stem.png`, N>1 fans out
    into `folder/stem_c0.png` / `folder/stem_c1.png` / ...; on success, old source images
    that are no longer part of the outputs are cleaned up, then train_replace_with_crops
    atomically replaces the manifest.
    """
    project_dir = projects.project_dir(project["id"], project["slug"])
    train_dir = preprocess.version_train_dir(project, version["label"])
    train_dir.mkdir(parents=True, exist_ok=True)

    crops_param = params.get("crops") or {}
    if not crops_param:
        log("[done] crops is empty, nothing to do")
        return 0
    sources = sorted(crops_param.keys())

    _last_emit_at = [0.0]

    def emit_throttled(*, force: bool, **payload) -> None:
        now = time.monotonic()
        if not force and (now - _last_emit_at[0]) < 1.0:
            return
        _last_emit_at[0] = now
        emit_event("crop_progress", **payload)

    total = len(sources)
    log(f"[start] stage=crop total={total} scope=train")

    succeeded = 0
    failed = 0
    skipped = 0

    for idx, src_rel in enumerate(sources, start=1):
        if _stop_requested:
            log(f"[cancel] cancellation signal received, processed {idx - 1}/{total}")
            break
        is_last = idx == total
        try:
            preprocess._validate_rel_name(src_rel)
        except DomainError as exc:
            log(f"[skip] {src_rel}: {exc}")
            skipped += 1
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_rel, status="skip",
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
            continue

        src_path = train_dir / src_rel
        if not src_path.is_file():
            log(f"[skip] ({idx}/{total}) {src_rel}: source doesn't exist")
            skipped += 1
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_rel, status="skip",
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
            continue

        # origin follows the manifest's existing entry root, otherwise use the src filename
        existing = preprocess_manifest.train_get_entry(
            project_dir, version["label"], src_rel
        )
        src_filename = src_rel.rsplit("/", 1)[-1]
        if existing is not None:
            origin = preprocess_manifest.entry_origin(existing, src_filename)
        else:
            origin = src_filename

        rects = crops_param[src_rel]
        n = len(rects)
        folder, _ = src_rel.split("/", 1)
        src_stem = Path(src_filename).stem
        out_rels = (
            [f"{folder}/{src_stem}.png"] if n == 1
            else [f"{folder}/{src_stem}_c{i}.png" for i in range(n)]
        )

        log(f"[crop] ({idx}/{total}) {src_rel} -> {n} outputs")
        try:
            t0 = time.monotonic()
            with Image.open(src_path) as raw:
                raw.load()
                src_img = raw.convert("RGB") if raw.mode != "RGB" else raw.copy()
            sw, sh = src_img.size
            outputs: list[dict[str, Any]] = []
            crop_boxes: list[tuple[int, int, int, int]] = []
            for r, out_rel in zip(rects, out_rels):
                left = int(round(r["x"] * sw))
                top = int(round(r["y"] * sh))
                right = int(round((r["x"] + r["w"]) * sw))
                bottom = int(round((r["y"] + r["h"]) * sh))
                right = max(left + 1, right)
                bottom = max(top + 1, bottom)
                crop_boxes.append((left, top, right, bottom))
                piece = src_img.crop((left, top, right, bottom))
                out_path = train_dir / out_rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                piece.save(tmp_path, format="PNG", optimize=False)
                import os as _os
                _os.replace(tmp_path, out_path)
                try:
                    st = out_path.stat()
                    sz, mt = st.st_size, st.st_mtime
                except OSError:
                    sz, mt = 0, time.time()
                outputs.append({
                    "name": out_rel,
                    "origin": origin,
                    "size": sz,
                    "mtime": mt,
                })

            # The output may switch to .png or fan out into several files; when the source
            # is no longer one of the outputs it must be deleted, otherwise train-only
            # data copied in from a bundle/version would keep both the original and the
            # cropped images. Known edge case (carried over from old behavior): `{stem}.png`
            # is included in the stale set to clean up outputs from historical N=1 crops;
            # if train happens to have two separate images with the same stem (X.jpg +
            # X.png), fanning out X.jpg will also delete the unrelated X.png.
            stale_rels = {src_rel, f"{folder}/{src_stem}.png"} - set(out_rels)
            output_stems = {Path(rel).stem for rel in out_rels}
            for stale_rel in sorted(stale_rels):
                stale_path = train_dir / stale_rel
                has_sidecar = (
                    stale_path.with_suffix(".txt").exists()
                    or stale_path.with_suffix(".json").exists()
                )
                if stale_path.exists() or has_sidecar:
                    try:
                        _unlink_image_and_sidecars(
                            stale_path,
                            keep_sidecars=Path(stale_rel).stem in output_stems,
                        )
                    except OSError as exc:
                        log(f"   Warning: failed to delete old {stale_rel}: {exc}")

            # mask sidecar follows along: same box crop + fan-out (no-op if source has no mask)
            try:
                train_masks.crop_mask_like(
                    train_dir, src_rel, crop_boxes, out_rels,
                )
            except Exception as exc:  # noqa: BLE001
                log(f"   Warning: mask failed to follow crop: {exc}")

            preprocess_manifest.train_replace_with_crops(
                project_dir, version["label"],
                source_name=src_rel,
                outputs=outputs,
            )
            # thumb prewarm
            try:
                from studio.services.dataset import thumb_cache
                for out_rel in out_rels:
                    out_path = train_dir / out_rel
                    with Image.open(out_path) as piece:
                        piece.load()
                        thumb_cache.prewarm_from_image(out_path, piece, [256, 768])
            except Exception as exc:  # noqa: BLE001
                log(f"   ⚠ thumb prewarm failed: {exc}")

            elapsed = time.monotonic() - t0
            succeeded += 1
            log(
                f"   OK {src_rel} -> {', '.join(out_rels)}  "
                f"({sw}x{sh} -> {n} piece(s), {elapsed:.2f}s)"
            )
            emit_throttled(
                force=(idx == 1 or is_last),
                idx=idx, total=total, name=src_rel, status="done",
                n_out=n, outputs=out_rels,
                succeeded=succeeded, failed=failed, skipped=skipped,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[fail] {src_rel}: {exc}")
            failed += 1
            emit_throttled(
                force=True,
                idx=idx, total=total, name=src_rel, status="fail",
                error=str(exc)[:200],
                succeeded=succeeded, failed=failed, skipped=skipped,
            )

    log(f"[done] succeeded={succeeded} failed={failed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    from ._base import worker_main
    worker_main(run)
