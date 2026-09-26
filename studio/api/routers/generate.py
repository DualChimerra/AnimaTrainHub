"""Test generation + daemon control + TAEFlux (PR-6 commit 5, extracted from server.py).

8 routes:
    POST /api/generate                          start a test generate task (runs on daemon)
    GET  /api/generate/{task_id}                query test task status
    GET  /api/generate/taeflux/status           whether the mid-step preview model is ready
    POST /api/generate/taeflux/install          synchronously download TAEFlux (~1.6MB, seconds)
    GET  /api/generate/daemon/status            daemon state / model_loaded / busy
    GET  /api/generate/daemon/logs              ring buffer log (since_seq / limit)
    POST /api/generate/daemon/unload            manual unload (409 while busy)
    GET  /api/generate/{task_id}/sample/{filename}  fetch PNG bytes from generate_cache

Test generation is not persisted (since commit 10): the daemon pushes PNG bytes back to
the server as base64 into generate_cache (an in-memory dict); HTTP reads from that cache.
The tempdir only holds config.json; when the task ends the supervisor still calls
cleanup_generate_tempdir to remove the empty directory. Server restart -> in-memory cache
is gone automatically; a force-kill leaves nothing behind either.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
import zlib
from datetime import date
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from ..deps import _resolve_model_paths
from ..errors import _validate_component_or_400
from ..schemas.generate import GenerateRequest
from ... import db, secrets
from ...domain import GenerateConfig
from ...domain.errors import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from ...domain.comfy_parity import force_comfy_parity_runtime_config
from ...infrastructure.event_bus import bus
from ...infrastructure.paths import STUDIO_DATA

router = APIRouter()
logger = logging.getLogger(__name__)

TEST_IMAGES_DIR = STUDIO_DATA / "test"


def _write_generate_cover(task_id: Optional[int], cover_path: Path) -> None:
    """0.17 P-I forward-write: when saving to disk, write the cover image's (on-disk)
    relative path into task.generate_cover (relative to TEST_IMAGES_DIR, _v14 column).
    The frontend doesn't read this yet; a future DB-driven generation timeline can use it
    to locate/verify existence. Silently skipped when task_id is missing (old frontend /
    exception) or the write fails -- this only accumulates data for the future and never
    affects the actual generation."""
    if task_id is None:
        return
    try:
        rel = str(cover_path.relative_to(TEST_IMAGES_DIR))
    except ValueError:
        rel = str(cover_path)
    try:
        with db.connection_for() as conn:
            db.update_task(conn, task_id, generate_cover=rel)
    except Exception:
        logger.warning("write generate_cover for task %s failed", task_id, exc_info=True)

# v2 naming (decision #6): parent directory distinguishes mode, filename is just "<label> N.png"
_DISPLAY_LABELS = {"single": "single image", "xy": "xy plot"}
_V2_SINGLE_RE = re.compile(r"^single image (\d+)\.png$")
_V2_XY_RE = re.compile(r"^xy plot (\d+)\.png$")
# v1 legacy: image_N.png (old naming), still read when scanning, but new writes only use v2
_V1_NAME_RE = re.compile(r"^image_(\d+)\.png$")

# XY folder layout (restores PreviewXYGrid history browsing):
#   <date>/xy/xy plot <N>/{xy plot.png, cell x<i> y<j>.png, ...}
# composite is the merged big image (export + thumbnail source); cell is each cell's
# original image (PreviewXYGrid + drag into Comfy)
_XY_FOLDER_RE = re.compile(r"^xy plot (\d+)$")
_XY_TMP_FOLDER_RE = re.compile(r"^\.xy plot \d+\.tmp$")
_XY_CELL_RE = re.compile(r"^cell x(\d+) y(\d+)\.png$")
_XY_COMPOSITE_NAME = "xy plot.png"

# Path validation (shared by the whole disk-image / thumb / delete family)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DISK_MODES = ("single", "xy")
_PNG_NAME_SAFE_RE = re.compile(r"^[a-zA-Z0-9 ._-]+\.png$")


def _next_image_index(dir_: Path, mode: str) -> int:
    """Scan the PNG files for the given mode under dir and return the next 1-based index.

    Decision #11: no concurrent generation scenario, so no O_EXCL / locking; scanning for
    max+1 plus an atomic write is enough.
    Decision #6: v2 naming is 1-based ("single image 1" reads more naturally than 0); if
    v1 legacy `image_N` files coexist in the same directory, scan both as one group and
    take max+1.
    """
    if not dir_.is_dir():
        return 1
    rx_v2 = _V2_SINGLE_RE if mode == "single" else _V2_XY_RE
    max_n = 0
    for p in dir_.iterdir():
        if not p.is_file():
            continue
        m_v2 = rx_v2.match(p.name)
        m_v1 = _V1_NAME_RE.match(p.name)
        if m_v2:
            max_n = max(max_n, int(m_v2.group(1)))
        elif m_v1:
            # v1 legacy is 0-based; map into v2's numbering space (+1) to avoid collisions
            max_n = max(max_n, int(m_v1.group(1)) + 1)
    return max_n + 1


def _next_xy_folder_index(xy_dir: Path) -> int:
    """Next 1-based folder index for XY mode.

    Scans two namespaces to avoid collisions:
    - new-format subfolders `xy plot N/` (_XY_FOLDER_RE)
    - legacy flat files `xy plot N.png` (_V2_XY_RE) -- written by early PR #245, which
      won't show up in history but if left on disk their numbers still can't be reused

    Decision #11: single user, no concurrent generation, so no locking; scan + atomic
    mkdir is enough.
    """
    if not xy_dir.is_dir():
        return 1
    max_n = 0
    for p in xy_dir.iterdir():
        if p.is_dir():
            m = _XY_FOLDER_RE.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
        elif p.is_file():
            m = _V2_XY_RE.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _cleanup_xy_tmp_folders() -> None:
    """At import time, clean up leftover `.xy plot N.tmp/` half-finished folders from a
    previous server crash.

    Save flow: first write to a sibling tmp folder, then once every cell is on disk,
    os.replace to the final name. A crash mid-way leaves a tmp folder behind. Swept once
    per module import.
    """
    if not TEST_IMAGES_DIR.is_dir():
        return
    for date_dir in TEST_IMAGES_DIR.iterdir():
        if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
            continue
        xy_dir = date_dir / "xy"
        if not xy_dir.is_dir():
            continue
        for p in xy_dir.iterdir():
            if p.is_dir() and _XY_TMP_FOLDER_RE.match(p.name):
                shutil.rmtree(p, ignore_errors=True)


# Import-time cleanup (tmp folders left over from a previous server crash)
_cleanup_xy_tmp_folders()


@router.post("/api/generate")
def enqueue_generate(body: GenerateRequest) -> dict[str, Any]:
    """Start a test generate task."""
    from ...services.inference.core import generate_tempdir
    from ...services.models.families import get_assets

    model_paths = _resolve_model_paths(body.base_model, family=body.model_family)
    # TE variant override (krea2): when the request explicitly sets bf16/fp8, override
    # the selected_te default (default_paths already resolves by selected_te); if fp8
    # isn't downloaded, raise an actionable error.
    if body.text_encoder and body.model_family == "krea2":
        from ...services.models.families.krea2 import qwen3_vl_dir_for
        from ...services.models.paths import models_root

        te_dir = qwen3_vl_dir_for(models_root(), body.text_encoder)
        if body.text_encoder == "fp8" and not (te_dir / "config.json").exists():
            raise HTTPException(
                status_code=409,
                detail="The Qwen3-VL fp8 text encoder is not downloaded — go to Settings → Model downloads "
                       "and retry once it has finished downloading.",
            )
        model_paths["text_encoder_path"] = str(te_dir)
    # Turbo detection (A4/C9): for the official distilled variant, the daemon defaults to
    # the 8-step/guidance-0/fixed-mu sampling schedule; custom weights with no purpose
    # metadata are treated as non-distilled.
    distilled = bool(get_assets(body.model_family).is_distilled_path(
        model_paths.get("transformer_path", "")))

    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name="generate", config_name="generate", priority=0,
        )
        db.update_task(conn, task_id, task_type="generate")

    # create_task already lands the task as pending+generate, but config_path isn't
    # written yet; supervisor's _dispatch_exclusive_tasks skips generate tasks with
    # config_path=NULL (treats them as still enqueuing) until config.json is persisted
    # below. If any step here fails we must mark the task failed, otherwise it stays
    # pending forever with config_path=NULL (the dispatcher will always skip it).
    try:
        tempdir = generate_tempdir(task_id)
        tempdir.mkdir(parents=True, exist_ok=True)

        # Test generation uses the Comfy-style runtime. The xformers backend can provide
        # exact KSampler parity with the pinned oracle; flash_attn/none can generate but
        # don't guarantee exact parity. Preview throttling still reads from settings;
        # backend selection for training / RegAI is unaffected.
        try:
            gen_cfg = secrets.load().generate
            attn_default = gen_cfg.attention_backend
            preview_n = int(gen_cfg.preview_every_n_steps or 0)
            vae_precision = str(getattr(gen_cfg, "vae_precision", "bf16") or "bf16")
            vram_policy = str(getattr(gen_cfg, "vram_policy", "auto") or "auto")
            ram_guard = bool(getattr(gen_cfg, "ram_guard", False))
        except Exception:
            attn_default = "auto"
            preview_n = 0
            vae_precision = "bf16"
            vram_policy = "auto"
            ram_guard = False
        attn = body.attention_backend or attn_default
        if attn == "auto":
            from ...services.runtime.xformers import detect_attention_backend
            attn = detect_attention_backend()

        cfg = GenerateConfig(
            **model_paths,
            model_family=body.model_family,
            distilled=distilled,
            output_dir=str(tempdir),
            prompts=body.prompts,
            negative_prompt=body.negative_prompt,
            width=body.width,
            height=body.height,
            steps=body.steps,
            cfg_scale=body.cfg_scale,
            sampler_name=body.sampler_name,
            scheduler=body.scheduler,
            count=body.count,
            seed=body.seed,
            lora_configs=[lc.model_dump() for lc in body.lora_configs],
            mixed_precision="bf16",
            vae_precision=vae_precision,
            attention_backend=attn,
            vram_policy=vram_policy,
            ram_guard=ram_guard,
            xy_matrix=body.xy_matrix.model_dump() if body.xy_matrix else None,
        )

        # commit 14: inject the preview throttling parameter used on the daemon side
        # (global toggle in settings)
        cfg_dict = force_comfy_parity_runtime_config(
            cfg.model_dump(),
            force_exact_ksampler_backend=False,
        )
        cfg_dict["preview_every_n_steps"] = preview_n

        # Decision #15: freeze save_test_images when the task starts, to avoid the user
        # flipping the toggle mid-task and having half the images in a task go to cache
        # and half to disk. daemon submit_task reads this field into
        # _ActiveTask.save_to_disk, and _handle_image_done decides SSE delivery from it.
        try:
            cfg_dict["save_test_images_at_dispatch"] = bool(
                secrets.load().generate.save_test_images
            )
        except Exception:
            cfg_dict["save_test_images_at_dispatch"] = False

        # Frontend params snapshot passthrough: route -> config.json -> supervisor ->
        # daemon.submit_task -> _ActiveTask.params_snapshot -> stuffed into the encrypted
        # payload header alongside the PNG bytes when cache.put runs. The underscore
        # prefix signals that the daemon subprocess doesn't read this field (cfg
        # passthrough doesn't parse it).
        if body.params_snapshot:
            cfg_dict["_anima_params_snapshot_"] = body.params_snapshot

        cfg_path = tempdir / "config.json"
        cfg_path.write_text(
            json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        import time as _time
        with db.connection_for() as conn:
            now = _time.time()
            db.update_task(
                conn, task_id, status="failed",
                started_at=now, finished_at=now,
                error_msg=f"enqueue failed: {e}",
            )
        bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "failed"})
        raise HTTPException(500, f"failed to enqueue generate task: {e}")

    # 0.17 P-I forward-write: persist the params snapshot to the DB (generate_params
    # column, _v14). The frontend doesn't read this yet; it accumulates data for a future
    # "DB-only generation timeline" -- at that point params + generate_cover (written
    # when generation finishes) can locate/backfill directly, with no disk scan and no
    # migration needed. The params are exactly the params_snapshot the frontend sends
    # with the body.
    generate_params = (
        json.dumps(body.params_snapshot, ensure_ascii=False)
        if body.params_snapshot else None
    )
    with db.connection_for() as conn:
        db.update_task(
            conn, task_id, config_path=str(cfg_path), generate_params=generate_params,
        )
        task = db.get_task(conn, task_id)

    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    return task or {"id": task_id}


@router.get("/api/generate/{task_id}")
def get_generate_task(task_id: int) -> dict[str, Any]:
    """Query the status of a test task."""
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task or task.get("task_type") != "generate":
        raise NotFoundError(
            "Task not found", code="task.not_found",
            details={"task_id": task_id}, http_status=404,
        )
    return task


# ---------------------------------------------------------------------------
# /api/generate/daemon -- test daemon status query + manual unload (commit 13)
# ---------------------------------------------------------------------------


@router.get("/api/generate/taeflux/status")
def get_taeflux_status() -> dict[str, Any]:
    """commit 14: query whether the TAEFlux model is ready (needed for mid-step preview)."""
    from ...services import models as _md
    d = _md.taeflux_dir()
    return {
        "available": _md.taeflux_available(),
        "dir": str(d),
        "files": _md.TAEFLUX_FILES,
    }


@router.post("/api/generate/taeflux/install")
def install_taeflux() -> dict[str, Any]:
    """Synchronously download TAEFlux (~1.6MB, seconds). Returns OK immediately if it already exists."""
    from ...services import models as _md
    if _md.taeflux_available():
        return {"ok": True, "noop": True}
    ok = _md.download_taeflux()
    if not ok:
        raise ValidationError(
            "Failed to download the preview model; check the server log",
            code="generate.preview_model_download_failed", http_status=500,
        )
    return {"ok": True}


_TOKENIZER_CACHE: dict[str, Any] = {}


@router.post("/api/generate/token_count")
def count_prompt_tokens(body: dict) -> dict[str, Any]:
    """Real token count of a prompt (for the frontend badge; same tokenizer as
    training/inference).

    krea2's text-conditioning training budget is 512 tokens; anything beyond that the
    model never saw (not blocked, not warned -- the quality consequences are the user's
    call, the frontend just shows a neutral count). The tokenizer is lazily loaded and
    cached; when unavailable, returns tokens=null and the frontend hides the badge.
    """
    text = str(body.get("text") or "")
    family = str(body.get("model_family") or "anima")
    try:
        from ...services.models.paths import models_root

        if family == "krea2":
            from ...services.models.families.krea2 import (
                qwen3_vl_dir_for, selected_te_variant,
            )

            tok_dir = str(qwen3_vl_dir_for(models_root(), selected_te_variant()))
        else:
            from ...services.models.families.anima import qwen_dir

            tok_dir = str(qwen_dir(models_root()))
        tokenizer = _TOKENIZER_CACHE.get(tok_dir)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tok_dir, local_files_only=True,
            )
            _TOKENIZER_CACHE[tok_dir] = tokenizer
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return {"tokens": tokens}
    except Exception:
        return {"tokens": None}


@router.get("/api/generate/daemon/status")
def get_daemon_status() -> dict[str, Any]:
    """Query the daemon's current status. Used by the frontend's DaemonControls."""
    from ...services.inference.daemon import get_daemon
    daemon = get_daemon()
    return {
        "state": daemon.state,
        "model_loaded": daemon.is_model_loaded,
        "busy": daemon.is_busy,
        "alive": daemon.is_alive,
    }


@router.get("/api/generate/daemon/logs")
def get_daemon_logs(since_seq: int = 0, limit: int = 2000) -> dict[str, Any]:
    """Read the daemon stderr ring buffer. Used by the frontend to pull history when the
    log drawer opens; incremental updates come via SSE.

    When since_seq>0, only returns lines newer than that seq.
    """
    from ...services.inference.daemon import get_daemon
    return get_daemon().read_logs(since_seq=since_seq, limit=limit)


@router.post("/api/generate/daemon/unload")
def unload_daemon() -> dict[str, Any]:
    """Manually unload the daemon's model (frees VRAM). Rejected (409) while busy.

    Once unloaded, the supervisor pushes a daemon_state_changed SSE and the frontend
    button auto-disables. The next time the user clicks "start generating," the daemon
    loads on demand.
    """
    from ...services.inference.daemon import get_daemon
    daemon = get_daemon()
    if daemon.is_busy:
        raise ConflictError(
            "Inference service is busy; try again after the current task finishes",
            code="generate.daemon_busy", http_status=409,
        )
    if not daemon.is_model_loaded:
        return {"ok": True, "noop": True}
    daemon.request_unload()
    return {"ok": True}


@router.get("/api/generate/{task_id}/sample/{filename}")
def get_generate_sample(task_id: int, filename: str) -> Any:
    """Read an output image for a generate task (commit 10: from the server's in-memory
    cache, no disk involved).

    Once the daemon finishes generating, it pushes the PNG bytes back to the server into
    generate_cache; HTTP returns the bytes directly here. LRU / client-disconnect cleanup
    was added in commit 11 -- before that, the cache is released alongside supervisor
    finalize (one group of entries per task, all cleared when the task ends).
    """
    _validate_component_or_400(filename)
    if not filename.lower().endswith(".png"):
        raise ValidationError(
            "Select a .png file", code="file.ext_invalid",
            details={"types": ".png"}, http_status=400,
        )
    from ...services.inference import disk_cache as generate_cache
    data = generate_cache.get_image(task_id, filename)
    if data is None:
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"task_id": task_id, "filename": filename}, http_status=404,
        )
    # Uses no-store rather than the no-cache + ETag combo from _thumb_response: content
    # for the same (task_id, filename) in the generate cache can be overwritten by a
    # rerun (user tweaks the prompt and regenerates), so there's no stable ETag to send;
    # no-store makes the browser always refetch and get the latest result.
    # Bandwidth cost is small: this endpoint is only hit when the user is actively
    # watching the test-generation page, so QPS is low.
    # (Thumbnails / dataset images, whose content is stable, keep using _thumb_response's ETag.)
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


SCHEMA_VERSION = 2


def _format_a1111_parameters(params: dict[str, Any]) -> str:
    """Assemble an a1111-compatible `parameters` tEXt block (generally understood by
    ComfyUI / WebUI / Civitai etc).

    Format:
        <prompt> [<lora:name:scale> ...]
        Negative prompt: <neg>
        Steps: N, Sampler: ..., Schedule type: ..., CFG scale: N, Seed: N, Size: WxH

    LoRA uses the <lora:basename-without-ext:scale> syntax (a1111/ComfyUI standard).
    xy_draft / dataset_pick are not included in this block (a1111 has no standard field
    for them; use anima_params instead).
    """
    prompts = params.get("prompts") or [""]
    prompt = prompts[0] if isinstance(prompts, list) else str(prompts)
    loras = params.get("loras") or []
    lora_tags: list[str] = []
    for lo in loras:
        if not isinstance(lo, dict):
            continue
        name = str(lo.get("name") or "").rsplit(".", 1)[0]  # strip .safetensors
        if not name:
            continue
        scale = lo.get("scale", 1.0)
        lora_tags.append(f"<lora:{name}:{scale}>")
    if lora_tags:
        prompt = f"{prompt} {' '.join(lora_tags)}".strip()

    neg = params.get("negative_prompt", "")
    width = params.get("width", 0)
    height = params.get("height", 0)
    parts = [
        f"Steps: {params.get('steps', '')}",
        f"Sampler: {params.get('sampler_name', 'er_sde')}",
        f"Schedule type: {params.get('scheduler', 'simple')}",
        f"CFG scale: {params.get('cfg_scale', '')}",
        f"Seed: {params.get('seed', '')}",
        f"Size: {width}x{height}",
    ]
    return f"{prompt}\nNegative prompt: {neg}\n{', '.join(parts)}"


def _inject_png_metadata(raw: bytes, params: dict[str, Any], *, mode: str) -> bytes:
    """Inject PNG tEXt blocks into an image:
       - `anima_params` -- structured JSON, **zTXt compressed** (decision #17), used by
         this program's own reimport
       - `parameters`   -- a1111-compatible text (decision #7: **not written** for xy,
         since dragging a single cell from a matrix image into a1111 doesn't make sense
         parameter-wise); only written in single mode

    Returns the original bytes on failure (does not block the main save flow).
    """
    try:
        from PIL import Image, PngImagePlugin
        img = Image.open(io.BytesIO(raw))
        info = PngImagePlugin.PngInfo()
        # zip=True -> zTXt compressed block (PIL 9+); with XY cells[], anima_params can
        # be 6KB+, usually compresses down to 1-2KB, and a1111 doesn't recognize
        # anima_params so it just skips it anyway
        info.add_text("anima_params", json.dumps(params, ensure_ascii=False), zip=True)
        if mode == "single":
            info.add_text("parameters", _format_a1111_parameters(params))
        out = io.BytesIO()
        img.save(out, format="PNG", pnginfo=info)
        return out.getvalue()
    except Exception:
        return raw


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _decode_png_text_chunk(ctype: bytes, data: bytes) -> str | None:
    """Extract the text for the `anima_params` keyword from a PNG text chunk; returns
    None for any other keyword or on parse failure.

    - tEXt: keyword\\0 + latin-1 plain text
    - zTXt: keyword\\0 + compression method (1 byte) + zlib compressed stream (latin-1)
    - iTXt: keyword\\0 + compression flag(1) + compression method(1) + language\\0 +
      translated keyword\\0 + text (utf-8)

    Decoding follows the same rules PIL uses for reading PNG text chunks (tEXt/zTXt ->
    latin-1, iTXt -> utf-8), to stay byte-for-byte consistent with the old PIL-based
    implementation.
    """
    keyword, sep, rest = data.partition(b"\x00")
    if not sep or keyword != b"anima_params":
        return None
    try:
        if ctype == b"tEXt":
            return rest.decode("latin-1")
        if ctype == b"zTXt":
            return zlib.decompress(rest[1:]).decode("latin-1") if rest else None
        if ctype == b"iTXt":
            if len(rest) < 2:
                return None
            comp_flag = rest[0]
            body = rest[2:]
            _, _, body = body.partition(b"\x00")  # skip language tag
            _, _, body = body.partition(b"\x00")  # skip translated keyword
            return (zlib.decompress(body) if comp_flag else body).decode("utf-8")
    except Exception:
        return None
    return None


def _read_png_anima_params(path: Path) -> dict[str, Any] | None:
    """Parse params from a PNG's `anima_params` tEXt / zTXt / iTXt block; returns None if
    absent or on parse failure.

    Scans PNG chunks in order directly, looking for the `anima_params` text block before
    the first IDAT (pixel data start), stopping there. `anima_params` is written by
    `PngInfo.add_text(..., zip=True)` as zTXt and sits before IDAT, so it's guaranteed to
    be found in the header region.

    The original implementation used `PIL.Image.open` to only read the header (no pixel
    decode), but measured PIL open at ~30-40ms per file, making a disk-history scan over
    hundreds of saved images take 10-15s (up to ~1min on a cold cache). A hand-written
    chunk scan runs ~0.1ms per file (measured: 356 files, 15s -> 0.04s, ~350x), and stays
    byte-for-byte consistent with the old implementation across the whole PNG history.
    """
    try:
        with open(path, "rb") as f:
            if f.read(8) != _PNG_SIGNATURE:
                return None
            while True:
                head = f.read(8)
                if len(head) < 8:
                    return None
                length = int.from_bytes(head[:4], "big")
                ctype = head[4:8]
                if ctype == b"IDAT":
                    return None  # reached pixel data; header region has no anima_params
                if ctype in (b"tEXt", b"zTXt", b"iTXt"):
                    text = _decode_png_text_chunk(ctype, f.read(length))
                    f.read(4)  # CRC
                    if text is not None:
                        parsed = json.loads(text)
                        return parsed if isinstance(parsed, dict) else None
                else:
                    f.seek(length + 4, 1)  # skip chunk data + CRC (IHDR etc.)
    except Exception:
        return None


def _migrate_anima_params(meta: dict[str, Any]) -> dict[str, Any]:
    """v1 -> v2 schema migration (decision #18).

    v1: `lora_configs[].path` is an absolute path (old schema stored the path directly)
    v2: `loras[].name` is a basename + project_id/version_id; no absolute path stored

    Migration rule: for a v1 PNG, take the basename of the last segment of
    `lora_configs[].path` as v2's `loras[].name`, keep project_id/version_id/scale, and
    drop the old path (privacy + it would be a dead link on another machine).
    """
    version = meta.get("schema_version", 1)
    if version >= 2:
        return meta
    if version == 1:
        legacy_loras = meta.pop("lora_configs", None)
        if isinstance(legacy_loras, list):
            new_loras: list[dict[str, Any]] = []
            for lc in legacy_loras:
                if not isinstance(lc, dict):
                    continue
                path = str(lc.get("path") or "")
                name = path.replace("\\", "/").rsplit("/", 1)[-1] if path else ""
                new_loras.append({
                    "name": name,
                    "scale": float(lc.get("scale", 1.0)),
                    "project_id": lc.get("project_id"),
                    "version_id": lc.get("version_id"),
                })
            meta["loras"] = new_loras
        meta["schema_version"] = 2
        return meta
    # Unknown version -> pass through as v2 (forward-compat)
    return meta


def _enrich_params_server_side(
    params: dict[str, Any], *, task_id: int | None, mode: str
) -> dict[str, Any]:
    """Fill in server-side info on params (avoids frontend forgery / missing fields).

    - `schema_version` is forced to the current version
    - `created_at` is the on-disk write timestamp (Unix seconds)
    - `task_id` comes from enqueue (not sent by / not trusted from the frontend)
    - `mode` comes from the route parameter (not sent by the frontend)
    """
    params = dict(params)
    params["schema_version"] = SCHEMA_VERSION
    params["created_at"] = time.time()
    if task_id is not None:
        params["task_id"] = int(task_id)
    params["mode"] = mode
    return params


def _atomic_write_png(target: Path, raw: bytes) -> None:
    """Atomically write a PNG: write to tmp + os.replace (decision #11 crash safety).

    If the server crashes mid-write, disk-history won't ever scan a half-written PNG
    (one with no IEND chunk fails PIL parsing and disk-history skips it), but the user
    would still see a half-written file in a file manager, which is noise. tmp + replace
    means the target only appears once its content is complete.
    """
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, target)


def _decode_params_field(raw: str, field: str) -> dict[str, Any]:
    """`params` / a per-cell manifest element -> dict. Raises HTTPException 400 on failure."""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValidationError(
            "Image parameters are not valid JSON",
            code="generate.params_invalid",
            details={"field": field, "reason": str(e)}, http_status=400,
        ) from e
    if not isinstance(decoded, dict):
        raise ValidationError(
            "Image parameters are not valid JSON",
            code="generate.params_invalid",
            details={"field": field}, http_status=400,
        )
    return decoded


@router.post("/api/generate/save")
async def save_test_image(
    mode: str = Form(...),
    image: UploadFile = File(...),
    params: str = Form(""),
    task_id: Optional[int] = Form(None),
    cells: list[UploadFile] = File(default=[]),
    cells_manifest: str = Form(""),
) -> dict[str, Any]:
    """Save a test-generated image to disk.

    **single mode** -> `studio_data/test/<YYYY-MM-DD>/single/single image <N>.png`
    returns `{path, index, filename}` -- `cells` / `cells_manifest` must be empty,
    otherwise 400.

    **xy mode** -> `studio_data/test/<YYYY-MM-DD>/xy/xy plot <N>/{xy plot.png, cell x<i> y<j>.png ...}`
    - `image` = the composite big image (export + thumbnail source), gets anima_params
      injected with mode='xy', no a1111 block
    - `cells` = N individual cell images; `cells_manifest` = a JSON array
      [{xi:int, yi:int, params:dict}], in the same order as `cells`; each cell gets
      anima_params + a1111 injected with mode='single'
    - Validation: len(cells)==len(manifest), no duplicate (xi,yi)
    - Atomic: first written to sibling `.xy plot <N>.tmp/`, once every cell is on disk
      `os.replace` to the final name; any failure -> `shutil.rmtree(tmp)` and raise 500
    - Returns `{folder, composite, cells: [path,...]}`

    Anything else (including "compare") -> 400. Settings.save_test_images=False -> 403.
    Server-side enrich always forces schema_version/created_at/task_id/mode.
    """
    if mode not in ("single", "xy"):
        raise ValidationError(
            f"Unsupported mode: {mode}", code="generate.mode_invalid",
            details={"mode": mode}, http_status=400,
        )
    if not secrets.load().generate.save_test_images:
        raise ForbiddenError(
            "Saving test images is disabled",
            code="generate.save_disabled", http_status=403,
        )
    raw = await image.read()
    if not raw:
        raise ValidationError(
            "The uploaded image is empty",
            code="generate.empty_image", http_status=400,
        )

    if mode == "single":
        if cells or cells_manifest:
            raise HTTPException(400, "single mode does not accept cells")
        if params:
            decoded = _decode_params_field(params, "params")
            enriched = _enrich_params_server_side(decoded, task_id=task_id, mode=mode)
            raw = _inject_png_metadata(raw, enriched, mode=mode)

        target_dir = TEST_IMAGES_DIR / date.today().isoformat() / mode
        target_dir.mkdir(parents=True, exist_ok=True)
        idx = _next_image_index(target_dir, mode)
        target = target_dir / f"{_DISPLAY_LABELS[mode]} {idx}.png"
        _atomic_write_png(target, raw)
        _write_generate_cover(task_id, target)  # 0.17 P-I forward-write
        return {"path": str(target), "index": idx, "filename": target.name}

    # ----- mode == "xy" -----
    if not cells_manifest:
        raise HTTPException(400, "xy mode requires cells_manifest")
    try:
        manifest = json.loads(cells_manifest)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"cells_manifest: invalid JSON ({e})")
    if not isinstance(manifest, list):
        raise HTTPException(400, "cells_manifest: must be a JSON array")
    if len(manifest) != len(cells):
        raise HTTPException(400, f"cells_manifest length {len(manifest)} != cells {len(cells)}")
    if not cells:
        raise HTTPException(400, "xy mode requires at least one cell")

    # Validate manifest entries + collect (xi, yi) to guard against duplicates
    seen_xy: set[tuple[int, int]] = set()
    cell_specs: list[tuple[int, int, dict[str, Any]]] = []
    for i, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            raise HTTPException(400, f"cells_manifest[{i}]: must be an object")
        try:
            xi = int(entry["xi"])
            yi = int(entry["yi"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, f"cells_manifest[{i}]: missing xi/yi")
        if xi < 0 or yi < 0:
            raise HTTPException(400, f"cells_manifest[{i}]: xi/yi must be non-negative")
        if (xi, yi) in seen_xy:
            raise HTTPException(400, f"cells_manifest[{i}]: duplicate (xi={xi}, yi={yi})")
        seen_xy.add((xi, yi))
        cell_params = entry.get("params")
        if cell_params is not None and not isinstance(cell_params, dict):
            raise HTTPException(400, f"cells_manifest[{i}].params: must be a JSON object")
        cell_specs.append((xi, yi, cell_params or {}))

    # Inject anima_params into the composite (mode='xy', no a1111 block)
    composite_bytes = raw
    if params:
        composite_decoded = _decode_params_field(params, "params")
        composite_enriched = _enrich_params_server_side(composite_decoded, task_id=task_id, mode="xy")
        composite_bytes = _inject_png_metadata(composite_bytes, composite_enriched, mode="xy")

    # Read all cell bytes (before allocating the folder, to avoid a half-written result)
    cell_bytes_list: list[bytes] = []
    for i, cell_upload in enumerate(cells):
        cb = await cell_upload.read()
        if not cb:
            raise HTTPException(400, f"cells[{i}]: empty body")
        cell_bytes_list.append(cb)

    # Allocate the folder + tmp path
    xy_dir = TEST_IMAGES_DIR / date.today().isoformat() / "xy"
    xy_dir.mkdir(parents=True, exist_ok=True)
    idx = _next_xy_folder_index(xy_dir)
    final_dir = xy_dir / f"{_DISPLAY_LABELS['xy']} {idx}"
    tmp_dir = xy_dir / f".{_DISPLAY_LABELS['xy']} {idx}.tmp"
    if final_dir.exists():
        raise HTTPException(500, f"folder collision: {final_dir} already exists")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        tmp_dir.mkdir(parents=False, exist_ok=False)
        # composite
        _atomic_write_png(tmp_dir / _XY_COMPOSITE_NAME, composite_bytes)
        # cells
        cell_paths: list[Path] = []
        for (xi, yi, cell_params), cb in zip(cell_specs, cell_bytes_list):
            cell_payload = cb
            if cell_params:
                enriched_cell = _enrich_params_server_side(cell_params, task_id=task_id, mode="single")
                cell_payload = _inject_png_metadata(cell_payload, enriched_cell, mode="single")
            cell_path = tmp_dir / f"cell x{xi} y{yi}.png"
            _atomic_write_png(cell_path, cell_payload)
            cell_paths.append(cell_path)
        # atomic rename tmp -> final (Windows requires the target not exist, which
        # _next_xy_folder_index already guarantees)
        os.replace(tmp_dir, final_dir)
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, f"failed to write xy folder: {e}")

    _write_generate_cover(task_id, final_dir / _XY_COMPOSITE_NAME)  # 0.17 P-I forward-write
    return {
        "folder": str(final_dir),
        "index": idx,
        "composite": str(final_dir / _XY_COMPOSITE_NAME),
        "cells": [str(final_dir / p.name) for p in cell_paths],
    }


# ---------------------------------------------------------------------------
# Disk history browsing: scan PNG `anima_params` tEXt blocks into entries, serve
# individual images by URL
# ---------------------------------------------------------------------------


def _disk_history_id(date_str: str, mode: str, filename: str) -> str:
    """Stable id for frontend dedup / merge.

    Uses a short sha1 hash instead of the raw filename -- filenames contain spaces
    (decision #6, "single image 1"), which causes trouble when stuffed into a React key
    / data-testid / URL fragment. A 12-char hash is unique enough globally.
    """
    h = hashlib.sha1(f"{date_str}/{mode}/{filename}".encode("utf-8")).hexdigest()[:12]
    return f"disk:{h}"


def _url_quote_filename(filename: str) -> str:
    """URL-encode spaces / non-ASCII characters etc. in a filename (decision #6:
    filenames may contain spaces). The backend encodes it before returning the URL; the
    frontend must never concatenate raw filenames itself."""
    return quote(filename, safe="")


def _scan_single_dir(single_dir: Path, date_str: str) -> list[dict[str, Any]]:
    """Scan all PNGs in a `<date>/single/` directory and return a list of disk-history entries."""
    out: list[dict[str, Any]] = []
    for img in single_dir.glob("*.png"):
        if img.name.endswith(".tmp.png"):
            continue  # fallback guard for atomic-write tmp files
        params = _read_png_anima_params(img)
        if params is None:
            continue
        params = _migrate_anima_params(params)
        try:
            created_at = img.stat().st_mtime
        except OSError:
            continue
        encoded = _url_quote_filename(img.name)
        out.append({
            "id": _disk_history_id(date_str, "single", img.name),
            "date": date_str,
            "mode": "single",
            "filename": img.name,
            "path": str(img),
            "image_url": f"/api/generate/disk/image/{date_str}/single/{encoded}",
            "thumb_url": f"/api/generate/disk/thumb/{date_str}/single/{encoded}?w=128",
            "created_at": float(created_at),
            "schema_version": int(params.get("schema_version", SCHEMA_VERSION)),
            "params": params,
        })
    return out


def _build_xy_meta_from_folder(
    folder: Path, composite_params: dict[str, Any], date_str: str, folder_name: str,
) -> dict[str, Any] | None:
    """Read every cell file under an XY folder, and using the composite's xy_draft, look
    up each cell's xv/yv to assemble the disk-history entry's `xy_meta` field.

    Design: only the composite's anima_params is read (one file open); each cell's xi/yi
    is parsed from the filename (regex), and xv/yv are looked up from
    composite.xy_draft.x.raw/y.raw after splitting. Anima_params is **not** opened per
    cell -- a 5x5 matrix means 1 open instead of 26.

    Returns None if the composite is missing xy_draft (an abnormal state; the frontend
    falls back to a plain <img>).
    """
    xy_draft = composite_params.get("xy_draft")
    if not isinstance(xy_draft, dict):
        return None
    x_axis_info = xy_draft.get("x")
    if not isinstance(x_axis_info, dict):
        return None
    x_raw = str(x_axis_info.get("raw", ""))
    x_values = [s.strip() for s in x_raw.split(",") if s.strip()]
    x_axis = x_axis_info.get("axis")

    y_axis_info = xy_draft.get("y") if xy_draft.get("y") else None
    y_values: list[str | None]
    y_axis: str | None
    if isinstance(y_axis_info, dict):
        y_raw = str(y_axis_info.get("raw", ""))
        y_values = [s.strip() for s in y_raw.split(",") if s.strip()]
        y_axis = y_axis_info.get("axis")
    else:
        y_values = [None]
        y_axis = None

    samples: list[dict[str, Any]] = []
    for cell_file in folder.glob("cell x*.png"):
        m = _XY_CELL_RE.match(cell_file.name)
        if not m:
            continue
        xi = int(m.group(1))
        yi = int(m.group(2))
        xv: str | None = x_values[xi] if 0 <= xi < len(x_values) else None
        yv: str | None = y_values[yi] if 0 <= yi < len(y_values) else None
        enc_folder = _url_quote_filename(folder_name)
        enc_file = _url_quote_filename(cell_file.name)
        samples.append({
            "path": cell_file.name,
            "xy": {"xi": xi, "yi": yi, "xv": xv, "yv": yv},
            "image_url": f"/api/generate/disk/image/{date_str}/xy/{enc_folder}/{enc_file}",
        })
    samples.sort(key=lambda s: (s["xy"]["yi"], s["xy"]["xi"]))
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_values": x_values,
        "y_values": y_values,
        "samples": samples,
    }


def _scan_xy_dir(xy_dir: Path, date_str: str) -> list[dict[str, Any]]:
    """Scan a `<date>/xy/` directory -- only look at subfolders (the new layout), skip
    all flat files (legacy).

    For each subfolder matching `_XY_FOLDER_RE`:
    - must have an `xy plot.png` composite, otherwise skip the whole folder
    - read the composite's anima_params, call `_build_xy_meta_from_folder` for the cells
    """
    out: list[dict[str, Any]] = []
    for folder in xy_dir.iterdir():
        if not folder.is_dir():
            continue  # legacy flat `xy plot N.png` files no longer appear in history (product decision)
        if not _XY_FOLDER_RE.match(folder.name):
            continue
        composite = folder / _XY_COMPOSITE_NAME
        if not composite.is_file():
            continue  # a folder with no composite (half-finished / user-created manually) is skipped
        params = _read_png_anima_params(composite)
        if params is None:
            continue
        params = _migrate_anima_params(params)
        try:
            created_at = composite.stat().st_mtime
        except OSError:
            continue
        xy_meta = _build_xy_meta_from_folder(folder, params, date_str, folder.name)
        enc_folder = _url_quote_filename(folder.name)
        enc_composite = _url_quote_filename(_XY_COMPOSITE_NAME)
        out.append({
            "id": _disk_history_id(date_str, "xy", folder.name),
            "date": date_str,
            "mode": "xy",
            "folder": folder.name,
            "path": str(folder),
            "image_url": f"/api/generate/disk/image/{date_str}/xy/{enc_folder}/{enc_composite}",
            "thumb_url": f"/api/generate/disk/thumb/{date_str}/xy/{enc_folder}/{enc_composite}?w=128",
            "created_at": float(created_at),
            "schema_version": int(params.get("schema_version", SCHEMA_VERSION)),
            "params": params,
            "xy_meta": xy_meta,
        })
    return out


def _scan_png_metadata(limit: int) -> list[dict[str, Any]]:
    """Scan anima_params under every <date>/{single,xy}/ in TEST_IMAGES_DIR.

    - single mode: scans `<date>/single/*.png`
    - xy mode: scans `<date>/xy/xy plot <N>/{composite + cells}` subfolders
      (legacy flat `<date>/xy/xy plot N.png` is excluded from the list -- product decision to hide it)

    PNGs with no anima_params are excluded from the list (old data / client didn't send params).
    Decision #16: only the composite header is read (no pixel load); cells aren't opened
    as PNGs at all (xv/yv is derived from the filename + the composite's xy_draft).
    Decision #18: v1->v2 migration.
    """
    if not TEST_IMAGES_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for date_dir in TEST_IMAGES_DIR.iterdir():
        if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
            continue
        single_dir = date_dir / "single"
        if single_dir.is_dir():
            out.extend(_scan_single_dir(single_dir, date_dir.name))
        xy_dir = date_dir / "xy"
        if xy_dir.is_dir():
            out.extend(_scan_xy_dir(xy_dir, date_dir.name))
    out.sort(key=lambda e: e["created_at"], reverse=True)
    return out[:limit]


@router.get("/api/generate/disk/history")
def list_disk_history(limit: int = 500) -> dict[str, Any]:
    """List every test image saved to disk (scanned from PNG `anima_params` tEXt),
    sorted by created_at desc.

    The frontend history sidebar pulls this once and merges into its IndexedDB view;
    entry.id is stable so the frontend dedups by id. Images without anima_params (old
    data / client didn't send params) are excluded from the list.
    """
    limit = max(1, min(int(limit), 2000))
    return {"entries": _scan_png_metadata(limit)}


@router.get("/api/generate/cache/index")
def list_cache_index() -> dict[str, Any]:
    """Index of every entry in the current session's encrypted disk cache (the only
    source for the frontend history sidebar when save_test_images=false).

    The server process's SessionCache tracks live entries -> just dumped here, sorted by
    createdAt desc. Each entry's params snapshot is the one stuffed into the encrypted
    payload header alongside the PNG bytes when the image entered the cache; it dies with
    the process.

    Refreshing / switching routes both pull from here -> zero persistence layer on the
    frontend, zero chance of stale data.
    """
    from ...services.inference import disk_cache as generate_cache
    try:
        return {"entries": generate_cache.list_index()}
    except RuntimeError:
        # cache not yet initialized (shouldn't happen in theory, lifespan startup already sets it up)
        return {"entries": []}


def _resolve_disk_png(date_str: str, mode: str, filename: str) -> Path:
    r"""Path validation + resolution shared by the three endpoints (image / thumb / delete).

    Validates: date format / mode enum / filename safe character set (no / \ .. etc.) /
    .png extension.
    Returns: the actual on-disk Path (existence not guaranteed; the caller decides when to 404).
    """
    if not _DATE_RE.match(date_str):
        raise HTTPException(400, "invalid date")
    if mode not in _DISK_MODES:
        raise HTTPException(400, "invalid mode")
    if not _PNG_NAME_SAFE_RE.match(filename):
        raise HTTPException(400, "invalid filename")
    # Extra defense: safe_join against traversal
    base = (TEST_IMAGES_DIR / date_str / mode).resolve()
    try:
        path = (base / filename).resolve()
    except OSError:
        raise HTTPException(400, "invalid filename")
    if not str(path).startswith(str(base)):
        raise HTTPException(400, "path escapes base dir")
    return path


@router.get("/api/generate/disk/image/{date_str}/{mode}/{filename}")
def get_disk_image(date_str: str, mode: str, filename: str) -> Any:
    """Read a test image saved to disk (used as the large-image source when the frontend
    history sidebar clicks a disk entry)."""
    path = _resolve_disk_png(date_str, mode, filename)
    if not path.is_file():
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "mode": mode, "filename": filename},
            http_status=404,
        )
    # Content of a saved image is stable (indices only increase, never overwritten), so
    # it can be cached aggressively
    return FileResponse(
        path, media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/api/generate/disk/thumb/{date_str}/{mode}/{filename}")
def get_disk_thumb(
    date_str: str, mode: str, filename: str,
    w: int = Query(128, ge=32, le=512),
) -> Any:
    """Generate a thumbnail on the fly with PIL (Dev v1 / Arch v2 decision) -- replaces
    the frontend's IDB dataURL cache.

    - ETag = sha1(file mtime + size + w); returned directly on a 304 hit
    - Cache-Control: public, max-age=86400 (saved-image content is stable)
    - Falls back to the original image on failure (so a thumbnail-generation bug doesn't
      block the history sidebar)
    """
    path = _resolve_disk_png(date_str, mode, filename)
    if not path.is_file():
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "mode": mode, "filename": filename},
            http_status=404,
        )
    try:
        st = path.stat()
        etag = hashlib.sha1(
            f"{st.st_mtime}:{st.st_size}:{w}".encode("utf-8")
        ).hexdigest()[:16]
    except OSError as exc:
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "mode": mode, "filename": filename},
            http_status=404,
        ) from exc
    # We don't read the request header directly here since letting FastAPI / Starlette
    # handle 304 conversion is more complex; simplified approach: return ETag +
    # Cache-Control and let the browser manage its own 304 conversion (no re-request
    # within max-age).
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.thumbnail((w, w), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
    except Exception:
        return FileResponse(path, media_type="image/png")
    return Response(
        content=data,
        media_type="image/png",
        headers={
            "ETag": f'"{etag}"',
            "Cache-Control": "public, max-age=86400",
        },
    )


# ---------------------------------------------------------------------------
# XY folder-specific routes (new layout)
#
# Note: DELETE /api/generate/disk/<date>/xy/<folder> must be registered before
# `delete_disk_image` (the 5-segment wildcard {date}/{mode}/{filename}), otherwise the
# latter would match first (FastAPI matches in registration order, and the 3-segment
# wildcard would swallow xy/<folder> first).
# ---------------------------------------------------------------------------


def _resolve_disk_xy_cell(date_str: str, folder: str, filename: str) -> Path:
    """Path validation + resolution for a composite / cell file inside an XY folder.

    Validates: date / folder (must match `xy plot N`) / filename (_PNG_NAME_SAFE_RE).
    Returns the actual on-disk Path (existence not guaranteed).
    """
    if not _DATE_RE.match(date_str):
        raise HTTPException(400, "invalid date")
    if not _XY_FOLDER_RE.match(folder):
        raise HTTPException(400, "invalid folder")
    if not _PNG_NAME_SAFE_RE.match(filename):
        raise HTTPException(400, "invalid filename")
    base = (TEST_IMAGES_DIR / date_str / "xy" / folder).resolve()
    try:
        path = (base / filename).resolve()
    except OSError:
        raise HTTPException(400, "invalid filename")
    if not str(path).startswith(str(base)):
        raise HTTPException(400, "path escapes base dir")
    return path


@router.get("/api/generate/disk/image/{date_str}/xy/{folder}/{filename}")
def get_disk_xy_image(date_str: str, folder: str, filename: str) -> Any:
    """Read the composite or a cell PNG inside an XY folder (reused by PreviewXYGrid
    history browsing + dragging into Comfy)."""
    path = _resolve_disk_xy_cell(date_str, folder, filename)
    if not path.is_file():
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "folder": folder, "filename": filename},
            http_status=404,
        )
    return FileResponse(
        path, media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/api/generate/disk/thumb/{date_str}/xy/{folder}/{filename}")
def get_disk_xy_thumb(
    date_str: str, folder: str, filename: str,
    w: int = Query(128, ge=32, le=512),
) -> Any:
    """PIL thumbnail for a file inside an XY folder (history sidebar's thumb_url uses the composite)."""
    path = _resolve_disk_xy_cell(date_str, folder, filename)
    if not path.is_file():
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "folder": folder, "filename": filename},
            http_status=404,
        )
    try:
        st = path.stat()
        etag = hashlib.sha1(
            f"{st.st_mtime}:{st.st_size}:{w}".encode("utf-8")
        ).hexdigest()[:16]
    except OSError as exc:
        raise NotFoundError(
            "Image not found", code="image.not_found",
            details={"date": date_str, "folder": folder, "filename": filename},
            http_status=404,
        ) from exc
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.thumbnail((w, w), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
    except Exception:
        return FileResponse(path, media_type="image/png")
    return Response(
        content=data,
        media_type="image/png",
        headers={
            "ETag": f'"{etag}"',
            "Cache-Control": "public, max-age=86400",
        },
    )


@router.delete("/api/generate/disk/{date_str}/xy/{folder}")
def delete_disk_xy_folder(date_str: str, folder: str) -> dict[str, Any]:
    """Delete an entire XY folder (composite + all cells).

    Called when the history sidebar's x is clicked; returns OK plus whether anything was
    actually deleted (noop=True means the folder didn't exist).
    """
    if not _DATE_RE.match(date_str):
        raise HTTPException(400, "invalid date")
    if not _XY_FOLDER_RE.match(folder):
        raise HTTPException(400, "invalid folder")
    base = (TEST_IMAGES_DIR / date_str / "xy" / folder).resolve()
    test_root = TEST_IMAGES_DIR.resolve()
    if not str(base).startswith(str(test_root)):
        raise HTTPException(400, "path escapes base dir")
    if not base.is_dir():
        return {"ok": True, "noop": True}
    try:
        shutil.rmtree(base)
        return {"ok": True, "noop": False}
    except OSError as e:
        raise HTTPException(500, f"delete failed: {e}")


@router.delete("/api/generate/disk/{date_str}/{mode}/{filename}")
def delete_disk_image(date_str: str, mode: str, filename: str) -> dict[str, Any]:
    """Delete a single saved test image (single mode / admin cleanup of legacy flat XY files).

    The new XY layout goes through `delete_disk_xy_folder`; this route mainly remains for
    single mode and legacy flat XY cleanup. Registered after the XY folder DELETE route --
    otherwise the 3-segment wildcard would swallow `xy/<folder>` paths first (FastAPI
    matches in registration order).
    Returns OK plus whether anything was actually deleted (noop=True means the file
    didn't exist). Same safety validation as image / thumb.
    """
    path = _resolve_disk_png(date_str, mode, filename)
    if not path.is_file():
        return {"ok": True, "noop": True}
    try:
        path.unlink()
        return {"ok": True, "noop": False}
    except OSError as e:
        raise HTTPException(500, f"delete failed: {e}")
