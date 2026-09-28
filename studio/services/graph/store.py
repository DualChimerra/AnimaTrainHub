"""Graph: a place to keep and compare LoRA training results.

A *board* holds its own parameter definitions (``params``), saved views and
the *runs* (one card = one training run). Each run can carry sample images,
copied into ``studio_data/graph/images/`` so moving or deleting the original
file — or deleting the task / project it came from — never breaks the card.

Storage is a dedicated SQLite file (``studio_data/graph/graph.db``) rather
than tables in ``studio.db``: the Graph is a self-contained tool and never
joins against the queue tables, so it has no reason to take part in the main
schema's migration chain.

Board ``params`` and run ``values`` are JSON the frontend owns; this module
only checks their shape loosely. Runs are never hard-deleted as a side effect
of anything outside the Graph.
"""
from __future__ import annotations

import io
import json
import logging
import re
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from ...infrastructure import paths as _paths
from ...infrastructure.db import connect
from .defaults import default_params

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS boards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    board_id    INTEGER NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
    data        TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_board ON runs(board_id);
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    board_id    INTEGER NOT NULL,
    filename    TEXT NOT NULL,
    width       INTEGER,
    height      INTEGER,
    step        INTEGER,
    prompt      TEXT,
    seed        INTEGER,
    comment     TEXT,
    rating      INTEGER,
    source      TEXT,
    sort        INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_run ON images(run_id);
"""

#: Run fields the API accepts; anything else in a request body is dropped.
RUN_FIELDS = (
    "name", "values", "status", "notes", "link", "favorite", "rating",
    "best_step", "tags", "task_id", "project_id", "version_id",
)
RUN_STATUSES = ("planned", "training", "done", "stopped")
IMAGE_FIELDS = ("step", "prompt", "seed", "comment", "rating", "sort")

_EXT_BY_FORMAT = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
MAX_IMAGE_BYTES = 64 * 1024 ** 2
#: The one preview size the Graph UI asks for (web/src/pages/tools/graph/ui.tsx THUMB_PX).
THUMB_PX = 384


class GraphError(Exception):
    """Something the user can fix (bad value, unsupported file…)."""


class GraphNotFound(GraphError):
    pass


# ── paths / connection ──────────────────────────────────────────────────────

def graph_root() -> Path:
    # Read through the module so tests that repoint STUDIO_DATA are honoured.
    return Path(_paths.STUDIO_DATA) / "graph"


def db_path() -> Path:
    return graph_root() / "graph.db"


def images_dir(board_id: int, run_id: int) -> Path:
    return graph_root() / "images" / str(int(board_id)) / str(int(run_id))


#: Database files whose schema was already ensured in this process.
_READY: set[str] = set()


@contextmanager
def conn_ctx() -> Iterator[sqlite3.Connection]:
    path = db_path()
    conn = connect(path)
    try:
        # CREATE ... IF NOT EXISTS takes a write lock; do it once per file, not
        # on every request (image previews load dozens at a time).
        if str(path) not in _READY or not path.exists():
            conn.executescript(SCHEMA)
            _READY.add(str(path))
        yield conn
    finally:
        conn.close()


# ── serialisation ───────────────────────────────────────────────────────────

def _loads(raw: Optional[str]) -> dict[str, Any]:
    try:
        val = json.loads(raw or "{}")
    except ValueError:
        return {}
    return val if isinstance(val, dict) else {}


def _image_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "width": row["width"],
        "height": row["height"],
        "step": row["step"],
        "prompt": row["prompt"] or "",
        "seed": row["seed"],
        "comment": row["comment"] or "",
        "rating": row["rating"],
        "source": row["source"] or "",
        "sort": row["sort"],
        "created_at": row["created_at"],
    }


def _run_row(row: sqlite3.Row, images: list[dict[str, Any]]) -> dict[str, Any]:
    data = _loads(row["data"])
    out = {k: data.get(k) for k in RUN_FIELDS}
    out["values"] = out["values"] if isinstance(out["values"], dict) else {}
    out["tags"] = out["tags"] if isinstance(out["tags"], list) else []
    out["status"] = out["status"] if out["status"] in RUN_STATUSES else "planned"
    out["favorite"] = bool(out["favorite"])
    out["name"] = out["name"] or ""
    out["notes"] = out["notes"] or ""
    out["link"] = out["link"] or ""
    out.update({
        "id": row["id"],
        "board_id": row["board_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "images": images,
    })
    return out


def _board_row(row: sqlite3.Row) -> dict[str, Any]:
    data = _loads(row["data"])
    return {
        "id": row["id"],
        "name": row["name"],
        "note": data.get("note") or "",
        "params": data.get("params") if isinstance(data.get("params"), list) else [],
        "views": data.get("views") if isinstance(data.get("views"), list) else [],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ── validation helpers ──────────────────────────────────────────────────────

def _clean_name(name: Any, what: str) -> str:
    s = str(name or "").strip()
    if not s:
        raise GraphError(f"{what} name can't be empty")
    return s[:200]


def _clean_run(patch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in RUN_FIELDS:
        if k not in patch:
            continue
        v = patch[k]
        if k == "values":
            if not isinstance(v, dict):
                raise GraphError("values must be an object")
            # Unset = key absent; never store null so "not set" can't be
            # confused with a value.
            v = {str(pk): pv for pk, pv in v.items()
                 if pv is not None and isinstance(pv, (str, int, float, bool))}
        elif k == "status":
            if v not in RUN_STATUSES:
                raise GraphError(f"Unknown status: {v}")
        elif k == "tags":
            v = [str(t).strip()[:60] for t in (v or []) if str(t).strip()][:30]
        elif k == "rating":
            v = None if v in (None, 0, "") else max(1, min(3, int(v)))
        elif k in ("task_id", "project_id", "version_id", "best_step"):
            v = None if v in (None, "") else int(v)
        elif k == "favorite":
            v = bool(v)
        elif k in ("name", "notes", "link"):
            v = str(v or "")[: (20000 if k == "notes" else 1000)]
        out[k] = v
    return out


# ── boards ──────────────────────────────────────────────────────────────────

def list_boards() -> list[dict[str, Any]]:
    with conn_ctx() as conn:
        rows = conn.execute(
            "SELECT b.*, "
            "(SELECT COUNT(*) FROM runs r WHERE r.board_id = b.id) AS run_count, "
            "(SELECT COUNT(*) FROM images i WHERE i.board_id = b.id) AS image_count "
            "FROM boards b ORDER BY b.updated_at DESC"
        ).fetchall()
    return [
        {"id": r["id"], "name": r["name"], "note": _loads(r["data"]).get("note") or "",
         "run_count": r["run_count"], "image_count": r["image_count"],
         "created_at": r["created_at"], "updated_at": r["updated_at"]}
        for r in rows
    ]


def create_board(name: str, *, copy_params_from: Optional[int] = None,
                 empty: bool = False) -> dict[str, Any]:
    name = _clean_name(name, "Board")
    if copy_params_from is not None:
        params = get_board(copy_params_from)["board"]["params"]
    else:
        params = [] if empty else default_params()
    now = time.time()
    with conn_ctx() as conn:
        cur = conn.execute(
            "INSERT INTO boards(name, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, json.dumps({"params": params, "views": []}), now, now),
        )
        conn.commit()
        bid = int(cur.lastrowid)
    return get_board(bid)


def _must_board(conn: sqlite3.Connection, bid: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM boards WHERE id = ?", (bid,)).fetchone()
    if not row:
        raise GraphNotFound("Board not found")
    return row


def get_board(bid: int) -> dict[str, Any]:
    with conn_ctx() as conn:
        board = _board_row(_must_board(conn, bid))
        run_rows = conn.execute(
            "SELECT * FROM runs WHERE board_id = ? ORDER BY created_at ASC", (bid,)
        ).fetchall()
        img_rows = conn.execute(
            "SELECT * FROM images WHERE board_id = ? ORDER BY sort ASC, id ASC", (bid,)
        ).fetchall()
    by_run: dict[int, list[dict[str, Any]]] = {}
    for r in img_rows:
        by_run.setdefault(r["run_id"], []).append(_image_row(r))
    return {"board": board, "runs": [_run_row(r, by_run.get(r["id"], [])) for r in run_rows]}


def update_board(bid: int, patch: dict[str, Any]) -> dict[str, Any]:
    with conn_ctx() as conn:
        row = _must_board(conn, bid)
        data = _loads(row["data"])
        name = row["name"]
        if "name" in patch:
            name = _clean_name(patch["name"], "Board")
        if "note" in patch:
            data["note"] = str(patch["note"] or "")[:5000]
        if "params" in patch:
            if not isinstance(patch["params"], list):
                raise GraphError("params must be a list")
            ids = [p.get("id") for p in patch["params"] if isinstance(p, dict)]
            if len(ids) != len(patch["params"]) or len(set(ids)) != len(ids) or not all(ids):
                raise GraphError("Every parameter needs a unique id")
            data["params"] = patch["params"]
        if "views" in patch:
            if not isinstance(patch["views"], list):
                raise GraphError("views must be a list")
            data["views"] = patch["views"][:100]
        conn.execute(
            "UPDATE boards SET name = ?, data = ?, updated_at = ? WHERE id = ?",
            (name, json.dumps(data), time.time(), bid),
        )
        conn.commit()
        return _board_row(_must_board(conn, bid))


def delete_board(bid: int) -> None:
    with conn_ctx() as conn:
        _must_board(conn, bid)
        conn.execute("DELETE FROM images WHERE board_id = ?", (bid,))
        conn.execute("DELETE FROM runs WHERE board_id = ?", (bid,))
        conn.execute("DELETE FROM boards WHERE id = ?", (bid,))
        conn.commit()
    shutil.rmtree(graph_root() / "images" / str(int(bid)), ignore_errors=True)


def remap_values(bid: int, param_id: str, mapping: list[dict[str, Any]]) -> int:
    """Rewrite one parameter's value across every run of a board.

    ``mapping`` is ``[{"from": old, "to": new_or_null}]``; ``to: null`` unsets
    the value. Numbers compare with a relative tolerance so ``1e-4`` and
    ``0.0001`` are the same value. Returns how many runs changed.
    """
    changed = 0
    with conn_ctx() as conn:
        _must_board(conn, bid)
        rows = conn.execute("SELECT id, data FROM runs WHERE board_id = ?", (bid,)).fetchall()
        now = time.time()
        for r in rows:
            data = _loads(r["data"])
            values = data.get("values") if isinstance(data.get("values"), dict) else {}
            if param_id not in values:
                continue
            cur = values[param_id]
            for m in mapping:
                if same_value(cur, m.get("from")):
                    if m.get("to") is None:
                        values.pop(param_id, None)
                    else:
                        values[param_id] = m["to"]
                    data["values"] = values
                    conn.execute("UPDATE runs SET data = ?, updated_at = ? WHERE id = ?",
                                 (json.dumps(data), now, r["id"]))
                    changed += 1
                    break
        conn.commit()
    return changed


def same_value(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        return abs(a - b) <= 1e-9 * max(abs(a), abs(b))
    return a == b


# ── runs ────────────────────────────────────────────────────────────────────

def _must_run(conn: sqlite3.Connection, rid: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (rid,)).fetchone()
    if not row:
        raise GraphNotFound("Training card not found")
    return row


def get_run(rid: int) -> dict[str, Any]:
    with conn_ctx() as conn:
        row = _must_run(conn, rid)
        imgs = conn.execute(
            "SELECT * FROM images WHERE run_id = ? ORDER BY sort ASC, id ASC", (rid,)
        ).fetchall()
    return _run_row(row, [_image_row(i) for i in imgs])


def create_run(bid: int, body: dict[str, Any]) -> dict[str, Any]:
    data = {"status": "planned", "values": {}, **_clean_run(body)}
    now = time.time()
    with conn_ctx() as conn:
        _must_board(conn, bid)
        cur = conn.execute(
            "INSERT INTO runs(board_id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (bid, json.dumps(data), now, now),
        )
        conn.execute("UPDATE boards SET updated_at = ? WHERE id = ?", (now, bid))
        conn.commit()
        rid = int(cur.lastrowid)
    return get_run(rid)


def update_run(rid: int, patch: dict[str, Any]) -> dict[str, Any]:
    clean = _clean_run(patch)
    with conn_ctx() as conn:
        row = _must_run(conn, rid)
        data = _loads(row["data"])
        data.update(clean)
        now = time.time()
        conn.execute("UPDATE runs SET data = ?, updated_at = ? WHERE id = ?",
                     (json.dumps(data), now, rid))
        conn.execute("UPDATE boards SET updated_at = ? WHERE id = ?", (now, row["board_id"]))
        conn.commit()
    return get_run(rid)


def duplicate_run(rid: int, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Copy a card's settings into a new planned card. Images are *not* copied
    (they belong to the run that produced them), nor are the task links."""
    src = get_run(rid)
    body = {k: src[k] for k in ("name", "values", "notes", "link", "tags")}
    body["name"] = f"{src['name']} (copy)" if src["name"] else ""
    body["status"] = "planned"
    if overrides:
        body.update(overrides)
    return create_run(src["board_id"], body)


def delete_run(rid: int) -> None:
    with conn_ctx() as conn:
        row = _must_run(conn, rid)
        conn.execute("DELETE FROM images WHERE run_id = ?", (rid,))
        conn.execute("DELETE FROM runs WHERE id = ?", (rid,))
        conn.commit()
    shutil.rmtree(images_dir(row["board_id"], rid), ignore_errors=True)


# ── images ──────────────────────────────────────────────────────────────────

def _sniff(data: bytes) -> tuple[str, int, int]:
    """Validate an upload and return (extension, width, height)."""
    from PIL import Image

    if len(data) > MAX_IMAGE_BYTES:
        raise GraphError("The image is larger than 64 MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "").upper()
            w, h = im.size
    except Exception as exc:  # noqa: BLE001 - PIL raises many types
        raise GraphError("Not a readable image") from exc
    ext = _EXT_BY_FORMAT.get(fmt)
    if not ext:
        raise GraphError("Only PNG, JPEG and WebP images are supported")
    return ext, int(w), int(h)


def _next_sort(conn: sqlite3.Connection, rid: int) -> int:
    row = conn.execute("SELECT COALESCE(MAX(sort), -1) + 1 FROM images WHERE run_id = ?",
                       (rid,)).fetchone()
    return int(row[0])


def add_image(rid: int, data: bytes, meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    ext, w, h = _sniff(data)
    meta = meta or {}
    with conn_ctx() as conn:
        run = _must_run(conn, rid)
        d = images_dir(run["board_id"], rid)
        d.mkdir(parents=True, exist_ok=True)
        fname = f"{uuid.uuid4().hex}{ext}"
        (d / fname).write_bytes(data)
        _warm_thumb(d / fname)
        step = meta.get("step")
        seed = meta.get("seed")
        cur = conn.execute(
            "INSERT INTO images(run_id, board_id, filename, width, height, step, prompt, "
            "seed, comment, source, sort, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, run["board_id"], fname, w, h,
             None if step in (None, "") else int(step),
             str(meta.get("prompt") or "")[:4000],
             None if seed in (None, "") else int(seed),
             str(meta.get("comment") or "")[:4000],
             str(meta.get("source") or "")[:500],
             _next_sort(conn, rid), time.time()),
        )
        conn.execute("UPDATE runs SET updated_at = ? WHERE id = ?", (time.time(), rid))
        conn.commit()
        row = conn.execute("SELECT * FROM images WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _image_row(row)


def _warm_thumb(path: Path) -> None:
    """Make the preview now, so the board never waits on a resize."""
    try:
        from ..dataset import thumb_cache

        thumb_cache.get_or_make_thumb(path, THUMB_PX)
    except Exception:  # noqa: BLE001 - a missing preview is made on first view instead
        logger.debug("graph: could not pre-make thumbnail for %s", path, exc_info=True)


def _must_image(conn: sqlite3.Connection, iid: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM images WHERE id = ?", (iid,)).fetchone()
    if not row:
        raise GraphNotFound("Image not found")
    return row


def update_image(iid: int, patch: dict[str, Any]) -> dict[str, Any]:
    sets: list[str] = []
    args: list[Any] = []
    for k in IMAGE_FIELDS:
        if k not in patch:
            continue
        v = patch[k]
        if k in ("step", "seed", "sort"):
            v = None if v in (None, "") else int(v)
        elif k == "rating":
            v = None if v in (None, 0, "") else max(1, min(3, int(v)))
        else:
            v = str(v or "")[:4000]
        sets.append(f"{k} = ?")
        args.append(v)
    with conn_ctx() as conn:
        _must_image(conn, iid)
        if sets:
            conn.execute(f"UPDATE images SET {', '.join(sets)} WHERE id = ?", (*args, iid))
            conn.commit()
        return _image_row(_must_image(conn, iid))


def delete_image(iid: int) -> None:
    with conn_ctx() as conn:
        row = _must_image(conn, iid)
        conn.execute("DELETE FROM images WHERE id = ?", (iid,))
        conn.commit()
    try:
        (images_dir(row["board_id"], row["run_id"]) / row["filename"]).unlink()
    except OSError:
        pass


def image_path(iid: int) -> Path:
    with conn_ctx() as conn:
        row = _must_image(conn, iid)
    p = images_dir(row["board_id"], row["run_id"]) / row["filename"]
    if not p.exists():
        raise GraphNotFound("Image file is missing")
    return p


# ── task samples → card ─────────────────────────────────────────────────────

_STEP_RE = re.compile(r"^step_(\d+)(?:_baseline_(\d+))?", re.IGNORECASE)
_EPOCH_RE = re.compile(r"^epoch_(\d+)", re.IGNORECASE)


def sample_prompts(config: dict[str, Any]) -> list[str]:
    prompts = [str(p) for p in (config.get("sample_prompts") or []) if str(p).strip()]
    if not prompts and str(config.get("sample_prompt") or "").strip():
        prompts = [str(config["sample_prompt"])]
    return prompts


#: Files the trainer writes next to samples that aren't samples.
_NOT_SAMPLES = re.compile(r"vae_roundtrip", re.IGNORECASE)


def is_sample_name(name: str) -> bool:
    return not _NOT_SAMPLES.search(name)


def describe_samples(names: list[str], config: dict[str, Any],
                     monitor: Optional[dict[str, Any]] = None) -> dict[str, dict[str, Any]]:
    """Step / prompt / seed for each sample file of a task.

    Epoch samples (``epoch_N.png``) get their step from the monitor's sample
    records, or from total_steps / total_epochs when the record has rolled
    off. The trainer rotates through the prompt list one sample at a time
    (step and epoch samples share the counter), so the n-th sample in step
    order used ``prompts[n % len]``; baseline files carry their prompt index
    in the name.
    """
    prompts = sample_prompts(config)
    seed = config.get("sample_seed")
    seed = int(seed) if isinstance(seed, (int, float)) and seed else None
    mon = monitor or {}
    recorded: dict[str, int] = {}
    for rec in mon.get("samples") or []:
        if isinstance(rec, dict) and isinstance(rec.get("step"), int) and rec.get("path"):
            recorded[Path(str(rec["path"])).name] = rec["step"]
    per_epoch: Optional[int] = None
    try:
        if mon.get("total_steps") and mon.get("total_epochs"):
            per_epoch = round(int(mon["total_steps"]) / int(mon["total_epochs"]))
    except (TypeError, ValueError, ZeroDivisionError):
        per_epoch = None

    out: dict[str, dict[str, Any]] = {}
    rotating: list[tuple[float, str]] = []
    for n in names:
        m = _STEP_RE.match(n)
        if m and m.group(2) is not None:
            i = int(m.group(2))
            out[n] = {"step": 0, "prompt": prompts[i] if i < len(prompts) else "",
                      "seed": (seed + i) if seed is not None else None, "epoch": None}
            continue
        if m:
            step = int(m.group(1))
            out[n] = {"step": step, "prompt": "", "seed": seed, "epoch": None}
            rotating.append((step, n))
            continue
        e = _EPOCH_RE.match(n)
        if e:
            epoch = int(e.group(1))
            step = recorded.get(n)
            if step is None and per_epoch:
                step = epoch * per_epoch
            out[n] = {"step": step, "prompt": "", "seed": seed, "epoch": epoch}
            rotating.append((step if step is not None else epoch * 1e9, n))
            continue
        out[n] = {"step": recorded.get(n), "prompt": "", "seed": seed, "epoch": None}
    if prompts:
        for k, (_, n) in enumerate(sorted(rotating)):
            out[n]["prompt"] = prompts[k % len(prompts)]
    return out
