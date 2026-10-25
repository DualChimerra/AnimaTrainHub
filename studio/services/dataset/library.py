"""Shared dataset library: image pools that live outside any project.

A shared dataset is a `projects` row with kind='dataset' whose folder sits in
`studio_data/datasets/{id}-{slug}/`:

    download/            the images (+ booru .txt / caption sidecars), same as a
                         project's download/ -- every download / upload / thumb
                         endpoint takes the dataset id as a project id
    cache/latents/<g>/   VAE latents written by training runs (content-addressed)
    cache/text/<g>/      caption encodings written by training runs
    project.json         row snapshot

Versions link datasets through `version_datasets`; curation then offers their
images next to the project's own download/. An image taken from a dataset is
remembered by its **pool key** `@{dataset_id}/{file name}` (manifest `origin`,
validation origins), while an image from the project's own download/ keeps its
bare file name -- old projects read exactly as before.

Training runs find the dataset caches through `train/.cache_routes.json`
(see runtime/training/dataset_cache.py), rewritten at every enqueue.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from studio.domain.errors import DomainError

from ..projects import projects, versions
from .scan import IMAGE_EXTS


class DatasetError(DomainError):
    default_code = "dataset.error"


# ---------------------------------------------------------------------------
# pool keys
# ---------------------------------------------------------------------------

_POOL_KEY = re.compile(r"^@(\d+)/([^\\/]+)$")
_META_EXTS = (".booru.txt", ".txt", ".json")

CACHE_DIRNAME = "cache"
CACHE_KINDS = ("latents", "text")
ROUTES_FILENAME = ".cache_routes.json"  # mirrors runtime/training/dataset_cache.py
VALIDATION_ORIGINS = "origins.json"


def pool_key(dataset_id: int, name: str) -> str:
    return f"@{int(dataset_id)}/{name}"


def parse_pool_key(key: str) -> tuple[Optional[int], str]:
    """`@7/a.png` -> (7, 'a.png'); a bare file name -> (None, name)."""
    m = _POOL_KEY.fullmatch(key or "")
    if m:
        return int(m.group(1)), m.group(2)
    return None, key


def is_pool_key(key: str) -> bool:
    return bool(_POOL_KEY.fullmatch(key or ""))


def pool_file_name(key: str) -> str:
    """The file name part of an origin (what a train copy is called on disk)."""
    return parse_pool_key(key)[1]


def dataset_dir_by_id(dataset_id: int) -> Optional[Path]:
    """Locate a dataset folder from its id alone (no db round trip).

    `{id}-` is a unique prefix: `1-*` never matches `12-foo`.
    """
    root = projects.datasets_root()
    if not root.is_dir():
        return None
    for d in root.glob(f"{int(dataset_id)}-*"):
        if d.is_dir():
            return d
    return None


def origin_source_path(project_dir: Path, origin: str) -> Path:
    """Where the original bytes of a train/validation image live.

    Pool keys resolve into the dataset's download/; bare names into the
    project's own download/. A deleted dataset yields a path that doesn't exist
    (callers already treat a missing original as "no_origin").
    """
    dataset_id, name = parse_pool_key(origin)
    if dataset_id is None:
        return project_dir / "download" / name
    ddir = dataset_dir_by_id(dataset_id)
    if ddir is None:
        return projects.datasets_root() / f"{dataset_id}-missing" / "download" / name
    return ddir / "download" / name


# ---------------------------------------------------------------------------
# version links
# ---------------------------------------------------------------------------


def _get_dataset(conn: sqlite3.Connection, dataset_id: int) -> dict[str, Any]:
    d = projects.get_project(conn, dataset_id)
    if not d or not projects.is_dataset(d):
        raise DatasetError(
            "Dataset not found", code="dataset.not_found",
            details={"id": dataset_id}, http_status=404,
        )
    return d


def get_dataset(conn: sqlite3.Connection, dataset_id: int) -> dict[str, Any]:
    return _get_dataset(conn, dataset_id)


def linked_datasets(conn: sqlite3.Connection, version_id: int) -> list[dict[str, Any]]:
    """The datasets a version curates from, in the order the user added them."""
    rows = conn.execute(
        "SELECT p.* FROM version_datasets vd JOIN projects p ON p.id = vd.dataset_id "
        "WHERE vd.version_id = ? AND p.kind = 'dataset' "
        "ORDER BY vd.position, vd.added_at",
        (version_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def set_linked_datasets(
    conn: sqlite3.Connection, version_id: int, dataset_ids: Iterable[int],
) -> list[dict[str, Any]]:
    """Replace a version's linked datasets. Images already copied into train/
    stay; unlinking only hides the dataset from the curation pool."""
    ids = list(dict.fromkeys(int(i) for i in dataset_ids))
    for i in ids:
        _get_dataset(conn, i)
    now = time.time()
    conn.execute("DELETE FROM version_datasets WHERE version_id = ?", (version_id,))
    conn.executemany(
        "INSERT INTO version_datasets(version_id, dataset_id, position, added_at) "
        "VALUES (?, ?, ?, ?)",
        [(version_id, i, pos, now) for pos, i in enumerate(ids)],
    )
    conn.commit()
    return linked_datasets(conn, version_id)


def copy_links(conn: sqlite3.Connection, src_version_id: int, dst_version_id: int) -> None:
    """A new version starts from the same datasets as the one it came from."""
    ids = [d["id"] for d in linked_datasets(conn, src_version_id)]
    if ids:
        set_linked_datasets(conn, dst_version_id, ids)


def usage(conn: sqlite3.Connection, dataset_id: int) -> list[dict[str, Any]]:
    """Versions that link this dataset: `[{project_id, project_title, version_id, version_label}]`."""
    rows = conn.execute(
        "SELECT p.id AS project_id, p.title AS project_title, p.kind AS project_kind, "
        "v.id AS version_id, v.label AS version_label "
        "FROM version_datasets vd "
        "JOIN versions v ON v.id = vd.version_id "
        "JOIN projects p ON p.id = v.project_id "
        "WHERE vd.dataset_id = ? ORDER BY p.title, v.label",
        (dataset_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------


def _image_files(download: Path) -> list[os.DirEntry]:
    if not download.is_dir():
        return []
    out = []
    with os.scandir(download) as it:
        for e in it:
            if e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_EXTS:
                out.append(e)
    return out


def _dir_size(root: Path) -> tuple[int, int]:
    """(file count, bytes) under root, recursively; group.json doesn't count."""
    files = 0
    size = 0
    if not root.is_dir():
        return 0, 0
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            if n == "group.json" or n.endswith(".tmp"):
                continue
            try:
                size += os.path.getsize(os.path.join(dirpath, n))
                files += 1
            except OSError:
                pass
    return files, size


def summarize(conn: sqlite3.Connection, d: dict[str, Any]) -> dict[str, Any]:
    """List-row payload: the row + image / cache stats + where it is used."""
    ddir = projects.project_dir(d["id"], d["slug"])
    images = _image_files(ddir / "download")
    stats = []
    for e in images:
        try:
            st = e.stat()
        except OSError:
            continue
        stats.append((e.name, st.st_size, st.st_mtime))
    stats.sort(key=lambda t: t[2], reverse=True)
    cache_files, cache_bytes = _dir_size(ddir / CACHE_DIRNAME)
    used = usage(conn, d["id"])
    out = dict(d)
    out.update({
        "image_count": len(stats),
        "image_bytes": sum(s for _, s, _ in stats),
        "last_added_at": stats[0][2] if stats else None,
        "covers": [n for n, _, _ in stats[:4]],
        "cache_files": cache_files,
        "cache_bytes": cache_bytes,
        "used_by": used,
        "path": str(ddir),
    })
    return out


def list_datasets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM projects WHERE kind = 'dataset' ORDER BY updated_at DESC"
    ).fetchall()
    return [summarize(conn, dict(r)) for r in rows]


_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya", "і": "i", "ї": "yi", "є": "ye", "ґ": "g",
})


def dataset_slug(title: str) -> str:
    """Readable folder name for a dataset: Cyrillic is transliterated (the user
    browses `studio_data/datasets/`), anything else non-ASCII falls back to
    'dataset' instead of the projects' 'project'."""
    slug = projects.slugify((title or "").lower().translate(_TRANSLIT))
    return "dataset" if slug == "project" and "project" not in (title or "").lower() else slug


def create_dataset(conn: sqlite3.Connection, *, title: str, note: Optional[str] = None) -> dict[str, Any]:
    return projects.create_project(conn, title=title, slug=dataset_slug(title), note=note, kind="dataset")


def delete_dataset(conn: sqlite3.Connection, dataset_id: int) -> None:
    """Deletes the folder (images + caches) and the row; links cascade away.
    Versions keep their train/ copies, so trainings are unaffected."""
    _get_dataset(conn, dataset_id)
    projects.delete_project(conn, dataset_id)


# ---------------------------------------------------------------------------
# cache folder
# ---------------------------------------------------------------------------


def cache_groups(d: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per cache format folder: what it is for, size, last use."""
    root = projects.project_dir(d["id"], d["slug"]) / CACHE_DIRNAME
    out: list[dict[str, Any]] = []
    for kind in CACHE_KINDS:
        kdir = root / kind
        if not kdir.is_dir():
            continue
        for g in sorted(p for p in kdir.iterdir() if p.is_dir()):
            meta: dict[str, Any] = {}
            mp = g / "group.json"
            if mp.is_file():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8")) or {}
                except (OSError, ValueError):
                    meta = {}
            files, size = _dir_size(g)
            try:
                mtime = max((f.stat().st_mtime for f in g.iterdir()), default=g.stat().st_mtime)
            except OSError:
                mtime = None
            out.append({
                "kind": kind,
                "group": g.name,
                "family": meta.get("family"),
                "fingerprint": meta.get("fingerprint"),
                "flip_copies": meta.get("flip_copies"),
                "files": files,
                "bytes": size,
                "created_at": meta.get("created_at"),
                "last_used_at": meta.get("last_used_at") or mtime,
                "path": str(g),
            })
    out.sort(key=lambda r: (r["last_used_at"] or 0), reverse=True)
    return out


_GROUP_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def clear_cache(d: dict[str, Any], kind: Optional[str] = None, group: Optional[str] = None) -> int:
    """Delete one cache group, one kind, or the whole cache. Returns bytes freed.
    The next training just re-encodes what it needs."""
    root = projects.project_dir(d["id"], d["slug"]) / CACHE_DIRNAME
    if kind is not None and kind not in CACHE_KINDS:
        raise DatasetError("Unknown cache kind", code="dataset.cache_kind_invalid",
                           details={"kind": kind}, http_status=400)
    if group is not None and not _GROUP_NAME.fullmatch(group):
        raise DatasetError("Invalid cache group", code="dataset.cache_group_invalid",
                           details={"group": group}, http_status=400)
    target = root
    if kind is not None:
        target = root / kind
        if group is not None:
            target = target / group
    if not target.exists():
        return 0
    _, size = _dir_size(target)
    shutil.rmtree(target, ignore_errors=True)
    return size


# ---------------------------------------------------------------------------
# import from a project's download/
# ---------------------------------------------------------------------------


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink when the volume allows (download/ originals are never edited in
    place, so both names can share the bytes); otherwise a plain copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def import_project_images(
    conn: sqlite3.Connection, dataset_id: int, project_id: int,
) -> dict[str, list[str]]:
    """Bring a project's own download/ images (and their sidecars) into a dataset.

    The project keeps its files, so its versions keep working exactly as before.
    Names already present in the dataset are skipped.
    """
    d = _get_dataset(conn, dataset_id)
    p = projects.get_project(conn, project_id)
    if not p or projects.is_dataset(p):
        raise DatasetError("Project not found", code="project.not_found",
                           details={"id": project_id}, http_status=404)
    src_dir = projects.project_dir(p["id"], p["slug"]) / "download"
    dst_dir = projects.project_dir(d["id"], d["slug"]) / "download"
    dst_dir.mkdir(parents=True, exist_ok=True)
    added: list[str] = []
    skipped: list[str] = []
    for e in sorted(_image_files(src_dir), key=lambda e: e.name):
        src = Path(e.path)
        dst = dst_dir / e.name
        if dst.exists():
            skipped.append(e.name)
            continue
        _link_or_copy(src, dst)
        stem = src.name[: -len(src.suffix)] if src.suffix else src.name
        for ext in _META_EXTS:
            side = src_dir / f"{stem}{ext}"
            if side.is_file() and not (dst_dir / side.name).exists():
                try:
                    _link_or_copy(side, dst_dir / side.name)
                except OSError:
                    pass
        added.append(e.name)
    projects.update_project(conn, dataset_id, note=d.get("note"))  # bumps updated_at
    return {"added": added, "skipped": skipped}


# ---------------------------------------------------------------------------
# training: cache routes
# ---------------------------------------------------------------------------


def write_cache_routes(project: dict[str, Any], version: dict[str, Any]) -> Optional[Path]:
    """Write `train/.cache_routes.json`: every train image that came from a
    shared dataset -> that dataset's cache folder. Removes a stale file when no
    image is routed, so the run falls back to plain sidecars."""
    from ..preprocess import manifest as preprocess_manifest

    pdir = projects.project_dir(project["id"], project["slug"])
    train = versions.version_dir(project["id"], project["slug"], version["label"]) / "train"
    target = train / ROUTES_FILENAME
    if not train.is_dir():
        return None
    try:
        tm = preprocess_manifest.train_load(pdir, version["label"])
    except Exception:  # noqa: BLE001 -- a broken manifest only disables sharing
        tm = {"images": {}}
    roots: dict[int, Optional[Path]] = {}
    routes: dict[str, str] = {}
    for rel, entry in (tm.get("images") or {}).items():
        if not isinstance(entry, dict) or preprocess_manifest.is_duplicate_removed_entry(entry):
            continue
        dataset_id, _ = parse_pool_key(str(entry.get("origin") or ""))
        if dataset_id is None:
            continue
        if dataset_id not in roots:
            ddir = dataset_dir_by_id(dataset_id)
            roots[dataset_id] = (ddir / CACHE_DIRNAME) if ddir else None
        root = roots[dataset_id]
        if root is not None and (train / rel).is_file():
            routes[rel] = str(root)
    if not routes:
        target.unlink(missing_ok=True)
        return None
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "routes": routes}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, target)
    return target


# ---------------------------------------------------------------------------
# validation origins (validation/ has no manifest; this tiny map remembers
# which dataset a held-out image came from, so it never reappears in the pool)
# ---------------------------------------------------------------------------


def read_validation_origins(val_dir: Path) -> dict[str, str]:
    path = val_dir / VALIDATION_ORIGINS
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def write_validation_origins(val_dir: Path, mapping: dict[str, str]) -> None:
    path = val_dir / VALIDATION_ORIGINS
    if not mapping:
        path.unlink(missing_ok=True)
        return
    val_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
