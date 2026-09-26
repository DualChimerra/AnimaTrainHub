"""Bulk tag operations (PP4).

scope = {kind: 'all' | 'folder' | 'files', folder?, names?}
All reads/writes go through `read_caption` / `write_caption`, which auto-adapt
to `.txt` (comma-separated) and `.json` (see
docs/user-guide/caption-format.md, simplified to {"tags": [...]}).

On write:
- If the image already has a `.json` -> write `.json`, updating the tags field
- Otherwise write `.txt`
- add/remove never changes the file format
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

from .scan import IMAGE_EXTS
from ..tagging.caption_format import caption_json_to_tags

ScopeKind = Literal["all", "folder", "files"]


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------


def caption_path(image: Path) -> Path | None:
    """Return the caption file path corresponding to the image; returns None if there is no caption file."""
    txt = image.with_suffix(".txt")
    js = image.with_suffix(".json")
    if js.exists():
        return js
    if txt.exists():
        return txt
    return None


def read_tags(image: Path) -> list[str]:
    """Uniformly read the caption (txt / json); returns [] if it doesn't exist."""
    p = caption_path(image)
    if p is None:
        return []
    if p.suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(data, dict):
            tags = data.get("tags")
            if (
                isinstance(tags, dict)
                or "ai_output" in data
                or "fixed" in data
                or "appearance" in data
                or "environment" in data
                or "quality" in data
            ):
                return caption_json_to_tags(data)
            if isinstance(tags, list):
                return [str(t) for t in tags]
        return []
    # txt: comma-separated
    text = p.read_text(encoding="utf-8")
    return [t.strip() for t in text.split(",") if t.strip()]


def write_tags(image: Path, tags: list[str]) -> Path:
    """Write the caption. Updates .json if it already exists; otherwise writes .txt."""
    js = image.with_suffix(".json")
    if js.exists():
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data["tags"] = list(tags)
        js.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return js
    txt = image.with_suffix(".txt")
    txt.write_text(", ".join(tags), encoding="utf-8")
    return txt


# ---------------------------------------------------------------------------
# scope resolution
# ---------------------------------------------------------------------------


def _scope_image_paths(scope: dict[str, Any], train_dir: Path) -> list[Path]:
    kind: ScopeKind = scope.get("kind", "all")  # type: ignore[assignment]
    if kind == "all":
        out: list[Path] = []
        if train_dir.exists():
            for sub in train_dir.iterdir():
                if sub.is_dir():
                    out.extend(_imgs_in(sub))
        return out
    if kind == "folder":
        folder = str(scope.get("name") or scope.get("folder") or "")
        if not folder:
            return []
        return _imgs_in(train_dir / folder)
    if kind == "files":
        # New form (used after the PP4 split): items=[{folder, name}, ...], spanning folders
        items = scope.get("items")
        if isinstance(items, list):
            out: list[Path] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                f = str(it.get("folder") or "")
                n = str(it.get("name") or "")
                if not f or not n:
                    continue
                p = train_dir / f / n
                if p.exists():
                    out.append(p)
            return out
        # Old form: folder + names (kept for compatibility)
        folder = str(scope.get("folder") or "")
        names = scope.get("names") or []
        if not folder or not isinstance(names, list):
            return []
        d = train_dir / folder
        return [d / n for n in names if (d / n).exists()]
    return []


def _imgs_in(d: Path) -> list[Path]:
    if not d.exists() or not d.is_dir():
        return []
    return sorted(
        f for f in d.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------


def stats(
    scope: dict[str, Any], train_dir: Path, top: int = 50
) -> list[tuple[str, int]]:
    """Tally tag frequency, returning the top N."""
    counter: Counter[str] = Counter()
    for img in _scope_image_paths(scope, train_dir):
        for tag in read_tags(img):
            counter[tag] += 1
    return counter.most_common(top)


def add_tags(
    scope: dict[str, Any],
    train_dir: Path,
    tags: list[str],
    *,
    position: Literal["front", "back"] = "back",
) -> int:
    """Add tags to every caption within scope (existing ones aren't duplicated). Returns the number of affected files."""
    new = [t.strip() for t in tags if t.strip()]
    if not new:
        return 0
    affected = 0
    for img in _scope_image_paths(scope, train_dir):
        cur = read_tags(img)
        cur_set = set(cur)
        to_add = [t for t in new if t not in cur_set]
        if not to_add:
            continue
        if position == "front":
            merged = to_add + cur
        else:
            merged = cur + to_add
        write_tags(img, merged)
        affected += 1
    return affected


def remove_tags(
    scope: dict[str, Any], train_dir: Path, tags: list[str]
) -> int:
    drop = {t.strip() for t in tags if t.strip()}
    if not drop:
        return 0
    affected = 0
    for img in _scope_image_paths(scope, train_dir):
        cur = read_tags(img)
        kept = [t for t in cur if t not in drop]
        if len(kept) != len(cur):
            write_tags(img, kept)
            affected += 1
    return affected


def replace_tag(
    scope: dict[str, Any], train_dir: Path, old: str, new: str
) -> int:
    old_s = old.strip()
    new_s = new.strip()
    if not old_s or not new_s or old_s == new_s:
        return 0
    affected = 0
    for img in _scope_image_paths(scope, train_dir):
        cur = read_tags(img)
        if old_s not in cur:
            continue
        # Replace while preserving order; if new is already in the list, just drop old to avoid a duplicate
        out: list[str] = []
        seen: set[str] = set()
        for t in cur:
            t_out = new_s if t == old_s else t
            if t_out in seen:
                continue
            seen.add(t_out)
            out.append(t_out)
        write_tags(img, out)
        affected += 1
    return affected


def dedupe(scope: dict[str, Any], train_dir: Path) -> int:
    """Dedupe within each file (order preserved, first occurrence kept). Returns the number of affected files."""
    affected = 0
    for img in _scope_image_paths(scope, train_dir):
        cur = read_tags(img)
        seen: set[str] = set()
        out: list[str] = []
        for t in cur:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        if len(out) != len(cur):
            write_tags(img, out)
            affected += 1
    return affected


# ---------------------------------------------------------------------------
# single-image helpers (used in GET/PUT /captions/{folder}/{filename})
# ---------------------------------------------------------------------------


def list_captions_in_folder(
    train_dir: Path, folder: str, *, preview: int = 5, full: bool = False
) -> list[dict[str, Any]]:
    """List all images in the folder plus tag info.

    When full=True, includes the complete tags and format (used by the
    frontend's cache model); otherwise only returns the tag count + the
    first N previews (used by the thumbnail list).
    """
    d = train_dir / folder
    out: list[dict[str, Any]] = []
    for img in _imgs_in(d):
        tags = read_tags(img)
        cap_path = caption_path(img)
        item: dict[str, Any] = {
            "name": img.name,
            "folder": folder,
            "tag_count": len(tags),
            "tags_preview": tags[:preview],
            "has_caption": cap_path is not None,
        }
        if full:
            item["tags"] = tags
            item["format"] = (
                "json" if cap_path and cap_path.suffix == ".json"
                else "txt" if cap_path
                else "none"
            )
        out.append(item)
    return out


def list_all_captions(
    train_dir: Path, *, preview: int = 5, full: bool = False
) -> list[dict[str, Any]]:
    """List images plus tag info across every folder under train/, with each item tagged with its owning folder."""
    if not train_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for sub in sorted(d for d in train_dir.iterdir() if d.is_dir()):
        out.extend(
            list_captions_in_folder(
                train_dir, sub.name, preview=preview, full=full
            )
        )
    return out


def read_one(train_dir: Path, folder: str, filename: str) -> dict[str, Any]:
    img = train_dir / folder / filename
    if not img.exists():
        raise FileNotFoundError(f"image not found: {folder}/{filename}")
    return {
        "name": img.name,
        "tags": read_tags(img),
        "format": (
            "json" if img.with_suffix(".json").exists()
            else "txt" if img.with_suffix(".txt").exists()
            else "none"
        ),
    }


def write_one(
    train_dir: Path, folder: str, filename: str, tags: list[str]
) -> dict[str, Any]:
    img = train_dir / folder / filename
    if not img.exists():
        raise FileNotFoundError(f"image not found: {folder}/{filename}")
    write_tags(img, tags)
    return read_one(train_dir, folder, filename)
