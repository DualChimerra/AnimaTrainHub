"""Tag list for autocomplete — the English Danbooru tags the prompt / caption
inputs suggest while you type.

Storage layout:
    studio_data/tag_dictionary/
        active.json   ← parsed tag list + meta (the frontend reads it through
                        GET /api/tag-dictionary/data)
        source.csv    ← the file the list was built from (kept for debugging)

Default source: the Danbooru tag list of the a1111 tagcomplete extension
(DominikDoom/a1111-sd-webui-tagcomplete, tags/danbooru.csv), one tag per line
as `name,category,post_count,"alias,alias"`, already sorted by post count, so
the file order is the popularity order the autocomplete ranks by.

A user upload may be the same CSV or a plain list with one tag per line;
only the first column is used.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests

from .paths import STUDIO_DATA

logger = logging.getLogger(__name__)

TAG_DICT_DIR = STUDIO_DATA / "tag_dictionary"
ACTIVE_JSON = TAG_DICT_DIR / "active.json"
SOURCE_FILE = TAG_DICT_DIR / "source.csv"
# Files an older version of the dictionary left behind; removed on download.
_LEGACY_FILES = ("source.sqlite", "source.sqlite.tmp", "source.sqlite.tmp-shm", "source.sqlite.tmp-wal")

DEFAULT_URL = (
    "https://raw.githubusercontent.com/DominikDoom/"
    "a1111-sd-webui-tagcomplete/main/tags/danbooru.csv"
)
DEFAULT_SOURCE_NAME = "a1111-tagcomplete/danbooru.csv"

MAX_BYTES = 10 * 1024 * 1024  # 10MB — user upload limit
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024  # 32MB — default source limit (currently ~3.5MB)
MAX_ENTRIES = 200_000


def parse_csv(text: str) -> list[str]:
    """Parse a tag list → tags in file order, without duplicates.

    Rules:
    - each line contributes its first comma-separated column; extra columns
      (category, post count, aliases) are ignored
    - the tag is stripped and '_' becomes a space (the form captions use)
    - blank lines and lines starting with '#' are skipped
    - a header line `name,...` / `tag,...` is skipped
    - past MAX_ENTRIES the rest is dropped with a warning
    """
    tags: list[str] = []
    seen: set[str] = set()
    truncated = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        head = line.split(",", 1)[0].strip().strip('"')
        tag = head.replace("_", " ").strip()
        if not tag or tag.lower() in ("name", "tag") and not tags:
            continue
        if tag in seen:
            continue
        if len(tags) >= MAX_ENTRIES:
            truncated = True
            break
        seen.add(tag)
        tags.append(tag)
    if truncated:
        logger.warning("tag_dictionary: more than %d tags, the rest was dropped", MAX_ENTRIES)
    return tags


def _meta(source_name: str, source_url: str, kind: str, count: int) -> dict[str, Any]:
    return {
        "source_name": source_name,
        "source_url": source_url,
        "entry_count": count,
        "downloaded_at": int(time.time()),
        "kind": kind,  # "default" | "user"
    }


def _write_active(tags: list[str], meta: dict[str, Any]) -> None:
    TAG_DICT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "tags": tags}
    ACTIVE_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def load_active() -> Optional[tuple[list[str], dict[str, Any]]]:
    """Read active.json; None when missing, damaged, or in the old format
    (a translation table), which makes the startup task fetch the new list."""
    if not ACTIVE_JSON.exists():
        return None
    try:
        raw = json.loads(ACTIVE_JSON.read_text(encoding="utf-8"))
        tags = raw.get("tags")
        meta = raw.get("meta") or {}
        if not isinstance(tags, list) or not isinstance(meta, dict):
            return None
        return [str(t) for t in tags], meta
    except Exception:
        logger.exception("tag_dictionary: active.json is damaged, treating it as not loaded")
        return None


def get_meta() -> Optional[dict[str, Any]]:
    """Just the meta (GET /api/tag-dictionary/meta)."""
    loaded = load_active()
    if loaded is None:
        return None
    _, meta = loaded
    return meta


def _drop_legacy_files() -> None:
    for name in _LEGACY_FILES:
        (TAG_DICT_DIR / name).unlink(missing_ok=True)


def download_default() -> dict[str, Any]:
    """Fetch the default tag list and commit it as active.json; returns the
    new meta. Raises RuntimeError (with the reason) on failure — the router
    turns it into a 502."""
    TAG_DICT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(DEFAULT_URL, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"download failed: {exc}") from exc
    content = resp.content
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(
            f"downloaded file too large: {len(content)} bytes > {MAX_DOWNLOAD_BYTES}"
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"downloaded file is not UTF-8: {exc}") from exc
    tags = parse_csv(text)
    if not tags:
        raise RuntimeError("downloaded file parsed to zero tags")
    tmp = SOURCE_FILE.with_suffix(".csv.tmp")
    tmp.write_bytes(content)
    os.replace(tmp, SOURCE_FILE)
    _drop_legacy_files()
    meta = _meta(DEFAULT_SOURCE_NAME, DEFAULT_URL, "default", len(tags))
    _write_active(tags, meta)
    logger.info("tag_dictionary: default tag list downloaded (%d tags)", len(tags))
    return meta


def apply_uploaded(content: bytes, filename: str) -> dict[str, Any]:
    """A user-uploaded csv/txt: check size → parse → write → new meta.

    Too large or zero tags → ValueError (the router turns it into a 400)."""
    if len(content) > MAX_BYTES:
        raise ValueError(
            f"The file is too large: {len(content)} bytes, the limit is {MAX_BYTES} bytes (10MB)"
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"The file must be UTF-8 encoded: {exc}") from exc
    tags = parse_csv(text)
    if not tags:
        raise ValueError("Parsed 0 tags; expected one tag per line (`tag` or `tag,category,count`)")
    TAG_DICT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_FILE.write_bytes(content)
    _drop_legacy_files()
    meta = _meta(filename or "user-upload", "", "user", len(tags))
    _write_active(tags, meta)
    logger.info("tag_dictionary: uploaded tag list loaded (%d tags, %s)", len(tags), filename)
    return meta


def reset_to_default() -> dict[str, Any]:
    """"Restore default" — download again and replace active.json."""
    return download_default()
