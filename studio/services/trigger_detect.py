"""Find the trigger word a dataset's captions already carry.

DOP (``runtime/training/dop.py``) needs a non-empty ``trigger_word``: its
preservation branch is "the same caption with the trigger removed". The trigger
lives on the version row and used to be written by the old Tagging step; with
that step gone nothing set it any more, so every DOP run died at startup.

In practice the trigger is already in every caption — a style dataset is
captioned as ``@handle, 1girl, ...`` — so the studio can simply read it back:
the comma-separated chunk that appears in (nearly) every caption, preferring one
that sits first and one that looks like a handle (``@name``).

Pure file reading, no model, fast enough to run on every enqueue.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

#: A candidate has to be in at least this share of captions to be suggested.
MIN_COVERAGE = 0.9

#: Chunks that are in every caption of an anime/furry dataset without being a
#: trigger — quality, rating and count tags.
_COMMON_TAGS = frozenset({
    "1girl", "1boy", "2girls", "2boys", "3girls", "multiple girls", "solo",
    "safe", "sensitive", "nsfw", "explicit", "questionable", "general",
    "masterpiece", "best quality", "high quality", "good quality", "newest",
    "absurdres", "highres", "score_9", "score_8_up", "score_7_up",
    "furry", "furry female", "furry male", "anthro", "no humans",
})

_CAPTION_EXTS = (".txt", ".caption", ".json")


def _caption_text(path: Path) -> Optional[str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if path.suffix.lower() != ".json":
        return raw
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    # Studio's own shapes: {"meta": {"trigger": ...}, "tags": [...] | {...}}
    parts: list[str] = []
    if isinstance(data, dict):
        meta = data.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("trigger"), str):
            parts.append(meta["trigger"])
        tags = data.get("tags")
        if isinstance(tags, list):
            parts.extend(str(t) for t in tags)
        elif isinstance(tags, dict):
            for v in tags.values():
                if isinstance(v, list):
                    parts.extend(str(t) for t in v)
                elif isinstance(v, str):
                    parts.append(v)
        for key in ("caption", "text", "description"):
            if isinstance(data.get(key), str):
                parts.append(data[key])
    return ", ".join(parts)


def _iter_captions(root: Path) -> Iterable[str]:
    if not root.is_dir():
        return
    seen_stems: set[Path] = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _CAPTION_EXTS:
            continue
        stem = path.with_suffix("")
        if stem in seen_stems:  # 1.txt + 1.json → count the image once
            continue
        seen_stems.add(stem)
        text = _caption_text(path)
        if text and text.strip():
            yield text


def _chunks(caption: str) -> list[str]:
    # A trailing natural-language block ("... blue background. A white cat")
    # is split off the last tag at the first sentence end.
    out: list[str] = []
    for raw in caption.replace("\n", ",").split(","):
        chunk = raw.strip()
        if not chunk:
            continue
        if ". " in chunk:
            chunk = chunk.split(". ", 1)[0].strip()
        chunk = chunk.strip(".").strip()
        if chunk and len(chunk) <= 64:
            out.append(chunk)
    return out


def detect(train_dir: Path, *, limit: int = 5) -> dict[str, Any]:
    """Scan captions under ``train_dir`` and rank trigger candidates.

    Returns ``{"total": N, "suggested": str | None, "candidates": [...]}`` where
    each candidate is ``{"word", "count", "coverage", "first"}``; ``first`` is
    the share of captions in which the chunk is the very first one.
    """
    total = 0
    present: Counter[str] = Counter()
    first: Counter[str] = Counter()
    spelling: dict[str, Counter[str]] = {}

    for caption in _iter_captions(train_dir):
        chunks = _chunks(caption)
        if not chunks:
            continue
        total += 1
        seen: set[str] = set()
        for i, chunk in enumerate(chunks):
            key = chunk.lower()
            spelling.setdefault(key, Counter())[chunk] += 1
            if key in seen:
                continue
            seen.add(key)
            present[key] += 1
            if i == 0:
                first[key] += 1

    if total == 0:
        return {"total": 0, "suggested": None, "candidates": []}

    ranked = []
    for key, count in present.items():
        coverage = count / total
        if coverage < MIN_COVERAGE or key in _COMMON_TAGS:
            continue
        first_share = first[key] / total
        # Handles (@name) and leading chunks are what a trigger looks like.
        score = coverage * 10 + first_share * 5 + (3 if key.startswith("@") else 0)
        ranked.append((score, key, count, coverage, first_share))
    ranked.sort(key=lambda r: (-r[0], r[1]))

    candidates = [
        {
            "word": spelling[key].most_common(1)[0][0],
            "count": count,
            "coverage": round(coverage, 4),
            "first": round(first_share, 4),
        }
        for _score, key, count, coverage, first_share in ranked[:limit]
    ]
    return {
        "total": total,
        "suggested": candidates[0]["word"] if candidates else None,
        "candidates": candidates,
    }


def confident_trigger(train_dir: Path) -> Optional[str]:
    """The trigger to fill in without asking: in every caption, and either first
    in most of them or an ``@handle``. ``None`` when the dataset is ambiguous."""
    result = detect(train_dir, limit=1)
    if not result["candidates"]:
        return None
    top = result["candidates"][0]
    if top["coverage"] < 1.0:
        return None
    if top["first"] >= 0.8 or str(top["word"]).startswith("@"):
        return str(top["word"])
    return None
