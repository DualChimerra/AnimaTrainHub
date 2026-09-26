from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from studio.server import app

from ._route_helpers import iter_leaf_routes

SNAPSHOT_PATH = Path(__file__).parent / "_snapshots" / "studio_routes.json"
SNAPSHOT_VERSION = 1


def _route_entry(route: Any) -> dict[str, Any]:
    methods = sorted(getattr(route, "methods", None) or [])
    return {
        "path": getattr(route, "path", ""),
        "methods": methods,
        "name": getattr(route, "name", ""),
        "type": type(route).__name__,
    }


def _collect_routes() -> list[dict[str, Any]]:
    entries = [_route_entry(r) for r in iter_leaf_routes(app.routes)]
    entries = [e for e in entries if not (e["type"] == "Mount" and e["name"] == "studio")]
    entries.sort(key=lambda e: (e["path"], ",".join(e["methods"]), e["name"]))
    return entries


def _entry_key(e: dict[str, Any]) -> tuple[str, str, str]:
    return (e["path"], ",".join(e["methods"]), e["type"])


def _load_snapshot() -> dict[str, Any] | None:
    if not SNAPSHOT_PATH.exists():
        return None
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _write_snapshot(routes: list[dict[str, Any]]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SNAPSHOT_VERSION,
        "generated_from": "studio.server.app",
        "count": len(routes),
        "routes": routes,
    }
    SNAPSHOT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _format_diff(current: list[dict[str, Any]], saved: list[dict[str, Any]]) -> str:
    cur_by_key = {_entry_key(e): e for e in current}
    saved_by_key = {_entry_key(e): e for e in saved}
    added = [k for k in cur_by_key if k not in saved_by_key]
    removed = [k for k in saved_by_key if k not in cur_by_key]
    changed = []
    for k in cur_by_key.keys() & saved_by_key.keys():
        if cur_by_key[k]["name"] != saved_by_key[k]["name"]:
            changed.append(
                (k, saved_by_key[k]["name"], cur_by_key[k]["name"])
            )

    lines: list[str] = []
    if added:
        lines.append(f"{len(added)} route(s) added (not in the snapshot):")
        for path, methods, type_ in sorted(added):
            lines.append(f"  + [{type_}] {methods or '-'} {path}")
    if removed:
        lines.append(f"{len(removed)} route(s) missing (in the snapshot but not currently present):")
        for path, methods, type_ in sorted(removed):
            lines.append(f"  - [{type_}] {methods or '-'} {path}")
    if changed:
        lines.append(f"{len(changed)} route(s) had their name changed:")
        for (path, methods, type_), old_name, new_name in sorted(changed):
            lines.append(f"  ~ [{type_}] {methods or '-'} {path}: {old_name} → {new_name}")
    if not lines:
        lines.append("(key sets match but JSON still differs -- maybe the snapshot version field or ordering changed)")
    lines.append("")
    lines.append("if you did intentionally add/remove/change routes: delete the snapshot file and rerun to regenerate it, ")
    lines.append(f"then commit: {SNAPSHOT_PATH.relative_to(Path(__file__).parent.parent)}")
    return "\n".join(lines)


def test_route_snapshot() -> None:
    current = _collect_routes()
    assert len(current) > 100, (
        f"app.routes only has {len(current)}, suspiciously few -- server.py wiring may be broken"
    )

    saved = _load_snapshot()
    if saved is None:
        _write_snapshot(current)
        warnings.warn(
            f"snapshot created: {SNAPSHOT_PATH} ({len(current)} routes)."
            "please commit this file to git.",
            stacklevel=1,
        )
        return

    if saved.get("routes") == current:
        return

    diff = _format_diff(current, saved.get("routes", []))
    pytest.fail(
        f"studio.server.app.routes snapshot mismatch.\n"
        f"snapshot path: {SNAPSHOT_PATH}\n"
        f"snapshot count = {saved.get('count')} / current count = {len(current)}\n\n"
        f"{diff}"
    )
