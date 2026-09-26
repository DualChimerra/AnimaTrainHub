"""v10 -> v11: ADR 0010 adds a `preprocessing` phase to VersionPhase.ORDER.

VersionPhase goes from 5 -> 6:

    curating -> preprocessing -> tagging -> editing -> regularizing -> ready

The new phase sits between curating / tagging; it's skippable (same as regularizing).

Backfill strategy (all silent, add-only -- same pattern as _v8):

  - phase = curating + `versions/{label}/train/{sub-folder}/` has images ->
    advance to preprocessing (the user already finished curating, the train set exists)
  - phase = curating + train empty -> stays curating
  - any other phase (tagging / editing / regularizing / ready) -> unchanged

Rationale: ADR 0010 SS Migration -- per the owner, "preprocessing images have already been
copied into the existing train images", so a non-empty train/ means curating is effectively
done; this provides a "silent advance to preprocessing" so users land directly on the new
phase after upgrading.

Zero schema change: the phase column is already TEXT (added in _v8), the new value needs no
extra field; the phase value constraint lives in backend code (VersionPhase.VALUES), the DB
layer is just TEXT.

Relationship to the _v9 destructive migration: this migration doesn't touch the stage column
(already dropped by _v9); it only UPDATEs the phase column. Ordering (`MIGRATIONS[10]` runs
after `_v9`) needs no special handling.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


# Kept consistent with dataset.scan.IMAGE_EXTS; inlined here to avoid the migration module
# depending on the services/ layer (migrations should be self-contained, so cold-start
# ordering stays flexible).
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})


def _train_has_image(project_dir: Path, version_label: str) -> bool:
    """Same logic as manifest._scan_train_images: scans train/{sub-folder}/{image}.
    Images placed directly in the train root are ignored (LoRA training only reads inside
    sub-folders).
    """
    train_dir = project_dir / "versions" / version_label / "train"
    if not train_dir.exists():
        return False
    for sub in train_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                return True
    return False


def migrate(conn: sqlite3.Connection) -> None:
    # Lazy import: avoids a hard dependency on the services layer when the migration module
    # loads; tests that monkeypatch services.projects.PROJECTS_DIR before running migrate get
    # the correct path.
    from ...services.projects import projects as _projects

    rows = conn.execute(
        "SELECT v.id, v.label, p.id, p.slug "
        "FROM versions v "
        "JOIN projects p ON v.project_id = p.id "
        "WHERE v.phase = 'curating'"
    ).fetchall()
    for vid, label, pid, slug in rows:
        try:
            project_dir = _projects.project_dir(int(pid), str(slug))
        except (TypeError, ValueError):
            continue
        if _train_has_image(project_dir, str(label)):
            conn.execute(
                "UPDATE versions SET phase = 'preprocessing' WHERE id = ?",
                (vid,),
            )
    conn.commit()
