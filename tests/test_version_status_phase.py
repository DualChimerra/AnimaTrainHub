"""ADR-0007 section 11.3-B: versions.status / phase dual-field + enum / accessor tests.

Note: the original v8 backfill tests were removed after the PR-5 v9 destructive migration --
those tests needed the stage column to exist to seed old data, and v9 physically dropped projects.stage / versions.stage.
Coverage of the v8 backfill function's logic moved to the PR-2 review + ADR section 11.3-B docs; only
5 test categories remain here: "v8 column addition + defaults + apply_all idempotence + enum / accessor".
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from studio import db
from studio.services.projects import versions
from studio.infrastructure.migrations import MIGRATIONS, current_version


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# schema column addition + defaults
# ---------------------------------------------------------------------------


def test_v8_adds_status_phase_columns(tmp_path: Path) -> None:
    """After init on a fresh DB, versions has the status / phase / last_failure_reason columns."""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(versions)")}
        assert {"status", "phase", "last_failure_reason"} <= cols


def test_v9_drops_stage_column(tmp_path: Path) -> None:
    """ADR-0007 PR-5 v9 destructive: projects.stage / versions.stage no longer exist."""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        v_cols = {r["name"] for r in c.execute("PRAGMA table_info(versions)")}
        p_cols = {r["name"] for r in c.execute("PRAGMA table_info(projects)")}
    assert "stage" not in v_cols
    assert "stage" not in p_cols


def test_default_values_for_new_versions(tmp_path: Path) -> None:
    """A newly created version without an explicit status/phase -> DEFAULTs to 'preparing' / 'curating'."""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    with _open(dbfile) as c:
        c.execute(
            "INSERT INTO projects(slug, title, created_at, updated_at) "
            "VALUES ('p', 'P', ?, ?)",
            (time.time(), time.time()),
        )
        pid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        c.execute(
            "INSERT INTO versions(project_id, label, created_at) "
            "VALUES (?, 'v', ?)",
            (pid, time.time()),
        )
        c.commit()

        row = c.execute(
            "SELECT status, phase, last_failure_reason FROM versions WHERE label='v'"
        ).fetchone()
        assert row["status"] == "preparing"
        assert row["phase"] == "curating"
        assert row["last_failure_reason"] is None


def test_apply_all_idempotent(tmp_path: Path) -> None:
    """Running init_db repeatedly should not corrupt the schema."""
    dbfile = tmp_path / "fresh.db"
    db.init_db(dbfile)
    db.init_db(dbfile)
    with _open(dbfile) as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(versions)")}
        assert {"status", "phase", "last_failure_reason"} <= cols
        assert current_version(c) == len(MIGRATIONS) + 1


# ---------------------------------------------------------------------------
# VersionStatus / VersionPhase enum + helper
# ---------------------------------------------------------------------------


def test_version_status_enum_values() -> None:
    assert versions.VersionStatus.PREPARING == "preparing"
    assert versions.VersionStatus.TRAINING == "training"
    assert versions.VersionStatus.COMPLETED == "completed"
    assert versions.VersionStatus.FAILED == "failed"
    assert versions.VersionStatus.CANCELED == "canceled"
    assert len(versions.VersionStatus.VALUES) == 5


def test_version_phase_order_and_skippable() -> None:
    # the auto-tagging phase has been removed: curating -> preprocessing -> editing -> regularizing -> ready
    assert versions.VersionPhase.ORDER == (
        "curating", "preprocessing", "editing", "regularizing", "ready",
    )
    assert len(versions.VersionPhase.VALUES) == 5
    assert versions.VersionPhase.SKIPPABLE == frozenset(
        {"preprocessing", "regularizing"}
    )


def test_get_status_fallback_to_preparing() -> None:
    assert versions.get_status({}) == "preparing"
    assert versions.get_status({"status": None}) == "preparing"
    assert versions.get_status({"status": ""}) == "preparing"
    assert versions.get_status({"status": "training"}) == "training"


def test_get_phase_fallback_to_curating() -> None:
    assert versions.get_phase({}) == "curating"
    assert versions.get_phase({"phase": None}) == "curating"
    assert versions.get_phase({"phase": "editing"}) == "editing"
