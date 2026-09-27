from __future__ import annotations

import time
from pathlib import Path

import pytest

from studio import db
from studio.services.projects import jobs as project_jobs
from studio.services.projects import projects, versions, phase as versions_phase
from studio.services.projects.phase import CheckResult


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    return {"db": dbfile}


def _make_version(isolated, label: str = "v1") -> dict:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label=label)
    return v


def _put_image(folder: Path, name: str, with_caption: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.png").write_bytes(b"fake")
    if with_caption:
        (folder / f"{name}.txt").write_text("tag1, tag2", encoding="utf-8")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_check_curating_empty() -> None:
    result = versions_phase.check_curating({"train_image_count": 0})
    assert not result.ok
    assert "Training set is empty" in result.reason


def test_check_curating_with_images() -> None:
    assert versions_phase.check_curating({"train_image_count": 1}).ok
    assert versions_phase.check_curating({"train_image_count": 100}).ok


def test_check_tagging_full_coverage() -> None:
    result = versions_phase.check_tagging({
        "train_image_count": 10, "tagged_image_count": 10,
    })
    assert result.ok


def test_check_tagging_partial_coverage() -> None:
    result = versions_phase.check_tagging({
        "train_image_count": 10, "tagged_image_count": 7,
    })
    assert not result.ok
    assert "3 image" in result.reason


def test_check_tagging_empty() -> None:
    result = versions_phase.check_tagging({"train_image_count": 0})
    assert not result.ok
    assert "Training set is empty" in result.reason


def test_check_editing_same_as_tagging() -> None:
    stats = {"train_image_count": 10, "tagged_image_count": 5}
    assert (
        versions_phase.check_editing(stats).reason
        == versions_phase.check_tagging(stats).reason
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_check_regularizing_no_jobs(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        result = versions_phase.check_regularizing(conn, v["id"])
    assert result.ok


def test_check_regularizing_blocks_when_job_running(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        _job = project_jobs.create_job(
            conn, project_id=v["project_id"], version_id=v["id"],
            kind="reg_build", params={},
        )
        project_jobs.update_status(conn, _job["id"], "running")
        result = versions_phase.check_regularizing(conn, v["id"])
    assert not result.ok
    assert "regularization" in result.reason


def test_check_regularizing_blocks_when_job_pending(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        _job = project_jobs.create_job(
            conn, project_id=v["project_id"], version_id=v["id"],
            kind="reg_build", params={},
        )
        result = versions_phase.check_regularizing(conn, v["id"])
    assert not result.ok


def test_check_regularizing_done_job_doesnt_block(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        _job = project_jobs.create_job(
            conn, project_id=v["project_id"], version_id=v["id"],
            kind="reg_build", params={},
        )
        project_jobs.update_status(conn, _job["id"], "done")
        result = versions_phase.check_regularizing(conn, v["id"])
    assert result.ok


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_check_ready_no_config(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        p = projects.get_project(conn, v["project_id"])
        result = versions_phase.check_ready(p, v)
    assert not result.ok
    assert "training config" in result.reason


# ---------------------------------------------------------------------------
# advance_phase / skip_phase
# ---------------------------------------------------------------------------


def test_advance_phase_blocked_by_failed_check(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        advanced, result, new_phase = versions_phase.advance_phase(conn, v["id"])
    assert not advanced
    assert not result.ok
    assert new_phase is None
    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.get_version(conn, v["id"])
    assert versions.get_phase(v2) == "curating"


def test_advance_phase_curating_to_preprocessing(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        p = projects.get_project(conn, v["project_id"])
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    _put_image(vdir / "train" / "5_concept", "001", with_caption=False)

    with db.connection_for(isolated["db"]) as conn:
        advanced, result, new_phase = versions_phase.advance_phase(conn, v["id"])
    assert advanced
    assert result.ok
    assert new_phase == "preprocessing"
    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.get_version(conn, v["id"])
    assert versions.get_phase(v2) == "preprocessing"


def test_advance_phase_editing_blocked_by_missing_caption(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        p = projects.get_project(conn, v["project_id"])
        versions.update_version(conn, v["id"], phase="editing")
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    _put_image(vdir / "train" / "5_concept", "001", with_caption=False)
    _put_image(vdir / "train" / "5_concept", "002", with_caption=True)

    with db.connection_for(isolated["db"]) as conn:
        advanced, result, _ = versions_phase.advance_phase(conn, v["id"])
    assert not advanced
    assert "1 image" in result.reason


def test_skip_phase_only_works_for_skippable(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        advanced, result, _ = versions_phase.skip_phase(conn, v["id"])
    assert not advanced
    assert "cannot be skipped" in result.reason


def test_skip_phase_preprocessing_jumps_to_editing(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], phase="preprocessing")
        advanced, result, new_phase = versions_phase.skip_phase(conn, v["id"])
    assert advanced
    assert result.ok
    assert new_phase == "editing"


def test_skip_phase_preprocessing_blocked_by_running_job(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], phase="preprocessing")
        _job = project_jobs.create_job(
            conn, project_id=v["project_id"], version_id=v["id"],
            kind="preprocess", params={},
        )
        project_jobs.update_status(conn, _job["id"], "running")
        advanced, result, _ = versions_phase.skip_phase(conn, v["id"])
    assert not advanced
    assert "preprocessing" in result.reason


def test_check_preprocessing_ok_without_running_job(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        result = versions_phase.check_preprocessing(conn, v["id"])
    assert result.ok


def test_skip_phase_regularizing_jumps_to_ready(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], phase="regularizing")
        advanced, result, new_phase = versions_phase.skip_phase(conn, v["id"])
    assert advanced
    assert result.ok
    assert new_phase == "ready"


def test_skip_phase_regularizing_blocked_by_running_job(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], phase="regularizing")
        _job = project_jobs.create_job(
            conn, project_id=v["project_id"], version_id=v["id"],
            kind="reg_build", params={},
        )
        project_jobs.update_status(conn, _job["id"], "running")
        advanced, result, _ = versions_phase.skip_phase(conn, v["id"])
    assert not advanced
    assert "regularization" in result.reason


def test_advance_phase_at_ready_returns_check_result(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], phase="ready")
        advanced, result, new_phase = versions_phase.advance_phase(conn, v["id"])
    assert not advanced
    assert new_phase is None
    assert not result.ok


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_update_version_accepts_valid_status(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.update_version(conn, v["id"], status="training")
    assert v2["status"] == "training"


def test_update_version_rejects_invalid_status(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(versions.VersionError, match="Invalid status"):
            versions.update_version(conn, v["id"], status="bogus")


def test_update_version_accepts_valid_phase(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.update_version(conn, v["id"], phase="editing")
    assert v2["phase"] == "editing"


def test_update_version_rejects_invalid_phase(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(versions.VersionError, match="Invalid phase"):
            versions.update_version(conn, v["id"], phase="bogus")


def test_update_version_accepts_last_failure_reason(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.update_version(conn, v["id"], last_failure_reason="OOM at step 500")
    assert v2["last_failure_reason"] == "OOM at step 500"
