from __future__ import annotations

from pathlib import Path

import pytest

from studio import db
from studio.services.projects import projects, versions as versions_mod
from studio.services.projects.versions import (
    list_lora_ckpts,
    list_project_lora_ckpts,
    list_project_state_ckpts,
    list_state_ckpts,
)


@pytest.fixture
def vdir(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return tmp_path


def test_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert list_lora_ckpts(tmp_path) == []


def test_scans_step_epoch_final(vdir: Path) -> None:
    out = vdir / "output"
    (out / "myproj_step1500.safetensors").touch()
    (out / "myproj_step2000.safetensors").touch()
    (out / "myproj_step2476.safetensors").touch()
    (out / "myproj_epoch5.safetensors").touch()
    (out / "myproj_final.safetensors").touch()

    items = list_lora_ckpts(vdir)
    kinds = [(it["kind"], it["value"]) for it in items]
    assert kinds[0] == ("final", 0)
    assert kinds[1:4] == [("step", 2476), ("step", 2000), ("step", 1500)]
    assert kinds[4] == ("epoch", 5)


def test_label_format(vdir: Path) -> None:
    out = vdir / "output"
    (out / "p_step100.safetensors").touch()
    (out / "p_epoch3.safetensors").touch()
    (out / "p_final.safetensors").touch()

    by_label = {it["label"]: it for it in list_lora_ckpts(vdir)}
    assert "step 100" in by_label
    assert "epoch 3" in by_label
    assert "final" in by_label


def test_unrecognized_filename_kind_other(vdir: Path) -> None:
    out = vdir / "output"
    (out / "weird_name_v9.safetensors").touch()
    items = list_lora_ckpts(vdir)
    assert len(items) == 1
    assert items[0]["kind"] == "other"
    assert items[0]["label"] == "weird_name_v9"


def test_path_is_absolute_string(vdir: Path) -> None:
    out = vdir / "output"
    (out / "p_step10.safetensors").touch()
    items = list_lora_ckpts(vdir)
    assert items[0]["path"].endswith("p_step10.safetensors")


def test_ignores_non_safetensors(vdir: Path) -> None:
    out = vdir / "output"
    (out / "p_step10.safetensors").touch()
    (out / "training_state_step10.pt").touch()
    (out / "readme.txt").touch()
    items = list_lora_ckpts(vdir)
    assert len(items) == 1
    assert items[0]["kind"] == "step"


def test_other_kind_sorts_by_natural_key(vdir: Path) -> None:
    out = vdir / "output"
    for name in ["a_9", "a_60", "a_5", "a_100"]:
        (out / f"{name}.safetensors").touch()

    items = list_lora_ckpts(vdir)
    labels = [it["label"] for it in items]
    assert labels == ["a_5", "a_9", "a_60", "a_100"]


def test_mixed_kinds_other_after_step_epoch(vdir: Path) -> None:
    out = vdir / "output"
    (out / "p_step100.safetensors").touch()
    (out / "p_step20.safetensors").touch()
    (out / "p_epoch3.safetensors").touch()
    (out / "p_final.safetensors").touch()
    (out / "custom_60.safetensors").touch()
    (out / "custom_5.safetensors").touch()

    items = list_lora_ckpts(vdir)
    labels = [it["label"] for it in items]
    assert labels == [
        "final", "step 100", "step 20", "epoch 3", "custom_5", "custom_60",
    ]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_state_ckpts_empty_dir(tmp_path: Path) -> None:
    assert list_state_ckpts(tmp_path) == []


def test_state_ckpts_scans_step_desc(vdir: Path) -> None:
    out = vdir / "output"
    (out / "training_state_step500.pt").touch()
    (out / "training_state_step1500.pt").touch()
    (out / "training_state_step100.pt").touch()

    items = list_state_ckpts(vdir)
    steps = [it["step"] for it in items]
    assert steps == [1500, 500, 100]
    assert items[0]["label"] == "step 1500"


def test_state_ckpts_ignores_lora_safetensors(vdir: Path) -> None:
    out = vdir / "output"
    (out / "training_state_step100.pt").touch()
    (out / "myproj_step100.safetensors").touch()
    (out / "ema_model.pt").touch()
    (out / "readme.txt").touch()

    items = list_state_ckpts(vdir)
    assert len(items) == 1
    assert items[0]["step"] == 100


def test_state_ckpts_path_is_string(vdir: Path) -> None:
    out = vdir / "output"
    (out / "training_state_step42.pt").touch()
    items = list_state_ckpts(vdir)
    assert items[0]["path"].endswith("training_state_step42.pt")
    assert "mtime" in items[0]




def test_state_ckpts_scans_new_per_task_subdir(vdir: Path) -> None:
    out = vdir / "output"
    task_dir = out / "state" / "task_42"
    task_dir.mkdir(parents=True)
    (task_dir / "training_state_step100.pt").touch()
    (task_dir / "training_state_step500.pt").touch()

    items = list_state_ckpts(vdir)
    steps = [it["step"] for it in items]
    assert steps == [500, 100]
    assert "task_42" in items[0]["path"]


def test_state_ckpts_old_and_new_paths_both_scanned(vdir: Path) -> None:
    out = vdir / "output"
    (out / "training_state_step100.pt").touch()
    task_dir = out / "state" / "task_7"
    task_dir.mkdir(parents=True)
    (task_dir / "training_state_step500.pt").touch()

    items = list_state_ckpts(vdir)
    steps = sorted(it["step"] for it in items)
    assert steps == [100, 500]


def test_state_ckpts_multi_task_isolated(vdir: Path) -> None:
    out = vdir / "output"
    for tid, steps in [(1, [500, 1000]), (2, [200])]:
        td = out / "state" / f"task_{tid}"
        td.mkdir(parents=True)
        for s in steps:
            (td / f"training_state_step{s}.pt").touch()

    items = list_state_ckpts(vdir)
    assert len(items) == 3
    paths = [it["path"] for it in items]
    assert any("task_1" in p and "step500" in p for p in paths)
    assert any("task_1" in p and "step1000" in p for p in paths)
    assert any("task_2" in p and "step200" in p for p in paths)


def test_state_ckpts_dedup_when_path_matches(vdir: Path) -> None:
    out = vdir / "output"
    task_dir = out / "state" / "task_3"
    task_dir.mkdir(parents=True)
    (task_dir / "training_state_step10.pt").touch()
    items = list_state_ckpts(vdir)
    assert len(items) == 1


def test_state_ckpts_includes_epoch_state(vdir: Path) -> None:
    out = vdir / "output"
    (out / "training_state_step100.pt").touch()
    (out / "training_state_epoch5.pt").touch()
    (out / "training_state_epoch3.pt").touch()

    items = list_state_ckpts(vdir)
    labels = [it["label"] for it in items]
    assert labels == ["step 100", "epoch 5", "epoch 3"]


def test_state_ckpts_pause_files_not_included(vdir: Path) -> None:
    out = vdir / "output"
    task_dir = out / "state" / "task_9"
    task_dir.mkdir(parents=True)
    (task_dir / "training_state_step100.pt").touch()
    (task_dir / "pause_step_200.pt").touch()
    (task_dir / "pause_step_200.config.json").touch()

    items = list_state_ckpts(vdir)
    assert len(items) == 1
    assert items[0]["step"] == 100


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.fixture
def project_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)

    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="Test Proj")
        v1 = versions_mod.create_version(conn, project_id=p["id"], label="baseline")
        v2 = versions_mod.create_version(conn, project_id=p["id"], label="high-lr")
        versions_mod.create_version(conn, project_id=p["id"], label="empty")

        v1dir = versions_mod.version_dir(p["id"], p["slug"], v1["label"])
        (v1dir / "output" / "training_state_step1500.pt").touch()
        (v1dir / "output" / "training_state_step500.pt").touch()
        (v1dir / "output" / "myproj_step1500.safetensors").touch()
        (v1dir / "output" / "myproj_final.safetensors").touch()

        v2dir = versions_mod.version_dir(p["id"], p["slug"], v2["label"])
        (v2dir / "output" / "training_state_step800.pt").touch()


    return dbfile, p


def test_project_state_ckpts_grouped_by_version(project_env) -> None:
    dbfile, p = project_env
    with db.connection_for(dbfile) as conn:
        groups = list_project_state_ckpts(conn, p)
    by_label = {g["label"]: g for g in groups}
    assert set(by_label.keys()) == {"baseline", "high-lr", "empty"}
    assert [it["step"] for it in by_label["baseline"]["items"]] == [1500, 500]
    assert [it["step"] for it in by_label["high-lr"]["items"]] == [800]
    assert by_label["empty"]["items"] == []
    for g in groups:
        assert isinstance(g["version_id"], int)


def test_project_lora_ckpts_grouped_by_version(project_env) -> None:
    dbfile, p = project_env
    with db.connection_for(dbfile) as conn:
        groups = list_project_lora_ckpts(conn, p)
    by_label = {g["label"]: g for g in groups}
    base_kinds = [(it["kind"], it["value"]) for it in by_label["baseline"]["items"]]
    assert base_kinds == [("final", 0), ("step", 1500)]
    assert by_label["high-lr"]["items"] == []
    assert by_label["empty"]["items"] == []


def test_project_ckpts_version_order_follows_created_at(project_env) -> None:
    dbfile, p = project_env
    with db.connection_for(dbfile) as conn:
        groups = list_project_state_ckpts(conn, p)
    assert [g["label"] for g in groups] == ["baseline", "high-lr", "empty"]
