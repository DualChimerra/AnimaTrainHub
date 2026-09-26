from __future__ import annotations

from pathlib import Path

import pytest

from studio.services import models_storage as ms


def _make_src(root: Path) -> Path:
    (root / "diffusion_models").mkdir(parents=True)
    (root / "diffusion_models" / "anima.safetensors").write_bytes(b"x" * 100)
    (root / "vae").mkdir()
    (root / "vae" / "vae.safetensors").write_bytes(b"y" * 50)
    (root / "top.txt").write_text("hi", encoding="utf-8")
    return root


@pytest.fixture
def reset_status():
    ms._set_status(state="idle", target="", error="")
    yield
    ms._set_status(state="idle", target="", error="")


def _fake_secrets_store(monkeypatch, old_root: Path):
    from studio import secrets
    state = {"s": secrets.Secrets(models={"root": str(old_root)})}
    monkeypatch.setattr(secrets, "load", lambda: state["s"])
    monkeypatch.setattr(secrets, "save", lambda s: state.update(s=s))
    return state


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

def test_scan_empty_or_missing_dir(tmp_path: Path) -> None:
    assert ms.scan_models_root(tmp_path / "nope") == {
        "total_files": 0, "total_bytes": 0, "entries": []
    }


def test_scan_counts_files_and_bytes(tmp_path: Path) -> None:
    src = _make_src(tmp_path / "models")
    res = ms.scan_models_root(src)
    assert res["total_files"] == 3
    assert res["total_bytes"] == 100 + 50 + 2
    by_name = {e["name"]: e for e in res["entries"]}
    assert by_name["diffusion_models"]["is_dir"] is True
    assert by_name["diffusion_models"]["files"] == 1
    assert by_name["diffusion_models"]["bytes"] == 100
    assert by_name["top.txt"]["is_dir"] is False


# ---------------------------------------------------------------------------
# validate_target
# ---------------------------------------------------------------------------

def test_validate_returns_models_subdir(tmp_path: Path) -> None:
    src = tmp_path / "cur"
    src.mkdir()
    target = tmp_path / "newroot"
    assert ms.validate_target(target, source=src) == target.resolve() / "models"


def test_validate_rejects_relative(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ms.validate_target(Path("relative/dir"), source=tmp_path / "cur")


def test_validate_rejects_same_as_current(tmp_path: Path) -> None:
    parent = tmp_path / "p"
    src = parent / "models"
    src.mkdir(parents=True)
    with pytest.raises(ValueError):
        ms.validate_target(parent, source=src)


def test_validate_rejects_nested(tmp_path: Path) -> None:
    src = tmp_path / "cur" / "models"
    src.mkdir(parents=True)
    with pytest.raises(ValueError):
        ms.validate_target(src / "sub", source=src)


def test_validate_rejects_nonempty_dst(tmp_path: Path) -> None:
    src = tmp_path / "cur"
    src.mkdir()
    target = tmp_path / "newroot"
    dst = target / "models"
    dst.mkdir(parents=True)
    (dst / "existing.bin").write_bytes(b"z")
    with pytest.raises(ValueError):
        ms.validate_target(target, source=src)


def test_validate_conflict_error_with_details(tmp_path: Path) -> None:
    src = _make_src(tmp_path / "cur")
    target = tmp_path / "newroot"
    dst = target / "models"
    (dst / "diffusion_models").mkdir(parents=True)
    (dst / "diffusion_models" / "anima.safetensors").write_bytes(b"other" * 4)
    (dst / "extra.bin").write_bytes(b"e" * 10)
    with pytest.raises(ms.TargetConflictError) as ei:
        ms.validate_target(target, source=src)
    d = ei.value.details
    assert d["target"] == str(target.resolve() / "models")
    assert d["existing_files"] == 2
    assert d["existing_bytes"] == 20 + 10
    assert d["same_name_files"] == 1


def test_validate_allows_nonempty_dst_with_on_conflict(tmp_path: Path) -> None:
    src = tmp_path / "cur"
    src.mkdir()
    target = tmp_path / "newroot"
    dst = target / "models"
    dst.mkdir(parents=True)
    (dst / "existing.bin").write_bytes(b"z")
    for mode in ("skip", "overwrite"):
        assert ms.validate_target(target, source=src, on_conflict=mode) == dst.resolve()


def test_validate_rejects_unknown_on_conflict(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ms.validate_target(tmp_path / "newroot", source=tmp_path / "cur", on_conflict="merge")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_run_migration_copies_and_updates_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_status
) -> None:
    src = _make_src(tmp_path / "old" / "models")
    dst = tmp_path / "new" / "models"
    state = _fake_secrets_store(monkeypatch, tmp_path / "old" / "models")
    events: list[dict] = []

    ms._run_migration(src, dst, publish=events.append)

    assert (dst / "diffusion_models" / "anima.safetensors").read_bytes() == b"x" * 100
    assert (dst / "vae" / "vae.safetensors").exists()
    assert (dst / "top.txt").read_text(encoding="utf-8") == "hi"
    assert state["s"].models.root == str(dst)
    assert any(e["type"] == "models_root_migrate_done" and e["ok"] for e in events)
    assert ms.migration_status()["state"] == "done"


def test_run_migration_rollback_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_status
) -> None:
    src = _make_src(tmp_path / "old" / "models")
    dst = tmp_path / "new" / "models"
    state = _fake_secrets_store(monkeypatch, tmp_path / "old" / "models")
    orig_root = state["s"].models.root

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(ms.shutil, "copy2", boom)

    events: list[dict] = []
    ms._run_migration(src, dst, publish=events.append)

    assert not dst.exists()
    assert state["s"].models.root == orig_root
    assert any(e["type"] == "models_root_migrate_done" and not e["ok"] for e in events)
    assert ms.migration_status()["state"] == "error"


def test_run_migration_skip_keeps_existing_and_fills_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_status
) -> None:
    src = _make_src(tmp_path / "old" / "models")
    dst = tmp_path / "new" / "models"
    (dst / "diffusion_models").mkdir(parents=True)
    (dst / "diffusion_models" / "anima.safetensors").write_bytes(b"theirs")
    state = _fake_secrets_store(monkeypatch, tmp_path / "old" / "models")
    events: list[dict] = []

    ms._run_migration(src, dst, publish=events.append, on_conflict="skip")

    assert (dst / "diffusion_models" / "anima.safetensors").read_bytes() == b"theirs"
    assert (dst / "vae" / "vae.safetensors").read_bytes() == b"y" * 50
    assert (dst / "top.txt").exists()
    assert state["s"].models.root == str(dst)
    assert any(e["type"] == "models_root_migrate_done" and e["ok"] for e in events)
    status = ms.migration_status()
    assert status["state"] == "done"
    assert status["total_files"] == 2
    assert not list(dst.rglob("*.part"))


def test_run_migration_overwrite_replaces_same_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_status
) -> None:
    src = _make_src(tmp_path / "old" / "models")
    dst = tmp_path / "new" / "models"
    (dst / "diffusion_models").mkdir(parents=True)
    (dst / "diffusion_models" / "anima.safetensors").write_bytes(b"theirs")
    (dst / "their-extra.bin").write_bytes(b"keep me")
    state = _fake_secrets_store(monkeypatch, tmp_path / "old" / "models")
    events: list[dict] = []

    ms._run_migration(src, dst, publish=events.append, on_conflict="overwrite")

    assert (dst / "diffusion_models" / "anima.safetensors").read_bytes() == b"x" * 100
    assert (dst / "their-extra.bin").read_bytes() == b"keep me"
    assert state["s"].models.root == str(dst)
    assert ms.migration_status()["state"] == "done"
    assert not list(dst.rglob("*.part"))


def test_run_migration_merge_failure_keeps_preexisting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_status
) -> None:
    src = _make_src(tmp_path / "old" / "models")
    dst = tmp_path / "new" / "models"
    dst.mkdir(parents=True)
    (dst / "their-extra.bin").write_bytes(b"keep me")
    state = _fake_secrets_store(monkeypatch, tmp_path / "old" / "models")
    orig_root = state["s"].models.root

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(ms.shutil, "copy2", boom)

    events: list[dict] = []
    ms._run_migration(src, dst, publish=events.append, on_conflict="skip")

    assert (dst / "their-extra.bin").read_bytes() == b"keep me"
    assert state["s"].models.root == orig_root
    assert not list(dst.rglob("*.part"))
    assert any(e["type"] == "models_root_migrate_done" and not e["ok"] for e in events)
    assert ms.migration_status()["state"] == "error"


def test_start_migration_single_flight(tmp_path: Path, reset_status) -> None:
    ms._set_status(state="running")
    with pytest.raises(RuntimeError):
        ms.start_migration(tmp_path / "newroot", source=tmp_path / "cur")
