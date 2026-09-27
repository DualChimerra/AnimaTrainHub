from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from studio import db
from studio.supervisor import Supervisor, _default_cmd_builder


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_cmd_builder_no_resume_when_no_paused_state(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("", encoding="utf-8")
    cmd = _default_cmd_builder({"task_type": "train", "id": 1}, cfg)
    assert "--resume-state" not in cmd


def test_cmd_builder_adds_resume_state_when_paused(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("", encoding="utf-8")
    pt = tmp_path / "pause_step_100.pt"
    cmd = _default_cmd_builder(
        {"task_type": "train", "id": 1, "paused_state_path": str(pt)},
        cfg,
    )
    assert "--resume-state" in cmd
    idx = cmd.index("--resume-state")
    assert cmd[idx + 1] == str(pt)


def test_cmd_builder_paused_path_works_for_reg_ai(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("", encoding="utf-8")
    pt = tmp_path / "pause_step_200.pt"
    cmd = _default_cmd_builder(
        {"task_type": "reg_ai", "paused_state_path": str(pt)},
        cfg,
    )
    assert "--resume-state" in cmd


def test_cmd_builder_resume_after_monitor_state_file(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("", encoding="utf-8")
    cmd = _default_cmd_builder(
        {
            "task_type": "train",
            "monitor_state_path": str(tmp_path / "ms.json"),
            "paused_state_path": str(tmp_path / "p.pt"),
        },
        cfg,
    )
    assert cmd.index("--config") < cmd.index("--monitor-state-file") < cmd.index("--resume-state")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_monitor_state_path_isolated_per_task_same_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.supervisor.cmd_builder import _resolve_monitor_state_path

    base = {"version_id": 999, "project_id": 1}
    p1 = _resolve_monitor_state_path({**base, "id": 1})
    p2 = _resolve_monitor_state_path({**base, "id": 2})

    assert p1 != p2
    assert p1.parts[-4:] == ("tasks", "1", "monitor", "state.json")
    assert p2.parts[-4:] == ("tasks", "2", "monitor", "state.json")


def test_monitor_state_path_no_version_still_task_scoped() -> None:
    from studio.supervisor.cmd_builder import _resolve_monitor_state_path

    pth = _resolve_monitor_state_path({"id": 7})
    assert pth.parts[-4:] == ("tasks", "7", "monitor", "state.json")


def test_task_paths_helpers_form_a_consistent_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.infrastructure import paths as _paths
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")

    msp = _paths.task_monitor_state_path(42)
    samples = _paths.task_samples_dir(42)
    log = _paths.task_log_path(42)
    snapshot = _paths.task_dir(42) / "snapshot" / "config.yaml"

    root = tmp_path / "tasks" / "42"
    assert msp == root / "monitor" / "state.json"
    assert samples == root / "samples"
    assert log == root / "run.log"
    from studio.services.task_snapshot import snapshot_dir
    assert snapshot_dir(42) == root / "snapshot"


# ---------------------------------------------------------------------------
# bootstrap_phase: _maybe_apply_pause_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_overrides_args(tmp_path: Path) -> None:
    from runtime.training.phases.bootstrap import _maybe_apply_pause_snapshot  # type: ignore[import-not-found]

    pt = tmp_path / "pause_step_100.pt"
    pt.write_bytes(b"")
    snap = pt.with_suffix(".config.json")
    snap.write_text(json.dumps({
        "version": 1,
        "args": {"lr": 5e-5, "optimizer": "Lion", "batch_size": 8},
        "sample_prompts": ["snap_prompt"],
    }), encoding="utf-8")

    args = argparse.Namespace(lr=1e-4, optimizer="AdamW", batch_size=2, resume_state=str(pt))
    _maybe_apply_pause_snapshot(args, pt)

    assert args.lr == 5e-5
    assert args.optimizer == "Lion"
    assert args.batch_size == 8
    assert args.sample_prompts == ["snap_prompt"]


def test_snapshot_missing_leaves_args_intact(tmp_path: Path) -> None:
    from runtime.training.phases.bootstrap import _maybe_apply_pause_snapshot

    pt = tmp_path / "training_state_step42.pt"
    pt.write_bytes(b"")

    args = argparse.Namespace(lr=1e-4, optimizer="AdamW", resume_state=str(pt))
    _maybe_apply_pause_snapshot(args, pt)
    assert args.lr == 1e-4
    assert args.optimizer == "AdamW"


def test_snapshot_keeps_resume_state_and_config(tmp_path: Path) -> None:
    from runtime.training.phases.bootstrap import _maybe_apply_pause_snapshot

    pt = tmp_path / "pause_step_50.pt"
    pt.write_bytes(b"")
    snap = pt.with_suffix(".config.json")
    snap.write_text(json.dumps({
        "version": 1,
        "args": {
            "lr": 1e-3,
            "resume_state": "/old/stale/path.pt",
            "config": "/old/stale.yaml",
        },
    }), encoding="utf-8")

    args = argparse.Namespace(
        lr=1e-4,
        resume_state=str(pt),
        config="/current/cfg.yaml",
    )
    _maybe_apply_pause_snapshot(args, pt)
    assert args.lr == 1e-3
    assert args.resume_state == str(pt)
    assert args.config == "/current/cfg.yaml"


def test_snapshot_malformed_falls_back_silently(tmp_path: Path) -> None:
    from runtime.training.phases.bootstrap import _maybe_apply_pause_snapshot

    pt = tmp_path / "pause_step_10.pt"
    pt.write_bytes(b"")
    snap = pt.with_suffix(".config.json")
    snap.write_text("not valid json {{{", encoding="utf-8")

    args = argparse.Namespace(lr=1e-4, resume_state=str(pt))
    _maybe_apply_pause_snapshot(args, pt)
    assert args.lr == 1e-4


def test_snapshot_schema_wrong_falls_back(tmp_path: Path) -> None:
    from runtime.training.phases.bootstrap import _maybe_apply_pause_snapshot

    pt = tmp_path / "pause_step_10.pt"
    pt.write_bytes(b"")
    snap = pt.with_suffix(".config.json")
    snap.write_text(json.dumps({"args": "not a dict"}), encoding="utf-8")

    args = argparse.Namespace(lr=1e-4, resume_state=str(pt))
    _maybe_apply_pause_snapshot(args, pt)
    assert args.lr == 1e-4


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_trigger_prepended_to_sample_prompt() -> None:
    from runtime.training.phases.bootstrap import _prepend_trigger_to_sample_prompts

    args = argparse.Namespace(
        trigger_word="ohwx",
        sample_prompt="1girl, masterpiece",
        sample_prompts=[],
    )
    _prepend_trigger_to_sample_prompts(args)
    assert args.sample_prompt == "ohwx, 1girl, masterpiece"


def test_trigger_prepended_to_sample_prompts_list() -> None:
    from runtime.training.phases.bootstrap import _prepend_trigger_to_sample_prompts

    args = argparse.Namespace(
        trigger_word="ohwx",
        sample_prompt="",
        sample_prompts=["a cat", "a dog"],
    )
    _prepend_trigger_to_sample_prompts(args)
    assert args.sample_prompts == ["ohwx, a cat", "ohwx, a dog"]


def test_trigger_skips_when_already_present_token_match() -> None:
    from runtime.training.phases.bootstrap import _prepend_trigger_to_sample_prompts

    args = argparse.Namespace(
        trigger_word="ohwx",
        sample_prompt="ohwx, 1girl",
        sample_prompts=["1girl, ohwx, blue eyes", "OHWX, masterpiece"],
    )
    _prepend_trigger_to_sample_prompts(args)
    assert args.sample_prompt == "ohwx, 1girl"
    assert args.sample_prompts == [
        "1girl, ohwx, blue eyes",
        "OHWX, masterpiece",
    ]


def test_trigger_empty_or_missing_is_noop() -> None:
    from runtime.training.phases.bootstrap import _prepend_trigger_to_sample_prompts

    args = argparse.Namespace(
        trigger_word="",
        sample_prompt="1girl",
        sample_prompts=["a cat"],
    )
    _prepend_trigger_to_sample_prompts(args)
    assert args.sample_prompt == "1girl"
    assert args.sample_prompts == ["a cat"]

    args2 = argparse.Namespace(sample_prompt="1girl", sample_prompts=["a"])
    _prepend_trigger_to_sample_prompts(args2)
    assert args2.sample_prompt == "1girl"
    assert args2.sample_prompts == ["a"]


def test_trigger_skips_empty_prompt_strings() -> None:
    from runtime.training.phases.bootstrap import _prepend_trigger_to_sample_prompts

    args = argparse.Namespace(
        trigger_word="ohwx",
        sample_prompt="",
        sample_prompts=["", "a cat", ""],
    )
    _prepend_trigger_to_sample_prompts(args)
    assert args.sample_prompt == ""
    assert args.sample_prompts == ["", "ohwx, a cat", ""]


# ---------------------------------------------------------------------------
# bootstrap_phase: _resolve_sample_seed
# ---------------------------------------------------------------------------


def test_resolve_sample_seed_zero_is_replaced_with_random_positive() -> None:
    from runtime.training.phases.bootstrap import _resolve_sample_seed

    args = argparse.Namespace(sample_seed=0)
    _resolve_sample_seed(args)
    assert isinstance(args.sample_seed, int)
    assert 1 <= args.sample_seed <= 2**31 - 1


def test_resolve_sample_seed_explicit_value_kept() -> None:
    from runtime.training.phases.bootstrap import _resolve_sample_seed

    args = argparse.Namespace(sample_seed=42)
    _resolve_sample_seed(args)
    assert args.sample_seed == 42


def test_resolve_sample_seed_idempotent_after_resolve() -> None:
    from runtime.training.phases.bootstrap import _resolve_sample_seed

    args = argparse.Namespace(sample_seed=0)
    _resolve_sample_seed(args)
    first = args.sample_seed
    _resolve_sample_seed(args)
    assert args.sample_seed == first


def test_resolve_sample_seed_missing_attr_treated_as_zero() -> None:
    from runtime.training.phases.bootstrap import _resolve_sample_seed

    args = argparse.Namespace()
    _resolve_sample_seed(args)
    assert isinstance(args.sample_seed, int)
    assert args.sample_seed > 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path):
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    logs.mkdir()
    configs.mkdir()
    return {"db": db_path, "logs": logs, "configs": configs, "root": tmp_path}


def _new_sup(env) -> Supervisor:
    return Supervisor(
        on_event=lambda _: None,
        cmd_builder=lambda *_: ["echo"],
        db_path=env["db"],
        logs_dir=env["logs"],
        configs_dir=env["configs"],
        poll_interval=10,
    )


def test_clear_pause_fields_keeps_files_and_clears_db(env) -> None:
    sup = _new_sup(env)
    state_pt = env["root"] / "auto_epoch_state.pt"
    cfg_json = env["root"] / "auto_epoch_state.config.json"
    state_pt.write_bytes(b"fake state")
    cfg_json.write_text("{}", encoding="utf-8")

    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            status="running",
            paused_state_path=str(state_pt),
            paused_config_path=str(cfg_json),
            paused_step=100,
            paused_at=time.time(),
        )

    sup._clear_pause_fields(tid)

    assert state_pt.exists()
    assert cfg_json.exists()
    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task is not None
    assert task["paused_state_path"] is None
    assert task["paused_config_path"] is None
    assert task["paused_step"] is None
    assert task["paused_at"] is None
    assert task["status"] == "running"


def test_clear_pause_fields_unknown_task_noop(env) -> None:
    sup = _new_sup(env)
    sup._clear_pause_fields(99999)


def test_clear_pause_fields_dangling_paths_robust(env) -> None:
    sup = _new_sup(env)
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            paused_state_path="/nonexistent/path.pt",
            paused_config_path="/nonexistent/path.config.json",
            paused_step=100,
        )
    sup._clear_pause_fields(tid)
    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task["paused_state_path"] is None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.fixture
def server_env(env, monkeypatch):
    if "studio.server" in sys.modules:
        del sys.modules["studio.server"]
    monkeypatch.setattr(db, "STUDIO_DB", env["db"])
    return env


def _create_paused_task(env, state_pt: Path, cfg_json: Path) -> int:
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            status="paused",
            paused_state_path=str(state_pt),
            paused_config_path=str(cfg_json),
            paused_step=100,
            paused_at=time.time(),
        )
    return tid


def _import_server_module():
    try:
        from fastapi import HTTPException  # noqa: F401  (kept for back-compat)
        from studio.api.routers.queue.lifecycle import resume_task as _resume_task
        from studio.domain.errors import ConflictError, NotFoundError
    except ImportError:
        pytest.skip("fastapi not installed; cannot import resume endpoint")

    class _ServerShim:
        pass

    shim = _ServerShim()
    # ADR 0009 Phase 2: resume_task now raises studio.domain.errors.* (DomainError
    # subclasses with .http_status / .message / .code), not fastapi HTTPException.
    shim.HTTPException = HTTPException
    shim.NotFoundError = NotFoundError
    shim.ConflictError = ConflictError
    shim.resume_task = _resume_task
    return shim


def test_resume_endpoint_rejects_unknown_task(server_env) -> None:
    server = _import_server_module()
    with pytest.raises(server.NotFoundError) as exc:
        server.resume_task(99999)
    assert exc.value.http_status == 404
    assert exc.value.code == "task.not_found"


def test_resume_endpoint_rejects_non_resumable_status(server_env) -> None:
    server = _import_server_module()
    with db.connection_for(server_env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        # pending status
    with pytest.raises(server.ConflictError) as exc:
        server.resume_task(tid)
    assert exc.value.http_status == 409
    assert exc.value.code == "task.not_resumable"


def test_resume_endpoint_rejects_when_state_file_missing(server_env) -> None:
    server = _import_server_module()
    state_pt = server_env["root"] / "pause_step_100.pt"
    cfg_json = server_env["root"] / "pause_step_100.config.json"
    cfg_json.write_text("{}", encoding="utf-8")  # state missing, config exists
    tid = _create_paused_task(server_env, state_pt, cfg_json)
    with pytest.raises(server.ConflictError) as exc:
        server.resume_task(tid)
    assert exc.value.http_status == 409
    assert exc.value.code == "task.resume_state_missing"
    assert "missing" in exc.value.message


def test_resume_endpoint_rejects_when_config_snapshot_missing(server_env) -> None:
    server = _import_server_module()
    state_pt = server_env["root"] / "pause_step_100.pt"
    cfg_json = server_env["root"] / "pause_step_100.config.json"
    state_pt.write_bytes(b"")  # state exists, config missing
    tid = _create_paused_task(server_env, state_pt, cfg_json)
    with pytest.raises(server.ConflictError) as exc:
        server.resume_task(tid)
    assert exc.value.http_status == 409
    # config snapshot missing reuses the resume_state_missing code (CATALOG: the
    # frozen config is part of the saved training state).
    assert exc.value.code == "task.resume_state_missing"


def test_resume_endpoint_success_flips_status_keeps_paused_fields(server_env) -> None:
    server = _import_server_module()
    state_pt = server_env["root"] / "pause_step_100.pt"
    cfg_json = server_env["root"] / "pause_step_100.config.json"
    state_pt.write_bytes(b"")
    cfg_json.write_text("{}", encoding="utf-8")
    tid = _create_paused_task(server_env, state_pt, cfg_json)

    result = server.resume_task(tid)
    assert result["status"] == "pending"
    assert result["task_id"] == tid
    with db.connection_for(server_env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task["status"] == "pending"
    assert task["paused_state_path"] == str(state_pt)
    assert task["paused_config_path"] == str(cfg_json)
    assert task["paused_step"] == 100
    assert task["finished_at"] is None
    assert task["exit_code"] is None
    assert task["error_msg"] is None
