from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from studio import db, server
from studio.services.projects import projects, versions
from studio.services import presets as preset_flow, version_config


@pytest.fixture(autouse=True)
def _fixed_selected(monkeypatch):
    from studio import secrets

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets(models={
        "selected": {"anima": "1.0", "krea2": "raw"},
    }))


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    from studio.services.presets import io as presets_io
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", presets_dir)
    return {"db": dbfile, "presets": presets_dir}


def _make_pv(env) -> tuple[dict, dict]:
    with db.connection_for(env["db"]) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
    return p, v


# ---------------------------------------------------------------------------
# project_specific_overrides
# ---------------------------------------------------------------------------


def test_project_specific_overrides_uses_version_dir(env) -> None:
    p, v = _make_pv(env)
    ov = version_config.project_specific_overrides(p, v)
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    assert ov["data_dir"] == str(vdir / "train")
    assert ov["output_dir"] == str(vdir / "output")
    assert ov["output_name"] == f"{p['slug']}_baseline"
    assert ov["reg_data_dir"] is None
    assert ov["resume_lora"] is None
    assert ov["resume_state"] is None


def test_project_specific_overrides_includes_reg_when_meta_exists(env) -> None:
    p, v = _make_pv(env)
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    (vdir / "reg").mkdir(parents=True, exist_ok=True)
    (vdir / "reg" / "meta.json").write_text("{}", encoding="utf-8")
    ov = version_config.project_specific_overrides(p, v)
    assert ov["reg_data_dir"] == str(vdir / "reg")


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------


def _minimal_config(**overrides) -> dict:
    from studio.schema import TrainingConfig
    return {**TrainingConfig().model_dump(), **overrides}


def test_has_version_config_false_initially(env) -> None:
    p, v = _make_pv(env)
    assert version_config.has_version_config(p, v) is False


def test_write_then_read(env) -> None:
    p, v = _make_pv(env)
    cfg_in = _minimal_config(lora_rank=64)
    version_config.write_version_config(p, v, cfg_in)
    assert version_config.has_version_config(p, v) is True
    cfg_out = version_config.read_version_config(p, v)
    assert cfg_out["lora_rank"] == 64


def test_write_prunes_inactive_fields(env) -> None:
    p, v = _make_pv(env)
    cfg_in = _minimal_config(optimizer_type="came", came_beta1=0.5)
    version_config.write_version_config(p, v, cfg_in)

    raw = yaml.safe_load(
        version_config.version_config_path(p, v).read_text(encoding="utf-8")
    )
    assert raw["came_beta1"] == 0.5
    assert "lion_beta1" not in raw
    assert "infonoise_K" not in raw
    assert "lr_scheduler" in raw

    cfg_out = version_config.read_version_config(p, v)
    assert cfg_out["came_beta1"] == 0.5
    assert "lion_beta1" in cfg_out


def test_write_tolerates_stale_preset_fields(env) -> None:
    p, v = _make_pv(env)
    cfg_in = _minimal_config(
        lora_rank=64,
        optimizer_type="made_up_optim",
        future_field_from_other_branch=True,
    )
    version_config.write_version_config(p, v, cfg_in)
    cfg_out = version_config.read_version_config(p, v)
    assert cfg_out["lora_rank"] == 64
    assert cfg_out["optimizer_type"] != "made_up_optim"
    assert "future_field_from_other_branch" not in cfg_out


def test_read_tolerates_stale_version_config(env) -> None:
    p, v = _make_pv(env)
    raw = _minimal_config(
        lora_rank=96,
        optimizer_type="made_up_optim",
        future_field_from_other_branch=True,
    )
    path = version_config.version_config_path(p, v)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    cfg_out = version_config.read_version_config(p, v)
    assert cfg_out["lora_rank"] == 96
    assert cfg_out["optimizer_type"] != "made_up_optim"
    assert "future_field_from_other_branch" not in cfg_out


def test_write_forces_project_overrides(env) -> None:
    p, v = _make_pv(env)
    cfg = _minimal_config(
        data_dir="/some/wrong/path",
        output_dir="/another/wrong/path",
        output_name="hacker",
    )
    version_config.write_version_config(p, v, cfg)
    out = version_config.read_version_config(p, v)
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    assert out["data_dir"] == str(vdir / "train")
    assert out["output_dir"] == str(vdir / "output")
    assert out["output_name"] == f"{p['slug']}_baseline"


def test_overrides_keep_resume_when_reset_disabled(env) -> None:
    p, v = _make_pv(env)
    ov = version_config.project_specific_overrides(p, v, reset_resume=False)
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    assert ov["data_dir"] == str(vdir / "train")
    assert "resume_lora" not in ov
    assert "resume_state" not in ov


def test_write_keeps_resume_when_reset_disabled(env, tmp_path) -> None:
    p, v = _make_pv(env)
    ckpt = tmp_path / "prev.safetensors"
    ckpt.write_bytes(b"x")
    cfg = _minimal_config(resume_lora=str(ckpt), data_dir="/some/wrong/path")
    version_config.write_version_config(
        p, v, cfg, force_project_overrides=True, reset_resume=False,
    )
    out = version_config.read_version_config(p, v)
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    assert out["resume_lora"] == str(ckpt)
    assert out["data_dir"] == str(vdir / "train")


def test_enqueue_keeps_user_resume_lora(env, tmp_path) -> None:
    p, v = _make_pv(env)
    ckpt = tmp_path / "prev.safetensors"
    ckpt.write_bytes(b"x")
    version_config.write_version_config(
        p, v, _minimal_config(resume_lora=str(ckpt)),
        force_project_overrides=False,
    )
    client = TestClient(server.app)
    resp = client.post(f"/api/projects/{p['id']}/versions/{v['id']}/queue")
    assert resp.status_code == 200, resp.text
    assert version_config.read_version_config(p, v)["resume_lora"] == str(ckpt)


def test_read_missing_raises(env) -> None:
    p, v = _make_pv(env)
    with pytest.raises(version_config.VersionConfigError):
        version_config.read_version_config(p, v)


def test_delete_version_config(env) -> None:
    p, v = _make_pv(env)
    version_config.write_version_config(p, v, _minimal_config())
    assert version_config.delete_version_config(p, v) is True
    assert version_config.delete_version_config(p, v) is False
    assert version_config.has_version_config(p, v) is False


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _seed_preset(env, name: str, **overrides) -> None:
    from studio.services.presets import io as presets_io
    presets_io.write_preset(name, _minimal_config(**overrides))


def test_fork_preset_for_version_applies_overrides(env) -> None:
    p, v = _make_pv(env)
    _seed_preset(env, "tpl", lora_rank=128, data_dir="/wrong")
    cfg = preset_flow.fork_preset_for_version("tpl", p, v)
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    assert cfg["data_dir"] == str(vdir / "train")
    assert cfg["output_name"] == f"{p['slug']}_baseline"
    assert cfg["lora_rank"] == 128


def test_fork_preset_for_version_reports_warnings(env) -> None:
    from studio.services.presets import io as presets_io
    p, v = _make_pv(env)
    raw = _minimal_config(
        lora_rank=128,
        optimizer_type="made_up_optim",
        future_field_from_other_branch=True,
    )
    (env["presets"] / "stale.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )

    cfg, dropped, defaulted = preset_flow.fork_preset_for_version_with_warnings(
        "stale", p, v
    )

    assert cfg["lora_rank"] == 128
    assert dropped == ["future_field_from_other_branch"]
    assert "optimizer_type" in defaulted
    assert presets_io.read_preset("stale")["optimizer_type"] != "made_up_optim"


def test_fork_preset_endpoint_returns_warnings(env) -> None:
    p, v = _make_pv(env)
    raw = _minimal_config(
        lora_rank=128,
        optimizer_type="made_up_optim",
        future_field_from_other_branch=True,
    )
    (env["presets"] / "stale.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )

    resp = TestClient(server.app).post(
        f"/api/projects/{p['id']}/versions/{v['id']}/config/from_preset",
        json={"name": "stale"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["config"]["lora_rank"] == 128
    assert body["dropped_fields"] == ["future_field_from_other_branch"]
    assert "optimizer_type" in body["defaulted_fields"]


def test_fork_then_modify_does_not_change_preset(env) -> None:
    p, v = _make_pv(env)
    _seed_preset(env, "tpl", lora_rank=32)
    preset_flow.fork_preset_for_version("tpl", p, v)
    cfg = version_config.read_version_config(p, v)
    cfg["lora_rank"] = 128
    version_config.write_version_config(p, v, cfg)
    from studio.services.presets import io as presets_io
    preset_now = presets_io.read_preset("tpl")
    assert preset_now["lora_rank"] == 32


def test_save_version_config_as_preset_clears_project_fields(env) -> None:
    p, v = _make_pv(env)
    _seed_preset(env, "tpl", lora_rank=64)
    preset_flow.fork_preset_for_version("tpl", p, v)

    saved = preset_flow.save_version_config_as_preset(p, v, "my-tuned")
    assert saved["data_dir"] == "./dataset"
    assert saved["output_dir"] == "./output"
    assert saved["output_name"] == "anima_lora"
    assert saved["reg_data_dir"] is None
    assert saved["lora_rank"] == 64

    yaml_path = env["presets"] / "my-tuned.yaml"
    assert yaml_path.exists()
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert raw["lora_rank"] == 64


def test_save_as_preset_rejects_existing_without_overwrite(env) -> None:
    from studio.services.presets import io as presets_io
    p, v = _make_pv(env)
    _seed_preset(env, "tpl", lora_rank=64)
    preset_flow.fork_preset_for_version("tpl", p, v)
    with pytest.raises(presets_io.PresetError):
        preset_flow.save_version_config_as_preset(p, v, "tpl", overwrite=False)
    preset_flow.save_version_config_as_preset(p, v, "tpl", overwrite=True)


def test_save_as_preset_rejects_invalid_name(env) -> None:
    from studio.services.presets import io as presets_io
    p, v = _make_pv(env)
    _seed_preset(env, "tpl")
    preset_flow.fork_preset_for_version("tpl", p, v)
    with pytest.raises(presets_io.PresetError):
        preset_flow.save_version_config_as_preset(p, v, "../etc/passwd")


# ---------------------------------------------------------------------------
# auto_sync_paths toggle (PP10.5)
# ---------------------------------------------------------------------------


def _custom_path() -> str:
    return Path("/tmp/anima-custom/foo.safetensors").resolve().as_posix()


def _normalize_default(path_str: str) -> str:
    return Path(path_str).as_posix()


def test_fork_with_toggle_on_overrides_model_paths(env, monkeypatch) -> None:
    from studio.services import models as model_downloader
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: True)
    p, v = _make_pv(env)
    custom = _custom_path()
    _seed_preset(env, "tpl", transformer_path=custom)
    cfg = preset_flow.fork_preset_for_version("tpl", p, v)
    expected = _normalize_default(model_downloader.default_paths_for_new_version()["transformer_path"])
    assert cfg["transformer_path"] == expected
    assert cfg["transformer_path"] != custom


def test_fork_with_toggle_off_respects_preset(env, monkeypatch) -> None:
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: False)
    p, v = _make_pv(env)
    custom = _custom_path()
    _seed_preset(env, "tpl", transformer_path=custom)
    cfg = preset_flow.fork_preset_for_version("tpl", p, v)
    assert cfg["transformer_path"] == custom


def test_fork_krea2_preset_syncs_krea2_paths(env, monkeypatch) -> None:
    from studio.services import models as model_downloader
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: True)
    p, v = _make_pv(env)
    _seed_preset(env, "tpl", model_family="krea2", shuffle_caption=False,
                 sample_sampler_name="euler", sample_scheduler="simple",
                 transformer_path=_custom_path())
    cfg = preset_flow.fork_preset_for_version("tpl", p, v)
    assert cfg["model_family"] == "krea2"
    expected = _normalize_default(
        model_downloader.default_paths_for_new_version(family="krea2")
        ["transformer_path"]
    )
    assert _normalize_default(cfg["transformer_path"]) == expected
    assert cfg["transformer_path"].endswith("krea2-raw-bf16.safetensors")
    assert "Qwen_Qwen3-VL-4B-Instruct" in cfg["text_encoder_path"]


def test_save_as_preset_toggle_on_clears_model_paths(env, monkeypatch) -> None:
    from studio.services import models as model_downloader
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: False)
    p, v = _make_pv(env)
    custom = _custom_path()
    _seed_preset(env, "tpl", transformer_path=custom)
    preset_flow.fork_preset_for_version("tpl", p, v)
    assert version_config.read_version_config(p, v)["transformer_path"] == custom
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: True)
    saved = preset_flow.save_version_config_as_preset(p, v, "saved")
    expected = _normalize_default(model_downloader.default_paths_for_new_version()["transformer_path"])
    assert saved["transformer_path"] == expected
    assert saved["transformer_path"] != custom


def test_save_as_preset_toggle_off_keeps_model_paths(env, monkeypatch) -> None:
    monkeypatch.setattr(preset_flow, "_auto_sync_paths", lambda: False)
    p, v = _make_pv(env)
    custom = _custom_path()
    _seed_preset(env, "tpl", transformer_path=custom)
    preset_flow.fork_preset_for_version("tpl", p, v)
    saved = preset_flow.save_version_config_as_preset(p, v, "saved")
    assert saved["transformer_path"] == custom
