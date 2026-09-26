from __future__ import annotations

from pathlib import Path

import pytest

from studio.services.presets import io as presets_io
from studio.schema import TrainingConfig


@pytest.fixture
def presets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pdir = tmp_path / "presets"
    pdir.mkdir()
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", pdir)
    return pdir


def _payload() -> dict:
    return TrainingConfig().model_dump(mode="python")


def test_write_then_read_roundtrip(presets_dir: Path) -> None:
    payload = _payload()
    payload["lora_rank"] = 64
    presets_io.write_preset("alpha", payload)
    assert (presets_dir / "alpha.yaml").exists()
    got = presets_io.read_preset("alpha")
    assert got["lora_rank"] == 64


def test_write_prunes_inactive_fields_read_reinflates(presets_dir: Path) -> None:
    import yaml

    payload = _payload()
    payload["came_beta1"] = 0.5
    presets_io.write_preset("pruned", payload)

    raw = yaml.safe_load((presets_dir / "pruned.yaml").read_text(encoding="utf-8"))
    assert "came_beta1" not in raw
    assert "infonoise_K" not in raw
    assert "lr_scheduler" in raw

    got = presets_io.read_preset("pruned")
    assert got["came_beta1"] == TrainingConfig().came_beta1


def test_write_invalid_rejected(presets_dir: Path) -> None:
    with pytest.raises(presets_io.PresetError):
        presets_io.write_preset("bad", {"lora_rank": "not-an-int"})
    assert not list(presets_dir.glob("*.yaml"))


def test_name_validation(presets_dir: Path) -> None:
    for bad in ("../escape", "name with space", "name/sub", "name.dot"):
        with pytest.raises(presets_io.PresetError, match="Invalid preset name"):
            presets_io.write_preset(bad, _payload())


def test_list_sorted_by_mtime(presets_dir: Path) -> None:
    import time
    presets_io.write_preset("first", _payload())
    time.sleep(0.05)
    presets_io.write_preset("second", _payload())
    items = presets_io.list_presets()
    assert [x["name"] for x in items[:2]] == ["second", "first"]


def test_delete(presets_dir: Path) -> None:
    presets_io.write_preset("to_delete", _payload())
    presets_io.delete_preset("to_delete")
    assert not (presets_dir / "to_delete.yaml").exists()


def test_delete_missing_raises(presets_dir: Path) -> None:
    with pytest.raises(presets_io.PresetError, match="not found"):
        presets_io.delete_preset("ghost")


def test_duplicate(presets_dir: Path) -> None:
    payload = _payload()
    payload["lora_rank"] = 16
    presets_io.write_preset("src", payload)
    presets_io.duplicate_preset("src", "src_copy")
    assert (presets_dir / "src_copy.yaml").exists()
    assert presets_io.read_preset("src_copy")["lora_rank"] == 16


def test_duplicate_conflict(presets_dir: Path) -> None:
    presets_io.write_preset("a", _payload())
    presets_io.write_preset("b", _payload())
    with pytest.raises(presets_io.PresetError, match="already exists"):
        presets_io.duplicate_preset("a", "b")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_parse_yaml_returns_config_and_suggested_name() -> None:
    import yaml
    payload = _payload()
    payload["epochs"] = 12
    raw = yaml.safe_dump(payload, allow_unicode=True).encode("utf-8")
    config, suggested = presets_io.parse_preset_bytes(raw, "my-run.yaml")
    assert config["epochs"] == 12
    assert suggested == "my-run"


def test_parse_json_works_via_yaml_superset() -> None:
    import json
    raw = json.dumps(_payload()).encode("utf-8")
    config, suggested = presets_io.parse_preset_bytes(raw, "old.json")
    assert config["lora_type"] == "lora"
    assert suggested == "old"


def test_parse_drops_unknown_field() -> None:
    import yaml
    bad = _payload()
    bad["nonexistent_field"] = 123
    raw = yaml.safe_dump(bad).encode("utf-8")
    cfg, suggested = presets_io.parse_preset_bytes(raw, "bad.yaml")
    assert suggested == "bad"
    assert "nonexistent_field" not in cfg


def test_parse_migrates_legacy_attention_fields() -> None:
    import yaml
    legacy = _payload()
    legacy.pop("attention_backend", None)
    legacy["flash_attn"] = False
    legacy["xformers"] = True
    raw = yaml.safe_dump(legacy).encode("utf-8")
    cfg, _ = presets_io.parse_preset_bytes(raw, "legacy.yaml")
    assert cfg["attention_backend"] == "xformers"
    assert "flash_attn" not in cfg
    assert "xformers" not in cfg


def test_parse_rejects_non_mapping() -> None:
    raw = b"- foo\n- bar\n"
    with pytest.raises(presets_io.PresetError, match="Preset file format is invalid"):
        presets_io.parse_preset_bytes(raw, "list.yaml")


def test_parse_rejects_invalid_utf8() -> None:
    raw = b"\xff\xfe\x00\x00bogus"
    with pytest.raises(presets_io.PresetError, match="UTF-8"):
        presets_io.parse_preset_bytes(raw, "binary.yaml")


def test_parse_sanitizes_suggested_name() -> None:
    import yaml
    raw = yaml.safe_dump(_payload()).encode("utf-8")
    _, suggested = presets_io.parse_preset_bytes(raw, "my preset (v2).yaml")
    assert all(c.isalnum() or c in "_-" for c in suggested)
    assert "v2" in suggested


def test_parse_empty_filename_fallback() -> None:
    import yaml
    raw = yaml.safe_dump(_payload()).encode("utf-8")
    _, suggested = presets_io.parse_preset_bytes(raw, "")
    assert suggested == "imported"


def test_preset_path_is_public_alias() -> None:
    assert presets_io.preset_path("foo").name == "foo.yaml"
    with pytest.raises(presets_io.PresetError, match="Invalid preset name"):
        presets_io.preset_path("bad/name")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_tolerant_validate_infonoise_mutex_disables_infonoise() -> None:
    cases = [
        {"infonoise_enabled": True, "noise_enhancement_type": "offset"},
        {"infonoise_enabled": True, "loss_weighting": "detail_inv_t"},
        {"infonoise_enabled": True, "loss_type": "huber"},
        {"infonoise_enabled": True, "timestep_schedule_shift": 2.0},
    ]
    for overrides in cases:
        cfg, _, defaulted = presets_io._tolerant_validate(overrides)
        assert cfg.infonoise_enabled is False, overrides
        assert "infonoise_enabled" in defaulted, overrides
        for k, v in overrides.items():
            if k == "infonoise_enabled":
                continue
            assert getattr(cfg, k) == v, overrides


def test_tolerant_validate_silently_drops_retired_monitor_keys() -> None:
    cfg, dropped, defaulted = presets_io._tolerant_validate({
        "no_monitor": True,
        "monitor_host": "127.0.0.1",
        "monitor_port": 8765,
        "no_browser": True,
        "epochs": 12,
    })
    assert dropped == []
    assert defaulted == []
    assert cfg.epochs == 12
    _, dropped2, _ = presets_io._tolerant_validate({"totally_unknown_key": 1})
    assert dropped2 == ["totally_unknown_key"]


def test_tolerant_validate_infonoise_mutex_multi_conflict_one_pass() -> None:
    cfg, _, defaulted = presets_io._tolerant_validate({
        "infonoise_enabled": True,
        "noise_enhancement_type": "offset",
        "loss_weighting": "detail_inv_t",
        "loss_type": "huber",
        "timestep_schedule_shift": 2.0,
    })
    assert cfg.infonoise_enabled is False
    assert cfg.noise_enhancement_type == "offset"
    assert cfg.loss_weighting == "detail_inv_t"
    assert cfg.loss_type == "huber"
    assert cfg.timestep_schedule_shift == 2.0
    assert defaulted == ["infonoise_enabled"]


def test_write_preset_strict_path_still_rejects_infonoise_mutex(
    presets_dir: Path,
) -> None:
    with pytest.raises(presets_io.PresetError):
        presets_io.write_preset("conflict", {
            **_payload(),
            "infonoise_enabled": True,
            "loss_type": "huber",
        })


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_read_preset_absolutizes_relative_paths(presets_dir: Path) -> None:
    import yaml
    from studio.paths import REPO_ROOT
    raw = TrainingConfig().model_dump()
    raw["transformer_path"] = "models/diffusion_models/anima-base-v1.0.safetensors"
    raw["vae_path"] = "models/vae/qwen_image_vae.safetensors"
    (presets_dir / "legacy.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )
    got = presets_io.read_preset("legacy")
    assert Path(got["transformer_path"]).is_absolute()
    assert "\\" not in got["transformer_path"]
    assert got["transformer_path"] == (
        REPO_ROOT / "models/diffusion_models/anima-base-v1.0.safetensors"
    ).resolve().as_posix()
    assert got["vae_path"] == (
        REPO_ROOT / "models/vae/qwen_image_vae.safetensors"
    ).resolve().as_posix()


def test_read_preset_normalizes_to_posix(presets_dir: Path) -> None:
    import yaml
    raw = TrainingConfig().model_dump()
    abs_path = str(Path("/data/anima/custom.safetensors").resolve())
    raw["transformer_path"] = abs_path
    (presets_dir / "modern.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )
    got = presets_io.read_preset("modern")
    assert "\\" not in got["transformer_path"]
    assert got["transformer_path"] == Path(abs_path).as_posix()


def test_write_preset_absolutizes_relative_paths(presets_dir: Path) -> None:
    import yaml
    payload = _payload()
    payload["transformer_path"] = "models/foo.safetensors"
    presets_io.write_preset("alpha", payload)
    raw = yaml.safe_load((presets_dir / "alpha.yaml").read_text(encoding="utf-8"))
    assert Path(raw["transformer_path"]).is_absolute()
    assert "\\" not in raw["transformer_path"]


def test_absolutize_preserves_windows_drive_letter_on_posix(monkeypatch) -> None:
    from studio.services.presets import io as presets_io
    from studio.services.presets.io import _absolutize_model_paths

    monkeypatch.setattr(
        presets_io.Path, "is_absolute", lambda self: False
    )

    data = {
        "transformer_path": "G:/models/diffusion_models/anima.safetensors",
        "vae_path": "D:\\anima\\vae.safetensors",
    }
    out = _absolutize_model_paths(data)
    assert out["transformer_path"] == "G:/models/diffusion_models/anima.safetensors"
    assert out["vae_path"] == "D:/anima/vae.safetensors"
