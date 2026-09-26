from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import secrets, server


@pytest.fixture
def secrets_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sf = tmp_path / "secrets.json"
    monkeypatch.setattr(secrets, "SECRETS_FILE", sf)
    return sf


@pytest.fixture
def client(secrets_file: Path) -> TestClient:  # noqa: ARG001 (fixture chains the patch)
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


def test_defaults_when_file_missing(secrets_file: Path) -> None:
    assert not secrets_file.exists()
    s = secrets.load()
    assert s.gelbooru.user_id == ""
    assert s.gelbooru.api_key == ""
    assert s.wd14.threshold_general == pytest.approx(0.35)
    joy = next(p for p in s.llm_tagger.presets if p.id == "joycaption")
    assert joy.base_url.startswith("http://")
    assert s.wandb.active.project == "AnimaLoraStudio"
    assert s.wandb.current_preset == s.wandb.presets[0].id


def test_load_corrupt_json_returns_defaults(secrets_file: Path) -> None:
    secrets_file.write_text("{not valid json", encoding="utf-8")
    s = secrets.load()
    assert s.gelbooru.user_id == ""


def test_wd14_defaults_include_candidate_list(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.wd14.model_id in s.wd14.model_ids
    assert set(secrets.DEFAULT_WD14_MODELS).issubset(set(s.wd14.model_ids))


def test_cltagger_defaults_use_1_02(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.cltagger.model_id == "cella110n/cl_tagger"
    assert s.cltagger.model_path == "cl_tagger_1_02/model.onnx"
    assert s.cltagger.tag_mapping_path == "cl_tagger_1_02/tag_mapping.json"
    assert s.cltagger.threshold_character == pytest.approx(0.6)

def test_legacy_local_dir_dropped_on_load(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({
            "wd14": {"local_dir": "/old/wd14"},
            "cltagger": {
                "local_dir": "/old/cltagger",
                "variant_local_dirs": {"cl_tagger_1_02": "/old/cltagger"},
            },
        }),
        encoding="utf-8",
    )
    s = secrets.load()
    assert not hasattr(s.wd14, "local_dir")
    assert not hasattr(s.cltagger, "local_dir")
    assert not hasattr(s.cltagger, "variant_local_dirs")


def test_eval_metrics_defaults_and_persistence(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.eval_metrics.clip_model_name == "openai/clip-vit-base-patch32"
    assert s.eval_metrics.dino_model_name == "facebook/dinov2-small"
    assert s.eval_metrics.eval_baseline_enabled is True

    secrets.update({
        "eval_metrics": {
            "clip_model_name": "/models/clip",
            "dino_model_name": "/models/dino",
            "eval_baseline_enabled": False,
        }
    })
    saved = secrets.load()
    assert saved.eval_metrics.clip_model_name == "/models/clip"
    assert saved.eval_metrics.dino_model_name == "/models/dino"
    assert saved.eval_metrics.eval_baseline_enabled is False


def test_llm_tagger_defaults(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.llm_tagger.current_preset == "style_json"
    assert [p.id for p in s.llm_tagger.presets] == [
        "style_json",
        "general_json",
        "txt_tags",
        "joycaption",
        "assist_json",
        "assist_text",
    ]
    assert all(p.builtin for p in s.llm_tagger.presets)
    joy = next(p for p in s.llm_tagger.presets if p.id == "joycaption")
    assert joy.base_url == "http://localhost:8000/v1"
    assert joy.model.endswith("joycaption-beta-one-hf-llava")
    assert joy.endpoint == "chat_completions"
    assert joy.output_format == "text"
    assert joy.temperature == pytest.approx(0.6)
    assert joy.max_tokens == 300
    assert joy.concurrency == 1
    assert joy.requests_per_second == pytest.approx(0.0)
    assert joy.max_requests_per_minute == 0


def test_llm_preset_normalizes_request_pool_settings(secrets_file: Path) -> None:
    s = secrets.update(
        {
            "llm_tagger": {
                "presets": [
                    {
                        "id": "style_json",
                        "concurrency": 99,
                        "requests_per_second": -5,
                        "max_requests_per_minute": 9999,
                    }
                ]
            }
        }
    )
    style = next(p for p in s.llm_tagger.presets if p.id == "style_json")
    assert style.concurrency == 8
    assert style.requests_per_second == pytest.approx(0.0)
    assert style.max_requests_per_minute == 3600


def test_llm_preset_keeps_model_in_model_ids(secrets_file: Path) -> None:
    s = secrets.update(
        {
            "llm_tagger": {
                "presets": [{"id": "joycaption", "model": "vision-a", "model_ids": []}]
            }
        }
    )
    joy = next(p for p in s.llm_tagger.presets if p.id == "joycaption")
    assert joy.model == "vision-a"
    assert joy.model_ids == ["vision-a"]


def test_llm_preset_assist_tagger_normalization() -> None:
    assert secrets.LLMPresetConfig(id="p", assist_tagger="wd14").assist_tagger == "wd14"
    assert (
        secrets.LLMPresetConfig(id="p", assist_tagger="cltagger").assist_tagger
        == "cltagger"
    )
    # Invalid values normalize to off.
    assert secrets.LLMPresetConfig(id="p", assist_tagger="bogus").assist_tagger == ""
    assert secrets.LLMPresetConfig(id="p", assist_tagger="joycaption").assist_tagger == ""
    assert secrets.LLMPresetConfig(id="p").assist_tagger == ""


def test_builtin_assist_presets_carry_tags_placeholder() -> None:
    from studio.infrastructure.llm_presets import builtin_llm_presets

    by_id = {p["id"]: p for p in builtin_llm_presets()}
    for pid in ("assist_json", "assist_text"):
        assert pid in by_id, f"missing builtin assist preset {pid}"
        cfg = secrets.LLMPresetConfig(**by_id[pid])
        assert cfg.assist_tagger in ("wd14", "cltagger")
        assert any("{{tags}}" in m.content for m in cfg.messages if m.type == "text")


def test_wd14_legacy_file_without_model_ids_gets_defaults(
    secrets_file: Path,
) -> None:
    secrets_file.write_text(
        json.dumps({"wd14": {"model_id": "Custom/my-tagger"}}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert "Custom/my-tagger" in s.wd14.model_ids
    for m in secrets.DEFAULT_WD14_MODELS:
        assert m in s.wd14.model_ids


def test_wd14_empty_model_ids_falls_back_to_defaults(
    secrets_file: Path,
) -> None:
    secrets.update({"wd14": {"model_ids": []}})
    s = secrets.load()
    assert list(s.wd14.model_ids) == list(secrets.DEFAULT_WD14_MODELS)


def test_wd14_cannot_drop_current_model_id(secrets_file: Path) -> None:
    secrets.update(
        {"wd14": {"model_id": "SmilingWolf/wd-vit-tagger-v3"}}
    )
    s = secrets.update({"wd14": {"model_ids": ["A/m1", "B/m2"]}})
    assert s.wd14.model_id == "SmilingWolf/wd-vit-tagger-v3"
    assert s.wd14.model_id in s.wd14.model_ids


def test_download_sources_default_seeds_huggingface(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.download_sources == {
        "training": "huggingface",
        "wd14": "huggingface",
        "upscaler": "huggingface",
    }


def test_download_sources_migrate_from_legacy_global(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({"download_source": "modelscope"}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.download_sources["training"] == "modelscope"
    assert s.download_sources["wd14"] == "modelscope"
    assert s.download_sources["upscaler"] == "modelscope"


def test_download_sources_explicit_override_not_clobbered_by_legacy(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({
            "download_source": "modelscope",
            "download_sources": {"training": "huggingface"},
        }),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.download_sources["training"] == "huggingface"
    assert s.download_sources["wd14"] == "modelscope"


def test_download_sources_persist_and_normalize(secrets_file: Path) -> None:
    secrets.update({"download_sources": {"wd14": "modelscope", "upscaler": "garbage"}})
    s = secrets.load()
    assert s.download_sources["wd14"] == "modelscope"
    assert s.download_sources["upscaler"] == "huggingface"


def test_download_image_settings_default(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.download.save_tags is False
    assert s.download.convert_to_png is True
    assert s.download.remove_alpha_channel is True


def test_migrate_gelbooru_image_settings_to_download(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({
            "gelbooru": {
                "user_id": "u",
                "save_tags": True,
                "convert_to_png": False,
                "remove_alpha_channel": False,
            }
        }),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.download.save_tags is True
    assert s.download.convert_to_png is False
    assert s.download.remove_alpha_channel is False
    assert s.gelbooru.user_id == "u"


def test_models_root_default_none(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.models.root is None


def test_models_root_persists(secrets_file: Path) -> None:
    secrets.update({"models": {"root": "/data/anima"}})
    assert secrets.load().models.root == "/data/anima"
    secrets.update({"models": {"root": None}})
    assert secrets.load().models.root is None


def test_model_downloader_uses_secrets_root(
    secrets_file: Path, tmp_path: Path
) -> None:
    from studio.services import models as model_downloader
    secrets.update({"models": {"root": None}})
    fallback = model_downloader.models_root()
    assert fallback.name == "models"
    custom = tmp_path / "custom_models"
    secrets.update({"models": {"root": str(custom)}})
    assert model_downloader.models_root() == custom


def test_find_anima_main_picks_latest(secrets_file: Path, tmp_path: Path) -> None:
    from studio.services import models as model_downloader
    secrets.update({"models": {"root": str(tmp_path)}})
    dm = tmp_path / "diffusion_models"
    dm.mkdir(parents=True)

    assert model_downloader.find_anima_main() is None

    (dm / "anima-preview2.safetensors").write_bytes(b"x")
    assert model_downloader.find_anima_main().name == "anima-preview2.safetensors"

    (dm / "anima-preview3-base.safetensors").write_bytes(b"y")
    assert (
        model_downloader.find_anima_main().name == "anima-preview3-base.safetensors"
    )

    (dm / "anima-base-v1.0.safetensors").write_bytes(b"z")
    assert (
        model_downloader.find_anima_main().name == "anima-base-v1.0.safetensors"
    )


def test_wd14_user_can_replace_current_then_drop(secrets_file: Path) -> None:
    s = secrets.update({"wd14": {"model_id": "A/m1"}})
    assert "A/m1" in s.wd14.model_ids
    s = secrets.update({"wd14": {"model_id": "B/m2"}})
    assert s.wd14.model_id == "B/m2"
    s = secrets.update({"wd14": {"model_ids": [m for m in s.wd14.model_ids if m != "A/m1"]}})
    assert "A/m1" not in s.wd14.model_ids
    assert "B/m2" in s.wd14.model_ids


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_reg_default_excluded_empty_by_default(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.reg.default_excluded_tags == []


def test_reg_default_excluded_round_trip(secrets_file: Path) -> None:
    secrets.update(
        {"reg": {"default_excluded_tags": ["white background", "signature"]}}
    )
    assert secrets.load().reg.default_excluded_tags == [
        "white background",
        "signature",
    ]


def test_reg_legacy_file_without_reg_field(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({"gelbooru": {"user_id": "alice"}}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.reg.default_excluded_tags == []
    assert s.gelbooru.user_id == "alice"


def test_reg_default_excluded_in_masked_dict(secrets_file: Path) -> None:
    secrets.update({"reg": {"default_excluded_tags": ["lowres"]}})
    masked = secrets.to_masked_dict(secrets.load())
    assert masked["reg"]["default_excluded_tags"] == ["lowres"]


# ---------------------------------------------------------------------------
# update / mask round-trip
# ---------------------------------------------------------------------------


def test_update_writes_file(secrets_file: Path) -> None:
    secrets.update({"gelbooru": {"user_id": "alice", "api_key": "k1"}})
    on_disk = json.loads(secrets_file.read_text(encoding="utf-8"))
    assert on_disk["gelbooru"]["user_id"] == "alice"
    assert on_disk["gelbooru"]["api_key"] == "k1"


def test_update_deep_merge_preserves_other_sections(secrets_file: Path) -> None:
    secrets.update({"huggingface": {"token": "hf_x"}})
    secrets.update({"gelbooru": {"user_id": "bob"}})
    s = secrets.load()
    assert s.huggingface.token == "hf_x"
    assert s.gelbooru.user_id == "bob"


def test_update_mask_keeps_existing_value(secrets_file: Path) -> None:
    secrets.update({"gelbooru": {"api_key": "real-key"}})
    secrets.update({"gelbooru": {"api_key": secrets.MASK, "user_id": "bob"}})
    s = secrets.load()
    assert s.gelbooru.api_key == "real-key"
    assert s.gelbooru.user_id == "bob"


def test_to_masked_dict_replaces_sensitive(secrets_file: Path) -> None:
    secrets.update(
        {
            "gelbooru": {"user_id": "alice", "api_key": "secret"},
            "huggingface": {"token": "hf_secret"},
            "wandb": {"presets": [{"id": "default", "api_key": "wandb_secret"}]},
            "llm_tagger": {
                "presets": [{"id": "joycaption", "api_key": "llm_secret"}]
            },
        }
    )
    masked = secrets.to_masked_dict(secrets.load())
    assert masked["gelbooru"]["user_id"] == "alice"
    assert masked["gelbooru"]["api_key"] == secrets.MASK
    assert masked["huggingface"]["token"] == secrets.MASK
    assert masked["wandb"]["presets"][0]["api_key"] == secrets.MASK
    joy_masked = next(p for p in masked["llm_tagger"]["presets"] if p["id"] == "joycaption")
    assert joy_masked["api_key"] == secrets.MASK


def test_to_masked_dict_keeps_empty_sensitive_empty(secrets_file: Path) -> None:
    masked = secrets.to_masked_dict(secrets.load())
    assert masked["gelbooru"]["api_key"] == ""
    assert masked["huggingface"]["token"] == ""
    for preset in masked["wandb"]["presets"]:
        assert preset["api_key"] == ""
    for preset in masked["llm_tagger"]["presets"]:
        assert preset["api_key"] == ""


def test_llm_tagger_legacy_schema_migration(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps(
            {
                "joycaption": {
                    "base_url": "http://my-vllm:9000/v1",
                    "model": "my-custom-joycaption",
                    "prompt_template": "My custom prompt",
                },
                "llm_tagger": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "sk-xxx",
                    "model": "gpt-4o-mini",
                    "model_ids": ["gpt-4o-mini", "gpt-4o"],
                    "endpoint": "chat_completions",
                    "prompt_preset": "style_json",
                    "prompt_presets": [
                        {"id": "style_json", "label": "Style", "prompt": "P1", "builtin": True, "output_format": "json"},
                    ],
                    "custom_prompt": "",
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
            }
        ),
        encoding="utf-8",
    )
    s = secrets.load()
    style = next(p for p in s.llm_tagger.presets if p.id == "style_json")
    assert style.base_url == "https://api.openai.com/v1"
    assert style.api_key == "sk-xxx"
    assert style.model == "gpt-4o-mini"
    assert style.endpoint == "chat_completions"
    assert style.temperature == pytest.approx(0.3)
    assert style.max_tokens == 800
    assert style.concurrency == 1
    assert style.requests_per_second == pytest.approx(0.0)
    assert style.max_requests_per_minute == 0
    joy = next(p for p in s.llm_tagger.presets if p.id == "joycaption")
    assert joy.base_url == "http://my-vllm:9000/v1"
    assert joy.model == "my-custom-joycaption"
    user_joy = next(p for p in s.llm_tagger.presets if p.id == "user_joycaption")
    assert user_joy.messages[0].type == "text"
    assert user_joy.messages[0].role == "system"
    assert user_joy.messages[0].content == "My custom prompt"
    assert user_joy.messages[-1].type == "image"
    assert user_joy.output_format == "text"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_get_dot_path(secrets_file: Path) -> None:
    secrets.update({"wd14": {"threshold_general": 0.5}})
    assert secrets.get("wd14.threshold_general") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_get_secrets_endpoint(client: TestClient) -> None:
    resp = client.get("/api/secrets")
    assert resp.status_code == 200
    body = resp.json()
    assert "gelbooru" in body
    assert "wd14" in body
    assert body["gelbooru"]["api_key"] == ""


def test_put_secrets_round_trip(client: TestClient, secrets_file: Path) -> None:
    resp = client.put(
        "/api/secrets",
        json={"gelbooru": {"user_id": "alice", "api_key": "k"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["gelbooru"]["user_id"] == "alice"
    assert body["gelbooru"]["api_key"] == secrets.MASK

    on_disk = json.loads(secrets_file.read_text(encoding="utf-8"))
    assert on_disk["gelbooru"]["api_key"] == "k"


def test_put_secrets_mask_keeps_value(client: TestClient) -> None:
    client.put("/api/secrets", json={"gelbooru": {"api_key": "first"}})
    client.put(
        "/api/secrets",
        json={"gelbooru": {"api_key": secrets.MASK, "user_id": "alice"}},
    )
    s = secrets.load()
    assert s.gelbooru.api_key == "first"
    assert s.gelbooru.user_id == "alice"


def test_has_gelbooru_credentials(secrets_file: Path) -> None:
    assert secrets.has_gelbooru_credentials() is False
    secrets.update({"gelbooru": {"user_id": "u", "api_key": "k"}})
    assert secrets.has_gelbooru_credentials() is True


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_system_defaults_update_channel_stable(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.system.update_channel == "stable"
    assert s.system.show_dev_channel is False


def test_system_update_channel_round_trip(secrets_file: Path) -> None:
    secrets.update({"system": {"update_channel": "dev"}})
    assert secrets.load().system.update_channel == "dev"
    secrets.update({"system": {"update_channel": "stable"}})
    assert secrets.load().system.update_channel == "stable"


def test_system_legacy_file_without_system_field(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({"gelbooru": {"user_id": "alice"}}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.system.update_channel == "stable"
    assert s.gelbooru.user_id == "alice"


def test_system_update_channel_in_masked_dict(secrets_file: Path) -> None:
    secrets.update({"system": {"update_channel": "dev"}})
    masked = secrets.to_masked_dict(secrets.load())
    assert masked["system"]["update_channel"] == "dev"


def test_system_show_dev_channel_migrated_to_update_channel(
    secrets_file: Path,
) -> None:
    secrets_file.write_text(
        json.dumps({"system": {"show_dev_channel": True}}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.system.update_channel == "dev"


def test_system_show_dev_channel_migration_does_not_overwrite_explicit_pref(
    secrets_file: Path,
) -> None:
    secrets_file.write_text(
        json.dumps({"system": {"show_dev_channel": True, "update_channel": "stable"}}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.system.update_channel == "stable"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_wandb_legacy_flat_schema_migration(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({
            "wandb": {
                "enabled": True,
                "api_key": "legacy-key",
                "project": "my-proj",
                "entity": "team",
                "base_url": "https://wandb.example",
                "mode": "offline",
                "log_samples": False,
                "sample_max_side": 768,
                "upload_model": True,
                "upload_model_policy": "all",
            }
        }),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.wandb.enabled is True
    assert s.wandb.current_preset == "default"
    assert len(s.wandb.presets) == 1
    wb = s.wandb.active
    assert wb.api_key == "legacy-key"
    assert wb.project == "my-proj"
    assert wb.entity == "team"
    assert wb.base_url == "https://wandb.example"
    assert wb.mode == "offline"
    assert wb.log_samples is False
    assert wb.sample_max_side == 768
    assert wb.upload_model is True
    assert wb.upload_model_policy == "all"


def test_wandb_preset_mask_roundtrip_keeps_real_key(secrets_file: Path) -> None:
    secrets.update({"wandb": {"presets": [{"id": "default", "api_key": "real-key"}]}})
    secrets.update({
        "wandb": {
            "current_preset": "default",
            "presets": [
                {"id": "default", "api_key": secrets.MASK, "project": "renamed"}
            ],
        }
    })
    s = secrets.load()
    assert s.wandb.active.api_key == "real-key"
    assert s.wandb.active.project == "renamed"


def test_wandb_get_preset_returns_real_key_for_export(secrets_file: Path) -> None:
    secrets.update({"wandb": {"presets": [{"id": "default", "api_key": "real-key"}]}})
    preset = secrets.get_wandb_preset("default")
    assert preset is not None
    assert preset.api_key == "real-key"
    assert secrets.get_wandb_preset("nonexistent") is None


def test_wandb_import_preset_appends_and_selects(secrets_file: Path) -> None:
    new, preset = secrets.import_wandb_preset(
        {"label": "Team B", "entity": "b", "api_key": "k2", "mode": "offline"}
    )
    assert preset.entity == "b"
    assert preset.api_key == "k2"
    assert new.wandb.current_preset == preset.id
    assert any(p.id == preset.id for p in secrets.load().wandb.presets)


def test_wandb_import_preset_unwraps_wrapper_and_uniquifies_id(secrets_file: Path) -> None:
    new, preset = secrets.import_wandb_preset({
        "kind": "anima-wandb-preset",
        "version": 1,
        "preset": {"label": "default", "api_key": secrets.MASK, "mode": "offline"},
    })
    assert preset.api_key == ""
    assert preset.mode == "offline"
    assert preset.id == "default_2"
    assert new.wandb.current_preset == "default_2"


def test_wandb_import_preset_rejects_non_mapping(secrets_file: Path) -> None:
    with pytest.raises(ValueError):
        secrets.import_wandb_preset(["not", "a", "dict"])


def test_wandb_current_preset_falls_back_when_missing(secrets_file: Path) -> None:
    secrets.update({
        "wandb": {
            "presets": [
                {"id": "default"},
                {"id": "team_b", "label": "Team B", "entity": "b"},
            ],
            "current_preset": "nonexistent",
        }
    })
    s = secrets.load()
    assert s.wandb.current_preset == "default"
    assert [p.id for p in s.wandb.presets] == ["default", "team_b"]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_update_new_style_selected_not_clobbered_by_stale_compat(secrets_file: Path) -> None:
    secrets.update({"models": {"selected": {"anima": "1.0", "krea2": "raw"}}})
    updated = secrets.update({"models": {"selected": {"anima": "preview2"}}})
    assert updated.models.selected["anima"] == "preview2"
    assert updated.models.selected["krea2"] == "raw"
    assert updated.models.selected_anima == "preview2"


def test_update_incoming_legacy_key_still_wins(secrets_file: Path) -> None:
    secrets.update({"models": {"selected": {"anima": "1.0"}}})
    updated = secrets.update({"models": {"selected_anima": "preview3-base"}})
    assert updated.models.selected["anima"] == "preview3-base"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_model_sources_default_empty(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.model_sources == {}


def test_legacy_wd14_model_ids_migrate_to_sources(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({"wd14": {"model_ids": [
            *secrets.DEFAULT_WD14_MODELS, "Custom/tagger-a", "Custom/tagger-b",
        ]}}),
        encoding="utf-8",
    )
    s = secrets.load()
    cands = s.model_sources["wd14"]
    assert [(c.kind, c.repo) for c in cands] == [
        ("download", "Custom/tagger-a"), ("download", "Custom/tagger-b"),
    ]
    assert list(s.wd14.model_ids) == [
        *secrets.DEFAULT_WD14_MODELS, "Custom/tagger-a", "Custom/tagger-b",
    ]


def test_legacy_models_custom_migrate_to_sources(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({"models": {"custom": {"anima": ["D:/w/a.safetensors"]}}}),
        encoding="utf-8",
    )
    s = secrets.load()
    cands = s.model_sources["anima"]
    assert [(c.kind, c.path) for c in cands] == [("local", "D:/w/a.safetensors")]
    assert s.models.custom == {"anima": ["D:/w/a.safetensors"]}
    secrets.save(s)
    on_disk = json.loads(secrets_file.read_text(encoding="utf-8"))
    assert on_disk["models"]["custom"] == {"anima": ["D:/w/a.safetensors"]}
    assert on_disk["wd14"]["model_ids"] == list(secrets.DEFAULT_WD14_MODELS)


def test_eval_custom_model_name_backfills_candidate(secrets_file: Path) -> None:
    secrets.update({"eval_metrics": {"clip_model_name": "laion/CLIP-ViT-H-14"}})
    s = secrets.load()
    cands = s.model_sources["eval_clip"]
    assert [(c.kind, c.repo) for c in cands] == [("download", "laion/CLIP-ViT-H-14")]
    assert s.eval_metrics.clip_model_name == "laion/CLIP-ViT-H-14"


def test_local_path_selected_backfills_local_candidate(secrets_file: Path) -> None:
    secrets.update({"eval_metrics": {"dino_model_name": "D:/models/dino-local"}})
    s = secrets.load()
    cands = s.model_sources["eval_dino"]
    assert [(c.kind, c.path) for c in cands] == [("local", "D:/models/dino-local")]


def test_cltagger_fork_repo_backfills_candidate_with_extra(secrets_file: Path) -> None:
    secrets.update({"cltagger": {"model_id": "someone/cl_tagger_fork"}})
    s = secrets.load()
    cands = s.model_sources["cltagger"]
    assert len(cands) == 1
    assert cands[0].kind == "download"
    assert cands[0].repo == "someone/cl_tagger_fork"
    assert cands[0].extra == {
        "model_path": "cl_tagger_1_02/model.onnx",
        "tag_mapping_path": "cl_tagger_1_02/tag_mapping.json",
    }


def test_model_sources_update_removal_not_resurrected(secrets_file: Path) -> None:
    secrets.update({"model_sources": {"wd14": [
        {"kind": "download", "repo": "Custom/tagger-a"},
        {"kind": "download", "repo": "Custom/tagger-b"},
    ]}})
    s = secrets.update({"model_sources": {"wd14": [
        {"kind": "download", "repo": "Custom/tagger-b"},
    ]}})
    assert [c.repo for c in s.model_sources["wd14"]] == ["Custom/tagger-b"]
    assert "Custom/tagger-a" not in s.wd14.model_ids
    s2 = secrets.load()
    assert [c.repo for c in s2.model_sources["wd14"]] == ["Custom/tagger-b"]


def test_model_sources_local_candidates_survive_legacy_model_ids_put(
    secrets_file: Path,
) -> None:
    secrets.update({"model_sources": {"wd14": [
        {"kind": "local", "path": "D:/models/wd14-local"},
        {"kind": "download", "repo": "Custom/tagger-a"},
    ]}})
    s = secrets.update({"wd14": {"model_ids": list(secrets.DEFAULT_WD14_MODELS)}})
    kinds = [(c.kind, c.repo or c.path) for c in s.model_sources["wd14"]]
    assert ("local", "D:/models/wd14-local") in kinds
    assert all(c.repo != "Custom/tagger-a" for c in s.model_sources["wd14"])


def test_model_sources_round_trip_persistence(secrets_file: Path) -> None:
    secrets.update({"model_sources": {"upscaler": [
        {"kind": "download", "repo": "Kim2091/UltraSharp", "filename": "4x-UltraSharp.pth"},
        {"kind": "local", "path": "D:/up/x.pth"},
    ]}})
    s = secrets.load()
    cands = s.model_sources["upscaler"]
    assert cands[0].kind == "download"
    assert cands[0].filename == "4x-UltraSharp.pth"
    assert cands[1].kind == "local"
    assert cands[1].path == "D:/up/x.pth"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_ram_guard_defaults_off(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.training.ram_guard is False
    assert s.generate.ram_guard is False
    secrets_file.write_text(
        json.dumps({"gelbooru": {"user_id": "alice"}}), encoding="utf-8"
    )
    s = secrets.load()
    assert s.training.ram_guard is False
    assert s.generate.ram_guard is False


def test_training_ram_guard_round_trip(secrets_file: Path) -> None:
    secrets.update({"training": {"ram_guard": True}})
    assert secrets.load().training.ram_guard is True
    assert secrets.load().training.ram_guard is True
    secrets.update({"training": {"ram_guard": False}})
    assert secrets.load().training.ram_guard is False


def test_ram_guard_legacy_true_discarded_once(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({
            "generate": {"ram_guard": True},
            "training": {"ram_guard": True},
        }),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.generate.ram_guard is False
    assert s.training.ram_guard is False
    assert s.system.ram_guard_default_off is True


def test_ram_guard_kept_when_sentinel_present(secrets_file: Path) -> None:
    secrets_file.write_text(
        json.dumps({
            "generate": {"ram_guard": True},
            "training": {"ram_guard": True},
            "system": {"ram_guard_default_off": True},
        }),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.generate.ram_guard is True
    assert s.training.ram_guard is True
