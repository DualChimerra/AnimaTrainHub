
from __future__ import annotations

import pytest

from studio.domain.family_switch import switch_family
from studio.schema import TrainingConfig
from studio.services.models import default_paths_for_new_version


@pytest.fixture(autouse=True)
def _fixed_selected(monkeypatch):
    from studio import secrets

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets(models={
        "selected": {"anima": "1.0", "krea2": "raw"},
    }))


def _paths(family: str) -> dict[str, str]:
    return default_paths_for_new_version(family=family)


def test_switch_anima_to_krea2_rewrites_paths_and_flavor():
    cfg = TrainingConfig(navit_packing=True, lora_rank=64).model_dump(mode="python")
    new, changes = switch_family(cfg, "krea2", _paths("krea2"))

    assert new["model_family"] == "krea2"
    assert new["transformer_path"].endswith("krea2-raw-bf16.safetensors")
    assert new["t5_tokenizer_path"] == ""
    assert new["sample_sampler_name"] == "euler"
    assert new["sample_scheduler"] == "simple"
    assert new["timestep_sampling"] == "krea2_shift"
    assert new["shuffle_caption"] is False
    assert new["navit_packing"] is False
    assert new["lora_rank"] == 64
    TrainingConfig(**new)
    changed_fields = {c["field"] for c in changes}
    assert {"model_family", "transformer_path", "text_encoder_path",
            "t5_tokenizer_path", "sample_sampler_name", "navit_packing",
            } <= changed_fields


def test_switch_krea2_to_anima_restores_schema_defaults():
    cfg = TrainingConfig(model_family="krea2").model_dump(mode="python")
    cfg["t5_tokenizer_path"] = ""
    new, changes = switch_family(cfg, "anima", _paths("anima"))

    assert new["model_family"] == "anima"
    assert new["sample_sampler_name"] == "er_sde"
    assert new["sample_scheduler"] == "simple"
    assert new["timestep_sampling"] == "logit_normal"
    assert (new["sample_infer_steps"], new["sample_cfg_scale"]) == (25, 4.0)
    assert new["t5_tokenizer_path"].endswith("t5_tokenizer")
    TrainingConfig(**new)
    assert any(c["field"] == "t5_tokenizer_path" for c in changes)


def test_switch_same_family_reports_no_path_drift():
    cfg = TrainingConfig().model_dump(mode="python")
    cfg.update(_paths("anima"))
    new, changes = switch_family(cfg, "anima", _paths("anima"))
    assert changes == []
    assert new == cfg


def test_switch_paths_slash_style_is_not_a_change():
    cfg = TrainingConfig().model_dump(mode="python")
    cfg.update({k: str(v).replace("\\", "/") for k, v in _paths("anima").items()})
    new, changes = switch_family(cfg, "anima", _paths("anima"))
    assert changes == []
    krea2_new, _ = switch_family(cfg, "krea2", _paths("krea2"))
    assert "\\" not in krea2_new["transformer_path"]


def test_switch_partial_config_fills_missing():
    new, changes = switch_family(
        {"model_family": "anima", "lora_rank": 32}, "krea2", _paths("krea2"))
    assert new["sample_sampler_name"] == "euler"
    assert new["lora_rank"] == 32
    by_field = {c["field"]: c for c in changes}
    assert by_field["sample_sampler_name"]["from"] is None


def test_endpoint_switches_and_rejects_unknown():
    from fastapi.testclient import TestClient
    from studio.server import app

    client = TestClient(app)
    cfg = TrainingConfig().model_dump(mode="json")
    r = client.post("/api/models/family-switch",
                    json={"target": "krea2", "config": cfg})
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["model_family"] == "krea2"
    assert body["config"]["sample_sampler_name"] == "euler"
    assert any(c["field"] == "transformer_path" for c in body["changes"])

    r = client.post("/api/models/family-switch",
                    json={"target": "no-such", "config": cfg})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "model.family_invalid"
