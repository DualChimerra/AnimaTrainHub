"""DOP needs a trigger word; queueing a version fills it in or fails early.

Covers ``_ensure_dop_trigger`` in the training router: before this, nothing in
the UI wrote ``versions.trigger_word`` any more, so every DOP run died at
startup with a traceback.
"""
from __future__ import annotations

import contextlib

import pytest

from studio.api.routers.projects import training as training_router
from studio.domain.errors import ValidationError


PROJECT = {"id": 1, "slug": "demo"}


@pytest.fixture
def saved(monkeypatch, tmp_path):
    calls: list[dict] = []

    def update_version(_conn, vid, **fields):
        calls.append({"id": vid, **fields})
        return {"id": vid, "label": "v1", "project_id": 1, **fields}

    monkeypatch.setattr(training_router.versions, "update_version", update_version)
    monkeypatch.setattr(training_router.versions, "version_dir", lambda *_a: tmp_path)
    monkeypatch.setattr(training_router.db, "connection_for", lambda: contextlib.nullcontext(None))
    return calls, tmp_path


def _caption(root, name, text):
    (root / "train" / "1_data").mkdir(parents=True, exist_ok=True)
    (root / "train" / "1_data" / name).write_text(text, encoding="utf-8")


def test_dop_off_is_untouched(saved):
    calls, _ = saved
    ver = {"id": 7, "label": "v1", "trigger_word": ""}
    assert training_router._ensure_dop_trigger(PROJECT, ver, {"dop_enabled": False}) is ver
    assert calls == []


def test_existing_version_trigger_is_kept(saved):
    calls, _ = saved
    ver = {"id": 7, "label": "v1", "trigger_word": "@mine"}
    assert training_router._ensure_dop_trigger(PROJECT, ver, {"dop_enabled": True}) is ver
    assert calls == []


def test_trigger_typed_into_the_yaml_is_rescued(saved):
    """The enqueue write forces the version's value into the yaml; without this
    a trigger written by hand into config.yaml would be wiped."""
    calls, _ = saved
    ver = {"id": 7, "label": "v1", "trigger_word": ""}
    out = training_router._ensure_dop_trigger(
        PROJECT, ver, {"dop_enabled": True, "trigger_word": "@fromyaml"},
    )
    assert out["trigger_word"] == "@fromyaml"
    assert calls == [{"id": 7, "trigger_word": "@fromyaml"}]


def test_trigger_is_read_from_the_captions(saved):
    calls, root = saved
    for i in range(4):
        _caption(root, f"{i}.txt", f"@mystyle, 1girl, tag{i}")
    ver = {"id": 7, "label": "v1", "trigger_word": ""}
    out = training_router._ensure_dop_trigger(PROJECT, ver, {"dop_enabled": True})
    assert out["trigger_word"] == "@mystyle"


def test_no_trigger_anywhere_fails_before_queueing(saved):
    calls, root = saved
    for i in range(4):
        _caption(root, f"{i}.txt", f"1girl, solo, tag{i}")
    ver = {"id": 7, "label": "v1", "trigger_word": ""}
    with pytest.raises(ValidationError) as exc:
        training_router._ensure_dop_trigger(PROJECT, ver, {"dop_enabled": True})
    assert "trigger word" in str(exc.value)
    assert calls == []
