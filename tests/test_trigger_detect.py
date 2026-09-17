"""Trigger word read back from captions (DOP needs one; see services/trigger_detect.py)."""
from __future__ import annotations

import json

from studio.services import trigger_detect


def _write(folder, name, text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")


def test_handle_in_every_caption_is_suggested(tmp_path):
    for i in range(10):
        _write(tmp_path / "1_data", f"{i}.txt",
               f"@mystyle, 1girl, solo, furry female, tag{i}. A sentence about image {i}.")
    r = trigger_detect.detect(tmp_path)
    assert r["total"] == 10
    assert r["suggested"] == "@mystyle"
    assert trigger_detect.confident_trigger(tmp_path) == "@mystyle"


def test_common_tags_are_never_suggested(tmp_path):
    for i in range(5):
        _write(tmp_path, f"{i}.txt", f"1girl, solo, safe, tag{i}")
    assert trigger_detect.detect(tmp_path)["suggested"] is None
    assert trigger_detect.confident_trigger(tmp_path) is None


def test_a_word_missing_from_some_captions_is_not_confident(tmp_path):
    for i in range(10):
        lead = "mystyle, " if i < 9 else ""
        _write(tmp_path, f"{i}.txt", f"{lead}1girl, tag{i}")
    r = trigger_detect.detect(tmp_path)
    assert r["suggested"] == "mystyle"  # 90% is still worth suggesting...
    assert trigger_detect.confident_trigger(tmp_path) is None  # ...but not auto-filling


def test_studio_json_captions_are_read(tmp_path):
    for i in range(3):
        _write(tmp_path, f"{i}.json", json.dumps({"meta": {"trigger": "@mystyle"}, "tags": ["1girl", f"t{i}"]}))
    assert trigger_detect.confident_trigger(tmp_path) == "@mystyle"


def test_empty_or_missing_folder(tmp_path):
    assert trigger_detect.detect(tmp_path / "nope") == {"total": 0, "suggested": None, "candidates": []}
