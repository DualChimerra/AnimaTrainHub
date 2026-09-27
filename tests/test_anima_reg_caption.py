"""Unit tests for anima_reg_ai's JSON caption filter + sidecar rewrite (PR #185
follow-up).

Verifies that after _rewrite_json_caption_for_prompt goes through
normalize_caption_json + single-pass filtering:
- meta.trigger is always dropped from both the reg sidecar + prompt (the base prior
  doesn't carry a LoRA handle)
- excluded_tags matches in both space/underscore forms
- documented_full / simplified shapes all collapse to the standard shape when written
  back (reg is a derived artifact)
- scalar fields containing commas are filtered per-tag against excluded
- nl natural language is preserved at the end of the prompt
- _clear_reg_dir is reused from reg_builder.clear_reg_dir (no longer duplicated locally)

Does not run on GPU, does not trigger a real anima_train import -- top-level stub.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def reg_module():
    """import runtime.anima_reg_ai once and reuse it, stubbing out anima_train's heavy deps."""
    if "anima_train" not in sys.modules or not hasattr(sys.modules["anima_train"], "sample_image"):
        at = types.ModuleType("anima_train")
        at.sample_image = lambda *a, **k: None
        at.find_diffusion_pipe_root = lambda: Path(".")
        at.resolve_path_best_effort = lambda p, bases: p
        at.load_anima_model = lambda *a, **k: None
        at.load_vae = lambda *a, **k: None
        at.load_text_encoders = lambda *a, **k: (None, None, None)
        at.enable_xformers = lambda *a, **k: None
        sys.modules["anima_train"] = at

    import importlib
    if "anima_reg_ai" in sys.modules:
        del sys.modules["anima_reg_ai"]
    return importlib.import_module("anima_reg_ai")


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cap.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# trigger never enters reg
# ---------------------------------------------------------------------------

def test_drops_meta_trigger_from_prompt_and_sidecar(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "meta": {"trigger": "aoirikko"},
        "tags": {
            "quality": ["masterpiece"],
            "appearance": ["brown hair"],
        },
    })
    prompt = reg_module._rewrite_json_caption_for_prompt(src, set())
    assert "aoirikko" not in prompt.lower()
    assert "brown hair" in prompt
    on_disk = _read(src)
    assert on_disk["meta"].get("trigger") in (None, "")


def test_drops_trigger_even_when_other_meta_kept(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "meta": {"trigger": "aoirikko", "tagger_version": "v2"},
        "tags": {"appearance": ["smile"]},
    })
    reg_module._rewrite_json_caption_for_prompt(src, set())
    on_disk = _read(src)
    assert "trigger" not in on_disk["meta"]
    assert on_disk["meta"].get("tagger_version") == "v2"


# ---------------------------------------------------------------------------
# excluded space / underscore equivalence
# ---------------------------------------------------------------------------

def test_excluded_underscore_matches_space_form(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "tags": {"appearance": ["brown hair", "blue eyes"]},
    })
    # excluded_tags is passed as underscore, matches the space form in the JSON
    excluded = {reg_module._tag_key("brown_hair")}
    prompt = reg_module._rewrite_json_caption_for_prompt(src, excluded)
    assert "brown hair" not in prompt
    assert "blue eyes" in prompt


def test_excluded_space_matches_underscore_form(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "tags": {"appearance": ["brown_hair", "blue_eyes"]},
    })
    excluded = {reg_module._tag_key("brown hair")}
    prompt = reg_module._rewrite_json_caption_for_prompt(src, excluded)
    assert "brown" not in prompt or "hair" not in prompt
    # blue_eyes gets collapsed by split_tags during normalize into a single
    # "blue_eyes" entry; use a light assertion here: the other tag is still present
    assert "blue" in prompt


# ---------------------------------------------------------------------------
# shape folding: documented_full -> standard
# ---------------------------------------------------------------------------

def test_documented_full_shape_folds_to_standard_on_disk(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "fixed": {"quality": "masterpiece, best quality", "series": "kaguya", "artist": ""},
        "character": {"name": "Misaka", "variant": "Sisters", "full": "Misaka Mikoto"},
        "ai_output": {
            "count": "1girl",
            "appearance": ["brown hair"],
            "tags": ["sitting"],
            "environment": ["indoors"],
            "nl": "She is reading a book.",
        },
        "from_path": {"appearance": ["smile"], "extra_tags": ["solo"]},
    })
    prompt = reg_module._rewrite_json_caption_for_prompt(src, set())
    on_disk = _read(src)

    # On-disk shape: standard shape, the original fixed/character/ai_output/from_path
    # top-level keys are all gone
    assert isinstance(on_disk.get("tags"), dict)
    assert "fixed" not in on_disk
    assert "ai_output" not in on_disk
    assert "from_path" not in on_disk
    assert "character" not in on_disk  # character folds into tags.character

    # prompt: character folds to full, appearance / tags merge the from_path part, nl appended at the end
    assert "misaka mikoto" in prompt.lower()
    assert "brown hair" in prompt
    assert "smile" in prompt
    assert "sitting" in prompt
    assert "solo" in prompt
    assert "She is reading a book." in prompt


def test_simplified_tags_list_top_level_shape(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "tags": ["1girl", "brown hair", "smile"],
        "meta": {"trigger": "aoirikko"},
    })
    prompt = reg_module._rewrite_json_caption_for_prompt(src, set())
    assert "aoirikko" not in prompt.lower()
    assert "1girl" in prompt
    assert "brown hair" in prompt
    on_disk = _read(src)
    assert isinstance(on_disk["tags"], dict)  # folded into the standard shape
    assert "1girl" in on_disk["tags"]["tags"]


# ---------------------------------------------------------------------------
# character / scalar fields
# ---------------------------------------------------------------------------

def test_character_dict_full_name_excluded_drops_character(reg_module, tmp_path: Path) -> None:
    """When character is dict-shaped it folds into a full string; excluded matching
    the full name clears it entirely."""
    src = _write(tmp_path, {
        "fixed": {"quality": "best quality"},
        "character": {"name": "Misaka", "variant": "", "full": "Misaka Mikoto"},
        "ai_output": {"tags": [], "appearance": [], "environment": []},
    })
    excluded = {reg_module._tag_key("Misaka Mikoto")}
    prompt = reg_module._rewrite_json_caption_for_prompt(src, excluded)
    assert "misaka" not in prompt.lower()
    assert "best quality" in prompt


def test_scalar_field_with_comma_per_tag_exclude(reg_module, tmp_path: Path) -> None:
    """In count="1girl, 1boy", exclude="1boy" only removes the 1boy token."""
    src = _write(tmp_path, {
        "tags": {"count": "1girl, 1boy", "appearance": []},
    })
    excluded = {reg_module._tag_key("1boy")}
    prompt = reg_module._rewrite_json_caption_for_prompt(src, excluded)
    assert "1girl" in prompt
    assert "1boy" not in prompt
    on_disk = _read(src)
    assert "1boy" not in on_disk["tags"]["count"]
    assert "1girl" in on_disk["tags"]["count"]


# ---------------------------------------------------------------------------
# nl natural language preservation
# ---------------------------------------------------------------------------

def test_nl_preserved_at_prompt_tail(reg_module, tmp_path: Path) -> None:
    src = _write(tmp_path, {
        "tags": {
            "appearance": ["brown hair"],
            "nl": "She is wearing a school uniform.",
        },
    })
    prompt = reg_module._rewrite_json_caption_for_prompt(src, set())
    assert prompt.endswith("She is wearing a school uniform.")
    assert "brown hair" in prompt


# ---------------------------------------------------------------------------
# reuse: clear_reg_dir comes from reg_builder
# ---------------------------------------------------------------------------

def test_clear_reg_dir_is_reused_from_reg_builder(reg_module) -> None:
    from studio.services.reg.builder import clear_reg_dir as upstream
    assert reg_module.clear_reg_dir is upstream
    # Also confirm there's no more local private copy defined
    assert not hasattr(reg_module, "_clear_reg_dir")


# ---------------------------------------------------------------------------
# txt path unchanged: a plain .txt caption is still written back in space-form,
# trigger is determined by the .txt content
# ---------------------------------------------------------------------------

def test_txt_caption_path_unchanged(reg_module, tmp_path: Path) -> None:
    src = tmp_path / "cap.txt"
    src.write_text("brown_hair, blue eyes, 1girl", encoding="utf-8")
    excluded = {reg_module._tag_key("1girl")}
    prompt = reg_module._rewrite_caption_for_prompt(src, excluded)
    assert "brown hair" in prompt  # underscore -> space normalization
    assert "blue eyes" in prompt
    assert "1girl" not in prompt
    on_disk = src.read_text(encoding="utf-8")
    assert on_disk == prompt


# ---------------------------------------------------------------------------
# _scan_train: mask sidecar does not count as a training image
# ---------------------------------------------------------------------------

def test_scan_train_ignores_mask_sidecar(reg_module, tmp_path: Path) -> None:
    """The {stem}.mask sidecar extension is not in IMAGE_EXTS -- the recursive scan
    naturally skips it, so masks are never counted toward the generation manifest
    (there was previously a bug: under the old masks/ directory layout, masks were
    treated as training images -> inflated generation totals, and counts wouldn't
    drop after deleting images and regenerating)."""
    train = tmp_path / "train"
    (train / "1_data").mkdir(parents=True)
    (train / "1_data" / "a.png").write_bytes(b"png")
    (train / "1_data" / "b.png").write_bytes(b"png")
    (train / "1_data" / "a.mask").write_bytes(b"png")

    entries = reg_module._scan_train(train)
    imgs = sorted(str(e["img"].relative_to(train)).replace("\\", "/") for e in entries)
    assert imgs == ["1_data/a.png", "1_data/b.png"]
