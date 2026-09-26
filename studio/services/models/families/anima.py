"""Anima family asset manifest (multi-model PR-4, migrated function-by-function
from paths.py).

Weight repo / variant list / download targets / selected-value resolution /
default paths for a new version -- all of this is family-specific knowledge.
Tool models (WD14 etc.) and models_root stay in ..paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .... import secrets
from ..paths import (
    custom_text_encoder_dir,
    models_root,
    qwen_image_vae_target,
    resolve_vae_path,
)

ANIMA_REPO = "circlestone-labs/Anima"
# Order: newest first. `find_anima_main`'s fallback lookup iterates this dict in
# order, and `build_catalog`'s variants list for the UI reuses the same order --
# so new versions are added at the top, older ones sink down.
ANIMA_VARIANTS: dict[str, str] = {
    "1.0":           "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "preview3-base": "split_files/diffusion_models/anima-preview3-base.safetensors",
    "preview2":      "split_files/diffusion_models/anima-preview2.safetensors",
    "preview":       "split_files/diffusion_models/anima-preview.safetensors",
}
LATEST_ANIMA = "1.0"
ANIMA_VAE_PATH = "split_files/vae/qwen_image_vae.safetensors"

QWEN_REPO = "Qwen/Qwen3-0.6B-Base"
# Note: Qwen3 bakes special tokens directly into tokenizer.json, so the repo has
# no `special_tokens_map.json` (older Qwen versions do -- copying that
# assumption over would 404).
QWEN_FILES = [
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "config.json",
]

T5_REPO = "google/t5-v1_1-xxl"
T5_FILES = [
    "spiece.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def anima_main_target(root: Path, variant: str) -> Path:
    if variant == "latest":
        variant = LATEST_ANIMA
    if variant not in ANIMA_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return root / "diffusion_models" / Path(ANIMA_VARIANTS[variant]).name


def qwen_dir(root: Path) -> Path:
    return root / "text_encoders"


def t5_tokenizer_dir(root: Path) -> Path:
    return root / "t5_tokenizer"


def selected_text_encoder_dir(root: Path) -> Path:
    """The text encoder directory Anima actually uses: the local directory
    selected in Settings takes priority.

    Anima has no official TE variants (only a single Qwen3-0.6B), so
    `selected_te["anima"]` is either empty = the official directory, or the
    absolute path of a user-registered local encoder directory.
    """
    custom = custom_text_encoder_dir("anima")
    return custom if custom is not None else qwen_dir(root)


def find_anima_main(root: Optional[Path] = None) -> Optional[Path]:
    """Find the first main model present on disk, in ANIMA_VARIANTS priority
    order (latest first).

    This is only a fallback (for bare CLI use / a missing yaml); when Studio
    creates a version it prefers `selected_anima_path()` to get the variant the
    user selected in settings.
    """
    r = root or models_root()
    order = [LATEST_ANIMA] + [v for v in ANIMA_VARIANTS if v != LATEST_ANIMA]
    for v in order:
        target = anima_main_target(r, v)
        if target.exists():
            return target
    return None


def selected_anima_variant() -> str:
    """Read `secrets.models.selected_anima`, falling back to LATEST_ANIMA."""
    try:
        v = secrets.load().models.selected_anima
    except Exception:
        v = None
    if v and v in ANIMA_VARIANTS:
        return v
    return LATEST_ANIMA


def selected_anima_transformer_path() -> str:
    """The selected main model's absolute transformer path (shared by new-training
    defaults and test generation).

    If `selected_anima` is an official variant key -> compute the path via
    `anima_main_target`; if it's a user-registered local custom path (not in
    ANIMA_VARIANTS and the file exists) -> return that path directly. If the
    custom path has gone stale (deleted / moved), falls back to the current
    variant, so it never returns a dead path that doesn't exist.
    """
    try:
        sel = secrets.load().models.selected_anima
    except Exception:
        sel = None
    if sel and sel not in ANIMA_VARIANTS:
        p = Path(str(sel).strip()).expanduser()
        if p.exists():
            return str(p)
    return str(anima_main_target(models_root(), selected_anima_variant()))


def anima_transformer_path_for(sel: Optional[str]) -> str:
    """Resolve an explicit main-model selection into a transformer absolute path.

    `sel` has the same semantics as `secrets.models.selected_anima`: an official
    variant key ("1.0" / "latest" etc.) or a registered local custom
    `.safetensors` absolute path. An empty value -> falls back to whatever
    Settings' `selected_anima` resolves to (`selected_anima_transformer_path`),
    i.e. prior generation / test generation follow "the base model selected in
    Settings". A stale custom path (deleted / moved) -> falls back to the
    current selection, never returning a dead path that doesn't exist.
    """
    s = (sel or "").strip()
    if not s:
        return selected_anima_transformer_path()
    if s == "latest" or s in ANIMA_VARIANTS:
        return str(anima_main_target(models_root(), s))
    p = Path(s).expanduser()
    if p.exists():
        return str(p)
    return selected_anima_transformer_path()


def default_paths_for_new_version(base_model: Optional[str] = None) -> dict[str, str]:
    """Used when Studio creates a new version: returns the **absolute path
    strings** for all 4 path fields.

    Computed from the current `secrets.models.root` and
    `secrets.models.selected_anima`. Once the user switches selected_anima in
    settings (an official variant or a registered local custom path), new
    versions created afterward automatically use the new selection; an existing
    version's yaml is left untouched (reproducibility).

    When `base_model` is non-empty, only transformer_path is overridden (prior
    generation / test generation render using the base model temporarily
    selected on the page); vae / text_encoder / t5 still follow the global setting.
    """
    root = models_root()
    return {
        "transformer_path": anima_transformer_path_for(base_model),
        # VAE / text encoder likewise follow the value selected in Settings
        # (local custom weights take priority, falling back to the official
        # location when stale) -- same convention as transformer.
        "vae_path": resolve_vae_path(root),
        "text_encoder_path": str(selected_text_encoder_dir(root)),
        "t5_tokenizer_path": str(t5_tokenizer_dir(root)),
    }


# ---------------------------------------------------------------------------
# catalog sections (migrated in from catalog.py's build_catalog; output shape unchanged -- zero frontend changes)
# ---------------------------------------------------------------------------


def _file_status(p: Path) -> dict[str, Any]:
    try:
        st = p.stat()
        return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}
    except OSError:
        return {"exists": False, "size": 0, "mtime": 0.0}


def text_encoder_presets(root: Path) -> list[dict[str, Any]]:
    """This family's official text-encoder candidates (the preset rows for the
    catalog's `anima_te` domain).

    Anima has only a single official encoder, Qwen3-0.6B, so value uses an empty
    string to mean "the official directory" (`selected_te["anima"]` empty =
    official); user-registered local directories are listed by absolute path.
    """
    d = qwen_dir(root)
    return [{
        "value": "",
        "label": QWEN_REPO,
        "description": str(d),
        "download_id": "qwen3",
        "status_key": "qwen3",
        "path": str(d),
        "files": [{"name": f, **_file_status(d / f)} for f in QWEN_FILES],
    }]


def catalog_sections(root: Path, models_cfg: Any) -> dict[str, Any]:
    """The Anima family section of /api/models/catalog (anima_main / anima_vae / qwen3 / t5_tokenizer)."""
    anima_variants = []
    for vname, _subpath in ANIMA_VARIANTS.items():
        target = anima_main_target(root, vname)
        anima_variants.append({
            "variant": vname,
            "is_latest": vname == LATEST_ANIMA,
            "target_path": str(target),
            **_file_status(target),
        })

    custom_anima = []
    for p in models_cfg.custom_anima_paths:
        target = Path(str(p)).expanduser()
        custom_anima.append({
            "path": p,
            "name": target.name,
            **_file_status(target),
        })

    vae_target = qwen_image_vae_target(root)
    qwen_d = qwen_dir(root)
    t5_d = t5_tokenizer_dir(root)
    return {
        "anima_main": {
            "id": "anima_main",
            "name": "Anima main model",
            "description": "Cosmos transformer (~4 GB)",
            "repo": ANIMA_REPO,
            "variants": anima_variants,
            "custom": custom_anima,
            "selected": models_cfg.selected_anima,
            "latest": LATEST_ANIMA,
        },
        "anima_vae": {
            "id": "anima_vae",
            "name": "Anima VAE",
            "description": "qwen_image_vae (~250 MB)",
            "repo": ANIMA_REPO,
            "target_path": str(vae_target),
            **_file_status(vae_target),
        },
        "qwen3": {
            "id": "qwen3",
            "name": "Qwen3-0.6B-Base",
            "description": "Text encoder (~1.2 GB)",
            "repo": QWEN_REPO,
            "target_dir": str(qwen_d),
            "files": [
                {"name": f, **_file_status(qwen_d / f)} for f in QWEN_FILES
            ],
        },
        "t5_tokenizer": {
            "id": "t5_tokenizer",
            "name": "T5 tokenizer",
            "description": "3 tokenizer files including spiece.model (no weights)",
            "repo": T5_REPO,
            "target_dir": str(t5_d),
            "files": [
                {"name": f, **_file_status(t5_d / f)} for f in T5_FILES
            ],
        },
    }


class _AnimaAssets:
    """Duck-typed family asset object (registered in families/__init__.py)."""

    family_id = "anima"
    display_name = "Anima"
    #: Fallback target for `selected` when unregistering a custom path (the latest official variant key)
    latest = LATEST_ANIMA

    default_paths_for_new_version = staticmethod(default_paths_for_new_version)
    transformer_path_for = staticmethod(anima_transformer_path_for)
    selected_variant = staticmethod(selected_anima_variant)
    catalog_sections = staticmethod(catalog_sections)
    text_encoder_presets = staticmethod(text_encoder_presets)
    # Anima has no distilled inference variant
    is_distilled_path = staticmethod(lambda path: False)


ASSETS = _AnimaAssets()
