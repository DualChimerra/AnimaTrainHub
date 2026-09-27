"""Krea 2 family asset manifest.

The Raw / Turbo main weights use the single-file checkpoint provided by Krea's
official repos; the text encoder uses the full Qwen3-VL-4B-Instruct
transformers directory. Krea 2 shares the same Qwen-Image VAE as Anima, so no
second VAE asset or on-disk copy is created.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .... import secrets
from ..paths import custom_text_encoder_dir, models_root, safe_dir_name
from ..paths import qwen_image_vae_target, resolve_vae_path

KREA2_VARIANTS: dict[str, dict[str, Any]] = {
    "raw": {
        "repo": "krea/Krea-2-Raw",
        "subpath": "raw.safetensors",
        "ms_repo": "Comfy-Org/Krea-2",
        "ms_subpath": "diffusion_models/krea2_raw_bf16.safetensors",
        "filename": "krea2-raw-bf16.safetensors",
        "purpose": "training",
        "size_estimate": 26_300_000_000,
    },
    # fp8_scaled Raw from Comfy-Org's official quantization pipeline: dual
    # purpose, training (fp8_base, weight VRAM 25.6->13.1GB, trainable on a
    # 24GB card) and inference (fp8 generation chain). purpose stays
    # "training" -- is_distilled_path judges Turbo distillation by
    # purpose=inference, and this file is Raw (not distilled), so it can't be
    # tagged inference.
    "raw_fp8": {
        "repo": "Comfy-Org/Krea-2",
        "subpath": "diffusion_models/krea2_raw_fp8_scaled.safetensors",
        "ms_repo": "Comfy-Org/Krea-2",
        "ms_subpath": "diffusion_models/krea2_raw_fp8_scaled.safetensors",
        "filename": "krea2-raw-fp8-scaled.safetensors",
        "purpose": "training",
        "size_estimate": 13_100_000_000,
    },
    "turbo": {
        "repo": "krea/Krea-2-Turbo",
        "subpath": "turbo.safetensors",
        "ms_repo": "Comfy-Org/Krea-2",
        "ms_subpath": "diffusion_models/krea2_turbo_bf16.safetensors",
        "filename": "krea2-turbo-bf16.safetensors",
        "purpose": "inference",
        "size_estimate": 26_300_000_000,
    },
    # fp8_scaled Turbo from Comfy-Org's official pipeline: the quantized
    # version of the TDM-distilled inference target (8 steps / no CFG, same
    # pipeline as raw_fp8). purpose=inference -> is_distilled_path judges it as
    # distilled, so selecting it on the test page auto-applies the distilled
    # sampling defaults.
    "turbo_fp8": {
        "repo": "Comfy-Org/Krea-2",
        "subpath": "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        "ms_repo": "Comfy-Org/Krea-2",
        "ms_subpath": "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        "filename": "krea2-turbo-fp8-scaled.safetensors",
        "purpose": "inference",
        "size_estimate": 13_100_000_000,
    },
}
LATEST_KREA2 = "raw"
KREA2_LICENSE = "Krea 2 Community License"
KREA2_LICENSE_URL = (
    "https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf"
)

QWEN3_VL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_FILES = [
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]
# The official single-file fp8_scaled TE from Comfy-Org (5.24GB vs bf16's
# 8.88GB). The weight keys use HF naming (the text side is missing a
# language_model. prefix, which the loader maps); the single file has no
# config / tokenizer etc., so those small files are downloaded from Qwen's
# official repo into the same directory (= QWEN3_VL_FILES minus the three
# weight shards/index).
QWEN3_VL_FP8_REPO = "Comfy-Org/Krea-2"
QWEN3_VL_FP8_SUBPATH = "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
QWEN3_VL_FP8_FILE = "qwen3vl_4b_fp8_scaled.safetensors"
QWEN3_VL_FP8_SMALL_FILES = [
    name for name in QWEN3_VL_FILES
    if not name.startswith("model-") and name != "model.safetensors.index.json"
]


def krea2_main_target(root: Path, variant: str) -> Path:
    if variant == "latest":
        variant = LATEST_KREA2
    try:
        filename = str(KREA2_VARIANTS[variant]["filename"])
    except KeyError:
        raise ValueError(f"unknown Krea 2 variant {variant!r}") from None
    return root / "diffusion_models" / filename


def qwen3_vl_dir(root: Path) -> Path:
    """Krea 2's text encoder directory; isolated from Anima's legacy flat directory."""
    return root / "text_encoders" / safe_dir_name(QWEN3_VL_REPO)


def qwen3_vl_fp8_dir(root: Path) -> Path:
    """Directory for the official single-file fp8_scaled TE (includes small config/tokenizer files)."""
    return root / "text_encoders" / "qwen3vl-4b-fp8"


#: TE variant -> directory resolution (bf16 first = default and UI order)
QWEN3_VL_TE_VARIANTS = ("bf16", "fp8")


def selected_te_variant() -> str:
    """The currently selected krea2 official TE variant; falls back to bf16 when
    missing/invalid (including a local directory)."""
    try:
        variant = secrets.load().models.selected_te.get("krea2")
    except Exception:
        variant = None
    return str(variant) if variant in QWEN3_VL_TE_VARIANTS else "bf16"


def qwen3_vl_dir_for(root: Path, variant: str) -> Path:
    return qwen3_vl_fp8_dir(root) if variant == "fp8" else qwen3_vl_dir(root)


def selected_text_encoder_dir(root: Path) -> Path:
    """The TE directory actually used for training / generation: a
    locally-registered directory takes priority, otherwise the official variant.

    `selected_te["krea2"]` carries both the official variant key (bf16/fp8) and
    a user-registered local encoder directory's absolute path; when the local
    directory has gone stale, falls back to official (bf16 as the ultimate
    fallback), never returning a dead path that doesn't exist.
    """
    custom = custom_text_encoder_dir("krea2")
    if custom is not None:
        return custom
    return qwen3_vl_dir_for(root, selected_te_variant())


def selected_krea2_variant() -> str:
    try:
        variant = secrets.load().models.selected.get("krea2")
    except Exception:
        variant = None
    return str(variant) if variant in KREA2_VARIANTS else LATEST_KREA2


def selected_krea2_transformer_path() -> str:
    """Returns the Krea2 official variant selected in Settings, or a valid local
    model path."""
    try:
        selected = secrets.load().models.selected.get("krea2")
    except Exception:
        selected = None
    if selected and selected not in KREA2_VARIANTS:
        path = Path(str(selected).strip()).expanduser()
        if path.is_file():
            return str(path)
    return str(krea2_main_target(models_root(), selected_krea2_variant()))


def krea2_transformer_path_for(sel: Optional[str]) -> str:
    selected = (sel or "").strip()
    if not selected:
        return selected_krea2_transformer_path()
    if selected == "latest" or selected in KREA2_VARIANTS:
        return str(krea2_main_target(models_root(), selected))
    path = Path(selected).expanduser()
    if path.is_file():
        return str(path)
    return selected_krea2_transformer_path()


def _training_variant() -> str:
    for name, info in KREA2_VARIANTS.items():
        if info["purpose"] == "training":
            return name
    return LATEST_KREA2


def _training_default_transformer_path() -> str:
    """Default main weight for a new training version: the official variant must
    have training purpose.

    Settings' `selected` is a single choice shared by training and inference --
    when the user has selected Turbo on the Generate page (purpose=inference, the
    TDM-distilled inference model), a new training version doesn't silently
    follow it, and falls back to the training variant (Raw) instead. A
    user-registered local custom path (community finetune etc.) carries no
    purpose metadata, so the user's choice is respected with no whitelist;
    explicitly passing base_model (including an explicit turbo selection) is
    likewise respected and doesn't go through this function.
    """
    try:
        selected = secrets.load().models.selected.get("krea2")
    except Exception:
        selected = None
    if selected and selected not in KREA2_VARIANTS:
        path = Path(str(selected).strip()).expanduser()
        if path.is_file():
            return str(path)
    variant = selected if selected in KREA2_VARIANTS else LATEST_KREA2
    if KREA2_VARIANTS[variant]["purpose"] != "training":
        variant = _training_variant()
    return str(krea2_main_target(models_root(), variant))


def default_paths_for_new_version(base_model: Optional[str] = None) -> dict[str, str]:
    """Returns the local asset paths a new Krea 2 version should use."""
    root = models_root()
    transformer = (
        krea2_transformer_path_for(base_model)
        if (base_model or "").strip()
        else _training_default_transformer_path()
    )
    return {
        "transformer_path": transformer,
        # VAE follows the global selection (local custom takes priority, falls back to official when stale).
        "vae_path": resolve_vae_path(root),
        # TE follows the selected value (local registered directory / bf16
        # directory / official fp8 single-file directory); training and test
        # generation share this default. fp8 training gets automatically
        # distinguished via the text-cache fingerprint (-tefp8).
        "text_encoder_path": str(selected_text_encoder_dir(root)),
        "t5_tokenizer_path": "",
    }


def is_distilled_path(path: str) -> bool:
    """Whether the transformer path is the official Turbo (TDM-distilled inference)
    variant.

    Turbo and Raw have identical structure (430 keys, same shapes), so a loader
    fingerprint physically cannot tell them apart -- this can only be decided by
    matching the catalog variant's filename. User-registered custom weights
    carry no purpose metadata, so they're always treated as non-distilled (A1:
    no whitelist, sampling parameters stay under user control).
    """
    if not path:
        return False
    name = Path(str(path)).name
    return any(
        info["purpose"] == "inference" and info["filename"] == name
        for info in KREA2_VARIANTS.values()
    )


def _file_status(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {"exists": True, "size": stat.st_size, "mtime": stat.st_mtime}
    except OSError:
        return {"exists": False, "size": 0, "mtime": 0.0}


def text_encoder_presets(root: Path) -> list[dict[str, Any]]:
    """This family's official TE candidates (the preset rows for the catalog's
    `krea2_te` domain).

    value has the same semantics as `selected_te["krea2"]`: an official variant
    key (bf16/fp8). User-registered local encoder directories are listed by
    absolute path (assembled uniformly on the catalog side).
    """
    bf16_dir = qwen3_vl_dir(root)
    fp8_dir = qwen3_vl_fp8_dir(root)
    return [
        {
            "value": "bf16",
            "label": QWEN3_VL_REPO,
            "description": str(bf16_dir),
            "download_id": "krea2_text_encoder",
            "status_key": "krea2_text_encoder",
            "path": str(bf16_dir),
            "files": [
                {"name": f, **_file_status(bf16_dir / f)}
                for f in QWEN3_VL_FILES
            ],
        },
        {
            "value": "fp8",
            "label": f"{QWEN3_VL_REPO} fp8",
            "description": str(fp8_dir),
            "download_id": "krea2_text_encoder_fp8",
            "status_key": "krea2_text_encoder_fp8",
            "path": str(fp8_dir),
            "files": [
                {"name": f, **_file_status(fp8_dir / f)}
                for f in [QWEN3_VL_FP8_FILE, *QWEN3_VL_FP8_SMALL_FILES]
            ],
        },
    ]


def catalog_sections(root: Path, models_cfg: Any) -> dict[str, Any]:
    variants = []
    for name, info in KREA2_VARIANTS.items():
        target = krea2_main_target(root, name)
        variants.append({
            "variant": name,
            "is_latest": name == LATEST_KREA2,
            "repo": info["repo"],
            "purpose": info["purpose"],
            "size_estimate": info["size_estimate"],
            "target_path": str(target),
            **_file_status(target),
        })

    custom_models = []
    for registered_path in models_cfg.custom.get("krea2", []):
        target = Path(str(registered_path)).expanduser()
        custom_models.append({
            "path": registered_path,
            "name": target.name,
            **_file_status(target),
        })

    text_dir = qwen3_vl_dir(root)
    fp8_dir = qwen3_vl_fp8_dir(root)
    # Selected TE: an official variant key is passed through as-is; a
    # local-registered directory is echoed back as its absolute path (the
    # frontend uses this to show "Custom" and disable the official radio);
    # any other invalid value normalizes to bf16.
    te_selected = str(
        (getattr(models_cfg, "selected_te", None) or {}).get("krea2") or "")
    if te_selected not in QWEN3_VL_TE_VARIANTS and not secrets.is_abs_path(
        te_selected
    ):
        te_selected = "bf16"
    return {
        "krea2_main": {
            "id": "krea2_main",
            "name": "Krea 2 main model",
            "description": (
                "Raw training base model / Turbo inference base model, each "
                "available in bf16 (26.3 GB) and official fp8 (13.1 GB, half "
                "the weight VRAM) versions"
            ),
            "repo": "krea/Krea-2-{Raw,Turbo}",
            "variants": variants,
            "custom": custom_models,
            "selected": models_cfg.selected.get("krea2") or LATEST_KREA2,
            "latest": LATEST_KREA2,
            "license": KREA2_LICENSE,
            "license_url": KREA2_LICENSE_URL,
        },
        "krea2_text_encoder": {
            "id": "krea2_text_encoder",
            "name": "Krea 2 - Qwen3-VL-4B-Instruct",
            "description": "Natural-language text encoder (approx. 8.89 GB)",
            "repo": QWEN3_VL_REPO,
            "target_dir": str(text_dir),
            # The selected TE variant (bf16/fp8) -- both the frontend's TE card
            # radio and the test page's TE dropdown read their default from here
            "selected": te_selected,
            "files": [
                {"name": filename, **_file_status(text_dir / filename)}
                for filename in QWEN3_VL_FILES
            ],
        },
        "krea2_text_encoder_fp8": {
            "id": "krea2_text_encoder_fp8",
            "name": "Krea 2 - Qwen3-VL fp8",
            "description": "Official fp8-quantized text encoder (approx. 5.24 GB, optional for test generation)",
            "repo": QWEN3_VL_FP8_REPO,
            "target_dir": str(fp8_dir),
            "files": [
                {"name": filename, **_file_status(fp8_dir / filename)}
                for filename in [QWEN3_VL_FP8_FILE, *QWEN3_VL_FP8_SMALL_FILES]
            ],
        },
    }


class _Krea2Assets:
    family_id = "krea2"
    display_name = "Krea 2"
    #: Fallback target for `selected` when unregistering a custom path (the latest official variant key)
    latest = LATEST_KREA2

    default_paths_for_new_version = staticmethod(default_paths_for_new_version)
    transformer_path_for = staticmethod(krea2_transformer_path_for)
    selected_variant = staticmethod(selected_krea2_variant)
    catalog_sections = staticmethod(catalog_sections)
    text_encoder_presets = staticmethod(text_encoder_presets)
    is_distilled_path = staticmethod(is_distilled_path)


ASSETS = _Krea2Assets()
