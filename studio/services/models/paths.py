"""Model path constants + local path resolution (the 1st piece of PR-3.8's
4-way split of the 1068-line model_downloader).

Only answers "where is the model locally": models_root / safe_dir_name +
**tool models** (WD14 / CLTagger / upscalers / eval / TAEFlux -- these are not
model families and never go into families/). Model family assets (Anima weight
manifest / target / selected-value resolution) live in families/<fam>.py
(multi-model PR-4). Does no downloading and doesn't read endpoints / mirrors
(those live in sources.py).

Note: download_* functions like `download_taeflux` have all been moved to
downloader.py; only "is it ready" queries like `taeflux_dir` / `taeflux_available`
stay here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from ... import secrets
from ...paths import REPO_ROOT

# ---------------------------------------------------------------------------
# Model manifest constants (edit here when a new version is released)
# ---------------------------------------------------------------------------

# TAEFlux: a 1.6MB tiny autoencoder for Flux/Anima, used by the daemon for
# intermediate-step previews. Loaded with diffusers.AutoencoderTiny.from_pretrained
# -> needs both the config.json and safetensors files.
TAEFLUX_REPO = "madebyollin/taef1"
TAEFLUX_FILES = [
    "diffusion_pytorch_model.safetensors",
    "config.json",
]

CLTAGGER_REPO = "cella110n/cl_tagger"
CLTAGGER_V2_REPO = "cella110n/cl_tagger_v2"

# CLTagger presets. v1 lives under cella110n/cl_tagger's version subdirectory;
# v2 is a separate gated repo, but its files are still under a version
# subdirectory. Add a new line here when a new version appears, and the UI
# automatically exposes it as a radio option.
#
# Each variant explicitly declares extra_files (files that need to be
# downloaded/validated alongside model_path / tag_mapping_path), instead of
# relying on the "v2 always has a same-named .data" heuristic:
#   - v2's onnx weights live in an external sidecar model.onnx.data (2GB+);
#     missing it makes onnxruntime blow up opaquely only when it tries to load
#     external data -> it must be included;
#   - model_metadata.json is downloaded alongside as the readiness signal for
#     "is the download complete".
# If a single-file (no .data) v2 variant shows up in the future, just leave its
# extra_files empty -- it won't mistakenly require .data.
CLTAGGER_VERSIONS: dict[str, dict[str, Any]] = {
    "cl_tagger_1_02": {
        "model_id": CLTAGGER_REPO,
        "model_path": "cl_tagger_1_02/model.onnx",
        "tag_mapping_path": "cl_tagger_1_02/tag_mapping.json",
        "extra_files": [],
        "description": "CLTagger 1.02 ONNX",
    },
    "cl_tagger_v2_v2_01a": {
        "model_id": CLTAGGER_V2_REPO,
        "model_path": "v2_01a/model.onnx",
        "tag_mapping_path": "v2_01a/model_vocabulary.json",
        "extra_files": [
            "v2_01a/model.onnx.data",
            "v2_01a/model_metadata.json",
        ],
        "description": "CL Tagger v2 provisional SigLIP2 ONNX",
    },
}


def cltagger_preset_for_paths(
    model_path: str, tag_mapping_path: str
) -> Optional[dict[str, Any]]:
    """Look up the matching preset by (model_path, tag_mapping_path); a custom
    path returns None.

    Each v1/v2 model_path + tag_mapping_path pair is unique, so it's enough to
    locate the preset (no need for model_id).
    """
    norm_model = model_path.replace("\\", "/")
    norm_mapping = tag_mapping_path.replace("\\", "/")
    for preset in CLTAGGER_VERSIONS.values():
        if (
            preset["model_path"] == norm_model
            and preset["tag_mapping_path"] == norm_mapping
        ):
            return preset
    return None


def cltagger_canonical_file_paths(
    model_id: str,
    model_path: str,
    tag_mapping_path: str,
) -> tuple[str, str]:
    """Restore an early v2 "bare root path" config into its canonical path with a
    version subdirectory.

    Early v2 support used to store files under bare repo-root names
    (model.onnx / model_vocabulary.json). This looks them up in CLTAGGER_VERSIONS
    by model_id + filename to recover the versioned-subdirectory path, without
    hardcoding a version number -- so it auto-adapts when variants like v2_02
    are added later; a path that's already versioned is returned unchanged.
    """
    normalized_model = model_path.replace("\\", "/")
    normalized_mapping = tag_mapping_path.replace("\\", "/")
    if model_id != CLTAGGER_V2_REPO:
        return model_path, tag_mapping_path
    for preset in CLTAGGER_VERSIONS.values():
        if (
            preset["model_id"] == model_id
            and Path(preset["model_path"]).name == normalized_model
            and Path(preset["tag_mapping_path"]).name == normalized_mapping
        ):
            return preset["model_path"], preset["tag_mapping_path"]
    return model_path, tag_mapping_path


def is_cltagger_v2_paths(model_path: str, tag_mapping_path: str) -> bool:
    joined = f"{model_path}/{tag_mapping_path}".replace("\\", "/").lower()
    return (
        "cl_tagger_v2" in joined
        or "cl-tagger-v2" in joined
        or Path(tag_mapping_path).name.lower() == "model_vocabulary.json"
    )


def cltagger_required_files(model_path: str, tag_mapping_path: str) -> tuple[str, ...]:
    """All files needed for one variant to be fully usable (shared by download +
    readiness validation).

    Prefers the extra_files explicitly declared in the preset; for a non-preset
    (user's custom path), falls back to the "v2 onnx must carry a same-named
    .data weight" heuristic, so hand-typed paths still validate correctly.
    """
    preset = cltagger_preset_for_paths(model_path, tag_mapping_path)
    if preset is not None:
        extra = list(preset.get("extra_files", []))
    elif is_cltagger_v2_paths(model_path, tag_mapping_path):
        extra = [f"{model_path}.data"]
    else:
        extra = []
    return (model_path, *extra, tag_mapping_path)

# WD14 model's standard filenames (HF SmilingWolf/* repos always have these two at the top level).
WD14_FILES = ("model.onnx", "selected_tags.csv")

# Preprocessing upscaler preset list.
#
# label -> metadata dict:
#   filename      the filename it lands as (also one of `selected_upscaler`'s persisted keys)
#   hf            (repo_id, repo_subpath) HuggingFace source; None means no stable HF mirror exists
#   ms            (repo_id, repo_subpath) ModelScope source; None means no mirror
#   size_mb       approximate download size, shown in the frontend
#   description   a one-line usage description (shown in the frontend)
#
# Routing: download_upscaler first picks the preferred source via
# _get_download_source(), and transparently falls back to the other source when
# that one is None. A preset with both sources None is considered invalid.
#
# Source choice notes: libfishopen/upscaler on ModelScope aggregates a batch of
# A1111-era mainstream weights, with filenames + byte sizes matching the
# original HF repos; the HF side uses each upstream author's official repo
# (more authoritative).
UPSCALER_VARIANTS: dict[str, dict[str, Any]] = {
    "4x-AnimeSharp": {
        "filename": "4x-AnimeSharp.pth",
        "hf": ("Kim2091/AnimeSharp", "4x-AnimeSharp.pth"),
        "ms": ("libfishopen/upscaler", "4x-AnimeSharp.pth"),
        "size_mb": 64,
        "description": "Good for anime lineart/flat colors (Kim2091, ESRGAN-RRDB)",
    },
    "R-ESRGAN_4x+Anime6B": {
        "filename": "R-ESRGAN_4x+Anime6B.pth",
        "hf": None,  # The upstream RealESRGAN repo doesn't publish a .pth directly, so MS only for now
        "ms": ("libfishopen/upscaler", "R-ESRGAN_4x+Anime6B.pth"),
        "size_mb": 18,
        "description": "Small anime-specific model (Real-ESRGAN, A1111 default)",
    },
    "4x_foolhardy_Remacri": {
        "filename": "4x_foolhardy_Remacri.pth",
        "hf": None,
        "ms": ("libfishopen/upscaler", "4x_foolhardy_Remacri.pth"),
        "size_mb": 64,
        "description": "Realistic style (well-regarded model)",
    },
    "ESRGAN_4x": {
        "filename": "ESRGAN_4x.pth",
        "hf": None,
        "ms": ("libfishopen/upscaler", "ESRGAN_4x.pth"),
        "size_mb": 64,
        "description": "General-purpose ESRGAN baseline",
    },
}
DEFAULT_UPSCALER = "4x-AnimeSharp"
# Allowed extensions for custom/uploaded upscalers (whitelist to prevent bad paths / uploaded executables).
UPSCALER_EXTS = (".pth", ".safetensors")
# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def safe_dir_name(model_id: str) -> str:
    """Convert an HF/MS repo id into a local directory name (replaces path
    separators with _).

    A general path-sanitization utility that used to live in tagging.onnx_base
    (PR-3.8 moved it here to break a cycle: models/paths.py <- tagging/onnx_base.py
    <- models/downloader.py).
    """
    return model_id.replace("/", "_").replace("\\", "_")


def models_root() -> Path:
    """The models root directory (shared by all training / WD14 models).

    Resolution priority:
      1. `secrets.models.root` (the absolute path the user configured in Settings)
      2. the `ALS_MODELS_ROOT` environment variable (injected by cloud notebooks;
         lets models be downloaded once to a persistent disk / Google Drive and
         reused across sessions, with no need to manually configure Settings)
      3. `{REPO_ROOT}/models/` (default)

    A cloud machine's system disk is ephemeral -- without setting 1 or 2, models
    get re-downloaded on every new connection (the Anima main model + Qwen3 + T5
    etc. add up to several GB). Pointing the root at Drive gets you "download
    once, reuse forever".

    Note on directory naming: aligned with schema.py's `transformer_path`
    default (also `models/`) + WD14's `models/wd14/`; the HF repo names it
    `diffusion_models/` internally, and the local flattened layout uses the
    same subdirectory name.
    """
    try:
        cfg_root = secrets.load().models.root
    except Exception:
        cfg_root = None
    if cfg_root and str(cfg_root).strip():
        return Path(str(cfg_root).strip()).expanduser()
    env_root = os.environ.get("ALS_MODELS_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser()
    return REPO_ROOT / "models"


def qwen_image_vae_target(root: Path) -> Path:
    """Local landing spot for the Qwen-Image VAE -- a **family-agnostic shared
    asset**, not owned by any model family.

    Both Anima and Krea 2 use this exact same VAE file (same Wan2.1 latent
    space, D6/D7); it's historically been filed under Anima's name only
    because Anima came first. The download channel (which repo to fetch it
    from) is still each family's asset-manifest knowledge; this function only
    answers "where does the file go / what should the training config point at".
    """
    return root / "vae" / "qwen_image_vae.safetensors"


#: Readiness marker for a local text-encoder directory. transformers
#: directories (Qwen3 / Qwen3-VL, including the official fp8 single-file
#: version) all carry a config.json; a directory missing it is always treated
#: as "not an encoder".
TEXT_ENCODER_MARKER = "config.json"


def custom_vae_path() -> Optional[Path]:
    """The local VAE selected by the user in Settings (`secrets.models.selected_vae`).

    Unselected / a stale file (deleted, moved, points at a directory) -> None,
    and the caller falls back to the official location. Same convention as the
    main model's custom path: never returns a dead path that doesn't exist.
    """
    try:
        selected = secrets.load().models.selected_vae
    except Exception:
        return None
    text = str(selected or "").strip()
    if not text:
        return None
    p = Path(text).expanduser()
    return p if p.is_file() else None


def resolve_vae_path(root: Optional[Path] = None) -> str:
    """The VAE absolute path actually used for training / generation (local
    selection takes priority, falls back to the official location)."""
    custom = custom_vae_path()
    if custom is not None:
        return str(custom)
    return str(qwen_image_vae_target(root or models_root()))


def custom_text_encoder_dir(family: str) -> Optional[Path]:
    """A family's selected **local** text-encoder directory
    (`secrets.models.selected_te[family]`).

    This field carries both an official variant key (krea2's "bf16" / "fp8")
    and a local absolute path, so this function only recognizes the single
    shape "absolute path + directory exists + has config.json"; anything else
    (an official key / a stale path) returns None, and each family resolves it
    against its own official directory.
    """
    try:
        selected = secrets.load().models.selected_te.get(str(family))
    except Exception:
        return None
    text = str(selected or "").strip()
    if not text or not secrets.is_abs_path(text):
        return None
    p = Path(text).expanduser()
    return p if (p / TEXT_ENCODER_MARKER).is_file() else None


def taeflux_dir(root: Optional[Path] = None) -> Path:
    """TAEFlux's local directory. The daemon loads it with AutoencoderTiny.from_pretrained."""
    r = root or models_root()
    return r / "taeflux"


def taeflux_available(root: Optional[Path] = None) -> bool:
    """Counts as ready only when both files are present."""
    d = taeflux_dir(root)
    return all((d / f).exists() for f in TAEFLUX_FILES)



def wd14_target_dir(root: Path, model_id: str) -> Path:
    """A single WD14 model_id's local directory. Matches wd14_tagger's
    _resolve_model_dir path layout.

    When `model_id` is an absolute path (the unified-source candidate's local
    type), points directly at that directory.
    """
    if secrets.is_abs_path(model_id):
        return Path(model_id)
    return root / "wd14" / safe_dir_name(model_id)


def eval_model_target_dir(root: Path, kind: str, model_id: str) -> Path:
    """Local directory for a CLIP / DINO eval-metric model (kind: ``clip`` | ``dino``).

    A multi-file transformers repo; the whole directory is landed by
    snapshot_download, and from_pretrained points here at eval time -- managed
    uniformly under the project's models/ rather than ~/.cache/huggingface.
    When `model_id` is an absolute path (a local candidate), points directly at
    that directory.
    """
    if secrets.is_abs_path(model_id):
        return Path(model_id)
    return root / "eval" / kind / safe_dir_name(model_id)


def ccip_model_dir(root: Path, variant: str) -> Path:
    """Local directory for a CCIP (anime character identity) ONNX variant.

    Each variant subdirectory under deepghs/ccip_onnx contains model_feat.onnx +
    model_metrics.onnx + metrics.json; only these 3 files are selectively
    downloaded here (the full repo is 3.5GB including torch checkpoints + pngs,
    so files are picked by name). When `variant` is an absolute path (a local
    candidate), points directly at that directory.
    """
    if secrets.is_abs_path(variant):
        return Path(variant)
    return root / "eval" / "ccip" / safe_dir_name(variant)


def cltagger_target_root(root: Path, model_id: str) -> Path:
    """The local root directory for a CLTagger repo. Subdirectory layout comes from CLTAGGER_VERSIONS."""
    return root / "cltagger" / safe_dir_name(model_id)


def upscaler_dir(root: Optional[Path] = None) -> Path:
    """The upscaler weights root directory `{models_root}/upscalers/`."""
    r = root or models_root()
    return r / "upscalers"


def upscaler_target(label: str, root: Optional[Path] = None) -> Path:
    """Target path for a single upscaler's weights.

    label can be:
      - a preset key (in UPSCALER_VARIANTS) -> uses the preset's filename
      - a direct filename (with .pth/.safetensors extension) -> treated as a custom/already-uploaded model
      - an absolute path (a unified-source local candidate, a file the user
        registered via PathPicker) -> returned directly, doesn't land under
        upscalers/

    Path-traversal protection: outside of the absolute-path case, label must
    not contain `/`, `\\`, or `..`, preventing relative segments from escaping
    outside of upscalers/.
    """
    if secrets.is_abs_path(label):
        if not label.lower().endswith(UPSCALER_EXTS):
            raise ValueError(f"unknown upscaler {label!r}")
        return Path(label)
    if "/" in label or "\\" in label or ".." in label:
        raise ValueError(f"invalid upscaler label {label!r}")
    if label in UPSCALER_VARIANTS:
        fname = UPSCALER_VARIANTS[label]["filename"]
    else:
        if not label.lower().endswith(UPSCALER_EXTS):
            raise ValueError(f"unknown upscaler {label!r}")
        fname = label
    return upscaler_dir(root) / fname


def find_upscaler(label: str, root: Optional[Path] = None) -> Optional[Path]:
    """Returns the local path if downloaded, None if not."""
    target = upscaler_target(label, root)
    return target if target.exists() else None


def selected_upscaler() -> str:
    """Reads `secrets.models.selected_upscaler`, falling back to DEFAULT_UPSCALER.

    The return value can be:
      - a preset label (in UPSCALER_VARIANTS)
      - an existing custom filename (with extension)
    Falls back to DEFAULT_UPSCALER (the 4x-AnimeSharp preset) when neither matches.
    """
    try:
        v = secrets.load().models.selected_upscaler
    except Exception:
        v = None
    if not v:
        return DEFAULT_UPSCALER
    if v in UPSCALER_VARIANTS:
        return v
    # custom: scan disk to see if the file exists
    if v.lower().endswith(UPSCALER_EXTS) and (upscaler_dir() / v).exists():
        return v
    return DEFAULT_UPSCALER
