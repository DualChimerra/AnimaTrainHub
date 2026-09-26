"""Download-source routing + mirror endpoints + low-level download primitives
(the 2nd piece of PR-3.8's 4-way split).

Answers two questions:
  1. Where to download from? (_get_download_source / _resolve_endpoint / _ms_token)
  2. How does a single file land on disk? (download_flat / download_flat_ms)

Holds no model path constants (those live in paths.py), and implements no
model-specific download flow (that lives in downloader.py).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from ... import secrets

# ---------------------------------------------------------------------------
# ModelScope mirror source mapping
# ---------------------------------------------------------------------------

# ModelScope mirror path constants.
# circlestone-labs publishes to both HF and ModelScope in sync, with matching
# repo IDs; on ModelScope, the Anima repo packs the main model / VAE / text
# encoder all under split_files/, and the text encoder is a single
# safetensors file (rather than the loose-file directory Qwen3 uses on HF).
MS_ANIMA_TEXT_ENCODER_PATH = "split_files/text_encoders/qwen_3_06b_base.safetensors"
# T5 tokenizer / TAEFlux / CLTagger have no corresponding mirror on ModelScope
# yet, so they fall back to HF.
# WD14: fireicewolf mirrors the SmilingWolf series on ModelScope, with repos
# named as SmilingWolf/{name} -> fireicewolf/{name}
_MS_WD14_OWNER = "fireicewolf"
_HF_WD14_OWNER = "SmilingWolf"


def _ms_wd14_repo_id(hf_repo_id: str) -> Optional[str]:
    """Converts SmilingWolf/wd-xxx to fireicewolf/wd-xxx; returns None for other repos."""
    if hf_repo_id.startswith(_HF_WD14_OWNER + "/"):
        name = hf_repo_id[len(_HF_WD14_OWNER) + 1:]
        return f"{_MS_WD14_OWNER}/{name}"
    return None


# ModelScope mirror mapping for CLIP / DINO eval-metric models. The community
# mirror organization AI-ModelScope syncs common vision/multimodal models from
# HF, generally under matching repo names. An unmapped model returns None ->
# falls back to HuggingFace (the same graceful fallback as wd14's
# non-SmilingWolf-prefix case). The MS source is only used once the user
# actively switches the eval source to modelscope; HF is the default. Whether a
# given repo actually exists is up to ModelScope's actual state -- if missing,
# that model's download fails and the user can switch back to the HF source.
_MS_EVAL_REPO_IDS = {
    "openai/clip-vit-base-patch32": "AI-ModelScope/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14": "AI-ModelScope/clip-vit-large-patch14",
    "facebook/dinov2-small": "AI-ModelScope/dinov2-small",
    "facebook/dinov2-base": "AI-ModelScope/dinov2-base",
}


def _ms_eval_repo_id(hf_repo_id: str) -> Optional[str]:
    """ModelScope mirror id for a CLIP / DINO model; returns None when unmapped (falls back to HF)."""
    return _MS_EVAL_REPO_IDS.get(hf_repo_id)


# ---------------------------------------------------------------------------
# Synchronous download helpers
# ---------------------------------------------------------------------------


def setup_mirror(use_mirror: bool) -> None:
    """[Legacy] Sets the HF_ENDPOINT environment variable.

    Since PR-S3, the Studio UI passes secrets.huggingface.endpoint to the HF
    library per-call instead of relying on the env var (the env var is only
    read once, when the huggingface_hub module is imported).
    This function is kept only for compatibility with `tools/download_models.py`'s
    early CLI setup flow.
    """
    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # Turning the mirror off does not actively unset HF_ENDPOINT -- left for the caller to manage explicitly


def _resolve_endpoint() -> Optional[str]:
    """Decides which HF endpoint to use for this download. Priority:

    1. the `HF_ENDPOINT` environment variable (set by the CLI's setup_mirror, or
       manually exported by the user)
    2. `secrets.huggingface.endpoint` (configured in the Studio UI)
    3. None (lets huggingface_hub use the default huggingface.co)

    Called once per download, so a config change in the UI needs no server restart.
    """
    env = os.environ.get("HF_ENDPOINT", "").strip()
    if env:
        return env
    try:
        endpoint = secrets.load().huggingface.endpoint
    except Exception:  # noqa: BLE001  corrupted secrets shouldn't block a download
        return None
    return endpoint or None


def _get_download_source() -> str:
    """Returns the currently configured download source ('huggingface' or 'modelscope').

    Prefers the MODELSCOPE_SOURCE env var (used by the CLI flag); otherwise reads from secrets.
    """
    env = os.environ.get("MODELSCOPE_SOURCE", "").strip()
    if env:
        return env
    try:
        return secrets.load().download_source or "huggingface"
    except Exception:  # noqa: BLE001
        return "huggingface"


def _source_for(type_key: str) -> str:
    """The currently selected source for a download type (training / wd14 / upscaler).

    MODELSCOPE_SOURCE env still acts as a global forced override (CLI flag /
    CI); otherwise reads secrets.download_sources[type_key], falling back to
    huggingface for a missing/invalid value. Types fixed to HF (cltagger / t5 /
    taeflux) don't go through here.
    """
    env = os.environ.get("MODELSCOPE_SOURCE", "").strip().lower()
    if env in ("huggingface", "modelscope"):
        return env
    try:
        src = secrets.load().download_sources.get(type_key, "huggingface")
    except Exception:  # noqa: BLE001
        return "huggingface"
    return src if src in ("huggingface", "modelscope") else "huggingface"


def _ms_token() -> Optional[str]:
    """Reads the ModelScope token: environment variable first, then secrets."""
    env = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if env:
        return env
    try:
        t = secrets.load().modelscope.token
        return t or None
    except Exception:  # noqa: BLE001
        return None


def _hf_token() -> Optional[str]:
    """Reads the HF token: environment variable first, then secrets.huggingface.token.

    Needed to download gated / private repos (e.g. cl_tagger_v2); public repos
    can be downloaded without one.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        env = os.environ.get(var, "").strip()
        if env:
            return env
    try:
        t = secrets.load().huggingface.token
        return t or None
    except Exception:  # noqa: BLE001  corrupted secrets shouldn't block a download
        return None


def _is_gated_auth_error(exc: BaseException) -> bool:
    """Roughly determines whether a download exception stems from a gated/private
    authorization failure, so an actionable hint can be appended."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "gated" in name
        or "gated" in msg
        or "401" in msg
        or "403" in msg
        or "unauthorized" in msg
        or "private or gated" in msg
        or "authentication" in msg
    )


def download_flat_ms(
    ms_repo_id: str,
    repo_subpath: str,
    target: Path,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Downloads a single file to target using the modelscope Python API.

    `model_file_download(local_dir=target.parent)` lands the file at
    `target.parent / repo_subpath` (preserving the repo's internal path
    structure), after which the exact same rename + empty-directory-cleanup
    logic as `download_flat` moves it to target.

    Requires ``pip install modelscope``; returns False and prints a hint if not
    installed. The token is read from the MODELSCOPE_API_TOKEN env var first,
    then secrets.modelscope.token.
    """
    if target.exists():
        on_log(f"   already have {target.name}, skipping")
        return True
    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError:
        on_log("   missing modelscope (pip install modelscope)")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    token = _ms_token()
    try:
        kwargs: dict = dict(
            model_id=ms_repo_id,
            file_path=repo_subpath,
            local_dir=str(target.parent),
        )
        if token:
            kwargs["token"] = token
        model_file_download(**kwargs)
    except Exception as exc:
        on_log(f"   failed {target.name} (ModelScope): {exc}")
        return False
    # model_file_download preserves the repo's internal path; the logic below matches download_flat exactly
    src = target.parent / repo_subpath
    if src != target:
        try:
            target.unlink(missing_ok=True)
            src.rename(target)
        except OSError as exc:
            on_log(f"   rename failed {src} -> {target}: {exc}")
            return False
        parent = src.parent
        while parent != target.parent and parent.exists():
            try:
                if any(parent.iterdir()):
                    break
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    on_log(f"   done {target.name} (via ModelScope)")
    return True


def download_flat(
    repo_id: str,
    repo_subpath: str,
    target: Path,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Downloads repo_subpath from HF and lands it flat at target; returns True = ready.

    Implementation: `hf_hub_download(local_dir=target.parent)` builds out the
    repo's internal directory structure, then renames it to target (atomic on
    the same volume, no duplicating 4 GB). Skips outright if already present.
    """
    if target.exists():
        on_log(f"   already have {target.name}, skipping")
        return True
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        on_log("   missing huggingface_hub")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    endpoint = _resolve_endpoint()
    token = _hf_token()
    try:
        kwargs = dict(
            repo_id=repo_id,
            filename=repo_subpath,
            local_dir=str(target.parent),
            local_dir_use_symlinks=False,
        )
        if endpoint:
            kwargs["endpoint"] = endpoint
        if token:
            kwargs["token"] = token
        hf_hub_download(**kwargs)
    except Exception as exc:
        on_log(f"   failed {target.name}: {exc}")
        if _is_gated_auth_error(exc):
            on_log(
                "   -> This repo may be gated/private: please apply for and accept "
                "the model license at huggingface.co first, then fill in your "
                "HuggingFace token under Settings -> Keys (or set the HF_TOKEN "
                "environment variable) and retry."
            )
        return False
    src = target.parent / repo_subpath
    if src != target:
        try:
            target.unlink(missing_ok=True)
            src.rename(target)
        except OSError as exc:
            on_log(f"   rename failed {src} -> {target}: {exc}")
            return False
        # Clean up empty intermediate directories
        parent = src.parent
        while parent != target.parent and parent.exists():
            try:
                if any(parent.iterdir()):
                    break
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    on_log(f"   done {target.name}")
    return True


def download_snapshot(
    repo_id: str,
    target_dir: Path,
    *,
    allow_patterns: Optional[list[str]] = None,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Downloads a whole repo from HF into target_dir (used for multi-file
    transformers models).

    Unlike download_flat (single file + flat rename), this lands the whole
    directory preserving the repo's structure, so from_pretrained can point
    directly at target_dir. Skips if already ready (has a config.json).
    """
    if (target_dir / "config.json").exists():
        on_log(f"   already have {target_dir.name}, skipping")
        return True
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        on_log("   missing huggingface_hub")
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    endpoint = _resolve_endpoint()
    token = _hf_token()
    try:
        kwargs: dict = dict(repo_id=repo_id, local_dir=str(target_dir))
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        if endpoint:
            kwargs["endpoint"] = endpoint
        if token:
            kwargs["token"] = token
        snapshot_download(**kwargs)
    except Exception as exc:
        on_log(f"   failed {target_dir.name}: {exc}")
        if _is_gated_auth_error(exc):
            on_log(
                "   -> This repo may be gated/private: please apply for and accept "
                "the model license at huggingface.co first, then fill in your "
                "HuggingFace token under Settings -> Keys and retry."
            )
        return False
    on_log(f"   done {target_dir.name}")
    return True


def download_snapshot_ms(
    ms_repo_id: str,
    target_dir: Path,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Downloads a whole repo from ModelScope into target_dir (used for multi-file models).

    Requires ``pip install modelscope``; returns False if not installed. Skips if already ready.
    """
    if (target_dir / "config.json").exists():
        on_log(f"   already have {target_dir.name}, skipping")
        return True
    try:
        from modelscope import snapshot_download as ms_snapshot
    except ImportError:
        on_log("   missing modelscope (pip install modelscope)")
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    token = _ms_token()
    try:
        kwargs: dict = dict(model_id=ms_repo_id, local_dir=str(target_dir))
        if token:
            kwargs["token"] = token
        ms_snapshot(**kwargs)
    except Exception as exc:
        on_log(f"   failed {target_dir.name} (ModelScope): {exc}")
        return False
    on_log(f"   done {target_dir.name} (via ModelScope)")
    return True


