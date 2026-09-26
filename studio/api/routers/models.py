"""Model catalog / downloads (PR-6 commit 2, extracted from server.py).

Routes (PP7 first-cut domain + unified model source candidates):
    GET    /api/models/catalog        list known models + each one's disk status + current download status
    GET    /api/models/path-defaults  absolute paths for the 4 model fields as computed from current Settings
    POST   /api/models/download       start a background download, returns a status key
    POST   /api/model-sources/{domain}   add a source candidate (download-type / local file)
    DELETE /api/model-sources/{domain}   remove a candidate (leaves disk untouched; falls back to default if selected)

The local base-model register/unregister endpoints (POST/DELETE
/api/models/{family}/custom) have been replaced by local candidates under
/api/model-sources/{family} (unified source candidates, D1).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..schemas.models import (
    FamilySwitchRequest,
    ModelDownloadRequest,
    ModelSourceCandidateRequest,
)
from ... import secrets
from ...domain.errors import ValidationError
from ...domain.family_switch import switch_family
from ...services import models as model_downloader

router = APIRouter()


@router.get("/api/models/catalog")
def get_models_catalog() -> dict[str, Any]:
    """Used by the frontend Settings page's Models section: lists known models + each
    one's disk status + current download status."""
    return model_downloader.build_catalog()


@router.get("/api/models/path-defaults")
def get_models_path_defaults(family: str = "anima") -> dict[str, str]:
    """Absolute paths for the 4 model fields, computed from current Settings (resolved
    by the `family` query parameter).

    Used by the presets page's reset button and the initial fill for "new preset" -- both
    scenarios have no project context, so they can't get project_specific_defaults from
    /api/projects/{pid}/versions/{vid}/config, hence this separate endpoint.
    """
    try:
        return model_downloader.default_paths_for_new_version(family=family)
    except ValueError as exc:
        raise ValidationError(
            f"Unknown model family: {family}",
            code="model.family_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc


@router.post("/api/models/family-switch")
def switch_model_family(body: FamilySwitchRequest) -> dict[str, Any]:
    """Preview computation for switching a training config's model family (multi-model P4-3).

    Pure computation, nothing is persisted: recomputes the 4 weight paths + resets
    family-flavored fields (sampler / scheduler / timestep etc.) + disables capability
    fields unsupported by the target family, and returns the full config after switching
    plus a change list. The frontend uses this to pop a confirmation dialog; once the user
    confirms it goes through the normal save path.
    """
    try:
        path_defaults = model_downloader.default_paths_for_new_version(
            family=body.target)
    except ValueError as exc:
        raise ValidationError(
            f"Unknown model family: {body.target}",
            code="model.family_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc
    new_config, changes = switch_family(body.config, body.target, path_defaults)
    return {"config": new_config, "changes": changes}


@router.delete("/api/models/asset")
def delete_model_asset(model_id: str, variant: str | None = None) -> dict[str, Any]:
    """Delete a downloaded asset (the inverse of download: the user deletes it first,
    then re-downloads if needed).

    The target path is resolved server-side (arbitrary paths are not accepted); errors if
    a download is in progress / the file is in use.
    Returns the full catalog after deletion.
    """
    try:
        model_downloader.delete_asset(model_id, variant)
    except ValueError as exc:
        raise ValidationError(
            f"Invalid model selection: {exc}",
            code="model.invalid", details={"reason": str(exc)}, http_status=400,
        ) from exc
    except RuntimeError as exc:
        raise ValidationError(
            str(exc), code="model.delete_failed", http_status=409,
        ) from exc
    return model_downloader.build_catalog()


@router.post("/api/models/download")
def start_model_download(body: ModelDownloadRequest) -> dict[str, Any]:
    """Start a background download, return a status key immediately; the frontend
    watches progress via SSE (`model_download_changed`) or by polling the catalog."""
    try:
        key = model_downloader.trigger(body.model_id, body.variant)
    except ValueError as exc:
        raise ValidationError(
            f"Invalid model selection: {exc}",
            code="model.invalid", details={"reason": str(exc)}, http_status=400,
        ) from exc
    snap = model_downloader.get_status_snapshot()
    return {"key": key, "status": snap.get(key, {}).get("status", "running")}


# ---------------------------------------------------------------------------
# Unified model source candidates (docs/design/model-source-unification.md §6)
# ---------------------------------------------------------------------------

# HF / MS repo ids look like owner/name. Validation kept minimal (D3): no network probe,
# a nonexistent repo just errors out at download time.
_REPO_ID_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")


def _source_domains() -> set[str]:
    from ...services.models.families import FAMILY_ASSETS

    return (
        set(secrets.MODEL_SOURCE_REPO_DOMAINS)
        | {"upscaler", secrets.VAE_DOMAIN}
        | set(FAMILY_ASSETS.keys())
        # per-family text encoders (anima_te / krea2_te): local transformers directory
        | {secrets.te_domain(f) for f in FAMILY_ASSETS}
    )


def _domain_or_400(domain: str) -> str:
    if domain not in _source_domains():
        raise ValidationError(
            f"Unknown model source domain: {domain}",
            code="model_source.domain_invalid",
            details={"domain": domain}, http_status=400,
        )
    return domain


def _require_file(p: Path, exts: tuple[str, ...]) -> None:
    if p.suffix.lower() not in exts:
        raise ValidationError(
            f"Select a {' / '.join(exts)} file",
            code="file.ext_invalid", details={"types": " / ".join(exts)},
            http_status=400,
        )
    if not p.is_file():
        raise ValidationError(
            "File not found", code="model.not_found",
            details={"path": str(p)}, http_status=400,
        )


def _require_dir_with(p: Path, required: tuple[str, ...]) -> None:
    missing = [f for f in required if not (p / f).exists()]
    if not p.is_dir() or missing:
        raise ValidationError(
            "Directory is missing required model files",
            code="model_source.dir_incomplete",
            details={"path": str(p), "missing": missing}, http_status=400,
        )


def _validate_candidate(domain: str, cand: "secrets.SourceCandidate") -> None:
    """Simple validation (D3): format / existence / domain structure; errors at runtime are the fallback."""
    from ...services.models.families import FAMILY_ASSETS
    from ...services.models.paths import (
        TEXT_ENCODER_MARKER,
        UPSCALER_EXTS,
        WD14_FILES,
    )

    if cand.kind == "download":
        # VAE / text encoders currently only support "pick a local file": official
        # weights go through their own download cards, and third-party repo downloads
        # would need new downloader placement rules; until that's implemented we error
        # out explicitly rather than silently saving a candidate that can never download.
        if domain == secrets.VAE_DOMAIN or secrets.te_domain_family(domain):
            raise ValidationError(
                "This model type only supports picking a local file or folder",
                code="model_source.download_unsupported",
                details={"domain": domain}, http_status=400,
            )
        if not _REPO_ID_RE.match(cand.repo):
            raise ValidationError(
                "Repository ID must look like owner/name",
                code="model_source.repo_invalid",
                details={"repo": cand.repo}, http_status=400,
            )
        # Single-file assets must have a filename + an allowed extension; directory-type assets don't accept filename
        if domain == "upscaler" or domain in FAMILY_ASSETS:
            exts = UPSCALER_EXTS if domain == "upscaler" else (".safetensors",)
            name = Path(cand.filename).name
            if not name or not name.lower().endswith(exts):
                raise ValidationError(
                    f"File name must end with {' / '.join(exts)}",
                    code="file.ext_invalid",
                    details={"types": " / ".join(exts)}, http_status=400,
                )
        elif cand.filename:
            raise ValidationError(
                "This model type downloads a whole repository (no file name)",
                code="model_source.filename_unexpected", http_status=400,
            )
        return

    # kind == "local"
    if not cand.path.strip() or not secrets.is_abs_path(cand.path):
        raise ValidationError(
            "An absolute local path is required",
            code="model_source.path_invalid",
            details={"path": cand.path}, http_status=400,
        )
    p = Path(cand.path).expanduser()
    if domain == "wd14":
        _require_dir_with(p, WD14_FILES)
    elif domain in ("eval_clip", "eval_dino"):
        _require_dir_with(p, ("config.json",))
    elif domain == "eval_ccip":
        _require_dir_with(
            p, ("model_feat.onnx", "model_metrics.onnx", "metrics.json"))
    elif domain == "upscaler":
        _require_file(p, UPSCALER_EXTS)
    elif domain == secrets.VAE_DOMAIN:
        _require_file(p, (".safetensors",))
    elif secrets.te_domain_family(domain):
        # A text encoder = a transformers directory (Qwen3 / Qwen3-VL / the official fp8
        # single-file variant all ship a config.json). If a bare weights file is picked,
        # this error names what's missing.
        _require_dir_with(p, (TEXT_ENCODER_MARKER,))
    elif domain == "cltagger":
        _require_file(p, (".onnx",))
        mapping = cand.extra.get("tag_mapping_path", "")
        if not mapping or not secrets.is_abs_path(mapping):
            raise ValidationError(
                "tag_mapping_path (absolute path) is required",
                code="model_source.path_invalid", http_status=400,
            )
        _require_file(Path(mapping).expanduser(), (".json",))
    else:  # base-model families: single-file .safetensors (same as PathPicker registration validation)
        _require_file(p, (".safetensors",))


def _candidate_from_request(body: ModelSourceCandidateRequest) -> "secrets.SourceCandidate":
    if body.kind not in ("download", "local"):
        raise ValidationError(
            f"Unknown candidate kind: {body.kind}",
            code="model_source.kind_invalid", http_status=400,
        )
    return secrets.SourceCandidate(
        kind=body.kind,
        repo=body.repo.strip(),
        filename=Path(body.filename).name if body.filename.strip() else "",
        path=body.path.strip(),
        extra={k: str(v).strip() for k, v in body.extra.items() if str(v).strip()},
    )


def _selected_value_reset(domain: str, removed: "secrets.SourceCandidate") -> dict[str, Any]:
    """When the removed candidate is the currently selected one -> also return an update
    partial that resets the selected value to its default."""
    from ...services.models.families import FAMILY_ASSETS
    from ...services.models.paths import DEFAULT_UPSCALER

    s = secrets.load()
    removed_value = removed.repo if removed.kind == "download" else removed.path
    if domain == "wd14" and s.wd14.model_id == removed_value:
        return {"wd14": {"model_id": secrets.DEFAULT_WD14_MODELS[0]}}
    if domain == "cltagger":
        # download=fork repo (compare to model_id); local=two files (compare to model_path)
        is_current = (
            removed.kind == "download" and s.cltagger.model_id == removed.repo
        ) or (
            removed.kind == "local" and s.cltagger.model_path == removed.path
        )
        if is_current:
            fields = secrets.CLTaggerConfig.model_fields
            return {"cltagger": {
                "model_id": fields["model_id"].default,
                "model_path": fields["model_path"].default,
                "tag_mapping_path": fields["tag_mapping_path"].default,
            }}
    if domain in ("eval_clip", "eval_dino", "eval_ccip"):
        field = {
            "eval_clip": "clip_model_name",
            "eval_dino": "dino_model_name",
            "eval_ccip": "ccip_model_name",
        }[domain]
        if getattr(s.eval_metrics, field) == removed_value:
            default = secrets.EvalMetricModelsConfig.model_fields[field].default
            return {"eval_metrics": {field: default}}
    if domain == "upscaler":
        sel = s.models.selected_upscaler
        if sel and sel in (removed_value, removed.filename):
            return {"models": {"selected_upscaler": DEFAULT_UPSCALER}}
    if domain == secrets.VAE_DOMAIN and s.models.selected_vae == removed_value:
        # empty string = follow the official location (ModelsConfig.selected_vae's default semantics)
        return {"models": {"selected_vae": ""}}
    te_family = secrets.te_domain_family(domain)
    if te_family and s.models.selected_te.get(te_family) == removed_value:
        # official default = the first entry of that family's text_encoder_presets
        # (anima: "" = official directory; krea2: "bf16")
        presets = FAMILY_ASSETS[te_family].text_encoder_presets(
            model_downloader.models_root())
        return {"models": {"selected_te": {
            **s.models.selected_te, te_family: str(presets[0]["value"]),
        }}}
    if domain in FAMILY_ASSETS and s.models.selected.get(domain) == removed_value:
        return {"models": {"selected": {
            **s.models.selected, domain: FAMILY_ASSETS[domain].latest,
        }}}
    return {}


@router.post("/api/model-sources/{domain}")
def add_model_source(
    domain: str, body: ModelSourceCandidateRequest
) -> dict[str, Any]:
    """Add a source candidate (dedup then append), return the new catalog.

    Validation kept minimal: repo looks like owner/name, extension allowlist for
    single-file assets, local path exists + domain structure (wd14's two files / eval's
    config.json etc.). No network probing.
    """
    _domain_or_400(domain)
    cand = _candidate_from_request(body)
    if domain == "cltagger" and cand.kind == "download":
        # A fork-repo candidate defaults to inheriting the current two-file relative path
        # (users usually fork the same version; falls back to the first built-in
        # preset's path if the current one is a local absolute path)
        from ...services.models.paths import CLTAGGER_VERSIONS

        cfg = secrets.load().cltagger
        mp, tmp = cfg.model_path, cfg.tag_mapping_path
        if secrets.is_abs_path(mp) or secrets.is_abs_path(tmp):
            first = next(iter(CLTAGGER_VERSIONS.values()))
            mp, tmp = str(first["model_path"]), str(first["tag_mapping_path"])
        cand.extra.setdefault("model_path", mp)
        cand.extra.setdefault("tag_mapping_path", tmp)
    _validate_candidate(domain, cand)
    existing = secrets.load().model_sources.get(domain, [])
    if all(c.identity() != cand.identity() for c in existing):
        secrets.update({"model_sources": {
            domain: [c.model_dump() for c in existing] + [cand.model_dump()],
        }})
    return model_downloader.build_catalog()


@router.delete("/api/model-sources/{domain}")
def remove_model_source(
    domain: str, body: ModelSourceCandidateRequest
) -> dict[str, Any]:
    """Remove a candidate (only removed from the list, disk files untouched), return the new catalog.

    When the removed candidate is the currently selected one, the selected value falls
    back to that domain's default (wd14's first built-in / eval schema default /
    cltagger's official preset / upscaler default / the latest official variant for base
    model families) -- consistent with the existing "unregistering a local base model
    falls back to latest" semantics.
    """
    _domain_or_400(domain)
    cand = _candidate_from_request(body)
    existing = secrets.load().model_sources.get(domain, [])
    remaining = [c for c in existing if c.identity() != cand.identity()]
    if len(remaining) != len(existing):
        partial: dict[str, Any] = {
            "model_sources": {domain: [c.model_dump() for c in remaining]},
        }
        for key, val in _selected_value_reset(domain, cand).items():
            partial[key] = val
        secrets.update(partial)
    return model_downloader.build_catalog()
