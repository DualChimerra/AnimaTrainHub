"""Model catalog -- scans disk to assemble "which models are downloaded, their
target paths, their sizes" (the 4th piece of PR-3.8's 4-way split).

build_catalog is the core of the /api/models/catalog endpoint; the frontend's
ModelsPage uses it to show install status. Depends on paths.py's constants +
target functions; never triggers a download (read-only disk scan).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ... import secrets
from .. import eval_registry
from .downloader import get_status_snapshot
from .families import FAMILY_ASSETS
from .paths import (
    CLTAGGER_VERSIONS,
    DEFAULT_UPSCALER,
    TAEFLUX_FILES,
    TAEFLUX_REPO,
    UPSCALER_EXTS,
    UPSCALER_VARIANTS,
    TEXT_ENCODER_MARKER,
    WD14_FILES,
    ccip_model_dir,
    cltagger_required_files,
    cltagger_target_root,
    eval_model_target_dir,
    models_root,
    qwen_image_vae_target,
    selected_upscaler,
    taeflux_dir,
    upscaler_dir,
    upscaler_target,
    wd14_target_dir,
)

# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


# Download size estimates for common CLIP / DINO models (bytes, approximate).
# Gives the user a size reference before downloading; unknown model_ids show no
# estimate. Once downloaded, the UI uses the actual directory size instead.
_EVAL_SIZE_ESTIMATES = {
    "openai/clip-vit-base-patch32": 605_000_000,
    "openai/clip-vit-base-patch16": 599_000_000,
    "openai/clip-vit-large-patch14": 1_710_000_000,
    "facebook/dinov2-small": 88_000_000,
    "facebook/dinov2-base": 346_000_000,
    "facebook/dinov2-large": 1_220_000_000,
    "facebook/dinov2-giant": 4_600_000_000,
    # CCIP (deepghs/ccip_onnx variants): only model_feat.onnx (~150MB) + metric + threshold are downloaded.
    "ccip-caformer-24-randaug-pruned": 152_000_000,
    "ccip-caformer_b36-24": 384_000_000,
}

# The 3 required files for a CCIP variant (all must be present to count as downloaded).
_CCIP_FILES = ("model_feat.onnx", "model_metrics.onnx", "metrics.json")


def _file_status(p: Path) -> dict[str, Any]:
    try:
        st = p.stat()
        return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}
    except OSError:
        return {"exists": False, "size": 0, "mtime": 0.0}


def build_catalog(root: Optional[Path] = None) -> dict[str, Any]:
    """Scan disk and assemble the catalog for frontend display.

    Each entry has `id` / `name` / `description` / target path / downloaded
    status. When the Anima main model has multiple versions, returns
    `variants[]`, each with its own status.
    The `downloads` field returns the status of any currently active download.
    """
    r = root or models_root()

    # Model-family sections are built by iterating the FAMILY_ASSETS registry
    # (multi-model PR-4); with a single family the output is byte-for-byte
    # identical to the old implementation, so the frontend needs zero changes
    models_cfg = secrets.load().models
    family_sections: dict[str, Any] = {}
    for _assets in FAMILY_ASSETS.values():
        family_sections.update(_assets.catalog_sections(r, models_cfg))

    _secrets = secrets.load()
    cl_cfg = _secrets.cltagger
    wd14_cfg = _secrets.wd14
    eval_cfg = _secrets.eval_metrics
    src_cfg = _secrets.download_sources
    source_cfg = _secrets.model_sources

    # CLIP / DINO eval-metric models: one variant row each; the whole directory
    # counts as "downloaded" once it has a config.json.
    eval_variants = []
    for kind, mid in (("clip", eval_cfg.clip_model_name), ("dino", eval_cfg.dino_model_name)):
        target = eval_model_target_dir(r, kind, mid)
        exists = (target / "config.json").exists()
        size = (
            sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            if target.exists() else 0
        )
        eval_variants.append({
            "kind": kind,
            "model_id": mid,
            "target_path": str(target),
            "exists": exists,
            "size": size,
            "size_estimate": _EVAL_SIZE_ESTIMATES.get(mid, 0),
        })
    # CCIP (anime character identity): counts as downloaded only when all 3 files exist (no config.json).
    ccip_mid = eval_cfg.ccip_model_name
    ccip_dir = ccip_model_dir(r, ccip_mid)
    eval_variants.append({
        "kind": "ccip",
        "model_id": ccip_mid,
        "target_path": str(ccip_dir),
        "exists": all((ccip_dir / f).exists() for f in _CCIP_FILES),
        "size": (
            sum(f.stat().st_size for f in ccip_dir.rglob("*") if f.is_file())
            if ccip_dir.exists() else 0
        ),
        "size_estimate": _EVAL_SIZE_ESTIMATES.get(ccip_mid, 0),
    })

    # One row per WD14 candidate model_id: counts as "downloaded" only when both files exist.
    wd14_variants = []
    for mid in wd14_cfg.model_ids:
        target = wd14_target_dir(r, mid)
        files = [{"name": f, **_file_status(target / f)} for f in WD14_FILES]
        all_exist = all(f["exists"] for f in files)
        total_size = sum(f["size"] for f in files)
        wd14_variants.append({
            "model_id": mid,
            "is_current": mid == wd14_cfg.model_id,
            "target_path": str(target),
            "exists": all_exist,
            "size": total_size,
            "files": files,
        })

    # CLTagger version presets (each variant can come from a different HF repo).
    cl_root = cltagger_target_root(r, cl_cfg.model_id)
    cl_variants = []
    for label, preset in CLTAGGER_VERSIONS.items():
        mid = preset["model_id"]
        mp = preset["model_path"]
        tmp = preset["tag_mapping_path"]
        variant_root = cltagger_target_root(r, mid)
        version_dir = variant_root / Path(mp).parent
        files = [
            {"name": f, **_file_status(variant_root / f)}
            for f in cltagger_required_files(mp, tmp)
        ]
        all_exist = all(f["exists"] for f in files)
        total_size = sum(f["size"] for f in files)
        cl_variants.append({
            "label": label,
            "model_id": mid,
            "model_path": mp,
            "tag_mapping_path": tmp,
            "description": preset.get("description", ""),
            # target_path = the repo's local root; version_dir = this version's subdirectory (used by the UI to show where files land).
            "target_path": str(variant_root),
            "version_dir": str(version_dir),
            "is_current": (
                cl_cfg.model_id == mid
                and cl_cfg.model_path == mp
                and cl_cfg.tag_mapping_path == tmp
            ),
            "exists": all_exist,
            "size": total_size,
            "files": files,
        })

    # -- Unified source candidate rows (docs/design/model-source-unification.md §6) --
    #
    # One flat list per domain: built-in presets + user candidates (download /
    # local); the capability flags (removable / deletable) are assembled here so
    # the frontend's generic candidate card no longer has to decide them itself.
    # This currently covers wd14 / eval_*; upscaler / main model family / cltagger
    # get folded in as their own sections migrate.

    def _source_row(
        *, kind: str, value: str, download_id: Optional[str],
        exists: bool, size: int, is_current: bool,
        label: Optional[str] = None, files: Optional[list] = None,
        size_estimate: int = 0, extra: Optional[dict] = None,
        download_variant: Optional[str] = None,
        status_key: Optional[str] = None,
        description: str = "",
        removable: Optional[bool] = None,
        candidate: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        variant = download_variant or value
        return {
            # The user candidate's raw storage record (the identity key for
            # DELETE /api/model-sources); absent for preset / scanned rows.
            "candidate": candidate,
            "kind": kind,               # preset | download | local | scanned
            "value": value,             # the value written into the selected-value field (repo id / absolute path / label)
            "label": label or value,
            "description": description,
            # Triggers a download: POST /api/models/download {model_id: download_id,
            # variant: download_variant}; status key is built by default and can be
            # overridden explicitly (upscaler custom's key looks like
            # upscaler:custom:{filename}). Not set for local.
            "download_id": download_id,
            "download_variant": variant if download_id else None,
            "status_key": (
                status_key if status_key is not None
                else (f"{download_id}:{variant}" if download_id else None)
            ),
            "exists": exists,
            "size": size,
            "files": files,
            "size_estimate": size_estimate,
            "is_current": is_current,
            # Built-ins can't be removed (protects the defaults); scanned rows
            # aren't in the candidate store either, so they can't be removed
            "removable": (
                removable if removable is not None
                else kind not in ("preset", "scanned")
            ),
            "deletable": kind != "local",    # local files are never deleted from the UI
            "extra": extra or {},
        }

    def _wd14_row(kind: str, value: str) -> dict[str, Any]:
        target = wd14_target_dir(r, value)
        files = [{"name": f, **_file_status(target / f)} for f in WD14_FILES]
        return _source_row(
            kind=kind, value=value,
            download_id="wd14" if kind != "local" else None,
            exists=all(f["exists"] for f in files),
            size=sum(f["size"] for f in files),
            files=files,
            is_current=value == wd14_cfg.model_id,
        )

    def _eval_row(domain: str, em_kind: str, kind: str, value: str) -> dict[str, Any]:
        if em_kind == "ccip":
            target = ccip_model_dir(r, value)
            exists = all((target / f).exists() for f in _CCIP_FILES)
        else:
            target = eval_model_target_dir(r, em_kind, value)
            exists = (target / "config.json").exists()
        size = (
            sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            if target.exists() else 0
        )
        current = {
            "eval_clip": eval_cfg.clip_model_name,
            "eval_dino": eval_cfg.dino_model_name,
            "eval_ccip": eval_cfg.ccip_model_name,
        }[domain]
        return _source_row(
            kind=kind, value=value,
            download_id=domain if kind != "local" else None,
            exists=exists, size=size,
            size_estimate=_EVAL_SIZE_ESTIMATES.get(value, 0),
            is_current=value == current,
        )

    def _user_rows(domain: str, row_fn) -> list[dict[str, Any]]:
        rows = []
        for c in source_cfg.get(domain, []):
            value = c.repo if c.kind == "download" else c.path
            row = row_fn(c.kind, value)
            row["extra"] = dict(c.extra)
            row["candidate"] = c.model_dump()
            rows.append(row)
        return rows

    model_source_rows: dict[str, list[dict[str, Any]]] = {
        "wd14": (
            [_wd14_row("preset", m) for m in secrets.DEFAULT_WD14_MODELS]
            + _user_rows("wd14", _wd14_row)
        ),
    }
    _eval_defaults = {
        "eval_clip": ("clip", secrets.EvalMetricModelsConfig.model_fields["clip_model_name"].default),
        "eval_dino": ("dino", secrets.EvalMetricModelsConfig.model_fields["dino_model_name"].default),
        "eval_ccip": ("ccip", secrets.EvalMetricModelsConfig.model_fields["ccip_model_name"].default),
    }
    for _domain, (_em_kind, _default) in _eval_defaults.items():
        model_source_rows[_domain] = (
            [_eval_row(_domain, _em_kind, "preset", str(_default))]
            + _user_rows(
                _domain,
                lambda kind, value, _d=_domain, _k=_em_kind: _eval_row(_d, _k, kind, value),
            )
        )

    # Upscalers: merge presets + disk scan.
    # - Pass 1: list all of UPSCALER_VARIANTS (offers a "download" entry point even if not downloaded)
    # - Pass 2: scan every .pth/.safetensors in upscalers/, and add any not in the
    #   presets to the list as custom entries (files landed there by a custom repo
    #   download or a future upload feature)
    selected_label = selected_upscaler()
    upscaler_variants = []
    seen_filenames: set[str] = set()
    for label, info in UPSCALER_VARIANTS.items():
        target = upscaler_target(label, r)
        seen_filenames.add(info["filename"])
        hf_repo = (info.get("hf") or (None,))[0]
        ms_repo = (info.get("ms") or (None,))[0]
        upscaler_variants.append({
            "label": label,
            "filename": info["filename"],
            "kind": "preset",
            "hf_repo": hf_repo,
            "ms_repo": ms_repo,
            "size_mb": info.get("size_mb"),
            "description": info.get("description", ""),
            "target_path": str(target),
            "is_current": label == selected_label,
            **_file_status(target),
        })
    up_dir = upscaler_dir(r)
    if up_dir.exists():
        for f in sorted(up_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in UPSCALER_EXTS:
                continue
            if f.name in seen_filenames:
                continue
            upscaler_variants.append({
                "label": f.name,
                "filename": f.name,
                "kind": "custom",
                "hf_repo": None,
                "ms_repo": None,
                "size_mb": None,
                "description": "Custom / already downloaded",
                "target_path": str(f),
                "is_current": f.name == selected_label or f.stem == selected_label,
                **_file_status(f),
            })

    # Unified CLTagger row: the selected value is the (model_id, model_path,
    # tag_mapping_path) triple; the row's value is a `|`-joined composite key;
    # the actual triple lives in extra, and the frontend writes all three fields
    # back from extra on selection. A fork repo (download candidate) carries its
    # own pair of relative file paths (mirror override retired, D4); a local
    # candidate is a pair of absolute file paths.
    def _cl_value(mid: str, mp: str, tmp: str) -> str:
        return f"{mid}|{mp}|{tmp}"

    cl_rows: list[dict[str, Any]] = []
    for v in cl_variants:
        cl_rows.append(_source_row(
            kind="preset",
            value=_cl_value(v["model_id"], v["model_path"], v["tag_mapping_path"]),
            label=v["label"],
            download_id="cltagger", download_variant=v["label"],
            status_key=f"cltagger:{v['label']}",
            exists=v["exists"], size=v["size"], files=v["files"],
            is_current=v["is_current"],
            description=str(v.get("description", "")),
            extra={
                "model_id": v["model_id"], "model_path": v["model_path"],
                "tag_mapping_path": v["tag_mapping_path"],
            },
        ))
    for c in source_cfg.get("cltagger", []):
        mp = c.path if c.kind == "local" else c.extra.get("model_path", "")
        tmp = c.extra.get("tag_mapping_path", "")
        if c.kind == "download":
            variant_root = cltagger_target_root(r, c.repo)
            files = [
                {"name": f, **_file_status(variant_root / f)}
                for f in cltagger_required_files(mp, tmp)
            ]
            cl_rows.append(_source_row(
                kind="download", value=_cl_value(c.repo, mp, tmp), label=c.repo,
                download_id="cltagger_custom", download_variant=c.repo,
                exists=all(f["exists"] for f in files),
                size=sum(f["size"] for f in files), files=files,
                is_current=(
                    cl_cfg.model_id == c.repo
                    and cl_cfg.model_path == mp
                    and cl_cfg.tag_mapping_path == tmp
                ),
                description=mp,
                extra={
                    "model_id": c.repo, "model_path": mp,
                    "tag_mapping_path": tmp,
                },
                candidate=c.model_dump(),
            ))
        else:
            p_model = Path(mp)
            p_map = Path(tmp) if tmp else None
            st = _file_status(p_model)
            cl_rows.append(_source_row(
                kind="local", value=_cl_value("", mp, tmp), label=p_model.name,
                download_id=None,
                exists=st["exists"] and bool(p_map and p_map.is_file()),
                size=st["size"],
                is_current=(
                    cl_cfg.model_path == mp
                    and cl_cfg.tag_mapping_path == tmp
                ),
                description=tmp,
                extra={
                    "model_id": "", "model_path": mp,
                    "tag_mapping_path": tmp,
                },
                candidate=c.model_dump(),
            ))
    model_source_rows["cltagger"] = cl_rows

    # Unified upscaler rows: presets + download/local candidates + disk-scan
    # fallback (D6). value semantics match selected_upscaler: preset=label /
    # download and scanned=filename / local=absolute path.
    up_rows: list[dict[str, Any]] = []
    for label, info in UPSCALER_VARIANTS.items():
        target = upscaler_target(label, r)
        st = _file_status(target)
        up_rows.append(_source_row(
            kind="preset", value=label,
            download_id="upscaler",
            exists=st["exists"], size=st["size"],
            is_current=label == selected_label,
            description=str(info.get("description", "")),
            size_estimate=int(info.get("size_mb") or 0) * 1_000_000,
            extra={
                "hf_repo": (info.get("hf") or ("",))[0] or "",
                "ms_repo": (info.get("ms") or ("",))[0] or "",
            },
        ))
    _custom_filenames: set[str] = set()
    for c in source_cfg.get("upscaler", []):
        if c.kind == "download":
            save_name = Path(c.filename).name
            _custom_filenames.add(save_name)
            target = upscaler_dir(r) / save_name
            st = _file_status(target)
            up_rows.append(_source_row(
                kind="download", value=save_name,
                download_id="upscaler_custom", download_variant=c.filename,
                status_key=f"upscaler:custom:{save_name}",
                exists=st["exists"], size=st["size"],
                is_current=save_name == selected_label,
                description=c.repo,
                candidate=c.model_dump(),
            ))
        else:
            p = Path(c.path)
            st = _file_status(p)
            up_rows.append(_source_row(
                kind="local", value=c.path, label=p.name, download_id=None,
                exists=st["exists"], size=st["size"],
                is_current=c.path == selected_label,
                candidate=c.model_dump(),
            ))
    for v in upscaler_variants:
        # Files found on disk that are neither a preset nor registered as a download candidate: deletion is the only option
        if v["kind"] != "custom" or v["filename"] in _custom_filenames:
            continue
        up_rows.append(_source_row(
            kind="scanned", value=v["filename"], label=v["label"],
            download_id="upscaler", download_variant=v["filename"],
            status_key=f"upscaler:custom:{v['filename']}",
            exists=bool(v.get("exists")), size=int(v.get("size") or 0),
            is_current=bool(v.get("is_current")),
        ))
    model_source_rows["upscaler"] = up_rows

    # Unified main-model-family rows: official variants (preset) + download
    # candidates (third-party finetunes, value=absolute path on disk, same
    # semantics as local/selected) + local (registered via PathPicker).
    for family_id in FAMILY_ASSETS:
        main = family_sections.get(f"{family_id}_main")
        if not main:
            continue
        selected_val = str(
            models_cfg.selected.get(family_id) or main.get("latest") or "")
        fam_rows: list[dict[str, Any]] = []
        for v in main["variants"]:
            fam_rows.append(_source_row(
                kind="preset", value=v["variant"],
                download_id=f"{family_id}_main",
                exists=bool(v.get("exists")), size=int(v.get("size") or 0),
                is_current=v["variant"] == selected_val,
                extra={"purpose": str(v.get("purpose") or "")},
            ))
        for c in source_cfg.get(family_id, []):
            if c.kind == "download":
                save_name = Path(c.filename).name
                target = r / "diffusion_models" / save_name
                st = _file_status(target)
                fam_rows.append(_source_row(
                    kind="download", value=str(target), label=save_name,
                    download_id=f"{family_id}_custom", download_variant=c.filename,
                    exists=st["exists"], size=st["size"],
                    is_current=str(target) == selected_val,
                    description=c.repo,
                    candidate=c.model_dump(),
                ))
            else:
                p = Path(c.path)
                st = _file_status(p)
                fam_rows.append(_source_row(
                    kind="local", value=c.path, label=p.name, download_id=None,
                    exists=st["exists"], size=st["size"],
                    is_current=c.path == selected_val,
                    candidate=c.model_dump(),
                ))
        model_source_rows[family_id] = fam_rows

    # VAE (a family-agnostic shared asset): the official location as a preset
    # (value="" = follows official) + user-registered local .safetensors files.
    # Selected value = models_cfg.selected_vae.
    vae_selected = str(models_cfg.selected_vae or "")
    vae_official = qwen_image_vae_target(r)
    official_st = _file_status(vae_official)
    vae_rows: list[dict[str, Any]] = [_source_row(
        kind="preset", value="", label=vae_official.name,
        download_id="anima_vae", download_variant=None,
        status_key="anima_vae",
        exists=official_st["exists"], size=official_st["size"],
        # Empty = official; an old config that explicitly stored the official
        # absolute path also counts as selecting official
        is_current=vae_selected in ("", str(vae_official)),
        description=str(vae_official),
    )]
    for c in source_cfg.get(secrets.VAE_DOMAIN, []):
        if c.kind != "local":
            continue
        p_vae = Path(c.path)
        st = _file_status(p_vae)
        vae_rows.append(_source_row(
            kind="local", value=c.path, label=p_vae.name, download_id=None,
            exists=st["exists"], size=st["size"],
            is_current=c.path == vae_selected,
            description=c.path,
            candidate=c.model_dump(),
        ))
    model_source_rows[secrets.VAE_DOMAIN] = vae_rows

    # Per-family text encoder ("clip"): the family's official variant presets +
    # user-registered local transformers directories. Selected value =
    # models_cfg.selected_te[family].
    for family_id, _assets in FAMILY_ASSETS.items():
        te_domain = secrets.te_domain(family_id)
        te_selected = str((models_cfg.selected_te or {}).get(family_id) or "")
        te_rows: list[dict[str, Any]] = []
        presets = _assets.text_encoder_presets(r)
        preset_values = {str(preset["value"]) for preset in presets}
        for preset in presets:
            files = list(preset["files"])
            te_rows.append(_source_row(
                kind="preset", value=str(preset["value"]),
                label=str(preset["label"]),
                download_id=str(preset["download_id"]),
                download_variant=None,
                status_key=str(preset["status_key"]),
                exists=all(f["exists"] for f in files),
                size=sum(f["size"] for f in files),
                files=files,
                # If the selected value isn't among the official variants (a
                # local directory / invalid) -> only the default variant (the
                # list's first entry, each family's fallback) is highlighted
                # when nothing has been selected yet
                is_current=(
                    te_selected == str(preset["value"])
                    or (
                        te_selected not in preset_values
                        and not secrets.is_abs_path(te_selected)
                        and preset is presets[0]
                    )
                ),
                description=str(preset["description"]),
            ))
        for c in source_cfg.get(te_domain, []):
            if c.kind != "local":
                continue
            p_te = Path(c.path)
            marker = p_te / TEXT_ENCODER_MARKER
            size = (
                sum(f.stat().st_size for f in p_te.rglob("*") if f.is_file())
                if p_te.is_dir() else 0
            )
            te_rows.append(_source_row(
                kind="local", value=c.path, label=p_te.name, download_id=None,
                exists=marker.is_file(), size=size,
                is_current=c.path == te_selected,
                description=c.path,
                candidate=c.model_dump(),
            ))
        model_source_rows[te_domain] = te_rows

    return {
        "models_root": str(r),
        **family_sections,
        "wd14": {
            "id": "wd14",
            "name": "WD14",
            "description": "SmilingWolf ONNX tagger family",
            "repo": "SmilingWolf/*",
            "current_model_id": wd14_cfg.model_id,
            "variants": wd14_variants,
        },
        "cltagger": {
            "id": "cltagger",
            "name": "CLTagger",
            "description": "cella110n CLTagger ONNX",
            "repo": cl_cfg.model_id,
            "target_dir": str(cl_root),
            "current_model_path": cl_cfg.model_path,
            "current_tag_mapping_path": cl_cfg.tag_mapping_path,
            "variants": cl_variants,
        },
        "eval_metrics": {
            "id": "eval_metrics",
            "name": "Eval metric models",
            "description": "CLIP / DINO, used for post-training LoRA metric evaluation",
            "variants": eval_variants,
        },
        # Eval metric registry (used by the Settings checkbox list): each metric's key/label/description/default.
        "eval_metric_catalog": eval_registry.public_catalog(),
        "upscalers": {
            "id": "upscalers",
            "name": "Upscalers",
            "description": "Super-resolution models used during preprocessing",
            "default": DEFAULT_UPSCALER,
            "current": selected_label,
            "target_dir": str(upscaler_dir(r)),
            "variants": upscaler_variants,
        },
        # Unified source candidate rows (consumed by the frontend's generic candidate card; key = domain).
        "model_sources": model_source_rows,
        # Download-source selection by type: dual-source types get a dropdown,
        # HF-only types get a single-choice indicator. current comes from
        # secrets.download_sources (already migrated seed); available decides
        # whether the frontend renders a real dropdown or a disabled 1-option box.
        "download_source_options": {
            "training": {"current": src_cfg.get("training", "huggingface"),
                         "available": ["huggingface", "modelscope"]},
            "wd14": {"current": src_cfg.get("wd14", "huggingface"),
                     "available": ["huggingface", "modelscope"]},
            "eval": {"current": src_cfg.get("eval", "huggingface"),
                     "available": ["huggingface", "modelscope"]},
            "upscaler": {"current": src_cfg.get("upscaler", "huggingface"),
                         "available": ["huggingface", "modelscope"]},
            "cltagger": {"current": "huggingface", "available": ["huggingface"]},
            "taeflux": {"current": "huggingface", "available": ["huggingface"]},
        },
        "downloads": get_status_snapshot(),
    }
