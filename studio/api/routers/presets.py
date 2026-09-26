"""Preset CRUD + import/export + schema (PR-5, extracted from server.py).

13 routes:
    GET  /api/schema                            TrainingConfig JSON schema + GROUP_ORDER
    GET  /api/presets                           list
    GET  /api/presets/{name}                    read
    PUT  /api/presets/{name}                    write
    DELETE /api/presets/{name}                  delete
    POST /api/presets/{name}/duplicate          duplicate
    GET  /api/presets/{name}/download           download yaml
    POST /api/presets/{name}/export             export to data_exports/
    POST /api/presets/import-from-data-exports  import from data_exports/
    POST /api/presets/import-from-path          import from absolute server path
    POST /api/presets/import                    upload + parse + schema validate
    *    /api/configs                           308 redirect -> /api/presets (legacy)
    *    /api/configs/{rest:path}               308 redirect -> /api/presets/{rest} (legacy)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from .. import errors as _errors
from ...domain.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from ..schemas.presets import (
    DuplicateRequest,
    PresetExportBody,
    PresetImportBody,
    PresetImportFromPathBody,
)
from ...services.presets import io as presets_io
from ...paths import DATA_EXPORTS
from ...schema import GROUP_ORDER, TrainingConfig

router = APIRouter()


@router.get("/api/schema")
def get_schema() -> dict[str, Any]:
    """Return TrainingConfig's JSON Schema + group order, which the frontend uses to render the form."""
    schema = TrainingConfig.model_json_schema()
    _apply_feature_flags(schema)
    return {
        "schema": schema,
        "groups": [
            {"key": k, "label": label, "default_collapsed": dc}
            for k, label, dc in GROUP_ORDER
        ],
    }


@router.post("/api/schema/preview-yaml")
def preview_config_yaml_endpoint(body: dict[str, Any]) -> dict[str, Any]:
    """Current form config -> yaml text using the same serialization path as the saved
    on-disk file (R4).

    Replaces the frontend's pruneInactiveConfig + configToYaml double mirror: the preview
    no longer just "claims to match" the saved file, it's physically identical. Pure
    computation, nothing is persisted; tolerant-fixup semantics match saving.
    """
    return {"yaml": presets_io.preview_config_yaml_text(dict(body.get("config") or {}))}


def _apply_feature_flags(schema: dict[str, Any]) -> None:
    """Dynamically adjust the schema based on SystemConfig's experimental flags (affects
    only UI rendering).

    Automagic v2 isn't officially released yet: when the flag is off, automagic_variant
    is marked hidden (the value still passes through, CLI/yaml are unaffected).
    agreement_threshold's show_when includes automagic_variant==v2, so once variant is
    hidden it stays on the default v1 and naturally doesn't show -- nothing else to
    handle. Enabled by manually editing system.enable_automagic_v2 in
    studio_data/secrets.json; the Settings page deliberately doesn't render this toggle.
    """
    from ...infrastructure import secrets as secrets_infra

    if not secrets_infra.load().system.enable_automagic_v2:
        props = schema.get("properties", {})
        if "automagic_variant" in props:
            props["automagic_variant"]["hidden"] = True


@router.get("/api/presets")
def list_presets_endpoint() -> dict[str, Any]:
    return {"items": presets_io.list_presets()}


@router.get("/api/presets/{name}")
def get_preset(name: str, warnings: bool = False) -> dict[str, Any]:
    if warnings:
        config, dropped, defaulted = presets_io.read_preset_with_warnings(name)
        return {
            "config": config,
            "dropped_fields": dropped,
            "defaulted_fields": defaulted,
        }
    return presets_io.read_preset(name)


@router.put("/api/presets/{name}")
def put_preset(name: str, body: dict[str, Any]) -> dict[str, str]:
    path = presets_io.write_preset(name, body)
    return {"name": name, "path": str(path)}


@router.delete("/api/presets/{name}")
def delete_preset_endpoint(name: str) -> dict[str, str]:
    presets_io.delete_preset(name)
    return {"deleted": name}


@router.post("/api/presets/{name}/duplicate")
def duplicate_preset_endpoint(name: str, body: DuplicateRequest) -> dict[str, str]:
    path = presets_io.duplicate_preset(name, body.new_name)
    return {"name": body.new_name, "path": str(path)}


@router.get("/api/presets/{name}/download")
def download_preset(name: str) -> FileResponse:
    """End-to-end file I/O: returns the raw `studio_data/presets/{name}.yaml` file directly."""
    path = presets_io.preset_path(name)
    if not path.exists():
        raise NotFoundError(
            f'Preset "{name}" not found',
            code="preset.not_found", details={"name": name},
        )
    return FileResponse(path, media_type="application/yaml", filename=f"{name}.yaml")


@router.post("/api/presets/{name}/export")
def export_preset_to_data_exports(name: str, body: PresetExportBody) -> dict[str, Any]:
    """Fully validate the current preset form's parameters, then save to data_exports/."""
    DATA_EXPORTS.mkdir(parents=True, exist_ok=True)
    dest = _errors._unique_data_export_path(f"{name}.yaml", (".yaml", ".yml"))
    path = presets_io.write_preset(dest.stem, body.config, DATA_EXPORTS)
    return _errors._export_result(path)


@router.post("/api/presets/import-from-data-exports")
def import_preset_from_data_exports(body: PresetImportBody) -> dict[str, Any]:
    """Import a yaml/json preset from data_exports/ into the user's preset pool."""
    src = _errors._data_export_path(body.filename, (".yaml", ".yml", ".json"))
    if not src.exists():
        raise NotFoundError(
            f'File "{body.filename}" not found',
            code="file.not_found", details={"filename": body.filename},
        )
    if not src.is_file():
        raise ValidationError(
            "Select a file", code="file.required", http_status=400,
        )
    config, suggested = presets_io.parse_preset_bytes(src.read_bytes(), src.name)
    if presets_io.preset_path(suggested).exists():
        raise ConflictError(
            f'Preset "{suggested}" already exists',
            code="preset.exists",
            details={
                "name": suggested,
                "config": config,
                "suggested_name": suggested,
            },
        )
    path = presets_io.write_preset(suggested, config)
    return {"name": suggested, "path": str(path)}


@router.post("/api/presets/import-from-path")
def import_preset_from_path(body: PresetImportFromPathBody) -> dict[str, Any]:
    """Import a preset (yaml/yml/json) from an absolute path on the server."""
    from pathlib import Path
    src = Path(body.path)
    if not src.is_file():
        raise NotFoundError(
            f'File "{src.name}" not found or not readable',
            code="file.not_found",
            details={"filename": src.name, "path": body.path},
            http_status=400,
        )
    if src.suffix.lower() not in (".yaml", ".yml", ".json"):
        raise ValidationError(
            "Select a .yaml, .yml, or .json file",
            code="file.ext_invalid",
            details={"types": ".yaml, .yml, .json"},
            http_status=400,
        )
    config, suggested = presets_io.parse_preset_bytes(src.read_bytes(), src.name)
    if presets_io.preset_path(suggested).exists():
        raise ConflictError(
            f'Preset "{suggested}" already exists',
            code="preset.exists",
            details={
                "name": suggested,
                "config": config,
                "suggested_name": suggested,
            },
        )
    path = presets_io.write_preset(suggested, config)
    return {"name": suggested, "path": str(path)}


@router.post("/api/presets/import")
async def import_preset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a .yaml/.yml/.json upload -> parse + schema validate -> save to `suggested_name`.

    No conflict -> write_preset writes it directly, returns 200 `{name, path}`.
    Conflict (`suggested_name.yaml` already exists) -> 409 + structured detail
    `{message, config, suggested_name}`; the frontend's ImportConflictDialog lets the user
    choose overwrite / save as / cancel, and the chosen option goes through
    PUT /api/presets/{name} to finish saving.
    Parse/validation failure -> 400/422.
    """
    raw = await file.read()
    config, suggested = presets_io.parse_preset_bytes(raw, file.filename or "")
    if presets_io.preset_path(suggested).exists():
        raise ConflictError(
            f'Preset "{suggested}" already exists',
            code="preset.exists",
            details={
                "name": suggested,
                "config": config,
                "suggested_name": suggested,
            },
        )
    path = presets_io.write_preset(suggested, config)
    return {"name": suggested, "path": str(path)}


# The old /api/configs/* endpoints are kept as a 308 redirect (protects any external scripts).
# 308 preserves method + body, so PUT/POST/DELETE all forward transparently.
@router.api_route(
    "/api/configs",
    methods=["GET", "POST", "PUT", "DELETE"],
    include_in_schema=False,
)
def _configs_root_redirect(request: Request) -> RedirectResponse:
    qs = ("?" + request.url.query) if request.url.query else ""
    return RedirectResponse(url=f"/api/presets{qs}", status_code=308)


@router.api_route(
    "/api/configs/{rest:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
    include_in_schema=False,
)
def _configs_redirect(rest: str, request: Request) -> RedirectResponse:
    qs = ("?" + request.url.query) if request.url.query else ""
    return RedirectResponse(url=f"/api/presets/{rest}{qs}", status_code=308)
