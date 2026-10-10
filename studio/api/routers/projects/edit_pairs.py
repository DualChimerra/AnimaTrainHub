"""Edit LoRA examples: list / recipe / upload / text / delete, recommended config (edit projects only)."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from ._shared import _publish_version_state, _version_dir_or_404
from ....domain.errors import ValidationError
from ....services.dataset import edit_pairs
from ....services import presets
from ....services.projects import projects, versions

router = APIRouter()


class PairCaption(BaseModel):
    caption: str = ""


class PairInstruction(BaseModel):
    instruction: str
    mode: Literal["replace", "prepend", "append"] = "prepend"


class PairRef(BaseModel):
    folder: str = edit_pairs.DEFAULT_FOLDER
    stem: str


class SlotRef(BaseModel):
    role: Literal["target", "source", "source2"]
    folder: str = edit_pairs.DEFAULT_FOLDER
    stem: str


class PairDelete(BaseModel):
    items: list[PairRef] = Field(default_factory=list)
    slots: list[SlotRef] = Field(default_factory=list)


class PairRecipe(BaseModel):
    recipe: Literal["text", "image"]


def _dirs(pid: int, vid: int, *, write: bool = False):
    p, v, vdir = _version_dir_or_404(pid, vid)
    if not projects.is_edit_project(p):
        raise ValidationError("This project is not an edit project", code="edit.not_edit_project")
    if write and versions.get_status(v) == "training":
        raise ValidationError(
            "This version is training; pairs can't change until it finishes",
            code="edit.version_training", http_status=409,
        )
    return p, v, edit_pairs.pair_dirs(vdir)


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except edit_pairs.EditPairsError as exc:
        raise ValidationError(str(exc), code="edit.invalid") from exc


@router.get("/api/projects/{pid}/versions/{vid}/pairs")
def list_pairs_endpoint(pid: int, vid: int) -> dict[str, Any]:
    _p, _v, dirs = _dirs(pid, vid)
    return edit_pairs.list_pairs(dirs)


@router.post("/api/projects/{pid}/versions/{vid}/pairs/upload")
async def upload_pairs_endpoint(
    pid: int, vid: int,
    role: Literal["source", "source2", "target"] = Form(...),
    folder: str = Form(edit_pairs.DEFAULT_FOLDER),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    _p, v, dirs = _dirs(pid, vid, write=True)
    saved, errors = [], []
    for upload in files:
        data = await upload.read()
        try:
            saved.append(edit_pairs.save_file(dirs, role, folder, upload.filename or "", data))
        except edit_pairs.EditPairsError as exc:
            errors.append(str(exc))
    _publish_version_state(v)
    return {"saved": saved, "errors": errors}


@router.put("/api/projects/{pid}/versions/{vid}/pairs/{folder}/{stem}/caption")
def put_pair_caption_endpoint(pid: int, vid: int, folder: str, stem: str, body: PairCaption) -> dict[str, Any]:
    _p, _v, dirs = _dirs(pid, vid, write=True)
    return {"caption": _guard(edit_pairs.set_caption, dirs, folder, stem, body.caption)}


@router.post("/api/projects/{pid}/versions/{vid}/pairs/instruction")
def batch_instruction_endpoint(pid: int, vid: int, body: PairInstruction) -> dict[str, Any]:
    _p, v, dirs = _dirs(pid, vid, write=True)
    if not body.instruction.strip():
        raise ValidationError("The instruction is empty", code="edit.instruction_empty")
    count = edit_pairs.batch_instruction(dirs, body.instruction, body.mode)
    _publish_version_state(v)
    return {"updated": count}


@router.post("/api/projects/{pid}/versions/{vid}/pairs/delete")
def delete_pairs_endpoint(pid: int, vid: int, body: PairDelete) -> dict[str, Any]:
    _p, v, dirs = _dirs(pid, vid, write=True)
    for ref in body.items:
        _guard(edit_pairs.delete_pair, dirs, ref.folder, ref.stem)
    for slot in body.slots:
        _guard(edit_pairs.delete_slot, dirs, slot.role, slot.folder, slot.stem)
    _publish_version_state(v)
    return {"deleted": len(body.items) + len(body.slots)}


@router.put("/api/projects/{pid}/versions/{vid}/pairs/recipe")
def put_recipe_endpoint(pid: int, vid: int, body: PairRecipe) -> dict[str, Any]:
    _p, v, dirs = _dirs(pid, vid, write=True)
    recipe = _guard(edit_pairs.write_recipe, dirs, body.recipe)
    _publish_version_state(v)
    return {"recipe": recipe}


@router.post("/api/projects/{pid}/versions/{vid}/pairs/recommended-config")
def recommended_config_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """Rewrite the version's config with the recommended edit settings."""
    p, v, _dirs_ = _dirs(pid, vid, write=True)
    config = presets.create_edit_version_config(p, v)
    _publish_version_state(v)
    return {"config": config}
