"""Training mask sidecar (`train/{folder}/{stem}.mask`) read/write + preprocessing transform propagation.

Design: docs/design/preprocess-inpaint-mask-design.md sections 2 / 7 / 9.

- The mask lives in the **same directory, same stem** as the training image, always with a `.mask` suffix (contents are grayscale PNG bytes).
  Structured like the .txt / .json caption sidecars: the suffix isn't in IMAGE_EXTS, so every image scan point
  (flat / recursive) naturally never treats it as a training image -- no exemption rule needed.
- The stem has no extension -- when crop / inpaint unify an X.jpg output to X.png, the mask path stays the same,
  so it's naturally immune to output renaming; the same-stem sidecar family (.txt/.json/.mask) shares one mental model
  for delete / export / transform propagation.
- Grayscale L semantics: 255 = learn normally, 0 = don't learn, in-between = partial weight (the trainer uses value/255 as the loss
  weight). No mask file = all 255; "clear mask" = delete the file.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from studio.domain.errors import ValidationError

MASK_SUFFIX = ".mask"


def mask_path_for(train_dir: Path, rel_name: str) -> Path:
    """`1_data/X.jpg` -> `{train_dir}/1_data/X.mask`.

    The write endpoint runs `_validate_rel_name` first (strict two segments); the delete / query path can also be reached
    via manifest mutation using the **legacy flat name** (no folder prefix, ADR 0004 compat
    data) -- a flat name maps to `{train_dir}/{stem}.mask`; if the file doesn't exist,
    the caller no-ops instead of crashing.
    """
    if "/" in rel_name:
        folder, filename = rel_name.split("/", 1)
        return train_dir / folder / f"{Path(filename).stem}{MASK_SUFFIX}"
    return train_dir / f"{Path(rel_name).stem}{MASK_SUFFIX}"


def write_mask(
    train_dir: Path,
    rel_name: str,
    data: bytes,
    *,
    expected_size: tuple[int, int],
) -> dict[str, Any]:
    """Writes the mask (grayscale PNG bytes, tmp + atomic replace).

    Size must match the corresponding training image's current size exactly -- the mask is a pixel-aligned data plane,
    a mismatch means the frontend export target is misaligned; reject outright.
    """
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # PIL raises inconsistent types on decode failure, normalize to 400
        raise ValidationError(
            "Uploaded mask is not a valid image file",
            code="preprocess.mask_image_invalid",
            details={"name": rel_name}, http_status=400,
        ) from exc
    if img.size != tuple(expected_size):
        raise ValidationError(
            "Mask size does not match the source image",
            code="preprocess.mask_size_mismatch",
            details={
                "name": rel_name,
                "expected": [expected_size[0], expected_size[1]],
                "got": [img.size[0], img.size[1]],
            },
            http_status=400,
        )
    if img.mode != "L":
        img = img.convert("L")

    out = mask_path_for(train_dir, rel_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    img.save(tmp, format="PNG", optimize=False)
    os.replace(tmp, out)
    st = out.stat()
    return {"name": rel_name, "mtime": st.st_mtime, "size": st.st_size}


def delete_mask(train_dir: Path, rel_name: str) -> bool:
    """Deletes the mask file. Returns whether it was actually deleted (returns False if it didn't exist, doesn't raise)."""
    p = mask_path_for(train_dir, rel_name)
    if not p.is_file():
        return False
    try:
        p.unlink()
    except OSError:
        return False
    return True


def delete_masks_for(train_dir: Path, rel_names: Iterable[str]) -> int:
    """Batch delete (sidecar cleanup that follows restore / dedup / removing a training image)."""
    n = 0
    for rel in rel_names:
        if delete_mask(train_dir, rel):
            n += 1
    return n


def crop_mask_like(
    train_dir: Path,
    src_rel: str,
    boxes: list[tuple[int, int, int, int]],
    out_rels: list[str],
) -> None:
    """Crop propagation: crops the mask with the same pixel box as the image, fanning out to the
    mask paths of out_rels; deletes the source mask if it isn't in the output set.

    No mask on the source image -> no-op. Mask size mismatched with the source image (missed by an external edit) -> delete the source mask
    (the geometry is already misaligned; keeping it would only pollute training; the trainer's size validation is the last line of defense, so we proactively clean up here).
    """
    from PIL import Image

    src_mask = mask_path_for(train_dir, src_rel)
    if not src_mask.is_file():
        return
    with Image.open(src_mask) as raw:
        raw.load()
        mask_img = raw.convert("L") if raw.mode != "L" else raw.copy()

    out_paths: list[Path] = []
    for box, out_rel in zip(boxes, out_rels):
        piece = mask_img.crop(box)
        out = mask_path_for(train_dir, out_rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        piece.save(tmp, format="PNG", optimize=False)
        os.replace(tmp, out)
        out_paths.append(out)

    if src_mask not in out_paths:
        try:
            src_mask.unlink()
        except OSError:
            pass


def resize_mask_like(
    train_dir: Path, rel_name: str, size: tuple[int, int],
) -> None:
    """Upscale propagation: NEAREST-resizes the mask to the new size (doesn't go through RealESRGAN,
    to avoid the grayscale values being polluted by the super-resolution model). No mask -> no-op."""
    from PIL import Image

    p = mask_path_for(train_dir, rel_name)
    if not p.is_file():
        return
    with Image.open(p) as raw:
        raw.load()
        img = raw.convert("L") if raw.mode != "L" else raw.copy()
    if img.size == tuple(size):
        return
    img = img.resize(size, Image.NEAREST)
    tmp = p.with_suffix(p.suffix + ".tmp")
    img.save(tmp, format="PNG", optimize=False)
    os.replace(tmp, p)


def mask_file(train_dir: Path, rel_name: str) -> Optional[Path]:
    """Mask file path (returns None if it doesn't exist). Used by the GET endpoint."""
    p = mask_path_for(train_dir, rel_name)
    return p if p.is_file() else None


def mask_stat(train_dir: Path, rel_name: str) -> Optional[dict[str, Any]]:
    """Returns {mtime, size} when the mask exists, else None (used by the workspace list's has_mask)."""
    p = mask_path_for(train_dir, rel_name)
    try:
        st = p.stat()
    except OSError:
        return None
    return {"mtime": st.st_mtime, "size": st.st_size}
