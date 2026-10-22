"""Edit LoRA datasets: examples of input image(s) + optional text -> result image.

Layout inside a version (mirrors diffusion-pipe's ``path`` / ``control_paths`` and
ai-toolkit's ``folder_path`` / ``control_path`` list)::

    train/<folder>/<stem>.<ext>      result (what the model learns to produce)
    train/<folder>/<stem>.txt        text prompt / instruction (may be empty)
    control/<folder>/<stem>.<ext>    input image  -> reference frame 1
    control2/<folder>/<stem>.<ext>   image prompt -> reference frame 2 (optional)
    edit.json                        {"recipe": ...} -- which slots the UI asks for

Files of one example share folder + stem. The runtime (training.dataset.ImageDataset
with control dirs) also accepts flat control folders; this module writes mirrored ones.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from .scan import IMAGE_EXTS

DEFAULT_FOLDER = "1_data"
CAPTION_EXT = ".txt"
RECIPE_FILE = "edit.json"
_SAFE_NAME = re.compile(r"[^\w.\- ]+", re.UNICODE)

Role = Literal["target", "source", "source2"]
Recipe = Literal["text", "image"]
#: text  = image + text instruction -> result
#: image = image + image prompt (+ optional text) -> result
RECIPES: tuple[str, ...] = ("text", "image")


class EditPairsError(ValueError):
    pass


@dataclass(frozen=True)
class PairDirs:
    train: Path
    control: Path
    control2: Optional[Path] = None
    version: Optional[Path] = None

    def root(self, role: Role) -> Path:
        if role == "target":
            return self.train
        if role == "source2":
            if self.control2 is None:
                raise EditPairsError("This dataset has no second input folder")
            return self.control2
        return self.control


def pair_dirs(version_dir: Path) -> PairDirs:
    return PairDirs(version_dir / "train", version_dir / "control", version_dir / "control2", version_dir)


def read_recipe(dirs: PairDirs) -> str:
    if dirs.version is not None:
        try:
            recipe = json.loads((dirs.version / RECIPE_FILE).read_text(encoding="utf-8")).get("recipe")
            if recipe in RECIPES:
                return recipe
        except (OSError, ValueError, AttributeError):
            pass
    # Existing data decides when nothing was chosen yet.
    return "image" if dirs.control2 is not None and _has_images(dirs.control2) else "text"


def write_recipe(dirs: PairDirs, recipe: str) -> str:
    if recipe not in RECIPES:
        raise EditPairsError(f"Unknown recipe: {recipe}")
    if dirs.version is None:
        raise EditPairsError("No version folder")
    (dirs.version / RECIPE_FILE).write_text(json.dumps({"recipe": recipe}), encoding="utf-8")
    return recipe


def _has_images(root: Path) -> bool:
    return root.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in root.rglob("*") if p.is_file())


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _folders(root: Optional[Path]) -> list[str]:
    if root is None or not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _check_folder(folder: str) -> str:
    folder = (folder or DEFAULT_FOLDER).strip()
    if not folder or folder in {".", ".."} or "/" in folder or "\\" in folder:
        raise EditPairsError(f"Invalid folder name: {folder!r}")
    return folder


def safe_filename(name: str) -> str:
    name = Path(str(name or "")).name.strip()
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = _SAFE_NAME.sub("_", stem).strip(" .")
    if not stem:
        raise EditPairsError(f"Invalid file name: {name!r}")
    return f"{stem}{ext}"


def find_image(folder: Path, stem: str) -> Optional[Path]:
    for path in _images(folder):
        if path.stem == stem:
            return path
    return None


def find_input(dirs: PairDirs, role: Role, folder: str, stem: str) -> Optional[Path]:
    """Input image: mirrored subfolder first, then the flat folder -- the runtime's order."""
    root = dirs.control2 if role == "source2" else dirs.control
    if root is None:
        return None
    return find_image(root / folder, stem) or find_image(root, stem)


def find_source(dirs: PairDirs, folder: str, stem: str) -> Optional[Path]:
    return find_input(dirs, "source", folder, stem)


def caption_path(dirs: PairDirs, folder: str, stem: str) -> Path:
    return dirs.train / folder / f"{stem}{CAPTION_EXT}"


def read_caption_at(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def read_caption(target: Path) -> str:
    return read_caption_at(target.with_suffix(CAPTION_EXT))


def _image_size(path: Optional[Path]) -> Optional[list[int]]:
    if path is None:
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            return [image.width, image.height]
    except Exception:
        return None


def _rel_folder(root: Path, path: Path) -> str:
    return path.parent.relative_to(root).as_posix().strip(".")


def list_pairs(dirs: PairDirs, *, with_sizes: bool = True) -> dict[str, Any]:
    """Every example (any file with a given folder + stem) with whatever halves it has."""
    recipe = read_recipe(dirs)
    keys: dict[tuple[str, str], None] = {}
    for root in (dirs.train, dirs.control, dirs.control2):
        for folder in _folders(root):
            for image in _images(root / folder):
                keys.setdefault((folder, image.stem), None)
        # Captions alone also start an example (text typed before the images arrive).
    for folder in _folders(dirs.train):
        for txt in (dirs.train / folder).glob(f"*{CAPTION_EXT}"):
            keys.setdefault((folder, txt.stem), None)

    items: list[dict[str, Any]] = []
    for folder, stem in sorted(keys):
        target = find_image(dirs.train / folder, stem)
        source = find_input(dirs, "source", folder, stem)
        source2 = find_input(dirs, "source2", folder, stem)
        files = [p for p in (target, source, source2) if p is not None]
        item: dict[str, Any] = {
            "folder": folder,
            "stem": stem,
            "target": target.name if target else None,
            "source": source.name if source else None,
            "source_folder": _rel_folder(dirs.control, source) if source else None,
            "source2": source2.name if source2 else None,
            "source2_folder": _rel_folder(dirs.control2, source2) if source2 and dirs.control2 else None,
            "caption": read_caption_at(caption_path(dirs, folder, stem)),
            "mtime": max((p.stat().st_mtime for p in files), default=0),
        }
        if with_sizes:
            item["target_size"] = _image_size(target)
            item["source_size"] = _image_size(source)
            item["source2_size"] = _image_size(source2)
        item["missing"] = _missing(item, recipe)
        items.append(item)
    return {
        "recipe": recipe,
        "items": items,
        "folders": sorted({i["folder"] for i in items}) or [DEFAULT_FOLDER],
        "count": len(items),
        "complete": sum(1 for i in items if not i["missing"]),
        "incomplete": sum(1 for i in items if i["missing"]),
    }


def _missing(item: dict[str, Any], recipe: str) -> list[str]:
    """Slots this example still needs for the recipe: target / source / source2 / caption."""
    out = []
    if not item["target"]:
        out.append("target")
    if not item["source"]:
        out.append("source")
    if recipe == "image" and not item["source2"]:
        out.append("source2")
    if recipe == "text" and not item["caption"]:
        out.append("caption")
    return out


def check_ready(dirs: PairDirs) -> tuple[bool, str]:
    """Phase gate: at least one example, and every example has what its recipe needs."""
    summary = list_pairs(dirs, with_sizes=False)
    if summary["count"] < 1:
        return False, "Add at least one example first"
    bad = [i for i in summary["items"] if i["missing"]]
    if bad:
        need = sorted({m for i in bad for m in i["missing"]})
        return False, f"{len(bad)} example(s) are incomplete (missing: {', '.join(need)})"
    if summary["recipe"] != "image" and dirs.control2 is not None and _has_images(dirs.control2):
        return False, "Second input images exist but the tool takes one input; remove them or switch the tool type"
    # Text is optional for some recipes; the trainer skips images without a caption file.
    for item in summary["items"]:
        path = caption_path(dirs, item["folder"], item["stem"])
        if not path.exists():
            path.write_text("", encoding="utf-8")
    return True, ""


def _remove_same_stem(folder: Path, stem: str, keep: Path) -> None:
    for other in _images(folder):
        if other.stem == stem and other != keep:
            other.unlink()


def _drop_latent_cache(folder: Path, stem: str) -> None:
    for npz in folder.glob(f"{stem}.*npz"):
        if npz.name == f"{stem}.npz" or re.fullmatch(rf"{re.escape(stem)}\.r\d+\.npz", npz.name):
            npz.unlink(missing_ok=True)


def save_file(dirs: PairDirs, role: str, folder: str, filename: str, data: bytes) -> dict[str, Any]:
    """Store one uploaded file in the role's folder. Results also accept a .txt caption."""
    if role not in ("target", "source", "source2"):
        raise EditPairsError(f"Unknown role: {role}")
    folder = _check_folder(folder)
    name = safe_filename(filename)
    ext = Path(name).suffix.lower()
    dest_dir = dirs.root(role) / folder  # type: ignore[arg-type]
    dest_dir.mkdir(parents=True, exist_ok=True)
    if ext == CAPTION_EXT and role == "target":
        (dest_dir / name).write_text(data.decode("utf-8", errors="replace").strip() + "\n", encoding="utf-8")
        return {"kind": "caption", "name": name}
    if ext not in IMAGE_EXTS:
        raise EditPairsError(f"Not an image: {filename}")
    dest = dest_dir / name
    dest.write_bytes(data)
    _remove_same_stem(dest_dir, dest.stem, dest)
    # Any changed half invalidates the example's cached latents (they live by the result).
    _drop_latent_cache(dirs.train / folder, dest.stem)
    return {"kind": role, "name": name}


def set_caption(dirs: PairDirs, folder: str, stem: str, caption: str) -> str:
    folder = _check_folder(folder)
    stem = Path(safe_filename(f"{stem}.x")).stem
    text = str(caption or "").strip()
    path = caption_path(dirs, folder, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n" if text else "", encoding="utf-8")
    return text


def batch_instruction(dirs: PairDirs, instruction: str,
                      mode: Literal["replace", "prepend", "append"]) -> int:
    """Write one text into every example's caption (prepend keeps existing tags after it)."""
    instruction = str(instruction or "").strip()
    count = 0
    for item in list_pairs(dirs, with_sizes=False)["items"]:
        path = caption_path(dirs, item["folder"], item["stem"])
        current = item["caption"]
        if mode == "replace" or not current:
            text = instruction
        elif mode == "prepend":
            text = current if current.startswith(instruction) else f"{instruction}\n{current}"
        else:
            text = current if current.endswith(instruction) else f"{current}\n{instruction}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n" if text else "", encoding="utf-8")
        count += 1
    return count


def delete_pair(dirs: PairDirs, folder: str, stem: str) -> None:
    folder = _check_folder(folder)
    found = False
    for role in ("source", "source2"):
        path = find_input(dirs, role, folder, stem)  # type: ignore[arg-type]
        if path is not None:
            path.unlink()
            found = True
    target = find_image(dirs.train / folder, stem)
    if target is not None:
        target.unlink()
        found = True
    for sidecar in (caption_path(dirs, folder, stem), dirs.train / folder / f"{stem}.json"):
        if sidecar.exists():
            sidecar.unlink()
            found = True
    _drop_latent_cache(dirs.train / folder, stem)
    if not found:
        raise EditPairsError(f"Example not found: {folder}/{stem}")


def delete_slot(dirs: PairDirs, role: str, folder: str, stem: str) -> None:
    """Remove one image of an example (to replace it, or to drop a stray second input)."""
    folder = _check_folder(folder)
    path = find_image(dirs.train / folder, stem) if role == "target" else find_input(dirs, role, folder, stem)  # type: ignore[arg-type]
    if path is None:
        raise EditPairsError(f"Nothing to remove: {role} of {folder}/{stem}")
    path.unlink()
    _drop_latent_cache(dirs.train / folder, stem)
