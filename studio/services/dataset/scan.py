"""Dataset directory scanning: recognizes the Kohya-style N_xxx prefix, counts samples and caption types.

No caching: every endpoint call rescans from scratch. Dataset directories are
usually < a few thousand images, so scanning is fast.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# The whole-pipeline image format whitelist — upload / download / curation /
# tag / reg / training all reference this set.
# Keep in sync with anima_train.py:EXTS (the trainer is a standalone script
# that doesn't import studio).
# Removed .jxl: PIL 12 doesn't register .jxl; it needs pillow-jxl-plugin to
# decode, and it's not seen in the booru ecosystem anyway.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
KOHYA_PREFIX = re.compile(r"^(\d+)_(.+)$")


def parse_repeat(folder_name: str) -> tuple[int, str]:
    """`5_concept` -> (5, 'concept'); returns (1, name) if there's no prefix."""
    m = KOHYA_PREFIX.match(folder_name)
    if m:
        return int(m.group(1)), m.group(2)
    return 1, folder_name


def caption_kind(image_path: Path) -> str:
    """Same-name .json > .txt > 'none'."""
    if image_path.with_suffix(".json").exists():
        return "json"
    if image_path.with_suffix(".txt").exists():
        return "txt"
    return "none"


def scan_folder(folder: Path, sample_limit: int = 4) -> dict[str, Any]:
    """Tally sample count and caption distribution for a single folder."""
    repeat, label = parse_repeat(folder.name)
    counts = {"json": 0, "txt": 0, "none": 0}
    samples: list[str] = []
    image_count = 0

    if folder.is_dir():
        for entry in sorted(folder.iterdir()):
            if entry.suffix.lower() not in IMAGE_EXTS:
                continue
            image_count += 1
            counts[caption_kind(entry)] += 1
            if len(samples) < sample_limit:
                samples.append(entry.name)

    return {
        "name": folder.name,
        "label": label,
        "repeat": repeat,
        "image_count": image_count,
        "caption_types": counts,
        "samples": samples,
        "path": str(folder),
    }


def scan_dataset_root(root: Path) -> dict[str, Any]:
    """Scan the dataset root directory, returning stats for each subdirectory; loose images in the root also count as one virtual entry."""
    if not root.exists() or not root.is_dir():
        return {"root": str(root), "exists": False, "folders": []}

    folders: list[dict[str, Any]] = []
    # subdirectories
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            folders.append(scan_folder(entry))

    # Images placed directly in the root (images with no prefix subdirectory count as repeat=1)
    root_loose = scan_folder(root, sample_limit=4)
    if root_loose["image_count"] > 0:
        root_loose["name"] = "(root directory)"
        root_loose["label"] = "(loose)"
        root_loose["repeat"] = 1
        folders.insert(0, root_loose)

    total_images = sum(f["image_count"] for f in folders)
    weighted = sum(f["image_count"] * f["repeat"] for f in folders)
    return {
        "root": str(root),
        "exists": True,
        "folders": folders,
        "total_images": total_images,
        "weighted_steps_per_epoch": weighted,
    }
