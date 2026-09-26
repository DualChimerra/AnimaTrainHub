"""
Caption processing utilities
- Read JSON tag files
- Normalize format (implemented in studio.services.caption_format, this module only re-exports)
- Categorized shuffle
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Optional

# Support running `python utils/caption_utils.py` directly as a script: by default python
# only adds the script's own directory to sys.path, so studio.* wouldn't be visible. Inject
# the repo root manually once; when imported as a module sys.path already contains the repo
# root, so setdefault-style insertion here is sufficient.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The authoritative implementation of normalize_caption_json lives in
# studio.services.caption_format -- PR #18 review found two subtly different
# implementations (dedup / appearance-merge strategy differed), consolidated back to one source.
from studio.services.tagging.caption_format import normalize_caption_json  # noqa: E402, F401


def load_caption_json(json_path: Path) -> dict | None:
    """Read a JSON tag file"""
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def dedupe_list(tags: list) -> list:
    """Deduplicate while preserving order"""
    seen = set()
    result = []
    for tag in tags:
        tag_lower = tag.lower().strip()
        if tag_lower and tag_lower not in seen:
            seen.add(tag_lower)
            result.append(tag.strip())
    return result


def build_caption_from_json(
    json_data: dict,
    shuffle_appearance: bool = True,
    shuffle_tags: bool = True,
    shuffle_environment: bool = True,
    tag_dropout: float = 0.0,
) -> str:
    """
    Build a caption from normalized JSON

    Args:
        json_data: normalized JSON data
        shuffle_appearance: whether to shuffle within appearance
        shuffle_tags: whether to shuffle within tags
        shuffle_environment: whether to shuffle within environment
        tag_dropout: drop probability for appearance/tags/environment (0-1)

    Returns:
        The final caption string
    """
    tags_dict = json_data.get("tags", {})
    meta = json_data.get("meta") if isinstance(json_data.get("meta"), dict) else {}

    # Fixed part (never shuffled, never dropped)
    parts = []

    # 0. trigger word (meta.trigger) -- injected by Studio during tagging, always first,
    # never participates in shuffle / dropout; equivalent to the .txt mode's keep_tokens=1 protection.
    trigger = (meta.get("trigger") or "").strip() if isinstance(meta.get("trigger"), str) else ""
    if trigger:
        parts.append(trigger)

    # 1. quality
    quality = tags_dict.get("quality", [])
    if quality:
        parts.extend(quality)

    # 2. count
    count = tags_dict.get("count", "")
    if count:
        parts.append(count)

    # 3. character
    character = tags_dict.get("character", "")
    if character:
        parts.append(character)

    # 4. series
    series = tags_dict.get("series", "")
    if series:
        parts.append(series)

    # 5. artist
    artist = tags_dict.get("artist", "")
    if artist:
        parts.append(artist)

    # Variable part (may be shuffled, may be dropped)
    def process_tag_list(tag_list: list, shuffle: bool, dropout: float) -> list:
        """Process a tag list: shuffle + dropout"""
        if not tag_list:
            return []

        result = list(tag_list)  # copy

        # Shuffle
        if shuffle:
            random.shuffle(result)

        # Dropout
        if dropout > 0:
            result = [t for t in result if random.random() > dropout]
            # Ensure at least one is kept
            if not result and tag_list:
                result = [random.choice(tag_list)]

        return result

    # 6. appearance
    appearance = tags_dict.get("appearance", [])
    parts.extend(process_tag_list(appearance, shuffle_appearance, tag_dropout))

    # 7. tags
    tags = tags_dict.get("tags", [])
    parts.extend(process_tag_list(tags, shuffle_tags, tag_dropout))

    # 8. environment
    environment = tags_dict.get("environment", [])
    parts.extend(process_tag_list(environment, shuffle_environment, tag_dropout))

    # Deduplicate
    parts = dedupe_list(parts)

    # Build the caption
    caption = ", ".join(parts)

    # 9. nl (natural language description)
    nl = tags_dict.get("nl", "")
    if nl:
        caption = f"{caption}. {nl}"

    return caption


def load_and_build_caption(
    json_path: Path,
    shuffle: bool = True,
    tag_dropout: float = 0.0,
) -> str | None:
    """
    Convenience function: load from a JSON file and build the caption

    Args:
        json_path: path to the JSON file
        shuffle: whether to shuffle by category
        tag_dropout: dropout probability

    Returns:
        The caption string, or None (if reading failed)
    """
    raw_json = load_caption_json(json_path)
    if raw_json is None:
        return None

    # Check whether it's already in standard format: tags must be a dict (categorized form)
    # plus a meta; otherwise always run normalize (including the simplified form Studio writes,
    # {"tags": [list], "meta": {trigger}} -- normalize moves the tags list into the tags.tags
    # field and keeps meta).
    if (
        isinstance(raw_json.get("tags"), dict)
        and isinstance(raw_json.get("meta"), dict)
    ):
        normalized = raw_json
    else:
        normalized = normalize_caption_json(raw_json)

    return build_caption_from_json(
        normalized,
        shuffle_appearance=shuffle,
        shuffle_tags=shuffle,
        shuffle_environment=shuffle,
        tag_dropout=tag_dropout,
    )


# ============================================================================
# Batch conversion tool
# ============================================================================

def convert_json_to_standard(input_path: Path, output_path: Path = None) -> dict:
    """
    Convert a single JSON file to standard format

    Args:
        input_path: input JSON path
        output_path: output path (optional, defaults to overwriting the original file)

    Returns:
        The normalized JSON data
    """
    raw_json = load_caption_json(input_path)
    if raw_json is None:
        raise ValueError(f"Cannot load JSON: {input_path}")

    normalized = normalize_caption_json(raw_json)

    if output_path is None:
        output_path = input_path

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    return normalized


def batch_convert_json(
    data_dir: Path,
    in_place: bool = True,
    output_suffix: str = "_std",
) -> int:
    """
    Batch-convert all JSON files under a directory to standard format

    Args:
        data_dir: data directory
        in_place: whether to overwrite in place
        output_suffix: output suffix when not overwriting in place

    Returns:
        Number of files converted
    """
    count = 0
    for json_path in data_dir.rglob("*.json"):
        try:
            raw_json = load_caption_json(json_path)
            if raw_json is None:
                continue

            # Skip files already in standard format -- tags must be a categorized dict (same
            # check as load_and_build_caption, #345); a flat {"tags": [...]} needs conversion
            # and must not be mistaken for already-standard.
            if (
                isinstance(raw_json.get("tags"), dict)
                and isinstance(raw_json.get("meta"), dict)
            ):
                continue

            normalized = normalize_caption_json(raw_json)

            if in_place:
                output_path = json_path
            else:
                output_path = json_path.with_stem(json_path.stem + output_suffix)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(normalized, f, ensure_ascii=False, indent=2)

            count += 1
        except Exception as e:
            print(f"Error converting {json_path}: {e}")

    return count


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Caption JSON tool")
    parser.add_argument("action", choices=["convert", "test"], help="Action type")
    parser.add_argument("--dir", type=str, help="Data directory")
    parser.add_argument("--file", type=str, help="Single file")
    parser.add_argument("--in-place", action="store_true", help="Overwrite in place")
    args = parser.parse_args()

    if args.action == "convert":
        if args.file:
            result = convert_json_to_standard(Path(args.file))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.dir:
            count = batch_convert_json(Path(args.dir), in_place=args.in_place)
            print(f"Converted {count} files")
        else:
            print("Please specify --dir or --file")

    elif args.action == "test":
        if args.file:
            caption = load_and_build_caption(Path(args.file), shuffle=True)
            print(caption)
        else:
            print("Please specify --file")
