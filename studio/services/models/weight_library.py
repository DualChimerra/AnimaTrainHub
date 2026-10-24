"""Weight discovery without importing torch or reading tensor data."""
import json
import os
import re
from functools import lru_cache
from pathlib import Path

CATEGORIES = {"checkpoints": "diffusion_models", "vae": "vae", "text_encoders": "text_encoders"}


def shard_files(path):
    match = re.match(r'^(.*)-(\d{5})-of-(\d{5})\.safetensors$', path.name)
    if not match:
        return [path]
    return [path.with_name(f'{match[1]}-{i:05d}-of-{match[3]}.safetensors') for i in range(1, int(match[3]) + 1)]


def weight_label(path):
    match = re.match(r'^(.*)-\d{5}-of-\d{5}\.safetensors$', path.name)
    if match:
        prefix = match[1]
        return path.parent.name if prefix in ("model", "diffusion_pytorch_model") else prefix
    return path.name


@lru_cache(maxsize=512)
def _header(filename, size, mtime):
    with open(filename, "rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 2 <= length <= min(size - 8, 32 * 1024 * 1024):
            raise ValueError("Invalid safetensors header")
        header = json.loads(stream.read(length))
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors metadata")
        return header


def identify(path, category):
    """Conservative structural hints; the runtime still validates every weight."""
    header = {}
    for shard in shard_files(path):
        if shard.is_file():
            st = shard.stat()
            header.update(_header(str(shard), st.st_size, st.st_mtime_ns))
    keys = list(header)
    has = lambda fragment: any(fragment in k for k in keys)
    if category == "checkpoints":
        if has("video_embedder.weight") and has("layers.0.proj."):
            return ["moss_image"]
        if has("txtfusion."):
            return ["krea2"]
        if has("llm_adapter.") or has("t_embedding_norm."):
            return ["anima"]
    elif category == "vae":
        if has("bn.running_mean") or has("decoder.conv_in.weight") and any(
            k.endswith("decoder.conv_in.weight") and v.get("shape", [0, 0])[1] == 32
            for k, v in header.items() if isinstance(v, dict)):
            return ["moss_image"]
        if any(k.endswith("decoder.conv1.weight") and len(v.get("shape", [])) == 5
               for k, v in header.items() if isinstance(v, dict)):
            return ["anima", "krea2"]
    else:
        config = path.parent / "config.json"
        if config.is_file():
            cfg = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                raise ValueError("Invalid encoder config.json")
            kind = cfg.get("model_type", "")
            if kind in ("qwen3_5", "qwen3_5_text"):
                return ["moss_image"]
            if kind in ("qwen3_vl", "qwen3_vl_text"):
                return ["krea2"]
            if kind == "qwen3":
                return ["anima"]
        if has("linear_attn.in_proj_qkv"):
            return ["moss_image"]
    return []


def build_library(root, cfg, sections, sources):
    from .families import FAMILY_ASSETS
    from .families.moss_image import asset_files
    result = {}
    active_family = cfg.active_family
    for section in sections.values():
        if isinstance(section, dict):
            for custom in section.get("custom", []):
                custom["name"] = weight_label(Path(custom["path"]))
    for category, subdir in CATEGORIES.items():
        folders = [str(root / subdir)]
        rows, errors, seen = [], [], set()

        def add(path, families, *, source=None, value=None, exists=None):
            p = Path(path)
            key = os.path.normcase(str(p.resolve()))
            if key in seen:
                return
            seen.add(key)
            if exists is None:
                exists = p.exists()
            selected = (cfg.selected if category == "checkpoints" else cfg.selected_te)
            current = (any(selected.get(f, "") == value for f in families) if category != "vae" else
                       (cfg.selected_vae_by_family.get("moss_image", "") == value if families == ["moss_image"] else cfg.selected_vae == value))
            files = shard_files(p) if p.is_file() else list(p.glob("*.safetensors")) if p.is_dir() else []
            rows.append({"path": str(p), "label": weight_label(p), "families": families,
                "value": value if value is not None else str(p), "exists": bool(exists),
                "is_current": active_family in families and bool(source.get("is_current", current) if source else current),
                "size": sum(f.stat().st_size for f in files if f.is_file()),
                "purpose": source.get("extra", {}).get("purpose", "") if source else "",
                "download_id": source.get("download_id") if source else None,
                "download_variant": source.get("download_variant") if source else None,
                "status_key": source.get("status_key") if source else None,
                "domain": ("vae" if category == "vae" else families[0] + ("_te" if category == "text_encoders" else "")) if families else "",
                "candidate": source.get("candidate") if source else None})

        for family, assets in FAMILY_ASSETS.items():
            if category == "checkpoints":
                main = sections[f"{family}_main"]
                presets = {v["variant"]: v for v in main["variants"]}
                for row in sources.get(family, []):
                    variant = presets.get(row["value"])
                    add(variant["target_path"] if variant else row["value"], [family], source=row,
                        value=row["value"], exists=row["exists"])
            elif category == "text_encoders":
                presets = {v["value"]: v for v in assets.text_encoder_presets(root)}
                for row in sources.get(f"{family}_te", []):
                    preset = presets.get(row["value"])
                    add(preset["path"] if preset else row["value"], [family], source=row,
                        value=row["value"], exists=row["exists"])
        if category == "vae":
            for row in sources.get("vae", []):
                add(row["description"], ["anima", "krea2"], source=row, value=row["value"], exists=row["exists"])
            _, p = asset_files(root, "moss_image_vae")[0]
            add(p, ["moss_image"], source={"download_id": "moss_image_vae", "status_key": "moss_image_vae"}, value="")
            custom = cfg.selected_vae_by_family.get("moss_image")
            if custom:
                add(custom, ["moss_image"], value=custom)

        for folder in folders:
            directory = Path(folder).expanduser()
            if not directory.is_dir():
                if directory != root / subdir:
                    errors.append(f"Folder unavailable: {directory}")
                continue
            for base_name, dirs, files in os.walk(directory, followlinks=False):
                base = Path(base_name)
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for name in sorted(files):
                    shard = re.search(r'-(\d{5})-of-\d{5}\.safetensors$', name)
                    if not name.endswith(".safetensors") or (shard and int(shard[1]) != 1):
                        continue
                    p = base / name
                    try:
                        families = identify(p, category)
                        target = base if category == "text_encoders" and (base / "config.json").is_file() else p
                        # Official encoder directories also cover all their shards.
                        if os.path.normcase(str(target.resolve())) in seen:
                            continue
                        complete = all(f.is_file() for f in shard_files(p))
                        add(target, families, value=str(target), exists=complete)
                        if category == "checkpoints" and families:
                            customs = sections[f"{families[0]}_main"]["custom"]
                            if not any(c["path"] == str(p) for c in customs):
                                customs.append({"path": str(p), "name": weight_label(p), "exists": complete})
                    except (OSError, ValueError, TypeError, IndexError) as exc:
                        errors.append(f"{p.name}: {exc}")
        # A shared VAE or a preset and its discovered alias must not produce
        # several checked radio buttons in the same category.
        current_seen = False
        for row in rows:
            row["is_current"] = row["is_current"] and not current_seen
            current_seen = current_seen or row["is_current"]
        result[category] = {"folders": folders, "rows": rows, "errors": errors}
    return result


def select_weight(category, path, catalog):
    from ... import secrets
    row = next((r for r in catalog["weight_library"].get(category, {}).get("rows", []) if r["path"] == path), None)
    if not row or not row["exists"] or not row["families"]:
        raise ValueError("Select an available, recognized model from the scanned folders")
    value = row["value"]
    if category == "checkpoints":
        patch = {"selected": {f: value for f in row["families"]}}
    elif category == "text_encoders":
        patch = {"selected_te": {f: value for f in row["families"]}}
    else:
        patch = ({"selected_vae_by_family": {"moss_image": value}}
                 if row["families"] == ["moss_image"] else {"selected_vae": value})
    active = secrets.load().models.active_family
    patch["active_family"] = active if active in row["families"] else row["families"][0]
    secrets.update({"models": patch})
