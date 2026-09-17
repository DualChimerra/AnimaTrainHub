"""Checkpoint "soup": merge several trained adapters into one file.

The idea comes from model soups (arXiv 2203.05482): averaging the weights of
several runs of the *same* architecture usually lands somewhere better than any
single run — the shared signal reinforces, the run-specific noise cancels. For
style LoRAs this is the cheap way to combine, say, an epoch that nailed the
lighting with one that nailed the linework, without training again.

What makes it sound (and what this module enforces):

* Averaging weights is only meaningful when the tensors describe the same
  parameter. Same algorithm, same rank, same target modules, same shapes. We
  check all of that up front and refuse with a readable reason instead of
  writing a file that produces noise at inference.
* ``alpha`` keys are scale constants, not weights. Averaging them silently
  rescales every layer, so we keep the first file's alpha and say so when the
  inputs disagree.
* Accumulation happens in fp32 even when the inputs are bf16: a bf16 running
  sum rounds away the tail of a 4-way average.

Everything here is pure and synchronous — the files involved are tens of MB, so
there is no need for a background task.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

from ..infrastructure.paths import STUDIO_DATA

logger = logging.getLogger(__name__)

#: Where merged files land, and where the user's own uploads are kept.
SOUP_DIR = STUDIO_DATA / "soup"
UPLOAD_DIR = SOUP_DIR / "uploads"
OUTPUT_DIR = SOUP_DIR / "output"

#: Scalar per-layer constants that must be carried over, never averaged.
_SCALE_KEY_RE = re.compile(r"\.(alpha|scale)$")

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9 ._()\[\]-]+$")

MAX_INPUTS = 12


class SoupError(Exception):
    """Anything the user can fix: bad file, incompatible inputs, bad name."""


# ── filesystem helpers ───────────────────────────────────────────────────────

def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """Validate a user-supplied basename and force the .safetensors suffix.

    Path separators are rejected rather than stripped: silently turning
    ``a/b`` into ``b`` writes to a file the caller did not name.
    """
    stem = str(name or "").strip()
    if "/" in stem or "\\" in stem or stem in (".", "..") or stem.startswith("."):
        raise SoupError("File name must not contain a path.")
    if stem.lower().endswith(".safetensors"):
        stem = stem[: -len(".safetensors")]
    stem = stem.strip()
    if not stem:
        raise SoupError("File name is empty.")
    if len(stem) > 120:
        raise SoupError("File name is too long (max 120 characters).")
    if not _SAFE_NAME_RE.match(stem):
        raise SoupError(
            "File name may only contain letters, digits, spaces and . _ - ( ) [ ]"
        )
    return f"{stem}.safetensors"


def _resolve_inside(directory: Path, name: str) -> Path:
    """Resolve ``name`` strictly inside ``directory`` (blocks ../ traversal)."""
    target = (directory / safe_filename(name)).resolve()
    root = directory.resolve()
    if root not in target.parents:
        raise SoupError("Invalid file name.")
    return target


def upload_path(name: str) -> Path:
    return _resolve_inside(UPLOAD_DIR, name)


def output_path(name: str) -> Path:
    return _resolve_inside(OUTPUT_DIR, name)


def unique_output_path(name: str) -> Path:
    """``name.safetensors`` → ``name (2).safetensors`` when taken."""
    path = output_path(name)
    if not path.exists():
        return path
    stem = path.stem
    for n in range(2, 1000):
        candidate = path.with_name(f"{stem} ({n}).safetensors")
        if not candidate.exists():
            return candidate
    raise SoupError("Too many files with this name.")


def _listing(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for f in sorted(directory.glob("*.safetensors")):
        if not f.is_file():
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        items.append({
            "name": f.name,
            "path": str(f),
            "size": st.st_size,
            "mtime": st.st_mtime,
        })
    items.sort(key=lambda x: -x["mtime"])
    return items


def list_uploads() -> list[dict[str, Any]]:
    return _listing(UPLOAD_DIR)


def list_outputs() -> list[dict[str, Any]]:
    return _listing(OUTPUT_DIR)


def delete_upload(name: str) -> None:
    _unlink(upload_path(name))


def delete_output(name: str) -> None:
    _unlink(output_path(name))


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        raise SoupError("File not found.") from None
    except OSError as exc:
        raise SoupError(f"Could not delete the file: {exc}") from exc


def save_upload(name: str, data: bytes) -> dict[str, Any]:
    """Store an uploaded adapter, after checking it is a readable safetensors."""
    ensure_dirs()
    if not data:
        raise SoupError("The uploaded file is empty.")
    path = upload_path(name)
    tmp = Path(tempfile.mkstemp(dir=str(UPLOAD_DIR), suffix=".part")[1])
    try:
        tmp.write_bytes(data)
        # Parse before accepting: a truncated upload should fail here, not two
        # screens later inside the merge.
        inspect(tmp, require_suffix=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    st = path.stat()
    return {"name": path.name, "path": str(path), "size": st.st_size, "mtime": st.st_mtime}


# ── inspection ───────────────────────────────────────────────────────────────

def _parse_metadata(meta: dict[str, str]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    raw = meta.get("ss_network_args")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                args = parsed
        except (ValueError, TypeError):
            args = {}

    def _num(key: str) -> Optional[float]:
        try:
            return float(meta[key])
        except (KeyError, TypeError, ValueError):
            return None

    return {
        "algo": str(args.get("algo") or "").lower() or None,
        "rank": _num("ss_network_dim"),
        "alpha": _num("ss_network_alpha"),
        "factor": args.get("factor"),
        "family": args.get("model_family") or args.get("family"),
        "module": meta.get("ss_network_module"),
    }


def inspect(path: str | Path, *, require_suffix: bool = True) -> dict[str, Any]:
    """Read one adapter's shape fingerprint without loading its tensors.

    ``safe_open`` reads only the header, so this stays instant even on a folder
    full of checkpoints. ``require_suffix=False`` is for validating an upload
    while it is still sitting under its temporary name.
    """
    from safetensors import safe_open

    p = Path(path)
    if not p.is_file():
        raise SoupError(f"File not found: {p.name}")
    if require_suffix and p.suffix.lower() != ".safetensors":
        raise SoupError(f"Not a .safetensors file: {p.name}")

    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: set[str] = set()
    try:
        with safe_open(str(p), framework="pt") as f:
            meta = f.metadata() or {}
            for key in f.keys():
                slice_ = f.get_slice(key)
                shapes[key] = tuple(int(x) for x in slice_.get_shape())
                dtypes.add(str(slice_.get_dtype()))
    except SoupError:
        raise
    except Exception as exc:  # noqa: BLE001 — any parse failure is user-facing
        raise SoupError(f"Could not read {p.name}: {exc}") from exc

    if not shapes:
        raise SoupError(f"{p.name} contains no tensors.")

    info = _parse_metadata(dict(meta))
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    info.update({
        "name": p.name,
        "path": str(p),
        "size": size,
        "tensor_count": len(shapes),
        "dtypes": sorted(dtypes),
        # The fingerprint the compatibility check actually compares on.
        "_shapes": shapes,
    })
    return info


def _public(info: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in info.items() if not k.startswith("_")}


def compatibility(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Can these files be averaged? Returns the verdict plus every reason."""
    infos = [inspect(p) for p in paths]
    if len(infos) < 2:
        raise SoupError("Pick at least two checkpoints to merge.")
    if len(infos) > MAX_INPUTS:
        raise SoupError(f"At most {MAX_INPUTS} checkpoints can be merged at once.")

    base = infos[0]
    errors: list[str] = []
    warnings: list[str] = []

    for other in infos[1:]:
        missing = sorted(set(base["_shapes"]) - set(other["_shapes"]))
        extra = sorted(set(other["_shapes"]) - set(base["_shapes"]))
        if missing or extra:
            errors.append(
                f"{other['name']} targets different modules than {base['name']} "
                f"({len(missing)} missing, {len(extra)} extra). They were trained "
                f"with different network settings and cannot be averaged."
            )
            continue
        bad = [k for k, s in other["_shapes"].items() if base["_shapes"][k] != s]
        if bad:
            example = bad[0]
            errors.append(
                f"{other['name']} has a different rank/shape than {base['name']} "
                f"(e.g. {example}: {other['_shapes'][example]} vs "
                f"{base['_shapes'][example]}). Merging needs the same rank."
            )
            continue
        if base["algo"] and other["algo"] and base["algo"] != other["algo"]:
            errors.append(
                f"{other['name']} is {other['algo']}, {base['name']} is {base['algo']}."
            )
        if _differs(base["alpha"], other["alpha"]):
            warnings.append(
                f"{other['name']} was trained with alpha {other['alpha']}, "
                f"{base['name']} with {base['alpha']}. The merged file keeps "
                f"{base['alpha']} — lower the LoRA strength a little if the result "
                f"looks stronger than expected."
            )
        if other["dtypes"] != base["dtypes"]:
            warnings.append(
                f"{other['name']} is stored as {'/'.join(other['dtypes'])}, "
                f"{base['name']} as {'/'.join(base['dtypes'])}. The merge is done in "
                f"fp32 and saved as {'/'.join(base['dtypes'])}."
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "items": [_public(i) for i in infos],
    }


def _differs(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


# ── the merge itself ─────────────────────────────────────────────────────────

def normalize_weights(weights: list[float], method: str) -> list[float]:
    """``average`` scales the weights to sum to 1; ``sum`` leaves them alone.

    Average is the soup proper: the result stays on the same scale as the
    inputs, so the LoRA strength you used before still applies. Sum is there for
    the occasional "I want 0.7 of this one *on top of* that one" case, and is
    free to overshoot — which is exactly why it is not the default.
    """
    if any(not math.isfinite(w) for w in weights):
        raise SoupError("Weights must be finite numbers.")
    if method == "sum":
        return list(weights)
    total = sum(weights)
    if math.isclose(total, 0.0, abs_tol=1e-9):
        raise SoupError("The weights add up to zero — nothing to average.")
    return [w / total for w in weights]


def merge(
    inputs: list[dict[str, Any]],
    out_name: str,
    *,
    method: str = "average",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Weighted-merge ``inputs`` (``[{"path": ..., "weight": ...}]``) into one file.

    Returns the written file's listing entry plus the warnings the user should
    see. Raises :class:`SoupError` for anything they can act on.
    """
    import torch
    from safetensors.torch import save_file
    from safetensors import safe_open

    if method not in ("average", "sum"):
        raise SoupError(f"Unknown merge method: {method}")
    if len(inputs) < 2:
        raise SoupError("Pick at least two checkpoints to merge.")

    paths = [Path(str(i.get("path") or "")) for i in inputs]
    raw_weights = []
    for i in inputs:
        try:
            raw_weights.append(float(i.get("weight", 1.0)))
        except (TypeError, ValueError):
            raise SoupError("Weights must be numbers.") from None

    verdict = compatibility(paths)
    if not verdict["ok"]:
        raise SoupError(verdict["errors"][0])

    weights = normalize_weights(raw_weights, method)
    infos = {i["path"]: i for i in verdict["items"]}
    base_info = verdict["items"][0]

    ensure_dirs()
    out = output_path(out_name) if overwrite else unique_output_path(out_name)

    # fp32 accumulator, cast back at the end — see the module docstring.
    acc: dict[str, torch.Tensor] = {}
    scale_keys: dict[str, torch.Tensor] = {}
    out_dtype: dict[str, torch.dtype] = {}

    for idx, (path, weight) in enumerate(zip(paths, weights)):
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if _SCALE_KEY_RE.search(key) or tensor.ndim == 0:
                    # Scale constants come from the first file, untouched.
                    if idx == 0:
                        scale_keys[key] = tensor
                    continue
                if idx == 0:
                    out_dtype[key] = tensor.dtype
                    acc[key] = tensor.to(torch.float32) * weight
                else:
                    acc[key].add_(tensor.to(torch.float32) * weight)

    merged = {k: v.to(out_dtype[k]) for k, v in acc.items()}
    merged.update(scale_keys)

    meta = _soup_metadata(base_info, paths, raw_weights, weights, method)

    tmp = Path(tempfile.mkstemp(dir=str(OUTPUT_DIR), suffix=".part")[1])
    try:
        save_file(merged, str(tmp), metadata=meta)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)

    st = out.stat()
    logger.info("soup: merged %d checkpoints into %s", len(paths), out.name)
    return {
        "name": out.name,
        "path": str(out),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "warnings": verdict["warnings"],
        "sources": [
            {"name": infos[str(p)]["name"], "weight": rw, "effective": ew}
            for p, rw, ew in zip(paths, raw_weights, weights)
        ],
    }


def _soup_metadata(
    base: dict[str, Any],
    paths: list[Path],
    raw_weights: list[float],
    weights: list[float],
    method: str,
) -> dict[str, str]:
    """Carry over the base file's network metadata and record the recipe.

    Inference code reads ``ss_network_*`` to rebuild the adapter, so those must
    survive the merge verbatim. ``ss_soup_sources`` is ours: without it a merged
    file is anonymous three weeks later.
    """
    from safetensors import safe_open

    try:
        with safe_open(str(paths[0]), framework="pt") as f:
            meta = dict(f.metadata() or {})
    except Exception:  # noqa: BLE001 — provenance is best-effort
        meta = {}

    meta["ss_soup_method"] = method
    meta["ss_soup_sources"] = json.dumps(
        [
            {"name": p.name, "weight": rw, "effective_weight": ew}
            for p, rw, ew in zip(paths, raw_weights, weights)
        ],
        ensure_ascii=False,
    )
    if base.get("alpha") is not None:
        meta.setdefault("ss_network_alpha", str(base["alpha"]))
    return {k: str(v) for k, v in meta.items()}
