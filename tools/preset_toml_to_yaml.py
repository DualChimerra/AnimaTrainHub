"""Convert preset files exported by the frontend `downloadCurrentPreset` bug between
2026-05-21 and 2026-05-24 -- "fake yaml, really TOML" -- into real yaml.

Bug details: during those 3 days, the Studio frontend, when exporting a preset, called
`generateToml(config)` to generate hand-rolled TOML (flat key = value + multi-line `{...}`
fake inline tables + null written as `key = `), but the blob MIME type / file extension were
both tagged as yaml. The server's `parse_preset_bytes` only recognizes yaml/json, so
re-uploading these files gets rejected with:
    "Invalid preset format (top level is not a mapping)"

The new frontend now goes through the server's `GET /api/presets/{name}/download` endpoint,
which sends the real yaml straight from disk, and no longer produces files like this. This
tool exists only to salvage historical downloaded files and is not part of the server's
dependency chain (avoids adding a TOML parser to production imports).

Usage:
    python tools/preset_toml_to_yaml.py broken1.yaml [broken2.yaml ...]
        Each file's output goes to `<stem>.fixed.yaml` in the same directory; the original is kept.
    python tools/preset_toml_to_yaml.py --in-place broken.yaml
        Overwrite the original file in place, backing it up as `<stem>.yaml.toml-bak` first.
    python tools/preset_toml_to_yaml.py --output out.yaml broken.yaml
        Specify the output path (only works for a single file).
    python tools/preset_toml_to_yaml.py --no-validate broken.yaml
        Skip TrainingConfig validation (emergency use when the schema has drifted; on by default).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib  # py 3.11+
except ImportError:  # pragma: no cover - py 3.10 and below fall back to tomli
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ImportError:
        sys.stderr.write(
            "Missing TOML parser library: py < 3.11 needs `pip install tomli` first.\n"
        )
        sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _preprocess(text: str) -> tuple[str, list[str]]:
    """Fix the non-standard TOML produced by the frontend's `generateToml`, returning
    (cleaned_text, warnings).

    Two known kinds of non-compliance:
    - **Multi-line `{...}` blocks**: a TOML inline table must be on a single line; here each
      `k = v` line inside a multi-line block is joined into one line
      `{ k = v, k2 = v2 }`.
    - **Empty value `key = ` (null/undefined)**: TOML doesn't allow an empty rhs; these keys
      are simply dropped, falling back to pydantic's schema defaults / Optional[None].
    """
    warnings: list[str] = []
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Start of a multi-line block: `key = {`
        m = re.match(r"^(\S.*?=\s*)\{\s*$", line)
        if m:
            prefix = m.group(1).rstrip()
            inner: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].strip() != "}":
                s = lines[j].strip()
                if s:
                    inner.append(s)
                j += 1
            if j >= len(lines):
                # No closing } found -- leave it for tomllib to report the original error
                out.append(line)
                i += 1
                continue
            out.append(f"{prefix} {{ {', '.join(inner)} }}")
            i = j + 1
            continue
        # Empty value: `key = ` or `key =` (rhs is all whitespace)
        if "=" in stripped and not stripped.startswith("#"):
            k, _, v = stripped.partition("=")
            if v.strip() == "":
                warnings.append(f"dropped empty-value key: {k.strip()}")
                i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out) + "\n", warnings


def _validate(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize via TrainingConfig.model_validate; returns (normalized dict, warnings)."""
    # Deferred import: by default this tool should be runnable in an environment without the
    # training stack installed (just yaml + tomllib)
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from studio.schema import TrainingConfig  # type: ignore[import-not-found]
    except ImportError as exc:
        return data, [f"skipped schema validation (import failed: {exc})"]
    try:
        cfg = TrainingConfig.model_validate(data)
    except Exception as exc:
        # A validation failure isn't fatal: write the raw data + a warning, and let the user
        # fix it by hand
        return data, [f"TrainingConfig validation failed (raw data written, please fix manually): {exc}"]
    return cfg.model_dump(mode="python"), []


def convert(text: str, validate: bool = True) -> tuple[str, list[str]]:
    """Main conversion: returns (yaml_text, warnings)."""
    warnings: list[str] = []
    cleaned, pre_warns = _preprocess(text)
    warnings.extend(pre_warns)
    try:
        data = tomllib.loads(cleaned)
    except Exception as exc:
        raise SystemExit(f"TOML parsing failed: {exc}\n(first 200 chars of preprocessed content)\n{cleaned[:200]}")
    if not isinstance(data, dict):
        raise SystemExit(f"TOML top level is not a mapping, got {type(data).__name__}")
    if validate:
        data, val_warns = _validate(data)
        warnings.extend(val_warns)
    yaml_text = yaml.safe_dump(
        data, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    return yaml_text, warnings


def _convert_file(src: Path, dst: Path, validate: bool, in_place: bool) -> None:
    if not src.exists():
        raise SystemExit(f"File not found: {src}")
    text = src.read_text(encoding="utf-8")
    yaml_text, warnings = convert(text, validate=validate)
    if in_place:
        bak = src.with_suffix(src.suffix + ".toml-bak")
        if bak.exists():
            raise SystemExit(f"Backup target already exists, delete it first and retry: {bak}")
        src.rename(bak)
        sys.stderr.write(f"  backed up original file -> {bak}\n")
    dst.write_text(yaml_text, encoding="utf-8")
    sys.stderr.write(f"  written -> {dst}\n")
    for w in warnings:
        sys.stderr.write(f"  ⚠ {w}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("files", nargs="+", type=Path, help="the .yaml (actually TOML) file(s) to convert")
    ap.add_argument(
        "--in-place", action="store_true",
        help="overwrite the original file in place, backing it up as .yaml.toml-bak",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="specify the output path (only takes effect for a single input file)",
    )
    ap.add_argument(
        "--no-validate", action="store_true",
        help="skip TrainingConfig schema validation (emergency use when the schema has drifted)",
    )
    args = ap.parse_args(argv)

    if args.output and len(args.files) > 1:
        ap.error("--output only supports a single input file")
    if args.output and args.in_place:
        ap.error("--output and --in-place are mutually exclusive")

    validate = not args.no_validate
    for src in args.files:
        sys.stderr.write(f"converting {src}\n")
        if args.in_place:
            dst = src
        elif args.output:
            dst = args.output
        else:
            dst = src.with_name(f"{src.stem}.fixed{src.suffix}")
        _convert_file(src, dst, validate=validate, in_place=args.in_place)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
