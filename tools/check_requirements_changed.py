#!/usr/bin/env python
"""Bootstrap helper: detects whether requirements.txt changed since the last venv sync.

studio.sh / studio.bat call this script during startup and decide whether to install
new dependencies based on stdout.

Modes:
- Default (read mode): compares the requirements.txt content hash against the marker file
  - Prints `stale`: content changed / no marker (first run or an old venv) -> caller should run pip install
  - Prints `current`: hash matches -> skip
  - Prints `missing`: requirements.txt doesn't exist -> skip

- `--update-marker`: writes the new hash after a successful sync, prints `written`

Why content hash instead of mtime:
- `git checkout` / `git pull` preserve commit timestamps under some git configs, so mtime
  would be misread as "stale" and trigger unnecessary pip installs
- a hash only reacts to actual content changes -- bulletproof

Stdlib only -- must run right after the venv is created (before pip has installed anything).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def compute_req_hash(req_path: Path) -> str:
    return hashlib.sha256(req_path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--marker", required=True,
        help="path to the marker file (e.g. venv/.studio-requirements.sha256)",
    )
    parser.add_argument(
        "--requirements", default="requirements.txt",
        help="requirements file to compare against",
    )
    parser.add_argument(
        "--update-marker", action="store_true",
        help="write the current hash to the marker (call after a successful sync)",
    )
    args = parser.parse_args(argv)

    req = Path(args.requirements)
    if not req.exists():
        print("missing")
        return 0

    current = compute_req_hash(req)
    marker = Path(args.marker)

    if args.update_marker:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(current, encoding="utf-8")
        print("written")
        return 0

    if not marker.exists():
        # Old venv with no marker -> treat as stale; caller runs pip once and writes the
        # marker, then it's fine on subsequent runs
        print("stale")
        return 0

    try:
        stored = marker.read_text(encoding="utf-8").strip()
    except OSError:
        # Marker is corrupted -> treat as stale, it gets rewritten on the next sync
        print("stale")
        return 0

    print("stale" if stored != current else "current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
