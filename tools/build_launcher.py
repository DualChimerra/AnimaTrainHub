#!/usr/bin/env python
"""Compiles `tools/launcher.py` into a single-file executable (AnimaLoraStudio.exe on Windows).

    python tools/build_launcher.py                 -> dist/AnimaLoraStudio(.exe)
    python tools/build_launcher.py --output-dir X   custom output directory
    python tools/build_launcher.py --console        keep the console window (default anyway)

The output is only a few MB: it bundles just the bootstrap logic (pure stdlib), it does
**not** include torch / the frontend / studio itself -- those stay in the repo, and the
exe installs the right versions for the user's GPU on their machine. See the module
docstring of `tools/launcher.py`.

PyInstaller is only needed on the build machine, so it's not in requirements.txt; this
script prints the install command instead of crashing when it's missing. Cross-compiling
isn't possible (PyInstaller produces an executable for the host platform), so the Windows
exe must be built on Windows -- CI does this via `.github/workflows/build-launcher.yml`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRY = REPO_ROOT / "tools" / "launcher.py"
NAME = "AnimaLoraStudio"


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401  -- only probing availability
    except ImportError:
        print(
            "PyInstaller is not installed.\n"
            f"  {sys.executable} -m pip install pyinstaller\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def build(output_dir: Path, *, windowed: bool, clean: bool) -> Path:
    work = output_dir / "_build"
    argv = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", NAME,
        "--distpath", str(output_dir),
        "--workpath", str(work),
        "--specpath", str(work),
        # The bootstrap is pure stdlib: exclude these packages explicitly, so that if the
        # build machine happens to have torch / numpy installed, PyInstaller's dependency
        # analysis doesn't bundle in hundreds of extra MB.
        "--exclude-module", "torch",
        "--exclude-module", "numpy",
        "--exclude-module", "PIL",
        "--exclude-module", "fastapi",
        "--exclude-module", "pydantic",
        "--exclude-module", "tkinter",
    ]
    if clean:
        argv.append("--clean")
    # Console kept by default: the first run installs several GB of dependencies, and this
    # window is the only way to see progress or errors. Hiding it would leave the user with
    # nothing to describe except "it sat there for ten minutes and did nothing."
    argv.append("--windowed" if windowed else "--console")
    icon = REPO_ROOT / "docs" / "images" / "launcher.ico"
    if icon.exists():
        argv += ["--icon", str(icon)]
    argv.append(str(ENTRY))

    print("[build]", " ".join(argv), flush=True)
    rc = subprocess.call(argv, cwd=str(REPO_ROOT))
    if rc != 0:
        raise SystemExit(rc)

    produced = output_dir / (f"{NAME}.exe" if sys.platform == "win32" else NAME)
    if not produced.exists():
        print(f"[build] expected {produced} but it was not created", file=sys.stderr)
        raise SystemExit(1)
    return produced


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", default=str(REPO_ROOT / "dist"),
                   help="where to put the executable (default: dist/)")
    p.add_argument("--windowed", action="store_true",
                   help="hide the console window (not recommended — see module docstring)")
    p.add_argument("--no-clean", action="store_true",
                   help="reuse PyInstaller's build cache")
    args = p.parse_args(argv)

    if not ENTRY.exists():
        print(f"entry point missing: {ENTRY}", file=sys.stderr)
        return 1
    ensure_pyinstaller()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    produced = build(output_dir, windowed=args.windowed, clean=not args.no_clean)

    size_mb = produced.stat().st_size / (1024 * 1024)
    print(f"\n[build] {produced}  ({size_mb:.1f} MB)")
    print("[build] put it in the repository root (next to requirements.txt) and double-click.")
    shutil.rmtree(output_dir / "_build", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
