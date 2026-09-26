#!/usr/bin/env python
"""flash_attn prebuilt wheel install CLI (command-line entry point that mirrors the Studio Settings UI).

Usage:
    python tools/install_flash_attn.py            # auto-pick and install the best wheel
    python tools/install_flash_attn.py --url URL  # manually specify a wheel URL
    python tools/install_flash_attn.py --dry-run  # only list environment + candidates, don't install
    python tools/install_flash_attn.py --force    # reinstall even if already installed

Exit codes: 0 success / 1 install failed / 2 unsupported environment

Implementation shared with studio.services.flash_attention_setup -- uses the exact same wheel
selection logic as the UI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Let the script run as `python tools/install_flash_attn.py` directly from the venv -- inject the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install a flash_attn prebuilt wheel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", help="manually specify a wheel URL (skips auto-matching)")
    parser.add_argument(
        "--dry-run", action="store_true", help="only list environment + candidates, don't install"
    )
    parser.add_argument(
        "--force", action="store_true", help="reinstall even if already installed"
    )
    args = parser.parse_args(argv)

    from studio.services.runtime import flash_attention as fa  # noqa: PLC0415

    env = fa.detect_env()
    print("[env] python:   ", env.get("python_tag"))
    print("[env] platform: ", env.get("platform"))
    print("[env] cuda:     ", env.get("cuda_tag"), f"({env.get('cuda_ver')})")
    print("[env] torch:    ", env.get("torch_tag"), f"({env.get('torch_ver')})")

    status = fa.current_status()
    if status["installed"]:
        print(f"[status] flash_attn=={status['version']} already installed")
        if not args.force and not args.dry_run:
            print("       use --force to reinstall; leaving as-is")
            return 0
    else:
        print("[status] flash_attn not installed")

    if not env.get("platform"):
        print(
            "[error] unsupported platform (prebuilt wheels only exist for linux_x86_64 / win_amd64)",
            file=sys.stderr,
        )
        return 2

    if args.dry_run or not args.url:
        candidates, fetch_error = fa.find_candidates(env)
        if fetch_error:
            print(f"[warn] failed to fetch the candidate list: {fetch_error}", file=sys.stderr)
            print(
                "       you can pass --url manually, pick one from "
                "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases",
                file=sys.stderr,
            )
            if args.dry_run:
                return 0
            return 2
        if not candidates:
            print("[warn] no matching wheel found (check whether the env above is complete)", file=sys.stderr)
            return 2

        print(f"\n[candidates] {len(candidates)} wheels total (sorted by score, descending):")
        for i, c in enumerate(candidates[:10]):
            mark = "✓" if c["usable"] else "✗"
            note_str = "; ".join(c["notes"]) if c["notes"] else ""
            print(f"  {mark} score={c['score']:>3}  {c['name']}")
            if note_str:
                print(f"      {note_str}")
        if len(candidates) > 10:
            print(f"  ... {len(candidates) - 10} more not shown")

    if args.dry_run:
        return 0

    try:
        result = fa.install(args.url)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"\n[ok] flash_attn=={result['version']} installed")
    print(f"     {result['url']}")
    if result.get("restart_required"):
        print("[note] flash_attn is a C extension; any running Studio / training process needs a restart to pick it up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
