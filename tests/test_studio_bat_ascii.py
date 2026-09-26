from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDIO_BAT = REPO_ROOT / "studio.bat"


def test_studio_bat_exists() -> None:
    assert STUDIO_BAT.exists(), "studio.bat should not disappear"


def test_studio_bat_is_pure_ascii() -> None:
    data = STUDIO_BAT.read_bytes()
    bad_offsets = [(i, b) for i, b in enumerate(data) if b > 127]
    if bad_offsets:
        snippets = []
        for off, b in bad_offsets[:5]:
            start = max(0, off - 20)
            end = min(len(data), off + 20)
            ctx = data[start:end].decode("utf-8", errors="replace")
            snippets.append(f"  offset {off} (byte 0x{b:02x}): ...{ctx!r}...")
        pytest.fail(
            f"studio.bat contains {len(bad_offsets)} non-ASCII byte(s) (first 5 detailed below):\n"
            + "\n".join(snippets)
            + "\n\nstudio.bat must stay pure ASCII -- cmd.exe parses it with the system ANSI "
            "codepage before chcp 65001 runs, and UTF-8 Chinese bytes get split into garbled "
            "commands. Put any non-ASCII messages inside echo, routed through a Python "
            "process that has already set PYTHONUTF8 / chcp 65001."
        )
