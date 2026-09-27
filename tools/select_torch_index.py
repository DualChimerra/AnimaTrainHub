#!/usr/bin/env python
"""Bootstrap helper: prints the PyTorch wheel index URL for the detected NVIDIA driver version.

studio.bat / studio.sh calls this script on the **first** venv install, to install
the right torch build for the GPU before requirements.txt. The `torch>=2.0.0`
constraint is already satisfied by that first step, so pip won't later overwrite it
with the default PyPI CPU build.

Stdlib only -- must run right after the venv is created (only pip + setuptools present).

Output:
- Suitable driver detected -> one URL line on stdout, e.g. `https://download.pytorch.org/whl/cu128`
- nvidia-smi missing / parse failure / driver too old -> silent, no output (caller falls back to PyPI default)
- Always exits 0, so bootstrap never fails because of this step

Driver -> cu wheel mapping: kept in sync with studio/services/torch_setup.py:_DRIVER_TO_BEST_CU.
Duplicated here so the bootstrap stage doesn't depend on the studio.services submodule load chain.
"""
from __future__ import annotations

import re
import subprocess
import sys

# Note: keep this table in sync with studio/services/torch_setup.py:_DRIVER_TO_BEST_CU
_DRIVER_TO_CU: list[tuple[int, str]] = [
    (555, "cu128"),
    (550, "cu126"),
    (545, "cu124"),
    (470, "cu118"),
]
_PYPI_BASE = "https://download.pytorch.org/whl"


def detect_driver_major() -> int | None:
    """Run nvidia-smi to get the driver major version; returns None on failure / if absent."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    line = (out.stdout or "").strip().split("\n")[0]
    m = re.match(r"^(\d+)\.", line)
    if not m:
        return None
    return int(m.group(1))


def select_index_url(driver_major: int | None) -> str | None:
    """Driver major version -> PyTorch wheel index URL; too old / None -> None."""
    if driver_major is None:
        return None
    for threshold, tag in _DRIVER_TO_CU:
        if driver_major >= threshold:
            return f"{_PYPI_BASE}/{tag}"
    return None


def main() -> int:
    url = select_index_url(detect_driver_major())
    if url:
        # No trailing newline -- avoids pitfalls when shell `for /f` / `$()` capture it
        sys.stdout.write(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
