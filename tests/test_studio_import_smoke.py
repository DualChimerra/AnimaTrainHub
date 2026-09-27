"""PR-1 safety net -- every .py under studio/ must be importable.

The most common accident during a refactor is a circular import, or forgetting to update an
import path after moving a directory. This test uses parametrize to give each module its own case, for precise localization.

Excluded:
- __pycache__/
- studio/web/ (frontend build output + node_modules)
- any .py under web/node_modules/
- studio/__main__.py (no `if __name__ == "__main__"` guard -- importing it runs main() directly)

Known import-time side effects (current state, to be fixed in PR-5/PR-7):
- studio.server: ensure_dirs() + db.init_db()
- studio.services.onnxruntime_setup: DLL preload
This test accepts those side effects and only verifies that import doesn't raise.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

STUDIO_ROOT = Path(__file__).parent.parent / "studio"


def _enumerate_modules() -> list[str]:
    modules: list[str] = []
    for py in sorted(STUDIO_ROOT.rglob("*.py")):
        rel = py.relative_to(STUDIO_ROOT.parent)
        parts = rel.with_suffix("").parts
        # skip cache
        if "__pycache__" in parts:
            continue
        # skip .py files that leaked in under the frontend dir (e.g. web/node_modules/flatted/python/flatted.py)
        if "web" in parts:
            continue
        # skip __main__: no if __name__ guard, importing it runs main() and raises SystemExit
        if parts[-1] == "__main__":
            continue
        # represent a package's __init__ by the package name
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(parts))
    return modules


MODULES = _enumerate_modules()


def test_module_list_not_empty() -> None:
    assert len(MODULES) > 50, f"only found {len(MODULES)} studio/ modules -- scanning may be broken"


@pytest.mark.parametrize("module_name", MODULES, ids=MODULES)
def test_import_module(module_name: str) -> None:
    importlib.import_module(module_name)
