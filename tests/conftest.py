"""Shared test configuration: makes sure `import studio.*` / `import train_monitor` /
`import anima_*` can be found.

`train_monitor` and `anima_*` (train / generate / daemon / reg_ai) all live in `runtime/`,
which was never converted to package imports (still bare-script style), so `runtime/`
needs to be injected into sys.path.

PR-1 C4 adds the _isolate_studio_logging session fixture: keeps api/lifespan, cli.main,
and workers/_base.worker_main from actually installing setup_logging when triggered by
tests (avoids polluting caplog + avoids writing real studio_data/logs/). See the fixture
docstring for details."""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "runtime"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


@pytest.fixture(scope="session", autouse=True)
def _isolate_studio_logging(tmp_path_factory: pytest.TempPathFactory):
    """PR-1 C4 -- makes application-code setup_logging a full no-op during tests
    (keeps caplog clean).

    Application entry points (api/lifespan / cli.main / workers/_base.worker_main) call
    setup_logging during their own startup. Any test that triggers one of these (e.g.
    TestClient(app) running lifespan) would install a real file handler that writes to
    the repo's studio_data/logs/, polluting it. Setting this env var makes setup_logging
    early-return at the top.

    tests/test_logging_setup.py, which tests setup_logging itself, uses
    monkeypatch.delenv to unset this env var on its own.
    """
    os.environ["ANIMA_LOGGING_NO_BOOTSTRAP"] = "1"
    # Also set ANIMA_LOG_DIR as a fallback (in case some test explicitly calls
    # setup_logging without passing log_dir)
    os.environ["ANIMA_LOG_DIR"] = str(tmp_path_factory.mktemp("studio_logs"))
    yield
