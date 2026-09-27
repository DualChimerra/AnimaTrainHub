"""Low-level infrastructure -- extracted from the studio/ top level starting in PR-7.

Submodules:
    paths.py           path constants + safe_join / validate_path_component
    event_bus.py        in-process SSE bus
    log_tail.py         per-task incremental log reads + monitor state polling
    argparse_bridge.py  pydantic model -> argparse argument derivation
    llm_presets.py      loads factory presets from studio/llm_presets/*.json

Upcoming PRs:
    secrets/  PR-7 commit 2  secrets.py 763 lines -> models/store/migrations, 3 files
    db/       PR-7 commit 3  db.py 188 lines + studio/migrations/ -> infrastructure/db/
"""
