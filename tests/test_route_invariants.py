"""PR-1 safety net -- coarse-grained invariants on route count and decorator count.

The snapshot is the fine-grained net (any character change triggers it); this file is the coarse-grained second line of defense:
- the count falls within a sane range
- the total number of @app.<verb> / @router.<verb> decorators across server.py + api/routers/*.py == the number of APIRoutes

Since PR-5 moved routers out of server.py into api/routers/, this test scans and sums the decorators
in both places. New router files (as PR-6 continues extracting) are picked up automatically.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.routing import APIRoute

from studio.server import app

from ._route_helpers import iter_leaf_routes

STUDIO_DIR = Path(__file__).parent.parent / "studio"
SERVER_PY = STUDIO_DIR / "server.py"
API_ROUTERS_DIR = STUDIO_DIR / "api" / "routers"


def test_route_count_in_sane_range() -> None:
    # FastAPI 0.137+ wraps include_router in an `_IncludedRouter` wrapper (see
    # tests/_route_helpers.py) -- in the new version, plain len(app.routes) only counts the wrapper, not the
    # inner APIRoute; recursively expanding it gives the real route count.
    n = sum(1 for _ in iter_leaf_routes(app.routes))
    assert 100 <= n <= 250, f"app.routes expanded = {n}, outside the sane range [100, 250]"


def test_decorator_count_matches_api_routes() -> None:
    # `@app.<verb>(...)` decorators in server.py
    src = SERVER_PY.read_text(encoding="utf-8")
    app_decorator_count = len(
        re.findall(
            r"^@app\.(get|post|put|delete|patch|api_route)\b",
            src,
            flags=re.MULTILINE,
        )
    )
    # `@router.<verb>(...)` decorators under api/routers/**/*.py (recursively covers subpackages like queue/)
    router_decorator_count = 0
    if API_ROUTERS_DIR.is_dir():
        for py in sorted(API_ROUTERS_DIR.rglob("*.py")):
            if py.name == "__init__.py":
                continue
            router_src = py.read_text(encoding="utf-8")
            router_decorator_count += len(
                re.findall(
                    r"^@router\.(get|post|put|delete|patch|api_route)\b",
                    router_src,
                    flags=re.MULTILINE,
                )
            )

    decorator_total = app_decorator_count + router_decorator_count
    api_route_count = sum(1 for r in iter_leaf_routes(app.routes) if isinstance(r, APIRoute))
    assert decorator_total == api_route_count, (
        f"total decorators {decorator_total} (server.py {app_decorator_count} + "
        f"api/routers/ {router_decorator_count}) != APIRoute instances in app.routes "
        f"{api_route_count} -- the discrepancy may be a missing include_router or a router that lost one at registration"
    )
