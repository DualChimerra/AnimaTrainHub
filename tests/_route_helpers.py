"""Shared route introspection helper -- compatible with FastAPI 0.137+ wrapping
include_router into the internal `_IncludedRouter` representation.

Behavior change (not an application bug, only affects introspection tests):

  0.136 and earlier:
    app.include_router(sub) -> each APIRoute of sub is appended directly to app.routes
  0.137 onward:
    app.include_router(sub) -> app.routes gains 1 extra `_IncludedRouter(original_router=sub)`
    wrapper; to get the underlying APIRoute you must recurse through `wrapper.original_router.routes`

HTTP routing dispatch works normally on both versions (verified with TestClient). The change only
affects code that walks `app.routes` directly to get APIRoute instances -- mainly tests.
"""
from __future__ import annotations

from typing import Any, Iterator


def iter_leaf_routes(routes: list[Any]) -> Iterator[Any]:
    """Recursively unwrap `_IncludedRouter` wrappers, yielding APIRoute / Mount / etc leaves.

    Compatible across fastapi versions:
    - 0.136: top level is already a leaf (else branch yields directly)
    - 0.137+: when a wrapper is found, recurse through `original_router.routes`
    - Nested include_router also unwraps correctly
    """
    for r in routes:
        if hasattr(r, "original_router") and hasattr(r.original_router, "routes"):
            yield from iter_leaf_routes(r.original_router.routes)
        else:
            yield r
