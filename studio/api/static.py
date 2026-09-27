"""SPA static file mount (extracted from server.py in PR-5)."""
from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# The Windows registry often tags .js as text/plain (polluted by IIS /
# antivirus / an old install), causing the ES module strict MIME check to
# reject it -> a blank screen on startup. See issue #228. Overridden
# proactively at import time to ensure cross-platform consistency.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/svg+xml", ".svg")

# ADR 0012: once the SPA is mounted at the root path, this catch-all takes
# over the entire URL space. These are the root-level namespaces owned by the
# server — requests under them that don't match a route must stay a clean
# 404 and must not fall back to index.html (otherwise a typo like /api/typo
# would return 200 and mislead the caller).
# **When adding a new non-/api root-level route, its first path segment must
# be added here.**
_RESERVED_SEGMENTS = frozenset({"api", "samples"})


class SPAStaticFiles(StaticFiles):
    """SPA routing fallback: returns index.html when no actual file matches
    and the request doesn't look like a static asset.

    This lets a direct refresh on a react-router route like
    `/studio/projects/1/v/1/curate` still get index.html, so BrowserRouter
    can resolve the path on the frontend. Requests with a file extension
    (.js/.css/.png etc.) keep the original 404 behavior, to avoid silently
    turning a missing resource into a 200 and misleading the browser.
    """

    async def get_response(self, path, scope):  # type: ignore[override]
        from starlette.exceptions import HTTPException as StarletteHTTPException
        # StaticFiles.get_path goes through os.path.normpath, and on Windows
        # the separator is a backslash — normalize to "/" first, otherwise
        # the namespace / extension checks below don't work on Windows.
        norm = path.replace("\\", "/")
        # Requests under a server-owned namespace (/api, /samples) that don't
        # match an explicit route -> a uniform clean 404 (for any method),
        # never reaching StaticFiles (ADR 0012). Must be intercepted before
        # calling super(): StaticFiles returns 405 first for non-GET/HEAD
        # methods and won't fall back to index.html — once mounted at the
        # root, an unmatched request like `PUT /api/typo` would become a 405
        # instead of the correct 404.
        first_seg = norm.split("/", 1)[0]
        if first_seg in _RESERVED_SEGMENTS:
            raise StarletteHTTPException(status_code=404)
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            # Last segment contains "." -> treated as a static asset request, no fallback
            last = norm.rsplit("/", 1)[-1]
            if "." in last:
                raise
            return FileResponse(Path(self.directory) / "index.html")
