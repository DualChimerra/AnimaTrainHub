"""Root path + legacy `/studio/*` compat redirect (ADR 0012).

routes:
    GET /                       dist already built → serve index.html directly (SPA entry
                                point, no longer redirects); dist missing → return a JSON
                                build hint
    GET /studio                 one-time legacy redirect → / (307, preserves query)
    GET /studio/{rest:path}     one-time legacy redirect → /{rest}

Historically the SPA was mounted at the `/studio/` subpath, with `/` 302-redirecting
there (to make way for now-deleted root-level pages like monitor_smooth.html). As of
ADR 0012 the SPA is mounted back at the root path: `/` no longer redirects, which
eliminates an infinite loop (issue #330) caused by reverse-proxy container-port
platforms (ModelScope Creative Space / HF Space) normalizing the trailing slash on
top of the mount's 307. The `/studio/*` redirect is kept for one release to give old
bookmarks / tabs still parked on `/studio/...` after a self-update a smooth landing
spot; it will be removed in the next version.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from ...paths import WEB_DIST

router = APIRouter()


@router.get("/", response_model=None, include_in_schema=False)
def root() -> FileResponse | JSONResponse:
    """Root path SPA entry point.

    dist already built → return index.html directly (ADR 0012: no longer
    302-redirects to /studio/); other frontend assets / deep links are
    handled by the SPAStaticFiles mounted at `/` in server.py.
    dist missing → return a JSON hint."""
    index = WEB_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {
            "message": "AnimaTrainHub is running. Build the React app at studio/web/ "
            "(npm install && npm run build) to enable the new UI."
        }
    )


@router.get("/studio", response_model=None, include_in_schema=False)
@router.get("/studio/{rest:path}", response_model=None, include_in_schema=False)
def legacy_studio_redirect(request: Request, rest: str = "") -> RedirectResponse:
    """ADR 0012 legacy: old `/studio/...` links get a one-time 307 redirect to the root path, kept for one release.

    307 preserves the method and isn't permanently cached (so no stale 301
    cache lingers once this is removed in the next version); query string is
    passed through."""
    qs = ("?" + request.url.query) if request.url.query else ""
    target = "/" + rest.lstrip("/") if rest else "/"
    return RedirectResponse(url=target + qs, status_code=307)
