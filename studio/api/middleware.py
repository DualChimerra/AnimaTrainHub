"""FastAPI middleware (extracted from server.py in PR-5)."""
from __future__ import annotations

from fastapi.middleware.gzip import GZipMiddleware

# Automatic gzip compression for JSON / text responses. Highly compressible
# for large arrays like /api/state (a 10k-step training run goes from ~500KB
# to ~100KB); skipped for small responses (< 1000B) to avoid framing
# overhead actually making them bigger.
#
# Two path categories are explicitly excluded:
#   - /api/events: SSE stream. GZipMiddleware buffers chunks until
#     minimum_size before sending, which breaks SSE real-time delivery, and
#     some EventSource implementations have compatibility issues parsing a
#     gzip stream.
#   - /samples/*: image bytes (PNG/JPEG/WEBP are already compressed formats),
#     so gzipping them again is pure wasted CPU and can even increase size
#     slightly.
_GZIP_SKIP_PREFIXES = ("/api/events", "/samples/")


class _SelectiveGZipMiddleware(GZipMiddleware):
    """Subclass of GZipMiddleware that bypasses specified routes by path prefix."""

    async def __call__(self, scope, receive, send):  # type: ignore[override]
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if any(path.startswith(p) for p in _GZIP_SKIP_PREFIXES):
                await self.app(scope, receive, send)
                return
        await super().__call__(scope, receive, send)
