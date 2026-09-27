"""SSE event stream (extracted from server.py in PR-5).

1 route:
    GET /api/events    SSE — broadcasts task/job status changes + monitor deltas +
                       daemon state etc. to all subscribers. 15s keepalive to avoid
                       proxy timeouts; await is_disconnected() breaks immediately so
                       no queue is left dangling (prevents memory leaks).
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ...infrastructure.event_bus import bus

router = APIRouter()


@router.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    """SSE: broadcasts task status-change events to all subscribers."""
    queue = await bus.subscribe()

    async def gen() -> AsyncIterator[bytes]:
        try:
            yield b": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(evt)}\n\n".encode("utf-8")
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")
