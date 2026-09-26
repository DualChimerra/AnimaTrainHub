"""Thread-safe event bus: supervisor (sync thread) -> FastAPI SSE (asyncio).

Usage:
    bus = EventBus()
    # bind the event loop during FastAPI lifespan startup
    bus.attach_loop(asyncio.get_running_loop())

    # SSE connection
    q = await bus.subscribe()
    try:
        evt = await q.get()
    finally:
        bus.unsubscribe(q)

    # publish from any thread
    bus.publish({"type": "task_state_changed", "task_id": 7, "status": "done"})
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Optional

_drop_logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # commit 11: connection lifecycle hooks (generate cache uses "last" -> 30s timer cleanup)
        self._on_first_subscribe: Optional[Callable[[], None]] = None
        self._on_last_unsubscribe: Optional[Callable[[], None]] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Call once on FastAPI startup to bind the main event loop."""
        self._loop = loop

    def set_connection_callbacks(
        self,
        on_first_subscribe: Optional[Callable[[], None]] = None,
        on_last_unsubscribe: Optional[Callable[[], None]] = None,
    ) -> None:
        """Sets the first-connect/last-disconnect hooks. The caller should handle reentrancy and
        exceptions -- the bus does not catch them."""
        self._on_first_subscribe = on_first_subscribe
        self._on_last_unsubscribe = on_last_unsubscribe

    def connection_count(self) -> int:
        with self._lock:
            return len(self._queues)

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        with self._lock:
            was_empty = len(self._queues) == 0
            self._queues.add(q)
        if was_empty and self._on_first_subscribe is not None:
            self._on_first_subscribe()
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._queues.discard(q)
            now_empty = len(self._queues) == 0
        if now_empty and self._on_last_unsubscribe is not None:
            self._on_last_unsubscribe()

    def publish(self, event: dict[str, Any]) -> None:
        """Thread-safe: sync code (e.g. the supervisor thread) can call this too."""
        loop = self._loop
        with self._lock:
            queues = list(self._queues)
        if not loop or not queues:
            return
        for q in queues:
            try:
                loop.call_soon_threadsafe(_safe_put, q, event)
            except RuntimeError:
                # the loop has already stopped
                pass


def _safe_put(q: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        # B-1.5: slow consumer: drop it, don't block the publisher; but this must be logged
        # (previously it dropped silently -> the frontend's progress/task_state_changed looked
        # like it was dropping frames, but backend logs were clean enough to hide the problem).
        # Uses the module logger through studio.log, which carries its own trace_id (if the
        # publisher is within a request context).
        _drop_logger.warning(
            "event_bus slow consumer: queue full, dropped event type=%s",
            event.get("type", "?"),
        )


# In-process singleton (used by server.py)
bus = EventBus()
