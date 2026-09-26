"""PP9 — Unified Booru API pool: Session keepalive + dual token bucket + concurrent image fetching + 429 backoff.

Why a pool is needed:
- `downloader.py` as it stands is fully synchronous/serial with a forced
  0.5s sleep per image — 5-10x slower in the cloud than on home broadband
- Rate limiting mostly falls on `gelbooru.com`; the CDN image domains
  (`img*.gelbooru.com`) allow a much looser rate
- When multiple tasks (download + reg_build) run at the same time, they
  don't know about each other's progress, making it easy to hit rate walls

Design:
- Dual token bucket: API host 2 req/s + CDN host 5 req/s (distinguished by
  the URL's netloc)
- ThreadPoolExecutor with 4 workers by default (matches the CDN bucket)
- On 429/503 -> sticky backoff of 60s + rate halved, permanent until the
  client is destroyed (conservative)
- `requests.Session` reuses TCP/TLS (HTTP keepalive alone is ~2x faster)

Public interface parallels `booru_api.py`: `search_posts` / `download_image` /
`parallel_download`. The raw `booru_api.search_posts(...)` can still be called
directly (for tests / legacy-code compatibility) — the pool is just a thin
wrapper on top.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import urlparse

import requests

from . import api as booru_api
from ..proxy_manager import patch_requests_session

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# host classification
# ---------------------------------------------------------------------------


# CDN host matching (both gelbooru / danbooru host images on subdomains):
# gelbooru: img3.gelbooru.com / video-cdn3.gelbooru.com / ...
# danbooru: cdn.donmai.us / sample.donmai.us / ...
_CDN_HOST_HINTS = ("cdn", "img", "video-cdn", "raikou", "sample")


def is_cdn_host(url: str) -> bool:
    """Determine by netloc whether this is a CDN (image host); everything else counts as an API host."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return any(h in host for h in _CDN_HOST_HINTS)


# ---------------------------------------------------------------------------
# token bucket
# ---------------------------------------------------------------------------


class TokenBucket:
    """A simple time-windowed token bucket: at most `rate` tokens per second, thread-safe.

    `acquire()` blocks until a token is available; no burst support (consecutive requests naturally get spaced out evenly).
    """

    def __init__(self, rate_per_sec: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self._interval = 1.0 / rate_per_sec
        self._next_time = 0.0
        self._lock = threading.Lock()

    @property
    def interval(self) -> float:
        return self._interval

    def set_rate(self, rate_per_sec: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        with self._lock:
            self._interval = 1.0 / rate_per_sec

    def acquire(self) -> None:
        """Block until the next token is available."""
        with self._lock:
            now = time.monotonic()
            wait = self._next_time - now
            if wait > 0:
                # Sleep while holding the lock — crude but simple; contention is acceptable when rate is small
                time.sleep(wait)
                now = time.monotonic()
            self._next_time = max(now, self._next_time) + self._interval


# ---------------------------------------------------------------------------
# config + client
# ---------------------------------------------------------------------------


@dataclass
class BooruPoolConfig:
    parallel_workers: int = 4
    api_rate_per_sec: float = 2.0
    cdn_rate_per_sec: float = 5.0
    backoff_on_429: float = 60.0
    # 503 = server temporarily unavailable, purely a transient issue; should not sticky + slow down for as long as 429 does
    backoff_on_503: float = 15.0


class BooruClient:
    """Session + ThreadPoolExecutor + dual token bucket + 429 sticky backoff.

    Usage pattern:
        with BooruClient() as client:
            posts = client.search_posts("gelbooru", "1girl", api_key=..., user_id=...)
            results = client.parallel_download(items, lambda item: client.download_image(...))

    Thread-safe; the same instance can be called concurrently by downloader / reg_builder.
    """

    def __init__(
        self,
        cfg: Optional[BooruPoolConfig] = None,
        *,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.cfg = cfg or BooruPoolConfig()
        self._session = session or requests.Session()
        patch_requests_session(self._session)
        # An externally passed-in session isn't necessarily a requests.Session
        # (tests use a minimal FakeSession that only implements .get)
        logger.info("BooruClient session proxies: %s", getattr(self._session, "proxies", {}))
        self._owns_session = session is None
        self._api_bucket = TokenBucket(self.cfg.api_rate_per_sec)
        self._cdn_bucket = TokenBucket(self.cfg.cdn_rate_per_sec)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, self.cfg.parallel_workers),
            thread_name_prefix="booru-pool",
        )
        # Sticky backoff state — API / CDN are independent, to avoid a CDN 503 storm locking up the API host too
        self._lock = threading.Lock()
        self._backoff_until: dict[str, float] = {"api": 0.0, "cdn": 0.0}
        self._rate_halved = False

    # -------------------- lifecycle --------------------

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        if self._owns_session:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "BooruClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -------------------- sticky backoff state --------------------

    def _wait_if_backoff(self, kind: str) -> None:
        """If this host class is within a backoff window, block until the window ends.

        No logging here — `_trigger_backoff` already logged once when entering
        backoff; having every worker log here too would spam the console
        (N workers -> N identical backoff lines).
        """
        with self._lock:
            until = self._backoff_until[kind]
        wait = until - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _trigger_backoff(self, status_code: int, kind: str) -> None:
        """On receiving 429/503 -> this host class enters sticky backoff.

        - 429 (server explicitly says "too fast"): long backoff + rate
          permanently halved (once)
        - 503 (server temporarily unavailable): short backoff, rate untouched
          — server downtime isn't our problem
        - Repeated triggers within one backoff window only log once (avoids
          spamming N lines during a burst of 503s)
        """
        is_429 = status_code == 429
        backoff = self.cfg.backoff_on_429 if is_429 else self.cfg.backoff_on_503
        log_new_window = False
        do_halve = False
        with self._lock:
            # Only treat this as a "new window" -> log once if we're not currently inside a backoff window
            if self._backoff_until[kind] <= time.monotonic():
                log_new_window = True
            self._backoff_until[kind] = time.monotonic() + backoff
            if is_429 and not self._rate_halved:
                self._rate_halved = True
                self.cfg.api_rate_per_sec /= 2
                self.cfg.cdn_rate_per_sec /= 2
                do_halve = True

        if log_new_window:
            logger.warning(
                "[booru_pool] %s received %d, sticky backoff %.0fs",
                kind, status_code, backoff,
            )
        if do_halve:
            self._api_bucket.set_rate(self.cfg.api_rate_per_sec)
            self._cdn_bucket.set_rate(self.cfg.cdn_rate_per_sec)
            logger.warning(
                "[booru_pool] 429 triggered a permanent rate halving (API %.2f / CDN %.2f req/s)",
                self.cfg.api_rate_per_sec,
                self.cfg.cdn_rate_per_sec,
            )

    def _check_response(self, resp: requests.Response, kind: str) -> None:
        """429/503 triggers sticky backoff; other 4xx/5xx leave the pool untouched (let the caller raise)."""
        if resp.status_code in (429, 503):
            self._trigger_backoff(resp.status_code, kind)

    # -------------------- public API --------------------

    def search_posts(self, api_source: str, tags_query: str, **kw: Any) -> list[dict[str, Any]]:
        """Goes through the API bucket. kw is passed through to booru_api.search_posts."""
        self._wait_if_backoff("api")
        self._api_bucket.acquire()
        kw.setdefault("session", self._session)
        try:
            return booru_api.search_posts(api_source, tags_query, **kw)
        except requests.HTTPError as exc:
            if exc.response is not None:
                self._check_response(exc.response, "api")
            raise

    def download_image(self, url: str, save_path: Path, **kw: Any) -> Path:
        """Goes through the CDN bucket. kw is passed through to booru_api.download_image."""
        self._wait_if_backoff("cdn")
        self._cdn_bucket.acquire()
        kw.setdefault("session", self._session)
        try:
            return booru_api.download_image(url, save_path, **kw)
        except requests.HTTPError as exc:
            if exc.response is not None:
                self._check_response(exc.response, "cdn")
            raise

    def parallel_download(
        self,
        items: list[T],
        fn: Callable[[T], Any],
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[tuple[T, Any, Optional[Exception]]]:
        """Run fn(item) concurrently across items; returns [(item, result, exc)] in the original order.

        Exceptions don't propagate (each item reports its own error);
        handled by the caller as needed.
        cancel_event is checked twice — before submit and after result;
        once triggered, un-run futures are canceled.
        """
        if not items:
            return []
        results: list[tuple[T, Any, Optional[Exception]]] = []
        futures = []
        # Batch-check cancel before submitting
        for it in items:
            if cancel_event is not None and cancel_event.is_set():
                break
            futures.append((it, self._executor.submit(fn, it)))
        for it, fut in futures:
            if cancel_event is not None and cancel_event.is_set():
                fut.cancel()
                results.append((it, None, RuntimeError("canceled")))
                continue
            try:
                r = fut.result()
                results.append((it, r, None))
            except Exception as exc:  # noqa: BLE001
                results.append((it, None, exc))
        return results
