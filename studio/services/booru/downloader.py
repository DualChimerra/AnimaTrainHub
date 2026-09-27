"""Gelbooru / Danbooru download library (pp2 + pp9).

Turned into a library from the original `danbooru_downloader.py`: dropped
input() / JSON config files, all parameters now go through `DownloadOptions`;
progress is pushed back to the caller via `on_progress(line)` (the worker
forwards it to logs + bus.publish).

PP9 rework:
- Image fetching is now concurrent, going through `BooruClient` (dual token
  bucket: API 2 / CDN 5 req/s by default)
- Removed the hard 0.5s sleep per image; rate is now controlled by the token
  bucket instead
- Kept `page_delay` (a polite 1s wait between pages)

Design:
- `download(opts, dest_dir, on_progress, on_image_saved, cancel_event, client)`
  downloads in blocking mode, returns the number of images successfully saved.
- Retries on failure 3 times (exponential backoff 1s/2s/4s), timeout 60s.
- Cancellation: `cancel_event.is_set()` is checked before every image / every
  page; once triggered, it returns the current saved count immediately, with
  no exception raised.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

from . import api as booru_api, pool as booru_pool
from ..proxy_manager import get_proxy_dict

ProgressFn = Callable[[str], None]
ImageSavedFn = Callable[[Path], None]


@dataclass
class DownloadOptions:
    tag: str
    count: int = 20
    api_source: str = "gelbooru"  # "gelbooru" | "danbooru"
    save_tags: bool = False
    convert_to_png: bool = True
    remove_alpha_channel: bool = False
    skip_existing: bool = True
    # gelbooru credentials
    user_id: str = ""
    # danbooru credentials
    username: str = ""
    # shared api key (both gelbooru / danbooru use .api_key)
    api_key: str = ""
    # global excluded tags (auto-appended as -tag when searching); from secrets.download.exclude_tags
    exclude_tags: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.exclude_tags is None:
            self.exclude_tags = []

    def base_url(self) -> str:
        return booru_api.default_base_url(self.api_source)

    def effective_tag_query(self) -> str:
        """Append -excluded after `tag` (same syntax for gelbooru / danbooru)."""
        parts = [self.tag.strip()]
        for ex in self.exclude_tags:
            ex = ex.strip().lstrip("-")
            if ex:
                parts.append(f"-{ex}")
        return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------


def download(
    opts: DownloadOptions,
    dest_dir: Path,
    *,
    on_progress: ProgressFn = print,
    on_image_saved: Optional[ImageSavedFn] = None,
    cancel_event: Optional[threading.Event] = None,
    session: Optional[requests.Session] = None,
    client: Optional[booru_pool.BooruClient] = None,
    page_delay: float = 1.0,
    max_retries: int = 3,
) -> int:
    """Blocking download into dest_dir.

    Returns the number of newly saved images this call (excludes skips). If
    interrupted (cancel_event triggered), returns the current saved count
    with no error raised.

    PP9: image fetching is concurrent (4 workers by default); rate is
    controlled by BooruClient's token bucket (API 2 / CDN 5 req/s).
    `session=` still accepts an external session (used by legacy tests);
    internally it's wrapped via `client.search_posts/download_image`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not opts.tag.strip():
        raise ValueError("tag must not be empty")
    if opts.count <= 0:
        raise ValueError("count must be > 0")
    if opts.api_source == "gelbooru" and not (opts.user_id and opts.api_key):
        raise ValueError(
            "gelbooru needs user_id + api_key (set secrets.gelbooru in Settings)"
        )
    if opts.api_source == "danbooru" and not (opts.username and opts.api_key):
        # hotfix: since danbooru went behind Cloudflare, an anonymous UA is no
        # longer reliable (even with our app UA, CF could tighten up at any
        # time); force binding an account so the UA carries (by username) —
        # when CF blocks anonymous requests we won't get swept in, and
        # danbooru also rate-limits per account (standard 2 req/s).
        raise ValueError(
            "danbooru needs username + api_key (set secrets.danbooru in Settings)"
        )

    # If no client was passed, build a temporary one (rate-tuned via
    # secrets.download.*); session takes priority
    owns_client = False
    if client is None:
        try:
            from .. import secrets as _secrets
            d = _secrets.load().download
            cfg = booru_pool.BooruPoolConfig(
                parallel_workers=d.parallel_workers,
                api_rate_per_sec=d.api_rate_per_sec,
                cdn_rate_per_sec=d.cdn_rate_per_sec,
            )
        except Exception:  # noqa: BLE001
            cfg = booru_pool.BooruPoolConfig()
        client = booru_pool.BooruClient(cfg, session=session)
        owns_client = True

    try:
        return _download_with_client(
            opts, dest_dir, client,
            on_progress=on_progress,
            on_image_saved=on_image_saved,
            cancel_event=cancel_event,
            page_delay=page_delay,
            max_retries=max_retries,
        )
    finally:
        if owns_client:
            client.close()


def _download_with_client(
    opts: DownloadOptions,
    dest_dir: Path,
    client: booru_pool.BooruClient,
    *,
    on_progress: ProgressFn,
    on_image_saved: Optional[ImageSavedFn],
    cancel_event: Optional[threading.Event],
    page_delay: float,
    max_retries: int,
) -> int:
    saved = 0
    skipped = 0
    failed = 0
    page = 1
    api_limit = 100 if opts.api_source == "gelbooru" else 200
    # An "emitted" counter shared across worker threads, only used by
    # _fetch_one to print live [N/count] progress. Matches the main thread's
    # `saved` on the normal path; not incremented on retry / failure.
    emit_lock = threading.Lock()
    emit_state = {"n": 0}

    while saved < opts.count:
        if cancel_event and cancel_event.is_set():
            on_progress("[cancel] user requested stop")
            return saved
        on_progress(f"[page {page}] fetching ...")
        try:
            posts = client.search_posts(
                opts.api_source,
                opts.effective_tag_query(),
                page=page,
                limit=api_limit,
                user_id=opts.user_id,
                api_key=opts.api_key,
                username=opts.username,
            )
        except requests.RequestException as exc:
            on_progress(f"[err] search failed: {exc}")
            return saved
        if not posts:
            on_progress("[done] no more posts (server returned empty page)")
            break

        # Collect all "to-download" candidates on this page; fetch them concurrently
        candidates: list[tuple[str, str, str, Optional[str], Path]] = []
        page_valid = 0
        for post in posts:
            if saved + len(candidates) >= opts.count:
                break
            post_id, file_url, file_ext, tags_str = booru_api.post_fields(
                post, opts.api_source
            )
            if not post_id or not file_url:
                continue
            page_valid += 1
            ext = "png" if opts.convert_to_png else file_ext
            target = dest_dir / f"{post_id}.{ext}"
            if opts.skip_existing and target.exists():
                skipped += 1
                on_progress(f"[skip] {target.name} already exists")
                continue
            candidates.append((post_id, file_url, file_ext, tags_str, target))

        # Image fetch function (with retry on failure); each image is scheduled independently to the worker pool
        referer = opts.base_url() + "/"

        def _fetch_one(item: tuple[str, str, str, Optional[str], Path]) -> Path:
            post_id, file_url, _file_ext, _tags, target = item
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("canceled")
                try:
                    final = client.download_image(
                        file_url,
                        target,
                        convert_to_png=opts.convert_to_png,
                        remove_alpha_channel=opts.remove_alpha_channel,
                        referer=referer,
                        username=opts.username,
                    )
                    # Live progress (inside the worker thread): emit immediately
                    # after each image finishes, so LogTailer can push SSE to
                    # the frontend. Otherwise parallel_download would just
                    # block the whole time, with no logs during the entire
                    # download phase, making the frontend look "stuck".
                    with emit_lock:
                        emit_state["n"] += 1
                        n = emit_state["n"]
                    on_progress(f"[{n}/{opts.count}] saved {final.name}")
                    return final
                except requests.RequestException as exc:
                    last_exc = exc
                    backoff = 2 ** (attempt - 1)
                    on_progress(
                        f"[retry {attempt}/{max_retries}] {target.name}: {exc}"
                    )
                    if cancel_event and cancel_event.wait(backoff):
                        raise RuntimeError("canceled") from exc
            raise RuntimeError(f"max_retries exceeded: {last_exc}")

        results = client.parallel_download(
            candidates, _fetch_one, cancel_event=cancel_event
        )

        for (post_id, _url, _ext, tags_str, target), final, exc in results:
            if cancel_event and cancel_event.is_set():
                on_progress("[cancel] user requested stop")
                return saved
            if exc is not None:
                # A RuntimeError("canceled") raised from _fetch_one due to
                # cancellation isn't reported as a "download failed" —
                # otherwise the user would see a pile of [err] on cancel.
                if isinstance(exc, RuntimeError) and "canceled" in str(exc):
                    continue
                on_progress(f"[err] {target.name}: {exc}")
                failed += 1
                continue
            assert isinstance(final, Path)
            if opts.save_tags and tags_str:
                final.with_suffix(".booru.txt").write_text(
                    str(tags_str), encoding="utf-8"
                )
            if on_image_saved:
                on_image_saved(final)
            saved += 1
            # The progress line was already emitted live inside _fetch_one;
            # here we just do the bookkeeping count / break early.
            if saved >= opts.count:
                break

        if len(posts) < api_limit:
            on_progress(
                f"[done] page returned {len(posts)} < limit {api_limit}, "
                "reached end"
            )
            break
        if page_valid == 0:
            on_progress("[done] no valid posts on this page; stopping")
            break
        page += 1
        if cancel_event and cancel_event.wait(page_delay):
            on_progress("[cancel] user requested stop")
            return saved

    on_progress(
        f"[summary] saved={saved} skipped={skipped} failed={failed}"
    )
    return saved


def estimate(opts: DownloadOptions) -> int:
    """Lightweight API call estimating a tag's (including exclude) hit count;
    returns -1 (unknown) on failure.

    v0.5.2 hotfix gap: search_posts already goes through booru_api's UA /
    Accept headers to get past CF, but this separate "lightweight" estimate
    path was still a bare requests.get (default UA python-requests/X.Y.Z)
    -> danbooru's CF treats it as a bot and blocks it -> raises an exception
    -> always returns -1 (unknown). Fix: reuse the same UA as
    booru_api._api_headers; also pass basic auth when available on the
    danbooru side so rate limiting is counted per account.
    """
    query = opts.effective_tag_query()
    proxies = get_proxy_dict()
    headers = booru_api._api_headers(opts.username)
    if opts.api_source == "gelbooru":
        try:
            params: dict[str, Any] = {
                "page": "dapi",
                "s": "post",
                "q": "index",
                "json": "1",
                "tags": query,
                "pid": 0,
                "limit": 1,
            }
            if opts.api_key and opts.user_id:
                params["api_key"] = opts.api_key
                params["user_id"] = opts.user_id
            r = requests.get(
                f"{opts.base_url()}/index.php",
                params=params, headers=headers, timeout=15, proxies=proxies
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "@attributes" in data:
                return int(data["@attributes"].get("count", -1))
        except Exception:
            return -1
        return -1
    try:
        auth = (opts.username, opts.api_key) if opts.username and opts.api_key else None
        r = requests.get(
            f"{opts.base_url()}/counts/posts.json",
            params={"tags": query},
            headers=headers, auth=auth, timeout=15, proxies=proxies
        )
        r.raise_for_status()
        return int(r.json().get("counts", {}).get("posts", -1))
    except Exception:
        return -1
