"""PP9 -- the BooruClient pool: separated token buckets / concurrent workers / 429 backoff.

No real HTTP is sent; monkeypatch replaces booru_api.search_posts /
download_image to verify routing (API/CDN buckets) + timing + 429 adaptation.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from studio.services.booru import api as booru_api, pool as booru_pool


# ---------------------------------------------------------------------------
# host classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,is_cdn",
    [
        ("https://img3.gelbooru.com/images/abc.png", True),
        ("https://video-cdn3.gelbooru.com/x.mp4", True),
        ("https://cdn.donmai.us/original/a/b.jpg", True),
        ("https://gelbooru.com/index.php", False),
        ("https://danbooru.donmai.us/posts.json", False),
    ],
)
def test_is_cdn_host_classifies_correctly(url: str, is_cdn: bool) -> None:
    assert booru_pool.is_cdn_host(url) is is_cdn


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


def test_token_bucket_enforces_minimum_interval() -> None:
    """rate=10/s -> two consecutive acquire calls are spaced >= 0.1s apart."""
    bucket = booru_pool.TokenBucket(rate_per_sec=10.0)
    bucket.acquire()  # the first call waits zero time (next_time=0)
    t0 = time.monotonic()
    bucket.acquire()
    bucket.acquire()
    elapsed = time.monotonic() - t0
    # each of the two subsequent acquire calls waits 0.1s
    assert elapsed >= 0.18, f"expected >= 0.18s, got {elapsed:.3f}s"


def test_token_bucket_set_rate_takes_effect() -> None:
    bucket = booru_pool.TokenBucket(rate_per_sec=100.0)
    bucket.set_rate(5.0)
    assert bucket.interval == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# BooruClient
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_client(monkeypatch: pytest.MonkeyPatch):
    """A high-speed bucket + mock booru_api, avoiding real waits and real requests."""
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=4,
        api_rate_per_sec=100.0,
        cdn_rate_per_sec=100.0,
        backoff_on_429=0.1,  # sped up for the test
    )
    return booru_pool.BooruClient(cfg)


def test_search_uses_api_bucket(monkeypatch: pytest.MonkeyPatch, fast_client) -> None:
    """search_posts calls the underlying booru_api.search_posts; goes through the API bucket."""
    seen = []

    def fake_search(api_source, query, **kw):
        seen.append((api_source, query, kw.get("page", 1)))
        return [{"id": 1, "file_url": "https://img/a.jpg"}]

    monkeypatch.setattr(booru_api, "search_posts", fake_search)
    out = fast_client.search_posts("gelbooru", "1girl", page=2, user_id="u", api_key="k")
    assert out == [{"id": 1, "file_url": "https://img/a.jpg"}]
    assert seen == [("gelbooru", "1girl", 2)]
    fast_client.close()


def test_download_uses_cdn_bucket(
    monkeypatch: pytest.MonkeyPatch, fast_client, tmp_path: Path
) -> None:
    seen: list[str] = []

    def fake_download(url, save_path, **kw):
        seen.append(url)
        save_path.write_bytes(b"x")
        return save_path

    monkeypatch.setattr(booru_api, "download_image", fake_download)
    out = fast_client.download_image(
        "https://img3.gelbooru.com/x.png",
        tmp_path / "x.png",
        convert_to_png=False,
        remove_alpha_channel=False,
    )
    assert out.exists()
    assert seen == ["https://img3.gelbooru.com/x.png"]
    fast_client.close()


def test_buckets_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """API rate=1/s + CDN rate=100/s -> 100 image fetches + 1 search take < 1s total (the API doesn't slow down the CDN)."""
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=4, api_rate_per_sec=1.0, cdn_rate_per_sec=100.0
    )
    client = booru_pool.BooruClient(cfg)

    monkeypatch.setattr(booru_api, "search_posts", lambda *a, **k: [])
    monkeypatch.setattr(booru_api, "download_image", lambda u, p, **k: p.write_bytes(b"x") or p)

    t0 = time.monotonic()
    # 1 search (consumes an API token) + 5 concurrent image fetches
    client.search_posts("gelbooru", "x", user_id="u", api_key="k")
    items = list(range(5))
    paths = [Path(f"/tmp/dummy_{i}.png") for i in items]
    # but use a tmp directory for paths to avoid actually writing to /tmp
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        paths = [Path(td) / f"x{i}.png" for i in items]
        client.parallel_download(
            list(zip(items, paths)),
            lambda pair: client.download_image(
                f"https://img3.gelbooru.com/{pair[0]}.png",
                pair[1],
                convert_to_png=False,
                remove_alpha_channel=False,
            ),
        )
    elapsed = time.monotonic() - t0
    # a single search isn't affected by the API's 1/s rate limit (the first call waits zero time); CDN's 100/s is barely limiting
    assert elapsed < 1.0, f"took {elapsed:.2f}s, the buckets may not be separated"
    client.close()


def test_429_triggers_backoff_and_halves_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=2,
        api_rate_per_sec=10.0,
        cdn_rate_per_sec=10.0,
        backoff_on_429=0.05,
    )
    client = booru_pool.BooruClient(cfg)

    # simulate a search raising HTTPError(status=429)
    resp = MagicMock()
    resp.status_code = 429
    err = requests.HTTPError("429"); err.response = resp

    def fake_search(*_a, **_k):
        raise err

    monkeypatch.setattr(booru_api, "search_posts", fake_search)
    with pytest.raises(requests.HTTPError):
        client.search_posts("gelbooru", "x", user_id="u", api_key="k")
    # the rate should now be halved
    assert client.cfg.api_rate_per_sec == pytest.approx(5.0)
    assert client.cfg.cdn_rate_per_sec == pytest.approx(5.0)
    # a second 429 should not halve it again
    with pytest.raises(requests.HTTPError):
        client.search_posts("gelbooru", "x", user_id="u", api_key="k")
    assert client.cfg.api_rate_per_sec == pytest.approx(5.0)
    client.close()


def test_503_triggers_backoff_but_does_not_halve_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """503 means the server is transiently unavailable (not "we're too fast") -- enters sticky mode but doesn't reduce the rate."""
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=2,
        api_rate_per_sec=10.0,
        cdn_rate_per_sec=10.0,
        backoff_on_429=0.05,
        backoff_on_503=0.05,
    )
    client = booru_pool.BooruClient(cfg)

    resp = MagicMock()
    resp.status_code = 503
    err = requests.HTTPError("503"); err.response = resp

    def fake_download(*_a, **_k):
        raise err

    monkeypatch.setattr(booru_api, "download_image", fake_download)
    with pytest.raises(requests.HTTPError):
        client.download_image(
            "https://img3.gelbooru.com/x.png",
            Path("/tmp/x.png"),
            convert_to_png=False,
            remove_alpha_channel=False,
        )
    # 503 doesn't reduce the rate
    assert client.cfg.api_rate_per_sec == pytest.approx(10.0)
    assert client.cfg.cdn_rate_per_sec == pytest.approx(10.0)
    # but the CDN sticky state is now set
    assert client._backoff_until["cdn"] > 0
    # the API side is unaffected (independent backoff)
    assert client._backoff_until["api"] == 0.0
    client.close()


def test_cdn_backoff_does_not_block_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CDN 503 shouldn't make a subsequent search (an API host) wait out the sticky window too."""
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=2,
        api_rate_per_sec=100.0,
        cdn_rate_per_sec=100.0,
        backoff_on_429=10.0,  # deliberately long, to prove the API isn't dragged by it
        backoff_on_503=10.0,
    )
    client = booru_pool.BooruClient(cfg)

    # CDN raises 503, triggering a 10s cdn sticky window
    resp = MagicMock(); resp.status_code = 503
    err = requests.HTTPError("503"); err.response = resp
    monkeypatch.setattr(booru_api, "download_image", lambda *_a, **_k: (_ for _ in ()).throw(err))
    monkeypatch.setattr(booru_api, "search_posts", lambda *_a, **_k: [{"id": 1}])

    with pytest.raises(requests.HTTPError):
        client.download_image(
            "https://img3.gelbooru.com/x.png",
            Path("/tmp/x.png"),
            convert_to_png=False,
            remove_alpha_channel=False,
        )
    # the subsequent API search should return immediately (not wait 10s)
    t0 = time.monotonic()
    client.search_posts("gelbooru", "x", user_id="u", api_key="k")
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"the API was dragged down by the CDN backoff, took {elapsed:.2f}s"
    client.close()


def test_burst_503_in_same_window_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Multiple workers firing 503 back-to-back within the same backoff window should log only once (avoid flooding the log).

    Simulates the original bug: 4 workers still in flight when the sticky
    window is set, all coming back with 503. This fires _trigger_backoff
    directly, back-to-back, to simulate concurrency; on the normal download
    path, _wait_if_backoff makes subsequent workers sleep out the window
    before triggering again, which is a genuinely "new event" that should
    be logged.
    """
    cfg = booru_pool.BooruPoolConfig(backoff_on_503=10.0)  # a window large enough
    client = booru_pool.BooruClient(cfg)
    caplog.set_level("WARNING", logger="studio.services.booru.pool")

    for _ in range(10):
        client._trigger_backoff(503, "cdn")

    sticky_logs = [r for r in caplog.records if "sticky backoff" in r.message]
    assert len(sticky_logs) == 1, (
        f"503 within the same window should log only once, got {len(sticky_logs)}: "
        f"{[r.message for r in sticky_logs]}"
    )
    client.close()


def test_parallel_download_respects_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """parallel_workers=2 -> the number of concurrently active fn calls is <= 2."""
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=2, api_rate_per_sec=100.0, cdn_rate_per_sec=100.0
    )
    client = booru_pool.BooruClient(cfg)

    active = 0
    max_active = 0
    lock = threading.Lock()

    def slow_fn(_item):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return "ok"

    out = client.parallel_download(list(range(10)), slow_fn)
    assert len(out) == 10
    assert all(r[2] is None for r in out)  # no exceptions
    assert max_active <= 2
    client.close()


def test_parallel_download_cancel_event(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=2, api_rate_per_sec=100.0, cdn_rate_per_sec=100.0
    )
    client = booru_pool.BooruClient(cfg)

    cancel = threading.Event()

    def fn(item):
        if item == 1:
            cancel.set()
        time.sleep(0.02)
        return item

    out = client.parallel_download(list(range(10)), fn, cancel_event=cancel)
    # at least the first few run; the rest get caught by cancel
    canceled = [r for r in out if isinstance(r[2], RuntimeError)]
    assert len(canceled) > 0
    client.close()


def test_context_manager_closes_resources() -> None:
    cfg = booru_pool.BooruPoolConfig(parallel_workers=2)
    with booru_pool.BooruClient(cfg) as client:
        assert client._executor is not None
    # calling close again after it's already closed should be idempotent
    client.close()


def test_external_session_not_owned() -> None:
    """When an external session is passed in, client.close should not close it."""
    sess = requests.Session()
    closed = []

    real_close = sess.close

    def track_close():
        closed.append(True)
        real_close()

    sess.close = track_close  # type: ignore[method-assign]
    client = booru_pool.BooruClient(session=sess)
    client.close()
    assert closed == [], "an external session should not be closed"
    sess.close = real_close  # type: ignore[method-assign]
