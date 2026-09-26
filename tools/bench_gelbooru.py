"""Three-step localization of the gelbooru download speed bottleneck: network / upstream rate limit / Studio code.

How to run (from the repo root):
    venv/bin/python scripts/bench_gelbooru.py
    # Windows: venv\\Scripts\\python.exe scripts\\bench_gelbooru.py

Credentials are read automatically from studio_data/secrets.json; if not found, falls back to the
GELBOORU_USER_ID / GELBOORU_API_KEY environment variables. Logs go to both stdout and bench_gelbooru.log.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOG_PATH = REPO_ROOT / "bench_gelbooru.log"
SAMPLE_SIZE = 20  # how many images to pull for the serial / parallel comparison
TAGS = "1girl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("bench")

HEADERS = {
    "Referer": "https://gelbooru.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def load_credentials() -> tuple[str, str]:
    secrets_path = REPO_ROOT / "studio_data" / "secrets.json"
    if secrets_path.exists():
        try:
            data = json.loads(secrets_path.read_text(encoding="utf-8"))
            g = data.get("gelbooru") or {}
            uid, key = g.get("user_id", ""), g.get("api_key", "")
            if uid and key:
                log.info("Credentials read from studio_data/secrets.json")
                return uid, key
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to read secrets.json: %s", exc)
    uid = os.environ.get("GELBOORU_USER_ID", "")
    key = os.environ.get("GELBOORU_API_KEY", "")
    if uid and key:
        log.info("Credentials read from environment variables")
        return uid, key
    log.error(
        "Gelbooru credentials not found. Make sure studio_data/secrets.json has "
        "user_id/api_key configured, or export the GELBOORU_USER_ID / GELBOORU_API_KEY environment variables."
    )
    sys.exit(1)


def fetch_post_urls(user_id: str, api_key: str, limit: int) -> list[str]:
    log.info("Fetching metadata for %d posts ...", limit)
    r = requests.get(
        "https://gelbooru.com/index.php",
        params={
            "page": "dapi", "s": "post", "q": "index", "json": "1",
            "tags": TAGS, "pid": 0, "limit": limit,
            "user_id": user_id, "api_key": api_key,
        },
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    posts = r.json().get("post") or []
    urls = [p["file_url"] for p in posts if p.get("file_url")]
    log.info("Got %d file_urls", len(urls))
    if not urls:
        log.error("Got zero URLs, credentials or tag are likely wrong, aborting")
        sys.exit(1)
    return urls


def ping_host(host: str, port: int = 443, attempts: int = 5) -> None:
    """TCP connect RTT -- doesn't rely on ICMP (cloud datacenters often block ping), measures TCP handshake time instead."""
    samples: list[float] = []
    for _ in range(attempts):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=10):
                samples.append((time.perf_counter() - t0) * 1000)
        except OSError as exc:
            log.warning("  TCP connect %s failed: %s", host, exc)
    if samples:
        log.info(
            "  TCP RTT %s:%d -> min=%.0fms median=%.0fms max=%.0fms (n=%d)",
            host, port, min(samples), statistics.median(samples), max(samples), len(samples),
        )


# ---------------------------------------------------------------------------
# (1) Single-image raw network (no Session, no concurrency, closest to curl)
# ---------------------------------------------------------------------------


def step1_raw_network(urls: list[str]) -> None:
    log.info("=" * 70)
    log.info("(1) Single-image raw network speed (new connection per image, no session reuse)")
    log.info("=" * 70)

    api_host = "gelbooru.com"
    cdn_host = urlparse(urls[0]).hostname or ""
    log.info("API host: %s", api_host)
    log.info("CDN host: %s", cdn_host)
    ping_host(api_host)
    if cdn_host and cdn_host != api_host:
        ping_host(cdn_host)

    # Test the first 3 images, measure speed = bytes / total_time
    speeds: list[float] = []
    for i, u in enumerate(urls[:3]):
        t0 = time.perf_counter()
        try:
            resp = requests.get(u, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            content = resp.content
        except Exception as exc:  # noqa: BLE001
            log.warning("  Image %d failed: %s", i + 1, exc)
            continue
        elapsed = time.perf_counter() - t0
        size = len(content)
        speed = size / elapsed if elapsed > 0 else 0
        speeds.append(speed)
        log.info(
            "  Image %d  size=%.2fMB  time=%.2fs  speed=%.2f MB/s  (%s)",
            i + 1, size / 1e6, elapsed, speed / 1e6, urlparse(u).hostname,
        )
    if speeds:
        log.info("  Average per-image speed %.2f MB/s", statistics.mean(speeds) / 1e6)


# ---------------------------------------------------------------------------
# (2) Serial vs parallel (bare requests, bypassing Studio)
# ---------------------------------------------------------------------------


def step2_serial_vs_parallel(urls: list[str]) -> tuple[float, float]:
    log.info("=" * 70)
    log.info("(2) Bare requests serial vs parallel comparison (n=%d)", len(urls))
    log.info("=" * 70)

    sess1 = requests.Session()
    t0 = time.perf_counter()
    total = 0
    fail = 0
    for u in urls:
        try:
            r = sess1.get(u, headers=HEADERS, timeout=60)
            r.raise_for_status()
            total += len(r.content)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            log.warning("  serial failed: %s", exc)
    serial_t = time.perf_counter() - t0
    sess1.close()
    log.info(
        "  SERIAL   : %d imgs / %d fail in %.1fs  (%.2f MB/s, %.2f img/s)",
        len(urls) - fail, fail, serial_t,
        total / serial_t / 1e6 if serial_t else 0,
        (len(urls) - fail) / serial_t if serial_t else 0,
    )

    sess2 = requests.Session()

    def fetch(u: str) -> int:
        r = sess2.get(u, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return len(r.content)

    t0 = time.perf_counter()
    total = 0
    fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(fetch, u) for u in urls]
        for f in concurrent.futures.as_completed(futs):
            try:
                total += f.result()
            except Exception as exc:  # noqa: BLE001
                fail += 1
                log.warning("  parallel failed: %s", exc)
    para_t = time.perf_counter() - t0
    sess2.close()
    log.info(
        "  PARALLEL4: %d imgs / %d fail in %.1fs  (%.2f MB/s, %.2f img/s)",
        len(urls) - fail, fail, para_t,
        total / para_t / 1e6 if para_t else 0,
        (len(urls) - fail) / para_t if para_t else 0,
    )
    if para_t > 0:
        log.info("  speedup  : %.2fx", serial_t / para_t)
    return serial_t, para_t


# ---------------------------------------------------------------------------
# (3) Studio's actual code (PP9 BooruClient)
# ---------------------------------------------------------------------------


def step3_studio_client(urls: list[str]) -> None:
    log.info("=" * 70)
    log.info("(3) Studio BooruClient (PP9 real code path)")
    log.info("=" * 70)
    try:
        from studio.services.booru.pool import BooruClient, BooruPoolConfig
    except Exception as exc:  # noqa: BLE001
        log.error("import studio.services.booru_pool failed: %s", exc)
        return

    cfg = BooruPoolConfig(parallel_workers=4, api_rate_per_sec=2.0, cdn_rate_per_sec=5.0)
    log.info(
        "  cfg: workers=%d api_rate=%.1f/s cdn_rate=%.1f/s",
        cfg.parallel_workers, cfg.api_rate_per_sec, cfg.cdn_rate_per_sec,
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        items = [(i, u, td_path / f"{i}.bin") for i, u in enumerate(urls)]
        with BooruClient(cfg) as client:
            t0 = time.perf_counter()

            def dl(item: tuple[int, str, Path]) -> Path:
                _, u, p = item
                return client.download_image(
                    u, p,
                    convert_to_png=False,
                    remove_alpha_channel=False,
                    referer="https://gelbooru.com/",
                )

            results = client.parallel_download(items, dl)
            elapsed = time.perf_counter() - t0

        ok = sum(1 for _, _, exc in results if exc is None)
        fail = len(results) - ok
        total = sum(p.stat().st_size for _, _, p in items if p.exists())
        log.info(
            "  PP9 BooruClient: %d ok / %d fail in %.1fs  (%.2f MB/s, %.2f img/s)",
            ok, fail, elapsed,
            total / elapsed / 1e6 if elapsed else 0,
            ok / elapsed if elapsed else 0,
        )
        for _, _, exc in results[:3]:
            if exc is not None:
                log.warning("  Example error: %s", exc)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("bench_gelbooru.py -- writing log to %s", LOG_PATH)
    log.info("Python: %s", sys.version.split()[0])
    log.info("requests: %s", requests.__version__)

    user_id, api_key = load_credentials()
    urls = fetch_post_urls(user_id, api_key, SAMPLE_SIZE)

    step1_raw_network(urls)
    step2_serial_vs_parallel(urls)
    step3_studio_client(urls)

    log.info("=" * 70)
    log.info("Done. Paste back the whole %s file.", LOG_PATH.name)


if __name__ == "__main__":
    main()
