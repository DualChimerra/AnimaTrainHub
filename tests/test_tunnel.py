"""Phone access over a Cloudflare quick tunnel.

Two failure modes matter here and both are silent:

* the gate locking the user out of their own local studio, and
* the gate *not* closing, leaving an unauthenticated public URL onto a machine
  whose API can read the filesystem and start jobs.

Everything below is about those two. The middleware is exercised as raw ASGI —
no HTTP client, no fastapi — so the tests run in any environment.
"""
from __future__ import annotations

import subprocess

import pytest

from studio.services import tunnel


# ── ASGI harness ─────────────────────────────────────────────────────────────

async def _ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"studio"})


def call(scope_extra: dict) -> dict:
    """Run one request through the middleware; returns status/headers/body."""
    import asyncio

    scope = {"type": "http", "headers": [], "query_string": b"", **scope_extra}
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    mw = tunnel.TunnelAuthMiddleware(_ok_app)
    asyncio.run(mw(scope, receive, send))

    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return {"status": start["status"], "headers": start.get("headers", []), "body": body}


@pytest.fixture
def running(monkeypatch):
    """Pretend a tunnel is up with a known key."""
    monkeypatch.setattr(tunnel, "current_key", lambda: "secret-key")
    return "secret-key"


CF = [(b"cf-ray", b"8abc123-FRA")]


# ── the gate ─────────────────────────────────────────────────────────────────

def test_local_requests_pass_when_no_tunnel_is_running(monkeypatch):
    monkeypatch.setattr(tunnel, "current_key", lambda: None)
    assert call({})["status"] == 200


def test_local_requests_pass_while_a_tunnel_is_running(running):
    """Turning the tunnel on must never lock the user out of their own studio.
    The machine's own browser never sends CF-Ray."""
    assert call({})["status"] == 200


def test_a_cloudflare_request_without_the_key_is_refused(running):
    r = call({"headers": CF})
    assert r["status"] == 401
    assert b"Access key required" in r["body"]


def test_a_cloudflare_request_with_the_key_in_the_url_passes(running):
    r = call({"headers": CF, "query_string": b"k=secret-key"})
    assert r["status"] == 200


def test_the_key_in_the_url_sets_a_cookie(running):
    """Otherwise every asset on the page would need the key glued to its URL."""
    r = call({"headers": CF, "query_string": b"k=secret-key"})
    cookies = [v for k, v in r["headers"] if k == b"set-cookie"]
    assert any(b"anima_tunnel_key=secret-key" in c for c in cookies)


def test_a_cloudflare_request_with_the_cookie_passes(running):
    r = call({"headers": CF + [(b"cookie", b"anima_tunnel_key=secret-key")]})
    assert r["status"] == 200


def test_a_wrong_key_is_refused(running):
    assert call({"headers": CF, "query_string": b"k=wrong"})["status"] == 401
    assert call({"headers": CF + [(b"cookie", b"anima_tunnel_key=wrong")]})["status"] == 401


def test_the_cookie_from_a_previous_key_is_refused(monkeypatch):
    """Rotating the key kills every link and cookie handed out before."""
    monkeypatch.setattr(tunnel, "current_key", lambda: "new-key")
    r = call({"headers": CF + [(b"cookie", b"anima_tunnel_key=old-key")]})
    assert r["status"] == 401


def test_other_cookies_do_not_confuse_the_gate(running):
    r = call({"headers": CF + [(b"cookie", b"theme=dark; anima_tunnel_key=secret-key; x=1")]})
    assert r["status"] == 200


def test_websocket_and_lifespan_scopes_are_passed_through(running):
    """The gate only understands http; anything else must not be swallowed."""
    import asyncio

    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    async def noop(*_a):
        return {}

    mw = tunnel.TunnelAuthMiddleware(app)
    asyncio.run(mw({"type": "lifespan"}, noop, noop))
    asyncio.run(mw({"type": "websocket", "headers": CF}, noop, noop))
    assert seen == ["lifespan", "websocket"]


# ── lifecycle ────────────────────────────────────────────────────────────────

class _FakeProc:
    """A cloudflared stand-in that prints the lines the real one prints."""

    def __init__(self, lines: list[str], *, exits: bool = False):
        self.stdout = iter(lines)
        self._exits = exits
        self.terminated = False

    def poll(self):
        return 0 if self._exits else None

    def terminate(self):
        self.terminated = True
        self._exits = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._exits = True


BANNER = [
    "2026-09-16T10:00:00Z INF Thank you for trying Cloudflare Tunnel.",
    "2026-09-16T10:00:01Z INF |  https://brave-lion-tiny.trycloudflare.com  |",
    "2026-09-16T10:00:02Z INF Connection registered",
]


@pytest.fixture
def settings(monkeypatch):
    """In-memory remote_access settings instead of the real secrets.json."""
    store = {
        "provider": "cloudflare", "autostart": False, "access_key": "",
        "ngrok_authtoken": "", "ngrok_domain": "",
    }
    monkeypatch.setattr(tunnel, "load_settings", lambda: dict(store))

    def save(**fields):
        store.update(fields)
        return dict(store)

    monkeypatch.setattr(tunnel, "save_settings", save)
    return store


@pytest.fixture
def fresh_tunnel(monkeypatch, tmp_path, settings):
    t = tunnel._Tunnel()
    monkeypatch.setattr(tunnel, "_tunnel", t)
    monkeypatch.setattr(tunnel, "find_binary", lambda: tmp_path / "cloudflared")
    monkeypatch.setattr(tunnel, "START_TIMEOUT_S", 2.0)
    return t


def test_start_returns_the_url_with_the_key_attached(fresh_tunnel, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc(BANNER))
    state = tunnel.start(8765)
    assert state["running"] is True
    assert state["url"].startswith("https://brave-lion-tiny.trycloudflare.com?k=")
    assert state["port"] == 8765
    # The key handed out is the key the gate checks.
    assert state["url"].split("k=")[1] == tunnel.current_key()


def test_starting_twice_does_not_spawn_a_second_process(fresh_tunnel, monkeypatch):
    spawned = []

    def popen(*a, **k):
        spawned.append(a)
        return _FakeProc(BANNER)

    monkeypatch.setattr(subprocess, "Popen", popen)
    first = tunnel.start(8765)
    second = tunnel.start(8765)
    assert len(spawned) == 1
    assert first["url"] == second["url"]


def test_start_without_the_binary_says_so(fresh_tunnel, monkeypatch):
    monkeypatch.setattr(tunnel, "find_binary", lambda: None)
    with pytest.raises(tunnel.TunnelError, match="not installed"):
        tunnel.start(8765)


def test_start_times_out_when_no_address_appears(fresh_tunnel, monkeypatch):
    proc = _FakeProc(["INF failed to connect to the edge"])
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    with pytest.raises(tunnel.TunnelError, match="did not produce an address"):
        tunnel.start(8765)
    assert proc.terminated
    # No half-open state left behind, and no key that the gate would accept.
    assert tunnel.state()["running"] is False
    assert tunnel.current_key() is None


def test_stop_invalidates_the_key(fresh_tunnel, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc(BANNER))
    tunnel.start(8765)
    assert tunnel.current_key() is not None
    state = tunnel.stop()
    assert state["running"] is False and state["url"] is None
    assert tunnel.current_key() is None


def test_a_process_that_died_is_reported_as_stopped(fresh_tunnel, monkeypatch):
    proc = _FakeProc(BANNER)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    tunnel.start(8765)
    proc._exits = True  # cloudflared lost its connection and quit
    state = tunnel.state()
    assert state["running"] is False
    assert state["url"] is None
    assert tunnel.current_key() is None


def test_state_before_anything_happens(fresh_tunnel):
    state = tunnel.state()
    assert state["running"] is False
    assert state["url"] is None


# ── the binary ───────────────────────────────────────────────────────────────

def test_the_url_regex_matches_a_real_banner_line():
    line = "2026-09-16T10:00:01Z INF |  https://brave-lion-tiny.trycloudflare.com  |"
    assert tunnel._URL_RE.search(line).group(0) == "https://brave-lion-tiny.trycloudflare.com"


def test_install_refuses_an_unknown_platform(monkeypatch):
    monkeypatch.setattr(tunnel, "_asset_name", lambda: None)
    with pytest.raises(tunnel.TunnelError, match="No prebuilt cloudflared"):
        tunnel.install()


@pytest.mark.parametrize("system,machine,asset", [
    ("linux", "x86_64", "cloudflared-linux-amd64"),
    ("linux", "aarch64", "cloudflared-linux-arm64"),
    ("darwin", "arm64", "cloudflared-darwin-arm64.tgz"),
    ("windows", "amd64", "cloudflared-windows-amd64.exe"),
])
def test_asset_is_picked_per_platform(monkeypatch, system, machine, asset):
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("platform.machine", lambda: machine)
    assert tunnel._asset_name() == asset


# ── persistent key ───────────────────────────────────────────────────────────

def test_the_key_survives_a_restart_of_the_tunnel(fresh_tunnel, monkeypatch, settings):
    """A permanent address is only useful if the link (with its key) is too."""
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc(BANNER))
    first = tunnel.start(8765)["url"].split("k=")[1]
    tunnel.stop()
    second = tunnel.start(8765)["url"].split("k=")[1]
    assert first == second == settings["access_key"]


def test_rotating_the_key_changes_the_link(fresh_tunnel, monkeypatch, settings):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc(BANNER))
    old = tunnel.start(8765)["url"].split("k=")[1]
    new = tunnel.rotate_key()["url"].split("k=")[1]
    assert new != old
    assert tunnel.current_key() == new == settings["access_key"]


def test_the_key_is_not_accepted_while_stopped(fresh_tunnel, settings):
    settings["access_key"] = "k" * 32
    assert tunnel.current_key() is None


def test_configure_rejects_an_unknown_provider(settings):
    with pytest.raises(tunnel.TunnelError, match="Unknown"):
        tunnel.configure(provider="carrier-pigeon")


def test_configure_saves_autostart_and_provider(fresh_tunnel, settings):
    state = tunnel.configure(provider="tailscale", autostart=True)
    assert settings["provider"] == "tailscale" and settings["autostart"] is True
    assert state["provider"] == "tailscale" and state["autostart"] is True
    assert state["permanent"] is True


def test_ngrok_without_a_domain_is_not_permanent(fresh_tunnel, settings):
    settings["provider"] = "ngrok"
    assert tunnel.state()["permanent"] is False
    settings["ngrok_domain"] = "calm-otter.ngrok-free.app"
    assert tunnel.state()["permanent"] is True


def test_permanent_providers_refuse_without_the_gate(fresh_tunnel, monkeypatch, tmp_path, settings):
    """No gate → no way to tell tunnel traffic apart → must not open."""
    monkeypatch.setattr(tunnel, "find_ngrok", lambda: tmp_path / "ngrok")
    monkeypatch.setattr(tunnel._gate, "_loop", None)
    with pytest.raises(tunnel.TunnelError):
        tunnel.start(8765, "ngrok")
    assert tunnel.state()["running"] is False


def test_autostart_does_nothing_when_off(monkeypatch, settings):
    called = []
    monkeypatch.setattr(tunnel, "start", lambda *a, **k: called.append(a))
    tunnel.autostart(8765)
    assert called == []


def test_autostart_starts_in_the_background_when_on(monkeypatch, settings):
    import threading

    settings["autostart"] = True
    done = threading.Event()

    def fake_start(port, provider=None):
        done.set()
        return {"url": "https://x"}

    monkeypatch.setattr(tunnel, "start", fake_start)
    tunnel.autostart(8765)
    assert done.wait(2)


# ── the gate relay ───────────────────────────────────────────────────────────

def test_requests_through_the_gate_need_the_key(running):
    """Header-agnostic: Tailscale / ngrok proxies send no CF-Ray, but their
    connections arrive from the relay's upstream ports."""
    tunnel._gate.client_ports.add(54321)
    try:
        via_gate = {"client": ("127.0.0.1", 54321)}
        assert call(via_gate)["status"] == 401
        assert call({**via_gate, "query_string": b"k=secret-key"})["status"] == 200
        # Same machine, different port: the user's own browser.
        assert call({"client": ("127.0.0.1", 54322)})["status"] == 200
    finally:
        tunnel._gate.client_ports.discard(54321)


def test_gate_traffic_without_a_running_tunnel_is_refused(monkeypatch):
    monkeypatch.setattr(tunnel, "current_key", lambda: None)
    tunnel._gate.client_ports.add(54323)
    try:
        assert call({"client": ("127.0.0.1", 54323)})["status"] == 401
    finally:
        tunnel._gate.client_ports.discard(54323)


def test_the_gate_relays_bytes_and_records_its_upstream_port():
    """End to end on real sockets: an echo 'studio' with the relay in front."""
    import asyncio

    async def scenario():
        seen_clients = []

        async def studio(reader, writer):
            seen_clients.append(writer.get_extra_info("peername")[1])
            data = await reader.read(100)
            writer.write(b"echo:" + data)
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(studio, "127.0.0.1", 0)
        studio_port = server.sockets[0].getsockname()[1]
        gate = tunnel._Gate()
        gate.bind_loop(asyncio.get_running_loop())
        gate_port = await gate._open(studio_port)

        reader, writer = await asyncio.open_connection("127.0.0.1", gate_port)
        writer.write(b"hello")
        await writer.drain()
        for _ in range(50):
            if seen_clients:
                break
            await asyncio.sleep(0.02)
        marked = gate.is_external_client(("127.0.0.1", seen_clients[0]))
        reply = await reader.read(100)
        writer.close()
        await gate._close()
        server.close()
        return marked, reply

    marked, reply = asyncio.run(scenario())
    assert marked is True
    assert reply == b"echo:hello"
