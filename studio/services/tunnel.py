"""Public URL for the studio, so the training UI is reachable from a phone.

The studio normally listens on ``127.0.0.1``, which is exactly right on a
desktop and useless when you are out of the house. This module asks an outbound
tunnel service to forward a public HTTPS address to the machine. Nothing is
opened on the router and nothing listens on the machine's network interfaces.
Three providers:

* ``cloudflare`` — Cloudflare *quick tunnel*. No account, no setup, but the
  address (``https://<random>.trycloudflare.com``) is new on every start.
* ``tailscale`` — Tailscale Funnel. A **permanent** address,
  ``https://<computer>.<tailnet>.ts.net``, free, no domain to buy. Needs the
  Tailscale app installed and signed in, and Funnel allowed once for the
  machine (the studio shows the link Tailscale asks you to open).
* ``ngrok`` — also **permanent** with the free static domain ngrok gives every
  account (``https://<name>.ngrok-free.app``). Needs an authtoken.

**The URL is public.** Anyone who has it can reach the machine. The studio has
no login, and its API can read the filesystem and start jobs — so every tunnel
carries an access key:

* The key lives in ``secrets.json`` and survives restarts, so a permanent
  address really is a permanent *link* (``https://…/?k=<key>``). It changes only
  when the user rotates it.
* :class:`TunnelAuthMiddleware` lets local requests through untouched and
  demands the key from anything arriving *through the tunnel*. The first request
  carrying a valid key gets a cookie, so the rest of the session works normally.

**How "through the tunnel" is recognised.** Not by headers: each provider's
proxy sets different ones, and guessing wrong would silently leave a public
address wide open. Instead the tunnel never points at the studio's own port. It
points at the *gate* — a tiny TCP relay on a private loopback port, run by this
process — which forwards to the studio. The relay records the local port of
every upstream connection it opens, and the middleware treats a request whose
client address is one of those ports as external. A browser on the machine
itself never goes through the relay, so switching the tunnel on cannot lock
anyone out of their own studio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from ..infrastructure.paths import STUDIO_DATA

logger = logging.getLogger(__name__)

BIN_DIR = STUDIO_DATA / "bin"

PROVIDERS = ("cloudflare", "tailscale", "ngrok")

#: cloudflared prints the quick-tunnel address to stderr inside a banner box.
_URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")

#: How long to wait for a provider to report its address before giving up.
START_TIMEOUT_S = 40.0

#: Query parameter and cookie carrying the access key.
KEY_PARAM = "k"
KEY_COOKIE = "anima_tunnel_key"
#: The key is persistent, so the cookie can outlive a browser session.
KEY_COOKIE_MAX_AGE = 30 * 86400

_DOWNLOAD_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"

#: (system, machine) → release asset name. Cloudflare publishes static binaries
#: for exactly these; anything else has to be installed by hand.
_ASSETS = {
    ("linux", "x86_64"): "cloudflared-linux-amd64",
    ("linux", "amd64"): "cloudflared-linux-amd64",
    ("linux", "aarch64"): "cloudflared-linux-arm64",
    ("linux", "arm64"): "cloudflared-linux-arm64",
    ("darwin", "x86_64"): "cloudflared-darwin-amd64.tgz",
    ("darwin", "arm64"): "cloudflared-darwin-arm64.tgz",
    ("windows", "amd64"): "cloudflared-windows-amd64.exe",
    ("windows", "x86_64"): "cloudflared-windows-amd64.exe",
}

_NGROK_BASE = "https://bin.equinox.io/c/bNyj1mQVY4c"
_NGROK_ASSETS = {
    ("linux", "x86_64"): "ngrok-v3-stable-linux-amd64.tgz",
    ("linux", "amd64"): "ngrok-v3-stable-linux-amd64.tgz",
    ("linux", "aarch64"): "ngrok-v3-stable-linux-arm64.tgz",
    ("linux", "arm64"): "ngrok-v3-stable-linux-arm64.tgz",
    ("darwin", "x86_64"): "ngrok-v3-stable-darwin-amd64.zip",
    ("darwin", "arm64"): "ngrok-v3-stable-darwin-arm64.zip",
    ("windows", "amd64"): "ngrok-v3-stable-windows-amd64.zip",
    ("windows", "x86_64"): "ngrok-v3-stable-windows-amd64.zip",
}

TAILSCALE_DOWNLOAD_URL = "https://tailscale.com/download"
NGROK_DASHBOARD_URL = "https://dashboard.ngrok.com/get-started/your-authtoken"
NGROK_DOMAINS_URL = "https://dashboard.ngrok.com/domains"


class TunnelError(Exception):
    """Something the user can act on: binary missing, start failed, timeout."""


# ── persisted settings ───────────────────────────────────────────────────────

def load_settings() -> dict[str, Any]:
    from ..infrastructure import secrets as _secrets

    return _secrets.load().remote_access.model_dump()


def save_settings(**fields: Any) -> dict[str, Any]:
    from ..infrastructure import secrets as _secrets

    return _secrets.update({"remote_access": fields}).remote_access.model_dump()


def _new_key() -> str:
    return secrets.token_urlsafe(24)[:32]


def _persistent_key() -> str:
    key = str(load_settings().get("access_key") or "")
    if len(key) < 16:
        key = _new_key()
        save_settings(access_key=key)
    return key


# ── the gate: a loopback relay that marks tunnel traffic ─────────────────────

class _Gate:
    """TCP relay ``127.0.0.1:<gate port> → 127.0.0.1:<studio port>``.

    Runs on the studio's own event loop (bound at startup). ``client_ports`` is
    the set of local ports of the relay's upstream connections — exactly the
    client ports uvicorn reports for requests that came through the tunnel.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[asyncio.base_events.Server] = None
        self.port: Optional[int] = None
        self.target: Optional[int] = None
        self.client_ports: set[int] = set()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def available(self) -> bool:
        return self._loop is not None and not self._loop.is_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        assert self.target is not None
        try:
            up_reader, up_writer = await asyncio.open_connection("127.0.0.1", self.target)
        except OSError:
            writer.close()
            return
        local_port = int(up_writer.get_extra_info("sockname")[1])
        self.client_ports.add(local_port)

        async def pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while True:
                    data = await src.read(65536)
                    if not data:
                        break
                    dst.write(data)
                    await dst.drain()
                if dst.can_write_eof():
                    dst.write_eof()
            except (ConnectionError, OSError):
                pass

        try:
            await asyncio.gather(pump(reader, up_writer), pump(up_reader, writer))
        finally:
            self.client_ports.discard(local_port)
            for w in (up_writer, writer):
                try:
                    w.close()
                except Exception:  # noqa: BLE001 — already gone
                    pass

    async def _open(self, target: int) -> int:
        if self._server is not None and self.target == target:
            return int(self.port)  # type: ignore[arg-type]
        await self._close()
        self.target = target
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = int(self._server.sockets[0].getsockname()[1])
        logger.info("tunnel gate: 127.0.0.1:%s → studio port %s", self.port, target)
        return self.port

    async def _close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        self.port = None

    def open(self, target: int) -> int:
        if not self.available:
            raise TunnelError("The studio event loop is not ready yet; try again in a moment.")
        fut = asyncio.run_coroutine_threadsafe(self._open(int(target)), self._loop)  # type: ignore[arg-type]
        return fut.result(timeout=10)

    def close(self) -> None:
        if self._server is None or not self.available:
            self._server = None
            self.port = None
            return
        try:
            asyncio.run_coroutine_threadsafe(self._close(), self._loop).result(timeout=5)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 — shutdown path
            self._server = None
            self.port = None

    def is_external_client(self, client: Any) -> bool:
        if not client or len(client) < 2:
            return False
        try:
            return int(client[1]) in self.client_ports
        except (TypeError, ValueError):
            return False


_gate = _Gate()


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Called once from the app lifespan with the running loop."""
    _gate.bind_loop(loop)


# ── helpers for provider processes ───────────────────────────────────────────

def _run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=timeout, **_no_window(),
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        return -1, out or ""
    except OSError as exc:
        return -1, str(exc)
    return proc.returncode, proc.stdout or ""


def _no_window() -> dict[str, Any]:
    """Keep console windows from flashing up on Windows."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _last_lines(lines: list[str], n: int = 3) -> str:
    tail = [ln.strip() for ln in lines if ln.strip()][-n:]
    return " | ".join(tail) if tail else "no output"


# ── the running tunnel ───────────────────────────────────────────────────────

class _Tunnel:
    """Owns at most one tunnel.

    A second tunnel would be pointless (same local port) and would leak the
    first, so ``start`` on an already-running tunnel returns the existing one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen[str]] = None
        self._provider: Optional[str] = None
        self._tailscale_on = False
        self._url: Optional[str] = None
        self._key: Optional[str] = None
        self._port: Optional[int] = None
        self._target: Optional[int] = None
        self._started_at: Optional[float] = None
        self._error: Optional[str] = None
        self._log: list[str] = []

    # ── state ────────────────────────────────────────────────────────────

    def _alive(self) -> bool:
        if self._provider == "tailscale":
            return self._tailscale_on
        return self._proc is not None and self._proc.poll() is None

    def state(self) -> dict[str, Any]:
        with self._lock:
            if self._provider is not None and not self._alive():
                # The process died on its own (network drop, provider hiccup).
                # Report it as stopped rather than handing out a dead URL.
                self._gate_close()
                self._clear()
            return self.state_unlocked()

    def _public_url(self) -> Optional[str]:
        if not self._url:
            return None
        return f"{self._url}?{KEY_PARAM}={self._key}" if self._key else self._url

    def key(self) -> Optional[str]:
        return self._key if self._alive() else None

    def is_running(self) -> bool:
        return self._alive()

    def _clear(self) -> None:
        self._proc = None
        self._provider = None
        self._tailscale_on = False
        self._url = None
        self._key = None
        self._port = None
        self._target = None
        self._started_at = None

    def _gate_close(self) -> None:
        if self._target is not None and self._target == _gate.port:
            _gate.close()

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self, port: int, provider: Optional[str] = None) -> dict[str, Any]:
        settings = load_settings()
        provider = provider or str(settings.get("provider") or "cloudflare")
        if provider not in PROVIDERS:
            raise TunnelError(f"Unknown remote access provider: {provider!r}")
        with self._lock:
            if self._alive():
                return self.state_unlocked()
            self._error = None
            self._log = []
            if provider == "cloudflare" and find_binary() is None:
                raise TunnelError(
                    "cloudflared is not installed. Use the install button, or "
                    "install it yourself and restart the studio."
                )
            # cloudflared still works without the gate (its requests carry
            # CF-Ray); the permanent providers do not, so they require it.
            if _gate.available:
                target = _gate.open(int(port))
            elif provider == "cloudflare":
                target = int(port)
            else:
                raise TunnelError("The studio event loop is not ready yet; try again in a moment.")
            key = _persistent_key()

            try:
                if provider == "cloudflare":
                    url = self._start_cloudflare(target)
                elif provider == "tailscale":
                    url = self._start_tailscale(target)
                else:
                    url = self._start_ngrok(target, settings)
            except TunnelError as exc:
                self._error = str(exc)
                if target == _gate.port:
                    _gate.close()
                self._clear()
                raise

            self._provider = provider
            self._url = url.rstrip("/")
            self._key = key
            self._port = int(port)
            self._target = target
            self._started_at = time.time()
            logger.info("tunnel: %s up at %s", provider, self._url)
            return self.state_unlocked()

    # cloudflare ─────────────────────────────────────────────────────────

    def _start_cloudflare(self, target: int) -> str:
        binary = find_binary()
        assert binary is not None
        cmd = [
            str(binary), "tunnel",
            "--no-autoupdate",
            "--url", f"http://127.0.0.1:{int(target)}",
        ]
        proc = self._spawn(cmd, os.environ | {"TZ": os.environ.get("TZ", "UTC")})

        def match(line: str) -> Optional[str]:
            m = _URL_RE.search(line)
            return m.group(0) if m else None

        url = self._await(proc, match)
        if url is None:
            self._stop_process(proc)
            detail = self._log[-1] if self._log else "no address was returned"
            raise TunnelError(f"cloudflared did not produce an address ({detail}).")
        self._proc = proc
        return url

    # tailscale ──────────────────────────────────────────────────────────

    def _start_tailscale(self, target: int) -> str:
        binary = find_tailscale()
        if binary is None:
            raise TunnelError(
                f"Tailscale is not installed. Install it from {TAILSCALE_DOWNLOAD_URL}, "
                "sign in, then try again."
            )
        code, out = _run([str(binary), "status", "--json"], timeout=15)
        try:
            status = json.loads(out[out.find("{"):]) if "{" in out else {}
        except ValueError:
            status = {}
        backend = str(status.get("BackendState") or "")
        if backend != "Running":
            raise TunnelError(
                "Tailscale is not signed in or not connected"
                + (f" (state: {backend})" if backend else "")
                + ". Open the Tailscale app, sign in, then try again."
            )
        dns = str((status.get("Self") or {}).get("DNSName") or "").rstrip(".")
        if not dns:
            raise TunnelError(
                "Tailscale did not report this computer's name. Enable MagicDNS "
                "in the Tailscale admin console, then try again."
            )

        base = [str(binary), "funnel", "--bg"]
        tail = ["--https=443", f"http://127.0.0.1:{int(target)}"]
        lines, exit_code = self._run_until_exit(base + ["--yes"] + tail)
        if exit_code != 0 and any("flag provided but not defined" in ln for ln in lines):
            lines, exit_code = self._run_until_exit(base + tail)  # older CLI

        text = "\n".join(lines)
        link = re.search(r"https://login\.tailscale\.com/\S+", text)
        if exit_code is None or (exit_code != 0 and link):
            if link:
                raise TunnelError(
                    "Tailscale Funnel is not allowed for this computer yet. Open "
                    f"{link.group(0)} , allow it, then press start again."
                )
            raise TunnelError(f"tailscale funnel did not finish ({_last_lines(lines)}).")
        if exit_code != 0:
            raise TunnelError(f"tailscale funnel failed ({_last_lines(lines)}).")
        self._tailscale_on = True
        return f"https://{dns}"

    def _run_until_exit(self, cmd: list[str]) -> tuple[list[str], Optional[int]]:
        """Run a CLI that normally exits quickly; ``None`` exit code = still
        waiting (killed after the timeout, e.g. Tailscale asking for consent)."""
        proc = self._spawn(cmd, dict(os.environ))
        self._await(proc, lambda _line: None, until_exit=True)
        # Output EOF can arrive a moment before the OS records the exit code
        # (seen on Windows: poll() still None ~30 ms after the CLI finished).
        try:
            code: Optional[int] = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = None
            self._stop_process(proc)
        return list(self._log), code

    def _stop_tailscale(self, target: Optional[int]) -> None:
        binary = find_tailscale()
        if binary is None:
            return
        if target is not None:
            code, _ = _run([str(binary), "funnel", "--https=443", f"http://127.0.0.1:{int(target)}", "off"])
            if code == 0:
                return
        _run([str(binary), "funnel", "--https=443", "off"])

    # ngrok ──────────────────────────────────────────────────────────────

    def _start_ngrok(self, target: int, settings: dict[str, Any]) -> str:
        binary = find_ngrok()
        if binary is None:
            raise TunnelError("ngrok is not installed. Use the install button, or install it yourself.")
        token = str(settings.get("ngrok_authtoken") or "").strip()
        if not token:
            raise TunnelError(
                f"ngrok needs your authtoken. Copy it from {NGROK_DASHBOARD_URL} "
                "and paste it into the field below."
            )
        domain = str(settings.get("ngrok_domain") or "").strip()
        env = dict(os.environ) | {"NGROK_AUTHTOKEN": token}

        def cmd(flag: str) -> list[str]:
            c = [str(binary), "http", f"127.0.0.1:{int(target)}",
                 "--log", "stdout", "--log-format", "json"]
            if domain:
                c += [flag, f"https://{domain}" if flag == "--url" else domain]
            return c

        def match(line: str) -> Optional[str]:
            try:
                obj = json.loads(line)
            except ValueError:
                return None
            if isinstance(obj, dict) and obj.get("url") and "tunnel" in str(obj.get("msg", "")):
                return str(obj["url"])
            return None

        proc = self._spawn(cmd("--url"), env)
        url = self._await(proc, match)
        if url is None and domain and any("unknown flag" in ln for ln in self._log):
            self._stop_process(proc)
            proc = self._spawn(cmd("--domain"), env)  # agents older than --url
            url = self._await(proc, match)
        if url is None:
            self._stop_process(proc)
            raise TunnelError(f"ngrok did not produce an address ({_ngrok_error(self._log)}).")
        self._proc = proc
        return url

    # shared ─────────────────────────────────────────────────────────────

    def _spawn(self, cmd: list[str], env: dict[str, str]) -> subprocess.Popen[str]:
        logger.info("tunnel: starting %s", " ".join(cmd[:4]))
        self._log = []
        try:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                **_no_window(),
            )
        except OSError as exc:
            raise TunnelError(f"Could not launch {Path(cmd[0]).name}: {exc}") from exc

    def _await(
        self,
        proc: subprocess.Popen[str],
        match: Callable[[str], Optional[str]],
        *,
        until_exit: bool = False,
    ) -> Optional[str]:
        """Read the process output until ``match`` finds the address (or the
        process exits, or time runs out).

        The reader thread keeps draining afterwards: tunnel agents write a
        heartbeat, and a full pipe buffer would wedge the process.
        """
        found: list[str] = []
        done = threading.Event()

        def _pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self._log.append(line)
                    del self._log[:-200]
                if not found and not until_exit:
                    hit = match(line)
                    if hit:
                        found.append(hit)
                        done.set()
            done.set()

        threading.Thread(target=_pump, daemon=True, name="tunnel-log").start()
        done.wait(START_TIMEOUT_S)
        return found[0] if found else None

    def state_unlocked(self) -> dict[str, Any]:
        """``state()`` without re-taking the lock (callers already hold it)."""
        running = self._alive()
        try:
            settings = load_settings()
        except Exception:  # noqa: BLE001 — a broken secrets file must not break the page
            settings = {}
        provider = str(settings.get("provider") or "cloudflare")
        return {
            "running": running,
            "url": self._public_url() if running else None,
            "port": self._port if running else None,
            "started_at": self._started_at if running else None,
            "running_provider": self._provider if running else None,
            "error": self._error,
            # the configured provider (what "start" will use)
            "provider": provider,
            "autostart": bool(settings.get("autostart")),
            "ngrok_domain": str(settings.get("ngrok_domain") or ""),
            "has_ngrok_token": bool(settings.get("ngrok_authtoken")),
            "permanent": provider in ("tailscale", "ngrok") and (provider != "ngrok" or bool(settings.get("ngrok_domain"))),
            # cloudflared (kept at the top level for older clients)
            "binary": str(find_binary() or ""),
            "installed": find_binary() is not None,
            "can_install": _asset_name() is not None,
            "providers": {
                "cloudflare": {"installed": find_binary() is not None, "can_install": _asset_name() is not None},
                "tailscale": {"installed": find_tailscale() is not None, "can_install": False,
                              "download_url": TAILSCALE_DOWNLOAD_URL},
                "ngrok": {"installed": find_ngrok() is not None, "can_install": _ngrok_asset_name() is not None,
                          "token_url": NGROK_DASHBOARD_URL, "domains_url": NGROK_DOMAINS_URL},
            },
            "log": list(self._log[-20:]),
        }

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._provider == "tailscale" and self._tailscale_on:
                self._stop_tailscale(self._target)
            if self._proc is not None:
                self._stop_process(self._proc)
            self._gate_close()
            self._clear()
            return self.state_unlocked()

    def rotate_key(self) -> dict[str, Any]:
        with self._lock:
            key = _new_key()
            save_settings(access_key=key)
            if self._alive():
                self._key = key
            return self.state_unlocked()

    @staticmethod
    def _stop_process(proc: subprocess.Popen[str]) -> None:
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                logger.warning("tunnel: agent process did not exit")


def _ngrok_error(lines: list[str]) -> str:
    for line in reversed(lines):
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and (obj.get("err") or str(obj.get("lvl")) in ("eror", "crit")):
            return str(obj.get("err") or obj.get("msg"))
    return _last_lines(lines)


_tunnel = _Tunnel()


def state() -> dict[str, Any]:
    return _tunnel.state()


def start(port: int, provider: Optional[str] = None) -> dict[str, Any]:
    return _tunnel.start(port, provider)


def stop() -> dict[str, Any]:
    return _tunnel.stop()


def rotate_key() -> dict[str, Any]:
    return _tunnel.rotate_key()


def configure(**fields: Any) -> dict[str, Any]:
    """Persist remote-access settings (provider / autostart / ngrok fields)."""
    allowed = {"provider", "autostart", "ngrok_domain", "ngrok_authtoken"}
    clean = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if "provider" in clean and clean["provider"] not in PROVIDERS:
        raise TunnelError(f"Unknown remote access provider: {clean['provider']!r}")
    if clean:
        save_settings(**clean)
    return _tunnel.state()


def current_key() -> Optional[str]:
    return _tunnel.key()


def is_running() -> bool:
    return _tunnel.is_running()


def autostart(port: int) -> None:
    """Open the link at studio start when the user asked for it. Runs in a
    background thread: a provider that is slow to answer must not delay the UI,
    and a failure is only logged (the settings page shows it)."""
    try:
        if not load_settings().get("autostart"):
            return
    except Exception:  # noqa: BLE001
        return

    def _run_start() -> None:
        try:
            s = start(port)
            logger.info("tunnel: autostart ok → %s", s.get("url", "").split("?")[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning("tunnel: autostart failed: %s", exc)

    threading.Thread(target=_run_start, name="tunnel-autostart", daemon=True).start()


def shutdown() -> None:
    """Close the tunnel when the studio exits — a stray agent would keep a
    public URL alive pointing at a port nothing is serving."""
    try:
        _tunnel.stop()
    except Exception:  # noqa: BLE001 — never block shutdown
        logger.debug("tunnel: shutdown failed", exc_info=True)


# ── binaries ─────────────────────────────────────────────────────────────────

def _exe(name: str) -> str:
    return f"{name}.exe" if platform.system().lower() == "windows" else name


def _exe_name() -> str:
    return _exe("cloudflared")


def find_binary() -> Optional[Path]:
    """cloudflared: our own copy first, then whatever is on PATH."""
    local = BIN_DIR / _exe_name()
    if local.is_file() and os.access(local, os.X_OK):
        return local
    found = shutil.which("cloudflared")
    return Path(found) if found else None


def find_ngrok() -> Optional[Path]:
    local = BIN_DIR / _exe("ngrok")
    if local.is_file() and os.access(local, os.X_OK):
        return local
    found = shutil.which("ngrok")
    return Path(found) if found else None


def find_tailscale() -> Optional[Path]:
    found = shutil.which("tailscale")
    if found:
        return Path(found)
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tailscale" / "tailscale.exe",
        Path("/Applications/Tailscale.app/Contents/MacOS/Tailscale"),
        Path("/usr/bin/tailscale"),
        Path("/usr/local/bin/tailscale"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _asset_name() -> Optional[str]:
    return _ASSETS.get((platform.system().lower(), platform.machine().lower()))


def _ngrok_asset_name() -> Optional[str]:
    return _NGROK_ASSETS.get((platform.system().lower(), platform.machine().lower()))


def _download(url: str, tmp: Path) -> None:
    logger.info("tunnel: downloading %s", url)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise TunnelError(f"Download failed: {exc}") from exc


def _make_executable(target: Path) -> None:
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install(provider: str = "cloudflare") -> dict[str, Any]:
    """Download the static agent binary for this machine.

    One file, no installer, no admin rights — it lands next to the studio's own
    data so uninstalling is deleting it. Tailscale is a system service with its
    own installer and cannot be installed this way.
    """
    if provider == "ngrok":
        return _install_ngrok()
    if provider != "cloudflare":
        raise TunnelError(f"{provider} has to be installed from its own website.")
    asset = _asset_name()
    if asset is None:
        raise TunnelError(
            f"No prebuilt cloudflared for {platform.system()} {platform.machine()}. "
            f"Install it manually and it will be picked up from PATH."
        )
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / _exe_name()
    tmp = target.with_suffix(target.suffix + ".part")
    _download(f"{_DOWNLOAD_BASE}/{asset}", tmp)
    try:
        if asset.endswith(".tgz"):
            _extract_one(tmp, target, "cloudflared")
            tmp.unlink(missing_ok=True)
        else:
            os.replace(tmp, target)
        _make_executable(target)
    except (OSError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise TunnelError(f"Could not unpack cloudflared: {exc}") from exc
    return {"installed": True, "binary": str(target)}


def _install_ngrok() -> dict[str, Any]:
    asset = _ngrok_asset_name()
    if asset is None:
        raise TunnelError(
            f"No prebuilt ngrok for {platform.system()} {platform.machine()}. "
            "Install it manually and it will be picked up from PATH."
        )
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / _exe("ngrok")
    tmp = BIN_DIR / (asset + ".part")
    _download(f"{_NGROK_BASE}/{asset}", tmp)
    try:
        _extract_one(tmp, target, _exe("ngrok"))
        _make_executable(target)
    except (OSError, ValueError) as exc:
        raise TunnelError(f"Could not unpack ngrok: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return {"installed": True, "binary": str(target)}


def _extract_one(archive: Path, target: Path, member_name: str) -> None:
    """Pull one named file out of a .tgz or .zip archive."""
    import tarfile
    import zipfile

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            info = next((i for i in zf.infolist() if Path(i.filename).name == member_name), None)
            if info is None:
                raise ValueError(f"the archive contains no {member_name}")
            with zf.open(info) as src, target.open("wb") as fh:
                shutil.copyfileobj(src, fh)
        return
    with tarfile.open(archive, "r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.isfile() and Path(m.name).name == member_name),
            None,
        )
        if member is None:
            raise ValueError(f"the archive contains no {member_name}")
        src = tar.extractfile(member)
        if src is None:
            raise ValueError("the archive entry could not be read")
        with src, target.open("wb") as fh:
            shutil.copyfileobj(src, fh)


def _extract_tgz(archive: Path, target: Path) -> None:
    """macOS ships cloudflared inside a tarball; pull out the one file we want."""
    _extract_one(archive, target, "cloudflared")


# ── access gate ──────────────────────────────────────────────────────────────

def _header(scope: dict, name: bytes) -> Optional[bytes]:
    for k, v in scope.get("headers", []):
        if k.lower() == name:
            return v
    return None


def _cookie_value(raw: Optional[bytes], name: str) -> Optional[str]:
    if not raw:
        return None
    for part in raw.decode("latin-1").split(";"):
        k, _, v = part.strip().partition("=")
        if k == name:
            return v
    return None


def is_external(scope: dict) -> bool:
    """Did this request come in through the tunnel?

    Primary signal: the client port belongs to the gate relay. CF-Ray is kept as
    a second signal for a cloudflared started before the gate existed.
    """
    return _gate.is_external_client(scope.get("client")) or _header(scope, b"cf-ray") is not None


class TunnelAuthMiddleware:
    """Require the tunnel key on requests that arrived through the tunnel.

    Pure ASGI rather than ``BaseHTTPMiddleware`` for the same reason as
    :class:`TraceIdMiddleware`: it has to see every request, including the ones
    that never reach a route.

    Local requests are untouched — they never pass the gate relay, so switching
    the tunnel on cannot lock anyone out of their own studio.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        key = current_key()
        if not is_external(scope):
            return await self.app(scope, receive, send)
        if key is None:
            # Traffic through the gate with no tunnel up (a stale connection
            # after stop): nothing to authenticate against, refuse.
            return await _deny(send)

        query = scope.get("query_string", b"").decode("latin-1")
        supplied = _query_key(query)
        if supplied is not None and secrets.compare_digest(supplied, key):
            # Hand out a cookie so the next request (and every asset on the
            # page) works without the key stuck in the URL.
            return await self.app(scope, receive, _with_cookie(send, key))

        cookie = _cookie_value(_header(scope, b"cookie"), KEY_COOKIE)
        if cookie is not None and secrets.compare_digest(cookie, key):
            return await self.app(scope, receive, send)

        await _deny(send)


def _query_key(query: str) -> Optional[str]:
    from urllib.parse import parse_qs

    values = parse_qs(query).get(KEY_PARAM)
    return values[0] if values else None


def _with_cookie(send, key: str):
    """Wrap ``send`` so the response start carries the session cookie."""
    async def _send(message):
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            headers.append((
                b"set-cookie",
                f"{KEY_COOKIE}={key}; Path=/; SameSite=Lax; Secure; HttpOnly; "
                f"Max-Age={KEY_COOKIE_MAX_AGE}".encode("latin-1"),
            ))
            message = {**message, "headers": headers}
        await send(message)
    return _send


_DENY_BODY = (
    b"<!doctype html><meta charset=utf-8>"
    b"<meta name=viewport content='width=device-width,initial-scale=1'>"
    b"<title>Access key required</title>"
    b"<style>body{font:16px/1.5 system-ui;margin:12vh auto;max-width:34rem;padding:0 1.5rem;"
    b"color:#222}h1{font-size:1.25rem}</style>"
    b"<h1>Access key required</h1>"
    b"<p>This studio is reachable through a public address, so it only "
    b"answers requests that carry the access key.</p>"
    b"<p>Open the full link shown in the studio's settings \xe2\x80\x94 the one ending in "
    b"<code>?k=\xe2\x80\xa6</code>.</p>"
)


async def _deny(send) -> None:
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [
            (b"content-type", b"text/html; charset=utf-8"),
            (b"content-length", str(len(_DENY_BODY)).encode("ascii")),
            (b"cache-control", b"no-store"),
        ],
    })
    await send({"type": "http.response.body", "body": _DENY_BODY})
