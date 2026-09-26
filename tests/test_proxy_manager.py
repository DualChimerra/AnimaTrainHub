from __future__ import annotations

import socket
import threading

import requests
import urllib3

from studio.infrastructure import secrets as secrets_mod
from studio.services import proxy_manager


def _use_proxy(monkeypatch, **kw):
    s = secrets_mod.Secrets(proxy=secrets_mod.ProxyConfig(**kw))
    monkeypatch.setattr(proxy_manager.secrets, "load", lambda: s)


def test_disabled_returns_none(monkeypatch):
    _use_proxy(monkeypatch, enabled=False, http_proxy="http://127.0.0.1:2333")
    assert proxy_manager.get_proxy_dict() is None


def test_empty_when_enabled_but_no_addresses(monkeypatch):
    _use_proxy(monkeypatch, enabled=True)
    assert proxy_manager.get_proxy_dict() is None


def test_http_only_falls_back_to_https(monkeypatch):
    _use_proxy(monkeypatch, enabled=True, http_proxy="http://127.0.0.1:2333")
    assert proxy_manager.get_proxy_dict() == {
        "http": "http://127.0.0.1:2333",
        "https": "http://127.0.0.1:2333",
    }


def test_https_only_falls_back_to_http(monkeypatch):
    _use_proxy(monkeypatch, enabled=True, https_proxy="http://127.0.0.1:2333")
    assert proxy_manager.get_proxy_dict() == {
        "http": "http://127.0.0.1:2333",
        "https": "http://127.0.0.1:2333",
    }


def test_both_explicit_are_respected(monkeypatch):
    _use_proxy(
        monkeypatch,
        enabled=True,
        http_proxy="http://127.0.0.1:2333",
        https_proxy="http://127.0.0.1:9999",
    )
    assert proxy_manager.get_proxy_dict() == {
        "http": "http://127.0.0.1:2333",
        "https": "http://127.0.0.1:9999",
    }


def test_socks5_single_field_covers_both(monkeypatch):
    _use_proxy(monkeypatch, enabled=True, http_proxy="socks5://127.0.0.1:1080")
    assert proxy_manager.get_proxy_dict() == {
        "http": "socks5://127.0.0.1:1080",
        "https": "socks5://127.0.0.1:1080",
    }


def test_whitespace_only_treated_as_empty(monkeypatch):
    _use_proxy(
        monkeypatch,
        enabled=True,
        http_proxy="  http://127.0.0.1:2333  ",
        https_proxy="   ",
    )
    assert proxy_manager.get_proxy_dict() == {
        "http": "http://127.0.0.1:2333",
        "https": "http://127.0.0.1:2333",
    }


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------


class _FakeProxy:

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.seen: list[str] = []
        self._stop = False

    def start(self) -> None:
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5)
            data = b""
            while b"\r\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            first = data.split(b"\r\n", 1)[0].decode("latin1")
            self.seen.append(first)
            if first.startswith("CONNECT"):
                conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            else:
                body = b"ok-via-proxy"
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%b"
                    % (len(body), body)
                )
        except OSError:
            pass
        finally:
            conn.close()

    def stop(self) -> None:
        self._stop = True
        try:
            self.sock.close()
        except OSError:
            pass


def test_e2e_http_only_routes_both_schemes_through_proxy(monkeypatch):
    urllib3.disable_warnings()
    proxy = _FakeProxy()
    proxy.start()
    try:
        _use_proxy(
            monkeypatch, enabled=True, http_proxy=f"http://127.0.0.1:{proxy.port}"
        )
        proxies = proxy_manager.get_proxy_dict()
        assert proxies == {
            "http": f"http://127.0.0.1:{proxy.port}",
            "https": f"http://127.0.0.1:{proxy.port}",
        }

        r = requests.get(
            "http://booru.invalid/index.php", proxies=proxies, timeout=5
        )
        assert r.text == "ok-via-proxy"
        assert any(
            s.startswith("GET http://booru.invalid/index.php") for s in proxy.seen
        ), f"the proxy did not see an absolute-form GET; seen={proxy.seen}"

        try:
            requests.get(
                "https://gelbooru.com/index.php",
                proxies=proxies,
                timeout=5,
                verify=False,
            )
        except requests.exceptions.RequestException:
            pass
        assert any(
            s.startswith("CONNECT gelbooru.com:443") for s in proxy.seen
        ), f"the proxy did not see a CONNECT (did https go direct?); seen={proxy.seen}"
    finally:
        proxy.stop()
