"""Global HTTP/HTTPS proxy helpers.

Reads the `secrets.proxy` config, builds the proxies dict expected by the requests library, and
injects proxies into an existing `requests.Session`. Currently only covers the booru download path;
transport-level proxy injection for `huggingface_hub` needs a separate monkeypatch, which is out of
scope for this module (the original `setup_global_httpx_client` from PR #162 used
`huggingface_hub.set_client_factory` -- that symbol doesn't exist in huggingface_hub, so it raised
ImportError on import; this hotfix removes that dead function too).
"""
from typing import Dict, Optional
import logging

from .. import secrets

logger = logging.getLogger(__name__)


def get_proxy_dict() -> Optional[Dict[str, str]]:
    """Read proxy settings from secrets, returning the dict format the requests library expects; returns None if disabled."""
    try:
        cfg = secrets.load().proxy
        if not cfg.enabled:
            return None

        proxies: Dict[str, str] = {}
        http_p = (cfg.http_proxy or "").strip()
        https_p = (cfg.https_proxy or "").strip()
        # If either field is blank, fall back to the other. Local proxies (clash / v2ray / socks5 etc.) usually
        # forward both http and https traffic through a single port, so users often only fill in one field. This also
        # honors the Settings page tip3 promise: "if HTTPS is blank, fall back to the HTTP proxy".
        #
        # The old logic only mirrored to https when `http_proxy` explicitly started with socks5://; a plain HTTP
        # proxy with only http_proxy set left proxies["https"] unset -> all booru (all-https) requests went
        # direct -> timeout. Now we fall back uniformly for all schemes; the socks5 special case is absorbed by
        # this logic, and when both fields are explicitly set, each is respected without overriding the other.
        if http_p and not https_p:
            https_p = http_p
        elif https_p and not http_p:
            http_p = https_p
        if http_p:
            proxies["http"] = http_p
        if https_p:
            proxies["https"] = https_p

        logger.info(f"Using proxies: {proxies}")
        return proxies if proxies else None
    except Exception as e:
        logger.error(f"Failed to get proxy: {e}")
        return None


def get_no_proxy_list() -> list[str]:
    """Split the `no_proxy` field into a host list."""
    try:
        proxy_cfg = secrets.load().proxy
        if not proxy_cfg.no_proxy:
            return []
        return [host.strip() for host in proxy_cfg.no_proxy.split(",")]
    except Exception:
        return []


def patch_requests_session(session):
    """Inject proxies into an existing requests.Session; returns it unchanged if proxying is disabled."""
    proxies = get_proxy_dict()
    if proxies:
        session.proxies.update(proxies)
        logger.info(f"Patched session proxies: {session.proxies}")
    else:
        logger.info("No proxy configured, session unchanged")
    return session
