"""Public URL for reaching the studio from a phone.

Routes:
    GET  /api/tunnel             state + settings (provider, autostart, ngrok fields)
    PUT  /api/tunnel/config      change provider / autostart / ngrok authtoken & domain
    POST /api/tunnel/start       open the tunnel, returns the keyed URL
    POST /api/tunnel/stop        close it
    POST /api/tunnel/rotate-key  replace the access key (old links stop working)
    POST /api/tunnel/install     download cloudflared / ngrok for this machine

The port is the one this server is actually listening on (the ASGI ``server``
tuple), not the Host header: behind the Vite dev proxy the Host header names the
dev server's port, which is not the one to forward.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ...domain.errors import ValidationError
from ...services import tunnel as tunnel_svc

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765


class TunnelConfigUpdate(BaseModel):
    provider: Optional[str] = None
    autostart: Optional[bool] = None
    ngrok_domain: Optional[str] = None
    ngrok_authtoken: Optional[str] = None


class TunnelInstallRequest(BaseModel):
    provider: str = "cloudflare"


def listening_port(request: Request) -> int:
    server = request.scope.get("server")
    if server and len(server) >= 2 and server[1]:
        return int(server[1])
    return int(request.url.port or DEFAULT_PORT)


def _fail(exc: Exception):
    raise ValidationError(str(exc), details={"reason": str(exc)}) from exc


@router.get("/api/tunnel")
def tunnel_state() -> dict[str, Any]:
    return tunnel_svc.state()


@router.put("/api/tunnel/config")
def tunnel_config(body: TunnelConfigUpdate) -> dict[str, Any]:
    fields = body.model_dump(exclude_none=True)
    # The settings page sends the mask back for an untouched token.
    if fields.get("ngrok_authtoken") == "***":
        fields.pop("ngrok_authtoken")
    try:
        return tunnel_svc.configure(**fields)
    except tunnel_svc.TunnelError as exc:
        _fail(exc)
        raise


@router.post("/api/tunnel/start")
def tunnel_start(request: Request) -> dict[str, Any]:
    try:
        return tunnel_svc.start(listening_port(request))
    except tunnel_svc.TunnelError as exc:
        _fail(exc)
        raise


@router.post("/api/tunnel/stop")
def tunnel_stop() -> dict[str, Any]:
    return tunnel_svc.stop()


@router.post("/api/tunnel/rotate-key")
def tunnel_rotate_key() -> dict[str, Any]:
    return tunnel_svc.rotate_key()


@router.post("/api/tunnel/install")
def tunnel_install(body: Optional[TunnelInstallRequest] = None) -> dict[str, Any]:
    try:
        tunnel_svc.install((body.provider if body else "cloudflare"))
    except tunnel_svc.TunnelError as exc:
        _fail(exc)
    return tunnel_svc.state()
