"""Runtime package install / LLM tagger admin endpoints (PR-6 commit 3, extracted from server.py).

10 routes, split into 5 subdomains by underlying component, sharing one router:

  wd14 (onnxruntime):     GET /api/wd14/runtime           currently installed package + available EPs
                           POST /api/wd14/install          switch GPU/CPU package (sync pip, needs restart)

  torch:                  GET /api/torch/status           current torch status + recommended cu tag
                           POST /api/torch/reinstall       register a reinstall request (runs at startup)

  flash-attention:        GET /api/flash-attention/status status + candidate wheel list
                           POST /api/flash-attention/install install a wheel (needs restart)

  xformers:                GET /api/xformers/status
                           POST /api/xformers/install      direct pip install (needs restart)

  llm-tagger admin:       POST /api/llm-tagger/models/refresh  pull /models, write to preset.model_ids
                           POST /api/llm-tagger/test            connectivity test (does not persist secrets)

One combined router instead of 5: each domain only has 2 routes and they share the
install pattern / restart_required semantics, so splitting further would be too granular.
The frontend Settings page also keeps them in a single drawer.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from ..schemas.installs import (
    FlashAttnInstallRequest,
    LLMConnectionTestRequest,
    LLMModelsRefreshRequest,
    TorchReinstallRequest,
    WD14InstallRequest,
)
from ... import secrets
from ...domain.errors import DomainError, ValidationError
from ...services.runtime import (
    flash_attention as flash_attention_setup,
    onnxruntime as onnxruntime_setup,
    pending_install,
    torch as torch_setup,
    xformers as xformers_setup,
)

router = APIRouter()


# WD14 runtime / GPU package install (PP8) -----------------------------------


@router.get("/api/wd14/runtime")
def wd14_runtime() -> dict[str, Any]:
    """Return which onnxruntime package is currently installed + available EPs + nvidia-smi detection."""
    rt = onnxruntime_setup.current_runtime()
    return {**rt, "cuda_detect": onnxruntime_setup.detect_cuda()}


@router.post("/api/wd14/install")
def wd14_install(body: WD14InstallRequest) -> dict[str, Any]:
    """Switch the onnxruntime package: uninstall the two mutually exclusive packages
    first, then install the target.

    Synchronous pip install, takes minutes; the frontend button must show a loading state.
    onnxruntime is a C extension, so **Studio must be restarted** after installing before
    the EP can actually switch (pip uninstall/reinstall can't hot-swap an already-imported
    .pyd/.so). Returns `restart_required=True` so the frontend can prompt explicitly.
    """
    if body.target not in ("auto", "gpu", "cpu", "directml"):
        raise ValidationError(
            "Unsupported runtime target",
            code="install.target_invalid",
            details={"target": body.target}, http_status=400,
        )
    try:
        res = onnxruntime_setup.install_runtime(body.target)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    stdout = res.pop("stdout", "")
    tail = "\n".join(stdout.splitlines()[-30:])
    # Also return the current process's view (providers is still the old one, for the UI to compare against)
    rt = onnxruntime_setup.current_runtime()
    return {
        **res,
        **rt,
        "cuda_detect": onnxruntime_setup.detect_cuda(),
        "stdout_tail": tail,
    }


# PyTorch runtime / reinstall (PR-S2) ---------------------------------------


@router.get("/api/torch/status")
def torch_status() -> dict[str, Any]:
    """Return torch's current status + driver detection + recommended cu tag +
    misinstall diagnostic flag.

    The UI uses `is_cpu_with_gpu` to decide whether to prominently warn "GPU detected
    but the CPU build is installed". `is_cuda_build_unavailable` flags a driver / WSL
    issue (not fixable by pip; the UI links to docs).
    """
    return torch_setup.current_status()


@router.post("/api/torch/reinstall")
def torch_reinstall(body: TorchReinstallRequest) -> dict[str, Any]:
    """Register a torch reinstall request; executed by the launcher process the next
    time Studio starts.

    Why not install directly: the server process has already imported torch (pulled in
    indirectly by flash_attention_setup etc.), and on Windows
    `torch\\_C.cp311-win_amd64.pyd` is locked, so pip uninstall / replace hits
    [WinError 5] access denied. Instead we write a marker -> user Ctrl+C's and restarts
    -> at that point cli.py hasn't imported torch yet, so pip can replace the files
    normally.

    Returns `{pending: true, target, tag, message}`; the UI shows "please close and restart Studio".
    """
    try:
        tag = torch_setup._decide_target_tag(body.target)
    except ValueError as exc:
        raise ValidationError(
            "Unsupported runtime target",
            code="install.target_invalid",
            details={"target": body.target}, http_status=400,
        ) from exc
    pending_install.register_torch_reinstall(body.target)
    return {
        "pending": True,
        "target": body.target,
        "tag": tag,
        "message": "Reinstall request registered. Press Ctrl+C to close Studio, then rerun studio.bat / studio.sh — torch will be installed automatically on startup (~3 GB, 5-30 minutes), after which the server starts normally.",
    }


# FlashAttention runtime (PR-7b) ---------------------------------------


@router.get("/api/flash-attention/status")
def flash_attn_status() -> dict[str, Any]:
    """Return flash_attn install status + current environment detection + candidate
    wheel list from GitHub.

    Fields the UI doesn't need (score / tags etc.) are stripped from candidates, keeping
    only url/name/notes/usable. At most the first 20 candidates are kept, to avoid a huge
    pile of GitHub release history flooding the screen.
    fetch_error being non-None means the GitHub API request failed (rate limit / network
    / firewall); the UI should show this so the user can paste a URL manually.

    Any unexpected exception is wrapped into fetch_error and returned as 200 -- this is
    diagnostic data pulled as soon as the Settings page mounts, so it's better to degrade
    gracefully to "couldn't fetch candidates" than to 500 the whole UI into "failed to
    load" and make the user think the backend is broken. Real issues get diagnosed from
    the traceback in the server log.
    """
    try:
        status = flash_attention_setup.current_status()
        env = flash_attention_setup.detect_env()
        candidates, fetch_error = flash_attention_setup.find_candidates(env)
        slim = [
            {"url": c["url"], "name": c["name"], "notes": c["notes"], "usable": c["usable"]}
            for c in candidates[:20]
        ]
        return {**status, "env": env, "candidates": slim, "fetch_error": fetch_error}
    except Exception as exc:  # noqa: BLE001
        # logger.exception persists the traceback + trace_id (bound by trace middleware)
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).exception("flash_attn status endpoint failed")
        return {
            "installed": False,
            "version": None,
            "env": {
                "python_tag": None,
                "cuda_tag": None,
                "cuda_ver": None,
                "driver_cuda_ver": None,
                "torch_tag": None,
                "torch_ver": None,
                "torch_cuda_build": None,
                "platform": None,
            },
            "candidates": [],
            "fetch_error": f"Diagnostics failed: {type(exc).__name__}: {exc}",
        }


@router.post("/api/flash-attention/install")
def flash_attn_install(body: FlashAttnInstallRequest) -> dict[str, Any]:
    """Install a flash_attn wheel; url=null uses the service's automatic matching.

    Synchronous pip install (remote wheel ~150MB), can take a few minutes; the UI button
    must show a loading state.
    flash_attn is a C extension, so Studio must be restarted after installing before it
    takes effect; returns restart_required=True.
    """
    try:
        return flash_attention_setup.install(body.url)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


# xformers runtime ------------------------------------------------------


@router.get("/api/xformers/status")
def xformers_status() -> dict[str, Any]:
    """Return xformers install status.

    Much simpler than flash_attention/status -- xformers is installed directly from
    PyPI, so it needs no GitHub candidate wheel list / environment detection detail
    (installed/version in status is enough).
    """
    return xformers_setup.current_status()


@router.post("/api/xformers/install")
def xformers_install() -> dict[str, Any]:
    """pip install xformers --index-url <torch-cu-index>.

    Runs synchronously; the remote wheel is usually tens to hundreds of MB, taking
    minutes. Raises 500 on failure, with the message including the tail of stderr (most
    failures mean the upstream wheel doesn't cover the current torch+cu combination).

    xformers is a C extension, so returns restart_required=True to have the UI prompt a restart.
    """
    try:
        return xformers_setup.install()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


# LLM tagger admin ------------------------------------------------------


def _select_preset(
    tagger_cfg: "secrets.LLMTaggerConfig", preset_id: Optional[str]
) -> "secrets.LLMPresetConfig":
    pid = preset_id or tagger_cfg.current_preset
    for preset in tagger_cfg.presets:
        if preset.id == pid:
            return preset
    return tagger_cfg.active


@router.post("/api/llm-tagger/models/refresh")
def refresh_llm_tagger_models(body: LLMModelsRefreshRequest) -> dict[str, Any]:
    """Read the OpenAI-compatible /models endpoint and save the result into the target
    preset's model_ids.

    Uses current_preset when `preset_id` is not given. Only persisted to secrets on
    success, to avoid writing dirty data when the request fails.
    """
    from ...services.tagging import llm as llm_tagger_svc

    tagger_cfg = secrets.load().llm_tagger
    target = _select_preset(tagger_cfg, body.preset_id)
    base_url = (body.base_url if body.base_url is not None else target.base_url).strip()
    api_key = (
        target.api_key
        if body.api_key is None or body.api_key == secrets.MASK
        else body.api_key.strip()
    )
    if not base_url:
        raise ValidationError(
            "API base URL is required",
            code="llm_tagger.base_url_required", http_status=400,
        )
    try:
        model_ids = llm_tagger_svc.fetch_openai_compatible_models(
            base_url,
            api_key,
            timeout=body.timeout or target.timeout,
        )
    except Exception as exc:  # noqa: BLE001
        raise DomainError(
            f"Could not reach the model service: {exc}",
            code="llm_tagger.connect_failed",
            details={"reason": str(exc)}, http_status=502,
        ) from exc
    selected = target.model if target.model in model_ids else (model_ids[0] if model_ids else target.model)
    preset_patch: dict[str, Any] = {
        "id": target.id,
        "base_url": base_url,
        "model_ids": model_ids,
        "model": selected,
    }
    if body.api_key not in (None, secrets.MASK):
        preset_patch["api_key"] = api_key
    new = secrets.update({"llm_tagger": {"presets": [preset_patch]}})
    return {
        "items": model_ids,
        "preset_id": target.id,
        "secrets": secrets.to_masked_dict(new),
    }


@router.post("/api/llm-tagger/test")
def test_llm_tagger_connection(body: LLMConnectionTestRequest) -> dict[str, Any]:
    """Run a text-only LLM connectivity test without saving form values.

    Defaults come from the target preset (preset_id or current); body fields
    override on top.
    """
    from ...services.tagging import llm as llm_tagger_svc

    tagger_cfg = secrets.load().llm_tagger
    target = _select_preset(tagger_cfg, body.preset_id)
    merged = target.model_dump()
    for key in ("base_url", "model", "endpoint", "timeout", "max_tokens", "temperature"):
        value = getattr(body, key)
        if value is not None:
            merged[key] = value
    if body.api_key is not None and body.api_key != secrets.MASK:
        merged["api_key"] = body.api_key
    cfg = secrets.LLMPresetConfig(**merged)
    if not cfg.base_url.strip():
        raise ValidationError(
            "API base URL is required",
            code="llm_tagger.base_url_required", http_status=400,
        )
    if not cfg.model.strip():
        raise ValidationError(
            "A model must be selected",
            code="llm_tagger.model_required", http_status=400,
        )
    return llm_tagger_svc.test_openai_compatible_connection(
        cfg.base_url,
        cfg.api_key,
        cfg.model,
        endpoint=cfg.endpoint,
        timeout=cfg.timeout,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
