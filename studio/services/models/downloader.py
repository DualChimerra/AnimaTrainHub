"""Per-model high-level download flows + async status tracking (PR-3.8, the 3rd of
a 4-way split).

Contains per-model download functions for Anima / Krea 2 training assets and
various tooling models; calls sources.py's download_flat[_ms] to do the actual
download, and paths.py / families to get the target Path and model manifest.

Async: DownloadStatus / start_download_async / trigger wrap a synchronous download
in a background thread, pushing model_download_changed to the event_bus.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ... import secrets
from ...infrastructure.event_bus import bus
from .families.anima import (
    ANIMA_REPO,
    ANIMA_VAE_PATH,
    ANIMA_VARIANTS,
    LATEST_ANIMA,
    QWEN_FILES,
    QWEN_REPO,
    T5_FILES,
    T5_REPO,
    anima_main_target,
    qwen_dir,
    selected_anima_variant,
    t5_tokenizer_dir,
)
from .families.krea2 import (
    KREA2_VARIANTS,
    LATEST_KREA2,
    QWEN3_VL_FILES,
    QWEN3_VL_FP8_FILE,
    QWEN3_VL_FP8_REPO,
    QWEN3_VL_FP8_SMALL_FILES,
    QWEN3_VL_FP8_SUBPATH,
    QWEN3_VL_REPO,
    krea2_main_target,
    qwen3_vl_dir,
    qwen3_vl_fp8_dir,
)
from .paths import (
    CLTAGGER_VERSIONS,
    DEFAULT_UPSCALER,
    TAEFLUX_FILES,
    TAEFLUX_REPO,
    UPSCALER_EXTS,
    UPSCALER_VARIANTS,
    WD14_FILES,
    ccip_model_dir,
    cltagger_canonical_file_paths,
    cltagger_required_files,
    cltagger_target_root,
    eval_model_target_dir,
    models_root,
    qwen_image_vae_target,
    selected_upscaler,
    taeflux_dir,
    upscaler_dir,
    upscaler_target,
    wd14_target_dir,
)
from . import sources as _sources
from .sources import MS_ANIMA_TEXT_ENCODER_PATH

# Note: cross-file calls to download_flat[_ms] / _get_download_source /
# _resolve_endpoint / _ms_wd14_repo_id must always go through _sources.X(...) --
# that's what makes test monkeypatching of `studio.services.models.sources.X`
# actually take effect. If instead written as `from .sources import X`, it would
# bind a local name in this module, and patching the sources module would have no
# effect on calls made from within downloader.

def download_taeflux(
    *, root: Optional[Path] = None,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Synchronously download TAEFlux (config + weights) locally. Returns False if
    any single file fails."""
    target_dir = taeflux_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for f in TAEFLUX_FILES:
        target = target_dir / f
        if not _sources.download_flat(TAEFLUX_REPO, f, target, on_log=on_log):
            ok = False
    return ok


def download_anima_main(
    root: Path, variant: str, *, on_log: Callable[[str], None] = print
) -> bool:
    if variant == "latest":
        variant = LATEST_ANIMA
    if variant not in ANIMA_VARIANTS:
        on_log(f"✗ Unknown variant {variant!r}")
        return False
    target = anima_main_target(root, variant)
    subpath = ANIMA_VARIANTS[variant]
    on_log(f"\n\U0001F4E5 Anima base model [{variant}] (~4 GB)")
    if _sources._source_for("training") == "modelscope":
        return _sources.download_flat_ms(ANIMA_REPO, subpath, target, on_log=on_log)
    return _sources.download_flat(ANIMA_REPO, subpath, target, on_log=on_log)


def download_anima_vae(root: Path, *, on_log: Callable[[str], None] = print) -> bool:
    # The destination is a family-agnostic shared asset (also used by Krea2); the
    # download channel goes through the Anima repo (where the file lives).
    target = qwen_image_vae_target(root)
    on_log("\n\U0001F4E5 Anima VAE (~250 MB)")
    if _sources._source_for("training") == "modelscope":
        return _sources.download_flat_ms(ANIMA_REPO, ANIMA_VAE_PATH, target, on_log=on_log)
    return _sources.download_flat(ANIMA_REPO, ANIMA_VAE_PATH, target, on_log=on_log)


def download_krea2_main(
    root: Path, variant: str, *, on_log: Callable[[str], None] = print
) -> bool:
    """Download Krea2 from the official HuggingFace repo or the ModelScope Comfy-Org
    mirror."""
    if variant == "latest":
        variant = LATEST_KREA2
    info = KREA2_VARIANTS.get(variant)
    if info is None:
        on_log(f"✗ Unknown Krea 2 variant {variant!r}")
        return False
    target = krea2_main_target(root, variant)
    size_gb = float(info.get("size_estimate", 0)) / 1e9
    on_log(f"\n\U0001F4E5 Krea 2 [{variant}] (~{size_gb:.1f} GB) -> {target}")
    if _sources._source_for("training") == "modelscope":
        return _sources.download_flat_ms(
            str(info["ms_repo"]), str(info["ms_subpath"]), target, on_log=on_log,
        )
    return _sources.download_flat(
        str(info["repo"]), str(info["subpath"]), target, on_log=on_log,
    )


def download_qwen3_vl(
    root: Path, *, on_log: Callable[[str], None] = print
) -> bool:
    """Download the full Qwen3-VL-4B-Instruct transformers directory used by Krea 2."""
    target_dir = qwen3_vl_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    use_modelscope = _sources._source_for("training") == "modelscope"
    source_label = "ModelScope" if use_modelscope else "HuggingFace"
    on_log(
        f"\n\U0001F4E5 Krea 2 text encoder Qwen3-VL-4B-Instruct "
        f"(~8.89 GB, {source_label}) -> {target_dir}"
    )
    ok = True
    for filename in QWEN3_VL_FILES:
        target = target_dir / filename
        if use_modelscope:
            downloaded = _sources.download_flat_ms(
                QWEN3_VL_REPO, filename, target, on_log=on_log,
            )
        else:
            downloaded = _sources.download_flat(
                QWEN3_VL_REPO, filename, target, on_log=on_log,
            )
        if not downloaded:
            ok = False
    return ok


def download_qwen3_vl_fp8(
    root: Path, *, on_log: Callable[[str], None] = print
) -> bool:
    """Download the official single-file fp8_scaled text encoder + small
    config/tokenizer files into a separate directory.

    The weights come from Comfy-Org/Krea-2 (same repo layout on HF and
    ModelScope); the small files come from the official Qwen repo (not present in
    the single-file build, but needed by the loader to build the structure from
    config and to encode with the tokenizer).
    """
    target_dir = qwen3_vl_fp8_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    use_ms = _sources._source_for("training") == "modelscope"
    on_log(f"\n\U0001F4E5 Krea 2 text encoder Qwen3-VL fp8 (~5.24 GB) -> {target_dir}")
    download = _sources.download_flat_ms if use_ms else _sources.download_flat
    ok = download(
        QWEN3_VL_FP8_REPO, QWEN3_VL_FP8_SUBPATH,
        target_dir / QWEN3_VL_FP8_FILE, on_log=on_log,
    )
    for filename in QWEN3_VL_FP8_SMALL_FILES:
        if not download(
            QWEN3_VL_REPO, filename, target_dir / filename, on_log=on_log,
        ):
            ok = False
    return ok


def download_qwen3(root: Path, *, on_log: Callable[[str], None] = print) -> bool:
    """Download the text encoder (Qwen3).

    - HuggingFace source: downloads the 6 files needed for a complete directory
      from Qwen/Qwen3-0.6B-Base.
    - ModelScope source: downloads weight files from circlestone-labs/Anima, and
      separately fills in tokenizer / config files from Qwen/Qwen3-0.6B-Base, so
      the local text_encoders/ ends up as a directory transformers can load
      directly.
    """
    target_dir = qwen_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    if _sources._source_for("training") == "modelscope":
        on_log(f"\n\U0001F4E5 Anima text encoder (ModelScope weights + HF tokenizer) -> {target_dir}")
        # The ModelScope Anima repo only has the weights; the training script
        # still requires a complete transformers directory.
        ok &= _sources.download_flat_ms(
            ANIMA_REPO,
            MS_ANIMA_TEXT_ENCODER_PATH,
            target_dir / "model.safetensors",
            on_log=on_log,
        )
        for f in QWEN_FILES:
            if f == "model.safetensors":
                continue
            if not _sources.download_flat(QWEN_REPO, f, target_dir / f, on_log=on_log):
                ok = False
        return ok

    on_log(f"\n\U0001F4E5 Qwen3-0.6B-Base (~1.2 GB) -> {target_dir}")
    for f in QWEN_FILES:
        if not _sources.download_flat(QWEN_REPO, f, target_dir / f, on_log=on_log):
            ok = False
    return ok


def download_t5_tokenizer(
    root: Path, *, on_log: Callable[[str], None] = print
) -> bool:
    target_dir = t5_tokenizer_dir(root)
    on_log(f"\n\U0001F4E5 T5 tokenizer (3 files) -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for f in T5_FILES:
        if not _sources.download_flat(T5_REPO, f, target_dir / f, on_log=on_log):
            ok = False
    return ok


def download_cltagger(
    target_root: Path,
    cfg: Optional["secrets.CLTaggerConfig"] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    cfg = cfg or secrets.load().cltagger
    model_path, tag_mapping_path = cltagger_canonical_file_paths(
        cfg.model_id,
        cfg.model_path,
        cfg.tag_mapping_path,
    )
    on_log(f"\n\U0001F4E5 CLTagger -> {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    ok = True
    for f in cltagger_required_files(model_path, tag_mapping_path):
        if not _sources.download_flat(cfg.model_id, f, target_root / f, on_log=on_log):
            ok = False
    return ok


def download_upscaler(
    label: str = DEFAULT_UPSCALER,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Download upscaler weights to `{models_root}/upscalers/{filename}`.

    Source selection: preference is taken from _sources._source_for("upscaler");
    when the preferred source doesn't have the file, transparently falls back to
    the other source (e.g. R-ESRGAN_4x+Anime6B has no HF mirror -> even if the
    user picked HF, it goes through MS).
    """
    if label not in UPSCALER_VARIANTS:
        on_log(f"✗ Unknown upscaler {label!r}")
        return False
    info = UPSCALER_VARIANTS[label]
    hf_src = info.get("hf")
    ms_src = info.get("ms")
    if hf_src is None and ms_src is None:
        on_log(f"✗ Upscaler {label!r} has no configured download source")
        return False

    target = upscaler_target(label, root)
    size_mb = info.get("size_mb", 64)
    prefer_ms = _sources._source_for("upscaler") == "modelscope"
    on_log(f"\n\U0001F4E5 Upscaler {label} (~{size_mb} MB) -> {target}")

    if prefer_ms and ms_src is not None:
        return _sources.download_flat_ms(ms_src[0], ms_src[1], target, on_log=on_log)
    if hf_src is not None:
        return _sources.download_flat(hf_src[0], hf_src[1], target, on_log=on_log)
    # HF preferred but missing -> fall back to ModelScope
    on_log("   ⚠ No HF mirror, falling back to ModelScope")
    return _sources.download_flat_ms(ms_src[0], ms_src[1], target, on_log=on_log)  # type: ignore[index]


def download_upscaler_custom(
    source: str,
    repo_id: str,
    filename: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Custom-repo download: the user specifies an HF/MS repo + filename, saved to
    `{upscalers}/{filename}`.

    The extension allowlist matches UPSCALER_EXTS (.pth / .safetensors). filename
    is only used for the saved filename -- the in-repo subpath is passed straight
    through as repo_id + filename (most upscaler repos keep their weights at the
    root; if a subdirectory is needed, the user can write a relative path like
    `subdir/foo.pth` in filename, but it's stripped down to a bare filename on
    save, to prevent path traversal).
    """
    if source not in ("hf", "ms"):
        on_log(f"✗ Unknown download source {source!r} (supported: hf / ms)")
        return False
    repo_subpath = filename
    save_name = Path(filename).name  # strip the directory prefix, keep only the bare filename
    if "/" in save_name or "\\" in save_name or ".." in save_name:
        on_log(f"✗ Invalid filename {save_name!r}")
        return False
    if not save_name.lower().endswith(UPSCALER_EXTS):
        on_log(f"✗ Only {UPSCALER_EXTS} extensions are supported, got {save_name!r}")
        return False
    target = upscaler_dir(root) / save_name
    on_log(f"\n\U0001F4E5 Custom upscaler [{source}] {repo_id}/{repo_subpath} -> {target}")
    if source == "ms":
        return _sources.download_flat_ms(repo_id, repo_subpath, target, on_log=on_log)
    return _sources.download_flat(repo_id, repo_subpath, target, on_log=on_log)


def download_main_custom(
    repo_id: str,
    filename: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Single-file download of a third-party base model from a unified source
    candidate -> `{models_root}/diffusion_models/`.

    filename is the in-repo path (may include subdirectories), stripped down to a
    bare filename on save (same directory as the official variants, where the
    filename itself is the identity). Goes through download_flat's default logic
    of "use MS if a mapping exists, otherwise HF" -- a custom repo with no mapping
    just connects to HF directly.
    """
    save_name = Path(filename).name
    if not save_name.lower().endswith(".safetensors"):
        on_log(f"✗ Only .safetensors is supported, got {save_name!r}")
        return False
    r = root or models_root()
    target = r / "diffusion_models" / save_name
    on_log(f"\n\U0001F4E5 Custom base model {repo_id}/{filename} -> {target}")
    return _sources.download_flat(repo_id, filename, target, on_log=on_log)


def _download_candidate(domain: str, filename: str) -> "secrets.SourceCandidate":
    """Look up a download-type candidate for this domain by filename (shared by
    trigger / delete)."""
    for c in secrets.load().model_sources.get(domain, []):
        if c.kind == "download" and c.filename == filename:
            return c
    raise ValueError(f"no download candidate {filename!r} for domain {domain!r}")


def _custom_family(model_id: str) -> Optional[str]:
    """A model_id of the form `{family}_custom` -> family id (returns None for an
    unregistered family)."""
    from .families import FAMILY_ASSETS

    family = model_id[: -len("_custom")]
    return family if family in FAMILY_ASSETS else None


def download_wd14(
    model_id: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Download the two files for a single WD14 model_id to
    `{models_root}/wd14/{safe_id}/`.

    ModelScope source: SmilingWolf/* -> fireicewolf/* (fireicewolf mirrors the
    full set on ModelScope). Automatically falls back to HF when there's no MS
    mapping (i.e. not a SmilingWolf-prefixed id).
    """
    r = root or models_root()
    target = wd14_target_dir(r, model_id)
    target.mkdir(parents=True, exist_ok=True)
    ok = True
    if _sources._source_for("wd14") == "modelscope":
        ms_repo = _sources._ms_wd14_repo_id(model_id)
        if ms_repo:
            on_log(f"\n\U0001F4E5 WD14 {model_id} -> {target} (via ModelScope: {ms_repo})")
            for f in WD14_FILES:
                if not _sources.download_flat_ms(ms_repo, f, target / f, on_log=on_log):
                    ok = False
            return ok
        on_log(f"\n\U0001F4E5 WD14 {model_id}: no ModelScope mapping, falling back to HuggingFace")
    else:
        on_log(f"\n\U0001F4E5 WD14 {model_id} -> {target}")
    for f in WD14_FILES:
        if not _sources.download_flat(model_id, f, target / f, on_log=on_log):
            ok = False
    return ok


def download_eval_model(
    kind: str,
    model_id: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Download the whole repo for a CLIP / DINO eval model to
    `{models_root}/eval/{kind}/{safe_id}/`.

    Source selection: goes through MS when the eval source is set to modelscope
    and a mirror mapping exists, otherwise (no mapping / HF selected) goes
    through HuggingFace -- the same "use MS if mapped, otherwise fall back to HF"
    logic as wd14.
    """
    r = root or models_root()
    target = eval_model_target_dir(r, kind, model_id)
    if _sources._source_for("eval") == "modelscope":
        ms_repo = _sources._ms_eval_repo_id(model_id)
        if ms_repo:
            on_log(f"\n\U0001F4E5 {kind.upper()} {model_id} -> {target} (via ModelScope: {ms_repo})")
            return _sources.download_snapshot_ms(ms_repo, target, on_log=on_log)
        on_log(f"\n\U0001F4E5 {kind.upper()} {model_id}: no ModelScope mapping, falling back to HuggingFace")
    else:
        on_log(f"\n\U0001F4E5 {kind.upper()} {model_id} -> {target}")
    return _sources.download_snapshot(model_id, target, on_log=on_log)


def ensure_eval_model(
    kind: str,
    model_id: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> Path:
    """Return the local directory for an eval model, downloading first if missing
    (lazy-load fallback, using the same path as the download card).

    Same pattern as wd14's `_resolve_model_dir`: if the user hasn't pre-downloaded
    when running eval, it auto-downloads into the project directory and
    `from_pretrained` points at it. If already present (has config.json), returns
    it directly.

    If ``model_id`` is itself an existing local model directory (the user typed a
    path directly into the text field), it's used as-is rather than treated as a
    repo id to download.
    """
    local = Path(str(model_id)).expanduser()
    if local.is_dir() and (local / "config.json").exists():
        return local
    r = root or models_root()
    target = eval_model_target_dir(r, kind, model_id)
    if (target / "config.json").exists():
        return target
    download_eval_model(kind, model_id, r, on_log=on_log)
    return target


# CCIP (anime character identity): only download the 3 files under the variant
# subdirectory, not the whole repo (which also has torch ckpt + png files).
CCIP_REPO = "deepghs/ccip_onnx"
CCIP_FILES = ("model_feat.onnx", "model_metrics.onnx", "metrics.json")


def download_ccip_model(
    variant: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """Download the 3 files for a given deepghs/ccip_onnx variant to
    `{models_root}/eval/ccip/{variant}/`."""
    r = root or models_root()
    eval_ccip_root = r / "eval" / "ccip"
    patterns = [f"{variant}/{f}" for f in CCIP_FILES]
    on_log(f"\n\U0001F4E5 CCIP {variant} -> {ccip_model_dir(r, variant)}")
    return _sources.download_snapshot(
        CCIP_REPO, eval_ccip_root, allow_patterns=patterns, on_log=on_log,
    )


def ensure_ccip_model(
    variant: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> Path:
    """Return the local directory for a CCIP variant, downloading first if any of
    the 3 files are missing (lazy-load fallback)."""
    r = root or models_root()
    target = ccip_model_dir(r, variant)
    if all((target / f).exists() for f in CCIP_FILES):
        return target
    download_ccip_model(variant, r, on_log=on_log)
    return target


# ---------------------------------------------------------------------------
# Async download state machine
# ---------------------------------------------------------------------------


@dataclass
class DownloadStatus:
    key: str
    status: str  # pending | running | done | failed
    started_at: float = 0.0
    finished_at: Optional[float] = None
    message: str = ""
    log: list[str] = field(default_factory=list)


_LOCK = threading.Lock()
_DOWNLOADS: dict[str, DownloadStatus] = {}


def get_status_snapshot() -> dict[str, dict[str, Any]]:
    """For endpoint serialization: a shallow copy of every current download status."""
    with _LOCK:
        return {
            k: {
                "key": v.key,
                "status": v.status,
                "started_at": v.started_at,
                "finished_at": v.finished_at,
                "message": v.message,
                "log_tail": v.log[-30:],
            }
            for k, v in _DOWNLOADS.items()
        }


def _failure_summary(log: list[str]) -> str:
    """Extract a single actionable failure reason from the download log (for the
    frontend toast / message).

    download_flat writes errors as `   ✗ ...`; gated / auth failures append a
    further `   ↳ ...hint` line (with token / access-request guidance).
    Prefers the line with a hint, otherwise falls back to the last ✗ error
    line, and finally to a generic string if neither exists. This avoids the
    frontend showing a red badge with no clue why (the reason otherwise only
    lives in the terminal / a collapsed log).
    """
    err = next((ln.strip() for ln in reversed(log) if ln.lstrip().startswith("✗")), "")
    hint = next((ln.strip() for ln in reversed(log) if "↳" in ln), "")
    if hint:
        return f"{err} {hint}".strip() if err else hint
    return err or "Download failed; see the download log for details"


def start_download_async(
    key: str, fn: Callable[[Callable[[str], None]], bool]
) -> DownloadStatus:
    """Start a background thread running `fn(on_log)`; fn returns True on success.

    `key` is the task identifier; starting again with the same key while it's
    still running reuses the existing status.
    On completion / failure, publishes a `model_download_changed` SSE event via
    `bus.publish`.
    """
    with _LOCK:
        existing = _DOWNLOADS.get(key)
        if existing and existing.status == "running":
            return existing
        ds = DownloadStatus(
            key=key, status="running", started_at=time.time(), log=[]
        )
        _DOWNLOADS[key] = ds

    def _on_log(line: str) -> None:
        with _LOCK:
            ds.log.append(line)
            if len(ds.log) > 200:
                del ds.log[:-200]
        # Echoed to backend stdout -- the UI ring buffer only holds 200 lines, so
        # early lines of a long download would otherwise get truncated; printing
        # keeps the full stream in studio_*.log / the terminal, so it can be
        # grepped directly when debugging / on-call. Done outside the lock to
        # avoid holding it during I/O and slowing down other download tasks'
        # log writes.
        print(line, flush=True)

    def _run() -> None:
        bus.publish({
            "type": "model_download_changed",
            "key": key,
            "status": "running",
        })
        try:
            ok = fn(_on_log)
            with _LOCK:
                ds.status = "done" if ok else "failed"
                ds.finished_at = time.time()
                if not ok:
                    ds.message = _failure_summary(ds.log)
        except Exception as exc:
            with _LOCK:
                ds.status = "failed"
                ds.finished_at = time.time()
                ds.message = str(exc)
                ds.log.append(f"[exception] {exc}")
        bus.publish({
            "type": "model_download_changed",
            "key": key,
            "status": ds.status,
        })

    threading.Thread(
        target=_run, daemon=True, name=f"model-dl-{key}"
    ).start()
    bus.publish({
        "type": "model_download_changed",
        "key": key,
        "status": "running",
    })
    return ds


def delete_asset(model_id: str, variant: Optional[str] = None) -> None:
    """Delete an already-downloaded asset (the reverse of the download button: the
    user deletes first, then re-downloads).

    Target paths are always resolved by server-side target functions -- arbitrary
    paths are never accepted; deletion is refused while the corresponding key's
    download is in progress. Covers every asset id in the download center:
    the training models section (base model variants / VAE / text encoder /
    tokenizer), tagging (wd14 / cltagger), eval metrics (clip / dino / ccip), and
    upscalers (presets + custom filenames). If the file is in use (model loaded /
    training in progress), the OSError is passed through as an actionable error.
    """
    import shutil

    root = models_root()
    target: Path
    key = model_id
    if model_id == "anima_main":
        v = variant or ""
        if v not in ANIMA_VARIANTS:
            raise ValueError(f"unknown anima variant {variant!r}")
        key = f"anima_main:{v}"
        target = anima_main_target(root, v)
    elif model_id == "krea2_main":
        v = variant or ""
        if v not in KREA2_VARIANTS:
            raise ValueError(f"unknown Krea 2 variant {variant!r}")
        key = f"krea2_main:{v}"
        target = krea2_main_target(root, v)
    elif model_id == "anima_vae":
        target = qwen_image_vae_target(root)
    elif model_id == "qwen3":
        target = qwen_dir(root)
    elif model_id == "t5_tokenizer":
        target = t5_tokenizer_dir(root)
    elif model_id == "krea2_text_encoder":
        target = qwen3_vl_dir(root)
    elif model_id == "krea2_text_encoder_fp8":
        target = qwen3_vl_fp8_dir(root)
    elif model_id == "wd14":
        if not variant:
            raise ValueError("wd14 needs variant=model_id")
        key = f"wd14:{variant}"
        target = wd14_target_dir(root, variant)
    elif model_id == "cltagger":
        preset = CLTAGGER_VERSIONS.get(variant or "")
        if preset is None:
            raise ValueError(f"unknown cltagger variant {variant!r}")
        key = f"cltagger:{variant}"
        # Only delete this version's subdirectory -- v1/v2 can coexist under the
        # same repo root
        target = cltagger_target_root(root, preset["model_id"]) / Path(
            preset["model_path"]
        ).parent
    elif model_id in ("eval_clip", "eval_dino"):
        if not variant:
            raise ValueError(f"{model_id} needs variant=model_id")
        kind = "clip" if model_id == "eval_clip" else "dino"
        key = f"{model_id}:{variant}"
        target = eval_model_target_dir(root, kind, variant)
    elif model_id == "eval_ccip":
        if not variant:
            raise ValueError("eval_ccip needs variant=<ccip variant name>")
        key = f"eval_ccip:{variant}"
        target = ccip_model_dir(root, variant)
    elif model_id == "upscaler":
        if not variant:
            raise ValueError("upscaler needs variant=label")
        # a preset label or a custom filename; upscaler_target has its own
        # path-traversal validation
        key = (
            f"upscaler:{variant}"
            if variant in UPSCALER_VARIANTS
            else f"upscaler:custom:{variant}"
        )
        target = upscaler_target(variant, root)
    elif model_id == "cltagger_custom":
        # delete the entire dedicated root for a fork repo (kept isolated from
        # the official repo directory, so this is safe)
        if not variant:
            raise ValueError("cltagger_custom needs variant=repo")
        key = f"cltagger_custom:{variant}"
        target = cltagger_target_root(root, variant)
    elif model_id == "upscaler_custom":
        # a file saved from a unified-source download candidate (filename is the identity)
        if not variant:
            raise ValueError("upscaler_custom needs variant=filename")
        save_name = Path(variant).name
        key = f"upscaler:custom:{save_name}"
        target = upscaler_dir(root) / save_name
    elif model_id.endswith("_custom") and _custom_family(model_id) is not None:
        if not variant:
            raise ValueError(f"{model_id} needs variant=filename")
        key = f"{model_id}:{variant}"
        target = root / "diffusion_models" / Path(variant).name
    else:
        raise ValueError(f"asset {model_id!r} does not support deletion")

    with _LOCK:
        existing = _DOWNLOADS.get(key)
        if existing and existing.status == "running":
            raise RuntimeError(f"{key} is still downloading and cannot be deleted")

    try:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Delete failed (the file may be in use — the model is loaded or training is running): {exc}"
        ) from exc


def trigger(model_id: str, variant: Optional[str] = None) -> str:
    """Convenience entry point for endpoints: picks the matching download_*
    function based on model_id + starts it asynchronously.

    Returns the status key (used by the frontend to build the SSE key it cares
    about).
    """
    root = models_root()
    if model_id == "anima_main":
        v = variant or "latest"
        if v == "latest":
            v = LATEST_ANIMA
        if v not in ANIMA_VARIANTS:
            raise ValueError(f"unknown anima variant {variant!r}")
        key = f"anima_main:{v}"
        start_download_async(
            key,
            lambda log: download_anima_main(root, v, on_log=log),
        )
        return key
    if model_id == "anima_vae":
        key = "anima_vae"
        start_download_async(
            key, lambda log: download_anima_vae(root, on_log=log)
        )
        return key
    if model_id == "krea2_main":
        v = variant or "latest"
        if v == "latest":
            v = LATEST_KREA2
        if v not in KREA2_VARIANTS:
            raise ValueError(f"unknown Krea 2 variant {variant!r}")
        key = f"krea2_main:{v}"
        start_download_async(
            key, lambda log: download_krea2_main(root, v, on_log=log)
        )
        return key
    if model_id == "krea2_text_encoder":
        key = "krea2_text_encoder"
        start_download_async(
            key, lambda log: download_qwen3_vl(root, on_log=log)
        )
        return key
    if model_id == "krea2_text_encoder_fp8":
        key = "krea2_text_encoder_fp8"
        start_download_async(
            key, lambda log: download_qwen3_vl_fp8(root, on_log=log)
        )
        return key
    if model_id == "qwen3":
        key = "qwen3"
        start_download_async(
            key, lambda log: download_qwen3(root, on_log=log)
        )
        return key
    if model_id == "t5_tokenizer":
        key = "t5_tokenizer"
        start_download_async(
            key, lambda log: download_t5_tokenizer(root, on_log=log)
        )
        return key
    if model_id == "cltagger":
        cfg = secrets.load().cltagger
        # variant can specify a preset label (overriding cfg's current
        # repo/path), letting the UI one-click download a version other than
        # the "currently selected" one. When unspecified, uses cfg's current
        # path.
        if variant:
            preset = CLTAGGER_VERSIONS.get(variant)
            if preset is None:
                raise ValueError(f"unknown cltagger variant {variant!r}")
            cfg = secrets.CLTaggerConfig(
                **{
                    **cfg.model_dump(),
                    "model_id": preset["model_id"],
                    "model_path": preset["model_path"],
                    "tag_mapping_path": preset["tag_mapping_path"],
                }
            )
            key = f"cltagger:{variant}"
        else:
            key = "cltagger"
        target = cltagger_target_root(root, cfg.model_id)
        start_download_async(
            key, lambda log: download_cltagger(target, cfg, on_log=log)
        )
        return key
    if model_id == "wd14":
        if not variant:
            raise ValueError("wd14 needs variant=model_id")
        key = f"wd14:{variant}"
        start_download_async(
            key, lambda log: download_wd14(variant, root, on_log=log)
        )
        return key
    if model_id in ("eval_clip", "eval_dino"):
        if not variant:
            raise ValueError(f"{model_id} needs variant=model_id")
        kind = "clip" if model_id == "eval_clip" else "dino"
        key = f"{model_id}:{variant}"
        start_download_async(
            key, lambda log: download_eval_model(kind, variant, root, on_log=log)
        )
        return key
    if model_id == "eval_ccip":
        if not variant:
            raise ValueError("eval_ccip needs variant=<ccip variant name>")
        key = f"eval_ccip:{variant}"
        start_download_async(
            key, lambda log: download_ccip_model(variant, root, on_log=log)
        )
        return key
    if model_id == "upscaler":
        label = variant or DEFAULT_UPSCALER
        if label not in UPSCALER_VARIANTS:
            raise ValueError(f"unknown upscaler variant {variant!r}")
        key = f"upscaler:{label}"
        start_download_async(
            key, lambda log: download_upscaler(label, root, on_log=log)
        )
        return key
    if model_id == "cltagger_custom":
        # a fork-repo candidate (a mirror replacement after retirement, D4):
        # variant=repo, both files' relative paths come from the candidate's
        # extra, downloaded into that fork's dedicated root.
        if not variant:
            raise ValueError("cltagger_custom needs variant=repo")
        cand = next(
            (c for c in secrets.load().model_sources.get("cltagger", [])
             if c.kind == "download" and c.repo == variant),
            None,
        )
        if cand is None:
            raise ValueError(f"no cltagger download candidate {variant!r}")
        cfg = secrets.CLTaggerConfig(**{
            **secrets.load().cltagger.model_dump(),
            "model_id": cand.repo,
            "model_path": cand.extra.get("model_path", ""),
            "tag_mapping_path": cand.extra.get("tag_mapping_path", ""),
        })
        key = f"cltagger_custom:{variant}"
        target = cltagger_target_root(root, cand.repo)
        start_download_async(
            key, lambda log: download_cltagger(target, cfg, on_log=log)
        )
        return key
    if model_id == "upscaler_custom":
        # a unified-source download candidate (variant=filename; repo comes from
        # the candidate record, source follows the global
        # download_sources.upscaler setting). key matches the disk-scan row.
        if not variant:
            raise ValueError("upscaler_custom needs variant=filename")
        cand = _download_candidate("upscaler", variant)
        key = f"upscaler:custom:{Path(variant).name}"
        source = "ms" if _sources._source_for("upscaler") == "modelscope" else "hf"
        start_download_async(
            key,
            lambda log: download_upscaler_custom(
                source, cand.repo, variant, root, on_log=log),
        )
        return key
    if model_id.endswith("_custom"):
        from .families import FAMILY_ASSETS

        family = model_id[: -len("_custom")]
        if family in FAMILY_ASSETS:
            if not variant:
                raise ValueError(f"{model_id} needs variant=filename")
            cand = _download_candidate(family, variant)
            key = f"{model_id}:{variant}"
            start_download_async(
                key,
                lambda log: download_main_custom(
                    cand.repo, variant, root, on_log=log),
            )
            return key
    raise ValueError(f"unknown model_id {model_id!r}")
