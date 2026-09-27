#!/usr/bin/env python3
"""Long-running daemon subprocess for test image generation.

Started by studio/services/inference_daemon.py; JSON-over-stdio protocol:
  stdin  ← {"id": "<req_id>", "action": "generate"|"unload"|"ping", ...}
  stdout → {"id": "<req_id>"|"_evt", "kind": "ready"|"started"|"image_done"|
             "done"|"error"|"loaded"|"unloaded", ...}

stdout carries only the protocol; all logging goes to stderr (to avoid
polluting the protocol stream).

Design:
  - Pushes _evt ready right after startup (means import / sys.path setup is
    done; the model isn't loaded yet)
  - Lazily loads the model when the first generate task arrives (30-60s),
    pushes _evt loaded
  - Subsequent tasks reuse the model + adapters; adapters are only
    unloaded/re-injected when lora_configs changes
  - Single-threaded, serial processing (one task at a time); the server side
    guarantees no concurrent submissions

Usage (CLI debugging):
    python runtime/anima_daemon.py
    then feed a line of JSON on stdin:
        {"id":"r1","action":"generate","task_id":1,"output_dir":"/tmp/g","config":{...}}
"""
from __future__ import annotations

import base64
import io
import json
import logging
import random
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import torch

# Same sys.path handling as anima_generate.py (so anima_train / studio can be
# imported). anima_train + train_monitor both live under runtime/, so
# _THIS_DIR alone is enough.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_train as _T  # noqa: E402

from studio.domain.comfy_parity import force_comfy_parity_runtime_config  # noqa: E402
from studio.services.inference.core import LoRAMeta, LoRASpec, apply_loras, read_lora_meta  # noqa: E402

# Warm up the transformers.generation -> sklearn -> scipy.special import
# chain. transformers 5.x's AutoModelForCausalLM.from_pretrained
# transitively imports this chain while loading the text encoder; a cold
# import of scipy.special on Windows + Python 3.13 + an environment that has
# already loaded a GB-scale model (system RAM tight) can take several
# minutes (measured with py-spy). Moved to the daemon's import phase, to pay
# for it once while RAM is still plentiful.
try:
    import transformers.generation.candidate_generator  # noqa: F401
except Exception:
    pass

# Logging goes to stderr, stdout is reserved for the protocol
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("anima_daemon")


# ---------------------------------------------------------------------------
# Protocol output
# ---------------------------------------------------------------------------


def _emit(msg: dict[str, Any]) -> None:
    """Write one protocol message to stdout (line-delimited JSON)."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _emit_evt(kind: str, **extra: Any) -> None:
    _emit({"id": "_evt", "kind": kind, **extra})


def _emit_for(req_id: str, kind: str, **extra: Any) -> None:
    _emit({"id": req_id, "kind": kind, **extra})


# ---------------------------------------------------------------------------
# Model management (lazy load + cache)
# ---------------------------------------------------------------------------


def _lora_topology(meta: LoRAMeta) -> tuple:
    # weight_decompose / rs_lora change the network structure (the former
    # adds a dora_scale tensor, the latter changes the effective-alpha
    # formula), so different settings can't take the hot weight-swap path and
    # must be re-injected. lora_reg_dims directly changes per-layer rank ->
    # even the same base rank with a different pattern config can't be reused.
    reg = meta.lora_reg_dims
    reg_key: Any = tuple(sorted(reg.items())) if reg else None
    return (meta.rank, meta.alpha, meta.algo, meta.factor,
            meta.weight_decompose, meta.rs_lora, reg_key)


def _load_lora_state_dict(path: str, device: str, dtype: Any) -> dict[str, Any]:
    from safetensors import safe_open

    sd: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).to(device=device, dtype=dtype)
    return sd


def _reload_adapter_weights(adapter: Any, spec: LoRASpec, device: str, dtype: Any) -> None:
    _set_lora_multiplier(adapter, spec.scale)
    result = adapter.load_state_dict(
        _load_lora_state_dict(spec.path, device, dtype),
        strict=False,
    )
    missing = len(getattr(result, "missing_keys", []) or [])
    unexpected = len(getattr(result, "unexpected_keys", []) or [])
    logger.info(
        f"hot-swapped LoRA weights: {Path(spec.path).name} "
        f"(scale={spec.scale}; missing={missing}, unexpected={unexpected})"
    )


def _move_module_to_device(module: Any, device: str) -> None:
    if module is None or not hasattr(module, "to"):
        return
    module.to(device)


def _move_adapter_to_device(adapter: Any, device: str, dtype: Any) -> None:
    network = getattr(adapter, "network", None)
    if network is None or not hasattr(network, "to"):
        return
    network.to(device=device, dtype=dtype)


class GenerationCanceled(BaseException):
    """Cancel signal.

    Deliberately subclasses BaseException rather than Exception: the cancel
    is raised by step_callback inside the sampling step, and the sampler
    wraps its call to step_callback in `except Exception: pass` (a callback
    failure shouldn't wreck the sampling run). Subclassing Exception would
    get silently swallowed by that layer, making it impossible to interrupt
    sampling. Every catch site explicitly writes `except GenerationCanceled`.
    """


_CANCEL_EVENTS: dict[str, threading.Event] = {}
_CANCEL_LOCK = threading.Lock()
_ACTIVE_WORKER: threading.Thread | None = None
_ACTIVE_WORKER_LOCK = threading.Lock()


def _register_cancel(req_id: str) -> threading.Event:
    event = threading.Event()
    with _CANCEL_LOCK:
        _CANCEL_EVENTS[req_id] = event
    return event


def _pop_cancel(req_id: str) -> None:
    with _CANCEL_LOCK:
        _CANCEL_EVENTS.pop(req_id, None)


def _request_cancel(req_id: str) -> bool:
    with _CANCEL_LOCK:
        event = _CANCEL_EVENTS.get(req_id)
    if event is None:
        return False
    event.set()
    return True


def _raise_if_canceled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCanceled()


def _torch_dtype_from_precision(value: str | None) -> torch.dtype:
    normalized = str(value or "fp32").lower().strip()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


class ModelCache:
    """Caches the loaded model / adapters.

    load_model_paths() runs on the first incoming task; if the paths don't
    change it's reused afterward; adapters are only re-injected when
    lora_configs changes (commit 9 simplification: re-injects every time,
    cost ~1-2s/LoRA, negligible compared to a 30s+ model load; to be
    optimized in a later commit).
    """

    def __init__(self) -> None:
        self.family_id: Optional[str] = None
        self.family: Any = None
        self.transformer_path: Optional[str] = None
        self.vae_path: Optional[str] = None
        self.text_encoder_path: Optional[str] = None
        self.t5_tokenizer_path: Optional[str] = None
        self.attention_backend: Optional[str] = None
        self.mixed_precision: Optional[str] = None
        self.vae_precision: Optional[str] = None
        self.text_encoder_backend: Optional[str] = None
        self.t5_tokenizer_backend: Optional[str] = None
        self.ram_guard: bool = False
        #: Identity key for the TE-first stack (family_id, text_encoder_path)
        #: -- ensure_text_ready reuses/rebuilds based on this; _load uses it
        #: to avoid a duplicate load
        self._text_ready_key: Optional[tuple] = None
        self.device: Optional[str] = None
        self.dtype: Any = None
        self.lora_dtype: Any = torch.float32
        self.model: Any = None
        self.vae: Any = None
        # Family-opaque text stack (a (qwen_model, qwen_tok, t5_tok) triple
        # for anima, Krea2TextStack for krea2) -- only consumed via
        # family.sample_image, the daemon never unpacks it
        self.text_stack: Any = None
        # adapters must keep a reference, or the forward hook stops working (lycoris closure)
        self.adapters: list[Any] = []
        self.last_lora_specs: list[LoRASpec] = []
        self.last_lora_metas: list[LoRAMeta] = []
        # latent2rgb linear projection used for intermediate-step previews
        # (see _decode_latent2rgb_preview) -- no external model / no
        # download involved, so CACHE holds no preview-decoder state at all.

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def ensure_text_ready(self, cfg: dict[str, Any]) -> None:
        """krea2's two-stage load, stage one: get the TE stack ready (without touching the DiT).

        TE-first orchestration (task-driven staging, the inference-side
        counterpart of the two-stage training scheme): at the start of a
        task, get the text stack ready first -> precache-encode -> release
        it entirely -> only then load the 13GB DiT -- at any given moment
        only one large model is on the GPU (measured: during the precache
        stage all three staying resident peaks at 24.1GB -> staggering them
        drops it to ~15GB, sparing a 16GB card from having to make room).
        A TE parameter change only rebuilds the text stack (invalidating
        the online LRU along with it), no longer triggering a full reload
        of everything. A no-op for the anima family (the TE-resident
        semantics go through the full reload in _load).
        """
        cfg = force_comfy_parity_runtime_config(
            cfg, force_exact_ksampler_backend=False,
        )
        family_id = str(cfg.get("model_family") or "anima")
        if family_id != "krea2":
            return
        repo_root = _T.find_diffusion_pipe_root()
        bases = [Path.cwd(), _THIS_DIR, repo_root]
        text_encoder_path = _T.resolve_path_best_effort(
            cfg["text_encoder_path"], bases,
        )
        key = (family_id, text_encoder_path)
        if self.text_stack is not None and self._text_ready_key == key:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _torch_dtype_from_precision(cfg.get("mixed_precision", "bf16"))
        family = _T.get_family(family_id)
        logger.info("loading text encoders (TE-first) %s", text_encoder_path)
        self.text_stack = family.load_text(
            text_encoder_path, device, dtype,
            purpose="generate",
            cache_enabled=False,
        )
        self._text_ready_key = key
        self.text_encoder_path = text_encoder_path

    def ensure_loaded(self, cfg: dict[str, Any]) -> None:
        """Decide whether a (re)load is needed based on cfg. A changed path or backend -> full reload."""
        cfg = force_comfy_parity_runtime_config(
            cfg,
            force_exact_ksampler_backend=False,
        )
        family_id = str(cfg.get("model_family") or "anima")
        backend = cfg.get("attention_backend", "none")
        precision = cfg.get("mixed_precision", "bf16")
        vae_precision = cfg.get("vae_precision", precision)
        text_encoder_backend = cfg.get("text_encoder_backend", "hf")
        t5_tokenizer_backend = cfg.get("t5_tokenizer_backend", "slow")
        transformer_path = cfg["transformer_path"]
        vae_path = cfg["vae_path"]
        text_encoder_path = cfg["text_encoder_path"]
        t5_tokenizer_path = cfg.get("t5_tokenizer_path", "")

        # Path resolution
        repo_root = _T.find_diffusion_pipe_root()
        bases = [Path.cwd(), _THIS_DIR, repo_root]
        transformer_path = _T.resolve_path_best_effort(transformer_path, bases)
        vae_path = _T.resolve_path_best_effort(vae_path, bases)
        text_encoder_path = _T.resolve_path_best_effort(text_encoder_path, bases)
        if t5_tokenizer_path:
            t5_tokenizer_path = _T.resolve_path_best_effort(t5_tokenizer_path, bases)

        self.ram_guard = bool(cfg.get("ram_guard", False))

        # Check whether a reload is needed (a family switch = a switch of the whole model stack, full reload)
        needs_reload = (
            not self.loaded
            or self.family_id != family_id
            or self.transformer_path != transformer_path
            or self.vae_path != vae_path
            or self.text_encoder_path != text_encoder_path
            or self.t5_tokenizer_path != t5_tokenizer_path
            or self.attention_backend != backend
            or self.mixed_precision != precision
            or self.vae_precision != vae_precision
            or self.text_encoder_backend != text_encoder_backend
            or self.t5_tokenizer_backend != t5_tokenizer_backend
        )

        if needs_reload:
            from training.sysmem import check_load_budget

            check_load_budget(
                self.ram_guard,
                weight_paths=[transformer_path, vae_path],
                stage="model load",
            )
            # keep_text: the TE-first stack just finished encoding (the LRU
            # is populated), a reload shouldn't clear it
            self.unload(keep_text=True)
            self._load(
                family_id=family_id,
                transformer_path=transformer_path,
                vae_path=vae_path,
                text_encoder_path=text_encoder_path,
                t5_tokenizer_path=t5_tokenizer_path,
                backend=backend,
                precision=precision,
                vae_precision=vae_precision,
                text_encoder_backend=text_encoder_backend,
                t5_tokenizer_backend=t5_tokenizer_backend,
            )
            _emit_evt("loaded")

    def _load(
        self,
        *,
        family_id: str,
        transformer_path: str,
        vae_path: str,
        text_encoder_path: str,
        t5_tokenizer_path: str,
        backend: str,
        precision: str,
        vae_precision: str,
        text_encoder_backend: str,
        t5_tokenizer_backend: str,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _torch_dtype_from_precision(precision)
        vae_dtype = _torch_dtype_from_precision(vae_precision)
        repo_root = _T.find_diffusion_pipe_root()
        use_flash = backend == "flash_attn"
        use_xformers = backend == "xformers"

        family = _T.get_family(family_id)
        logger.info("loading transformer [%s] %s", family_id, transformer_path)
        model = family.load_dit(
            transformer_path, device, dtype,
            attention_backend=("flash_attn" if use_flash else "none"), repo_root=repo_root,
            purpose="generate",
        )
        if use_xformers and not _T.enable_xformers(model):
            raise RuntimeError(
                "Exact ComfyUI KSampler parity is guaranteed only with xformers, "
                "but xformers could not be enabled"
            )

        logger.info("loading vae %s", vae_path)
        vae = family.load_vae(vae_path, device, vae_dtype)

        # Family-opaque text stack, never unpacked. When krea2's TE-first
        # stack (ensure_text_ready) is already ready and its identity
        # matches, reuse it -- preserving the precache LRU, avoiding a
        # duplicate load.
        if (
            family_id == "krea2"
            and self.text_stack is not None
            and self._text_ready_key == (family_id, text_encoder_path)
        ):
            text_stack = self.text_stack
        else:
            logger.info("loading text encoders %s", text_encoder_path)
            text_stack = family.load_text(
                text_encoder_path, device, dtype,
                t5_tokenizer_path=t5_tokenizer_path or None,
                comfy_qwen=text_encoder_backend == "comfy_qwen3",
                t5_fast=t5_tokenizer_backend == "fast",
                purpose="generate",
                cache_enabled=False,
            )

        self.family_id = family_id
        self.family = family
        self.model = model
        self.vae = vae
        self.text_stack = text_stack
        self.transformer_path = transformer_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.t5_tokenizer_path = t5_tokenizer_path
        self.attention_backend = backend
        self.mixed_precision = precision
        self.vae_precision = vae_precision
        self.text_encoder_backend = text_encoder_backend
        self.t5_tokenizer_backend = t5_tokenizer_backend
        self.device = device
        self.dtype = dtype
        self.lora_dtype = torch.float32
        self.adapters = []
        self.last_lora_specs = []
        self.last_lora_metas = []
        # Large weights are loaded now: return the mmap'd file cache pages
        # (DiT 13-26GB + VAE) to the system, preventing a paging stall on
        # machines with tight physical memory (TE gets trimmed again by
        # ensure_model after its own lazy load)
        from training.sysmem import trim_working_set

        trim_working_set()

    def apply_loras(self, lora_configs: list[dict[str, Any]]) -> list[Any]:
        """Inject adapters based on lora_configs; when switching between checkpoints of the same structure, only hot-swaps the weights."""
        self._move_runtime_to_device()

        specs = [
            LoRASpec(path=str(lc.get("path", "")), scale=float(lc.get("scale", 1.0)))
            for lc in lora_configs
        ]
        if specs == self.last_lora_specs and self.adapters:
            return self.adapters

        current_metas: list[LoRAMeta] = []
        if specs:
            try:
                for spec in specs:
                    if not spec.path or not Path(spec.path).exists():
                        current_metas = []
                        break
                    current_metas.append(read_lora_meta(spec.path))
            except Exception:
                logger.exception("read LoRA metadata failed")
                current_metas = []

        can_hot_reload = (
            bool(self.adapters)
            and bool(self.last_lora_specs)
            and len(specs) == len(self.adapters) == len(self.last_lora_metas) == len(current_metas)
            and [_lora_topology(m) for m in current_metas]
            == [_lora_topology(m) for m in self.last_lora_metas]
            # An fp8 merge handle has no resident network weights to hot-swap,
            # must detach (restore the original fp8 weights) and re-merge
            and all(getattr(a, "supports_hot_reload", True) for a in self.adapters)
        )
        if can_hot_reload:
            try:
                for adapter, spec in zip(self.adapters, specs):
                    _reload_adapter_weights(adapter, spec, self.device, self.lora_dtype)
            except Exception:
                logger.exception("LoRA hot reload failed; reinjecting adapters")
            else:
                self.last_lora_specs = specs
                self.last_lora_metas = current_metas
                self.model.eval()
                return self.adapters

        all_detached = True
        for adapter in self.adapters:
            try:
                if not adapter.detach():
                    all_detached = False
            except Exception:
                logger.exception("adapter detach failed")
                all_detached = False
        self.adapters = []

        if not all_detached and self.last_lora_specs:
            logger.warning("detach failed, reloading model to ensure clean state")
            saved_paths = (
                self.transformer_path, self.vae_path,
                self.text_encoder_path, self.t5_tokenizer_path,
                self.attention_backend, self.mixed_precision,
                self.vae_precision, self.text_encoder_backend,
                self.t5_tokenizer_backend,
            )
            # family_id gets cleared inside unload(), so it must be saved
            # first -- forgetting this once caused a silent TypeError (a
            # required parameter of _load that this code path missed
            # updating when P4-4 was introduced)
            saved_family = self.family_id or "anima"
            self.unload()
            self._load(
                family_id=saved_family,
                transformer_path=saved_paths[0],
                vae_path=saved_paths[1],
                text_encoder_path=saved_paths[2],
                t5_tokenizer_path=saved_paths[3] or "",
                backend=saved_paths[4],
                precision=saved_paths[5],
                vae_precision=saved_paths[6] or saved_paths[5],
                text_encoder_backend=saved_paths[7] or "hf",
                t5_tokenizer_backend=saved_paths[8] or "slow",
            )
            _emit_evt("loaded")
        self.last_lora_specs = []
        self.last_lora_metas = []

        self.adapters = apply_loras(
            self.model, specs, self.device, self.lora_dtype,
            family_id=self.family_id or "anima",
        )
        self.last_lora_specs = specs
        self.last_lora_metas = current_metas
        self.model.eval()
        return self.adapters

    def _move_runtime_to_device(self) -> None:
        if not self.device:
            return
        _move_module_to_device(self.model, self.device)
        # anima's text stack is a (qwen_model, qwen_tok, t5_tok) triple --
        # the decode offload inside sampling moves the TE to CPU, this moves
        # it back. A stack that manages its own device (krea2's
        # Krea2TextStack) has no bare module members, so the loop is
        # naturally a no-op for it.
        if isinstance(self.text_stack, (tuple, list)):
            for member in self.text_stack:
                if isinstance(member, torch.nn.Module):
                    _move_module_to_device(member, self.device)
        for adapter in self.adapters:
            _move_adapter_to_device(adapter, self.device, self.lora_dtype)

    def unload(self, *, keep_text: bool = False) -> None:
        """Unload the model stack. ``keep_text``: keep the TE-first stack
        (including the precache LRU) -- used by ensure_loaded's reload path,
        to avoid wiping out results that were just encoded."""
        if not keep_text:
            self.text_stack = None
            self._text_ready_key = None
        if not self.loaded:
            # Model not loaded != VRAM clean: after an OOM partway through
            # loading, self.model can still be None, but the exception
            # traceback's reference cycle pins a half-loaded state_dict
            # (its refcount never drops) -- when the manual "free cache"
            # action hits this branch it must sweep unconditionally, or the
            # button spins for nothing (upstream #499, measured on real
            # hardware: 20GB stayed completely stuck).
            _reclaim_cuda_leftovers()
            return
        logger.info("unloading model")
        self.model = None
        self.vae = None
        self.family_id = None
        self.family = None
        self.adapters = []
        self.last_lora_specs = []
        self.last_lora_metas = []
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    # The cuBLAS workspace is a C++-level resident
                    # allocation (invisible to Python's gc, only ~10MB), but
                    # it pins down the entire allocator segment it lives in
                    # -- measured: after fp8 sampling, 8GB+ reserved
                    # couldn't be released by empty_cache (reproduced in
                    # tmp/diag_vram_leak.py). Same handling as ComfyUI's
                    # soft_empty_cache; internal API, failure can be
                    # ignored (the next load will just reuse the cache).
                    torch._C._cuda_clearCublasWorkspaces()
                except Exception:
                    pass
                torch.cuda.empty_cache()
        except Exception:
            pass


def _reclaim_cuda_leftovers() -> None:
    """Reclaim VRAM / pinned-memory leftovers that are no longer referenced
    by business logic but still hold onto memory (upstream #499).

    An exception's traceback (especially from an OOM partway through
    loading / sampling) references its frame and vice versa, pinning large
    tensors sitting in the frame's locals; only gc can free that, and then
    empty_cache returns the freed blocks to the driver. The pinned host
    cache is explicitly returned the same way (a no-op if there's no cache).
    The model itself is unaffected -- only "orphaned" parts are cleared.
    """
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            try:
                # cuBLAS workspace pins down the segment it lives in (see
                # the comment inside unload); it must be cleared before
                # empty_cache can return the whole segment
                torch._C._cuda_clearCublasWorkspaces()
            except Exception:
                pass
            torch.cuda.empty_cache()
        try:
            from training.block_swap import release_pinned_host_cache

            release_pinned_host_cache()
        except Exception:
            pass
    except Exception:
        logger.exception("CUDA leftovers reclaim failed")


CACHE = ModelCache()


# ---------------------------------------------------------------------------
# Generate implementation (reuses anima_generate.py's loop logic)
# ---------------------------------------------------------------------------


def _precache_prompts_and_release(
    prompts: list[str],
    vram_policy: str,
    phase_callback: Any = None,
) -> None:
    """krea2's task-level prompt precache + full TE release (the inference-side counterpart of the two-stage training scheme).

    The prompt set for an XY / multi-prompt generate task is closed before
    the task starts -- encode every prompt into the online LRU first, then
    **fully release** the TE (release, not offload: doesn't leave behind a
    ~5GB CPU copy; conditioning is already in the LRU, and the next task's
    LRU miss just re-loads from disk in ~3s). Together with
    ensure_text_ready this forms the TE-first orchestration: the first task
    encodes before the DiT loads, so at any moment only one large model is
    on GPU.

    - Not released under the performance tier (the user explicitly asked
      for everything resident, zero transfers).
    - The anima text stack (a tuple) has no such API, safely skipped.
    - A precache failure doesn't block the task: the per-cell lazy encode
      path is the fallback.
    """
    precache = getattr(CACHE.text_stack, "precache_online_prompts", None)
    if not callable(precache):
        return
    # The TE is lazily loaded here (fp8 5GB / bf16 8.9GB read from an mmap'd
    # file) -- budget RAM/VRAM based on the TE file size. Must be outside
    # the fallback try: a guardrail error must abort the task, not get
    # swallowed by "fall back to per-cell encoding" and still load the TE,
    # hanging. When the TE is already resident on the card, zero budget
    # passes straight through.
    from training.sysmem import check_load_budget

    te_paths = (
        [] if getattr(CACHE.text_stack, "is_model_loaded", False)
        else [CACHE.text_encoder_path]
    )
    check_load_budget(CACHE.ram_guard, weight_paths=te_paths, stage="text encoder load")
    try:
        if phase_callback is not None:
            phase_callback("clip")
        encoded = precache([str(p) for p in prompts])
    except Exception:
        logger.exception("prompt precache failed; falling back to per-cell lazy encoding")
        return
    if vram_policy != "performance":
        release = getattr(CACHE.text_stack, "release_model", None)
        if callable(release):
            release()
    if encoded:
        logger.info("krea2 precached %d prompts; TE released, zero usage during sampling", encoded)


def _set_lora_multiplier(adapter: Any, scale: float) -> None:
    if adapter.network is None:
        # An fp8 merge handle (network=None): the scale is already baked
        # into the weights, safe to skip setting a per-cell value (fp8's
        # lora_scale axis takes effect via _cell_lora_configs ->
        # CACHE.apply_loras re-merging)
        return
    adapter.network.multiplier = float(scale)
    for lora in getattr(adapter.network, "loras", []):
        if hasattr(lora, "multiplier"):
            lora.multiplier = float(scale)


def _swap_ckpt_for_axis(spec: dict[str, Any], val: Any,
                        lora_configs: list[dict[str, Any]]) -> None:
    """When axis=lora_ckpt, change lora_configs[lora_index].path to val."""
    if spec.get("axis") != "lora_ckpt":
        return
    idx = int(spec.get("lora_index") or 0)
    if 0 <= idx < len(lora_configs):
        lora_configs[idx]["path"] = str(val)


def _cell_lora_configs(
    x_spec: dict[str, Any],
    y_spec: dict[str, Any] | None,
    xv: Any,
    yv: Any,
    base_paths: list[str],
    base_scales: list[float],
    *,
    fp8_scale_axes: bool,
) -> list[dict[str, Any]] | None:
    """Assemble the lora_configs that this cell needs to re-mount; returns None if no re-mount is needed.

    - lora_ckpt axis: swaps a single path by lora_index (applies to both bf16/fp8).
    - lora_scale axis only takes effect here for an fp8 base model
      (``fp8_scale_axes=True``): a merge has no resident network, so
      changing the strength requires detach-and-restore + re-merge --
      CACHE.apply_loras automatically takes this path for a handle with
      supports_hot_reload=False (same as the lora_ckpt axis); cells with
      identical specs get deduped and skipped at zero cost. bf16 uses
      _apply_axis's multiplier hot-swap instead, and doesn't go through here.
      Global-axis semantics: every entry gets scale=the cell's value; when
      both x/y are scale axes, whichever is written second wins, matching
      _apply_axis's x->y call order.
    """
    x_axis = x_spec.get("axis")
    y_axis = y_spec.get("axis") if y_spec is not None else None
    needs = x_axis == "lora_ckpt" or y_axis == "lora_ckpt" or (
        fp8_scale_axes and "lora_scale" in (x_axis, y_axis)
    )
    if not needs:
        return None
    configs = [
        {"path": p, "scale": s} for p, s in zip(base_paths, base_scales)
    ]
    _swap_ckpt_for_axis(x_spec, xv, configs)
    if y_spec is not None and yv is not None:
        _swap_ckpt_for_axis(y_spec, yv, configs)
    if fp8_scale_axes:
        if x_axis == "lora_scale":
            for lc in configs:
                lc["scale"] = float(xv)
        if y_axis == "lora_scale" and yv is not None:
            for lc in configs:
                lc["scale"] = float(yv)
    return configs


def _apply_axis(
    axis: dict[str, Any],
    value: Any,
    *,
    cur_steps: int,
    cur_cfg_scale: float,
    adapters: list[Any],
) -> tuple[int, float]:
    """Handle a plain numeric/scale axis. lora_ckpt is not handled here (it
    needs a re-injection, handled separately by _run_xy via the
    CACHE.apply_loras path).

    lora_scale is a **global axis** -- it sets every adapter's multiplier to
    the cell's value; the original relative weighting between different
    LoRAs disappears, but the weight axis's semantics in the UI are "sweep
    an absolute value", not "sweep one LoRA's relative value".
    """
    axis_type = axis["axis"]
    if axis_type == "steps":
        cur_steps = int(value)
    elif axis_type == "cfg_scale":
        cur_cfg_scale = float(value)
    elif axis_type == "lora_scale":
        for ad in adapters:
            _set_lora_multiplier(ad, float(value))
    return cur_steps, cur_cfg_scale


def _setup_monitor(cfg: dict[str, Any]) -> Any:
    """Initialize train_monitor (one independent monitor_state.json per task).

    The frontend gets samples + xy metadata via SSE monitor_progress; the
    image bytes themselves go through the protocol's image_done event into
    the server's in-memory cache (since commit 10). The sample_path field
    holds a virtual path (the frontend only does split+pop to get the
    filename to build the /api/generate/{tid}/sample/{fn} URL); no such
    file actually exists on disk.
    """
    msf = cfg.get("__monitor_state_file")
    if not msf:
        return None
    try:
        from train_monitor import reset_monitor, set_state_file, update_monitor
        # Important: when the daemon process is reused across tasks,
        # MONITOR_STATE persists, so it must be cleared. Otherwise the
        # previous task's samples would leak into the new task's
        # monitor_state.json, and the frontend building a URL from
        # currentTask.id with the old filename would get a 404 broken image.
        reset_monitor()
        set_state_file(msf)
        update_monitor(config={
            "type": "generate",
            "prompts": len(cfg.get("prompts") or []),
            "count": int(cfg.get("count", 1)),
            "steps": int(cfg.get("steps", 25)),
            "cfg_scale": float(cfg.get("cfg_scale", 4.0)),
        })
        return update_monitor
    except Exception as e:
        logger.warning("monitor init failed: %s", e)
        return None


def _encode_png(img: Any) -> tuple[str, int]:
    """PIL.Image -> PNG bytes -> base64 string. Returns (b64_str, raw_byte_size)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), len(raw)


def _encode_jpeg(img: Any, quality: int = 80) -> tuple[str, int]:
    """Intermediate-step preview encoding: JPEG at 80% by default, ~5x smaller than PNG. Returns (b64_str, byte_size)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), len(raw)


def _build_preview_callback(
    req_id: str,
    every_n: int,
    cancel_event: threading.Event | None = None,
) -> Any:
    """Pushes a preview_step event every step; attaches a latent2rgb preview image_b64 when the throttle hits.

    User feedback: the progress bar should always be visible ("what's it
    doing right now, which step"), the preview image is on-demand. This
    callback splits into two paths:
      - always emit preview_step { step, total } -- for the frontend progress bar
      - when every_n>0 and the step hits (including the last step) -> latent2rgb decode + attach image_b64
    The callback runs synchronously on the daemon's main thread; the empty
    path is ~microsecond-scale, latent2rgb (a pure linear projection, no NN
    forward pass) + JPEG encoding is ~1-2ms.

    Once cancel_event is injected, it's checked every step, so cancellation
    latency drops from "a whole image" to "one step".
    """
    def _cb(step: int, total: int, latent: Any) -> None:
        _raise_if_canceled(cancel_event)
        # Whether to attach a preview image: preview_every_n_steps>0 + throttle hit (including the last step).
        with_image = False
        b64: Optional[str] = None
        byte_size = 0
        if every_n > 0 and (step % every_n == 0 or step == total - 1):
            img = _decode_latent2rgb_preview(latent)
            if img is not None:
                b64, byte_size = _encode_jpeg(img, quality=80)
                with_image = True
        payload: dict[str, Any] = {"step": step + 1, "total": total}
        if with_image:
            payload["image_b64"] = b64
            payload["byte_size"] = byte_size
        _emit_for(req_id, "preview_step", **payload)
    return _cb


# The latent->RGB linear projection coefficients are taken from the
# **currently loaded family's spec** (folded into ModelSpec in D17, single
# source of truth in families/latent_spaces.py). Anima and Krea2 share the
# same Wan2.1 16-channel latent space (Qwen-Image VAE: ComfyUI's
# supported_models.py `QwenImage.latent_format = latent_formats.Wan21`);
# when a family with a different latent space is added in the future, the
# preview will automatically follow. Previously mistakenly used TAEFlux (the
# Flux decoder) -> inverted colors, now deprecated.
from training.families.latent_spaces import WAN21_F8C16 as _WAN21_F8C16  # noqa: E402

# Preview upscale target (longest-edge pixels). The latent is at 1/8
# resolution (1024^2 -> 128^2), latent2rgb produces 128^2 directly; upscaling
# to 512 keeps it from looking too blurry when the frontend fills its area,
# and the JPEG is still tiny.
_PREVIEW_TARGET_PX = 512


def _preview_latent_spec():
    """The currently loaded family's LatentSpec; falls back to the shared Wan21 space when nothing is loaded (unit tests / early startup)."""
    family = getattr(CACHE, "family", None)
    return family.spec.latent if family is not None else _WAN21_F8C16


def _decode_latent2rgb_preview(latent: Any) -> Optional[Any]:
    """latent -> Wan2.1 latent2rgb linear projection -> PIL.Image. Returns None on failure (preview must never block).

    Anima latent shape: [B, 16, F=1, H, W]. Matches ComfyUI's Latent2RGBPreviewer:
      x0 = latent[0, :, 0]                          # [16, H, W]
      rgb[h,w,r] = sum_c x0[c,h,w]*factors[c,r] + bias[r]
      img = ((rgb + 1) / 2).clamp(0,1) * 255        # [-1,1] -> [0,255]
    """
    try:
        import numpy as np
        from PIL import Image
        latent_spec = _preview_latent_spec()
        with torch.no_grad():
            x = latent[0, :, 0].float()  # [16, H, W]
            factors = torch.tensor(
                [list(r) for r in latent_spec.rgb_factors],
                device=x.device, dtype=x.dtype,
            )  # [16, 3]
            bias = torch.tensor(
                list(latent_spec.rgb_bias), device=x.device, dtype=x.dtype,
            )  # [3]
            rgb = torch.einsum("chw,cr->hwr", x, factors) + bias  # [H, W, 3]
            rgb = ((rgb + 1.0) / 2.0).clamp(0.0, 1.0)
            arr = (rgb.cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
            img = Image.fromarray(arr)
            # Upscale to _PREVIEW_TARGET_PX on the longest edge (keeping
            # aspect ratio); the frontend then fills the result area
            w, h = img.size
            scale = _PREVIEW_TARGET_PX / max(w, h)
            if scale > 1.0:
                img = img.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    Image.BILINEAR,
                )
            return img
    except Exception:
        logger.exception("latent2rgb preview decode failed")
        return None


def _virtual_path(task_id: int, filename: str) -> str:
    """The frontend only does split+pop to get the filename, so hand it a string that looks like an absolute path."""
    return f"/anima_gen_{task_id}/{filename}"


def _run_generate(
    req_id: str,
    task_id: int,
    cfg: dict[str, Any],
    output_dir: Path,
    cancel_event: threading.Event | None = None,
) -> None:
    """Run one complete generate (optionally including XY).

    Since commit 10: PNG bytes are pushed to stdout as base64 (the
    image_done event) -> the server-side InferenceDaemon puts them into
    generate_cache; output_dir is no longer written to disk (the parameter
    is kept for the anima_generate.py CLI usage's fallback path).

    monitor_state.json is still written (for compatibility with the
    frontend's sample_path SSE pipeline), but sample_path is a virtual path
    with no corresponding file on disk -- the frontend only does split+pop
    on it to get the filename, used to build the
    /api/generate/{tid}/sample/{fn} URL.
    """
    cfg = force_comfy_parity_runtime_config(
        cfg,
        force_exact_ksampler_backend=False,
    )
    update_monitor = _setup_monitor(cfg)

    _raise_if_canceled(cancel_event)

    def _phase_cb(name: str) -> None:
        _emit_for(req_id, "phase", name=name)

    prompts: list[str] = cfg.get("prompts") or [
        "newest, safe, 1girl, masterpiece, best quality"
    ]
    negative_prompt: str = cfg.get("negative_prompt", "")
    width: int = int(cfg.get("width", 1024))
    height: int = int(cfg.get("height", 1024))
    steps: int = int(cfg.get("steps", 25))
    cfg_scale: float = float(cfg.get("cfg_scale", 4.0))
    sampler_name: str = cfg.get("sampler_name", "er_sde")
    scheduler: str = cfg.get("scheduler", "simple")
    count: int = max(1, int(cfg.get("count", 1)))
    base_seed: int = int(cfg.get("seed", 0))
    # Distilled inference base model (Krea2 Turbo): studio injects this
    # after detecting it via the catalog variant purpose; the anima family
    # accepts and ignores it
    distilled: bool = bool(cfg.get("distilled", False))

    # phase reporting: model/LoRA loading stage (clip/sample/vae are
    # reported from inside sample_image) -> the progress bar covers the
    # whole pipeline
    _emit_for(req_id, "phase", name="load")
    # TE-first orchestration (krea2): the prompt set is closed -- get the TE
    # stack ready before the DiT loads, encode every prompt, then fully
    # release the TE. At any moment only one large model is on GPU
    # (during the precache stage all three staying resident peaks at
    # 24.1GB -> ~15GB; spares a 16GB card from having to make room). Both
    # steps are no-ops for the anima family.
    CACHE.ensure_text_ready(cfg)
    _precache_prompts_and_release(
        [*prompts, negative_prompt],
        str(cfg.get("vram_policy") or "auto"),
        _phase_cb,
    )
    CACHE.ensure_loaded(cfg)
    adapters = CACHE.apply_loras(cfg.get("lora_configs", []))

    # Progress push: always builds a callback that pushes preview_step
    # (including step/total); when preview_every_n_steps>0, attaches an
    # intermediate preview image_b64 (commit 14).
    preview_every = int(cfg.get("preview_every_n_steps", 0) or 0)
    preview_callback = _build_preview_callback(req_id, preview_every, cancel_event)

    xy_matrix = cfg.get("xy_matrix")
    if xy_matrix is not None:
        _run_xy(
            req_id=req_id, task_id=task_id, cfg=cfg, output_dir=output_dir,
            xy_matrix=xy_matrix, adapters=adapters,
            prompt=prompts[0], negative_prompt=negative_prompt,
            base_seed=base_seed, base_steps=steps, base_cfg_scale=cfg_scale,
            base_sampler=sampler_name, scheduler=scheduler,
            height=height, width=width,
            update_monitor=update_monitor,
            preview_callback=preview_callback,
            phase_callback=_phase_cb,
            cancel_event=cancel_event,
        )
        return

    total = count * len(prompts)
    _emit_for(req_id, "started", task_id=task_id, total=total)

    img_idx = 0
    image_done_count = 0
    image_errors: list[str] = []
    for pi, prompt in enumerate(prompts):
        for ci in range(count):
            _raise_if_canceled(cancel_event)
            seed = (
                (base_seed + img_idx) if base_seed != 0
                else random.randint(0, 2**31 - 1)
            )
            torch.manual_seed(seed)
            random.seed(seed)
            _emit_for(
                req_id, "image_started",
                batch_idx=img_idx, batch_total=total, total_steps=steps,
            )
            try:
                img = CACHE.family.sample_image(
                    CACHE.model, CACHE.vae, CACHE.text_stack,
                    prompt,
                    height=height,
                    width=width,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    negative_prompt=negative_prompt,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    distilled=distilled,
                    device=CACHE.device,
                    dtype=CACHE.dtype,
                    step_callback=preview_callback,
                    phase_callback=_phase_cb,
                    seed=seed,
                    vram_policy=str(cfg.get("vram_policy") or "auto"),
                )
                CACHE._move_runtime_to_device()
                fname = f"gen_{img_idx:04d}_p{pi}_c{ci}_s{seed}.png"
                vpath = _virtual_path(task_id, fname)
                b64, byte_size = _encode_png(img)
                if update_monitor:
                    update_monitor(sample_path=vpath, step=img_idx + 1)
                _emit_for(
                    req_id, "image_done",
                    filename=fname, path=vpath,
                    step=img_idx + 1, total=total,
                    image_b64=b64, byte_size=byte_size,
                )
                image_done_count += 1
            except GenerationCanceled:
                raise
            except Exception as e:
                logger.exception("generate failed")
                image_errors.append(str(e))
                _emit_for(req_id, "image_error", step=img_idx + 1, message=str(e))
            img_idx += 1

    if image_done_count == 0 and image_errors:
        raise RuntimeError(f"all generated images failed: {image_errors[-1]}")


def _run_xy(
    *,
    req_id: str,
    task_id: int,
    cfg: dict[str, Any],
    output_dir: Path,
    xy_matrix: dict[str, Any],
    adapters: list[Any],
    prompt: str,
    negative_prompt: str,
    base_seed: int,
    base_steps: int,
    base_cfg_scale: float,
    base_sampler: str,
    scheduler: str,
    height: int,
    width: int,
    update_monitor: Any,
    preview_callback: Any = None,
    phase_callback: Any = None,
    cancel_event: threading.Event | None = None,
) -> None:
    x_spec = xy_matrix["x"]
    y_spec = xy_matrix.get("y")
    x_values = x_spec["values"]
    y_values = y_spec["values"] if y_spec else [None]
    distilled: bool = bool(cfg.get("distilled", False))

    # An fp8 base model's LoRA is merged into the weights (no resident
    # network), so the lora_scale axis can't hot-swap via multiplier --
    # each cell goes through detach-and-restore + re-merge
    # (_cell_lora_configs assembles the config -> CACHE.apply_loras, same
    # as the lora_ckpt axis). Every distinct scale value triggers one
    # full-model re-merge: when scale is on the Y axis, each row only
    # merges once (specs deduped); on the X axis, once per cell.
    fp8_model = False
    if CACHE.model is not None:
        from training.families.krea2.quant_fp8 import model_has_fp8_layers

        fp8_model = model_has_fp8_layers(CACHE.model)

    if base_seed == 0:
        base_seed = random.randint(0, 2**31 - 1)
        logger.info("XY shared seed (cfg.seed=0 randomized): %d", base_seed)

    base_scales = [float(s.scale) for s in CACHE.last_lora_specs]
    base_lora_paths = [str(s.path) for s in CACHE.last_lora_specs]
    total = len(x_values) * len(y_values)
    _emit_for(req_id, "started", task_id=task_id, total=total)

    # XY has no prompt axis -- prompt/negative have already been uniformly
    # precached and released by _run_generate's TE-first orchestration; each
    # cell hits the LRU here, no duplicate work needed.

    img_idx = 0
    image_done_count = 0
    image_errors: list[str] = []
    for yi, yv in enumerate(y_values):
        for xi, xv in enumerate(x_values):
            _raise_if_canceled(cancel_event)
            # lora_ckpt swaps the file / (under fp8) lora_scale swaps the
            # strength: assemble this cell's lora_configs and call
            # CACHE.apply_loras -- detach-and-restore + re-mount (bf16
            # reinject / fp8 re-merge). base_paths/base_scales is a
            # snapshot taken outside the loop, so cells never contaminate
            # each other.
            lora_configs = _cell_lora_configs(
                x_spec, y_spec, xv, yv, base_lora_paths, base_scales,
                fp8_scale_axes=fp8_model,
            )
            if lora_configs is not None:
                adapters = CACHE.apply_loras(lora_configs)

            for i, s in enumerate(base_scales):
                if i < len(adapters):
                    _set_lora_multiplier(adapters[i], s)

            cur_steps = base_steps
            cur_cfg_scale = base_cfg_scale

            cur_steps, cur_cfg_scale = _apply_axis(
                x_spec, xv,
                cur_steps=cur_steps, cur_cfg_scale=cur_cfg_scale,
                adapters=adapters,
            )
            if y_spec is not None and yv is not None:
                cur_steps, cur_cfg_scale = _apply_axis(
                    y_spec, yv,
                    cur_steps=cur_steps, cur_cfg_scale=cur_cfg_scale,
                    adapters=adapters,
                )

            cur_seed = base_seed
            torch.manual_seed(cur_seed)
            random.seed(cur_seed)

            _emit_for(
                req_id, "image_started",
                batch_idx=img_idx, batch_total=total, total_steps=cur_steps,
            )
            try:
                img = CACHE.family.sample_image(
                    CACHE.model, CACHE.vae, CACHE.text_stack,
                    prompt,
                    height=height,
                    width=width,
                    steps=cur_steps,
                    step_callback=preview_callback,
                    phase_callback=phase_callback,
                    cfg_scale=cur_cfg_scale,
                    negative_prompt=negative_prompt,
                    sampler_name=base_sampler,
                    scheduler=scheduler,
                    distilled=distilled,
                    device=CACHE.device,
                    dtype=CACHE.dtype,
                    seed=cur_seed,
                    vram_policy=str(cfg.get("vram_policy") or "auto"),
                )
                CACHE._move_runtime_to_device()
                fname = f"xy_x{xi:02d}_y{yi:02d}_s{cur_seed}.png"
                vpath = _virtual_path(task_id, fname)
                b64, byte_size = _encode_png(img)
                if update_monitor:
                    update_monitor(
                        sample_path=vpath,
                        step=img_idx + 1,
                        xy={"xi": xi, "yi": yi, "xv": xv, "yv": yv},
                    )
                _emit_for(
                    req_id, "image_done",
                    filename=fname, path=vpath,
                    step=img_idx + 1, total=total,
                    xy={"xi": xi, "yi": yi, "xv": xv, "yv": yv},
                    image_b64=b64, byte_size=byte_size,
                )
                image_done_count += 1
            except GenerationCanceled:
                raise
            except Exception as e:
                logger.exception("XY [%d,%d] failed", xi, yi)
                image_errors.append(str(e))
                _emit_for(
                    req_id, "image_error",
                    step=img_idx + 1, message=str(e),
                    xy={"xi": xi, "yi": yi, "xv": xv, "yv": yv},
                )
            img_idx += 1

    if image_done_count == 0 and image_errors:
        raise RuntimeError(f"all generated images failed: {image_errors[-1]}")


def _run_generate_worker(
    req_id: str,
    task_id: int,
    cfg: dict[str, Any],
    output_dir: Path,
    cancel_event: threading.Event,
) -> None:
    failed = False
    try:
        _run_generate(req_id, task_id, cfg, output_dir, cancel_event)
        _emit_for(req_id, "done", task_id=task_id)
    except GenerationCanceled:
        logger.info("generate canceled: task_id=%s", task_id)
        _emit_for(req_id, "canceled", task_id=task_id)
    except Exception as e:
        logger.exception("generate failed")
        _emit_for(req_id, "error", task_id=task_id, message=str(e))
        failed = True
    finally:
        if failed:
            # Must sweep only after the except block ends (the exception
            # object is implicitly deleted), or the frame locals pinned by
            # the traceback (e.g. a half-loaded state_dict from an OOM)
            # can't be freed
            _reclaim_cuda_leftovers()
        _pop_cancel(req_id)
        with _ACTIVE_WORKER_LOCK:
            global _ACTIVE_WORKER
            _ACTIVE_WORKER = None


def _start_generate_worker(req_id: str, task_id: int, cfg: dict[str, Any], output_dir: Path) -> bool:
    global _ACTIVE_WORKER
    with _ACTIVE_WORKER_LOCK:
        if _ACTIVE_WORKER is not None and _ACTIVE_WORKER.is_alive():
            return False
        cancel_event = _register_cancel(req_id)
        worker = threading.Thread(
            target=_run_generate_worker,
            args=(req_id, task_id, cfg, output_dir, cancel_event),
            daemon=False,
            name=f"generate-{task_id}",
        )
        _ACTIVE_WORKER = worker
        worker.start()
        return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _handle_message(msg: dict[str, Any]) -> None:
    action = msg.get("action")
    req_id = msg.get("id", "")

    if action == "ping":
        _emit_for(req_id, "pong")
        return

    if action == "unload":
        CACHE.unload()
        _emit_evt("unloaded")
        return

    if action == "cancel":
        if _request_cancel(str(msg.get("target_id") or req_id)):
            _emit_for(req_id, "cancel_ack")
        else:
            _emit_for(req_id, "cancel_missed")
        return

    if action == "generate":
        task_id = int(msg.get("task_id", 0))
        cfg = msg.get("config") or {}
        output_dir = Path(msg.get("output_dir") or ".")
        if not _start_generate_worker(req_id, task_id, cfg, output_dir):
            _emit_for(req_id, "error", task_id=task_id, message="daemon is already running a task")
        return

    logger.warning("unknown action: %r", action)
    _emit_for(req_id, "error", message=f"unknown action: {action!r}")


def main() -> int:
    _emit_evt("ready")
    logger.info("anima daemon ready, waiting for stdin commands")
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("non-JSON stdin line: %r (%s)", line[:200], e)
                continue
            try:
                _handle_message(msg)
            except Exception:
                logger.exception("message handler crashed")
    except KeyboardInterrupt:
        pass
    finally:
        with _ACTIVE_WORKER_LOCK:
            worker = _ACTIVE_WORKER
        if worker is not None:
            worker.join()
        CACHE.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
