"""Shared infrastructure for local ONNX taggers.

WD14 and CLTagger both go through the same local-onnxruntime batch / CUDA fallback /
preprocess thread-pool logic; the differences are in model file layout, tag metadata format, the preprocess flow,
and postprocess threshold rules. The base class seals off what's shared; subclasses fill in the specifics.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image

from ..runtime import onnxruntime as onnxruntime_setup
from ..models.paths import safe_dir_name  # noqa: F401  (re-exported for shim consumers)
from .base import ProgressFn, TagResult

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def silenced_fd_stderr():
    """Temporarily redirects fd 2 to devnull, swallowing output that C++ libraries write directly to fd 2.

    When onnxruntime's CUDA dlopen fails (missing cublasLt64_12.dll / libcurand etc.), it
    dumps colored ANSI (plus extra NUL bytes on Windows) straight to fd 2, bypassing Python's
    sys.stderr -- we've already caught InferenceSession's Python exception and surfaced it into
    the Settings UI's cuda_load_error, so these raw bytes would only pollute the worker log.
    """
    try:
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    saved = os.dup(2)
    devnull = open(os.devnull, "wb")
    try:
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        devnull.close()


class OnnxTaggerBase:
    """Shared logic for local ONNX taggers (CUDA fallback / batch decision / preprocess thread pool).

    Subclasses need to implement:
    - `prepare()`: parses the model file -> calls `_create_session(model_path)` -> loads metadata
    - `_preprocess(img) -> ndarray`: preprocesses a single image
    - `_postprocess_one(logits) -> (tags, raw_scores)`: postprocesses a single result
    - `_get_batch_size_cfg() -> int`: reads batch_size from secrets

    Subclasses must set `name`, used for logging / thread naming.
    """
    name: str = "onnx_base"
    requires_service = False

    def __init__(self) -> None:
        self._session = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[list[str]] = None
        # Set by _fallback_to_cpu_session() and rebuilt on it when CUDA fails during inference.
        # Set by _create_session once the session is created successfully.
        self._model_path: Optional[Path] = None

    # -------------------- subclass implementation --------------------

    def prepare(self) -> None:
        raise NotImplementedError

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        raise NotImplementedError

    def _postprocess_one(
        self, logits: np.ndarray
    ) -> tuple[list[str], dict[str, float]]:
        raise NotImplementedError

    def _get_batch_size_cfg(self) -> int:
        raise NotImplementedError

    # -------------------- session creation (including GPU EP fallback) --------------------

    def _create_session(self, model_path: Path) -> None:
        """Creates the onnxruntime InferenceSession, auto-falling back to CPU if the GPU EP fails.

        GPU EP selection priority: CUDA > DirectML > CPU. If onnxruntime-gpu is installed, uses CUDA;
        if onnxruntime-directml is installed, uses DirectML (Windows + DX12 backend, vendor-agnostic);
        the CPU package uses CPU.

        Side effects: sets `_session` / `_input_name` / `_output_names` / `_model_path`,
        and stashes any CUDA error for the Settings UI (a successful path also clears the old error record). A silent
        DirectML downgrade is only logged, without dedicated UI display (DX12 compat issues are extremely rare).
        """
        self._model_path = model_path
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - install hint
            raise RuntimeError(
                "onnxruntime is not installed; install onnxruntime / onnxruntime-gpu / onnxruntime-directml"
            ) from exc

        avail = ort.get_available_providers()
        gpu_ep: Optional[str] = None
        if "CUDAExecutionProvider" in avail:
            gpu_ep = "CUDAExecutionProvider"
        elif "DmlExecutionProvider" in avail:
            gpu_ep = "DmlExecutionProvider"
        providers = [gpu_ep, "CPUExecutionProvider"] if gpu_ep else ["CPUExecutionProvider"]

        # PP9.5 -- the CUDA EP's `get_available_providers()` reporting available doesn't mean it can actually dlopen.
        # When the system CUDA runtime is missing, this hangs during InferenceSession creation. fd-level stderr silently
        # swallows the polluting C-layer log; the Python exception already has the full reason. DirectML has fewer diagnostics,
        # so it's not silenced; the CPU fallback isn't silenced either.
        silence_stderr = gpu_ep == "CUDAExecutionProvider"
        ctx = silenced_fd_stderr() if silence_stderr else contextlib.nullcontext()
        try:
            with ctx:
                self._session = ort.InferenceSession(
                    str(model_path), providers=providers
                )
        except Exception as exc:  # noqa: BLE001
            if not gpu_ep:
                raise
            err = str(exc)
            logger.warning(
                "%s %s session creation failed, retrying with a CPU downgrade: %s", self.name, gpu_ep, err
            )
            if gpu_ep == "CUDAExecutionProvider":
                onnxruntime_setup.record_cuda_load_error(err)
            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        else:
            # onnxruntime does **not** raise an exception when a GPU EP fails to dlopen -- internally it silently
            # falls back to the next EP (CPU). A try/except alone isn't enough; you have to compare against the actual
            # session.get_providers() -- if it differs, the user is actually running on CPU but the UI can't see it.
            if gpu_ep:
                actual = list(self._session.get_providers())
                if gpu_ep not in actual:
                    msg = (
                        f"{gpu_ep} silently downgraded to CPU (InferenceSession raised no exception, "
                        f"but get_providers={{actual}}). Common causes: driver version too old / "
                        f"missing runtime .so/.dll / cuDNN ABI mismatch / DX12 unsupported."
                    )
                    logger.warning("%s %s", self.name, msg)
                    if gpu_ep == "CUDAExecutionProvider":
                        onnxruntime_setup.record_cuda_load_error(msg)
                elif gpu_ep == "CUDAExecutionProvider":
                    onnxruntime_setup.record_cuda_load_error(None)

        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def _fallback_to_cpu_session(self) -> bool:
        """Falls back to CPU and rebuilds the session after a CUDA inference failure; returns True on success.

        Difference from `_create_session`: that one hangs during session creation (dlopen failure);
        this one hangs during inference after creation (typically a cuBLAS / cuDNN ABI mismatch -> CUBLAS_STATUS_*).
        After rebuilding, `_session.get_providers()` only has CPU left, and `_effective_batch_size`
        automatically drops batch_n to 1, so later batches no longer go through CUDA.
        """
        if self._model_path is None:
            return False
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            return False
        try:
            logger.warning(
                "%s CUDA inference failed, downgrading to a CPU InferenceSession (later batches also go through CPU)",
                self.name,
            )
            self._session = ort.InferenceSession(
                str(self._model_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            onnxruntime_setup.record_cuda_load_error(
                "A cuBLAS / CUDA error occurred during CUDA inference; automatically downgraded to running on CPU"
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("%s CPU session downgrade failed: %s", self.name, exc)
            return False

    @staticmethod
    def _is_cuda_inference_error(exc: BaseException) -> bool:
        """Determines whether an inference exception is CUDA-related -> triggers one CPU-fallback retry.

        Typical keywords: CUBLAS_STATUS_*, CUDNN_STATUS_*, the CUDAExecutionProvider name.
        OOM is identified separately ("out of memory" / "OOM"); after downgrading to CPU it usually still can't run, but at least gives
        the user a clear error instead of an opaque crash.
        """
        msg = str(exc)
        keywords = ("CUBLAS", "CUDNN", "CUDAExecutionProvider", "out of memory", "OOM")
        return any(k in msg for k in keywords) or "CUDA" in msg

    # -------------------- batch + iteration --------------------

    def _effective_batch_size(self) -> int:
        n = max(1, int(self._get_batch_size_cfg() or 1))
        if self._session is not None:
            providers = list(self._session.get_providers())
            if "CUDAExecutionProvider" not in providers:
                return 1
        return n

    def _preprocess_one_safe(
        self, indexed: tuple[int, Path]
    ) -> tuple[int, Optional[np.ndarray], Optional[str]]:
        """`(j, path) -> (j, arr_or_None, err_or_None)`, for use with ThreadPool.

        PIL.Image.open / resize / paste release the GIL at the C layer -- multithreading can genuinely run in parallel,
        unblocking preprocess from being the bottleneck on a weak-single-core-but-strong-GPU setup (e.g. an EPYC + RTX 5090 in the cloud).
        """
        j, p = indexed
        try:
            with Image.open(p) as raw:
                return j, self._preprocess(raw), None
        except Exception as exc:  # noqa: BLE001
            return j, None, str(exc)

    def tag(
        self,
        image_paths: list[Path],
        on_progress: ProgressFn = lambda d, t: None,
    ) -> Iterator[TagResult]:
        if self._session is None:
            self.prepare()
        assert self._session is not None
        assert self._input_name is not None
        total = len(image_paths)
        batch_n = self._effective_batch_size()
        done = 0
        i = 0

        # The pool is only opened when batch > 1: the CPU EP path already forces batch_n to 1, so pool=None goes through the single-threaded
        # compat path with zero regression.
        pool: Optional[ThreadPoolExecutor] = None
        if batch_n > 1:
            pool = ThreadPoolExecutor(
                max_workers=batch_n, thread_name_prefix=f"{self.name}-prep"
            )
        try:
            yield from self._tag_loop(
                image_paths, batch_n, pool, total, done, i, on_progress
            )
        finally:
            if pool is not None:
                pool.shutdown(wait=False)

    def _tag_loop(
        self,
        image_paths: list[Path],
        batch_n: int,
        pool: Optional[ThreadPoolExecutor],
        total: int,
        done: int,
        i: int,
        on_progress: ProgressFn,
    ) -> Iterator[TagResult]:
        assert self._session is not None
        while i < total:
            chunk = image_paths[i : i + batch_n]
            arrs: list[np.ndarray] = []
            ok_idx: list[int] = []
            errs: dict[int, str] = {}
            if pool is None:
                for j, p in enumerate(chunk):
                    j, arr, err = self._preprocess_one_safe((j, p))
                    if err is None and arr is not None:
                        arrs.append(arr)
                        ok_idx.append(j)
                    else:
                        errs[j] = err or "preprocess failed"
            else:
                for j, arr, err in pool.map(
                    self._preprocess_one_safe, list(enumerate(chunk))
                ):
                    if err is None and arr is not None:
                        arrs.append(arr)
                        ok_idx.append(j)
                    else:
                        errs[j] = err or "preprocess failed"
            logits_batch: Optional[np.ndarray] = None
            if arrs:
                batch = np.stack(arrs, axis=0).copy()
                try:
                    logits_batch = self._session.run(
                        self._output_names, {self._input_name: batch}
                    )[0]
                except Exception as exc:  # noqa: BLE001
                    # CUDA inference failed (cuBLAS / cuDNN / OOM) -> retry once after downgrading to a CPU session
                    if self._is_cuda_inference_error(exc) and self._fallback_to_cpu_session():
                        try:
                            logits_batch = self._session.run(
                                self._output_names, {self._input_name: batch}
                            )[0]
                        except Exception as exc2:  # noqa: BLE001
                            for j in ok_idx:
                                errs[j] = f"inference failed: {exc2}"
                            logits_batch = None
                    else:
                        for j in ok_idx:
                            errs[j] = f"inference failed: {exc}"
                        logits_batch = None
            ok_pos = 0
            for j, p in enumerate(chunk):
                if j in errs:
                    yield {"image": p, "tags": [], "error": errs[j]}
                elif logits_batch is not None and ok_pos < logits_batch.shape[0]:
                    tags, raw_scores = self._postprocess_one(logits_batch[ok_pos])
                    ok_pos += 1
                    yield {"image": p, "tags": tags, "raw_scores": raw_scores}
                else:
                    yield {"image": p, "tags": [], "error": "no logits"}
                done += 1
                on_progress(done, total)
            i += batch_n
