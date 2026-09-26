"""PR-3 -- OnnxTaggerBase CPU fallback: auto-retry on CPU after a CUDA error during inference.

The CUDA fallback during session creation is already covered by _create_session; this file
specifically tests the fallback path during inference (_session.run raising a cuBLAS / CUDA
error).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from studio.services.tagging import onnx_base as onnx_tagger_base
from studio.services.tagging.onnx_base import OnnxTaggerBase


class _StubTagger(OnnxTaggerBase):
    """Minimal tagger implementation bypassing the filesystem + ONNX: fixed shape for a single
    image, single logit passthrough."""
    name = "stub"

    def __init__(self, batch: int = 2) -> None:
        super().__init__()
        self._batch = batch

    def prepare(self) -> None:  # tests inject the session directly, prepare is never called
        raise AssertionError("prepare() should not be called in tests")

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.float32)

    def _postprocess_one(self, logits: np.ndarray):
        return [f"score={float(logits[0]):.2f}"], {"score": float(logits[0])}

    def _get_batch_size_cfg(self) -> int:
        return self._batch


# ---------------------------------------------------------------------------
# _is_cuda_inference_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("msg", [
    "CUBLAS_STATUS_INVALID_VALUE",
    "CUBLAS_STATUS_EXECUTION_FAILED while running ConvBatch",
    "CUDNN_STATUS_BAD_PARAM",
    "CUDA error: invalid device function",
    "CUDAExecutionProvider failed to bind input",
    "out of memory: tried to allocate 12 GiB",
    "OOM at layer x",
])
def test_is_cuda_inference_error_recognizes_cuda_failures(msg: str) -> None:
    assert OnnxTaggerBase._is_cuda_inference_error(RuntimeError(msg)) is True


@pytest.mark.parametrize("msg", [
    "model file truncated",
    "tensor shape mismatch [3,4] vs [4,3]",
    "permission denied",
])
def test_is_cuda_inference_error_ignores_unrelated_failures(msg: str) -> None:
    assert OnnxTaggerBase._is_cuda_inference_error(RuntimeError(msg)) is False


# ---------------------------------------------------------------------------
# _create_session -- detecting a silent CUDA EP downgrade
# ---------------------------------------------------------------------------


def _make_fake_session(providers: list[str]):
    """Build a mock onnxruntime session: input/output name + given providers."""
    sess = MagicMock()
    sess.get_inputs.return_value = [MagicMock()]
    sess.get_inputs.return_value[0].name = "x"
    sess.get_outputs.return_value = [MagicMock()]
    sess.get_outputs.return_value[0].name = "y"
    sess.get_providers.return_value = list(providers)
    return sess


def test_create_session_records_error_on_silent_cuda_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """onnxruntime doesn't raise on a CUDA dlopen failure, it silently falls back to CPU;
    `_create_session` must compare the actual providers and stash an error, or the UI never
    sees it."""
    t = _StubTagger()

    fake_session = _make_fake_session(["CPUExecutionProvider"])  # requested GPU, but only CPU remains
    requested_providers: list[list[str]] = []

    def fake_session_ctor(path, providers):
        requested_providers.append(list(providers))
        return fake_session

    fake_ort = MagicMock(InferenceSession=fake_session_ctor)
    fake_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    # start with no prior error recorded
    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)

    try:
        t._create_session(Path("/fake/model.onnx"))
        # the caller requested CUDA
        assert requested_providers and "CUDAExecutionProvider" in requested_providers[0]
        # but onnxruntime silently downgraded to CPU internally -> cuda_load_error must be set so the UI can show it
        err = onnx_tagger_base.onnxruntime_setup.get_cuda_load_error()
        assert err is not None
        assert "静默降级" in err or "silently" in err.lower()
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


def test_create_session_clears_error_when_cuda_actually_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA requested and the actual providers include CUDA -> clears the old error (success path)."""
    t = _StubTagger()
    fake_session = _make_fake_session(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    fake_ort = MagicMock(InferenceSession=lambda _p, providers: fake_session)
    fake_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    # pre-seed an old error
    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error("old failure")

    try:
        t._create_session(Path("/fake/model.onnx"))
        assert onnx_tagger_base.onnxruntime_setup.get_cuda_load_error() is None
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


def test_create_session_no_false_positive_when_cuda_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No GPU on the machine -> CUDA was never requested -> CPU-only providers is normal, should not record an error."""
    t = _StubTagger()
    fake_session = _make_fake_session(["CPUExecutionProvider"])

    fake_ort = MagicMock(InferenceSession=lambda _p, providers: fake_session)
    fake_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)

    try:
        t._create_session(Path("/fake/model.onnx"))
        assert onnx_tagger_base.onnxruntime_setup.get_cuda_load_error() is None
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


def test_create_session_uses_directml_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With onnxruntime-directml installed -> providers should be [Dml, CPU], the DirectML EP
    must not be dropped (or InferenceSession would actually only run on CPU)."""
    t = _StubTagger()
    fake_session = _make_fake_session(
        ["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    requested_providers: list[list[str]] = []

    def fake_session_ctor(path, providers):
        requested_providers.append(list(providers))
        return fake_session

    fake_ort = MagicMock(InferenceSession=fake_session_ctor)
    # onnxruntime-directml wheel: DmlExecutionProvider available, no CUDA
    fake_ort.get_available_providers.return_value = [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)

    try:
        t._create_session(Path("/fake/model.onnx"))
        assert requested_providers
        assert requested_providers[0][0] == "DmlExecutionProvider"
        assert "CPUExecutionProvider" in requested_providers[0]
        # the DirectML success path must not accidentally stash into cuda_load_error
        assert onnx_tagger_base.onnxruntime_setup.get_cuda_load_error() is None
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


def test_create_session_directml_silent_downgrade_does_not_stash_cuda_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DirectML silently downgrades (available providers claim it's there, but get_providers
    actually only returns CPU) -> must not pollute the cuda_load_error field (that field is
    semantically CUDA-EP-only)."""
    t = _StubTagger()
    fake_session = _make_fake_session(["CPUExecutionProvider"])

    fake_ort = MagicMock(InferenceSession=lambda _p, providers: fake_session)
    fake_ort.get_available_providers.return_value = [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)

    try:
        t._create_session(Path("/fake/model.onnx"))
        # went through the DirectML silent-downgrade path, cuda_load_error should still be None
        assert onnx_tagger_base.onnxruntime_setup.get_cuda_load_error() is None
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


# ---------------------------------------------------------------------------
# _fallback_to_cpu_session
# ---------------------------------------------------------------------------


def test_fallback_returns_false_without_model_path() -> None:
    """_model_path unset (exceptional path, shouldn't happen in theory) -> silently returns False, doesn't raise."""
    t = _StubTagger()
    assert t._model_path is None
    assert t._fallback_to_cpu_session() is False


def test_fallback_creates_cpu_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """_model_path is set -> rebuilds the session with CPUExecutionProvider and stashes the error."""
    t = _StubTagger()
    t._model_path = Path("/fake/model.onnx")

    fake_session = MagicMock()
    fake_session.get_inputs.return_value = [MagicMock(name="input")]
    fake_session.get_inputs.return_value[0].name = "input"
    fake_session.get_outputs.return_value = [MagicMock()]
    fake_session.get_outputs.return_value[0].name = "out"

    captured: dict = {}

    def fake_session_ctor(path, providers):
        captured["path"] = path
        captured["providers"] = providers
        return fake_session

    fake_ort = MagicMock(InferenceSession=fake_session_ctor)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    # clear any old stashed error
    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)
    try:
        ok = t._fallback_to_cpu_session()
        assert ok is True
        assert captured["providers"] == ["CPUExecutionProvider"]
        assert Path(captured["path"]) == Path("/fake/model.onnx")
        assert t._session is fake_session
        assert t._input_name == "input"
        assert t._output_names == ["out"]
        assert (
            "cuBLAS"
            in (onnx_tagger_base.onnxruntime_setup.get_cuda_load_error() or "")
        )
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


def test_fallback_returns_false_when_session_ctor_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = _StubTagger()
    t._model_path = Path("/fake/model.onnx")

    def boom(*_a, **_k):
        raise RuntimeError("model corrupt")

    fake_ort = MagicMock(InferenceSession=boom)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    assert t._fallback_to_cpu_session() is False


# ---------------------------------------------------------------------------
# _tag_loop CUDA inference failure -> CPU fallback retry
# ---------------------------------------------------------------------------


def _attach_session(tagger: _StubTagger, run_side_effect, providers=("CPUExecutionProvider",)):
    """Build a session so _tag_loop can run directly (bypassing prepare)."""
    sess = MagicMock()
    sess.get_inputs.return_value = [MagicMock()]
    sess.get_inputs.return_value[0].name = "x"
    sess.get_outputs.return_value = [MagicMock()]
    sess.get_outputs.return_value[0].name = "y"
    sess.get_providers.return_value = list(providers)
    sess.run.side_effect = run_side_effect
    tagger._session = sess
    tagger._input_name = "x"
    tagger._output_names = ["y"]
    return sess


def test_inference_cuda_error_triggers_cpu_fallback_and_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CUDA error raised -> fallback rebuilds a CPU session -> retries on the CPU session -> yields a successful result."""
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(img)

    t = _StubTagger(batch=2)
    t._model_path = tmp_path / "fake.onnx"

    # the first run raises a cuBLAS error; after fallback rebuilds the session, the second run returns logits
    fail_then_succeed = [
        RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED"),
    ]
    success_logits = np.array([[0.7]], dtype=np.float32)

    cuda_sess = _attach_session(
        t, fail_then_succeed, providers=("CUDAExecutionProvider", "CPUExecutionProvider")
    )

    # fallback rebuild: sys.modules["onnxruntime"].InferenceSession returns the CPU session
    cpu_sess = MagicMock()
    cpu_sess.get_inputs.return_value = [MagicMock()]
    cpu_sess.get_inputs.return_value[0].name = "x"
    cpu_sess.get_outputs.return_value = [MagicMock()]
    cpu_sess.get_outputs.return_value[0].name = "y"
    cpu_sess.get_providers.return_value = ["CPUExecutionProvider"]
    cpu_sess.run.return_value = [success_logits]

    fake_ort = MagicMock(InferenceSession=lambda _p, providers: cpu_sess)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)
    try:
        results = list(t.tag([img]))
        assert len(results) == 1
        # retry succeeds after falling back to the CPU session -> should have no error field
        assert "error" not in results[0], results[0]
        assert results[0]["tags"] == ["score=0.70"]
        # session has switched to the CPU instance (not the original CUDA mock)
        assert t._session is cpu_sess
        assert "cuBLAS" in (onnx_tagger_base.onnxruntime_setup.get_cuda_load_error() or "")
    finally:
        onnx_tagger_base.onnxruntime_setup.record_cuda_load_error(None)


def test_non_cuda_inference_error_does_not_trigger_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plain inference error (e.g. shape mismatch) -> should not trigger CPU fallback, reports directly to the user."""
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (0, 255, 0)).save(img)

    t = _StubTagger(batch=2)
    t._model_path = tmp_path / "fake.onnx"
    cuda_sess = _attach_session(
        t,
        [RuntimeError("tensor shape mismatch [3,4] vs [4,3]")],
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )

    # probe: the fallback path should not be triggered -> InferenceSession should not be called
    fallback_called: list[bool] = []

    def _explode(*_a, **_k):
        fallback_called.append(True)
        raise AssertionError("CPU fallback shouldn't have been triggered")

    fake_ort = MagicMock(InferenceSession=_explode)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    results = list(t.tag([img]))
    assert fallback_called == []
    assert len(results) == 1
    assert "error" in results[0]
    assert "shape mismatch" in results[0]["error"]


def test_fallback_failure_propagates_original_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CUDA error + fallback session rebuild also fails -> the user sees the first (more diagnostically useful) error."""
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (0, 0, 255)).save(img)

    t = _StubTagger(batch=2)
    t._model_path = tmp_path / "fake.onnx"
    _attach_session(
        t,
        [RuntimeError("CUDNN_STATUS_BAD_PARAM")],
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )

    def boom(*_a, **_k):
        raise RuntimeError("model file truncated")

    fake_ort = MagicMock(InferenceSession=boom)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    results = list(t.tag([img]))
    assert "error" in results[0]
    # reports the original CUDA error (not the fallback's "model corrupt" failure)
    assert "CUDNN" in results[0]["error"]
