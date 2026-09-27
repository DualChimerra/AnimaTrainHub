from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "runtime"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_daemon  # noqa: E402


class _Stack:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.precached: list[list[str]] = []
        self.released = 0
        self.is_model_loaded = False

    def precache_online_prompts(self, prompts):
        if self.fail:
            raise RuntimeError("boom")
        self.precached.append(list(prompts))
        return len(prompts)

    def release_model(self):
        self.released += 1


def _with_stack(monkeypatch, stack):
    monkeypatch.setattr(anima_daemon.CACHE, "text_stack", stack, raising=False)
    return stack


def test_precache_encodes_and_releases(monkeypatch):
    stack = _with_stack(monkeypatch, _Stack())
    phases: list[str] = []

    anima_daemon._precache_prompts_and_release(
        ["p", "neg"], "auto", phases.append,
    )

    assert stack.precached == [["p", "neg"]]
    assert stack.released == 1
    assert phases == ["clip"]


def test_precache_performance_policy_keeps_te_resident(monkeypatch):
    stack = _with_stack(monkeypatch, _Stack())

    anima_daemon._precache_prompts_and_release(["p"], "performance")

    assert stack.precached == [["p"]]
    assert stack.released == 0


def test_precache_failure_falls_back_without_release(monkeypatch):
    stack = _with_stack(monkeypatch, _Stack(fail=True))

    anima_daemon._precache_prompts_and_release(["p"], "auto")

    assert stack.released == 0


def test_precache_skips_stacks_without_api(monkeypatch):
    monkeypatch.setattr(
        anima_daemon.CACHE, "text_stack",
        (SimpleNamespace(), SimpleNamespace(), SimpleNamespace()),
        raising=False,
    )
    anima_daemon._precache_prompts_and_release(["p"], "auto")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _te_cfg(path="G:/models/te-dir"):
    return {
        "model_family": "krea2",
        "text_encoder_path": path,
        "transformer_path": "G:/models/dit.safetensors",
        "vae_path": "G:/models/vae.safetensors",
        "mixed_precision": "bf16",
    }


def test_ensure_text_ready_builds_and_reuses(monkeypatch):
    cache = anima_daemon.ModelCache()
    built = []

    class _Family:
        def load_text(self, path, device, dtype, **kw):
            built.append(str(path))
            return _Stack()

    monkeypatch.setattr(anima_daemon._T, "get_family", lambda fid: _Family())
    monkeypatch.setattr(
        anima_daemon._T, "resolve_path_best_effort", lambda p, bases: str(p),
    )

    cache.ensure_text_ready(_te_cfg())
    assert len(built) == 1
    stack_first = cache.text_stack

    cache.ensure_text_ready(_te_cfg())
    assert len(built) == 1
    assert cache.text_stack is stack_first

    cache.ensure_text_ready(_te_cfg(path="G:/models/te-fp8"))
    assert len(built) == 2
    assert cache.text_stack is not stack_first


def test_ensure_text_ready_noop_for_anima(monkeypatch):
    cache = anima_daemon.ModelCache()
    cfg = _te_cfg()
    cfg["model_family"] = "anima"
    cache.ensure_text_ready(cfg)
    assert cache.text_stack is None


def test_unload_keep_text_preserves_stack(monkeypatch):
    cache = anima_daemon.ModelCache()
    stack = _Stack()
    cache.text_stack = stack
    cache._text_ready_key = ("krea2", "p")
    cache.model = object()
    cache.vae = object()

    cache.unload(keep_text=True)
    assert cache.text_stack is stack
    assert cache._text_ready_key == ("krea2", "p")
    assert cache.model is None

    cache.unload()
    assert cache.text_stack is None
    assert cache._text_ready_key is None
