
from __future__ import annotations

import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _daemon():
    import anima_daemon

    return anima_daemon


def test_unload_reclaims_leftovers_even_when_model_absent(monkeypatch):
    d = _daemon()
    calls = []
    monkeypatch.setattr(d, "_reclaim_cuda_leftovers", lambda: calls.append(1))

    cache = d.ModelCache()
    assert not cache.loaded
    cache.unload()

    assert calls, "the early-exit branch must clean up (gc + empty_cache + pinned release)"


def test_generate_worker_reclaims_leftovers_on_failure(monkeypatch):
    d = _daemon()
    calls = []
    monkeypatch.setattr(d, "_reclaim_cuda_leftovers", lambda: calls.append(1))
    monkeypatch.setattr(d, "_emit_for", lambda *a, **k: None)

    def _boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(d, "_run_generate", _boom)
    d._run_generate_worker("req-x", 1, {}, Path("."), threading.Event())
    assert calls, "the failure path must clean up"

    calls.clear()
    monkeypatch.setattr(d, "_run_generate", lambda *a, **k: None)
    d._run_generate_worker("req-y", 2, {}, Path("."), threading.Event())
    assert not calls, "the success path does not clean up"
