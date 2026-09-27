from __future__ import annotations

import importlib
import logging
import sys

import pytest


@pytest.fixture
def fresh_patch_module(monkeypatch: pytest.MonkeyPatch):
    if "utils.lycoris_patch" in sys.modules:
        del sys.modules["utils.lycoris_patch"]
    mod = importlib.import_module("utils.lycoris_patch")

    try:
        from lycoris.modules.lokr import LokrModule
        orig_get_weight = LokrModule.get_weight
        had_flag = getattr(LokrModule, mod._PATCHED_FLAG, False)
        if had_flag:
            delattr(LokrModule, mod._PATCHED_FLAG)
    except Exception:
        LokrModule = None  # type: ignore[assignment]
        orig_get_weight = None
        had_flag = False

    yield mod

    if LokrModule is not None and orig_get_weight is not None:
        LokrModule.get_weight = orig_get_weight
        if hasattr(LokrModule, mod._PATCHED_FLAG):
            delattr(LokrModule, mod._PATCHED_FLAG)


@pytest.mark.parametrize("affected_version", ["3.4.0", "4.0.0"])
def test_apply_on_known_affected_version_patches_get_weight(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch, affected_version: str,
) -> None:
    pytest.importorskip("lycoris.modules.lokr")
    import torch
    from lycoris.modules.lokr import LokrModule

    upstream_calls: list[object] = []

    def _fake_upstream(self, shape):
        upstream_calls.append(shape)
        assert self.rank_dropout == 0.0
        return self.weight.clone()

    monkeypatch.setattr(LokrModule, "get_weight", _fake_upstream)
    monkeypatch.setattr(fresh_patch_module, "version", lambda _: affected_version)

    status = fresh_patch_module.apply_lokr_device_patch()
    assert status == "applied"
    assert getattr(LokrModule, fresh_patch_module._PATCHED_FLAG) is True

    captured: dict[str, object] = {}
    real_rand = torch.rand

    def _spy_rand(*args, **kwargs):
        captured["device"] = kwargs.get("device", None)
        return real_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", _spy_rand)

    class _FakeSelf:
        training = True
        rank_dropout = 0.5
        rank_dropout_scale = False
        weight = torch.eye(4)

    fake = _FakeSelf()
    LokrModule.get_weight(fake, (4, 4))
    assert upstream_calls == [(4, 4)]
    assert fake.rank_dropout == 0.5, "wrapper must restore the module's rank_dropout"
    assert captured["device"] == fake.weight.device


def test_apply_when_not_installed_returns_skipped(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(_pkg):
        raise fresh_patch_module.PackageNotFoundError
    monkeypatch.setattr(fresh_patch_module, "version", _raise)

    assert fresh_patch_module.apply_lokr_device_patch() == "skipped_not_installed"


def test_apply_unknown_version_skips_and_warns(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("lycoris.modules.lokr")
    monkeypatch.setattr(fresh_patch_module, "version", lambda _: "999.0.0")
    with caplog.at_level(logging.WARNING, logger="utils.lycoris_patch"):
        status = fresh_patch_module.apply_lokr_device_patch()
    assert status == "skipped_version_unknown"
    assert any("999.0.0" in rec.message for rec in caplog.records)


def test_apply_idempotent(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("lycoris.modules.lokr")
    monkeypatch.setattr(fresh_patch_module, "version", lambda _: "4.0.0")
    assert fresh_patch_module.apply_lokr_device_patch() == "applied"
    assert fresh_patch_module.apply_lokr_device_patch() == "skipped_already_patched"
