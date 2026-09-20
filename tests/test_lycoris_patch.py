"""utils.lycoris_patch — LyCORIS 3.4/4.0 LoKr rank-dropout device patch。

覆盖：
- 命中受影响版本 → patch 生效且保留 upstream weight builder
- 未装 lycoris → skipped_not_installed
- 未知版本 → skipped_version_unknown + warn
- 同进程内幂等 → skipped_already_patched

测试用 monkeypatch 改写 importlib.metadata.version 与 KNOWN_AFFECTED_VERSIONS
模拟不同环境，避免真实重新安装 lycoris。
"""
from __future__ import annotations

import importlib
import logging
import sys

import pytest


@pytest.fixture
def fresh_patch_module(monkeypatch: pytest.MonkeyPatch):
    """每个测试拿一份新 imported 的 lycoris_patch + 重置 LokrModule.get_weight。

    LokrModule 是单例；同进程内 patch 后属性会持久。fixture 在测试前后
    把 get_weight 还原到上游版本，并清掉 _PATCHED_FLAG，让每个测试都从
    「未 patch」状态起跑。
    """
    # 重新加载模块（清掉 module 级缓存）
    if "utils.lycoris_patch" in sys.modules:
        del sys.modules["utils.lycoris_patch"]
    mod = importlib.import_module("utils.lycoris_patch")

    # 备份并清理 LokrModule.get_weight，让幂等检查重置
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

    # 还原
    if LokrModule is not None and orig_get_weight is not None:
        LokrModule.get_weight = orig_get_weight
        if hasattr(LokrModule, mod._PATCHED_FLAG):
            delattr(LokrModule, mod._PATCHED_FLAG)


@pytest.mark.parametrize("affected_version", ["3.4.0", "4.0.0"])
def test_apply_on_known_affected_version_patches_get_weight(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch, affected_version: str,
) -> None:
    """受影响版本的 mask 应跟随 weight device，且 upstream builder 仍被调用。"""
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

    # 行为验证：让 get_weight 走到 rank_dropout 分支并触发 torch.rand —— mask
    # 必须生成在 weight.device 上。用 patch 把 torch.rand 替换成探针。
    captured: dict[str, object] = {}
    real_rand = torch.rand

    def _spy_rand(*args, **kwargs):
        captured["device"] = kwargs.get("device", None)
        return real_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", _spy_rand)

    # 用 mock self，不依赖不同 LyCORIS 版本的 LoKr constructor 细节。
    class _FakeSelf:
        training = True
        rank_dropout = 0.5
        rank_dropout_scale = False
        weight = torch.eye(4)

    fake = _FakeSelf()
    LokrModule.get_weight(fake, (4, 4))
    assert upstream_calls == [(4, 4)]
    assert fake.rank_dropout == 0.5, "wrapper 必须恢复 module 的 rank_dropout"
    assert captured["device"] == fake.weight.device


def test_apply_when_not_installed_returns_skipped(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没装 lycoris-lora（PackageNotFoundError）→ 静默跳过。"""
    def _raise(_pkg):
        raise fresh_patch_module.PackageNotFoundError
    monkeypatch.setattr(fresh_patch_module, "version", _raise)

    assert fresh_patch_module.apply_lokr_device_patch() == "skipped_not_installed"


def test_apply_unknown_version_skips_and_warns(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """未知版本（如上游已修的 3.5.0）→ 跳过并 warn，避免覆盖上游修复。"""
    pytest.importorskip("lycoris.modules.lokr")
    monkeypatch.setattr(fresh_patch_module, "version", lambda _: "999.0.0")
    with caplog.at_level(logging.WARNING, logger="utils.lycoris_patch"):
        status = fresh_patch_module.apply_lokr_device_patch()
    assert status == "skipped_version_unknown"
    assert any("999.0.0" in rec.message for rec in caplog.records)


def test_apply_idempotent(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同进程内重复调用：第一次 applied，之后 skipped_already_patched。"""
    pytest.importorskip("lycoris.modules.lokr")
    monkeypatch.setattr(fresh_patch_module, "version", lambda _: "4.0.0")
    assert fresh_patch_module.apply_lokr_device_patch() == "applied"
    assert fresh_patch_module.apply_lokr_device_patch() == "skipped_already_patched"
