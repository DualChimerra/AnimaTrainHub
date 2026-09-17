"""DOP（差分输出保持）：触发词剥离、参照前向、schema 接线。

核心语义：同一批图、去掉触发词的 caption，比较「开着适配器」与「关掉适配器」的
预测。带触发词=风格，不带=什么都不改；两条分支内容完全相同，所以「抄数据集内容」
拿不到奖励。详见 runtime/training/dop.py 的模块 docstring。
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from training import dop


# ── 触发词剥离 ────────────────────────────────────────────────────────────────

def test_strips_trigger_as_a_standalone_tag():
    caption = "@mystyle, 1girl, solo, furry female"
    assert dop.strip_trigger(caption, "@mystyle") == "1girl, solo, furry female"


def test_strips_trigger_from_the_middle_of_the_tag_list():
    caption = "1girl, @mystyle, solo"
    assert dop.strip_trigger(caption, "@mystyle") == "1girl, solo"


def test_strips_trigger_inside_a_natural_language_sentence():
    caption = "1girl, solo. A @mystyle styled girl reclines on sand."
    out = dop.strip_trigger(caption, "@mystyle")
    assert "@mystyle" not in out
    assert "styled girl reclines on sand" in out
    assert "  " not in out


def test_strip_is_case_insensitive():
    assert "@Style" not in dop.strip_trigger("@Style, 1girl", "@style")


def test_does_not_touch_words_that_merely_contain_the_trigger():
    """``@mystyle2`` 不是触发词，不能被吃掉一半。"""
    caption = "@mystyle2, 1girl"
    assert dop.strip_trigger(caption, "@mystyle") == caption


def test_empty_trigger_returns_caption_unchanged():
    assert dop.strip_trigger("1girl, solo", "") == "1girl, solo"
    assert dop.strip_trigger("1girl, solo", None) == "1girl, solo"


def test_leaves_no_dangling_separators():
    for caption in ("@t, 1girl", "1girl, @t", "@t", "1girl, @t, solo"):
        out = dop.strip_trigger(caption, "@t")
        assert not out.startswith(","), caption
        assert not out.endswith(","), caption
        assert ", ," not in out, caption


def test_preservation_captions_maps_the_whole_batch():
    out = dop.preservation_captions(["@t, a", "@t, b"], "@t")
    assert out == ["a", "b"]


def test_has_trigger_detects_presence():
    assert dop.has_trigger(["@t, a", "b"], "@t")
    assert not dop.has_trigger(["a", "b"], "@t")
    assert not dop.has_trigger(["@t, a"], "")


# ── 参照前向 ──────────────────────────────────────────────────────────────────

class _FakeInjector:
    """最小适配器替身：disabled() 里把 delta 置 0。"""

    def __init__(self):
        self.delta = 1.0
        self.disable_calls = 0

    from contextlib import contextmanager as _cm

    @_cm
    def disabled(self):
        self.disable_calls += 1
        previous = self.delta
        self.delta = 0.0
        try:
            yield
        finally:
            self.delta = previous


class _FakeFamily:
    """forward_train 的输出 = 输入 + injector.delta，便于断言两支的差。"""

    def __init__(self, injector):
        self.injector = injector
        self.checkpoint_flags: list[bool] = []

    def forward_train(self, model, noisy, t, cross, use_checkpoint=False):
        self.checkpoint_flags.append(use_checkpoint)
        return noisy + self.injector.delta * cross.sum() * torch.ones_like(noisy)


def _pieces():
    inj = _FakeInjector()
    fam = _FakeFamily(inj)
    noisy = torch.zeros(1, 2, requires_grad=False)
    t = torch.tensor([0.5])
    cross = torch.ones(1, 1, requires_grad=True)
    return inj, fam, noisy, t, cross


def test_reference_branch_runs_with_the_adapter_disabled():
    inj, fam, noisy, t, cross = _pieces()
    loss = dop.compute_dop_loss(
        family=fam, model=None, injector=inj,
        noisy=noisy, t=t, cross_wo_trigger=cross,
    )
    assert inj.disable_calls == 1          # 参照分支关了适配器
    assert inj.delta == 1.0                # 之后恢复
    # 有适配器 = noisy + 1*1, 无适配器 = noisy + 0 → MSE = 1
    assert pytest.approx(float(loss.detach())) == 1.0


def test_loss_is_zero_when_the_adapter_changes_nothing():
    inj, fam, noisy, t, cross = _pieces()
    inj.delta = 0.0
    loss = dop.compute_dop_loss(
        family=fam, model=None, injector=inj,
        noisy=noisy, t=t, cross_wo_trigger=cross,
    )
    assert pytest.approx(float(loss)) == 0.0


def test_gradient_flows_only_through_the_adapter_branch():
    """参照分支在 no_grad 里 —— 它是常量目标，不能反向传播。"""
    inj, fam, noisy, t, cross = _pieces()
    loss = dop.compute_dop_loss(
        family=fam, model=None, injector=inj,
        noisy=noisy, t=t, cross_wo_trigger=cross,
    )
    assert loss.requires_grad
    loss.backward()
    assert cross.grad is not None


def test_reference_branch_never_uses_gradient_checkpointing():
    """no_grad 下 checkpoint 只是白白重算一遍，没有任何收益。"""
    inj, fam, noisy, t, cross = _pieces()
    dop.compute_dop_loss(
        family=fam, model=None, injector=inj,
        noisy=noisy, t=t, cross_wo_trigger=cross, use_checkpoint=True,
    )
    assert fam.checkpoint_flags == [False, True]   # 参照分支 False，训练分支 True


# ── ratio / 启动期校验 ────────────────────────────────────────────────────────

def test_should_apply_edges():
    rng = random.Random(0)
    assert dop.should_apply(1.0, rng) is True
    assert dop.should_apply(0.0, rng) is False
    assert dop.should_apply(None, rng) is True


def test_should_apply_is_roughly_the_requested_fraction():
    rng = random.Random(0)
    hits = sum(dop.should_apply(0.3, rng) for _ in range(4000))
    assert 0.25 < hits / 4000 < 0.35


def test_adapter_without_disabled_fails_fast():
    with pytest.raises(RuntimeError, match="disabled"):
        dop.assert_adapter_supports_dop(SimpleNamespace())
    dop.assert_adapter_supports_dop(_FakeInjector())   # 有 disabled() → 通过


# ── schema ───────────────────────────────────────────────────────────────────

def test_schema_fields_and_exclusivity():
    from studio.schema import TrainingConfig

    cfg = TrainingConfig(dop_enabled=True)
    assert cfg.dop_weight == 1.0 and cfg.dop_ratio == 1.0

    for conflict in ({"leap_enabled": True}, {"navit_packing": True}):
        with pytest.raises(Exception):
            TrainingConfig(dop_enabled=True, **conflict)

    for name in ("dop_weight", "dop_ratio"):
        meta = TrainingConfig.model_fields[name].json_schema_extra
        assert meta["show_when"] == "dop_enabled==true", name
