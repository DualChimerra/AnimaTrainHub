"""DOP —— Differential Output Preservation（差分输出保持）。

来源：kohya-ss/sd-scripts PR #1710 与 ostris/ai-toolkit 的同名功能。

**要解决的问题。** 风格 LoRA 训练完经常有两个毛病：一是触发词不在提示词里也照样
改图（风格「漏」得到处都是）；二是把数据集里的内容（同一个典型角色、同一种背景）
一起搬进生成结果，也就是「不是把风格贴上去，而是把数据集抄过来」。

**传统做法**是正则集：另外准备一堆中性图片，让 LoRA 学着别动它们。麻烦在于图从哪来、
和训练集分布差多少都要操心。

**DOP 换了个思路**：不需要额外图片。拿**同一批训练图**，把 caption 里的触发词去掉，
然后

1. 关掉适配器跑一次前向 → 这就是「底模本来会画成什么样」（no_grad，当参照）；
2. 开着适配器跑同样的输入 → 现在 LoRA 会画成什么样；
3. 两者的 MSE 作为惩罚项加进总 loss。

于是 LoRA 被同时教两件事：**带触发词 = 我的风格；不带触发词 = 我什么都不改**。
内容（角色、姿势、背景）在两条分支里完全相同，所以「抄内容」这条路拿不到任何奖励——
这正是「风格贴上去而不是抄过来」想要的约束。

**代价**：每个启用 DOP 的 step 多两次前向（一次 no_grad 参照 + 一次带梯度），
约 2-2.5× 单步耗时。``dop_ratio`` 可以只在一部分 step 上启用来换回速度。

**范围（v1）**：标准 rectified flow 路径。NaViT 打包（逐图 cross 需要重新打包）与
LeapAlign（自带目标函数）在 schema 层互斥，不静默跳过。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def strip_trigger(caption: str, trigger: str) -> str:
    """从 caption 里去掉触发词，得到「保持分支」用的提示词。

    两遍处理，因为触发词在本项目的混合 caption 里有两种出现形态：

    1. **独立的 tag**（``@mystyle, 1girl, solo, ...``）—— 按逗号切开后整块丢弃；
    2. **夹在自然语言句子里** —— 按词边界删除，再收拾多余空格。

    大小写不敏感。触发词为空时原样返回（调用方负责 fail-fast，见 schema 校验）。
    """
    trig = (trigger or "").strip()
    if not trig:
        return caption
    low = trig.lower()

    kept = [
        chunk for chunk in str(caption).split(",")
        if chunk.strip().lower() != low
    ]
    text = ",".join(kept)

    # 词边界删除残留（\b 对 '@' 开头的触发词不成立，故自己写前后界）
    pattern = re.compile(
        r"(?<![\w@])" + re.escape(trig) + r"(?![\w])",
        flags=re.IGNORECASE,
    )
    text = pattern.sub("", text)

    # 收拾删除后留下的 ",  ,"、行首逗号、重复空格
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = text.strip().strip(",").strip()
    return text


def preservation_captions(captions: Sequence[str], trigger: str) -> list[str]:
    return [strip_trigger(c, trigger) for c in captions]


def has_trigger(captions: Sequence[str], trigger: str) -> bool:
    """本批 caption 里是否真的出现过触发词（用于启动期自检与告警）。"""
    trig = (trigger or "").strip().lower()
    if not trig:
        return False
    return any(trig in str(c).lower() for c in captions)


def assert_adapter_supports_dop(injector: Any) -> None:
    """DOP 需要适配器能临时关掉自己；不支持就在启动期报错而不是训到一半才发现。"""
    if not hasattr(injector, "disabled"):
        raise RuntimeError(
            f"DOP 需要适配器实现 disabled() 上下文（临时把缩放置 0 跑「无 LoRA」参照前向），"
            f"当前适配器 {type(injector).__name__} 没有。"
            f"请关闭 dop_enabled，或给该适配器补上 disabled()。"
        )


def compute_dop_loss(
    *,
    family: Any,
    model: Any,
    injector: Any,
    noisy: torch.Tensor,
    t: torch.Tensor,
    cross_wo_trigger: torch.Tensor,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """保持损失：MSE(带适配器的预测, 关掉适配器的预测)，同一输入、同一无触发词条件。

    参照分支在 ``no_grad`` + ``injector.disabled()`` 里跑，所以它是常量目标，
    梯度只从「带适配器」那一支回流——正是我们想约束的对象。

    复用主 step 的 ``noisy`` / ``t``：噪声档由采样器决定，DOP 不该另立一套分布，
    否则保持约束只作用在某个窄噪声段上。
    """
    with torch.no_grad():
        with injector.disabled():
            base_pred = family.forward_train(
                model, noisy, t, cross_wo_trigger, use_checkpoint=False,
            )
    lora_pred = family.forward_train(
        model, noisy, t, cross_wo_trigger, use_checkpoint=use_checkpoint,
    )
    return F.mse_loss(lora_pred.float(), base_pred.float().detach())


def should_apply(ratio: float, rng) -> bool:
    """按 dop_ratio 掷骰子。1.0 = 每个 step 都做；0 = 从不（等于关闭）。"""
    r = 1.0 if ratio is None else float(ratio)
    if r >= 1.0:
        return True
    if r <= 0.0:
        return False
    return rng.random() < r
