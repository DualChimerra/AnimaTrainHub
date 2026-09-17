"""适配器权重 EMA（指数滑动平均）。

训练到最后一步的权重是「最后一次随机批次把参数推到哪」的结果；相邻 epoch 之间
经常一个偏这边一个偏那边，选哪个 checkpoint 变成看运气。EMA 在训练途中维护一份
**平滑副本**：每次 optimizer.step() 之后

    shadow ← decay · shadow + (1 - decay) · 当前权重

于是 shadow 是最近若干步的加权平均（decay=0.999 的有效窗口 ≈ 1000 步）。实践里
平滑副本比任意单点更稳、常常还略好，过拟合也来得更平缓——对风格 LoRA 尤其实用，
因为「第几个 epoch 最好」的抓阄环节直接消失。

两个实现要点：

* **影子必须 fp32。** 训练权重是 bf16，而 bf16 只有 8 位尾数（≈2-3 位十进制）。
  decay=0.999 时每步的增量是权重的千分之一 —— 在 bf16 累加器里会被直接舍成 0，
  EMA 静默变成「一份不动的旧权重拷贝」。
* **影子从「开始 EMA 的那一刻的权重」起步，而不是从零初始化起步。** LoRA 的初始
  ΔW=0，若从它起步，2760 步后仍有 0.999^2760 ≈ 6% 的「零」留在平均里，等于白白
  把 LoRA 削弱 6%。配合 warmup（见 ``effective_decay``）早期几乎等于跟随当前权重。

落盘复用 adapter 自己的 ``save()``：``applied()`` 上下文临时把影子写进参数，退出
时原样还原。这样 alpha 重写 / ss_* metadata / 族标记等全部逻辑不用复制一份。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterable, Sequence

import torch

logger = logging.getLogger(__name__)


class AdapterEma:
    """可训练参数的 EMA 影子副本。

    Args:
        params: 可训练参数（``ctx.trainable_params``，即 optimizer 的 param_groups 展平）。
        decay: 平滑系数。越大窗口越长、越稳但越滞后。0.999 ≈ 最近千步。
        start_step: 从第几个 optimizer step 开始累计（之前的 update 调用直接忽略）。
            用来跳过前期的剧烈阶段，只平均「已经像样了」的那一段。
        warmup: 早期用 ``(1+n)/(10+n)`` 压低实际 decay，让影子迅速追上当前权重，
            而不是被起点拖着。默认开。
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        decay: float = 0.999,
        start_step: int = 0,
        warmup: bool = True,
    ) -> None:
        self.params: list[torch.nn.Parameter] = [p for p in params if p.requires_grad]
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.warmup = bool(warmup)
        self.updates = 0
        self._shadow: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------ 状态
    @property
    def active(self) -> bool:
        """影子是否已经建立（start_step 之前为 False）。"""
        return self._shadow is not None

    def effective_decay(self) -> float:
        """本次 update 实际使用的 decay。

        warmup 阶段取 ``min(decay, (1+n)/(10+n))``：n=0 时 0.09（几乎直接跟随当前
        权重），n=100 时 0.917，n≈10000 后收敛到设定值。标准 EMA warmup 写法。
        """
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    def _init_shadow(self) -> None:
        self._shadow = [p.detach().float().clone() for p in self.params]

    # ------------------------------------------------------------------ 更新
    @torch.no_grad()
    def update(self, global_step: int) -> None:
        """在一次真实的 optimizer.step() 之后调用（不是每个 micro-batch）。"""
        if global_step < self.start_step:
            return
        if self._shadow is None:
            self._init_shadow()
            logger.info(
                "EMA 影子已建立（step %d，decay=%.4f%s）",
                global_step, self.decay, "，含 warmup" if self.warmup else "",
            )
            self.updates += 1
            return
        d = self.effective_decay()
        for shadow, param in zip(self._shadow, self.params):
            shadow.mul_(d).add_(param.detach().float(), alpha=1.0 - d)
        self.updates += 1

    # ------------------------------------------------------------------ 落盘
    @contextmanager
    def applied(self):
        """临时把影子权重放进参数；``yield`` 出是否真的换了。

        用法::

            with ema.applied() as ok:
                if ok:
                    injector.save(path_ema)

        退出时无条件还原原权重（包括异常路径）——训练必须继续用真权重。
        """
        if self._shadow is None:
            yield False
            return
        backup = [p.detach().clone() for p in self.params]
        try:
            for shadow, param in zip(self._shadow, self.params):
                param.data.copy_(shadow.to(dtype=param.dtype))
            yield True
        finally:
            for saved, param in zip(backup, self.params):
                param.data.copy_(saved)

    # ------------------------------------------------------------- 断点续训
    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "start_step": self.start_step,
            "warmup": self.warmup,
            "updates": self.updates,
            "shadow": [s.cpu() for s in self._shadow] if self._shadow is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.decay = float(state.get("decay", self.decay))
        self.start_step = int(state.get("start_step", self.start_step))
        self.warmup = bool(state.get("warmup", self.warmup))
        self.updates = int(state.get("updates", 0))
        shadow: Sequence[torch.Tensor] | None = state.get("shadow")
        if not shadow:
            self._shadow = None
            return
        if len(shadow) != len(self.params):
            logger.warning(
                "EMA 影子数量不匹配（存档 %d / 当前 %d），放弃恢复、下次 update 重新建立",
                len(shadow), len(self.params),
            )
            self._shadow = None
            return
        self._shadow = [
            t.to(device=p.device, dtype=torch.float32)
            for t, p in zip(shadow, self.params)
        ]


def build_ema(args: Any, trainable_params: Iterable[torch.nn.Parameter], total_steps: int | None):
    """按 args 构建 AdapterEma；未启用返回 None。

    ``ema_start_ratio`` 是占总步数的比例（0.3 = 前 30% 不计入平均）。总步数未知时
    退化为「从头开始」——比静默跳过整个 EMA 安全。
    """
    if not bool(getattr(args, "ema_enabled", False)):
        return None
    decay = getattr(args, "ema_decay", 0.999)
    decay = 0.999 if decay is None else float(decay)
    ratio = getattr(args, "ema_start_ratio", 0.0)
    ratio = 0.0 if ratio is None else float(ratio)
    start_step = int(round(ratio * total_steps)) if (total_steps and ratio > 0) else 0
    ema = AdapterEma(trainable_params, decay=decay, start_step=start_step)
    logger.info(
        "EMA 已启用：decay=%.4f，从 step %d 开始累计（ema_start_ratio=%.2f）",
        decay, start_step, ratio,
    )
    return ema
