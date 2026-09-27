"""EMA (exponential moving average) of adapter weights.

The weights at the final training step are whatever the last random batch happened to
push the parameters to; adjacent epochs often land on opposite sides of some optimum,
so picking a checkpoint becomes a coin flip. EMA maintains a **smoothed copy** during
training: after every optimizer.step()

    shadow <- decay * shadow + (1 - decay) * current weights

so shadow is a weighted average over the last several steps (with decay=0.999, the
effective window is roughly 1000 steps). In practice the smoothed copy is more stable
than any single point, often slightly better too, and overfits more gradually -- which
is especially useful for style LoRAs, since the "which epoch is best" guessing game
just disappears.

Two implementation details matter:

* **The shadow must be fp32.** Training weights are bf16, and bf16 only has 8 bits of
  mantissa (~2-3 decimal digits). With decay=0.999, each step's increment is a
  thousandth of the weight -- which would get rounded straight to 0 in a bf16
  accumulator, silently turning EMA into "an unchanging copy of old weights."
* **The shadow starts from the weights at the moment EMA begins, not from a zero
  initialization.** LoRA's initial ΔW=0; if the shadow started there, after 2760 steps
  there would still be 0.999^2760 ≈ 6% of "zero" baked into the average, effectively
  weakening the LoRA by 6% for nothing. Combined with warmup (see ``effective_decay``),
  the shadow tracks the current weights almost exactly early on.

Saving to disk reuses the adapter's own ``save()``: the ``applied()`` context manager
temporarily writes the shadow into the parameters and restores them exactly on exit.
That way none of the alpha-rewriting / ss_* metadata / family-tag logic needs to be
duplicated.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterable, Sequence

import torch

logger = logging.getLogger(__name__)


class AdapterEma:
    """An EMA shadow copy of trainable parameters.

    Args:
        params: the trainable parameters (``ctx.trainable_params``, i.e. the
            optimizer's param_groups flattened).
        decay: the smoothing coefficient. Larger = longer window, more stable, but more
            lag. 0.999 ≈ roughly the last thousand steps.
        start_step: which optimizer step to start accumulating from (calls to update()
            before this are simply ignored). Used to skip the early, volatile phase and
            only average over the part that's "already decent."
        warmup: when on, uses ``(1+n)/(10+n)`` to reduce the effective decay early on,
            so the shadow quickly catches up to the current weights instead of being
            held back by its starting point. On by default.
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

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        """Whether the shadow has been established yet (False before start_step)."""
        return self._shadow is not None

    def effective_decay(self) -> float:
        """The decay actually used for this update.

        During warmup this is ``min(decay, (1+n)/(10+n))``: 0.09 at n=0 (almost
        exactly tracking the current weights), 0.917 at n=100, converging to the
        configured value by n≈10000. The standard EMA warmup formula.
        """
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    def _init_shadow(self) -> None:
        self._shadow = [p.detach().float().clone() for p in self.params]

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(self, global_step: int) -> None:
        """Call this after an actual optimizer.step() (not every micro-batch)."""
        if global_step < self.start_step:
            return
        if self._shadow is None:
            self._init_shadow()
            logger.info(
                "EMA shadow established (step %d, decay=%.4f%s)",
                global_step, self.decay, ", with warmup" if self.warmup else "",
            )
            self.updates += 1
            return
        d = self.effective_decay()
        for shadow, param in zip(self._shadow, self.params):
            shadow.mul_(d).add_(param.detach().float(), alpha=1.0 - d)
        self.updates += 1

    # ------------------------------------------------------------------ saving
    @contextmanager
    def applied(self):
        """Temporarily swap the shadow weights into the parameters; ``yield``s whether a swap actually happened.

        Usage::

            with ema.applied() as ok:
                if ok:
                    injector.save(path_ema)

        Unconditionally restores the original weights on exit (including on the
        exception path) -- training must keep using the real weights.
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

    # ------------------------------------------------------------- checkpoint resume
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
                "EMA shadow count mismatch (saved %d / current %d), discarding and rebuilding on next update",
                len(shadow), len(self.params),
            )
            self._shadow = None
            return
        self._shadow = [
            t.to(device=p.device, dtype=torch.float32)
            for t, p in zip(shadow, self.params)
        ]


def build_ema(args: Any, trainable_params: Iterable[torch.nn.Parameter], total_steps: int | None):
    """Build an AdapterEma from args; returns None if not enabled.

    ``ema_start_ratio`` is a fraction of the total steps (0.3 = the first 30% is
    excluded from the average). If the total step count is unknown, this falls back to
    "start from the beginning" -- safer than silently skipping EMA altogether.
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
        "EMA enabled: decay=%.4f, accumulating from step %d (ema_start_ratio=%.2f)",
        decay, start_step, ratio,
    )
    return ema
