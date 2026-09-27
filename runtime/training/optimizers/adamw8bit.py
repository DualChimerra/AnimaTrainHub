"""AdamW8bit optimizer build wrapper (ADR 0003 PR-C).

bitsandbytes' 8-bit AdamW: quantizes both momentum states (exp_avg /
exp_avg_sq) block-wise to int8, so state VRAM is ~25% of fp32 AdamW. Update
math matches AdamW, and hyperparameters (lr / betas / weight_decay) carry
over from AdamW unchanged -- unlike Lion / Automagic, which need lr
recalibration. The implementation lives in bitsandbytes; this file is just
the registry's build/validate shell, mirroring adamw.py.

Small tensors below `min_8bit_size` (utils-side default 4096) stay fp32:
quantizing them saves little and loses relatively more precision. LoRA's A/B
matrices are far above this threshold, so in practice everything runs 8-bit.

bitsandbytes is an **optional dependency** (commented out in
requirements.txt: it doesn't always install cleanly on Windows), so validate
turns "not installed" into an actionable error at startup instead of letting
build raise a bare ImportError from deep inside utils.
"""

from __future__ import annotations


def validate(args) -> None:
    """Startup check that the optional bitsandbytes dependency is available."""
    from utils.optimizer_utils import BITSANDBYTES_AVAILABLE

    if not BITSANDBYTES_AVAILABLE:
        raise SystemExit(
            "optimizer_type=adamw8bit requires bitsandbytes, which is not "
            "installed in this environment.\n"
            "  Install: pip install bitsandbytes\n"
            "  If it won't install: use lion (state VRAM ~= 50% of fp32 "
            "AdamW, but lr should be ~1/3 of the AdamW value) or came (state "
            "is even smaller, lr stays at AdamW scale)."
        )


def build(args, params, lr: float, weight_decay: float):
    """Build an 8-bit AdamW instance.

    Reads no extra args beyond the standard ones; keeps the same signature as
    the other builders so the registry can dispatch uniformly (same as
    adamw.py).
    """
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="adamw8bit",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
    )
