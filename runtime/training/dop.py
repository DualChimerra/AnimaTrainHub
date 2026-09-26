"""DOP -- Differential Output Preservation.

Source: kohya-ss/sd-scripts PR #1710 and the feature of the same name in ostris/ai-toolkit.

**The problem this solves.** Style LoRA training often ends up with two flaws: first,
the trigger word affects the image even when it's absent from the prompt (the style
"leaks" everywhere); second, content from the dataset (the same recurring character,
the same background) gets carried into generated results too -- in other words, "the
style gets copied in wholesale instead of being applied on top."

**The traditional approach** is a regularization set: prepare a separate batch of
neutral images and train the LoRA not to touch them. The hassle is where those images
come from, and how far their distribution is from the training set.

**DOP takes a different approach**: no extra images needed. Take the **same batch of
training images**, strip the trigger word out of the caption, then:

1. Run a forward pass with the adapter disabled -> this is "what the base model would
   draw anyway" (no_grad, used as the reference);
2. Run the same input with the adapter enabled -> what LoRA draws now;
3. Add the MSE between the two as a penalty term to the total loss.

This teaches the LoRA two things at once: **with the trigger word = my style; without
the trigger word = I change nothing**. The content (character, pose, background) is
identical on both branches, so "copying content" gets no reward on either path --
which is exactly the constraint needed for "apply the style on top, don't copy the
content."

**Cost**: each step with DOP enabled runs two extra forward passes (one no_grad
reference + one with gradients), roughly 2-2.5x the per-step time. ``dop_ratio`` lets
you enable it on only a fraction of steps to trade some of that speed back.

**Scope (v1)**: the standard rectified flow path. NaViT packing (which needs
per-image cross-attention repacking) and LeapAlign (which brings its own objective)
are mutually exclusive with DOP at the schema level -- they don't silently get skipped.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def strip_trigger(caption: str, trigger: str) -> str:
    """Strip the trigger word out of a caption to get the prompt used by the preservation branch.

    Done in two passes, because in this project's hybrid captions the trigger word
    shows up in two forms:

    1. As a **standalone tag** (``@mystyle, 1girl, solo, ...``) -- dropped as a whole
       chunk after splitting on commas;
    2. **Embedded in a natural-language sentence** -- removed at word boundaries, with
       leftover whitespace cleaned up afterward.

    Case-insensitive. Returns the caption unchanged when the trigger is empty (the
    caller is responsible for fail-fast behavior here, see the schema validation).
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

    # Remove leftovers at word boundaries (\b doesn't work for triggers starting with
    # '@', so the boundaries are written out by hand)
    pattern = re.compile(
        r"(?<![\w@])" + re.escape(trig) + r"(?![\w])",
        flags=re.IGNORECASE,
    )
    text = pattern.sub("", text)

    # Clean up leftover ",  ,", leading commas, and repeated spaces after removal
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = text.strip().strip(",").strip()
    return text


def preservation_captions(captions: Sequence[str], trigger: str) -> list[str]:
    return [strip_trigger(c, trigger) for c in captions]


def has_trigger(captions: Sequence[str], trigger: str) -> bool:
    """Whether the trigger word actually appears in this batch of captions (used for the startup self-check and warning)."""
    trig = (trigger or "").strip().lower()
    if not trig:
        return False
    return any(trig in str(c).lower() for c in captions)


def assert_adapter_supports_dop(injector: Any) -> None:
    """DOP needs the adapter to be able to temporarily disable itself; error out at startup instead of discovering this mid-training."""
    if not hasattr(injector, "disabled"):
        raise RuntimeError(
            f"DOP requires the adapter to implement a disabled() context manager "
            f"(temporarily zeroing the scale to run a 'no LoRA' reference forward pass), "
            f"but the current adapter {type(injector).__name__} does not have one. "
            f"Either turn off dop_enabled, or add disabled() to this adapter."
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
    """Preservation loss: MSE(prediction with adapter, prediction with adapter disabled), same input, same trigger-free condition.

    The reference branch runs inside ``no_grad`` + ``injector.disabled()``, so it's a
    constant target -- gradients only flow back through the "with adapter" branch,
    which is exactly what we want to constrain.

    Reuses the main step's ``noisy`` / ``t``: the noise level is determined by the
    sampler, and DOP shouldn't set up a separate distribution of its own, or the
    preservation constraint would only apply to a narrow slice of noise levels.
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
    """Roll the dice according to dop_ratio. 1.0 = every step; 0 = never (equivalent to disabled)."""
    r = 1.0 if ratio is None else float(ratio)
    if r >= 1.0:
        return True
    if r <= 0.0:
        return False
    return rng.random() < r
