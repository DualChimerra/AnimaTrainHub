"""LyCORIS LoKr ``rank_dropout`` device bug compatibility patch.

Upstream bug: `torch.rand(weight.size(0))` doesn't pass `device=`, so it produces a
CPU mask that raises a device mismatch when multiplied with a CUDA weight. Only
triggered when `rank_dropout > 0` and the module is in training mode.

Why we can't rely solely on lycoris_adapter.py's model.train() hijack:
- The hijack only guarantees the network is in eval mode during sample/eval (which
  doesn't trigger the rank_dropout branch)
- But if the user configures `rank_dropout > 0`, a normal training step still goes
  through the rank_dropout branch -- the hijack doesn't cover this path, so the bug
  still hits
- Hence we wrap ``LokrModule.get_weight`` so the dropout mask follows the weight's device

Version guard:
- Only patches versions within KNOWN_AFFECTED_VERSIONS
- Other versions (including ones upstream has already fixed) log a warning and skip;
  this avoids overwriting an implementation upstream has already fixed
- Once upstream fixes this, remove the corresponding ``KNOWN_AFFECTED_VERSIONS`` entry

The implementation deliberately no longer imports LyCORIS internals like
``make_kron`` / ``rebuild_tucker``: 4.0 moved these implementations to the functional
kernel API. Wrapping the original method preserves 4.0's fused kernel dispatch while
keeping the patch stable across 3.4/4.0's internal refactors.

Upstream issue: https://github.com/KohakuBlueleaf/LyCORIS/issues -- to be filed
"""
from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

logger = logging.getLogger(__name__)

# lycoris-lora versions confirmed to be affected by the rank_dropout device bug.
# Verified in practice: 3.4.0 / 4.0.0's `lycoris/modules/lokr.py:get_weight` goes
# through `torch.rand(weight.size(0))` (a CPU mask), which fails when multiplied with a CUDA weight.
KNOWN_AFFECTED_VERSIONS: frozenset[str] = frozenset({"3.4.0", "4.0.0"})

PatchStatus = Literal[
    "applied",  # matched an affected version, patch applied
    "skipped_not_installed",  # lycoris not installed
    "skipped_version_unknown",  # installed but version not in the known-affected set (warn)
    "skipped_already_patched",  # already patched in this process, idempotent return
]

_PATCHED_FLAG = "_anima_lokr_device_patched"


def apply_lokr_device_patch() -> PatchStatus:
    """Check the lycoris-lora version and patch LokrModule.get_weight as needed.

    Idempotent: calling multiple times in the same process only patches once.
    """
    try:
        installed = version("lycoris-lora")
    except PackageNotFoundError:
        return "skipped_not_installed"

    try:
        from lycoris.modules.lokr import LokrModule
    except Exception as exc:  # pragma: no cover - edge case where lycoris-lora is installed but import fails
        logger.warning(
            "lycoris-lora %s is installed but importing lycoris.modules.lokr failed: %s; skipping device patch",
            installed,
            exc,
        )
        return "skipped_not_installed"

    if getattr(LokrModule, _PATCHED_FLAG, False):
        return "skipped_already_patched"

    if installed not in KNOWN_AFFECTED_VERSIONS:
        logger.warning(
            "lycoris-lora %s is not in the set of versions known to be affected by the "
            "rank_dropout device bug %s; skipping patch (assuming upstream already fixed it. "
            "If you hit a device mismatch during training, please report your version in the issue)",
            installed,
            sorted(KNOWN_AFFECTED_VERSIONS),
        )
        return "skipped_version_unknown"

    import torch  # noqa: PLC0415  deferred here to avoid top-level import side effects

    original_get_weight = LokrModule.get_weight

    def _get_weight_fixed(self, shape):  # type: ignore[no-untyped-def]
        # Ask upstream to build the weight (and, in 4.x, select its fused
        # kernel) with only the broken dropout branch temporarily disabled.
        # Restoring in ``finally`` keeps module state intact even if upstream
        # raises for an unsupported shape.
        rank_dropout = self.rank_dropout
        apply_dropout = bool(self.training and rank_dropout)
        if apply_dropout:
            self.rank_dropout = 0.0
        try:
            weight = original_get_weight(self, shape)
        finally:
            if apply_dropout:
                self.rank_dropout = rank_dropout

        if apply_dropout:
            drop = (
                torch.rand(weight.size(0), device=weight.device) > rank_dropout
            ).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop
        return weight

    LokrModule.get_weight = _get_weight_fixed
    setattr(LokrModule, _PATCHED_FLAG, True)
    logger.info(
        "lycoris-lora %s: patched LokrModule.get_weight (rank_dropout device fix)",
        installed,
    )
    return "applied"
