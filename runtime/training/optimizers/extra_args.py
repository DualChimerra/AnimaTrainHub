"""Merge the user's optimizer_args over a builder's constructor kwargs."""

from __future__ import annotations

import logging
from typing import Any

from studio.domain.optimizer_args import parse_optimizer_args

logger = logging.getLogger(__name__)


def with_optimizer_args(args, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return ``kwargs`` with ``args.optimizer_args`` applied on top.

    Unknown names are left for the optimizer constructor to reject, so a typo
    fails at startup instead of being ignored.
    """
    extra = parse_optimizer_args(getattr(args, "optimizer_args", None))
    if extra:
        logger.info("optimizer_args: %s", ", ".join(f"{k}={v!r}" for k, v in extra.items()))
    return {**kwargs, **extra}
