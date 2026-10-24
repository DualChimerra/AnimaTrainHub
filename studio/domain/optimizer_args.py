"""optimizer_args: extra optimizer constructor arguments as ``name=value`` lines.

Same convention as kohya's ``--optimizer_args``: values are Python literals
(``0.99``, ``(0.9, 0.99)``, ``True``); anything else is kept as a string, so
``adjust_lr_fn=original`` works without quotes. ``true`` / ``false`` / ``none``
are accepted in lower case too. Lives in studio/domain (no torch import) so
the config validator and the trainer share one parser.
"""
from __future__ import annotations

import ast
from typing import Any, Iterable

#: Set through their own fields; a second source would be ambiguous.
RESERVED_OPTIMIZER_ARGS = frozenset({"params", "lr", "learning_rate", "weight_decay"})

_LOWER_LITERALS = {"true": True, "false": False, "none": None}


def _parse_value(raw: str) -> Any:
    text = raw.strip()
    if text.lower() in _LOWER_LITERALS:
        return _LOWER_LITERALS[text.lower()]
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_optimizer_args(lines: Iterable[str] | None) -> dict[str, Any]:
    """``["betas=(0.9, 0.99)", "eps=1e-6"]`` -> ``{"betas": (0.9, 0.99), "eps": 1e-6}``.

    Raises ValueError naming the bad line: missing ``=``, a non-identifier
    name, a reserved name, or a name given twice.
    """
    result: dict[str, Any] = {}
    for line in lines or ():
        text = str(line).strip()
        if not text:
            continue
        name, sep, value = text.partition("=")
        name = name.strip()
        if not sep or not value.strip():
            raise ValueError(f"optimizer_args: expected name=value, got {text!r}")
        if not name.isidentifier():
            raise ValueError(f"optimizer_args: {name!r} is not a valid argument name")
        if name in RESERVED_OPTIMIZER_ARGS:
            raise ValueError(
                f"optimizer_args: set {name!r} with its own field (learning rate / weight decay)"
            )
        if name in result:
            raise ValueError(f"optimizer_args: {name!r} is given more than once")
        result[name] = _parse_value(value)
    return result
