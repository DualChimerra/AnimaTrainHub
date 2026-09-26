"""Eval metric registry -- the single source of truth for all eval metrics.

Shared by three consumers: (1) the Settings checkbox list (which metrics the user
enables); (2) eval orchestration (`eval_auto` only runs the runner matching the
"enabled set"); (3) frontend metric descriptions / display.

A **runner** = one metric job (a single inference pass that produces one or more
metrics):
- the `clip` runner produces clip_t + clip_i (shared CLIP encoding)
- the `dino` runner produces dino_i
- the `ccip` runner produces ccip_i (character identity)
- the `tag`  runner produces tag_recall (prompt following, reuses WD14)

Gating is at the runner level: a runner runs if any of its metrics is enabled
(metrics on the same runner share inference, so disabling just one doesn't save
compute -- it only affects what's displayed). `models` lists the download-center
entry keys that metric depends on.
"""
from __future__ import annotations

from typing import Any, Iterable

# Order is the frontend display order. default=True items go into the default
# enabled set (preserves existing behavior).
METRICS: list[dict[str, Any]] = [
    {
        "key": "clip_t", "label": "CLIP-T", "runner": "clip",
        "models": ["clip"], "default": True,
        "desc": "CLIP similarity between generated image and prompt text (prompt following)",
        "note": "The CLIP text tower doesn't understand booru tags -- tag-style captions score low and noisy",
    },
    {
        "key": "clip_i", "label": "CLIP-I", "runner": "clip",
        "models": ["clip"], "default": True,
        "desc": "CLIP image similarity between generated and reference images (overall semantics)",
        "note": "Natural-image domain, fairly coarse; not anime-tuned",
    },
    {
        "key": "dino_i", "label": "DINO-I", "runner": "dino",
        "models": ["dino"], "default": True,
        "desc": "DINOv2 feature similarity between generated and reference images (subject/structure fidelity)",
        "note": "Not anime-tuned; the small variant has weak discriminative power",
    },
    {
        "key": "ccip_i", "label": "CCIP-I", "runner": "ccip",
        "models": ["ccip"], "default": False,
        "desc": "Fraction of generated images judged as the same anime character as the reference set (character identity fidelity, anime domain)",
        "note": "Only for single-character character LoRAs; weak on hair color/skin tone",
    },
    {
        "key": "tag_recall", "label": "Tag-Recall", "runner": "tag",
        "models": ["wd14"], "default": False,
        "desc": "Re-tags the generated image and measures recall of the prompt's booru tags (anime-native prompt following)",
        "note": "Only applies to booru-tag captions",
    },
]

_BY_KEY = {m["key"]: m for m in METRICS}
ALL_KEYS: list[str] = [m["key"] for m in METRICS]
DEFAULT_ENABLED: list[str] = [m["key"] for m in METRICS if m["default"]]


def metric(key: str) -> dict[str, Any] | None:
    return _BY_KEY.get(key)


def normalize_enabled(enabled: Iterable[str] | None) -> set[str]:
    """Filter down to valid metric keys; None / empty -> default set (preserves existing behavior)."""
    if not enabled:
        return set(DEFAULT_ENABLED)
    out = {k for k in enabled if k in _BY_KEY}
    return out or set(DEFAULT_ENABLED)


def enabled_runners(enabled: Iterable[str] | None) -> list[str]:
    """Enabled set -> runners that need to run (deduped, in METRICS order). A
    runner is included as soon as any one of its metrics is enabled."""
    active = normalize_enabled(enabled)
    out: list[str] = []
    for m in METRICS:
        if m["key"] in active and m["runner"] not in out:
            out.append(m["runner"])
    return out


def runner_metrics(runner: str) -> list[str]:
    return [m["key"] for m in METRICS if m["runner"] == runner]


def public_catalog() -> list[dict[str, Any]]:
    """For the frontend Settings checkboxes + metric descriptions (no internal
    implementation details beyond the runner field)."""
    return [
        {
            "key": m["key"], "label": m["label"], "runner": m["runner"],
            "models": m["models"], "default": m["default"],
            "desc": m["desc"], "note": m["note"],
        }
        for m in METRICS
    ]
