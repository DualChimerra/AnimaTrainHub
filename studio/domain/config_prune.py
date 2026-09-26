"""Prune fields not needed on disk from the yaml, based on UI visibility metadata.

Two pruning rules (studio/web/src/lib/schema.ts's pruneInactiveConfig is the
frontend mirror of the same semantics; the YAML preview drawer depends on it
matching what's actually saved to disk, so keep both sides in sync if the
semantics change):

1. show_when evaluates false -- a field that's not visible in the UI and
   whose value doesn't take effect under the current config (the
   eval_show_when evaluator mirrors the frontend's evalShowWhen verbatim).
2. hidden=True and the value equals the schema default -- a field the UI
   never renders (advanced/expert-only knobs, trigger_word, etc); saving the
   default value to disk is pure noise, while a non-default value (a bare
   CLI user's manual override, or trigger_word written by the Tagging page)
   is kept as usual.

Runtime-side safety (since config pipeline cut 1 / R1): the trainer loads the
yaml through the same TrainingConfig construction path as Studio
(argparse_bridge.namespace_from_config); missing keys get filled in via
pydantic defaults + the FAMILY_CONFIG_DEFAULTS family overlay -- matching the
saving side's pre-prune values field-for-field. Lesson learned: the old
merge_yaml_into_namespace filled missing keys with bare argparse defaults and
skipped the family overlay, so a pruned shuffle_caption on krea2 fell back to
anima's semantic default of True and broke training. Pruning safety is always
judged by "trainer read-back == saving side's read-back".

disable_when is never pruned -- a field it hits gets reset by the frontend to
disable_value, and disable_value may not equal the field's default (e.g. with
Prodigy, lr_scheduler is pinned to "none"); pruning it would change behavior
by making runtime read back the default instead, so it's kept.
"""
from typing import Any

from pydantic import BaseModel

# The evaluator lives in config_rules (a zero-dependency leaf that training's
# validator also needs; putting it here would create a
# `config_prune -> training` cycle); re-exported here to keep existing
# consumers unchanged.
from .config_rules import _MISSING, _js_str, eval_show_when  # noqa: F401
from .training import TrainingConfig


def prune_inactive_fields(
    dumped: dict[str, Any], model_cls: type[BaseModel] = TrainingConfig
) -> dict[str, Any]:
    """Remove fields whose show_when evaluates false, and fields with hidden=True still at the schema default, from a model_dump result.

    Every expression is evaluated against the full `dumped` dict (matching
    the frontend evaluating against the whole form state), so fields removed
    earlier don't affect later checks. disable_when fields and fields without
    metadata are kept as-is.
    """
    out = dict(dumped)
    for name, field in model_cls.model_fields.items():
        extra = field.json_schema_extra
        if not isinstance(extra, dict):
            continue
        expr = extra.get("show_when")
        if isinstance(expr, str) and expr and not eval_show_when(expr, dumped):
            out.pop(name, None)
            continue
        if extra.get("hidden") and name in out:
            if out[name] == field.get_default(call_default_factory=True):
                out.pop(name)
    return out
