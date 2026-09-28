"""Starting parameter set for a new Graph board.

Every value here is only a suggestion: the user can rename, reorder, hide,
archive or delete any parameter or option, and add their own. The ``config``
block on an option (or on a number parameter) ties it to a training-config
key, which is what lets the Graph

* recognise queue tasks whose frozen config matches a card or an empty slot,
* build a new version's config from a slot ("create training from here").

``apply`` is the patch written into the version config; ``match`` is how a
task config is recognised (defaults to ``apply``). ``match`` values may be a
literal or one operator object: ``{"$gt": n}``, ``{"$gte": n}``,
``{"$truthy": bool}``.
"""
from __future__ import annotations

from typing import Any


def _opt(oid: str, label: str, apply: dict[str, Any] | None = None,
         match: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"id": oid, "label": label}
    if apply is not None or match is not None:
        out["config"] = {"apply": apply or {}, **({"match": match} if match is not None else {})}
    return out


def _num(oid: str, value: float) -> dict[str, Any]:
    return {"id": oid, "value": value}


def default_params() -> list[dict[str, Any]]:
    return [
        {
            "id": "model", "name": "Model", "type": "list",
            "options": [
                _opt("anima", "Anima", {"model_family": "anima"}),
                _opt("krea2", "Krea2", {"model_family": "krea2"}),
            ],
        },
        {
            "id": "resolution", "name": "Resolution", "type": "number",
            "config": {"key": "resolution", "list": True},
            "options": [_num("r1024", 1024), _num("r1536", 1536)],
        },
        {
            "id": "reg", "name": "Regulation set", "type": "bool",
            "description": "Whether the version trains with a regularization set. "
                           "It comes from the version's data, so creating a training "
                           "from the Graph can't switch it.",
            "options": [
                _opt("yes", "Yes", {}, {"reg_data_dir": {"$truthy": True}}),
                _opt("no", "No", {}, {"reg_data_dir": {"$truthy": False}}),
            ],
        },
        {
            "id": "navit", "name": "NaViT packing", "type": "bool",
            "options": [
                _opt("yes", "Yes", {"navit_packing": True}),
                _opt("no", "No", {"navit_packing": False}),
            ],
        },
        {
            "id": "lora_type", "name": "LoRA type", "type": "list",
            "options": [
                _opt("tlora", "TLoRA (Ortho)", {"lora_type": "tlora", "tlora_use_ortho": True}),
                _opt("lora", "LoRA", {"lora_type": "lora"}),
                _opt("lokr", "LoKr", {"lora_type": "lokr"}),
            ],
        },
        {
            "id": "rank_alpha", "name": "Rank/Alpha", "type": "list",
            "description": "Full matrix = LoKr with a huge dim, so LyCORIS stops "
                           "decomposing the second block.",
            "options": [
                _opt("full", "Full matrix", {"lora_rank": 10000000000, "lora_alpha": 10000000000},
                     {"lora_rank": {"$gte": 10000}}),
                _opt("r32a32", "32/32", {"lora_rank": 32, "lora_alpha": 32}),
                _opt("r64a64", "64/64", {"lora_rank": 64, "lora_alpha": 64}),
                _opt("r48a32", "48/32", {"lora_rank": 48, "lora_alpha": 32}),
            ],
        },
        {
            "id": "dora", "name": "DoRA", "type": "bool",
            "options": [
                _opt("yes", "Yes", {"lora_dora": True}),
                _opt("no", "No", {"lora_dora": False}),
            ],
        },
        {
            "id": "rslora", "name": "RS LoRA", "type": "bool",
            "options": [
                _opt("yes", "Yes", {"lora_rs": True}),
                _opt("no", "No", {"lora_rs": False}),
            ],
        },
        {
            "id": "steps", "name": "Total steps", "type": "number",
            "config": {"key": "max_steps"},
            "options": [_num("s1500", 1500), _num("s2000", 2000), _num("s2500", 2500), _num("s3000", 3000)],
        },
        {
            "id": "lr", "name": "LR", "type": "number",
            "config": {"key": "learning_rate"},
            "options": [
                _num("lr1", 0.0001), _num("lr2", 0.00001), _num("lr3", 0.00002),
                _num("lr4", 0.00003), _num("lr5", 0.00005),
            ],
        },
        {
            "id": "scheduler", "name": "Scheduler", "type": "list",
            "options": [
                _opt("constant", "Constant", {"lr_scheduler": "none"}),
                _opt("cosine", "Cosine", {"lr_scheduler": "cosine"}),
            ],
        },
        {
            "id": "optimizer", "name": "Optimizer", "type": "list",
            "options": [
                _opt("adamw", "AdamW", {"optimizer_type": "adamw"}),
                _opt("came", "CAME", {"optimizer_type": "came"}),
                _opt("prodigy", "Prodigy+", {"optimizer_type": "prodigy_plus_schedulefree"}),
            ],
        },
        {
            "id": "wd", "name": "Weight decay", "type": "bool",
            "options": [
                _opt("yes", "Yes", {"weight_decay": 0.01}, {"weight_decay": {"$gt": 0}}),
                _opt("no", "No", {"weight_decay": 0}),
            ],
        },
        {
            "id": "sampling", "name": "Sampling", "type": "list",
            "options": [
                _opt("style", "Style-friendly", {"timestep_sampling": "style_friendly"}),
                _opt("logit", "Logit-normal", {"timestep_sampling": "logit_normal"}),
            ],
        },
        {
            "id": "snr_mean", "name": "SNR mean", "type": "number",
            "config": {"key": "style_snr_mean"},
            "condition": {"param": "sampling", "options": ["style"]},
            "options": [],
        },
        {
            "id": "snr_sigma", "name": "SNR sigma", "type": "number",
            "config": {"key": "style_snr_sigma"},
            "condition": {"param": "sampling", "options": ["style"]},
            "options": [],
        },
    ]
