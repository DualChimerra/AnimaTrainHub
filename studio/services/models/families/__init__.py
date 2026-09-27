"""Model-family asset registry (one of three studio-side homes, multi-model PR-4).

Shares the family-name join key ("anima" / "krea2") with the runtime
`training/families` SPECS. Download assets are allowed to land before the
training-side implementation; tests guarantee every runtime family has a
corresponding asset manifest.
Classification rule (01 §8.1): whatever appears in TrainingConfig's weight-path
fields is a "model family asset" and belongs in this package; "tool models"
like tagging / upscaling / eval / preview decoding always stay in ..paths.

Each family module exposes an ASSETS object (duck-typed):
- family_id / display_name
- default_paths_for_new_version(base_model) -- absolute weight paths for a new version
- transformer_path_for(sel) -- explicit base-model selection -> transformer absolute path
- selected_variant() -- the variant currently selected in Settings
- catalog_sections(root, models_cfg) -- this family's section of /api/models/catalog
"""
from __future__ import annotations

from typing import Optional

from . import anima as _anima
from . import krea2 as _krea2

FAMILY_ASSETS = {
    "anima": _anima.ASSETS,
    "krea2": _krea2.ASSETS,
}


def get_assets(family_id: str):
    try:
        return FAMILY_ASSETS[str(family_id)]
    except KeyError:
        raise ValueError(
            f"Unknown model family '{family_id}'; registered: {sorted(FAMILY_ASSETS)}"
        ) from None


def default_paths_for_new_version(
    base_model: Optional[str] = None, *, family: str = "anima"
) -> dict[str, str]:
    """Resolve a new version's 4 weight-path fields, per family (registry
    dispatch, multi-model P4-1).

    Historically this name was bound directly to the Anima implementation, so
    all 6 call sites (preset fork / save as preset / version config hint /
    bundle import / path-defaults endpoint / prior generation) got anima paths
    -- for a krea2 version, switching presets would overwrite the config with
    anima paths. Callers must pass in the config's `model_family`; an unknown
    family raises ValueError (listing registered ones).
    """
    return get_assets(family).default_paths_for_new_version(base_model)
