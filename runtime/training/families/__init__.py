"""Model family registry -- the 8th plugin registry, the first at the architecture
level (multi-model support Phase 1).

PR-1 only carries ModelSpec (single source of declarative constants + latent
cache fingerprint protocol); the ModelFamily behavior interface and
get_family() dispatch land with PR-2b, with dispatch funneling all 6 call
sites through the anima_train public namespace (docs/design/multi-model/04-synthesis.md D8'/S5).
"""

from __future__ import annotations

# The families subtree uses relative imports: the studio server reuses this
# subtree under the `runtime.training.*` namespace (bucket distribution
# preview via dataset.py); over there sys.path only has the repo root, so an
# absolute `training.*` import would raise ModuleNotFoundError
# (tests/test_bucket_histogram.py has a regression import test for this).
from .spec import (  # noqa: F401  (re-export)
    ConstantShift,
    KNOWN_CAPABILITIES,
    LatentSpec,
    LoraOutputSpec,
    ModelSpec,
    ResolutionAwareShift,
    SamplingDefaults,
    TextSpec,
    validate_spec,
)
from .anima import ANIMA_SPEC
from .krea2 import KREA2_SPEC

SPECS: dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    validate_spec(spec)
    if spec.family_id in SPECS:
        raise ValueError(f"Model family registered twice: {spec.family_id}")
    SPECS[spec.family_id] = spec


_register(ANIMA_SPEC)
_register(KREA2_SPEC)


_FAMILIES: dict[str, object] = {}


def get_family(family_id: str):
    """Get the ModelFamily instance for a family id (lazily constructed). Unknown id -> ValueError."""
    get_spec(family_id)  # Raises here for an unknown id and lists what's registered
    fam = _FAMILIES.get(family_id)
    if fam is None:
        if family_id == "anima":
            from .anima.family import AnimaFamily

            fam = AnimaFamily()
        elif family_id == "krea2":
            from .krea2 import Krea2Family

            fam = Krea2Family()
        else:  # pragma: no cover - registry is kept in sync with SPECS
            raise ValueError(f"Model family '{family_id}' has no ModelFamily implementation")
        _FAMILIES[family_id] = fam
    return fam


def resolve_family(args):
    """Resolve model_family from args/cfg (defaults to anima, D7 zero migration).
    Supports both argparse.Namespace and dict carriers (bypass callers' cfg is a dict)."""
    if isinstance(args, dict):
        fid = args.get("model_family") or "anima"
    else:
        fid = getattr(args, "model_family", "anima") or "anima"
    return get_family(str(fid))


def get_spec(family_id: str) -> ModelSpec:
    """Get the ModelSpec for a family id. Unknown id -> ValueError listing what's registered."""
    try:
        return SPECS[str(family_id)]
    except KeyError:
        raise ValueError(
            f"Unknown model family '{family_id}', registered: {sorted(SPECS)}"
        ) from None
