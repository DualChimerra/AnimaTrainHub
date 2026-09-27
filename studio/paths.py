"""Re-export shim -- the real module is studio.infrastructure.paths (PR-7).

The sys.modules alias makes monkeypatching / private access through the old
path transparently forward to the real module (same shim pattern as the PR-3
services/ shim). New code should use `from studio.infrastructure.paths import X`
directly.
"""
import sys as _sys

from .infrastructure import paths as _real

_sys.modules[__name__] = _real
