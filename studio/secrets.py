"""Re-export shim -- the real module is studio.infrastructure.secrets (PR-7).

The 763-line file plus 170 lines of legacy migration hasn't been split into
3 files yet (the planned models/store/migrations split was pushed to 0.11.1
-- risk of circular Pydantic v2 validators across files, safer as its own
PR). This PR only does the move + shim.
"""
import sys as _sys

from .infrastructure import secrets as _real

_sys.modules[__name__] = _real
