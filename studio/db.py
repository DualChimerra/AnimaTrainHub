"""Re-export shim -- the real module is studio.infrastructure.db (PR-7).

The 188-line file was kept as a single module instead of the planned 3-way
split into connection/tasks/settings -- same rationale as secrets.py: the
current shape doesn't hurt readability or edits, the split buys little, and
it's safer as its own PR.

The migrations/ subpackage also moved to studio/infrastructure/migrations/;
the `from .migrations import apply_all` relative import inside
`db.init_db()` keeps working transparently.
"""
import sys as _sys

from .infrastructure import db as _real

_sys.modules[__name__] = _real
