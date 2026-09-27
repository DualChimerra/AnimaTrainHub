"""Re-export shim -- PR-3.8's real module is studio.services.models (after the
4-way split this became an alias for the whole package).

The sys.modules alias transparently forwards old-path monkeypatches / private
access to the models package entry point; the original
`from studio.services.models import X` still works via the package's
__init__.py re-export.
New code should import directly with `from studio.services.models import X`.
"""
import sys as _sys

from . import models as _real

_sys.modules[__name__] = _real
