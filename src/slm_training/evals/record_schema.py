"""Moved to ``slm_training.harness_core.record_schema`` (structural extraction).

This file keeps the historical import path working: it aliases the relocated
module in ``sys.modules`` so both paths resolve to the same module object.
New code should import from ``slm_training.harness_core``
(docs/design/harness-core.md).
"""

import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module(
    "slm_training.harness_core.record_schema"
)
