"""Moved to ``slm_training.harness_core.lineage`` (structural extraction).

This package keeps the historical import paths working: it aliases the
relocated package and every submodule in ``sys.modules`` so old and new paths
resolve to the same module objects (class identity and monkeypatching behave
identically through either path). New code should import from
``slm_training.harness_core.lineage`` (docs/design/harness-core.md).
"""

import importlib as _importlib
import sys as _sys

for _name in (
    "data_cycle",
    "evaluation_snapshot",
    "interventions",
    "merge",
    "promotion",
    "records",
    "store",
    "tracks",
):
    _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(
        f"slm_training.harness_core.lineage.{_name}"
    )
_sys.modules[__name__] = _importlib.import_module("slm_training.harness_core.lineage")
