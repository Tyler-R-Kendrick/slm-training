"""Pre-extraction import paths must alias the relocated harness-core modules.

The harness-core extraction (docs/design/harness-core.md) kept every old
import path as a ``sys.modules`` shim. Old and new paths must resolve to the
*same* module object so class identity and monkeypatching behave identically
through either path.
"""

from __future__ import annotations

import importlib

import pytest

MODULE_PAIRS = [
    ("slm_training.versioning", "slm_training.harness_core.versioning"),
    ("slm_training.lineage", "slm_training.harness_core.lineage"),
    ("slm_training.lineage.data_cycle", "slm_training.harness_core.lineage.data_cycle"),
    (
        "slm_training.lineage.evaluation_snapshot",
        "slm_training.harness_core.lineage.evaluation_snapshot",
    ),
    (
        "slm_training.lineage.interventions",
        "slm_training.harness_core.lineage.interventions",
    ),
    ("slm_training.lineage.merge", "slm_training.harness_core.lineage.merge"),
    ("slm_training.lineage.promotion", "slm_training.harness_core.lineage.promotion"),
    ("slm_training.lineage.records", "slm_training.harness_core.lineage.records"),
    ("slm_training.lineage.store", "slm_training.harness_core.lineage.store"),
    ("slm_training.lineage.tracks", "slm_training.harness_core.lineage.tracks"),
    (
        "slm_training.harnesses.model_build.checkpoint_reference",
        "slm_training.harness_core.checkpoint_reference",
    ),
    (
        "slm_training.harnesses.experiments.efficiency_gain",
        "slm_training.harness_core.efficiency_gain",
    ),
    (
        "slm_training.harnesses.experiments.scaling_fit",
        "slm_training.harness_core.scaling_fit",
    ),
    ("slm_training.evals.record_schema", "slm_training.harness_core.record_schema"),
    ("slm_training.evals.eval_cache", "slm_training.harness_core.eval_cache"),
    ("slm_training.evals.score_policy", "slm_training.harness_core.score_policy"),
]


@pytest.mark.parametrize("old,new", MODULE_PAIRS, ids=[old for old, _ in MODULE_PAIRS])
def test_old_path_aliases_new_module(old: str, new: str) -> None:
    assert importlib.import_module(old) is importlib.import_module(new)


def test_class_identity_across_paths() -> None:
    from slm_training.harness_core.checkpoint_reference import (
        CheckpointReferenceV1 as NewRef,
    )
    from slm_training.harness_core.lineage.records import DataSnapshot as NewSnap
    from slm_training.harnesses.model_build.checkpoint_reference import (
        CheckpointReferenceV1 as OldRef,
    )
    from slm_training.lineage.records import DataSnapshot as OldSnap

    assert OldSnap is NewSnap
    assert OldRef is NewRef


def test_versioning_registry_resolves_from_new_location() -> None:
    """The pre-extraction shim must resolve the same version as the new path.

    Asserts consistency between the two import paths, not a specific version
    number -- the registry version legitimately bumps whenever a
    harness_core file changes (see src/slm_training/resources/versions.json).
    """
    from slm_training.harness_core.versioning import (
        build_version_stamp as new_build_version_stamp,
    )
    from slm_training.versioning import build_version_stamp

    stamp = build_version_stamp("harness.core")
    new_stamp = new_build_version_stamp("harness.core")
    assert stamp["stamp_schema"] == "version_stamp/v1"
    assert stamp["components"]["harness.core"] == new_stamp["components"]["harness.core"]
    assert stamp["components"]["harness.core"].startswith("v")
