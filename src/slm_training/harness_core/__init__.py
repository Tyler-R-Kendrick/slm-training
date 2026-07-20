"""Frozen, DSL-agnostic harness machinery (the harness core).

This package owns the stable contracts every harness builds on: component
version stamping, immutable lineage/provenance records, the checkpoint
reference schema, the ship-gate and promotion decision engines, scaling-law
math, and eval bookkeeping (run-honesty record schema, content-addressed eval
cache, score-policy registry).

Contract (docs/design/harness-core.md):

- **DSL-agnostic** — modules here never import ``slm_training.dsl``,
  ``models``, ``evals``, ``harnesses``, ``web``, or other DSL-coupled layers
  at module level (enforced by ``tests/test_harness_core``). DSL- or
  metric-specific behavior enters via parameters and callbacks (the ship-gate
  ``normalize_suite`` hook, the promotion ``gate_evaluator`` and
  ``hard_categories``).
- **Frozen** — any change under ``src/slm_training/harness_core/`` must bump
  the ``harness.core`` component (or carry a ``no-bump:`` history note) in
  ``src/slm_training/resources/versions.json``.
- **Import-light** — importing this package never pulls torch; heavy
  dependencies stay behind function-level imports.

Pre-extraction import paths (``slm_training.versioning``,
``slm_training.lineage``, ``slm_training.evals.record_schema``, …) remain
valid shims that alias these modules in ``sys.modules``.
"""

from slm_training.harness_core.checkpoint_reference import CheckpointReferenceV1
from slm_training.harness_core.efficiency_gain import efficiency_gain_lcb
from slm_training.harness_core.gate_engine import run_gate_checks
from slm_training.harness_core.lineage.records import canonical_json, content_sha
from slm_training.harness_core.promotion_engine import (
    PromotionCriteria,
    check_rank_stability,
)
from slm_training.harness_core.versioning import (
    build_version_stamp,
    component_version,
)

__all__ = [
    "CheckpointReferenceV1",
    "PromotionCriteria",
    "build_version_stamp",
    "canonical_json",
    "check_rank_stability",
    "component_version",
    "content_sha",
    "efficiency_gain_lcb",
    "run_gate_checks",
]
