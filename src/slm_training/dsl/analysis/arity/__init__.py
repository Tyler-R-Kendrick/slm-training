"""Calculated arity and adaptive precision analyzer (CAP0-02)."""

from __future__ import annotations

from slm_training.dsl.analysis.arity.analyzer import ArityAnalyzer
from slm_training.dsl.analysis.arity.coding import (
    CodeVerification,
    build_mds_7_4_2_3,
    build_shortened_ternary_hamming_7_4_3,
    gilbert_greedy_guarantees,
    hamming_ball_volume,
    hamming_sphere_packing_holds,
    minimum_distance,
    singleton_upper_bound,
    smallest_injective_arity,
    verify_code,
)
from slm_training.dsl.analysis.arity.precision import (
    ResidualScaleMode,
    assert_geometric_only,
    balanced_ternary_levels,
    minimum_margin_trit_planes,
    ternary_ecoc_width,
)
from slm_training.dsl.analysis.arity.report import ArityReport, ContinuationSummary
from slm_training.dsl.analysis.arity.types import (
    AnalysisBounds,
    StateAtom,
    StateSignature,
    SupportOracle,
    SupportQuery,
    SupportResult,
    SupportVerdict,
)

__all__ = [
    "AnalysisBounds",
    "ArityAnalyzer",
    "ArityReport",
    "CodeVerification",
    "ContinuationSummary",
    "ResidualScaleMode",
    "StateAtom",
    "StateSignature",
    "SupportOracle",
    "SupportQuery",
    "SupportResult",
    "SupportVerdict",
    "assert_geometric_only",
    "balanced_ternary_levels",
    "build_mds_7_4_2_3",
    "build_shortened_ternary_hamming_7_4_3",
    "gilbert_greedy_guarantees",
    "hamming_ball_volume",
    "hamming_sphere_packing_holds",
    "minimum_distance",
    "minimum_margin_trit_planes",
    "singleton_upper_bound",
    "smallest_injective_arity",
    "ternary_ecoc_width",
    "verify_code",
]
