"""Calculated arity and adaptive precision analyzer (CAP0-02)."""

from __future__ import annotations

from slm_training.dsl.analysis.arity.analyzer import ArityAnalyzer
from slm_training.dsl.analysis.arity.certificate import (
    ArityCertificate,
    ArityCertificateBundle,
    ArityProvenance,
    ArityResult,
    ConstraintFrame,
    EstimatedEvidence,
    EvidenceKind,
    ExactEvidence,
    certificate_digest,
    exact_certificate_from_report,
)
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
from slm_training.dsl.analysis.arity.profiles import (
    AnalysisProfile,
    OPENVUI_CAP_V1,
    get_profile,
    register_profile,
)
from slm_training.dsl.analysis.arity.report import ArityReport, CodingMetadata, ContinuationSummary
from slm_training.dsl.analysis.arity.state_graph import (
    STATE_GRAPH_VERSION,
    GraphEdge,
    GraphNode,
    StateFingerprint,
    StateGraph,
    StateGraphReport,
)
from slm_training.dsl.analysis.arity.suggest import RobustArm, suggest_robust_arms, smallest_feasible_alphabet
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
    "AnalysisProfile",
    "ArityAnalyzer",
    "ArityCertificate",
    "ArityCertificateBundle",
    "ArityProvenance",
    "ArityReport",
    "ArityResult",
    "CodeVerification",
    "CodingMetadata",
    "ConstraintFrame",
    "ContinuationSummary",
    "EstimatedEvidence",
    "EvidenceKind",
    "ExactEvidence",
    "GraphEdge",
    "GraphNode",
    "OPENVUI_CAP_V1",
    "ResidualScaleMode",
    "RobustArm",
    "STATE_GRAPH_VERSION",
    "StateAtom",
    "StateFingerprint",
    "StateGraph",
    "StateGraphReport",
    "StateSignature",
    "SupportOracle",
    "SupportQuery",
    "SupportResult",
    "SupportVerdict",
    "assert_geometric_only",
    "balanced_ternary_levels",
    "build_mds_7_4_2_3",
    "build_shortened_ternary_hamming_7_4_3",
    "certificate_digest",
    "exact_certificate_from_report",
    "get_profile",
    "gilbert_greedy_guarantees",
    "hamming_ball_volume",
    "hamming_sphere_packing_holds",
    "minimum_distance",
    "minimum_margin_trit_planes",
    "register_profile",
    "singleton_upper_bound",
    "smallest_feasible_alphabet",
    "smallest_injective_arity",
    "suggest_robust_arms",
    "ternary_ecoc_width",
    "verify_code",
]
