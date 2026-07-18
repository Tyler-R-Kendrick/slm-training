"""CAP0-02 exact arity analyzer for the bounded arith-sketch fixture.

A deterministic, Torch-free pipeline that turns a declared bounded frame into a
replayable arity certificate:

    enumerate canonical ASTs -> validate/type-reject -> prefix trie
    -> acyclic minimisation -> branching / completion / K^d capacity.

Public API is re-exported here for convenience; see the submodules for detail
and ``docs/design/cap0-02-arity-analyzer-20260718.md`` for the certified counts
and the honesty boundary (external CAP0-01 estimates are **not** reproduced).
"""

from __future__ import annotations

from slm_training.dsl.analysis.arity.canonical import (
    CanonicalProgram,
    NUMBER_CLASS,
    is_type_valid,
    materialize,
    program_actions,
    program_from_source,
)
from slm_training.dsl.analysis.arity.minimize import MinimizedDFA, minimize
from slm_training.dsl.analysis.arity.report import (
    CODEC_VERSION,
    PARSER_VERSION,
    SCHEMA_VERSION,
    SIGNATURE_VERSION,
    AnalysisBounds,
    ExactArityReport,
    SchemaError,
    StateSignature,
    analyze,
    min_alphabet_for_capacity,
)
from slm_training.dsl.analysis.arity.state_graph import (
    FIXTURES,
    EnumerationBounds,
    build_trie,
    enumerate_programs,
)

# CAP0-03 (SLM-79): exact robust-code constructions + residual-precision
# reference functions. These are standalone, Torch-free helpers layered on the
# canonical CAP0-02 API above (``coding.smallest_injective_arity`` delegates to
# ``report.min_alphabet_for_capacity``); they never re-introduce the retired
# ``analyzer``/``explorer``/``types`` stub. See
# ``docs/design/cap0-03-coding-precision-20260718.md``.
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
from slm_training.dsl.analysis.arity.suggest import (
    RobustArm,
    smallest_feasible_alphabet,
    suggest_robust_arms,
)

# CAP0-04 (SLM-80): exact-vs-estimated arity certificates + provenance. A
# versioned certificate schema layered on the canonical CAP0-02 report (via the
# ``ReportView`` bridge) and the standalone CAP0-03 coding API; it never
# re-introduces the retired ``analyzer``/``explorer``/``types`` stub or a
# report-attached ``CodingMetadata``. Builds on the CAP0-02/03 design docs
# (``cap0-02-arity-analyzer-20260718.md`` / ``cap0-03-coding-precision-20260718.md``).
from slm_training.dsl.analysis.arity.certificate import (
    ArityCertificate,
    ArityCertificateBundle,
    ArityProvenance,
    ArityResult,
    ConstraintFrame,
    EstimatedEvidence,
    EvidenceKind,
    ExactEvidence,
    ReportView,
    certificate_digest,
    exact_certificate_from_report,
    report_view,
)
from slm_training.dsl.analysis.arity.render import (
    one_line_summary,
    to_csv,
    to_markdown,
)
from slm_training.dsl.analysis.arity.conditional_rate import (
    ConditionalRateReport,
    FanoBound,
    PosteriorEffectiveSupport,
    RateDistortionPoint,
    analyze_conditional_rate,
    blahut_arimoto_rate_distortion,
    conditional_entropy,
    entropy,
    fano_lower_bound,
    mutual_information,
    posterior_effective_support,
)
from slm_training.dsl.analysis.arity.task_quotient import (
    AlignedActionRecord,
    ConfusabilityGraph,
    ColoringResult,
    QuotientReport,
    TaskDistortionSpec,
    analyze_task_quotient,
    build_confusability_graph,
    build_state_profiles,
    capacity_feasible,
    color_graph,
    refine_quotient,
)

__all__ = [
    "AnalysisBounds",
    "CanonicalProgram",
    "CODEC_VERSION",
    "EnumerationBounds",
    "ExactArityReport",
    "FIXTURES",
    "MinimizedDFA",
    "NUMBER_CLASS",
    "PARSER_VERSION",
    "SCHEMA_VERSION",
    "SIGNATURE_VERSION",
    "SchemaError",
    "StateSignature",
    "analyze",
    "build_trie",
    "enumerate_programs",
    "is_type_valid",
    "materialize",
    "min_alphabet_for_capacity",
    "minimize",
    "program_actions",
    "program_from_source",
    # --- CAP0-03: coding-theory + residual precision (SLM-79) ---
    "CodeVerification",
    "ResidualScaleMode",
    "RobustArm",
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
    "smallest_feasible_alphabet",
    "smallest_injective_arity",
    "suggest_robust_arms",
    "ternary_ecoc_width",
    "verify_code",
    # --- CAP0-04: exact-vs-estimated arity certificates (SLM-80) ---
    "ArityCertificate",
    "ArityCertificateBundle",
    "ArityProvenance",
    "ArityResult",
    "ConstraintFrame",
    "EstimatedEvidence",
    "EvidenceKind",
    "ExactEvidence",
    "ReportView",
    "certificate_digest",
    "exact_certificate_from_report",
    "one_line_summary",
    "report_view",
    "to_csv",
    "to_markdown",
    # --- CAP1-03: task-confusability graph / neural state quotient (SLM-83) ---
    "AlignedActionRecord",
    "ConfusabilityGraph",
    "ColoringResult",
    "QuotientReport",
    "TaskDistortionSpec",
    "analyze_task_quotient",
    "build_confusability_graph",
    "build_state_profiles",
    "capacity_feasible",
    "color_graph",
    "refine_quotient",
    # --- CAP1-04: conditional task rate / Fano / RD (SLM-84) ---
    "ConditionalRateReport",
    "FanoBound",
    "PosteriorEffectiveSupport",
    "RateDistortionPoint",
    "analyze_conditional_rate",
    "blahut_arimoto_rate_distortion",
    "conditional_entropy",
    "entropy",
    "fano_lower_bound",
    "mutual_information",
    "posterior_effective_support",
]
