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
]
