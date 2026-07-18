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
]
