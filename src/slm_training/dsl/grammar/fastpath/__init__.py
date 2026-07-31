"""Grammar-baked deterministic fast-path for OpenUI decode / train aux."""

from typing import Any

from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine, engine_for_dsl
from slm_training.dsl.grammar.fastpath.force_emit import draft_forced_ids, force_next_token_id
from slm_training.dsl.grammar.fastpath.maskgit_constrain import admit_fill
from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
    build_completion_forest,
)


def __getattr__(name: str) -> Any:
    """Load the optional PyTorch gate only when explicitly requested."""
    if name == "FastPathGate":
        from slm_training.dsl.grammar.fastpath.gate import FastPathGate

        return FastPathGate
    if name in {
        "DomainSnapshot",
        "compare_domain_parity",
        "production_domain_snapshot",
        "require_certified_static_projection",
        "e9_warm_hard_prefix_classification",
    }:
        from slm_training.dsl.grammar.fastpath import static_control_domain as _scd

        return getattr(_scd, name)
    if name in {
        "ResidualSupportResult",
        "joint_multi_hole_support",
        "rank_inside_legal_residual",
    }:
        from slm_training.dsl.grammar.fastpath import residual_support as _rs

        return getattr(_rs, name)
    raise AttributeError(name)

__all__ = [
    "FastPathGate",
    "OpenUIIncrementalEngine",
    "CompletionForest",
    "CompletionPath",
    "DomainSnapshot",
    "ResidualSupportResult",
    "admit_fill",
    "compare_domain_parity",
    "draft_forced_ids",
    "e9_warm_hard_prefix_classification",
    "engine_for_dsl",
    "force_next_token_id",
    "build_completion_forest",
    "joint_multi_hole_support",
    "production_domain_snapshot",
    "rank_inside_legal_residual",
    "require_certified_static_projection",
]
