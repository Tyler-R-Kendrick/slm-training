"""Legacy DSL-pack adapter for the canonical arithmetic grammar owner."""

from slm_training.dsl.packs.types import DSLPack, PlaceholderPolicy
from slm_training.dsl.arith_sketch import (
    evaluate_answer,
    _canonicalize,
    _canonical_equal,
    _corpus_generator,
    _scope_check,
    _validity_oracle,
)

__all__ = ["build_pack", "evaluate_answer"]


def build_pack() -> DSLPack:
    return DSLPack(
        id="arith-sketch",
        description="Straight-line arithmetic sketch DSL — checkable-answer "
        "reasoning traces (G4)",
        grammar="arith-sketch",
        canonicalize=_canonicalize,
        canonical_equal=_canonical_equal,
        validity_oracle=_validity_oracle,
        corpus_generator=_corpus_generator,
        scope_check=_scope_check,
        placeholders=PlaceholderPolicy(
            is_placeholder=lambda value: False,
            extract=lambda source: [],
        ),
        notes=(
            "identity canonicalizer — no codec round-trip normal form",
            "oracle = deterministic straight-line evaluator; validity and "
            "answer scoring share one code path",
            "no placeholder routing — answers are computed, not copied",
        ),
    )
