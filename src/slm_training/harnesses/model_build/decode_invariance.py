"""Paired decode-outcome attribution for the decode-invariance audit (EFS0-02).

Given two decode outcomes for the same frozen checkpoint × example under two
decode paths, classify the disagreement into a stable taxonomy and, over a
suite, decide whether the paths are invariant within preregistered equivalence
bands. This is what turns "byte-identical weights, parse 0 → 1.0 after a decoder
change" (the E288 defect) into a machine-checkable *decoder-sensitive* verdict.

Torch-free: it operates on already-computed per-example outcomes, so it can be
unit-tested (including an intentional decoder-defect regression) without a GPU
or a real checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

__all__ = [
    "DISAGREEMENT_CLASSES",
    "EQUIVALENCE_BANDS",
    "DecodeOutcome",
    "DisagreementClass",
    "classify_disagreement",
    "pair_disagreement_summary",
    "is_invariant",
]

DisagreementClass = Literal[
    "agree",
    "surface_only",
    "syntax_placeholder",
    "semantic_binding",
    "empty_vs_populated",
    "timeout_fallback",
    "exact_choice_derivation",
]

# Ordered most-decision-relevant first; the first matching class wins.
DISAGREEMENT_CLASSES: tuple[DisagreementClass, ...] = (
    "empty_vs_populated",
    "semantic_binding",
    "syntax_placeholder",
    "timeout_fallback",
    "exact_choice_derivation",
    "surface_only",
    "agree",
)

# Preregistered equivalence bands (EFS0-02): declare a checkpoint invariant only
# when every paired metric delta and the paired disagreement rate stay within
# these. Recorded before execution; do not loosen to green a run.
EQUIVALENCE_BANDS: dict[str, float] = {
    "max_abs_metric_delta": 0.01,
    "max_disagreement_rate": 0.01,
}

# Classes that count as a *substantive* (non-surface) disagreement.
_SUBSTANTIVE = frozenset(
    {
        "empty_vs_populated",
        "semantic_binding",
        "syntax_placeholder",
        "timeout_fallback",
        "exact_choice_derivation",
    }
)


@dataclass(frozen=True)
class DecodeOutcome:
    """One checkpoint × decode-path × example decode outcome (metrics only)."""

    example_id: str
    parse_ok: bool
    meaningful: bool
    non_empty: bool
    canonical_output: str | None = None
    raw_output: str | None = None
    fallback: bool = False
    timeout: bool = False
    error_class: str | None = None
    choice_derivation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_disagreement(a: DecodeOutcome, b: DecodeOutcome) -> DisagreementClass:
    """Classify the disagreement between two outcomes (most-severe class wins)."""
    if a.example_id != b.example_id:
        raise ValueError(
            f"cannot compare different examples {a.example_id!r} vs {b.example_id!r}"
        )
    if a.non_empty != b.non_empty:
        return "empty_vs_populated"
    if a.meaningful != b.meaningful:
        return "semantic_binding"
    if a.parse_ok != b.parse_ok:
        return "syntax_placeholder"
    if a.timeout != b.timeout or a.fallback != b.fallback:
        return "timeout_fallback"
    if a.choice_derivation != b.choice_derivation:
        return "exact_choice_derivation"
    if (
        a.canonical_output is not None
        and a.canonical_output == b.canonical_output
        and a.raw_output != b.raw_output
    ):
        return "surface_only"
    return "agree"


def _rates(outcomes: Sequence[DecodeOutcome]) -> dict[str, float]:
    n = len(outcomes) or 1
    return {
        "parse_rate": sum(o.parse_ok for o in outcomes) / n,
        "meaningful_program_rate": sum(o.meaningful for o in outcomes) / n,
        "non_empty_rate": sum(o.non_empty for o in outcomes) / n,
    }


def pair_disagreement_summary(
    path_a: str,
    path_b: str,
    outcomes_a: Sequence[DecodeOutcome],
    outcomes_b: Sequence[DecodeOutcome],
    *,
    bands: Mapping[str, float] = EQUIVALENCE_BANDS,
) -> dict[str, Any]:
    """Compare two decode paths over one checkpoint's paired outcomes.

    Returns paired metric deltas, a per-class confusion count, the paired
    disagreement rate, and an honest ``decoder_sensitive`` verdict. Both
    sequences must cover the same examples in the same order.
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("paired outcome sequences must be the same length")
    counts: dict[str, int] = {cls: 0 for cls in DISAGREEMENT_CLASSES}
    substantive = 0
    for a, b in zip(outcomes_a, outcomes_b):
        cls = classify_disagreement(a, b)
        counts[cls] += 1
        if cls in _SUBSTANTIVE:
            substantive += 1
    n = len(outcomes_a) or 1
    rates_a, rates_b = _rates(outcomes_a), _rates(outcomes_b)
    deltas = {k: round(rates_b[k] - rates_a[k], 6) for k in rates_a}
    disagreement_rate = substantive / n
    max_delta = max((abs(v) for v in deltas.values()), default=0.0)
    invariant = max_delta <= bands["max_abs_metric_delta"] and (
        disagreement_rate <= bands["max_disagreement_rate"]
    )
    return {
        "path_a": path_a,
        "path_b": path_b,
        "n": len(outcomes_a),
        "rates_a": rates_a,
        "rates_b": rates_b,
        "metric_deltas": deltas,
        "max_abs_metric_delta": round(max_delta, 6),
        "disagreement_counts": counts,
        "substantive_disagreements": substantive,
        "disagreement_rate": round(disagreement_rate, 6),
        "equivalence_bands": dict(bands),
        "invariant": invariant,
        "decoder_sensitive": not invariant,
    }


def is_invariant(summary: Mapping[str, Any]) -> bool:
    """True when a pair summary declares the two paths invariant."""
    return bool(summary.get("invariant", False))
