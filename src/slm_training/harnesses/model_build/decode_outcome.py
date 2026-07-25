"""SLM-303 (LAR1-05): decode-outcome taxonomy for budget/interference audits.

Every decode attempt lands in exactly one of ``DECODE_OUTCOMES``. The point of
the taxonomy is to stop laundering runtime/harness artifacts (timeouts,
fallbacks, harness exceptions) into model-quality claims: a fallback output is
never a model success, and a timeout says something about the budget, not the
model.

Classification precedence is strict and total:

    harness_error > runtime_timeout > fallback_output > model_abstain
        > model_invalid / model_valid
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

DECODE_OUTCOMES: tuple[str, ...] = (
    "model_valid",
    "model_invalid",
    "model_abstain",
    "runtime_timeout",
    "fallback_output",
    "harness_error",
)

MODEL_VALID, MODEL_INVALID, MODEL_ABSTAIN = DECODE_OUTCOMES[0:3]
RUNTIME_TIMEOUT, FALLBACK_OUTPUT, HARNESS_ERROR = DECODE_OUTCOMES[3:6]

# DecodeStats counter names that mean "the emitted output came from a
# non-model fallback path". Mirrors the eval_runner fallback_count set.
FALLBACK_COUNTER_NAMES: tuple[str, ...] = (
    "unconstrained_retries",
    "compiler_fallbacks",
    "seeded_fallbacks",
    "template_fallback_count",
    "certified_fallbacks",
)


def fallback_counter_total(stats: object | None) -> int:
    """Total fallback counter value across a DecodeStats-like row (0 if none)."""
    if stats is None:
        return 0
    if isinstance(stats, Mapping):
        return sum(int(stats.get(name) or 0) for name in FALLBACK_COUNTER_NAMES)
    return sum(int(getattr(stats, name, 0) or 0) for name in FALLBACK_COUNTER_NAMES)


def classify_decode_outcome(
    *,
    parse_ok: bool | None,
    error: str | None = None,
    fallback_counters: int = 0,
    timed_out: bool = False,
    abstained: bool = False,
    harness_exception: bool = False,
) -> str:
    """Classify one decode attempt into the taxonomy.

    Precedence is strict: a harness exception dominates everything; a timeout
    is a budget artifact, not a model verdict; a fallback output is never a
    model success even when it happens to parse; an abstention (empty output
    without timeout/fallback) outranks a parse verdict.
    """
    if harness_exception:
        return HARNESS_ERROR
    if timed_out:
        return RUNTIME_TIMEOUT
    if fallback_counters > 0:
        return FALLBACK_OUTPUT
    if abstained:
        return MODEL_ABSTAIN
    # parse_ok None = parse not evaluated; a produced candidate with no
    # contrary evidence is a model output, not a failure verdict.
    return MODEL_INVALID if parse_ok is False else MODEL_VALID


@dataclass
class DecodeOutcomeRecordV1:
    """One decode attempt with its budget, evidence, and classified outcome."""

    request_id: str
    outcome: str
    stop_reason: str
    budget: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float | None = None
    forwards: int | None = None
    verifier_calls: int | None = None
    fallback_used: bool = False
    detail: str | None = None

    schema: str = "decode_outcome_record/v1"

    def __post_init__(self) -> None:
        if self.outcome not in DECODE_OUTCOMES:
            raise ValueError(f"unknown decode outcome {self.outcome!r}")
        if self.outcome == FALLBACK_OUTPUT and not self.fallback_used:
            raise ValueError("fallback_output outcome requires fallback_used=True")
        if self.outcome == MODEL_VALID and self.fallback_used:
            raise ValueError("a fallback output may never classify as model_valid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecodeOutcomeRecordV1":
        return cls(
            request_id=str(data["request_id"]),
            outcome=str(data["outcome"]),
            stop_reason=str(data["stop_reason"]),
            budget=dict(data.get("budget") or {}),
            elapsed_ms=data.get("elapsed_ms"),
            forwards=data.get("forwards"),
            verifier_calls=data.get("verifier_calls"),
            fallback_used=bool(data.get("fallback_used", False)),
            detail=data.get("detail"),
        )


def outcome_counts(outcomes: list[str]) -> dict[str, int]:
    """Per-taxonomy counts with every key present (zeros included)."""
    counts = {name: 0 for name in DECODE_OUTCOMES}
    for outcome in outcomes:
        if outcome not in counts:
            raise ValueError(f"unknown decode outcome {outcome!r}")
        counts[outcome] += 1
    return counts


__all__ = [
    "DECODE_OUTCOMES",
    "FALLBACK_COUNTER_NAMES",
    "DecodeOutcomeRecordV1",
    "classify_decode_outcome",
    "fallback_counter_total",
    "outcome_counts",
]
