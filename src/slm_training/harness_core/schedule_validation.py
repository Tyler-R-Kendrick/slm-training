"""Fail-closed validation primitives for numeric weight/schedule vectors.

RSC-A06 (SLM-242): generalizes the deep-supervision length/zero-sum fix into a
small set of composable, DSL-agnostic primitives every numeric
weight/schedule/mixture/bucket vector in the model-build config surface can be
checked against. Primitives never silently coerce, truncate, pad, renormalize,
or drop values a caller did not explicitly ask for — they either return a
typed, read-only (tuple) view of the input or raise :class:`ScheduleValidationError`
naming the offending field and reason.

Docs: docs/design/rsc-a06-numeric-schedule-validation-20260721.md
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

__all__ = [
    "ScheduleValidationError",
    "finite_scalar",
    "non_negative_scalar",
    "positive_scalar",
    "exact_length_vector",
    "non_empty_vector",
    "positive_sum_vector",
    "normalized_probability_vector",
    "strictly_increasing_sequence",
    "paired_equal_length_sequences",
    "unique_enum_sequence",
    "supported_capability_requirement",
]


class ScheduleValidationError(ValueError):
    """A numeric weight/schedule/mixture vector failed a fail-closed check.

    Carries the offending ``field`` name and a human-readable ``reason`` so
    callers can surface one consistent, typed error across every vector-shaped
    config surface (recursive-depth weights, grammar LTR stages, diffusion
    policies/buckets, slot-component class weights/priors, margin family
    weights, ...).
    """

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


def finite_scalar(value: object, *, field: str) -> float:
    """Return ``value`` as a finite float or raise. Rejects NaN/inf/non-numeric."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ScheduleValidationError(field, f"must be numeric, got {value!r}") from exc
    if math.isnan(v) or math.isinf(v):
        raise ScheduleValidationError(field, f"must be finite, got {value!r}")
    return v


def non_negative_scalar(value: object, *, field: str) -> float:
    """Finite and ``>= 0``. Rejects the "unvalidated negative value" pattern."""
    v = finite_scalar(value, field=field)
    if v < 0.0:
        raise ScheduleValidationError(field, f"must be >= 0, got {v!r}")
    return v


def positive_scalar(value: object, *, field: str) -> float:
    """Finite and ``> 0``."""
    v = finite_scalar(value, field=field)
    if v <= 0.0:
        raise ScheduleValidationError(field, f"must be > 0, got {v!r}")
    return v


def exact_length_vector(
    vector: Sequence[object], expected_length: int, *, field: str
) -> tuple:
    """Length must equal ``expected_length`` exactly.

    Rejects the ``min(len(vector), len(outputs))`` silent-truncation pattern:
    a vector shorter *or* longer than the quantity it is meant to weight is a
    configuration error, never an implicit crop.
    """
    v = tuple(vector)
    if len(v) != expected_length:
        raise ScheduleValidationError(
            field,
            f"length {len(v)} must equal required length {expected_length} "
            "(no implicit truncation/padding)",
        )
    return v


def non_empty_vector(vector: Sequence[object], *, field: str) -> tuple:
    """Reject the empty vector (as opposed to the sentinel "feature off" value)."""
    v = tuple(vector)
    if not v:
        raise ScheduleValidationError(field, "must be non-empty")
    return v


def positive_sum_vector(vector: Sequence[object], *, field: str) -> tuple[float, ...]:
    """Non-empty, every element finite and ``>= 0``, and the sum is ``> 0``.

    Rejects the "all-zero erasure" pattern: a non-empty vector whose entries
    are all zero silently disables the feature it configures with no signal.
    Individual zero entries remain valid (invariant: a zero weight disables
    only its own contribution, not the whole feature) as long as at least one
    entry is strictly positive.
    """
    v = tuple(non_negative_scalar(x, field=field) for x in non_empty_vector(vector, field=field))
    if sum(v) <= 0.0:
        raise ScheduleValidationError(
            field, f"sum of weights must be > 0 (all-zero vector silently disables), got {v!r}"
        )
    return v


def normalized_probability_vector(
    vector: Sequence[object], *, field: str, atol: float = 1e-6
) -> tuple[float, ...]:
    """Positive-sum vector rescaled to sum to 1.0.

    Invariant under positive uniform rescaling of the input: ``(2, 2, 4)`` and
    ``(1, 1, 2)`` normalize to the same output.
    """
    v = positive_sum_vector(vector, field=field)
    total = sum(v)
    normalized = tuple(x / total for x in v)
    residual = abs(sum(normalized) - 1.0)
    if residual > atol:
        raise ScheduleValidationError(
            field, f"normalized weights must sum to 1.0 (+/-{atol}), got {sum(normalized)!r}"
        )
    return normalized


def strictly_increasing_sequence(vector: Sequence[object], *, field: str) -> tuple:
    """Non-empty and strictly increasing (rejects unsorted values and duplicates)."""
    v = non_empty_vector(vector, field=field)
    for a, b in zip(v, v[1:]):
        if not (a < b):  # type: ignore[operator]
            raise ScheduleValidationError(
                field, f"must be strictly increasing (no duplicates/unsorted), got {v!r}"
            )
    return v


def paired_equal_length_sequences(
    pairs: Iterable[tuple[object, Sequence[object]]], *, field: str
) -> tuple[tuple[object, tuple[float, ...]], ...]:
    """Every ``(key, vector)`` pair has a unique key and all vectors share one length.

    Used for lexeme/span prior tables where every key's score vector must line
    up with the same fixed class ordering.
    """
    items = list(pairs)
    if not items:
        return ()
    seen_keys: set = set()
    lengths: set[int] = set()
    normalized: list[tuple[object, tuple[float, ...]]] = []
    for key, vector in items:
        if key in seen_keys:
            raise ScheduleValidationError(field, f"duplicate key {key!r}")
        seen_keys.add(key)
        vec = tuple(finite_scalar(x, field=field) for x in vector)
        if not vec:
            raise ScheduleValidationError(field, f"score vector for key {key!r} must be non-empty")
        lengths.add(len(vec))
        normalized.append((key, vec))
    if len(lengths) > 1:
        raise ScheduleValidationError(
            field, f"paired vectors must share one length, got lengths {sorted(lengths)}"
        )
    return tuple(normalized)


def unique_enum_sequence(
    vector: Sequence[object], *, field: str, allowed: frozenset | None = None
) -> tuple:
    """Non-empty, no duplicates, and (optionally) every value in ``allowed``."""
    v = non_empty_vector(vector, field=field)
    if len(set(v)) != len(v):
        dupes = sorted({str(x) for x in v if v.count(x) > 1})
        raise ScheduleValidationError(field, f"duplicate values: {dupes}")
    if allowed is not None:
        unknown = sorted(str(x) for x in set(v) - allowed)
        if unknown:
            raise ScheduleValidationError(
                field, f"unknown values {unknown}; allowed={sorted(allowed)}"
            )
    return v


def supported_capability_requirement(
    *, condition: bool, capability_ok: bool, field: str, reason: str
) -> None:
    """Raise when ``condition`` (a non-empty/explicitly-set vector or weight)
    is true but the active architecture/config does not support it.

    This is the generic form of "non-empty recursive depth weights requires a
    denoiser that supports recursive_outputs": no feature may be silently
    ignored because the active architecture lacks the capability it needs.
    """
    if condition and not capability_ok:
        raise ScheduleValidationError(field, reason)
