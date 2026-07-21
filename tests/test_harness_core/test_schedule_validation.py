"""Table-driven property tests for the generic schedule-validation primitives.

RSC-A06 (SLM-242). ``hypothesis`` is not a repo dependency (checked
``pyproject.toml``/``uv.lock``); these are hand-rolled parametrized/
table-driven equivalents covering the required shapes: valid vectors of
varying length/scale; empty/all-zero/negative/NaN/inf/duplicate/unsorted/
mismatched inputs; scale-invariant normalization; and one consistent typed
error class across every primitive.
"""

from __future__ import annotations

import math

import pytest

from slm_training.harness_core.schedule_validation import (
    ScheduleValidationError,
    exact_length_vector,
    finite_scalar,
    non_empty_vector,
    non_negative_scalar,
    normalized_probability_vector,
    paired_equal_length_sequences,
    positive_scalar,
    positive_sum_vector,
    strictly_increasing_sequence,
    supported_capability_requirement,
    unique_enum_sequence,
)


# --------------------------------------------------------------------------
# finite_scalar / non_negative_scalar / positive_scalar
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 0.0, 1, -1, 3.5, -3.5, 1e18, -1e18])
def test_finite_scalar_accepts_any_finite_number(value: float) -> None:
    assert finite_scalar(value, field="f") == pytest.approx(float(value))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_finite_scalar_rejects_nan_and_inf(value: float) -> None:
    with pytest.raises(ScheduleValidationError):
        finite_scalar(value, field="f")


def test_finite_scalar_rejects_non_numeric() -> None:
    with pytest.raises(ScheduleValidationError):
        finite_scalar("not-a-number", field="f")


@pytest.mark.parametrize("value", [0, 0.0, 1, 1000.0])
def test_non_negative_scalar_accepts_zero_and_positive(value: float) -> None:
    assert non_negative_scalar(value, field="f") == pytest.approx(float(value))


@pytest.mark.parametrize("value", [-1e-9, -1, -1000.0])
def test_non_negative_scalar_rejects_negative(value: float) -> None:
    with pytest.raises(ScheduleValidationError):
        non_negative_scalar(value, field="f")


@pytest.mark.parametrize("value", [0, 0.0, -1])
def test_positive_scalar_rejects_zero_and_negative(value: float) -> None:
    with pytest.raises(ScheduleValidationError):
        positive_scalar(value, field="f")


# --------------------------------------------------------------------------
# exact_length_vector / non_empty_vector
# --------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 2, 3, 8, 32])
def test_exact_length_vector_accepts_matching_length(length: int) -> None:
    vector = tuple(range(length))
    assert exact_length_vector(vector, length, field="f") == vector


@pytest.mark.parametrize(
    "vector,expected", [((1, 2), 1), ((1, 2), 3), ((), 1), ((1,), 0)]
)
def test_exact_length_vector_rejects_mismatched_length(vector, expected) -> None:
    """The silent min()/truncation pattern: a length mismatch must always raise."""
    with pytest.raises(ScheduleValidationError, match="length"):
        exact_length_vector(vector, expected, field="f")


def test_non_empty_vector_rejects_empty() -> None:
    with pytest.raises(ScheduleValidationError):
        non_empty_vector((), field="f")


# --------------------------------------------------------------------------
# positive_sum_vector / normalized_probability_vector
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector",
    [
        (1.0,),
        (1.0, 2.0),
        (0.0, 1.0),  # a zero entry is fine as long as the sum is positive
        (0.5, 1.0, 0.5),
        tuple(float(i) for i in range(1, 33)),  # varying length
        (1e-6, 1e6),  # varying scale
    ],
)
def test_positive_sum_vector_accepts_valid_vectors(vector) -> None:
    assert positive_sum_vector(vector, field="f") == vector


def test_positive_sum_vector_rejects_empty() -> None:
    with pytest.raises(ScheduleValidationError):
        positive_sum_vector((), field="f")


@pytest.mark.parametrize(
    "vector",
    [(0.0,), (0.0, 0.0), (0.0, 0.0, 0.0, 0.0)],
)
def test_positive_sum_vector_rejects_all_zero(vector) -> None:
    """The all-zero-erasure pattern: a non-empty, all-zero vector must raise."""
    with pytest.raises(ScheduleValidationError, match="all-zero"):
        positive_sum_vector(vector, field="f")


@pytest.mark.parametrize(
    "vector", [(-1.0,), (1.0, -1.0), (-0.5, 2.0), (1.0, math.nan), (1.0, math.inf)]
)
def test_positive_sum_vector_rejects_negative_nan_inf(vector) -> None:
    """The unvalidated-negative-value pattern (plus NaN/inf) must raise."""
    with pytest.raises(ScheduleValidationError):
        positive_sum_vector(vector, field="f")


@pytest.mark.parametrize(
    "base,scale",
    [
        ((1.0, 1.0, 2.0), 1.0),
        ((1.0, 1.0, 2.0), 2.0),
        ((1.0, 1.0, 2.0), 0.5),
        ((1.0, 1.0, 2.0), 100.0),
        ((3.0, 1.0), 7.0),
    ],
)
def test_normalized_probability_vector_is_invariant_under_positive_rescaling(
    base: tuple[float, ...], scale: float
) -> None:
    """Core invariant #3: normalized weighted means are invariant under
    positive uniform rescaling of the input weight vector."""
    scaled = tuple(x * scale for x in base)
    assert normalized_probability_vector(base, field="f") == pytest.approx(
        normalized_probability_vector(scaled, field="f")
    )


def test_normalized_probability_vector_sums_to_one() -> None:
    normalized = normalized_probability_vector((2.0, 2.0, 4.0), field="f")
    assert sum(normalized) == pytest.approx(1.0)
    assert normalized == pytest.approx((0.25, 0.25, 0.5))


# --------------------------------------------------------------------------
# strictly_increasing_sequence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector",
    [(1,), (1, 2), (1, 2, 3, 4), (32, 64, 96, 128, 192, 256, 384, 512)],
)
def test_strictly_increasing_sequence_accepts_valid(vector) -> None:
    assert strictly_increasing_sequence(vector, field="f") == vector


@pytest.mark.parametrize(
    "vector", [(), (1, 1), (2, 1), (1, 2, 2, 3), (64, 128, 192, 128)]
)
def test_strictly_increasing_sequence_rejects_unsorted_or_duplicate(vector) -> None:
    with pytest.raises(ScheduleValidationError):
        strictly_increasing_sequence(vector, field="f")


# --------------------------------------------------------------------------
# paired_equal_length_sequences
# --------------------------------------------------------------------------


def test_paired_equal_length_sequences_accepts_matching_lengths() -> None:
    pairs = (("a", (1.0, 2.0)), ("b", (3.0, 4.0)))
    result = paired_equal_length_sequences(pairs, field="f")
    assert result == (("a", (1.0, 2.0)), ("b", (3.0, 4.0)))


def test_paired_equal_length_sequences_accepts_empty() -> None:
    assert paired_equal_length_sequences((), field="f") == ()


def test_paired_equal_length_sequences_rejects_mismatched_length() -> None:
    pairs = (("a", (1.0, 2.0)), ("b", (3.0,)))
    with pytest.raises(ScheduleValidationError, match="one length"):
        paired_equal_length_sequences(pairs, field="f")


def test_paired_equal_length_sequences_rejects_duplicate_key() -> None:
    pairs = (("a", (1.0,)), ("a", (2.0,)))
    with pytest.raises(ScheduleValidationError, match="duplicate"):
        paired_equal_length_sequences(pairs, field="f")


def test_paired_equal_length_sequences_rejects_empty_score_vector() -> None:
    with pytest.raises(ScheduleValidationError):
        paired_equal_length_sequences((("a", ()),), field="f")


# --------------------------------------------------------------------------
# unique_enum_sequence
# --------------------------------------------------------------------------


def test_unique_enum_sequence_accepts_unique_known_values() -> None:
    allowed = frozenset({"uniform", "contiguous", "statement"})
    assert unique_enum_sequence(
        ("uniform", "contiguous"), field="f", allowed=allowed
    ) == ("uniform", "contiguous")


def test_unique_enum_sequence_rejects_duplicates() -> None:
    with pytest.raises(ScheduleValidationError, match="duplicate"):
        unique_enum_sequence(("uniform", "uniform"), field="f")


def test_unique_enum_sequence_rejects_unknown_values() -> None:
    allowed = frozenset({"uniform"})
    with pytest.raises(ScheduleValidationError, match="unknown"):
        unique_enum_sequence(("bogus",), field="f", allowed=allowed)


def test_unique_enum_sequence_rejects_empty() -> None:
    with pytest.raises(ScheduleValidationError):
        unique_enum_sequence((), field="f")


# --------------------------------------------------------------------------
# supported_capability_requirement
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "condition,capability_ok", [(False, False), (False, True), (True, True)]
)
def test_supported_capability_requirement_passes_when_capability_present_or_unneeded(
    condition: bool, capability_ok: bool
) -> None:
    supported_capability_requirement(
        condition=condition, capability_ok=capability_ok, field="f", reason="r"
    )  # must not raise


def test_supported_capability_requirement_raises_when_missing() -> None:
    """The non-recursive-capability-ignore pattern: a required-but-absent
    capability must raise, never silently no-op."""
    with pytest.raises(ScheduleValidationError, match="reason-text"):
        supported_capability_requirement(
            condition=True, capability_ok=False, field="f", reason="reason-text"
        )


# --------------------------------------------------------------------------
# One consistent typed error class (invariant #2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: finite_scalar(math.nan, field="f"),
        lambda: non_negative_scalar(-1, field="f"),
        lambda: positive_scalar(0, field="f"),
        lambda: exact_length_vector((1,), 2, field="f"),
        lambda: non_empty_vector((), field="f"),
        lambda: positive_sum_vector((0.0,), field="f"),
        lambda: normalized_probability_vector((0.0,), field="f"),
        lambda: strictly_increasing_sequence((2, 1), field="f"),
        lambda: paired_equal_length_sequences((("a", (1,)), ("b", (1, 2))), field="f"),
        lambda: unique_enum_sequence((1, 1), field="f"),
        lambda: supported_capability_requirement(
            condition=True, capability_ok=False, field="f", reason="r"
        ),
    ],
)
def test_every_primitive_raises_the_same_typed_error_class(call) -> None:
    with pytest.raises(ScheduleValidationError) as excinfo:
        call()
    assert excinfo.value.field == "f"
    assert isinstance(excinfo.value.reason, str) and excinfo.value.reason
    assert isinstance(excinfo.value, ValueError)
