"""Fail-closed numeric/schedule validation for TwoTower configs.

SLM-242 (RSC-A06): every weight and schedule vector used by the TwoTower
training, decode, curriculum, and auxiliary-loss surfaces is validated before
execution.  Invalid configs raise ``NumericValidationError`` with a field path
instead of silently truncating, zeroing, or ignoring a vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields as _fields
from typing import Any, Iterable, Sequence

from slm_training.levers import (
    MAX_RUN_MINUTES,
    lever_configuration_errors,
)


class NumericValidationError(ValueError):
    """Raised when a numeric config field violates its documented contract."""

    def __init__(self, field: str, value: Any, reason: str) -> None:
        super().__init__(f"{field}: {reason} (got {value!r})")
        self.field = field
        self.value = value
        self.reason = reason


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def finite_scalar(field: str, value: float | None) -> float | None:
    """Require a scalar to be a finite real number (None is passed through)."""
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise NumericValidationError(field, value, "must be a real number")
    if math.isnan(value) or math.isinf(value):
        raise NumericValidationError(field, value, "must be finite")
    return float(value)


def non_negative_scalar(field: str, value: float | None) -> float | None:
    """Require ``value >= 0`` (None passes through)."""
    value = finite_scalar(field, value)
    if value is None:
        return None
    if value < 0:
        raise NumericValidationError(field, value, "must be non-negative")
    return value


def positive_scalar(field: str, value: float | int | None) -> float | int | None:
    """Require ``value > 0`` (None passes through)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericValidationError(field, value, "must be a real number")
    if math.isnan(value) or math.isinf(value):
        raise NumericValidationError(field, value, "must be finite")
    if value <= 0:
        raise NumericValidationError(field, value, "must be positive")
    return value


def interval_scalar(
    field: str, value: float | None, low: float, high: float
) -> float | None:
    """Require ``low <= value <= high`` (None passes through)."""
    value = finite_scalar(field, value)
    if value is None:
        return None
    if not (low <= value <= high):
        raise NumericValidationError(
            field, value, f"must be in [{low}, {high}]"
        )
    return value


def exact_length_vector(
    field: str, vector: Sequence[float], length: int
) -> Sequence[float]:
    """Require ``len(vector) == length``."""
    if len(vector) != length:
        raise NumericValidationError(
            field, vector, f"must have length {length}, got {len(vector)}"
        )
    return vector


def non_empty_vector(field: str, vector: Sequence[Any]) -> Sequence[Any]:
    """Require a non-empty sequence."""
    if len(vector) == 0:
        raise NumericValidationError(field, vector, "must be non-empty")
    return vector


def finite_non_negative_vector(
    field: str, vector: Sequence[float]
) -> Sequence[float]:
    """Require every element to be finite and non-negative."""
    for i, x in enumerate(vector):
        finite_scalar(f"{field}[{i}]", x)
        if x < 0:
            raise NumericValidationError(
                f"{field}[{i}]", x, "must be non-negative"
            )
    return vector


def positive_sum_vector(field: str, vector: Sequence[float]) -> Sequence[float]:
    """Require a non-empty vector of finite non-negative values with positive sum."""
    non_empty_vector(field, vector)
    finite_non_negative_vector(field, vector)
    total = sum(vector)
    if total <= 0:
        raise NumericValidationError(
            field, vector, "must have a positive sum (all-zero is not allowed)"
        )
    return vector


def normalized_probability_vector(
    field: str, vector: Sequence[float], tol: float = 1e-6
) -> Sequence[float]:
    """Require finite non-negative values that sum to 1 (within ``tol``)."""
    finite_non_negative_vector(field, vector)
    total = sum(vector)
    if not (1.0 - tol <= total <= 1.0 + tol):
        raise NumericValidationError(
            field, vector, f"must sum to 1 (sum={total})"
        )
    return vector


def strictly_increasing_sequence(
    field: str, vector: Sequence[int]
) -> Sequence[int]:
    """Require strictly increasing positive integers."""
    for i, x in enumerate(vector):
        if not isinstance(x, int) or isinstance(x, bool):
            raise NumericValidationError(
                f"{field}[{i}]", x, "must be an integer"
            )
        if x <= 0:
            raise NumericValidationError(
                f"{field}[{i}]", x, "must be positive"
            )
    for i in range(1, len(vector)):
        if vector[i] <= vector[i - 1]:
            raise NumericValidationError(
                field, vector, "must be strictly increasing"
            )
    return vector


def unique_enum_sequence(
    field: str, vector: Sequence[str], allowed: Iterable[str]
) -> Sequence[str]:
    """Require every value to be one of ``allowed`` with no duplicates."""
    allowed_set = set(allowed)
    seen: set[str] = set()
    for i, x in enumerate(vector):
        if x not in allowed_set:
            raise NumericValidationError(
                f"{field}[{i}]", x, f"must be one of {sorted(allowed_set)}"
            )
        if x in seen:
            raise NumericValidationError(
                f"{field}[{i}]", x, "duplicate value"
            )
        seen.add(x)
    return vector


def paired_equal_length_sequences(
    field_a: str,
    a: Sequence[Any],
    field_b: str,
    b: Sequence[Any],
) -> None:
    """Require two sequences to have equal length."""
    if len(a) != len(b):
        raise NumericValidationError(
            f"{field_a}/{field_b}", (a, b),
            f"must have equal length ({len(a)} vs {len(b)})",
        )


def supported_capability_requirement(
    field: str,
    value: Any,
    enabled: bool,
    capability: str,
) -> None:
    """Fail closed when a non-empty/enabled config requires an unsupported capability."""
    if not enabled:
        return
    is_non_empty = False
    if isinstance(value, (str, bool)):
        is_non_empty = bool(value)
    elif isinstance(value, Sequence):
        is_non_empty = len(value) > 0
    elif value is not None:
        is_non_empty = True
    if is_non_empty:
        raise NumericValidationError(
            field, value, f"requires unsupported capability {capability}"
        )


@dataclass(frozen=True)
class ValidatedNumericSchedule:
    """Read-only view of a validated schedule/weight vector."""

    field: str
    raw: tuple[float, ...]
    normalized: tuple[float, ...]
    sum: float


@dataclass
class NumericScheduleValidationReportV1:
    """Machine-readable report listing every audited numeric field."""

    schema: str = "numeric_schedule_validation_report/v1"
    valid: bool = True
    field_results: dict[str, dict[str, Any]] = None  # type: ignore[assignment]
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.field_results is None:
            object.__setattr__(self, "field_results", {})
        if self.errors is None:
            object.__setattr__(self, "errors", [])

    def record(self, field: str, ok: bool, message: str = "") -> None:
        self.field_results[field] = {"ok": ok, "message": message}
        if not ok:
            self.valid = False
            self.errors.append(f"{field}: {message}")


def validate_schedule_vector(
    field: str,
    vector: Sequence[float],
    require_non_empty: bool = False,
    require_positive_sum: bool = False,
    expected_length: int | None = None,
) -> ValidatedNumericSchedule:
    """Validate a generic weight/schedule vector."""
    if require_non_empty or require_positive_sum:
        non_empty_vector(field, vector)
    if expected_length is not None:
        exact_length_vector(field, vector, expected_length)
    finite_non_negative_vector(field, vector)
    total = sum(vector)
    if require_positive_sum and total <= 0:
        raise NumericValidationError(
            field, vector, "must have a positive sum"
        )
    normalized = tuple(x / total if total > 0 else 0.0 for x in vector)
    return ValidatedNumericSchedule(
        field=field,
        raw=tuple(vector),
        normalized=normalized,
        sum=total,
    )




def _type_is_tuple(field_type: str) -> bool:
    return "tuple" in field_type and "..." in field_type


def _type_is_numeric_scalar(field_type: str) -> bool:
    return "float" in field_type or "int" in field_type


def validate_numeric_config(
    cfg: Any,
    *,
    context: str = "config",
    require_trained_decode: bool = False,
) -> NumericScheduleValidationReportV1:
    """Validate every weight/schedule/numeric field on ``cfg``.

    Uses dataclass field introspection so new *_weight fields are gated
    automatically.  Specific schedule families (LTR stages, diffusion buckets,
    etc.) get stricter structural checks.
    """
    report = NumericScheduleValidationReportV1()

    # Scalars that must be > 0 (architectural dimensions, counts that have no
    # meaningful disabled state).
    positive_int_names = {
        "batch_size", "d_model", "n_heads", "context_layers",
        "denoiser_layers", "max_prompt_len", "max_target_len", "gen_steps",
        "recursive_steps", "grammar_top_k", "grammar_ltr_max_tokens",
        "grammar_block_size", "generate_batch_size", "grad_accum_steps",
        "parallel_workers", "design_md_budget", "diffusion_overallocate",
        "topology_max_nodes", "topology_max_active", "topology_max_arity",
        "topology_max_depth", "topology_max_phases",
        "topology_global_sync_interval", "solver_max_nodes", "solver_max_depth",
        "solver_max_backtracks", "solver_max_verifier_calls", "target_token_budget",
        "eval_shards", "pointer_hidden_dim", "pointer_heads",
        "connector_hidden_dim", "connector_rank", "connector_n_queries",
        "action_shortlist_k", "action_shortlist_min_legal_size",
        "grammar_multitoken_max", "surface_ar_d_model", "surface_ar_n_layers",
        "surface_ar_n_heads", "surface_ar_max_bytes", "surface_ar_top_k",
        "generate_max_attempts", "slot_component_owner_rare_multiplier",
        "root_reference_identity_strict_subset_multiplier", "cluster_max_size",
        "speculative_fanout",
    }
    # Scalars that are allowed to be 0 (feature disabled) or have other
    # documented non-positive sentinel values.
    non_negative_int_names = {
        "steps", "recursive_transition_layers", "eval_every", "loss_eval_every",
        "suffix_rollback_window", "slot_component_owner_rare_threshold",
        "grammar_canvas_lookahead", "retrieval_k", "solver_max_wall_ms",
        "stability_min_persistence",
    }
    positive_float_names = {"lr", "mdlm_eps"}

    for f in _fields(cfg):
        name = f.name
        value = getattr(cfg, name)
        ft = str(f.type)

        # Tuple-shaped schedules/weights.
        if _type_is_tuple(ft):
            if value is None:
                continue
            if name.endswith("_stages"):
                try:
                    strictly_increasing_sequence(name, value)
                    report.record(name, True)
                except NumericValidationError as exc:
                    report.record(name, False, exc.reason)
            elif name.endswith("_buckets"):
                try:
                    strictly_increasing_sequence(name, value)
                    report.record(name, True)
                except NumericValidationError as exc:
                    report.record(name, False, exc.reason)
            elif name == "diffusion_policies":
                # Allowed values are validated by the caller/diffusion adapter;
                # here we just enforce finite strings and uniqueness.
                try:
                    unique_enum_sequence(name, value, value)
                    report.record(name, True)
                except NumericValidationError as exc:
                    report.record(name, False, exc.reason)
            elif name in {
                "recursive_depth_supervision_weights",
                "slot_component_class_weights",
            }:
                try:
                    finite_non_negative_vector(name, value)
                    report.record(name, True)
                except NumericValidationError as exc:
                    report.record(name, False, exc.reason)
            elif name in {"slot_component_lexeme_priors", "slot_component_span_priors"}:
                try:
                    for family, sub in value:
                        for i, x in enumerate(sub):
                            finite_scalar(f"{name}[{family}][{i}]", x)
                    report.record(name, True)
                except NumericValidationError as exc:
                    report.record(name, False, exc.reason)
            elif name == "targeted_margin_family_weights":
                try:
                    for family, w in value:
                        finite_scalar(f"{name}[{family}]", w)
                        if w < 0:
                            raise NumericValidationError(
                                f"{name}[{family}]", w, "must be non-negative"
                            )
                    report.record(name, True)
                except NumericValidationError as exc:
                    report.record(name, False, exc.reason)
            else:
                # Any other tuple of numbers (e.g. mixture knobs) -- finite/non-negative.
                if value and isinstance(value[0], (int, float)) and not isinstance(value[0], bool):
                    try:
                        finite_non_negative_vector(name, value)
                        report.record(name, True)
                    except NumericValidationError as exc:
                        report.record(name, False, exc.reason)
            continue

        # Scalar numeric fields.
        if not _type_is_numeric_scalar(ft):
            continue

        if name in positive_int_names:
            try:
                positive_scalar(name, value)
                report.record(name, True)
            except NumericValidationError as exc:
                report.record(name, False, exc.reason)
        elif name in positive_float_names:
            try:
                positive_scalar(name, value)
                report.record(name, True)
            except NumericValidationError as exc:
                report.record(name, False, exc.reason)
        elif name in non_negative_int_names:
            try:
                non_negative_scalar(name, value)
                report.record(name, True)
            except NumericValidationError as exc:
                report.record(name, False, exc.reason)
        elif name in {"max_wall_minutes"}:
            try:
                if value is not None:
                    finite_scalar(name, value)
                    if not (0.0 <= value <= MAX_RUN_MINUTES):
                        raise NumericValidationError(
                            name,
                            value,
                            f"must be at most {MAX_RUN_MINUTES} minutes",
                        )
                report.record(name, True)
            except NumericValidationError as exc:
                report.record(name, False, exc.reason)
        elif name in {"decode_min_content"}:
            # A4: 0 off | >0 floor | -1 auto-from-inventory
            try:
                if value is not None and value != -1:
                    non_negative_scalar(name, value)
                report.record(name, True)
            except NumericValidationError as exc:
                report.record(name, False, exc.reason)
        elif name in {"mask_min", "mask_max"}:
            try:
                interval_scalar(name, value, 0.0, 1.0)
                report.record(name, True)
            except NumericValidationError as exc:
                report.record(name, False, exc.reason)
        elif name.endswith(("_weight", "_weights", "_threshold", "_margin",
                            "_rate", "_frac", "_temperature", "_eps",
                            "_loss_weight", "_decode_weight", "_bias")):
            try:
                non_negative_scalar(name, value)
                report.record(name, True)
            except NumericValidationError as exc:
                report.record(name, False, exc.reason)

    # Cross-field ordering invariants.
    if hasattr(cfg, "mask_min") and hasattr(cfg, "mask_max"):
        mn = getattr(cfg, "mask_min")
        mx = getattr(cfg, "mask_max")
        if mn is not None and mx is not None and mn > mx:
            report.record("mask_min/max", False, "mask_min must be <= mask_max")

    lever_errors = lever_configuration_errors(
        cfg, require_trained_decode=require_trained_decode
    )
    if lever_errors:
        report.record(
            "lever_capability_compatibility",
            False,
            "; ".join(lever_errors),
        )

    if not report.valid:
        raise NumericValidationError(
            "numeric_schedule_validation", report.errors, "; ".join(report.errors)
        )
    return report


def validate_model_build_config(cfg: Any) -> NumericScheduleValidationReportV1:
    """Fail-closed numeric gate for ``ModelBuildConfig``."""
    return validate_numeric_config(cfg, context="ModelBuildConfig")


def validate_twotower_config(cfg: Any) -> NumericScheduleValidationReportV1:
    """Fail-closed numeric gate for ``TwoTowerConfig``.

    Recursive-depth supervision is *not* checked here; call
    ``validate_recursive_depth_supervision`` in the model owner so that the
    architecture-specific capability gate stays with the denoiser code.
    """
    return validate_numeric_config(
        cfg, context="TwoTowerConfig", require_trained_decode=True
    )
