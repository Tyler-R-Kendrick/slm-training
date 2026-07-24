"""Default-off, advisory semantic-collapse telemetry primitives (SLM-264)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

SEMANTIC_TRAJECTORY_VERSION = "semantic_trajectory/v1"


class FeatureAvailability(str, Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class EarlyWarningAction(str, Enum):
    CONTINUE = "continue"
    STOP_AS_COLLAPSE = "stop_as_collapse"
    EXTEND_EXPOSURE = "extend_exposure"
    ROUTE_TO_TARGETED_DATA_FAMILY = "route_to_targeted_data_family"
    ROUTE_DECODE_FOR_DIAGNOSTIC_ONLY = "route_decode_for_diagnostic_only"
    UNKNOWN_NO_ACTION = "unknown_no_action"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class SemanticTrajectoryTelemetryV1:
    """One immutable observation; unavailable features never become exact."""

    run_id: str
    checkpoint_sha256: str
    exposure_updates: int
    data_family: str
    example_id: str
    state_fingerprint: str
    feature_availability: FeatureAvailability
    full_vocab_entropy: float | None
    legal_entropy: float | None
    top1_top2_margin: float | None
    legal_set_size: int | None
    legal_mass: float | None
    empty_vs_populated_log_score_gap: float | None
    constrained_empty_mass: float | None
    constraint_debt_ref: str | None = None
    failure_trace_ref: str | None = None
    schema_version: str = SEMANTIC_TRAJECTORY_VERSION
    fingerprint: str = ""

    def __post_init__(self) -> None:
        payload = asdict(self)
        supplied = payload.pop("fingerprint")
        payload["feature_availability"] = self.feature_availability.value
        expected = _fingerprint(payload)
        if supplied and supplied != expected:
            raise ValueError("semantic trajectory fingerprint mismatch")
        object.__setattr__(self, "fingerprint", expected)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["feature_availability"] = self.feature_availability.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticTrajectoryTelemetryV1:
        return cls(
            **{
                **data,
                "feature_availability": FeatureAvailability(data["feature_availability"]),
            }
        )


@dataclass(frozen=True)
class SemanticEarlyWarningPolicyV1:
    """A frozen advisory policy. Unknown observations can never stop a run."""

    collapse_margin_ceiling: float = 0.0
    action_on_known_collapse: EarlyWarningAction = EarlyWarningAction.ROUTE_DECODE_FOR_DIAGNOSTIC_ONLY
    default_off: bool = True
    schema_version: str = "semantic_early_warning_policy/v1"

    def action_for(self, row: SemanticTrajectoryTelemetryV1) -> EarlyWarningAction:
        if self.default_off or row.feature_availability is not FeatureAvailability.EXACT:
            return EarlyWarningAction.UNKNOWN_NO_ACTION
        if (
            row.top1_top2_margin is not None
            and row.top1_top2_margin <= self.collapse_margin_ceiling
        ):
            return self.action_on_known_collapse
        return EarlyWarningAction.CONTINUE


def entropy_and_margin(probabilities: Iterable[float]) -> tuple[float, float]:
    """Return Shannon entropy and top-1/top-2 margin for a normalized finite set."""
    values = tuple(float(value) for value in probabilities)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("probabilities must be a non-empty non-negative sequence")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")
    ordered = sorted(values, reverse=True)
    entropy = -sum(value * math.log(value) for value in values if value)
    return entropy, ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)


def bounded_empty_mass(
    masses: Iterable[float], *, coverage: FeatureAvailability
) -> tuple[float | None, FeatureAvailability]:
    """Only a complete declared finite set can report an exact mass."""
    if coverage is not FeatureAvailability.EXACT:
        return None, coverage
    values = tuple(float(value) for value in masses)
    if any(value < 0 for value in values) or sum(values) > 1 + 1e-9:
        raise ValueError("bounded completion masses must be probabilities")
    return sum(values), FeatureAvailability.EXACT
