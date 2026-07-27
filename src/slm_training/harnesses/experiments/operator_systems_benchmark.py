"""SLM-407 operator-serving work accounting.

The serving path does more than call a policy model: it builds a legal set,
dry-runs candidates, replays selected actions, validates, and materializes.
This module keeps those costs in one comparable per-request denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Literal, Mapping


OPERATOR_SERVING_WORK_SCHEMA = "operator_serving_work/v1"
OPERATOR_SERVING_REPORT_SCHEMA = "operator_serving_benchmark_report/v1"
REQUIRED_ARMS = (
    "x22",
    "full_generation",
    "serialized_operator",
    "typed_operator",
    "compiler_forced",
)
Outcome = Literal["success", "fallback", "timeout", "failure"]


def _nearest_rank(values: Iterable[float], percentile: int) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, (len(ordered) * percentile + 99) // 100 - 1)
    return ordered[index]


@dataclass(frozen=True)
class OperatorServingIdentityV1:
    """Fields that must agree before serving work can be compared."""

    target_model: str
    target_model_revision: str
    device: str
    precision: str
    batch_size: int
    cache_mode: str
    context_fingerprint: str
    quality_metric: str = "meaningful_parse"
    schema: str = "operator_serving_identity/v1"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not all(
            (
                self.target_model,
                self.target_model_revision,
                self.device,
                self.precision,
                self.cache_mode,
                self.context_fingerprint,
                self.quality_metric,
            )
        ):
            raise ValueError("serving identity fields must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target_model": self.target_model,
            "target_model_revision": self.target_model_revision,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "cache_mode": self.cache_mode,
            "context_fingerprint": self.context_fingerprint,
            "quality_metric": self.quality_metric,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorServingIdentityV1":
        return cls(
            target_model=str(data["target_model"]),
            target_model_revision=str(data["target_model_revision"]),
            device=str(data["device"]),
            precision=str(data["precision"]),
            batch_size=int(data["batch_size"]),
            cache_mode=str(data["cache_mode"]),
            context_fingerprint=str(data["context_fingerprint"]),
            quality_metric=str(data.get("quality_metric", "meaningful_parse")),
            schema=str(data.get("schema", "operator_serving_identity/v1")),
        )


@dataclass(frozen=True)
class OperatorServingWorkV1:
    """Raw serving work for one request, including unsuccessful requests."""

    request_id: str
    arm: str
    identity: OperatorServingIdentityV1
    outcome: Outcome
    meaningful_parse: bool | None
    quality_score: float | None
    state_size: int
    legal_set_actions: int
    legal_set_builds: int
    dry_run_calls: int
    executor_calls: int
    validator_calls: int
    materialization_calls: int
    cache_hits: int
    cache_misses: int
    target_model_forwards: int
    non_target_forward_equivalents: float
    legal_set_ms: float
    dry_run_ms: float
    executor_ms: float
    validator_ms: float
    materialization_ms: float
    model_device_ms: float
    cpu_ms: float
    wall_ms: float
    peak_memory_bytes: int
    prefix_shared_tokens: int = 0
    stratum: str = "all"
    schema: str = OPERATOR_SERVING_WORK_SCHEMA

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "fallback", "timeout", "failure"}:
            raise ValueError(f"unsupported outcome {self.outcome!r}")
        if self.schema != OPERATOR_SERVING_WORK_SCHEMA:
            raise ValueError("unsupported operator-serving work schema")
        integer_fields = (
            self.state_size,
            self.legal_set_actions,
            self.legal_set_builds,
            self.dry_run_calls,
            self.executor_calls,
            self.validator_calls,
            self.materialization_calls,
            self.cache_hits,
            self.cache_misses,
            self.target_model_forwards,
            self.peak_memory_bytes,
            self.prefix_shared_tokens,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("operator-serving counters cannot be negative")
        float_fields = (
            self.non_target_forward_equivalents,
            self.legal_set_ms,
            self.dry_run_ms,
            self.executor_ms,
            self.validator_ms,
            self.materialization_ms,
            self.model_device_ms,
            self.cpu_ms,
            self.wall_ms,
        )
        if any(value < 0 for value in float_fields):
            raise ValueError("operator-serving times cannot be negative")
        if self.quality_score is not None and not 0 <= self.quality_score <= 1:
            raise ValueError("quality_score must be within [0, 1]")
        if not self.request_id or not self.arm or not self.stratum:
            raise ValueError("request_id, arm, and stratum must be non-empty")

    @property
    def target_forward_equivalents(self) -> float:
        """The primary compatible numeraire, reconciled from raw components."""
        return self.target_model_forwards + self.non_target_forward_equivalents

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "arm": self.arm,
            "identity": self.identity.to_dict(),
            "outcome": self.outcome,
            "meaningful_parse": self.meaningful_parse,
            "quality_score": self.quality_score,
            "state_size": self.state_size,
            "stratum": self.stratum,
            "raw_components": {
                "legal_set_actions": self.legal_set_actions,
                "legal_set_builds": self.legal_set_builds,
                "dry_run_calls": self.dry_run_calls,
                "executor_calls": self.executor_calls,
                "validator_calls": self.validator_calls,
                "materialization_calls": self.materialization_calls,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "target_model_forwards": self.target_model_forwards,
                "non_target_forward_equivalents": self.non_target_forward_equivalents,
                "legal_set_ms": self.legal_set_ms,
                "dry_run_ms": self.dry_run_ms,
                "executor_ms": self.executor_ms,
                "validator_ms": self.validator_ms,
                "materialization_ms": self.materialization_ms,
                "model_device_ms": self.model_device_ms,
                "cpu_ms": self.cpu_ms,
                "wall_ms": self.wall_ms,
                "peak_memory_bytes": self.peak_memory_bytes,
                "prefix_shared_tokens": self.prefix_shared_tokens,
            },
            "primary_numeraire": {
                "name": "target_forward_equivalents",
                "value": self.target_forward_equivalents,
                "reconciliation": (
                    "target_model_forwards + non_target_forward_equivalents"
                ),
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorServingWorkV1":
        raw = data["raw_components"]
        return cls(
            request_id=str(data["request_id"]),
            arm=str(data["arm"]),
            identity=OperatorServingIdentityV1.from_dict(data["identity"]),
            outcome=data["outcome"],
            meaningful_parse=data.get("meaningful_parse"),
            quality_score=data.get("quality_score"),
            state_size=int(data["state_size"]),
            stratum=str(data.get("stratum", "all")),
            legal_set_actions=int(raw["legal_set_actions"]),
            legal_set_builds=int(raw["legal_set_builds"]),
            dry_run_calls=int(raw["dry_run_calls"]),
            executor_calls=int(raw["executor_calls"]),
            validator_calls=int(raw["validator_calls"]),
            materialization_calls=int(raw["materialization_calls"]),
            cache_hits=int(raw["cache_hits"]),
            cache_misses=int(raw["cache_misses"]),
            target_model_forwards=int(raw["target_model_forwards"]),
            non_target_forward_equivalents=float(raw["non_target_forward_equivalents"]),
            legal_set_ms=float(raw["legal_set_ms"]),
            dry_run_ms=float(raw["dry_run_ms"]),
            executor_ms=float(raw["executor_ms"]),
            validator_ms=float(raw["validator_ms"]),
            materialization_ms=float(raw["materialization_ms"]),
            model_device_ms=float(raw["model_device_ms"]),
            cpu_ms=float(raw["cpu_ms"]),
            wall_ms=float(raw["wall_ms"]),
            peak_memory_bytes=int(raw["peak_memory_bytes"]),
            prefix_shared_tokens=int(raw.get("prefix_shared_tokens", 0)),
            schema=str(data.get("schema", OPERATOR_SERVING_WORK_SCHEMA)),
        )


def _summary(rows: tuple[OperatorServingWorkV1, ...]) -> dict[str, Any]:
    fields = (
        "legal_set_actions",
        "legal_set_builds",
        "dry_run_calls",
        "executor_calls",
        "validator_calls",
        "materialization_calls",
        "cache_hits",
        "cache_misses",
        "target_model_forwards",
        "non_target_forward_equivalents",
        "legal_set_ms",
        "dry_run_ms",
        "executor_ms",
        "validator_ms",
        "materialization_ms",
        "model_device_ms",
        "cpu_ms",
        "wall_ms",
        "peak_memory_bytes",
        "prefix_shared_tokens",
    )
    totals = {field: sum(getattr(row, field) for row in rows) for field in fields}
    total_numeraire = sum(row.target_forward_equivalents for row in rows)
    quality = [row.quality_score for row in rows if row.quality_score is not None]
    meaningful = [row.meaningful_parse for row in rows if row.meaningful_parse is not None]
    return {
        "n": len(rows),
        "outcomes": {
            outcome: sum(row.outcome == outcome for row in rows)
            for outcome in ("success", "fallback", "timeout", "failure")
        },
        "totals": totals,
        "primary_numeraire": {
            "name": "target_forward_equivalents",
            "total": total_numeraire,
            "reconciles": total_numeraire
            == totals["target_model_forwards"]
            + totals["non_target_forward_equivalents"],
        },
        "quality": {
            "mean_score": None if not quality else fmean(quality),
            "meaningful_parse_rate": (
                None if not meaningful else sum(meaningful) / len(meaningful)
            ),
        },
        "distribution": {
            "wall_ms_p50": _nearest_rank((row.wall_ms for row in rows), 50),
            "wall_ms_p95": _nearest_rank((row.wall_ms for row in rows), 95),
            "target_forward_equivalents_p50": _nearest_rank(
                (row.target_forward_equivalents for row in rows), 50
            ),
            "target_forward_equivalents_p95": _nearest_rank(
                (row.target_forward_equivalents for row in rows), 95
            ),
            "target_model_forwards_p50": _nearest_rank(
                (float(row.target_model_forwards) for row in rows), 50
            ),
            "target_model_forwards_p95": _nearest_rank(
                (float(row.target_model_forwards) for row in rows), 95
            ),
            "legal_set_actions_p50": _nearest_rank(
                (float(row.legal_set_actions) for row in rows), 50
            ),
            "legal_set_actions_p95": _nearest_rank(
                (float(row.legal_set_actions) for row in rows), 95
            ),
            "dry_run_calls_p50": _nearest_rank(
                (float(row.dry_run_calls) for row in rows), 50
            ),
            "dry_run_calls_p95": _nearest_rank(
                (float(row.dry_run_calls) for row in rows), 95
            ),
            "executor_calls_p50": _nearest_rank(
                (float(row.executor_calls) for row in rows), 50
            ),
            "executor_calls_p95": _nearest_rank(
                (float(row.executor_calls) for row in rows), 95
            ),
            "throughput_requests_per_second": (
                None if totals["wall_ms"] == 0 else len(rows) * 1000 / totals["wall_ms"]
            ),
        },
    }


def _observed_crossovers(
    rows: tuple[OperatorServingWorkV1, ...], baseline_arm: str
) -> dict[str, list[dict[str, Any]]]:
    """Report only sign changes between observed strata; never interpolate."""
    by_arm_stratum: dict[tuple[str, str], list[OperatorServingWorkV1]] = {}
    for row in rows:
        by_arm_stratum.setdefault((row.arm, row.stratum), []).append(row)
    result: dict[str, list[dict[str, Any]]] = {}
    arms = sorted({row.arm for row in rows} - {baseline_arm})
    for arm in arms:
        points = []
        for stratum in sorted({row.stratum for row in rows}):
            baseline = by_arm_stratum.get((baseline_arm, stratum), [])
            candidate = by_arm_stratum.get((arm, stratum), [])
            if not baseline or not candidate:
                continue
            points.append(
                {
                    "stratum": stratum,
                    "delta_target_forward_equivalents": fmean(
                        row.target_forward_equivalents for row in candidate
                    )
                    - fmean(row.target_forward_equivalents for row in baseline),
                    "delta_wall_ms": fmean(row.wall_ms for row in candidate)
                    - fmean(row.wall_ms for row in baseline),
                }
            )
        crossings = []
        for before, after in zip(points, points[1:]):
            if before["delta_wall_ms"] * after["delta_wall_ms"] < 0:
                crossings.append({"between": [before["stratum"], after["stratum"]]})
        result[arm] = [{"observed_points": points, "crossovers": crossings}]
    return result


def build_operator_serving_report(
    *,
    identity: OperatorServingIdentityV1,
    rows: Iterable[OperatorServingWorkV1],
    baseline_arm: str = "full_generation",
    required_arms: tuple[str, ...] = REQUIRED_ARMS,
    quality_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Aggregate full-denominator work and fail closed on unmatched quality."""
    if quality_tolerance < 0:
        raise ValueError("quality_tolerance cannot be negative")
    frozen_rows = tuple(rows)
    if any(row.identity != identity for row in frozen_rows):
        raise ValueError("all work rows must share one serving identity")
    arms = sorted({row.arm for row in frozen_rows})
    summaries = {
        arm: _summary(tuple(row for row in frozen_rows if row.arm == arm))
        for arm in arms
    }
    missing = [arm for arm in required_arms if arm not in summaries]
    baseline = summaries.get(baseline_arm)
    comparisons: dict[str, Any] = {}
    for arm, summary in summaries.items():
        if arm == baseline_arm or baseline is None:
            continue
        baseline_quality = baseline["quality"]["mean_score"]
        candidate_quality = summary["quality"]["mean_score"]
        score_matched = (
            baseline_quality is not None
            and candidate_quality is not None
            and candidate_quality + quality_tolerance >= baseline_quality
        )
        baseline_meaningful = baseline["quality"]["meaningful_parse_rate"]
        candidate_meaningful = summary["quality"]["meaningful_parse_rate"]
        meaningful_matched = (
            baseline_meaningful is not None
            and candidate_meaningful is not None
            and candidate_meaningful >= baseline_meaningful
        )
        quality_matched = meaningful_matched and score_matched
        faster = summary["totals"]["wall_ms"] < baseline["totals"]["wall_ms"]
        lower_work = (
            summary["primary_numeraire"]["total"]
            < baseline["primary_numeraire"]["total"]
        )
        comparisons[arm] = {
            "quality_matched": quality_matched,
            "meaningful_parse_matched": meaningful_matched,
            "lower_total_work": lower_work,
            "lower_total_wall": faster,
            "efficiency_claim_supported": quality_matched and lower_work and faster,
        }
    supported = [
        arm for arm, comparison in comparisons.items()
        if comparison["efficiency_claim_supported"]
    ]
    if missing:
        verdict = "unavailable"
        reason = "required arms are not all measured under one serving identity"
    elif baseline is None:
        verdict = "unavailable"
        reason = "the full_generation baseline is missing"
    elif not supported:
        verdict = "reject_efficiency_claim"
        reason = "no compared arm improved total work and wall time at matched quality"
    else:
        verdict = "measured"
        reason = "one or more arms improved observed total work and wall time"
    return {
        "schema": OPERATOR_SERVING_REPORT_SCHEMA,
        "identity": identity.to_dict(),
        "primary_metric": "target_forward_equivalents",
        "required_arms": list(required_arms),
        "measured_arms": arms,
        "missing_arms": missing,
        "per_arm": summaries,
        "comparisons": comparisons,
        "observed_crossover": _observed_crossovers(frozen_rows, baseline_arm),
        "verdict": verdict,
        "reason": reason,
        "efficiency_claim_supported_arms": supported,
        "raw_rows": [row.to_dict() for row in frozen_rows],
    }
