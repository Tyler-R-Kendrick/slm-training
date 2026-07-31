"""Canonical SFF scoreboard metrics contract (quality + runtime).

Required field sets are loaded from the versioned external metrics resource
(``resources/experiments/semantic_factor_frontier/metrics.v1.json``) so the
contract surface can grow without rewriting harness Python.

Writers call :func:`validate_scoreboard_metrics` before persisting results.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from slm_training.harnesses.experiments.semantic_factor_config import (
    load_sff_metrics,
)

__all__ = [
    "SCOREBOARD_KIND",
    "REQUIRED_ARM_METRIC_FIELDS",
    "REQUIRED_ARM_RUNTIME_FIELDS",
    "REQUIRED_RUNTIME_BLOCK_FIELDS",
    "REQUIRED_CAMPAIGN_RUNTIME_ENDPOINTS",
    "validate_scoreboard_metrics",
    "ScoreboardMetricsError",
    "refresh_contract_from_resource",
]


class ScoreboardMetricsError(ValueError):
    """Scoreboard violates the SFF quality+runtime metrics contract."""


def refresh_contract_from_resource(metrics_name: str = "metrics.v1.json") -> None:
    """Reload required field sets from the external metrics resource."""

    global SCOREBOARD_KIND
    global REQUIRED_ARM_METRIC_FIELDS
    global REQUIRED_ARM_RUNTIME_FIELDS
    global REQUIRED_RUNTIME_BLOCK_FIELDS
    global REQUIRED_CAMPAIGN_RUNTIME_ENDPOINTS

    cfg = load_sff_metrics(metrics_name)
    SCOREBOARD_KIND = cfg.scoreboard_kind
    # Split is informational only for docs; validation uses the full required set.
    all_fields = cfg.required_arm_fields
    runtime_keys = frozenset(
        k
        for k in all_fields
        if k.startswith("wall_ms")
        or k.startswith("project_ms")
        or k.startswith("score_ms")
        or k
        in {
            "quality_per_ms",
            "delta_accuracy_per_extra_ms",
            "dominates_control_quality_runtime",
            "delta_accuracy_vs_control",
        }
    )
    REQUIRED_ARM_RUNTIME_FIELDS = runtime_keys
    REQUIRED_ARM_METRIC_FIELDS = frozenset(all_fields - runtime_keys)
    REQUIRED_RUNTIME_BLOCK_FIELDS = cfg.required_runtime_block
    REQUIRED_CAMPAIGN_RUNTIME_ENDPOINTS = cfg.required_campaign_runtime_endpoints


# Initialize from resource at import (single source of truth).
SCOREBOARD_KIND: str
REQUIRED_ARM_METRIC_FIELDS: frozenset[str]
REQUIRED_ARM_RUNTIME_FIELDS: frozenset[str]
REQUIRED_RUNTIME_BLOCK_FIELDS: frozenset[str]
REQUIRED_CAMPAIGN_RUNTIME_ENDPOINTS: frozenset[str]
refresh_contract_from_resource()


def validate_scoreboard_metrics(payload: Mapping[str, Any]) -> None:
    """Fail closed if a scoreboard omits required quality or runtime metrics."""

    # Always re-read contract so file edits apply without process restart in tests.
    refresh_contract_from_resource()
    errors: list[str] = []
    kind = str(payload.get("kind") or "")
    if not kind.startswith("semantic_factor_frontier_results/"):
        errors.append(f"unexpected kind {kind!r}")
    if kind in {"semantic_factor_frontier_results/v1"}:
        errors.append(
            f"kind {kind!r} predates runtime contract; use {SCOREBOARD_KIND}"
        )

    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append("missing runtime block")
    else:
        for key in sorted(REQUIRED_RUNTIME_BLOCK_FIELDS):
            if key not in runtime:
                errors.append(f"runtime missing {key}")
        if runtime.get("timer") != "time.perf_counter":
            errors.append(
                f"runtime.timer must be time.perf_counter, got {runtime.get('timer')!r}"
            )

    arms = payload.get("arms")
    required_arm = REQUIRED_ARM_METRIC_FIELDS | REQUIRED_ARM_RUNTIME_FIELDS
    if not isinstance(arms, Sequence) or not arms:
        errors.append("arms must be a non-empty list")
    else:
        for i, arm in enumerate(arms):
            if not isinstance(arm, Mapping):
                errors.append(f"arms[{i}] not an object")
                continue
            arm_id = arm.get("arm_id", f"index_{i}")
            for key in sorted(required_arm):
                if key not in arm:
                    errors.append(f"arm {arm_id!r} missing {key}")
            for key in (
                "wall_ms_total",
                "wall_ms_mean",
                "wall_ms_p50",
                "wall_ms_p95",
                "project_ms_mean",
                "score_ms_mean",
            ):
                if key not in arm:
                    continue
                val = arm[key]
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    errors.append(f"arm {arm_id!r} {key} not numeric")
                elif float(val) < 0:
                    errors.append(f"arm {arm_id!r} {key} negative")

    outcomes = payload.get("outcomes")
    if isinstance(outcomes, Sequence) and outcomes:
        sample = outcomes[0]
        if isinstance(sample, Mapping):
            for key in ("decision_ms", "score_ms", "project_ms"):
                if key not in sample:
                    errors.append(f"outcomes[] missing {key}")

    # Future runs must bind the external versioned config files they used.
    resources = payload.get("config_resources")
    if not isinstance(resources, Mapping):
        errors.append("missing config_resources (external metrics/scorer_params bind)")
    else:
        for key in ("metrics", "scorer_params", "campaign"):
            block = resources.get(key)
            if not isinstance(block, Mapping):
                errors.append(f"config_resources.{key} missing")
                continue
            for field in ("path", "schema", "version", "sha256"):
                if not block.get(field):
                    errors.append(f"config_resources.{key} missing {field}")

    if errors:
        raise ScoreboardMetricsError(
            "SFF scoreboard metrics contract failed:\n- " + "\n- ".join(errors)
        )


def campaign_declares_runtime_endpoints(endpoint_ids: Sequence[str]) -> bool:
    refresh_contract_from_resource()
    return REQUIRED_CAMPAIGN_RUNTIME_ENDPOINTS.issubset(set(endpoint_ids))
