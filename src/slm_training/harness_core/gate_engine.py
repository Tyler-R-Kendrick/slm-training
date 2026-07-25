"""Generic multi-suite gate-check engine (DSL-agnostic).

The frozen check loop extracted verbatim from the OpenUI ship-gate policy
owner (``slm_training.harnesses.model_build.ship_gates``), which remains the
home of the policy itself: thresholds, metric names, the suite normalizer,
payload assembly, and ``gates.json`` writing. The engine knows nothing about
any particular DSL or metric family — the caller supplies the policy dict,
the per-suite metric normalizer, and the evidence floor.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

SuiteNormalizer = Callable[[Mapping[str, Any]], dict[str, Any]]


def _finite_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _integral_count(value: Any) -> bool:
    return _finite_real(value) and float(value).is_integer() and float(value) >= 0


def _json_scalar(value: Any) -> Any:
    """Keep criterion evidence JSON-safe without making it passable."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def build_gate_criteria(
    suites: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Mapping[str, float]],
    *,
    normalize_suite: SuiteNormalizer,
    default_min_n: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Lower a gate policy to raw deterministic AgentEvals criteria."""
    actual: dict[str, dict[str, Any]] = {}
    criteria: list[dict[str, Any]] = []

    for suite_name, mins in policy.items():
        metrics = suites.get(suite_name)
        if metrics is None:
            criteria.append(
                {
                    "id": f"{suite_name}:missing_suite",
                    "suite": suite_name,
                    "actual": None,
                    "operator": "present",
                    "expected": True,
                }
            )
            continue
        slim = normalize_suite(metrics)
        actual[suite_name] = slim
        fallback_count = metrics.get("fallback_count")
        criteria.append(
            {
                "id": f"{suite_name}:certified_fallback",
                "suite": suite_name,
                "actual": (
                    int(fallback_count)
                    if _integral_count(fallback_count)
                    else _json_scalar(fallback_count)
                ),
                "operator": "eq",
                "expected": 0,
            }
        )
        criteria.append(
            {
                "id": f"{suite_name}:insufficient_n",
                "suite": suite_name,
                "actual": _json_scalar(metrics.get("n")),
                "operator": "gte",
                "expected": int(mins.get("min_n", default_min_n)),
            }
        )
        criteria.extend(
            {
                "id": f"{suite_name}:{metric}",
                "suite": suite_name,
                "actual": _json_scalar(slim.get(metric, metrics.get(metric))),
                "operator": "gte",
                "expected": minimum,
            }
            for metric, minimum in mins.items()
            if metric != "min_n"
        )
        timeout_count = metrics.get("decode_timeout_count")
        if (
            isinstance(timeout_count, (int, float))
            and not isinstance(timeout_count, bool)
            and timeout_count > 0
        ):
            criteria.append(
                {
                    "id": f"{suite_name}:decode_timeout_count",
                    "suite": suite_name,
                    "actual": _json_scalar(timeout_count),
                    "operator": "eq",
                    "expected": 0,
                }
            )
    return actual, criteria


def criterion_passes(criterion: Mapping[str, Any]) -> bool:
    """Project one raw criterion to a boolean for non-authoritative previews."""
    actual = criterion.get("actual")
    operator = criterion.get("operator")
    expected = criterion.get("expected")
    criterion_id = str(criterion.get("id", ""))
    if operator == "present":
        return actual is not None
    if operator == "eq":
        if criterion_id.endswith(":certified_fallback"):
            return _integral_count(actual) and int(actual) == expected
        return actual is not None and actual == expected
    if operator == "gte":
        valid = (
            _integral_count(actual)
            if criterion_id.endswith(":insufficient_n")
            else _finite_real(actual)
        )
        return valid and _finite_real(expected) and float(actual) >= float(expected)
    raise ValueError(f"unsupported gate criterion operator: {operator!r}")


def criterion_failure(criterion: Mapping[str, Any]) -> str:
    """Render the stable legacy failure text for a failed raw criterion."""
    key = str(criterion["id"])
    actual = criterion.get("actual")
    expected = criterion.get("expected")
    if key.endswith(":missing_suite"):
        return key
    if key.endswith(":certified_fallback"):
        if actual is None:
            return (
                f"{key} unmeasured (fallback_count absent) need=0 "
                "for learned-quality claims"
            )
        if not _integral_count(actual):
            return (
                f"{key} invalid (fallback_count={actual!r}) need=0 "
                "for learned-quality claims"
            )
        return f"{key} actual={actual} need=0 for learned-quality claims"
    if criterion.get("operator") == "eq":
        return f"{key} actual={actual!r} need={expected}"
    return f"{key} actual={actual!r} need>={expected}"


def criterion_failure_category(criterion: Mapping[str, Any]) -> str:
    key = str(criterion["id"])
    actual = criterion.get("actual")
    if key.endswith(":missing_suite"):
        return "evidence_volume_failures"
    if key.endswith(":certified_fallback"):
        return "measurement_integrity_failures"
    if key.endswith(":decode_timeout_count"):
        return "runtime_failures"
    if key.endswith(":insufficient_n"):
        return (
            "evidence_volume_failures"
            if _integral_count(actual)
            else "measurement_integrity_failures"
        )
    return (
        "quality_threshold_failures"
        if _finite_real(actual)
        else "measurement_integrity_failures"
    )


def run_gate_checks(
    suites: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Mapping[str, float]],
    *,
    normalize_suite: SuiteNormalizer,
    default_min_n: int,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, bool],
    list[str],
    dict[str, list[str]],
]:
    """Check every suite present in ``policy`` against its floors.

    Returns normalized metrics, per-gate booleans, flat failures, and an exact
    failure-category partition. Missing metrics fail measurement integrity.
    Suites absent from the scoreboard fail evidence volume. A policy entry may
    override the evidence floor per suite via a ``"min_n"`` key.
    """
    actual, criteria = build_gate_criteria(
        suites,
        policy,
        normalize_suite=normalize_suite,
        default_min_n=default_min_n,
    )
    checks = {str(item["id"]): criterion_passes(item) for item in criteria}
    failures = [
        criterion_failure(item) for item in criteria if not checks[str(item["id"])]
    ]
    categories: dict[str, list[str]] = {
        "evidence_volume_failures": [],
        "measurement_integrity_failures": [],
        "quality_threshold_failures": [],
        "runtime_failures": [],
    }
    for item, failure in zip(
        (item for item in criteria if not checks[str(item["id"])]),
        failures,
        strict=True,
    ):
        categories[criterion_failure_category(item)].append(failure)

    return actual, checks, failures, categories
