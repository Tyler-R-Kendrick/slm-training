"""Generic multi-suite gate-check engine (DSL-agnostic).

The frozen check loop extracted verbatim from the OpenUI ship-gate policy
owner (``slm_training.harnesses.model_build.ship_gates``), which remains the
home of the policy itself: thresholds, metric names, the suite normalizer,
payload assembly, and ``gates.json`` writing. The engine knows nothing about
any particular DSL or metric family — the caller supplies the policy dict,
the per-suite metric normalizer, and the evidence floor.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

SuiteNormalizer = Callable[[Mapping[str, Any]], dict[str, Any]]


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
    checks: dict[str, bool] = {}
    actual: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    categories: dict[str, list[str]] = {
        "evidence_volume_failures": [],
        "measurement_integrity_failures": [],
        "quality_threshold_failures": [],
        "runtime_failures": [],
    }

    def fail(message: str, category: str) -> None:
        failures.append(message)
        categories[category].append(message)

    for suite_name, mins in policy.items():
        metrics = suites.get(suite_name)
        if metrics is None:
            key = f"{suite_name}:missing_suite"
            checks[key] = False
            fail(key, "evidence_volume_failures")
            continue
        slim = normalize_suite(metrics)
        actual[suite_name] = slim
        fallback_count = metrics.get("fallback_count")
        fallback_key = f"{suite_name}:certified_fallback"
        if fallback_count is None:
            # Unmeasured must never certify: a board without fallback telemetry
            # cannot claim learned (fallback-free) quality.
            checks[fallback_key] = False
            fail(
                f"{fallback_key} unmeasured (fallback_count absent) need=0 "
                "for learned-quality claims",
                "measurement_integrity_failures",
            )
        else:
            fallback_count = int(fallback_count)
            checks[fallback_key] = fallback_count == 0
            if fallback_count:
                fail(
                    f"{fallback_key} actual={fallback_count} need=0 for learned-quality claims",
                    "measurement_integrity_failures",
                )
        min_n = int(mins.get("min_n", default_min_n))
        n_value = metrics.get("n")
        n_key = f"{suite_name}:insufficient_n"
        n_ok = isinstance(n_value, (int, float)) and int(n_value) >= min_n
        checks[n_key] = n_ok
        if not n_ok:
            fail(
                f"{n_key} actual={n_value!r} need>={min_n}",
                "evidence_volume_failures",
            )
        for metric, minimum in mins.items():
            if metric == "min_n":
                continue
            key = f"{suite_name}:{metric}"
            value = slim.get(metric, metrics.get(metric))
            try:
                ok = value is not None and float(value) >= float(minimum)
            except (TypeError, ValueError):
                ok = False
            checks[key] = ok
            if not ok:
                category = (
                    "measurement_integrity_failures"
                    if value is None or not isinstance(value, (int, float))
                    else "quality_threshold_failures"
                )
                fail(
                    f"{key} actual={value!r} need>={minimum}",
                    category,
                )

    return actual, checks, failures, categories
