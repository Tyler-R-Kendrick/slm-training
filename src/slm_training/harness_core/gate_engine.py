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
) -> tuple[dict[str, dict[str, Any]], dict[str, bool], list[str]]:
    """Check every suite present in ``policy`` against its floors.

    Returns ``(actual, checks, failures)``: the normalized (slim) metrics per
    suite, the per-gate booleans, and human-readable failure descriptions in
    check order. Missing metrics fail their threshold. Suites absent from the
    scoreboard are reported as ``missing_suite`` failures (cannot claim pass
    without evidence). A policy entry may override the evidence floor per
    suite via a ``"min_n"`` key (never treated as a metric bar).
    """
    checks: dict[str, bool] = {}
    actual: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for suite_name, mins in policy.items():
        metrics = suites.get(suite_name)
        if metrics is None:
            key = f"{suite_name}:missing_suite"
            checks[key] = False
            failures.append(key)
            continue
        slim = normalize_suite(metrics)
        actual[suite_name] = slim
        fallback_count = metrics.get("fallback_count")
        fallback_key = f"{suite_name}:certified_fallback"
        if fallback_count is None:
            # Unmeasured must never certify: a board without fallback telemetry
            # cannot claim learned (fallback-free) quality.
            checks[fallback_key] = False
            failures.append(
                f"{fallback_key} unmeasured (fallback_count absent) need=0 "
                "for learned-quality claims"
            )
        else:
            fallback_count = int(fallback_count)
            checks[fallback_key] = fallback_count == 0
            if fallback_count:
                failures.append(
                    f"{fallback_key} actual={fallback_count} need=0 for learned-quality claims"
                )
        min_n = int(mins.get("min_n", default_min_n))
        n_value = metrics.get("n")
        n_key = f"{suite_name}:insufficient_n"
        n_ok = isinstance(n_value, (int, float)) and int(n_value) >= min_n
        checks[n_key] = n_ok
        if not n_ok:
            failures.append(f"{n_key} actual={n_value!r} need>={min_n}")
        for metric, minimum in mins.items():
            if metric == "min_n":
                continue
            key = f"{suite_name}:{metric}"
            value = slim.get(metric, metrics.get(metric))
            ok = value is not None and float(value) >= float(minimum)
            checks[key] = ok
            if not ok:
                failures.append(
                    f"{key} actual={value!r} need>={minimum}"
                )

    return actual, checks, failures
