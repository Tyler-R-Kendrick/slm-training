"""Honest multi-suite ship gates (adversarial-review policy)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Per-suite minimums. Smoke is a canary; generalization requires the rest.
DEFAULT_SHIP_GATES: dict[str, dict[str, float]] = {
    "smoke": {
        "parse_rate": 0.66,
        "structural_similarity": 0.35,
        "placeholder_fidelity": 0.25,
        "reward_score": 0.30,
    },
    "held_out": {
        "parse_rate": 0.40,
        "structural_similarity": 0.30,
        "placeholder_fidelity": 0.15,
    },
    "adversarial": {
        "parse_rate": 0.25,
        "structural_similarity": 0.25,
    },
    "ood": {
        "parse_rate": 0.25,
        "structural_similarity": 0.25,
    },
    "rico_held": {
        "parse_rate": 0.10,
        "structural_similarity": 0.20,
    },
}


def evaluate_ship_gates(
    suites: dict[str, dict[str, Any]],
    *,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """
    Check every suite present in `suites` against the ship policy.

    Missing metrics fail that threshold. Suites absent from the scoreboard are
    reported as `missing_suite` failures (cannot claim pass without evidence).
    """
    policy = thresholds or DEFAULT_SHIP_GATES
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
        slim = {
            "n": metrics.get("n"),
            "parse_rate": metrics.get("parse_rate"),
            "placeholder_fidelity": metrics.get("placeholder_fidelity"),
            "placeholder_validity": metrics.get("placeholder_validity"),
            "structural_similarity": metrics.get("structural_similarity"),
            "reward_score": metrics.get("reward_score"),
        }
        actual[suite_name] = slim
        fallback_count = int(metrics.get("fallback_count") or 0)
        fallback_key = f"{suite_name}:certified_fallback"
        checks[fallback_key] = fallback_count == 0
        if fallback_count:
            failures.append(
                f"{fallback_key} actual={fallback_count} need=0 for learned-quality claims"
            )
        for metric, minimum in mins.items():
            key = f"{suite_name}:{metric}"
            value = metrics.get(metric)
            ok = value is not None and float(value) >= float(minimum)
            checks[key] = ok
            if not ok:
                failures.append(
                    f"{key} actual={value!r} need>={minimum}"
                )

    return {
        "policy": policy,
        "actual": actual,
        "gates": checks,
        "failures": failures,
        "pass": all(checks.values()) if checks else False,
        "note": (
            "Honest ship gates require all policy suites and score structure only "
            "(parse / structural_similarity / placeholder_fidelity / reward_score). "
            "DESIGN.md style lint is never a ship gate. "
            "See docs/design/adversarial-review.md and docs/design/structure-only-eval.md."
        ),
    }


def write_ship_gates(
    run_dir: Path | str,
    suites: dict[str, dict[str, Any]],
    *,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Write gates.json under the run directory; return the payload."""
    payload = evaluate_ship_gates(suites, thresholds=thresholds)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "gates.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["output"] = str(path)
    return payload
