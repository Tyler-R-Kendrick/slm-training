"""Honest multi-suite ship gates (adversarial-review policy)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Per-suite minimums. Smoke is a canary; generalization requires the rest.
#
# ``component_type_recall`` is the **semantic-density floor** (E2): the fraction
# of the gold's component *types* the prediction recovers. It collapses toward 0
# for the trivial/empty program, so a compression- or decode-driven change that
# emits shorter-but-emptier output cannot green these gates on syntax alone. The
# floors sit at or below the structural bars (density must be at least as present
# as structure) and only make the policy stricter — never weaker.
DEFAULT_SHIP_GATES: dict[str, dict[str, float]] = {
    "smoke": {
        "meaningful_program_rate": 0.66,
        "structural_similarity": 0.35,
        "component_type_recall": 0.35,
        "placeholder_fidelity": 0.25,
        "reward_score": 0.30,
    },
    "held_out": {
        "meaningful_program_rate": 0.40,
        "structural_similarity": 0.30,
        "component_type_recall": 0.30,
        "placeholder_fidelity": 0.15,
    },
    "adversarial": {
        "meaningful_program_rate": 0.25,
        "structural_similarity": 0.25,
        "component_type_recall": 0.20,
    },
    "ood": {
        "meaningful_program_rate": 0.25,
        "structural_similarity": 0.25,
        "component_type_recall": 0.20,
    },
    "rico_held": {
        "meaningful_program_rate": 0.10,
        "structural_similarity": 0.20,
        "component_type_recall": 0.15,
    },
}

# Production readiness requires the complete held-out RICO corpus, not a
# diagnostic prefix that happens to clear the metric thresholds.
MIN_SHIP_SUITE_N: dict[str, int] = {"rico_held": 1500}


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
            "meaningful_program_rate": metrics.get("meaningful_program_rate"),
            "syntax_parse_rate": metrics.get("syntax_parse_rate"),
            "placeholder_fidelity": metrics.get("placeholder_fidelity"),
            "placeholder_validity": metrics.get("placeholder_validity"),
            "structural_similarity": metrics.get("structural_similarity"),
            "component_type_recall": metrics.get("component_type_recall"),
            "reward_score": metrics.get("reward_score"),
        }
        if (
            slim["meaningful_program_rate"] is None
            and metrics.get("syntax_parse_rate") is None
        ):
            # Historical scoreboards used parse_rate for meaningful-program
            # quality. New scoreboards persist both metrics explicitly.
            slim["meaningful_program_rate"] = metrics.get("parse_rate")
        actual[suite_name] = slim
        complete_key = f"{suite_name}:complete_evaluation"
        diagnostic_subset = bool(metrics.get("diagnostic_subset"))
        checks[complete_key] = not diagnostic_subset
        if diagnostic_subset:
            failures.append(f"{complete_key} actual=diagnostic_subset need=full_suite")
        minimum_n = MIN_SHIP_SUITE_N.get(suite_name)
        if minimum_n is not None:
            sample_key = f"{suite_name}:sample_count"
            sample_n = metrics.get("n")
            sample_ok = (
                isinstance(sample_n, (int, float)) and int(sample_n) >= minimum_n
            )
            checks[sample_key] = sample_ok
            if not sample_ok:
                failures.append(
                    f"{sample_key} actual={sample_n!r} need>={minimum_n}"
                )
        fallback_count = int(metrics.get("fallback_count") or 0)
        fallback_key = f"{suite_name}:certified_fallback"
        checks[fallback_key] = fallback_count == 0
        if fallback_count:
            failures.append(
                f"{fallback_key} actual={fallback_count} need=0 for learned-quality claims"
            )
        for metric, minimum in mins.items():
            key = f"{suite_name}:{metric}"
            value = slim.get(metric, metrics.get(metric))
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
            "(meaningful_program_rate / structural_similarity / component_type_recall "
            "/ placeholder_fidelity / reward_score). component_type_recall is the "
            "semantic-density floor: shorter-but-emptier output cannot pass on "
            "syntax alone. Diagnostic subsets cannot pass, and production RICO "
            "requires n>=1500. Syntax parse is reported separately and is not a "
            "learned-quality substitute. DESIGN.md style lint is never a ship gate. "
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
