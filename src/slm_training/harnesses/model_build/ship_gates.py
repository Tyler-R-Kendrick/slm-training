"""Honest multi-suite ship gates (adversarial-review policy)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from slm_training.harness_core.gate_engine import (
    build_gate_criteria,
    criterion_failure,
    criterion_failure_category,
    run_gate_checks,
)

_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "resources/evals/openui_ship_gates_v6.json"
)
_REQUIRED_SUITES = {"smoke", "held_out", "adversarial", "ood", "rico_held"}
_BEQ_METRICS = {"ast_beq_rate", "canonical_beq_rate"}
_REQUIRED_METRICS = {
    "smoke": {
        "meaningful_program_rate",
        "structural_similarity",
        "component_type_recall",
        "ast_beq_rate",
        "canonical_beq_rate",
        "placeholder_fidelity",
        "reward_score",
    },
    "held_out": {
        "meaningful_program_rate",
        "structural_similarity",
        "component_type_recall",
        "ast_beq_rate",
        "canonical_beq_rate",
        "placeholder_fidelity",
    },
    "adversarial": {
        "meaningful_program_rate",
        "structural_similarity",
        "component_type_recall",
        "ast_beq_rate",
        "canonical_beq_rate",
    },
    "ood": {
        "meaningful_program_rate",
        "structural_similarity",
        "component_type_recall",
        "ast_beq_rate",
        "canonical_beq_rate",
    },
    "rico_held": {
        "meaningful_program_rate",
        "structural_similarity",
        "component_type_recall",
        "ast_beq_rate",
        "canonical_beq_rate",
    },
}

# When certificates are present, require digest BEq agreement.
CERTIFICATE_EQUIVALENCE_FLOOR = 1.0


def load_ship_gate_policy(path: Path | str = _POLICY_PATH) -> dict[str, Any]:
    """Load the committed default policy, rejecting malformed or weaker shapes."""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ship-gate policy {source}: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != "openui_ship_gate_policy/v1"
    ):
        raise ValueError(f"{source}: invalid ship-gate policy schema")
    version = value.get("version")
    if not isinstance(version, str) or source.stem != version:
        raise ValueError(f"{source}: version must match the resource filename")
    minimum = value.get("default_min_suite_n")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0:
        raise ValueError(f"{source}: default_min_suite_n must be a positive integer")
    suites = value.get("suites")
    if not isinstance(suites, dict) or set(suites) != _REQUIRED_SUITES:
        raise ValueError(f"{source}: all canonical ship suites are required")
    for suite, thresholds in suites.items():
        if not isinstance(thresholds, dict) or not thresholds:
            raise ValueError(f"{source}: {suite} thresholds must be a non-empty object")
        metrics = set(thresholds) - {"min_n"}
        if metrics != _REQUIRED_METRICS[suite]:
            raise ValueError(f"{source}: {suite} must define its canonical metrics")
        for metric, threshold in thresholds.items():
            if metric == "min_n":
                if (
                    not isinstance(threshold, int)
                    or isinstance(threshold, bool)
                    or threshold < minimum
                ):
                    raise ValueError(
                        f"{source}: {suite}.min_n cannot weaken the evidence floor"
                    )
            elif (
                not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)
                or not math.isfinite(float(threshold))
                or not 0.0 <= float(threshold) <= 1.0
            ):
                raise ValueError(f"{source}: invalid threshold {suite}.{metric}")
        # Exact BEq floors must not exceed soft structural similarity.
        struct = float(thresholds["structural_similarity"])
        if float(thresholds["ast_beq_rate"]) > struct:
            raise ValueError(f"{source}: {suite}.ast_beq_rate cannot exceed structural")
        if float(thresholds["canonical_beq_rate"]) > float(thresholds["ast_beq_rate"]):
            raise ValueError(
                f"{source}: {suite}.canonical_beq_rate cannot exceed ast_beq_rate"
            )
    meaningful = value.get("meaningful_metric_policy")
    if (
        not isinstance(meaningful, dict)
        or meaningful.get("active_primary") != "meaningful_program_v1"
        or meaningful.get("threshold_version") != version
        or (meaningful.get("meaningful_program_v1") or {}).get("wire_field")
        != "meaningful_program_rate"
        or (meaningful.get("meaningful_program_v1") or {}).get("thresholds")
        != "DEFAULT_SHIP_GATES"
        or not isinstance(meaningful.get("binding_aware_meaningful_v2"), dict)
        or (meaningful.get("binding_aware_meaningful_v2") or {}).get("thresholds")
        is not None
        or (meaningful.get("binding_aware_meaningful_v2") or {}).get("status")
        != "candidate_pending_calibration"
        or not isinstance(meaningful.get("semantic_fidelity"), dict)
        or (meaningful.get("semantic_fidelity") or {}).get("version") is None
    ):
        raise ValueError(f"{source}: invalid meaningful-metric provenance")
    return value


_POLICY = load_ship_gate_policy()
DEFAULT_MIN_SUITE_N = int(_POLICY["default_min_suite_n"])
DEFAULT_SHIP_GATES: dict[str, dict[str, float]] = _POLICY["suites"]
MEANINGFUL_METRIC_POLICY: dict[str, Any] = _POLICY["meaningful_metric_policy"]


def _meaningful_metric_policy(
    policy: dict[str, dict[str, float]], *, custom: bool
) -> dict[str, Any]:
    policy_id = str(MEANINGFUL_METRIC_POLICY.get("threshold_version") or "openui_ship_gates_v6")
    source = "DEFAULT_SHIP_GATES"
    if custom:
        encoded = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        policy_id = f"custom:{hashlib.sha256(encoded).hexdigest()}"
        source = "request_thresholds"
    return {
        **MEANINGFUL_METRIC_POLICY,
        "threshold_version": policy_id,
        "meaningful_program_v1": {
            **MEANINGFUL_METRIC_POLICY["meaningful_program_v1"],
            "thresholds": source,
        },
    }


def _slim_suite(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Project a scoreboard suite onto the gated OpenUI metric fields."""
    slim = {
        "n": metrics.get("n"),
        "parse_rate": metrics.get("parse_rate"),
        "meaningful_program_rate": metrics.get("meaningful_program_rate"),
        "meaningful_program_v1_rate": metrics.get(
            "meaningful_program_v1_rate",
            metrics.get("meaningful_program_rate"),
        ),
        "binding_aware_meaningful_v2_rate_strict": metrics.get(
            "binding_aware_meaningful_v2_rate_strict"
        ),
        "binding_aware_meaningful_v2_rate_coverage_conditioned": metrics.get(
            "binding_aware_meaningful_v2_rate_coverage_conditioned"
        ),
        "binding_aware_meaningful_v2_coverage": metrics.get(
            "binding_aware_meaningful_v2_coverage"
        ),
        "syntax_parse_rate": metrics.get("syntax_parse_rate"),
        "placeholder_fidelity": metrics.get("placeholder_fidelity"),
        "placeholder_validity": metrics.get("placeholder_validity"),
        "structural_similarity": metrics.get("structural_similarity"),
        "component_type_recall": metrics.get("component_type_recall"),
        "ast_beq_rate": metrics.get("ast_beq_rate", metrics.get("exact_match")),
        "canonical_beq_rate": metrics.get("canonical_beq_rate"),
        "certificate_equivalence_rate": metrics.get("certificate_equivalence_rate"),
        "reward_score": metrics.get("reward_score"),
    }
    if (
        slim["meaningful_program_rate"] is None
        and metrics.get("syntax_parse_rate") is None
    ):
        # Historical scoreboards used parse_rate for meaningful-program
        # quality. New scoreboards persist both metrics explicitly.
        slim["meaningful_program_rate"] = metrics.get("parse_rate")
    # Historical scoreboards used exact_match for structure BEq.
    if slim["ast_beq_rate"] is None and metrics.get("exact_match") is not None:
        slim["ast_beq_rate"] = metrics.get("exact_match")
    return slim


def _certificates_present(metrics: Mapping[str, Any]) -> bool:
    for key in ("certificates_compared", "certificates_emitted"):
        raw = metrics.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
            return True
    return False


def _certificate_integrity_failures(
    suites: Mapping[str, Mapping[str, Any]],
    *,
    floor: float = CERTIFICATE_EQUIVALENCE_FLOOR,
) -> list[str]:
    """Conditional integrity: certificate BEq when certificates were compared."""

    failures: list[str] = []
    for suite_name, metrics in suites.items():
        if not _certificates_present(metrics):
            continue
        replay_failures = metrics.get("certificate_replay_failures")
        if (
            isinstance(replay_failures, (int, float))
            and not isinstance(replay_failures, bool)
            and replay_failures > 0
        ):
            failures.append(
                f"{suite_name}:certificate_replay_failures "
                f"actual={int(replay_failures)} need==0"
            )
        rate = metrics.get("certificate_equivalence_rate")
        if rate is None:
            failures.append(
                f"{suite_name}:certificate_equivalence_rate "
                f"actual=None need>={floor}"
            )
        elif not (
            isinstance(rate, (int, float))
            and not isinstance(rate, bool)
            and float(rate) >= floor
        ):
            failures.append(
                f"{suite_name}:certificate_equivalence_rate "
                f"actual={rate} need>={floor}"
            )
    return failures


def evaluate_ship_gates(
    suites: dict[str, dict[str, Any]],
    *,
    thresholds: dict[str, dict[str, float]] | None = None,
    suite_reachability: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Check every suite present in `suites` against the ship policy.

    Missing metrics fail that threshold. Suites absent from the scoreboard are
    reported as `missing_suite` failures (cannot claim pass without evidence).

    When `suite_reachability` is provided (SLM-299 edit-space reachability
    audit), gated suites below 1.0 reachable fail as measurement integrity.

    When a suite reports compared/emitted certificates, certificate digest
    equivalence (BEq on certificates) and zero replay failures are required.
    """
    policy = thresholds or DEFAULT_SHIP_GATES
    actual, checks, failures, categorized = run_gate_checks(
        suites,
        policy,
        normalize_suite=_slim_suite,
        default_min_n=DEFAULT_MIN_SUITE_N,
        suite_reachability=suite_reachability,
    )
    cert_failures = _certificate_integrity_failures(suites)
    if cert_failures:
        failures = list(failures) + cert_failures
        categorized = {
            **categorized,
            "measurement_integrity_failures": list(
                categorized.get("measurement_integrity_failures") or []
            )
            + cert_failures,
        }
        for item in cert_failures:
            gate_id = item.split(" actual=")[0]
            checks[gate_id] = False
    return {
        "authority": "Python preview; durable ship verdicts require AgentEvals assertions",
        "policy": policy,
        "meaningful_metric_policy": _meaningful_metric_policy(
            policy, custom=bool(thresholds)
        ),
        "actual": actual,
        "gates": checks,
        "failures": failures,
        **categorized,
        "pass": all(checks.values()) if checks else False,
        "note": (
            "Honest ship gates require all policy suites and score structure + "
            "semantic fidelity only (meaningful_program_rate / structural_similarity / "
            "component_type_recall / ast_beq_rate / canonical_beq_rate / "
            "placeholder_fidelity / reward_score). component_type_recall is the "
            "semantic-density floor; ast_beq_rate and canonical_beq_rate are BEq "
            "analogues on AST/grammar pairs — soft similarity or syntax parse alone "
            "cannot promote. When certificates are compared, certificate_equivalence "
            "must hold (digest BEq) with zero replay failures. DESIGN.md style lint "
            "is never a ship gate. When suite_reachability is supplied, gated suites "
            "below 1.0 reachable fail as reachability_unproven (measurement integrity). "
            "See docs/design/adversarial-review.md, docs/design/structure-only-eval.md, "
            "and docs/design/semantic-fidelity-ship-gates.md."
        ),
    }


def write_ship_gates(
    run_dir: Path | str,
    suites: dict[str, dict[str, Any]],
    *,
    thresholds: dict[str, dict[str, float]] | None = None,
    suite_reachability: dict[str, float] | None = None,
    evals_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the AgentEvals-authoritative gates.json payload."""
    from slm_training.versioning import build_version_stamp

    payload = evaluate_ship_gates(
        suites,
        thresholds=thresholds,
        suite_reachability=suite_reachability,
    )
    policy = thresholds or DEFAULT_SHIP_GATES
    _, raw_criteria = build_gate_criteria(
        suites,
        policy,
        normalize_suite=_slim_suite,
        default_min_n=DEFAULT_MIN_SUITE_N,
        suite_reachability=suite_reachability,
    )
    for suite_name, metrics in suites.items():
        if not _certificates_present(metrics):
            continue
        raw_criteria.append(
            {
                "id": f"{suite_name}:certificate_equivalence_rate",
                "suite": suite_name,
                "actual": metrics.get("certificate_equivalence_rate"),
                "operator": "gte",
                "expected": CERTIFICATE_EQUIVALENCE_FLOOR,
            }
        )
        raw_criteria.append(
            {
                "id": f"{suite_name}:certificate_replay_failures",
                "suite": suite_name,
                "actual": metrics.get("certificate_replay_failures", 0),
                "operator": "eq",
                "expected": 0,
            }
        )
    observed = {
        str(item.get("id")): item
        for item in (evals_result.get("criteria") or {}).get("results", [])
    }
    checks = {
        str(item["id"]): (
            observed.get(str(item["id"]), {}).get("pass") is True
            and all(
                observed[str(item["id"])].get(field) == item.get(field)
                for field in ("actual", "operator", "expected")
            )
        )
        for item in raw_criteria
    }
    failed_criteria = [item for item in raw_criteria if not checks[str(item["id"])]]
    categorized = {
        "evidence_volume_failures": [],
        "measurement_integrity_failures": [],
        "quality_threshold_failures": [],
        "runtime_failures": [],
    }
    for item in failed_criteria:
        categorized[criterion_failure_category(item)].append(criterion_failure(item))
    payload.update(
        {
            "authority": "AgentEvals assertions",
            "gates": checks,
            "failures": [criterion_failure(item) for item in failed_criteria],
            **categorized,
            "pass": bool(checks)
            and all(checks.values())
            and evals_result.get("authority") == "AgentEvals assertions"
            and (evals_result.get("criteria") or {}).get("pass") is True
            and int((evals_result.get("runner") or {}).get("execution_errors") or 0)
            == 0,
            "evals": {
                "format": evals_result.get("format"),
                "spec": evals_result.get("spec"),
                "criteria": evals_result.get("criteria"),
                "runner": evals_result.get("runner"),
            },
        }
    )
    payload["version_stamp"] = build_version_stamp(
        "gates.ship",
        "harness.core",
        "evals.meaningful_program",
        "evals.semantic_fidelity",
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "gates.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["output"] = str(path)
    return payload
