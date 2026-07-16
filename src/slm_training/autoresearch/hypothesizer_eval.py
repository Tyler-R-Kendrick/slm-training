"""Frozen AgentV evaluation for hypothesis-matrix quality and feedback lineage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_training.autoresearch.schemas import (
    HypothesizerBenchmarkReport,
    HypothesisMatrix,
)
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.lineage.records import content_sha


def evaluate_hypothesizer(
    cases_path: Path | str,
    predictions_path: Path | str,
    *,
    run_dir: Path | str,
    hypothesizer_id: str,
    pass_threshold: float = 0.8,
    human_approved: bool = False,
) -> HypothesizerBenchmarkReport:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("hypothesizer benchmark requires frozen cases")
    predictions = [
        json.loads(line)
        for line in Path(predictions_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row.get("case_id")): row for row in predictions}
    scored: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        failures: list[str] = []
        matrix = None
        try:
            matrix = HypothesisMatrix.model_validate(by_id.get(case_id, {}).get("matrix"))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"invalid HypothesisMatrix: {exc}")
        valid = (
            matrix is not None
            and len(matrix.hypotheses) >= int(case.get("min_hypotheses", 5))
            and matrix.campaign_id == case["campaign_id"]
            and matrix.evidence_snapshot_id == case["evidence_snapshot_id"]
            and all(
                item.experiment.campaign_id == matrix.campaign_id
                for item in matrix.hypotheses
            )
        )
        evidence_roles = {
            str(item["uri"]): str(item["role"])
            for item in case.get("evidence") or []
        }
        used_roles = {
            use.role
            for candidate in (matrix.hypotheses if matrix else ())
            for use in candidate.evidence_uses
            if evidence_roles.get(use.citation) == use.role
        }
        required_roles = set(case.get("required_roles") or [])
        grounded = bool(matrix) and required_roles <= used_roles and all(
            evidence_roles.get(use.citation) == use.role
            for candidate in matrix.hypotheses
            for use in candidate.evidence_uses
        )
        novel = bool(matrix) and any(
            candidate.novelty.transition_kind == "regime_transition_candidate"
            and candidate.novelty.residual_elements
            for candidate in matrix.hypotheses
        )
        recommended = (
            next(
                (
                    item.experiment
                    for item in matrix.hypotheses
                    if item.experiment.experiment_id
                    == matrix.recommended_experiment_id
                ),
                None,
            )
            if matrix
            else None
        )
        expected_knobs = set(case.get("expected_knobs") or [])
        actionable = bool(recommended) and bool(
            set(recommended.knobs.model_dump(exclude_none=True)) & expected_knobs
        )
        expected_feedback = set(case.get("feedback_ids") or [])
        expected_predecessor = case.get("predecessor_matrix_id")
        feedback_lineage = bool(matrix) and (
            set(matrix.feedback_ids) == expected_feedback
            and matrix.predecessor_matrix_id == expected_predecessor
        )
        checks = {
            "valid": valid,
            "grounded": grounded,
            "novel": novel,
            "actionable": actionable,
            "feedback_lineage": feedback_lineage,
        }
        failures.extend(name for name, passed in checks.items() if not passed)
        scored.append(
            {
                "id": case_id,
                "criteria": str(case["criteria"]),
                "pass": not failures,
                "failures": failures,
                "result": checks,
                "metadata": {"frozen_case": True},
            }
        )

    def rate(key: str) -> float:
        return sum(bool(row["result"][key]) for row in scored) / len(scored)

    agentv = publish_agentv_evaluation(
        run_dir,
        name=f"autoresearch-hypothesizer-{hypothesizer_id}",
        claim="grounded_novel_feedback_linked_hypothesis_matrices",
        cases=scored,
    )
    agentv = _relative_paths(agentv, Path(__file__).resolve().parents[3])
    rates = {
        "valid_matrix_rate": rate("valid"),
        "grounded_rate": rate("grounded"),
        "novel_rate": rate("novel"),
        "actionable_rate": rate("actionable"),
        "feedback_lineage_rate": rate("feedback_lineage"),
    }
    passed = min(rates.values()) >= pass_threshold
    identity = {
        "hypothesizer_id": hypothesizer_id,
        "cases_sha": content_sha(cases),
        "predictions_sha": content_sha(predictions),
        "rates": rates,
    }
    return HypothesizerBenchmarkReport(
        benchmark_id=f"hypothesizer-{content_sha(identity)[:16]}",
        hypothesizer_id=hypothesizer_id,
        cases=len(scored),
        pass_threshold=pass_threshold,
        passed=passed,
        human_approved=human_approved,
        promotable=passed and human_approved,
        agentv=agentv,
        **rates,
    )


def _relative_paths(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _relative_paths(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [_relative_paths(child, root) for child in value]
    if isinstance(value, str):
        try:
            path = Path(value)
            return str(path.relative_to(root)) if path.is_absolute() else value
        except ValueError:
            return value
    return value
