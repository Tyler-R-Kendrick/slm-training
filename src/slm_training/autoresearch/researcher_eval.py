"""Frozen, publishable evaluation for researcher proposal quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_training.autoresearch.schemas import (
    ExperimentSpec,
    ResearcherBenchmarkReport,
)
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.lineage.records import content_sha


def evaluate_researcher(
    cases_path: Path | str,
    predictions_path: Path | str,
    *,
    run_dir: Path | str,
    researcher_id: str,
    pass_threshold: float = 0.8,
    human_approved: bool = False,
) -> ResearcherBenchmarkReport:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in Path(predictions_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row.get("case_id")): row for row in predictions}
    scored: list[dict[str, Any]] = []
    changed_signatures: list[tuple[str, ...]] = []
    for case in cases:
        case_id = str(case["case_id"])
        raw = by_id.get(case_id, {}).get("experiment")
        failures: list[str] = []
        experiment = None
        try:
            experiment = ExperimentSpec.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"invalid ExperimentSpec: {exc}")
        known = set(case.get("evidence_uris") or [])
        grounded = bool(experiment and set(experiment.citations) & known)
        if not grounded:
            failures.append("no citation matches frozen evidence")
        expected = set(case.get("expected_knobs") or [])
        changed = (
            set(experiment.knobs.model_dump(exclude_none=True)) if experiment else set()
        )
        actionable = bool(experiment and experiment.stop_conditions and expected & changed)
        if not actionable:
            failures.append("proposal does not change an expected bounded knob")
        signature = tuple(sorted(changed))
        novel = bool(signature and signature not in changed_signatures)
        changed_signatures.append(signature)
        if not novel:
            failures.append("proposal duplicates an earlier knob signature")
        scored.append(
            {
                "id": case_id,
                "criteria": str(case["criteria"]),
                "pass": not failures,
                "failures": failures,
                "result": {
                    "grounded": grounded,
                    "valid": experiment is not None,
                    "novel": novel,
                    "actionable": actionable,
                },
                "metadata": {"frozen_case": True},
            }
        )
    count = len(scored)

    def rate(key: str) -> float:
        return sum(bool(row["result"][key]) for row in scored) / count

    agentv = publish_agentv_evaluation(
        run_dir,
        name=f"autoresearch-researcher-{researcher_id}",
        claim="grounded_novel_actionable_experiment_proposals",
        cases=scored,
    )
    agentv = _relative_paths(agentv, Path(__file__).resolve().parents[3])
    rates = {
        "grounded_rate": rate("grounded"),
        "valid_spec_rate": rate("valid"),
        "novel_rate": rate("novel"),
        "actionable_rate": rate("actionable"),
    }
    passed = min(rates.values()) >= pass_threshold and all(row["pass"] for row in scored)
    identity = {
        "researcher_id": researcher_id,
        "cases_sha": content_sha(cases),
        "predictions_sha": content_sha(predictions),
        "rates": rates,
    }
    return ResearcherBenchmarkReport(
        benchmark_id=f"researcher-{content_sha(identity)[:16]}",
        researcher_id=researcher_id,
        cases=count,
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
            return str(Path(value).relative_to(root)) if Path(value).is_absolute() else value
        except ValueError:
            return value
    return value
