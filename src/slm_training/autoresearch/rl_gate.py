"""Fail-closed competence gate shared by every reinforcement-learning path."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from slm_training.autoresearch.schemas import RLReadinessReport
from slm_training.lineage.evaluation_snapshot import REQUIRED_SUITES
from slm_training.lineage.records import content_sha


def assess_rl_readiness(evaluation: Path | str | dict[str, Any]) -> RLReadinessReport:
    """Require frozen full-suite AgentEvals gates and useful reward variance."""
    if isinstance(evaluation, (str, Path)):
        raw = Path(evaluation).read_bytes()
        payload = json.loads(raw)
        evaluation_sha = hashlib.sha256(raw).hexdigest()
    else:
        payload = dict(evaluation)
        evaluation_sha = content_sha(payload)

    suites = payload.get("suites") or payload.get("scoreboard", {}).get("suites") or {}
    snapshot = payload.get("evaluation_snapshot") or payload.get("snapshot") or {}
    metadata = snapshot.get("metadata") or payload.get("snapshot_metadata") or {}
    frozen = metadata.get("kind") == "frozen_production_evaluation"
    suite_sizes = {
        name: int((suites.get(name) or {}).get("n") or 0) for name in REQUIRED_SUITES
    }
    declared_sizes = metadata.get("suite_sizes") or {}
    for name in REQUIRED_SUITES:
        suite_sizes[name] = max(suite_sizes[name], int(declared_sizes.get(name) or 0))
    gates = payload.get("gates") or payload.get("ship_gates") or {}
    evals = payload.get("evals") or gates.get("evals") or {}
    criteria = evals.get("criteria") if isinstance(evals, dict) else {}
    criteria = criteria if isinstance(criteria, dict) else {}
    runner = evals.get("runner") if isinstance(evals, dict) else {}
    runner = runner if isinstance(runner, dict) else {}
    eval_criteria_pass = (
        criteria.get("pass") is True
        and int(criteria.get("total") or 0) > 0
        and int(runner.get("execution_errors") or 0) == 0
    )
    ship_gates_pass = (
        isinstance(gates, dict)
        and gates.get("authority") == "AgentEvals assertions"
        and gates.get("pass") is True
        and eval_criteria_pass
    )
    rewards = [
        float(value)
        for value in payload.get("reward_samples", [])
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    variance = statistics.pvariance(rewards) if len(rewards) >= 2 else 0.0
    failures: list[str] = []
    if not frozen:
        failures.append("evaluation snapshot is not frozen production evidence")
    if int(metadata.get("human_feedback_holdout_n") or 0) < 1:
        failures.append("missing never-trained human-feedback holdout")
    for suite in REQUIRED_SUITES:
        if suite not in suites:
            failures.append(f"missing suite: {suite}")
    if suite_sizes.get("rico_held", 0) < 1500:
        failures.append("rico_held requires n>=1500")
    if not ship_gates_pass:
        failures.append("AgentEvals-authoritative ship gates did not pass")
    if not eval_criteria_pass:
        failures.append("AgentEvals criteria did not pass")
    if len(rewards) < 2 or variance <= 0.0:
        failures.append("reward samples must have nonzero variance")
    identity = {
        "evaluation_sha256": evaluation_sha,
        "suite_sizes": suite_sizes,
        "ship_gates_pass": ship_gates_pass,
        "eval_criteria_pass": eval_criteria_pass,
        "reward_sample_count": len(rewards),
        "reward_variance": variance,
        "failures": failures,
    }
    return RLReadinessReport(
        report_id=f"rl-ready-{content_sha(identity)[:16]}",
        evaluation_sha256=evaluation_sha,
        frozen_snapshot=frozen,
        required_suites=tuple(REQUIRED_SUITES),
        suite_sizes=suite_sizes,
        ship_gates_pass=ship_gates_pass,
        eval_criteria_pass=eval_criteria_pass,
        reward_sample_count=len(rewards),
        reward_variance=variance,
        approved=not failures,
        failures=tuple(failures),
    )


def load_rl_readiness(path: Path | str) -> RLReadinessReport:
    return RLReadinessReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def assert_rl_ready(report: RLReadinessReport | Path | str | None) -> RLReadinessReport:
    if report is None:
        raise ValueError(
            "RL is locked: provide an approved --rl-readiness-report produced from "
            "the frozen full-suite evaluation bundle"
        )
    resolved = load_rl_readiness(report) if isinstance(report, (str, Path)) else report
    if not resolved.approved or resolved.failures:
        detail = "; ".join(resolved.failures) or "report is not approved"
        raise ValueError(f"RL is locked: {detail}")
    if not (
        resolved.frozen_snapshot
        and resolved.ship_gates_pass
        and resolved.eval_criteria_pass
        and resolved.suite_sizes.get("rico_held", 0) >= 1500
        and resolved.reward_variance > 0
    ):
        raise ValueError("RL is locked: readiness report is internally inconsistent")
    return resolved


def write_rl_readiness(
    path: Path | str, report: RLReadinessReport, *, overwrite: bool = False
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(report.model_dump_json(indent=2) + "\n")
    return path
