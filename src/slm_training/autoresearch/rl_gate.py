"""Fail-closed competence gate shared by every reinforcement-learning path."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from slm_training.autoresearch.schemas import RLReadinessReport
from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates
from slm_training.lineage.evaluation_snapshot import REQUIRED_SUITES
from slm_training.lineage.records import content_sha


def assess_rl_readiness(evaluation: Path | str | dict[str, Any]) -> RLReadinessReport:
    """Require frozen full-suite competence, AgentV, and useful reward variance."""
    if isinstance(evaluation, (str, Path)):
        evaluation_path = Path(evaluation)
        raw = evaluation_path.read_bytes()
        payload = json.loads(raw)
        evaluation_sha = hashlib.sha256(raw).hexdigest()
        evaluation_uri = str(evaluation_path.resolve())
    else:
        payload = dict(evaluation)
        evaluation_sha = content_sha(payload)
        evaluation_uri = None

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
    gates = evaluate_ship_gates(suites)
    agentv_pass = _agentv_pass(payload.get("agentv") or payload.get("agentv_result"))
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
    if not gates.get("pass"):
        failures.append("canonical honest ship gates did not pass")
    if not agentv_pass:
        failures.append("AgentV evaluation did not pass")
    if len(rewards) < 2 or variance <= 0.0:
        failures.append("reward samples must have nonzero variance")
    identity = {
        "evaluation_sha256": evaluation_sha,
        "suite_sizes": suite_sizes,
        "ship_gates_pass": bool(gates.get("pass")),
        "agentv_pass": agentv_pass,
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
        ship_gates_pass=bool(gates.get("pass")),
        agentv_pass=agentv_pass,
        reward_sample_count=len(rewards),
        reward_variance=variance,
        approved=not failures,
        failures=tuple(failures),
        evaluation_uri=evaluation_uri,
    )


def load_rl_readiness(path: Path | str) -> RLReadinessReport:
    return RLReadinessReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def assert_rl_ready(
    report: RLReadinessReport | Path | str | None,
    *,
    evaluation: Path | str | dict[str, Any] | None = None,
) -> RLReadinessReport:
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
        and resolved.agentv_pass
        and resolved.suite_sizes.get("rico_held", 0) >= 1500
        and resolved.reward_variance > 0
    ):
        raise ValueError("RL is locked: readiness report is internally inconsistent")
    evidence = evaluation or resolved.evaluation_uri
    if evidence is None:
        raise ValueError(
            "RL is locked: readiness report is not bound to a verifiable evaluation"
        )
    verified = assess_rl_readiness(evidence)
    if (
        not verified.approved
        or verified.evaluation_sha256 != resolved.evaluation_sha256
        or verified.model_dump(exclude={"evaluation_uri", "created_at"})
        != resolved.model_dump(exclude={"evaluation_uri", "created_at"})
    ):
        raise ValueError("RL is locked: readiness evidence does not match report")
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


def _agentv_pass(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("pass") is True or value.get("passed") is True or value.get("success") is True:
        return True
    summary = value.get("summary")
    if isinstance(summary, dict):
        total = int(summary.get("total") or 0)
        passed = int(summary.get("passed") or 0)
        errors = int(summary.get("executionErrors") or 0)
        failed = int(summary.get("failed") or 0)
        if total > 0 and passed == total and failed == 0 and errors == 0:
            return True
    return str(value.get("status") or "").lower() in {"pass", "passed", "success", "succeeded"}
