"""HARN-03 / SLM-536: deterministic revmath runner, replay, and report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.judgment import JudgmentOutcome, UnknownReason
from slm_training.harnesses.reasoning.revmath.plugins import (
    RevmathCheckPlan,
    default_plugin_registry,
    override_hermetic_mode,
    resolve_plugin,
)
from slm_training.harnesses.reasoning.revmath.report import build_revmath_report
from slm_training.harnesses.reasoning.revmath.replay import (
    assert_revmath_replay_integrity,
    build_revmath_replay_bundle,
)
from slm_training.harnesses.reasoning.revmath.runner import run_revmath_task
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathTaskV1,
    parse_revmath_task,
)

REPO = Path(__file__).resolve().parents[3]
FIXTURE_TASK = (
    REPO
    / "src"
    / "slm_training"
    / "resources"
    / "revmath"
    / "fixtures"
    / "hermetic_forward_theorem.task.json"
)


@pytest.fixture
def hermetic_task() -> RevmathTaskV1:
    return parse_revmath_task(json.loads(FIXTURE_TASK.read_text(encoding="utf-8")))


def _with_mode(task: RevmathTaskV1, mode: str) -> RevmathCheckPlan:
    plugin = resolve_plugin(task)
    assert plugin is not None
    plan = plugin.plan_check(task, lean_root=REPO, hermetic=True)
    return RevmathCheckPlan(
        command=override_hermetic_mode(plan.command, mode),
        cwd=plan.cwd,
        env=plan.env,
        requires_lean_tool=False,
        lean_project_root=None,
        checker_id=plan.checker_id,
    )


def test_hermetic_forward_is_deterministic(hermetic_task: RevmathTaskV1) -> None:
    first = run_revmath_task(hermetic_task, hermetic=True)
    second = run_revmath_task(hermetic_task, hermetic=True)
    assert first.result.judgment == second.result.judgment
    assert first.result.task_identity_digest == hermetic_task.identity_digest()
    assert second.result.task_identity_digest == hermetic_task.identity_digest()
    assert first.replay_bundle.bundle_hash == second.replay_bundle.bundle_hash
    assert first.result.solver_judgment().outcome is JudgmentOutcome.WITNESSED
    assert_revmath_replay_integrity(first.replay_bundle)
    assert first.identity_snapshot["statement_sha256"] == (
        hermetic_task.proposition.statement_sha256
    )
    assert first.identity_snapshot["campaign_manifest_sha256"] == (
        hermetic_task.campaign.campaign_manifest_sha256
    )


def test_timeout_stays_unknown_never_refuted(hermetic_task: RevmathTaskV1) -> None:
    task = hermetic_task.model_copy(
        update={
            "budget": hermetic_task.budget.model_copy(update={"max_wall_seconds": 0.2})
        }
    )
    record = run_revmath_task(
        task,
        hermetic=True,
        plan_override=_with_mode(task, "sleep"),
    )
    judgment = record.result.solver_judgment()
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    assert judgment.payload is UnknownReason.TIMEOUT
    assert record.result.proof is None


def test_malformed_proof_is_invalid(hermetic_task: RevmathTaskV1) -> None:
    record = run_revmath_task(
        hermetic_task,
        hermetic=True,
        plan_override=_with_mode(hermetic_task, "malformed"),
    )
    assert record.result.solver_judgment().outcome is JudgmentOutcome.INVALID
    assert record.result.proof is None


def test_incomplete_check_stays_unknown(hermetic_task: RevmathTaskV1) -> None:
    record = run_revmath_task(
        hermetic_task,
        hermetic=True,
        plan_override=_with_mode(hermetic_task, "incomplete"),
    )
    judgment = record.result.solver_judgment()
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    assert judgment.payload is UnknownReason.INCOMPLETE_COVERAGE


def test_unsupported_task_kind_stays_unknown(hermetic_task: RevmathTaskV1) -> None:
    task = hermetic_task.model_copy(update={"task_kind": "assumption_ablation"})
    record = run_revmath_task(task, hermetic=True)
    judgment = record.result.solver_judgment()
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    assert judgment.payload is UnknownReason.UNSUPPORTED_CAPABILITY


def test_missing_lean_tool_stays_unknown(
    hermetic_task: RevmathTaskV1, tmp_path: Path
) -> None:
    empty = tmp_path / "no-lean"
    empty.mkdir()
    record = run_revmath_task(hermetic_task, lean_root=empty, hermetic=False)
    judgment = record.result.solver_judgment()
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    assert judgment.payload is UnknownReason.MISSING_TOOL


def test_report_aggregates_without_mutating_campaign(
    hermetic_task: RevmathTaskV1,
) -> None:
    record = run_revmath_task(hermetic_task, hermetic=True)
    report = build_revmath_report(
        report_id="report.hermetic.1",
        tasks=[hermetic_task],
        results=[record.result],
        repair_ids=("repair.hist.ref.1",),
    )
    assert report.campaign_manifest_sha256 == (
        hermetic_task.campaign.campaign_manifest_sha256
    )
    assert report.judgment_counts["witnessed"] == 1
    assert report.repair_ids == ("repair.hist.ref.1",)
    assert hermetic_task.campaign.campaign_manifest_sha256 == (
        record.identity_snapshot["campaign_manifest_sha256"]
    )


def test_plugin_registry_covers_kinds_without_duplicating_runner() -> None:
    registry = default_plugin_registry()
    assert "forward_theorem" in registry
    assert "assumption_ablation" in registry


def test_replay_bundle_round_trip_fields(hermetic_task: RevmathTaskV1) -> None:
    record = run_revmath_task(hermetic_task, hermetic=True)
    rebuilt = build_revmath_replay_bundle(
        task=hermetic_task,
        result=record.result,
        capture=record.capture,
        identity_snapshot=record.identity_snapshot,
        repair_history_ref=None,
    )
    assert rebuilt.bundle_hash == record.replay_bundle.bundle_hash
    evidence = rebuilt.final_evidence
    assert evidence["proposition_sha256"] == hermetic_task.proposition.statement_sha256
    assert evidence["campaign_manifest_sha256"] == (
        hermetic_task.campaign.campaign_manifest_sha256
    )
