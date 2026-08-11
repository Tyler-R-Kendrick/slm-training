"""HARN-09 / SLM-560: proposition-preserving self-healing + mutation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from slm_training.formal.judgment import (
    JudgmentOutcome,
    SolverJudgmentV1,
    UnknownReason,
    WitnessPayloadV1,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.replay import build_revmath_replay_bundle
from slm_training.harnesses.reasoning.revmath.runner import RevmathRunRecord
from slm_training.harnesses.reasoning.revmath.schemas import (
    REVMATH_MUTABLE_REPAIR_KNOBS,
    AxisCertificateRefV1,
    RevmathBaseTheoryV1,
    RevmathBudgetV1,
    RevmathCampaignBindingV1,
    RevmathCorpusIdentityV1,
    RevmathFourAxisAnalysisV1,
    RevmathProofArtifactRefV1,
    RevmathPropositionIdentityV1,
    RevmathResultV1,
    RevmathSchemaError,
    RevmathTaskV1,
    RevmathVerifierJudgeIdentityV1,
    assert_repair_preserves_identity,
)
from slm_training.harnesses.reasoning.revmath.self_healing import (
    REVMATH_FROZEN_REPAIR_FIELDS,
    RevmathRepairKnobStateV1,
    assert_serialized_repair_freeze,
    propose_repair_attempts,
    reject_forbidden_repair,
    run_self_healing,
)
from tests.casefiles import case_values

REPO = Path(__file__).resolve().parents[3]
_HEX = "ab" * 32
_HEX2 = "cd" * 32
_HEX3 = "ef" * 32


def _prop(statement: str = "forall n, n + 0 = n") -> RevmathPropositionIdentityV1:
    return RevmathPropositionIdentityV1(
        proposition_id="prop.add_zero",
        statement=statement,
        statement_sha256=content_sha(statement),
    )


def _task(**overrides: object) -> RevmathTaskV1:
    base: dict[str, Any] = {
        "task_id": "task.forward.heal",
        "task_kind": "forward_theorem",
        "proposition": _prop(),
        "base_theory": RevmathBaseTheoryV1(
            base_theory_id="RCA0",
            allowed_assumption_ids=("ISigma1",),
            interpretation_status="explicit_interpretation",
        ),
        "theorem_direction": "forward",
        "corpus": RevmathCorpusIdentityV1(
            corpus_id="corpus.rm.v1",
            corpus_sha256=_HEX,
            entry_id="entry.heal",
        ),
        "budget": RevmathBudgetV1(
            max_wall_seconds=30.0,
            max_tactic_steps=100,
            stopping_rule_ids=("wall_or_steps",),
        ),
        "verifier_judge": RevmathVerifierJudgeIdentityV1(
            verifier_id="lean.kernel",
            judge_id="solver_judgment.v1",
        ),
        "campaign": RevmathCampaignBindingV1(
            campaign_id="camp.rm",
            experiment_id="exp.rm.heal",
            campaign_manifest_sha256=_HEX2,
            locked_arm_ids=("control", "candidate"),
        ),
    }
    base.update(overrides)
    return RevmathTaskV1(**base)


def _four_axis() -> RevmathFourAxisAnalysisV1:
    return RevmathFourAxisAnalysisV1(
        assumption_strength=AxisCertificateRefV1(
            axis="assumption_strength", status="not_claimed"
        ),
        computability=AxisCertificateRefV1(axis="computability", status="not_claimed"),
        resource_bounds=AxisCertificateRefV1(
            axis="resource_bounds", status="not_claimed"
        ),
        implementation_refinement=AxisCertificateRefV1(
            axis="implementation_refinement", status="not_claimed"
        ),
    )


def _unknown_result(task: RevmathTaskV1, *, result_id: str) -> RevmathResultV1:
    return RevmathResultV1(
        result_id=result_id,
        task_id=task.task_id,
        task_identity_digest=task.identity_digest(),
        judgment=SolverJudgmentV1(
            outcome=JudgmentOutcome.UNKNOWN,
            payload=UnknownReason.INCONCLUSIVE,
            checked=False,
        ).to_dict(),
        proof=None,
        four_axis=_four_axis(),
    )


def _witnessed_result(task: RevmathTaskV1, *, result_id: str, proof_sha: str) -> RevmathResultV1:
    return RevmathResultV1(
        result_id=result_id,
        task_id=task.task_id,
        task_identity_digest=task.identity_digest(),
        judgment=SolverJudgmentV1(
            outcome=JudgmentOutcome.WITNESSED,
            payload=WitnessPayloadV1(witness_digest=proof_sha, source="lean"),
            checked=True,
        ).to_dict(),
        proof=RevmathProofArtifactRefV1(
            proof_sha256=proof_sha,
            checker_id="revmath.hermetic",
            checker_report_sha256=_HEX3,
        ),
        four_axis=_four_axis(),
    )


def _run_record(task: RevmathTaskV1, result: RevmathResultV1) -> RevmathRunRecord:
    identity_snapshot = {
        "task_id": task.task_id,
        "task_identity_digest": task.identity_digest(),
        "proposition_id": task.proposition.proposition_id,
        "statement_sha256": task.proposition.statement_sha256,
        "base_theory_id": task.base_theory.base_theory_id,
        "allowed_assumption_ids": list(task.base_theory.allowed_assumption_ids),
        "campaign_id": task.campaign.campaign_id,
        "experiment_id": task.campaign.experiment_id,
        "campaign_manifest_sha256": task.campaign.campaign_manifest_sha256,
        "locked_arm_ids": list(task.campaign.locked_arm_ids),
        "task_canonical_hash": task.canonical_hash(),
    }
    capture = {
        "note": "test",
        "timed_out": False,
        "stdout_sha256": content_sha(""),
        "stderr_sha256": content_sha(""),
    }
    return RevmathRunRecord(
        task=task,
        result=result,
        replay_bundle=build_revmath_replay_bundle(
            task=task,
            result=result,
            capture=capture,
            identity_snapshot=identity_snapshot,
        ),
        capture=capture,
        identity_snapshot=identity_snapshot,
        decode_stats_record={},
    )


def test_frozen_fields_match_parity_matrix() -> None:
    parity = json.loads(
        (REPO / "src/slm_training/resources/revmath_harness_parity.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(parity["self_healing"]["must_freeze"]) == set(REVMATH_FROZEN_REPAIR_FIELDS)
    overlap = set(parity["self_healing"]["may_modify"]) & set(
        parity["self_healing"]["must_freeze"]
    )
    assert not overlap
    assert set(parity["self_healing"]["may_modify"]) == set(REVMATH_MUTABLE_REPAIR_KNOBS)


@pytest.mark.parametrize(
    "mutation",
    case_values(__file__, "test_mutation_forbidden_repairs_fail_closed"),
)
def test_mutation_forbidden_repairs_fail_closed(mutation: str) -> None:
    with pytest.raises(RevmathSchemaError, match="forbidden|frozen|unknown"):
        reject_forbidden_repair(mutation)


def test_propose_repairs_only_mutable_knobs() -> None:
    task = _task()
    state = RevmathRepairKnobStateV1()
    proposals = propose_repair_attempts(
        task=task,
        knob_state=state,
        failure={"first_failure_family": "search_or_pruning_regret"},
        max_proposals=5,
    )
    assert proposals
    assert all(p.knob in REVMATH_MUTABLE_REPAIR_KNOBS for p in proposals)
    assert proposals[0].failure_family == "search_or_pruning_regret"


def test_successful_repair_proves_original_locked_task() -> None:
    task = _task()
    identity = task.identity_digest()
    initial = _run_record(task, _unknown_result(task, result_id="res.0"))
    calls: list[str] = []

    def runner(t: RevmathTaskV1, **kwargs: Any) -> RevmathRunRecord:
        del kwargs
        assert t.identity_digest() == identity
        calls.append("run")
        # First repair re-run succeeds on the same locked task.
        return _run_record(
            t, _witnessed_result(t, result_id=f"res.{len(calls)}", proof_sha=_HEX2)
        )

    session = run_self_healing(
        task,
        initial_run=initial,
        max_attempts=3,
        runner=runner,
        failure={"first_failure_family": "search_coverage_or_candidate_absence"},
    )
    assert session.status == "repaired"
    assert session.task_identity_digest == identity
    assert session.final_result is not None
    assert session.final_result.solver_judgment().outcome is JudgmentOutcome.WITNESSED
    assert session.attempts
    repair = session.attempts[0].repair
    assert repair is not None
    assert_repair_preserves_identity(task, repair)
    assert session.attempts[0].invalidated_proof_sha256s == ()
    assert session.semantic_lineage["extends"] == "semantic_repair/v1"


def test_cycle_detection_terminates_unknown() -> None:
    task = _task()
    initial = _run_record(task, _unknown_result(task, result_id="res.u0"))

    def runner(t: RevmathTaskV1, **kwargs: Any) -> RevmathRunRecord:
        del kwargs
        return _run_record(t, _unknown_result(t, result_id="res.loop"))

    # Force the same proposal twice by allowing two attempts with deterministic knobs.
    session = run_self_healing(
        task,
        initial_run=initial,
        initial_knob_state=RevmathRepairKnobStateV1(),
        max_attempts=3,
        runner=runner,
        failure={"first_failure_family": "search_or_pruning_regret"},
    )
    # After first unknown attempt, the next equivalent proposal must cycle-stop.
    # With changing salts per attempt index inside propose, signatures differ unless
    # we re-propose the same delta — simulate by feeding a runner that never
    # advances state and a tiny max that still records lineage, then assert that
    # repeating an identical knob change raises via a second session seed.
    assert session.status in {"unknown", "budget_exhausted", "cycle_detected"}
    assert task.identity_digest() == session.task_identity_digest
    # Explicit cycle: run again with a knob state already matching the only proposal.
    state = RevmathRepairKnobStateV1()
    proposals = propose_repair_attempts(
        task=task,
        knob_state=state,
        failure={"first_failure_family": "local_scoring_regret"},
        max_proposals=1,
    )
    assert len(proposals) == 1
    # Apply once, then craft a controller path that sees the same signature twice.
    seen: list[str] = []

    def cycling_runner(t: RevmathTaskV1, **kwargs: Any) -> RevmathRunRecord:
        del kwargs
        seen.append("x")
        return _run_record(
            t,
            _unknown_result(t, result_id=f"res.c{len(seen)}"),
        )

    # Monkeypatch propose to always return the same knob/after_value.
    import slm_training.harnesses.reasoning.revmath.self_healing as heal

    fixed = proposals

    def _fixed_propose(**kwargs: Any):
        del kwargs
        return fixed

    original = heal.propose_repair_attempts
    heal.propose_repair_attempts = _fixed_propose  # type: ignore[assignment]
    try:
        cycled = run_self_healing(
            task,
            initial_run=initial,
            max_attempts=3,
            runner=cycling_runner,
        )
    finally:
        heal.propose_repair_attempts = original  # type: ignore[assignment]
    assert cycled.status == "cycle_detected"
    assert any(a.outcome == "cycle_detected" for a in cycled.attempts)
    assert cycled.final_result is not None
    assert cycled.final_result.solver_judgment().outcome is JudgmentOutcome.UNKNOWN


def test_stale_proof_invalidated_on_rerun() -> None:
    task = _task()
    initial = _run_record(task, _unknown_result(task, result_id="res.stale0"))
    phase = {"n": 0}

    def runner(t: RevmathTaskV1, **kwargs: Any) -> RevmathRunRecord:
        del kwargs
        phase["n"] += 1
        if phase["n"] == 1:
            return _run_record(t, _unknown_result(t, result_id="res.u1"))
        return _run_record(
            t, _witnessed_result(t, result_id="res.ok", proof_sha=_HEX2)
        )

    session = run_self_healing(
        task,
        initial_run=initial,
        max_attempts=3,
        runner=runner,
        failure={"first_failure_family": "local_scoring_regret"},
    )
    assert session.status == "repaired"
    assert len(session.attempts) >= 2
    # Second attempt invalidates the synthetic after-proof digest from attempt 1.
    assert session.attempts[1].invalidated_proof_sha256s
    assert (
        session.attempts[1].invalidated_proof_sha256s
        == (session.attempts[0].repair.after_proof_sha256,)  # type: ignore[union-attr]
    )
    assert session.attempts[1].parent_attempt_id == session.attempts[0].attempt_id


def test_serialized_artifact_cannot_rewrite_identity() -> None:
    task = _task()
    initial = _run_record(task, _unknown_result(task, result_id="res.s0"))

    def runner(t: RevmathTaskV1, **kwargs: Any) -> RevmathRunRecord:
        del kwargs
        return _run_record(
            t, _witnessed_result(t, result_id="res.s1", proof_sha=_HEX2)
        )

    session = run_self_healing(
        task, initial_run=initial, max_attempts=1, runner=runner
    )
    payload = session.attempts[0].to_dict()
    assert_serialized_repair_freeze(task, payload)

    bad = dict(payload)
    bad["repair"] = dict(payload["repair"])
    bad["repair"]["proposition_id"] = "prop.other"
    with pytest.raises(RevmathSchemaError, match="proposition"):
        assert_serialized_repair_freeze(task, bad)

    # Drifted digest in session envelope.
    with pytest.raises(RevmathSchemaError, match="digest"):
        assert_serialized_repair_freeze(
            task, {"task_identity_digest": _HEX3, "schema_version": "revmath_repair_attempt/v1"}
        )


def test_resource_allocation_stays_within_locked_budget() -> None:
    state = RevmathRepairKnobStateV1()
    ok = state.with_knob(
        "resource_allocation_within_locked_budget", {"search": 0.25, "check": 0.75}
    )
    assert sum(ok.resource_allocation_within_locked_budget.values()) == pytest.approx(1.0)
    with pytest.raises(RevmathSchemaError, match="fractions"):
        state.with_knob(
            "resource_allocation_within_locked_budget",
            {"search": 0.9, "check": 0.9},
        )


def test_no_repair_needed_when_already_decisive() -> None:
    task = _task()
    initial = _run_record(
        task, _witnessed_result(task, result_id="res.done", proof_sha=_HEX)
    )
    session = run_self_healing(task, initial_run=initial, max_attempts=2, runner=lambda t, **k: initial)
    assert session.status == "no_repair_needed"
    assert session.attempts == ()
