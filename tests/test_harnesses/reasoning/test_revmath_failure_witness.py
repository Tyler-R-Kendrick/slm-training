"""INTEG-05 / SLM-571: revmath failure → VerifierWitnessV1 adapter tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from slm_training.evals.semantic_failure import (
    WITNESS_SCHEMA_VERSION,
    VerifierLocalizationV1,
    VerifierWitnessV1,
    seal_verifier_witness,
    witness_integrity_ok,
)
from slm_training.formal.judgment import (
    JudgmentOutcome,
    SolverJudgmentV1,
    UnknownReason,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.failure_witness import (
    EVIDENCE_SCHEMA,
    FEEDBACK_SCHEMA,
    REVMATH_FAILURE_TAXONOMY_VERSION,
    RevmathFailureClass,
    RevmathFailureEvidenceV1,
    assert_diagnostic_cannot_prove_success,
    bounded_repair_actions_for,
    build_revmath_verifier_witness,
    feed_repair_session_to_disposition,
    route_revmath_failure,
    semantic_family_for_revmath_class,
)
from slm_training.harnesses.reasoning.revmath.replay import build_revmath_replay_bundle
from slm_training.harnesses.reasoning.revmath.runner import RevmathRunRecord
from slm_training.harnesses.reasoning.revmath.schemas import (
    AxisCertificateRefV1,
    RevmathBaseTheoryV1,
    RevmathBudgetV1,
    RevmathCampaignBindingV1,
    RevmathCorpusIdentityV1,
    RevmathFourAxisAnalysisV1,
    RevmathPropositionIdentityV1,
    RevmathResultV1,
    RevmathSchemaError,
    RevmathTaskV1,
    RevmathVerifierJudgeIdentityV1,
)
from slm_training.harnesses.reasoning.revmath.self_healing import (
    RevmathRepairKnobStateV1,
    run_self_healing,
)

REPO = Path(__file__).resolve().parents[3]
_HEX = "ab" * 32


def _prop(statement: str = "forall n, n + 0 = n") -> RevmathPropositionIdentityV1:
    return RevmathPropositionIdentityV1(
        proposition_id="prop.add_zero",
        statement=statement,
        statement_sha256=content_sha(statement),
    )


def _four_axis() -> RevmathFourAxisAnalysisV1:
    def axis(name: str) -> AxisCertificateRefV1:
        return AxisCertificateRefV1(axis=name, status="unknown_axis")  # type: ignore[arg-type]

    return RevmathFourAxisAnalysisV1(
        assumption_strength=axis("assumption_strength"),
        computability=axis("computability"),
        resource_bounds=axis("resource_bounds"),
        implementation_refinement=axis("implementation_refinement"),
    )


def _task(*, task_kind: str = "forward_theorem", task_id: str = "task.fw") -> RevmathTaskV1:
    return RevmathTaskV1(
        task_id=task_id,
        task_kind=task_kind,  # type: ignore[arg-type]
        proposition=_prop(),
        base_theory=RevmathBaseTheoryV1(
            base_theory_id="RCA0",
            allowed_assumption_ids=("ISigma1",),
            interpretation_status="explicit_interpretation",
        ),
        theorem_direction="forward",
        corpus=RevmathCorpusIdentityV1(
            corpus_id="corpus.rm.v1",
            corpus_sha256=_HEX,
            entry_id="entry.fw",
        ),
        budget=RevmathBudgetV1(
            max_wall_seconds=30.0,
            max_tactic_steps=100,
            stopping_rule_ids=("wall_or_steps",),
        ),
        verifier_judge=RevmathVerifierJudgeIdentityV1(
            verifier_id="lean.kernel",
            judge_id="solver_judgment.v1",
        ),
        campaign=RevmathCampaignBindingV1(
            campaign_id="camp.rm",
            experiment_id="exp.rm",
            campaign_manifest_sha256=_HEX,
            locked_arm_ids=("arm.a",),
        ),
    )


def _unknown_result(task: RevmathTaskV1) -> RevmathResultV1:
    judgment = SolverJudgmentV1(
        outcome=JudgmentOutcome.UNKNOWN,
        payload=UnknownReason.INCOMPLETE_COVERAGE,
        checked=False,
    )
    return RevmathResultV1(
        result_id=f"result.{task.task_id}",
        task_id=task.task_id,
        task_identity_digest=task.identity_digest(),
        judgment=judgment.to_dict(),
        proof=None,
        four_axis=_four_axis(),
    )


def _run(
    task: RevmathTaskV1,
    *,
    note: str = "",
    launch_error: str | None = None,
    timed_out: bool = False,
) -> RevmathRunRecord:
    result = _unknown_result(task)
    capture = {
        "outcome": None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "stdout_sha256": content_sha(""),
        "stderr_sha256": content_sha(""),
        "duration_seconds": 0.0,
        "timed_out": timed_out,
        "launch_error": launch_error,
        "note": note,
    }
    bundle = build_revmath_replay_bundle(
        task=task,
        result=result,
        capture=capture,
        identity_snapshot={
            "task_id": task.task_id,
            "task_identity_digest": task.identity_digest(),
        },
    )
    return RevmathRunRecord(
        task=task,
        result=result,
        replay_bundle=bundle,
        capture=capture,
        identity_snapshot={"task_id": task.task_id},
        decode_stats_record={"schema": "decode_stats_record/v1"},
    )


@pytest.mark.parametrize(
    ("failure_class", "hints", "task_kind"),
    [
        (
            RevmathFailureClass.MISSING_NECESSARY_ASSUMPTION_CANDIDATE,
            {"necessary_assumption_ids": ("ISigma1",)},
            "assumption_ablation",
        ),
        (
            RevmathFailureClass.FAILED_REVERSAL_DIRECTION,
            {"failed_direction": "reverse"},
            "reversal",
        ),
        (
            RevmathFailureClass.NONCONSTRUCTIVE_REMAINDER,
            {"remainder_note": "choice on infinite branch"},
            "constructivization",
        ),
        (
            RevmathFailureClass.QUANTITATIVE_BOUND_UNAVAILABLE,
            {"missing_assumptions": ("finite_parameter_dimension",)},
            "quantitative_bound_extraction",
        ),
        (
            RevmathFailureClass.CHECKED_COUNTEREXAMPLE,
            {"independently_checked": True},
            "computable_finite_counterexample",
        ),
        (
            RevmathFailureClass.CHECKER_TOOL_ENVIRONMENT_ERROR,
            {},
            "forward_theorem",
        ),
        (
            RevmathFailureClass.BLOCKED_SEMANTIC_MUTATION,
            {"blocked_semantic_mutation": True},
            "forward_theorem",
        ),
    ],
)
def test_route_covers_each_failure_class(
    failure_class: RevmathFailureClass,
    hints: dict[str, Any],
    task_kind: str,
) -> None:
    task = _task(task_kind=task_kind, task_id=f"task.{failure_class.value[:12]}")
    note = "missing_lean_tool" if failure_class is RevmathFailureClass.CHECKER_TOOL_ENVIRONMENT_ERROR else ""
    run = _run(task, note=note)
    evidence = route_revmath_failure(
        run,
        localization_hints=hints or None,
        blocked_semantic_mutation=failure_class
        is RevmathFailureClass.BLOCKED_SEMANTIC_MUTATION,
    )
    assert evidence.failure_class == failure_class.value
    assert evidence.schema_version == EVIDENCE_SCHEMA
    assert evidence.failure_taxonomy_version == REVMATH_FAILURE_TAXONOMY_VERSION
    assert evidence.semantic_failure_family == semantic_family_for_revmath_class(
        failure_class
    ).value
    assert evidence.judgment_outcome == JudgmentOutcome.UNKNOWN.value
    assert evidence.witness.schema_version == WITNESS_SCHEMA_VERSION
    assert witness_integrity_ok(evidence.witness)
    assert evidence.lineage["task_identity_digest"] == task.identity_digest()
    assert evidence.lineage["proposition_id"] == task.proposition.proposition_id
    round_trip = RevmathFailureEvidenceV1.from_dict(evidence.to_dict())
    assert round_trip.content_sha256 == evidence.content_sha256
    assert round_trip.witness.witness_digest == evidence.witness.witness_digest


def test_unlocalized_stays_explicit_unknown() -> None:
    run = _run(_task())
    evidence = route_revmath_failure(run)
    assert evidence.failure_class == RevmathFailureClass.UNLOCALIZED.value
    assert evidence.completeness_class == "UNKNOWN"
    assert any("unlocalized" in u for u in evidence.unresolved)
    actions = bounded_repair_actions_for(RevmathFailureClass.UNLOCALIZED)
    assert actions  # advisory only
    assert all(a.knob for a in actions)


def test_blocked_and_assumption_classes_have_empty_or_safe_actions() -> None:
    assert bounded_repair_actions_for(
        RevmathFailureClass.BLOCKED_SEMANTIC_MUTATION
    ) == ()
    assert bounded_repair_actions_for(
        RevmathFailureClass.MISSING_NECESSARY_ASSUMPTION_CANDIDATE
    ) == ()
    assert bounded_repair_actions_for(
        RevmathFailureClass.QUANTITATIVE_BOUND_UNAVAILABLE
    ) == ()
    reversal = bounded_repair_actions_for(RevmathFailureClass.FAILED_REVERSAL_DIRECTION)
    assert {a.knob for a in reversal} <= {
        "proof_search_strategy",
        "tactic_sequence",
        "retrieval_choices",
    }


def test_negative_fabricated_completeness_rejected() -> None:
    with pytest.raises(RevmathSchemaError, match="fabricated completeness"):
        build_revmath_verifier_witness(
            failure_class=RevmathFailureClass.UNLOCALIZED,
            completeness_class="compiler-hard",  # type: ignore[arg-type]
            detail="nope",
            source_trace_fingerprint=_HEX,
            judgment_outcome="unknown",
            certificate_id=None,
            unresolved=(),
        )


def test_negative_exact_without_certificate_rejected() -> None:
    with pytest.raises(RevmathSchemaError, match="certificate_id"):
        build_revmath_verifier_witness(
            failure_class=RevmathFailureClass.FAILED_REVERSAL_DIRECTION,
            completeness_class="EXACT",
            detail="reverse failed",
            source_trace_fingerprint=_HEX,
            judgment_outcome="unknown",
            certificate_id=None,
            unresolved=(),
        )


def test_negative_unlocalized_cannot_claim_exact() -> None:
    with pytest.raises(RevmathSchemaError, match="UNKNOWN"):
        build_revmath_verifier_witness(
            failure_class=RevmathFailureClass.UNLOCALIZED,
            completeness_class="EXACT",
            detail="pretend",
            source_trace_fingerprint=_HEX,
            judgment_outcome="unknown",
            certificate_id="cert",
            unresolved=(),
        )


def test_negative_from_dict_rejects_tampered_digest() -> None:
    evidence = route_revmath_failure(
        _run(_task(task_kind="reversal")),
        localization_hints={"failed_direction": "reverse"},
    )
    payload = evidence.to_dict()
    payload["content_sha256"] = "00" * 32
    with pytest.raises(RevmathSchemaError, match="digest mismatch"):
        RevmathFailureEvidenceV1.from_dict(payload)


def test_negative_from_dict_rejects_fabricated_localization_completeness() -> None:
    evidence = route_revmath_failure(
        _run(_task(task_kind="reversal")),
        localization_hints={"failed_direction": "reverse"},
    )
    payload = evidence.to_dict()
    payload["witness"]["localizations"][0]["completeness_class"] = "compiler-hard"
    # Recompute would fail either at localization parse or digest — both closed.
    with pytest.raises(ValueError, match="completeness_class"):
        RevmathFailureEvidenceV1.from_dict(payload)


def test_diagnostic_cannot_upgrade_unknown_to_witnessed() -> None:
    run = _run(_task())
    evidence = route_revmath_failure(run)
    forged = RevmathFailureEvidenceV1(
        schema_version=evidence.schema_version,
        failure_taxonomy_version=evidence.failure_taxonomy_version,
        failure_class=evidence.failure_class,
        semantic_failure_family=evidence.semantic_failure_family,
        completeness_class=evidence.completeness_class,
        proposition_id=evidence.proposition_id,
        task_id=evidence.task_id,
        task_identity_digest=evidence.task_identity_digest,
        checker_id=evidence.checker_id,
        judgment_outcome=JudgmentOutcome.WITNESSED.value,
        judgment_checked=True,
        lineage=evidence.lineage,
        unresolved=evidence.unresolved,
        witness=evidence.witness,
        repair_action_suggestions=evidence.repair_action_suggestions,
        content_sha256=evidence.content_sha256,
    )
    with pytest.raises(RevmathSchemaError, match="judgment_outcome drifted"):
        assert_diagnostic_cannot_prove_success(forged, run.result)


def test_seal_verifier_witness_parity_with_owner() -> None:
    loc = VerifierLocalizationV1(
        source="revmath",
        gate=None,
        reason_code="revmath.unlocalized",
        ast_path=None,
        source_span=None,
        expected=None,
        observed="unknown",
        completeness_class="UNKNOWN",
        certificate_id=None,
        detail="x",
    )
    witness = seal_verifier_witness(
        source_trace_fingerprint=_HEX,
        taxonomy_version="semantic_failure_taxonomy/v1",
        evaluator_version="test",
        evaluator_hash="00" * 32,
        first_failed_gate=None,
        localizations=(loc,),
        unresolved_localizations=("revmath:unlocalized",),
    )
    assert witness_integrity_ok(witness)
    assert VerifierWitnessV1.from_dict(witness.to_dict()).witness_digest == witness.witness_digest


def test_harn09_before_after_feed_disposition_surface() -> None:
    task = _task(task_id="task.heal.feed")
    initial = _run(task, note="incomplete")
    session = run_self_healing(
        task,
        initial_run=initial,
        initial_knob_state=RevmathRepairKnobStateV1(),
        max_attempts=1,
        runner=lambda t, **k: initial,
    )
    evidence = route_revmath_failure(initial, repair_session=session)
    feedback = feed_repair_session_to_disposition(session, failure=evidence)
    assert feedback["schema"] == FEEDBACK_SCHEMA
    record = feedback["disposition_record"]
    assert record["mechanism_id"] == "revmath.self_healing"
    assert record["disposition"] in {
        "retain_diagnostic",
        "inconclusive",
        "blocked",
    }
    assert record["default_state"] == "off"
    assert record["evidence_class"] == "fixture"
    assert feedback["before_after_refs"] is not None
    assert evidence.lineage["repair_session"]["session_id"] == session.session_id
    # Activation payload carries immutable before/after refs for replay.
    activation = json.loads(record["activation_evidence"])
    assert activation["schema"] == FEEDBACK_SCHEMA
    assert activation["task_identity_digest"] == task.identity_digest()


def test_owner_map_registers_failure_witness_surface() -> None:
    owner_map = json.loads(
        (REPO / "src/slm_training/resources/revmath_owner_map.json").read_text(
            encoding="utf-8"
        )
    )
    surfaces = {s["id"]: s for s in owner_map["surfaces"]}
    assert "revmath_failure_witness" in surfaces
    assert (
        surfaces["revmath_failure_witness"]["owner_module"]
        == "src/slm_training/harnesses/reasoning/revmath/failure_witness.py"
    )
    parity = json.loads(
        (REPO / "src/slm_training/resources/revmath_harness_parity.json").read_text(
            encoding="utf-8"
        )
    )
    obligation_ids = {o["id"] for o in parity["obligations"]}
    assert "evidence.revmath_failure_witness" in obligation_ids
