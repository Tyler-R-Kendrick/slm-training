"""HARN-02 / SLM-527: strict revmath schema fail-closed tests (torch-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.judgment import (
    JudgmentOutcome,
    SolverJudgmentV1,
    UnknownReason,
    WitnessPayloadV1,
)
from slm_training.harness_core.lineage.records import content_sha
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
    RevmathRepairKnobChangeV1,
    RevmathRepairRecordV1,
    RevmathResultV1,
    RevmathSchemaError,
    RevmathTaskV1,
    RevmathVerifierJudgeIdentityV1,
    assert_repair_preserves_identity,
    canonical_hash,
    export_json_schemas,
    parse_revmath_repair,
    parse_revmath_result,
    parse_revmath_task,
)

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
    base = {
        "task_id": "task.forward.1",
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
            entry_id="entry.001",
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
            experiment_id="exp.rm.1",
            campaign_manifest_sha256=_HEX2,
            locked_arm_ids=("control", "candidate"),
        ),
    }
    base.update(overrides)
    return RevmathTaskV1(**base)  # type: ignore[arg-type]


def _four_axis(**status_overrides: str) -> RevmathFourAxisAnalysisV1:
    def axis(name: str, status: str) -> AxisCertificateRefV1:
        kwargs: dict[str, object] = {"axis": name, "status": status}
        if status in ("proved_axis", "refuted_axis"):
            kwargs["certificate_sha256"] = _HEX
            kwargs["theorem_ref"] = f"thm.{name}"
            if name == "resource_bounds":
                kwargs["bound_ast_id"] = "bound.work.v1"
        return AxisCertificateRefV1(**kwargs)  # type: ignore[arg-type]

    return RevmathFourAxisAnalysisV1(
        assumption_strength=axis(
            "assumption_strength", status_overrides.get("assumption_strength", "not_claimed")
        ),
        computability=axis(
            "computability", status_overrides.get("computability", "unknown_axis")
        ),
        resource_bounds=axis(
            "resource_bounds", status_overrides.get("resource_bounds", "not_claimed")
        ),
        implementation_refinement=axis(
            "implementation_refinement",
            status_overrides.get("implementation_refinement", "empirical_remainder"),
        ),
    )


def test_task_kinds_and_round_trip() -> None:
    kinds = (
        "assumption_ablation",
        "forward_theorem",
        "reversal",
        "constructivization",
        "computable_finite_counterexample",
        "quantitative_bound_extraction",
        "computability_classification",
    )
    for kind in kinds:
        task = _task(task_kind=kind, task_id=f"task.{kind}")
        rebuilt = parse_revmath_task(task.model_dump(mode="json"))
        assert rebuilt.task_kind == kind
        assert rebuilt.canonical_hash() == canonical_hash(task)
        assert len(rebuilt.identity_digest()) == 64


def test_underspecified_task_fails_closed() -> None:
    raw = _task().model_dump(mode="json")
    del raw["campaign"]
    with pytest.raises(RevmathSchemaError, match="invalid task"):
        parse_revmath_task(raw)


def test_unknown_schema_version_fails_closed() -> None:
    raw = _task().model_dump(mode="json")
    raw["schema_version"] = "revmath_task/v0"
    with pytest.raises(RevmathSchemaError, match="unsupported task schema_version"):
        parse_revmath_task(raw)


def test_rca0_requires_explicit_interpretation() -> None:
    with pytest.raises(ValueError, match="explicit_reversal or explicit_interpretation"):
        RevmathBaseTheoryV1(
            base_theory_id="WKL0",
            allowed_assumption_ids=(),
            interpretation_status="uninterpreted",
        )


def test_result_requires_solver_judgment_v1() -> None:
    task = _task()
    witnessed = SolverJudgmentV1(
        outcome=JudgmentOutcome.WITNESSED,
        payload=WitnessPayloadV1(witness_digest=_HEX, source="lean"),
        checked=True,
    )
    result = RevmathResultV1(
        result_id="res.1",
        task_id=task.task_id,
        task_identity_digest=task.identity_digest(),
        judgment=witnessed.to_dict(),
        proof=RevmathProofArtifactRefV1(
            proof_sha256=_HEX,
            checker_id="lean.kernel",
            checker_report_sha256=_HEX2,
        ),
        four_axis=_four_axis(assumption_strength="proved_axis"),
    )
    assert result.solver_judgment().outcome is JudgmentOutcome.WITNESSED
    round_trip = parse_revmath_result(result.model_dump(mode="json"))
    assert round_trip.judgment["schema"] == "solver_judgment/v1"


def test_unknown_payload_cannot_become_success() -> None:
    task = _task()
    with pytest.raises(ValueError, match="cannot normalize unknown payloads"):
        RevmathResultV1(
            result_id="res.bad",
            task_id=task.task_id,
            task_identity_digest=task.identity_digest(),
            judgment={"ok": True, "proved": True},
            four_axis=_four_axis(),
        )


def test_timeout_unknown_stays_unknown_without_proof() -> None:
    task = _task()
    unknown = SolverJudgmentV1(
        outcome=JudgmentOutcome.UNKNOWN,
        payload=UnknownReason.TIMEOUT,
        checked=False,
    )
    result = RevmathResultV1(
        result_id="res.timeout",
        task_id=task.task_id,
        task_identity_digest=task.identity_digest(),
        judgment=unknown.to_dict(),
        proof=None,
        four_axis=_four_axis(),
    )
    assert result.solver_judgment().outcome is JudgmentOutcome.UNKNOWN
    assert result.solver_judgment().authorizes_semantic_conclusion() is False


def test_witnessed_without_proof_fails() -> None:
    task = _task()
    witnessed = SolverJudgmentV1(
        outcome=JudgmentOutcome.WITNESSED,
        payload=WitnessPayloadV1(witness_digest=_HEX, source="lean"),
        checked=True,
    )
    with pytest.raises(ValueError, match="require proof artifact"):
        RevmathResultV1(
            result_id="res.np",
            task_id=task.task_id,
            task_identity_digest=task.identity_digest(),
            judgment=witnessed.to_dict(),
            proof=None,
            four_axis=_four_axis(),
        )


def test_repair_preserves_identity_and_rejects_frozen_knobs() -> None:
    task = _task()
    repair = RevmathRepairRecordV1(
        repair_id="repair.1",
        task_identity_digest=task.identity_digest(),
        proposition_id=task.proposition.proposition_id,
        statement_sha256=task.proposition.statement_sha256,
        base_theory_id=task.base_theory.base_theory_id,
        allowed_assumption_ids=task.base_theory.allowed_assumption_ids,
        campaign_manifest_sha256=task.campaign.campaign_manifest_sha256,
        campaign_id=task.campaign.campaign_id,
        experiment_id=task.campaign.experiment_id,
        before_proof_sha256=_HEX,
        after_proof_sha256=_HEX2,
        knob_changes=(
            RevmathRepairKnobChangeV1(
                knob="tactic_sequence",
                before_value_sha256=_HEX,
                after_value_sha256=_HEX3,
            ),
        ),
    )
    assert_repair_preserves_identity(task, repair)
    parse_revmath_repair(repair.model_dump(mode="json"))

    with pytest.raises(ValueError, match="not mutable"):
        RevmathRepairKnobChangeV1(
            knob="proposition",
            before_value_sha256=_HEX,
            after_value_sha256=_HEX2,
        )

    bad = repair.model_copy(update={"proposition_id": "prop.other"})
    with pytest.raises(RevmathSchemaError, match="cannot alter proposition_id"):
        assert_repair_preserves_identity(task, bad)

    bad_campaign = repair.model_copy(update={"campaign_manifest_sha256": _HEX3})
    with pytest.raises(RevmathSchemaError, match="campaign_manifest_sha256"):
        assert_repair_preserves_identity(task, bad_campaign)


def test_mutable_knobs_match_parity_matrix() -> None:
    parity = json.loads(
        (REPO / "src/slm_training/resources/revmath_harness_parity.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(parity["self_healing"]["may_modify"]) == set(REVMATH_MUTABLE_REPAIR_KNOBS)
    overlap = set(parity["self_healing"]["may_modify"]) & set(
        parity["self_healing"]["must_freeze"]
    )
    assert not overlap


def test_four_axis_slot_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="four-axis slot"):
        RevmathFourAxisAnalysisV1(
            assumption_strength=AxisCertificateRefV1(
                axis="computability", status="not_claimed"
            ),
            computability=AxisCertificateRefV1(
                axis="computability", status="not_claimed"
            ),
            resource_bounds=AxisCertificateRefV1(
                axis="resource_bounds", status="not_claimed"
            ),
            implementation_refinement=AxisCertificateRefV1(
                axis="implementation_refinement", status="not_claimed"
            ),
        )


def test_json_schema_export_covers_contracts() -> None:
    schemas = export_json_schemas()
    assert set(schemas) == {
        "revmath_task/v1",
        "revmath_corpus_entry/v1",
        "revmath_result/v1",
        "revmath_repair/v1",
        "revmath_report/v1",
        "revmath_four_axis/v1",
    }
    for schema in schemas.values():
        assert "properties" in schema
        assert schema.get("additionalProperties") is False or "title" in schema
