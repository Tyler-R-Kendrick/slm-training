"""Tests for goal-verifier profiles and terminal-evidence contracts (PGS-B01 / SLM-497)."""

from __future__ import annotations

import pytest

from tests.casefiles import case_values
from pydantic import ValidationError

from slm_training.data.progspec.goal_constraints import (
    CompiledGoalConstraintSetV1,
    GoalConstraintAmbiguityGroup,
    GoalConstraintEvaluationV1,
    GoalConstraintV1,
)
from slm_training.data.progspec.synthesis_problem import PackIdentityV1
from slm_training.dsl.solver.goal_support import (
    EvaluatorIdentityV1,
    GoalEvaluatorResultV1,
    GoalFailureAtomV1,
    GoalGateResultV1,
    GoalTerminalEvidenceV1,
    GoalUnknownAtomV1,
    GoalVerifierProfileV1,
    MetricIdentityV1,
    compute_pack_identity_digest,
    profile_digest_inputs,
    profile_mode_authority_table,
    redact_bounded_string,
    terminal_status_decision_table,
    validate_profile_against_constraint_set,
)


def _pack_identity(**overrides: object) -> PackIdentityV1:
    defaults: dict[str, object] = {
        "pack_id": "openui",
        "contract_version": 2,
        "grammar_sha256": "a" * 64,
        "tokenizer_id": "tok/v1",
        "canonicalizer_id": "canon/v1",
        "artifact_schema": "openui/v1",
        "artifact_sha256": "b" * 64,
    }
    defaults.update(overrides)
    return PackIdentityV1(**defaults)


def _constraint(**overrides: object) -> GoalConstraintV1:
    defaults: dict[str, object] = {
        "constraint_id": "c_slot_button",
        "kind": "slot_present",
        "parameters": {"role_id": "primary_action"},
        "source_kind": "pack_contract",
        "source_id": "pack/openui",
        "source_digest": "abc123",
        "authority_tier": "compiler-hard",
        "completeness": "EXACT",
        "may_prune": True,
    }
    defaults.update(overrides)
    return GoalConstraintV1(**defaults)


def _compiled_set(**overrides: object) -> CompiledGoalConstraintSetV1:
    hard = _constraint()
    advisory = _constraint(
        constraint_id="adv_hint",
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
        source_kind="prompt_requirement",
    )
    evaluation = _constraint(
        constraint_id="eval_gate",
        kind="required_gate_passes",
        parameters={"gate": "G3"},
        authority_tier="evaluation-only",
        completeness="EXACT",
        may_prune=False,
        source_kind="evaluation_fixture",
    )
    defaults: dict[str, object] = {
        "problem_digest": "a" * 64,
        "request_digest": "b" * 64,
        "pack_identity_digest": compute_pack_identity_digest(_pack_identity()),
        "compiler_version": "compiler/v1",
        "constraints": (hard, advisory, evaluation),
        "hard_constraint_ids": (hard.constraint_id,),
        "advisory_constraint_ids": (advisory.constraint_id,),
        "evaluation_constraint_ids": (evaluation.constraint_id,),
    }
    defaults.update(overrides)
    payload = CompiledGoalConstraintSetV1(**defaults).to_dict(include_digest=True)
    return CompiledGoalConstraintSetV1.from_dict(payload)


def _profile(**overrides: object) -> GoalVerifierProfileV1:
    pack = _pack_identity()
    constraint_set = _compiled_set(pack_identity_digest=compute_pack_identity_digest(pack))
    defaults: dict[str, object] = {
        "profile_id": "profile/prod",
        "mode": "production_exact",
        "problem_digest": constraint_set.problem_digest,
        "constraint_set_digest": constraint_set.digest,
        "pack_identity": pack,
        "pack_identity_digest": compute_pack_identity_digest(pack),
        "required_constraint_ids": ("c_slot_button",),
        "required_gates": ("G0", "G3"),
        "required_evaluators": (
            EvaluatorIdentityV1(
                evaluator_id="meaningful_program/v2",
                version="v2",
                implementation_hash="c" * 64,
            ),
        ),
        "metric_identities": (
            MetricIdentityV1(metric_id="meaningful_program", version="v2"),
        ),
        "authority_tier": "compiler-hard",
    }
    defaults.update(overrides)
    payload = GoalVerifierProfileV1(**defaults).to_dict(include_digest=True)
    return GoalVerifierProfileV1.from_dict(payload)


def _terminal_evidence(**overrides: object) -> GoalTerminalEvidenceV1:
    profile = _profile()
    defaults: dict[str, object] = {
        "profile_digest": profile.digest,
        "program_digest": "1" * 64,
        "canonical_program_digest": "2" * 64,
        "program_spec_digest": "3" * 64,
        "semantic_plan_digest": "4" * 64,
        "constraint_evaluations": (
            GoalConstraintEvaluationV1(constraint_id="c_slot_button", status="PASS"),
        ),
        "required_gate_results": (GoalGateResultV1(gate_id="G0", status="ACCEPT"),),
        "required_evaluator_results": (
            GoalEvaluatorResultV1(
                evaluator_id="meaningful_program/v2",
                status="ACCEPT",
            ),
        ),
        "mandatory_failure_atoms": (),
        "mandatory_unknown_atoms": (),
        "structural_status": "ACCEPT",
        "overall_status": "ACCEPT",
    }
    defaults.update(overrides)
    payload = GoalTerminalEvidenceV1(**defaults).to_dict(include_digest=True)
    return GoalTerminalEvidenceV1.from_dict(payload)


def test_round_trip_preserves_profile_and_terminal_evidence() -> None:
    profile = _profile()
    restored_profile = GoalVerifierProfileV1.from_dict(profile.to_dict())
    assert restored_profile == profile

    evidence = _terminal_evidence(profile_digest=profile.digest)
    restored_evidence = GoalTerminalEvidenceV1.from_dict(evidence.to_dict())
    assert restored_evidence == evidence


def test_from_dict_rejects_unknown_profile_version() -> None:
    payload = _profile().to_dict()
    payload["schema_version"] = "goal_verifier_profile/v2"
    with pytest.raises(ValueError, match="unsupported GoalVerifierProfileV1 version"):
        GoalVerifierProfileV1.from_dict(payload)


def test_from_dict_rejects_unknown_terminal_evidence_version() -> None:
    payload = _terminal_evidence().to_dict()
    payload["schema_version"] = "goal_terminal_evidence/v2"
    with pytest.raises(ValueError, match="unsupported GoalTerminalEvidenceV1 version"):
        GoalTerminalEvidenceV1.from_dict(payload)


def test_extra_field_rejected_on_profile() -> None:
    with pytest.raises(ValidationError):
        _profile(extra_field="nope")


def test_extra_field_rejected_on_terminal_evidence() -> None:
    with pytest.raises(ValidationError):
        _terminal_evidence(extra_field="nope")


@pytest.mark.parametrize(
    ("mode", "authority_tier"),
    case_values(__file__, "test_mode_authority_mismatch_rejected"),
)
def test_mode_authority_mismatch_rejected(mode: str, authority_tier: str) -> None:
    with pytest.raises(ValidationError, match="authority_tier .* incompatible with mode"):
        _profile(mode=mode, authority_tier=authority_tier)


def test_stale_problem_digest_rejected_against_constraint_set() -> None:
    profile = _profile(problem_digest="f" * 64)
    constraint_set = _compiled_set()
    with pytest.raises(ValueError, match="problem_digest is stale"):
        validate_profile_against_constraint_set(profile, constraint_set)


def test_stale_constraint_set_digest_rejected() -> None:
    profile = _profile(constraint_set_digest="f" * 64)
    constraint_set = _compiled_set()
    with pytest.raises(ValueError, match="constraint_set_digest is stale"):
        validate_profile_against_constraint_set(profile, constraint_set)


def test_stale_pack_identity_digest_rejected() -> None:
    pack_a = _pack_identity()
    pack_b = _pack_identity(pack_id="other-pack")
    constraint_set = _compiled_set(pack_identity_digest=compute_pack_identity_digest(pack_a))
    profile = _profile(
        pack_identity=pack_b,
        pack_identity_digest=compute_pack_identity_digest(pack_b),
        problem_digest=constraint_set.problem_digest,
        constraint_set_digest=constraint_set.digest,
    )
    with pytest.raises(ValueError, match="pack_identity_digest is stale"):
        validate_profile_against_constraint_set(profile, constraint_set)


def test_unknown_required_constraint_id_rejected() -> None:
    profile = _profile(required_constraint_ids=("missing",))
    constraint_set = _compiled_set(
        problem_digest=profile.problem_digest,
        pack_identity_digest=profile.pack_identity_digest,
    )
    with pytest.raises(ValueError, match="unknown constraints"):
        validate_profile_against_constraint_set(profile, constraint_set)


def test_unresolved_ambiguity_rejected_in_production_exact() -> None:
    hard = _constraint()
    second = _constraint(constraint_id="c_alt", parameters={"role_id": "alt"})
    constraint_set = _compiled_set(
        constraints=(hard, second),
        hard_constraint_ids=(hard.constraint_id, second.constraint_id),
        advisory_constraint_ids=(),
        evaluation_constraint_ids=(),
        ambiguity_groups=(
            GoalConstraintAmbiguityGroup(
                group_id="g1",
                member_constraint_ids=(hard.constraint_id, second.constraint_id),
            ),
        ),
    )
    profile = _profile(
        required_constraint_ids=(hard.constraint_id,),
        problem_digest=constraint_set.problem_digest,
        constraint_set_digest=constraint_set.digest,
        pack_identity_digest=constraint_set.pack_identity_digest,
    )
    with pytest.raises(ValueError, match="unresolved hard ambiguity"):
        validate_profile_against_constraint_set(profile, constraint_set)


def test_mandatory_failure_atom_rejects_heuristic_completeness() -> None:
    with pytest.raises(ValidationError, match="mandatory_failure_atoms cannot claim EXACT"):
        _terminal_evidence(
            mandatory_failure_atoms=(
                GoalFailureAtomV1(
                    atom_id="a1",
                    reason_code="binding_unresolved",
                    source_kind="gate",
                    completeness_class="HEURISTIC",
                    source_identity="G3",
                ),
            )
        )


def test_mandatory_failure_atom_rejects_heuristic_gate_source() -> None:
    with pytest.raises(ValidationError, match="mandatory_failure_atoms cannot claim EXACT"):
        _terminal_evidence(
            mandatory_failure_atoms=(
                GoalFailureAtomV1(
                    atom_id="a1",
                    reason_code="judge_disagreement",
                    source_kind="gate",
                    completeness_class="EXACT",
                    source_identity="G11",
                ),
            )
        )


def test_mandatory_unknown_atoms_round_trip() -> None:
    evidence = _terminal_evidence(
        mandatory_unknown_atoms=(
            GoalUnknownAtomV1(
                atom_id="u1",
                reason_code="evaluator_unavailable",
                source_kind="evaluator",
                source_identity="meaningful_program/v2",
            ),
        ),
        overall_status="UNAVAILABLE",
        structural_status="UNAVAILABLE",
    )
    restored = GoalTerminalEvidenceV1.from_dict(evidence.to_dict())
    assert restored.mandatory_unknown_atoms == evidence.mandatory_unknown_atoms


def test_secret_shaped_content_redacted() -> None:
    secret = "token=hf_" + ("x" * 20)
    redacted = redact_bounded_string(secret)
    assert "[REDACTED]" in redacted
    assert "hf_" not in redacted


def test_long_literal_redacted_in_terminal_evidence_to_dict() -> None:
    long_reason = "x" * 400
    evidence = _terminal_evidence(
        required_gate_results=(
            GoalGateResultV1(gate_id="G0", status="REJECT", reason_code=long_reason),
        ),
        overall_status="REJECT",
        structural_status="REJECT",
    )
    payload = evidence.to_dict()
    reason = payload["required_gate_results"][0]["reason_code"]
    assert len(reason) <= 280
    assert "truncated:" in reason


def test_profile_digest_tampering_rejected() -> None:
    payload = _profile().to_dict()
    payload["digest"] = "f" * 64
    with pytest.raises(ValidationError, match="digest does not match"):
        GoalVerifierProfileV1.from_dict(payload)


def test_terminal_evidence_digest_tampering_rejected() -> None:
    payload = _terminal_evidence().to_dict()
    payload["evidence_digest"] = "f" * 64
    with pytest.raises(ValidationError, match="evidence_digest does not match"):
        GoalTerminalEvidenceV1.from_dict(payload)


def test_profile_digest_order_invariant() -> None:
    profile_a = _profile(
        required_constraint_ids=("b", "a"),
        required_gates=("G3", "G0"),
    )
    profile_b = _profile(
        required_constraint_ids=("a", "b"),
        required_gates=("G0", "G3"),
    )
    assert profile_a.compute_digest() == profile_b.compute_digest()


def test_terminal_evidence_digest_order_invariant() -> None:
    gate_a = GoalGateResultV1(gate_id="G0", status="ACCEPT")
    gate_b = GoalGateResultV1(gate_id="G3", status="ACCEPT")
    evidence_a = _terminal_evidence(required_gate_results=(gate_b, gate_a))
    evidence_b = _terminal_evidence(required_gate_results=(gate_a, gate_b))
    assert evidence_a.compute_digest() == evidence_b.compute_digest()


def test_reference_tables_cover_closed_modes_and_statuses() -> None:
    mode_table = profile_mode_authority_table()
    assert set(mode_table) == {
        "production_exact",
        "evaluation_oracle",
        "advisory_diagnostic",
    }
    status_table = terminal_status_decision_table()
    assert set(status_table) == {"ACCEPT", "REJECT", "UNAVAILABLE"}
    assert len(profile_digest_inputs()) > 0


def test_forbidden_terminal_field_rejected_on_from_dict() -> None:
    payload = _terminal_evidence().to_dict()
    payload["raw_prompt"] = "secret user text"
    with pytest.raises(ValueError, match="forbidden terminal-evidence fields"):
        GoalTerminalEvidenceV1.from_dict(payload)


def test_production_exact_rejects_heuristic_gate_in_profile() -> None:
    with pytest.raises(ValidationError, match="heuristic gate identities"):
        _profile(required_gates=("G11",))


def test_evaluation_oracle_profile_validates_evaluation_partition() -> None:
    constraint_set = _compiled_set()
    profile = _profile(
        mode="evaluation_oracle",
        authority_tier="evaluation-only",
        required_constraint_ids=("eval_gate",),
        required_gates=(),
    )
    validate_profile_against_constraint_set(profile, constraint_set)


def test_advisory_profile_validates_advisory_partition() -> None:
    constraint_set = _compiled_set()
    profile = _profile(
        mode="advisory_diagnostic",
        authority_tier="advisory-learned",
        required_constraint_ids=("adv_hint",),
        required_gates=(),
    )
    validate_profile_against_constraint_set(profile, constraint_set)
