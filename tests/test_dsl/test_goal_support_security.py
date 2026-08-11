"""Property-style attacks on goal profiles and terminal evidence."""

from __future__ import annotations

import json
import random

import pytest
from pydantic import ValidationError

from slm_training.data.progspec.goal_constraints import (
    CompiledGoalConstraintSetV1,
    GoalConstraintEvaluationV1,
    GoalConstraintV1,
)
from slm_training.data.progspec.synthesis_problem import PackIdentityV1
from slm_training.dsl.solver.goal_support import (
    EvaluatorIdentityV1,
    GoalFailureAtomV1,
    GoalGateResultV1,
    GoalTerminalEvidenceV1,
    GoalUnknownAtomV1,
    GoalVerifierProfileV1,
    compute_pack_identity_digest,
    validate_profile_against_constraint_set,
)


def _pack(**updates: object) -> PackIdentityV1:
    fields: dict[str, object] = {
        "pack_id": "openui",
        "contract_version": 2,
        "grammar_sha256": "a" * 64,
        "tokenizer_id": "tok/v1",
        "canonicalizer_id": "canon/v1",
        "artifact_schema": "openui/v1",
        "artifact_sha256": "b" * 64,
    }
    fields.update(updates)
    return PackIdentityV1(**fields)


def _constraint(**updates: object) -> GoalConstraintV1:
    fields: dict[str, object] = {
        "constraint_id": "hard.slot",
        "kind": "slot_present",
        "parameters": {"slot": ":slot_0"},
        "source_kind": "generation_request",
        "source_id": "request.slot_contract",
        "source_digest": "c" * 64,
        "authority_tier": "compiler-hard",
        "completeness": "EXACT",
        "may_prune": True,
    }
    fields.update(updates)
    return GoalConstraintV1(**fields)


def _constraint_set(constraint: GoalConstraintV1 | None = None) -> CompiledGoalConstraintSetV1:
    item = constraint or _constraint()
    hard_ids = (item.constraint_id,) if item.authority_tier in {"compiler-hard", "verifier-hard"} else ()
    advisory_ids = (
        (item.constraint_id,)
        if item.authority_tier in {"advisory-learned", "oracle-diagnostic"}
        else ()
    )
    evaluation_ids = (item.constraint_id,) if item.authority_tier == "evaluation-only" else ()
    provisional = CompiledGoalConstraintSetV1(
        problem_digest="d" * 64,
        request_digest="e" * 64,
        pack_identity_digest=compute_pack_identity_digest(_pack()),
        compiler_version="compiler/v1",
        constraints=(item,),
        hard_constraint_ids=hard_ids,
        advisory_constraint_ids=advisory_ids,
        evaluation_constraint_ids=evaluation_ids,
    )
    return CompiledGoalConstraintSetV1.from_dict(provisional.to_dict(include_digest=True))


def _profile(
    constraint_set: CompiledGoalConstraintSetV1 | None = None,
    **updates: object,
) -> GoalVerifierProfileV1:
    constraints = constraint_set or _constraint_set()
    fields: dict[str, object] = {
        "profile_id": "production/security",
        "mode": "production_exact",
        "problem_digest": constraints.problem_digest,
        "constraint_set_digest": constraints.digest,
        "pack_identity": _pack(),
        "pack_identity_digest": constraints.pack_identity_digest,
        "required_constraint_ids": constraints.hard_constraint_ids,
        "authority_tier": "compiler-hard",
    }
    fields.update(updates)
    provisional = GoalVerifierProfileV1(**fields)
    return GoalVerifierProfileV1.from_dict(provisional.to_dict(include_digest=True))


def _evaluation(status: str = "PASS") -> GoalConstraintEvaluationV1:
    return GoalConstraintEvaluationV1(
        constraint_id="hard.slot",
        status=status,
        authority_tier="compiler-hard",
        completeness_achieved="EXACT",
        may_prune=True,
        reason_code="checked",
        evaluator_digest="f" * 64,
    )


def _terminal(**updates: object) -> GoalTerminalEvidenceV1:
    fields: dict[str, object] = {
        "profile_digest": _profile().digest,
        "program_digest": "1" * 64,
        "canonical_program_digest": "2" * 64,
        "program_spec_digest": "3" * 64,
        "semantic_plan_digest": "4" * 64,
        "constraint_evaluations": (_evaluation(),),
        "required_gate_results": (GoalGateResultV1(gate_id="G0", status="ACCEPT"),),
        "structural_status": "ACCEPT",
        "overall_status": "ACCEPT",
    }
    fields.update(updates)
    provisional = GoalTerminalEvidenceV1(**fields)
    return GoalTerminalEvidenceV1.from_dict(provisional.to_dict(include_digest=True))


def test_profile_loader_rejects_authority_and_mode_string_injection() -> None:
    payload = _profile().to_dict()
    attacks = [
        ("mode", "production_exact\x00"),
        ("mode", "production_exact,evaluation_oracle"),
        ("authority_tier", "compiler-hard "),
        ("authority_tier", "evaluation-only/compiler-hard"),
    ]
    random.Random(497).shuffle(attacks)
    for field, attack in attacks:
        forged = {**payload, field: attack, "digest": ""}
        with pytest.raises(ValidationError):
            GoalVerifierProfileV1.from_dict(forged)


def test_model_copy_cannot_launder_advisory_profile_to_production() -> None:
    constraints = _constraint_set(
        _constraint(
            source_kind="prompt_requirement",
            authority_tier="advisory-learned",
            completeness="HEURISTIC",
            may_prune=False,
        )
    )
    advisory = _profile(
        constraints,
        mode="advisory_diagnostic",
        authority_tier="advisory-learned",
        required_constraint_ids=constraints.advisory_constraint_ids,
    )
    with pytest.raises((ValidationError, ValueError)):
        advisory.model_copy(
            update={"mode": "production_exact", "authority_tier": "compiler-hard"}
        )


def test_profile_validation_recomputes_mutated_constraint_set_digest() -> None:
    constraints = _constraint_set()
    profile = _profile(constraints)
    parameters = constraints.constraints[0].parameters
    try:
        parameters["slot"] = ":slot_attacker"
    except TypeError:
        return
    with pytest.raises(ValueError, match="digest|stale|mutat"):
        validate_profile_against_constraint_set(profile, constraints)


def test_production_profile_refuses_hard_constraint_without_prune_authority() -> None:
    constraints = _constraint_set(_constraint(may_prune=False))
    profile = _profile(constraints)
    with pytest.raises(ValueError, match="prun|hard-eligible"):
        validate_profile_against_constraint_set(profile, constraints)


def test_profile_digest_changes_for_every_pinned_identity() -> None:
    baseline = _profile()
    variants = (
        _profile(implementation_version="goal_support/v2"),
        _profile(required_gates=("G0",)),
        _profile(
            required_evaluators=(
                EvaluatorIdentityV1(
                    evaluator_id="exact/evaluator",
                    version="v1",
                    implementation_hash="7" * 64,
                ),
            )
        ),
        _profile(pack_identity=_pack(tokenizer_id="tok/v2"), pack_identity_digest=compute_pack_identity_digest(_pack(tokenizer_id="tok/v2"))),
    )
    assert all(item.digest != baseline.digest for item in variants)


def test_terminal_loader_rejects_unknown_version_and_nested_extra_field() -> None:
    payload = _terminal().to_dict()
    with pytest.raises(ValueError, match="unsupported GoalTerminalEvidenceV1 version"):
        GoalTerminalEvidenceV1.from_dict(
            {**payload, "schema_version": "goal_terminal_evidence/v99"}
        )
    forged = json.loads(json.dumps(payload))
    forged["required_gate_results"][0]["raw_prompt"] = "secret"
    forged["evidence_digest"] = ""
    with pytest.raises(ValidationError):
        GoalTerminalEvidenceV1.from_dict(forged)


def test_terminal_accept_cannot_carry_mandatory_failure() -> None:
    with pytest.raises((ValidationError, ValueError), match="status|ACCEPT|failure"):
        _terminal(
            mandatory_failure_atoms=(
                GoalFailureAtomV1(
                    atom_id="constraint:hard.slot",
                    reason_code="definite_failure",
                    source_kind="constraint",
                    completeness_class="EXACT",
                    source_identity="hard.slot",
                ),
            ),
            constraint_evaluations=(_evaluation("FAIL"),),
        )


def test_terminal_accept_cannot_carry_mandatory_unknown() -> None:
    with pytest.raises((ValidationError, ValueError), match="status|ACCEPT|unknown"):
        _terminal(
            mandatory_unknown_atoms=(
                GoalUnknownAtomV1(
                    atom_id="gate:G5",
                    reason_code="required_gate_unavailable",
                    source_kind="gate",
                    source_identity="G5",
                ),
            )
        )


@pytest.mark.parametrize("status", ("SKIPPED", "NOT_APPLICABLE"))
def test_terminal_accept_cannot_carry_unresolved_constraint(status: str) -> None:
    with pytest.raises((ValidationError, ValueError), match="status|ACCEPT|UNAVAILABLE"):
        _terminal(constraint_evaluations=(_evaluation(status),))


def test_terminal_accept_cannot_hide_structural_or_gate_rejection() -> None:
    with pytest.raises((ValidationError, ValueError), match="status|ACCEPT|structural"):
        _terminal(structural_status="REJECT")
    with pytest.raises((ValidationError, ValueError), match="status|ACCEPT|gate"):
        _terminal(
            required_gate_results=(
                GoalGateResultV1(gate_id="G0", status="REJECT", reason_code="failed"),
            )
        )


def test_terminal_serialization_removes_secret_shaped_values_everywhere() -> None:
    secret = "token=hf_" + "x" * 24
    evidence = _terminal(
        required_gate_results=(
            GoalGateResultV1(gate_id="G0", status="REJECT", reason_code=secret),
        ),
        mandatory_failure_atoms=(
            GoalFailureAtomV1(
                atom_id="gate:G0",
                reason_code=secret,
                source_kind="gate",
                completeness_class="EXACT",
                source_identity=secret,
            ),
        ),
        structural_status="REJECT",
        overall_status="REJECT",
    )
    wire = json.dumps(evidence.to_dict(), sort_keys=True)
    assert "hf_" not in wire
    assert secret not in wire
    assert "[REDACTED]" in wire


def test_terminal_serialization_does_not_retain_long_opaque_prefix() -> None:
    opaque = "customer-private-literal-" + "x" * 400
    evidence = _terminal(
        required_gate_results=(
            GoalGateResultV1(gate_id="G0", status="REJECT", reason_code=opaque),
        ),
        structural_status="REJECT",
        overall_status="REJECT",
    )
    wire = json.dumps(evidence.to_dict(), sort_keys=True)
    assert "customer-private-literal" not in wire
    assert opaque not in wire


def test_terminal_digest_is_invariant_to_evidence_order() -> None:
    forward = _terminal(
        required_gate_results=(
            GoalGateResultV1(gate_id="G3", status="ACCEPT"),
            GoalGateResultV1(gate_id="G0", status="ACCEPT"),
        )
    )
    reverse = _terminal(
        required_gate_results=tuple(reversed(forward.required_gate_results))
    )
    assert forward.evidence_digest == reverse.evidence_digest
