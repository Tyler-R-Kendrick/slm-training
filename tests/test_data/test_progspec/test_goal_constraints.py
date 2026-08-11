"""Tests for compiled goal-constraint contract (PGS-A01 / SLM-494)."""

from __future__ import annotations

import pytest

from tests.casefiles import case_values
from pydantic import ValidationError

from slm_training.data.progspec.goal_constraints import (
    COMPILED_GOAL_CONSTRAINT_SET_SCHEMA_VERSION,
    GOAL_CONSTRAINT_EVALUATION_SCHEMA_VERSION,
    GOAL_CONSTRAINT_SCHEMA_VERSION,
    CompiledGoalConstraintSetV1,
    GoalConstraintAmbiguityGroup,
    GoalConstraintEvaluationV1,
    GoalConstraintV1,
    authority_matrix,
    combine_authority,
    combine_completeness,
    combine_may_prune,
    merge_goal_constraints,
)


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
    constraint = _constraint()
    defaults: dict[str, object] = {
        "problem_digest": "problem",
        "request_digest": "request",
        "pack_identity_digest": "pack",
        "compiler_version": "compiler/v1",
        "constraints": (constraint,),
        "hard_constraint_ids": (constraint.constraint_id,),
        "advisory_constraint_ids": (),
        "evaluation_constraint_ids": (),
    }
    defaults.update(overrides)
    payload = CompiledGoalConstraintSetV1(**defaults).to_dict(include_digest=True)
    return CompiledGoalConstraintSetV1.from_dict(payload)


def test_round_trip_preserves_goal_constraint_fields() -> None:
    constraint = _constraint(
        kind="required_gate_passes",
        parameters={"gate": "G3"},
        source_kind="verification_requirement",
        authority_tier="verifier-hard",
    )
    restored = GoalConstraintV1.from_dict(constraint.to_dict())
    assert restored == constraint


def test_from_dict_rejects_unknown_goal_constraint_version() -> None:
    payload = _constraint().to_dict()
    payload["schema_version"] = "goal_constraint/v2"
    with pytest.raises(ValueError, match="unsupported GoalConstraintV1 version"):
        GoalConstraintV1.from_dict(payload)


def test_extra_field_rejected_on_goal_constraint() -> None:
    with pytest.raises(ValidationError):
        _constraint(extra_field="nope")


def test_nonfinite_parameter_rejected() -> None:
    with pytest.raises(ValidationError, match="JSON-compatible|non-finite"):
        _constraint(parameters={"threshold": float("nan")})


def test_may_prune_without_exact_hard_authority_rejected() -> None:
    with pytest.raises(ValidationError, match="may_prune=True requires"):
        _constraint(
            authority_tier="advisory-learned",
            completeness="EXACT",
            source_kind="pack_contract",
            may_prune=True,
        )


@pytest.mark.parametrize(
    ("source_kind", "authority_tier", "completeness"),
    case_values(__file__, "test_may_prune_fail_closed_unless_independently_exact_and_exact"),
)
def test_may_prune_fail_closed_unless_independently_exact_and_exact(
    source_kind: str,
    authority_tier: str,
    completeness: str,
) -> None:
    with pytest.raises(ValidationError, match="may_prune=True requires"):
        _constraint(
            source_kind=source_kind,
            authority_tier=authority_tier,
            completeness=completeness,
            may_prune=True,
        )


def test_prompt_requirement_cannot_gain_prune_even_with_compiler_hard() -> None:
    with pytest.raises(ValidationError, match="may_prune=True requires"):
        _constraint(
            source_kind="prompt_requirement",
            authority_tier="compiler-hard",
            completeness="EXACT",
            may_prune=True,
        )


def test_duplicate_constraint_id_rejected_in_compiled_set() -> None:
    duplicate = _constraint()
    with pytest.raises(ValidationError, match="constraint_id values must be unique"):
        CompiledGoalConstraintSetV1(
            problem_digest="p",
            request_digest="r",
            pack_identity_digest="k",
            compiler_version="v1",
            constraints=(duplicate, duplicate),
            hard_constraint_ids=(duplicate.constraint_id,),
        )


def test_partition_mismatch_rejected() -> None:
    advisory = _constraint(
        constraint_id="adv",
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
        source_kind="prompt_requirement",
    )
    with pytest.raises(ValidationError, match="hard_constraint_ids contains"):
        CompiledGoalConstraintSetV1(
            problem_digest="p",
            request_digest="r",
            pack_identity_digest="k",
            compiler_version="v1",
            constraints=(advisory,),
            hard_constraint_ids=(advisory.constraint_id,),
        )


def test_missing_partition_member_rejected() -> None:
    constraint = _constraint()
    with pytest.raises(ValidationError, match="constraint partitions must cover"):
        CompiledGoalConstraintSetV1(
            problem_digest="p",
            request_digest="r",
            pack_identity_digest="k",
            compiler_version="v1",
            constraints=(constraint,),
            hard_constraint_ids=(),
            advisory_constraint_ids=(),
            evaluation_constraint_ids=(),
        )


def test_malformed_digest_rejected() -> None:
    payload = _compiled_set().to_dict()
    payload["digest"] = "0" * 63
    with pytest.raises(ValidationError):
        CompiledGoalConstraintSetV1.from_dict(payload)


def test_digest_tampering_rejected() -> None:
    payload = _compiled_set().to_dict()
    payload["digest"] = "f" * 64
    with pytest.raises(ValidationError, match="digest does not match"):
        CompiledGoalConstraintSetV1.from_dict(payload)


def test_digest_order_invariant() -> None:
    first = _constraint(constraint_id="a", parameters={"z": 1})
    second = _constraint(
        constraint_id="b",
        kind="component_count_at_least",
        parameters={"count": 2},
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
        source_kind="prompt_requirement",
    )
    forward = CompiledGoalConstraintSetV1(
        problem_digest="p",
        request_digest="r",
        pack_identity_digest="k",
        compiler_version="v1",
        constraints=(first, second),
        hard_constraint_ids=(first.constraint_id,),
        advisory_constraint_ids=(second.constraint_id,),
    ).to_dict(include_digest=True)
    reverse = CompiledGoalConstraintSetV1(
        problem_digest="p",
        request_digest="r",
        pack_identity_digest="k",
        compiler_version="v1",
        constraints=(second, first),
        hard_constraint_ids=(first.constraint_id,),
        advisory_constraint_ids=(second.constraint_id,),
    ).to_dict(include_digest=True)
    assert forward["digest"] == reverse["digest"]
    CompiledGoalConstraintSetV1.from_dict(forward)
    CompiledGoalConstraintSetV1.from_dict(reverse)


def test_ambiguity_group_unresolved_member_rejected() -> None:
    constraint = _constraint()
    with pytest.raises(ValidationError, match="unknown constraint ids"):
        CompiledGoalConstraintSetV1(
            problem_digest="p",
            request_digest="r",
            pack_identity_digest="k",
            compiler_version="v1",
            constraints=(constraint,),
            hard_constraint_ids=(constraint.constraint_id,),
            ambiguity_groups=(
                GoalConstraintAmbiguityGroup(
                    group_id="g0",
                    member_constraint_ids=(constraint.constraint_id, "missing"),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    case_values(__file__, "test_combine_authority_never_escalates"),
)
def test_combine_authority_never_escalates(left: str, right: str, expected: str) -> None:
    assert combine_authority(left, right) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    case_values(__file__, "test_combine_completeness_never_escalates"),
)
def test_combine_completeness_never_escalates(left: str, right: str, expected: str) -> None:
    assert combine_completeness(left, right) == expected


def test_combine_may_prune_never_escalates() -> None:
    assert combine_may_prune(True, False) is False
    assert combine_may_prune(True, True) is True


def test_merge_identical_constraints_takes_weaker_authority_and_completeness() -> None:
    strong = _constraint(
        authority_tier="compiler-hard",
        completeness="EXACT",
        may_prune=True,
    )
    weak = _constraint(
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
    )
    merged = merge_goal_constraints(strong, weak)
    assert merged.authority_tier == "advisory-learned"
    assert merged.completeness == "HEURISTIC"
    assert merged.may_prune is False


def test_merge_conflicting_same_id_fails() -> None:
    left = _constraint(parameters={"role_id": "a"})
    right = _constraint(parameters={"role_id": "b"})
    with pytest.raises(ValueError, match="conflicting GoalConstraintV1"):
        merge_goal_constraints(left, right)


@pytest.mark.parametrize("authority", ["compiler-hard", "verifier-hard"])
def test_hard_authorities_deserialize_for_compiled_constraints(authority: str) -> None:
    constraint = _constraint(authority_tier=authority)
    restored = GoalConstraintV1.from_dict(constraint.to_dict())
    assert restored.authority_tier == authority


def test_compiled_set_from_dict_rejects_unknown_version() -> None:
    payload = _compiled_set().to_dict()
    payload["schema_version"] = "compiled_goal_constraint_set/v9"
    with pytest.raises(ValueError, match="unsupported CompiledGoalConstraintSetV1 version"):
        CompiledGoalConstraintSetV1.from_dict(payload)


def test_evaluation_record_round_trip() -> None:
    evaluation = GoalConstraintEvaluationV1(
        constraint_id="c_slot_button",
        status="UNKNOWN",
        detail="candidate not evaluated in fixture mode",
    )
    restored = GoalConstraintEvaluationV1.from_dict(evaluation.to_dict())
    assert restored == evaluation


def test_evaluation_rejects_unknown_version() -> None:
    payload = GoalConstraintEvaluationV1(
        constraint_id="c",
        status="PASS",
    ).to_dict()
    payload["schema_version"] = "goal_constraint_evaluation/v9"
    with pytest.raises(ValueError, match="unsupported GoalConstraintEvaluationV1 version"):
        GoalConstraintEvaluationV1.from_dict(payload)


def test_authority_matrix_is_machine_readable() -> None:
    matrix = authority_matrix()
    assert matrix["may_prune_requires"]["source_kind"] == [
        "generation_request",
        "pack_contract",
        "verification_requirement",
    ]
    assert "compiler-hard" in matrix["partitions"]["hard_constraint_ids"]


def test_public_schema_versions_are_stable() -> None:
    assert GOAL_CONSTRAINT_SCHEMA_VERSION == "goal_constraint/v1"
    assert COMPILED_GOAL_CONSTRAINT_SET_SCHEMA_VERSION == "compiled_goal_constraint_set/v1"
    assert GOAL_CONSTRAINT_EVALUATION_SCHEMA_VERSION == "goal_constraint_evaluation/v1"
