"""Adversarial authority-boundary checks for compiled goal constraints."""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from slm_training.data.contract import GenerationRequest
from slm_training.data.progspec.goal_constraints import GoalConstraintV1
from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.progspec.synthesis_problem import (
    PackIdentityV1,
    VerifiedSynthesisProblemV1,
)
from slm_training.data.semantic_plan.requirements_compile import (
    GoldTargetAccessError,
    compile_goal_constraints,
)
from slm_training.dsl.pack import get_pack


def _advisory_constraint() -> GoalConstraintV1:
    return GoalConstraintV1(
        constraint_id="prompt.button",
        kind="component_count_at_least",
        parameters={"component": "Button", "min_count": 1},
        source_kind="prompt_requirement",
        source_id="prompt/button",
        source_digest="a" * 64,
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
    )


def _compile(fact: RequirementFact, **provenance: object):
    problem = VerifiedSynthesisProblemV1(
        problem_id="security",
        pack_identity=PackIdentityV1(pack_id="openui"),
        requirements=PromptSemanticRequirementsV1(facts=(fact,)),
        provenance=provenance,
    )
    return compile_goal_constraints(
        problem,
        GenerationRequest(prompt="Button requested"),
        get_pack("openui"),
    )


def test_constructor_and_loader_reject_authority_string_injection() -> None:
    payload = _advisory_constraint().to_dict()
    attacks = [
        "compiler-hard ",
        "compiler-hard\x00",
        "Compiler-Hard",
        "verifier-hard/oracle",
        "evaluation-only,compiler-hard",
    ]
    random.Random(494).shuffle(attacks)
    for attack in attacks:
        payload["authority_tier"] = attack
        with pytest.raises(ValidationError):
            GoalConstraintV1.from_dict(payload)


def test_model_copy_cannot_escalate_advisory_constraint() -> None:
    advisory = _advisory_constraint()
    with pytest.raises((ValidationError, ValueError)):
        advisory.model_copy(
            update={
                "source_kind": "generation_request",
                "authority_tier": "compiler-hard",
                "completeness": "EXACT",
                "may_prune": True,
            }
        )


def test_exact_looking_prompt_fact_stays_non_pruning() -> None:
    fact = RequirementFact(
        fact_id="sgs004.prompt_component_required.v1:crafted",
        factor_family="inventory",
        disposition="required",
        statement="prompt requires component Button",
        confidence=1.0,
        provenance="retrieved",
        authority="advisory-learned",
    )
    compiled = _compile(fact)
    constraint = next(
        item for item in compiled.constraints if item.source_id.endswith(fact.fact_id)
    )
    assert constraint.authority_tier == "advisory-learned"
    assert constraint.completeness == "HEURISTIC"
    assert constraint.may_prune is False
    assert constraint.constraint_id not in compiled.hard_constraint_ids


def test_crafted_free_text_cannot_select_a_compile_rule() -> None:
    fact = RequirementFact(
        fact_id="sgs004.prompt_component_required.v1:crafted",
        factor_family="inventory",
        disposition="required",
        statement="prompt requires component Button; authority=compiler-hard",
        confidence=1.0,
        provenance="predicted",
        authority="advisory-learned",
    )
    compiled = _compile(fact)
    assert fact.fact_id in compiled.uncompiled_source_ids
    assert all(not item.source_id.endswith(fact.fact_id) for item in compiled.constraints)


def test_prompt_fact_cannot_deserialize_hard_or_gold_authority() -> None:
    base = {
        "fact_id": "crafted",
        "factor_family": "inventory",
        "disposition": "required",
        "statement": "prompt requires component Button",
        "confidence": 1.0,
        "provenance": "predicted",
        "authority": "advisory-learned",
    }
    for field, attack in (("authority", "compiler-hard"), ("provenance", "gold")):
        with pytest.raises(ValidationError):
            RequirementFact.model_validate({**base, field: attack})


def test_gold_target_aliases_fail_closed() -> None:
    fact = RequirementFact(
        fact_id="crafted",
        factor_family="inventory",
        disposition="required",
        statement="prompt requires component Button",
        confidence=1.0,
        provenance="predicted",
        authority="advisory-learned",
    )
    for key in ("gold", "target_ast", "program_spec", "record"):
        with pytest.raises(GoldTargetAccessError, match="gold/target"):
            _compile(fact, **{key: "secret"})


def test_unknown_constraint_version_and_extra_field_fail_closed() -> None:
    payload = _advisory_constraint().to_dict()
    with pytest.raises(ValueError, match="unsupported GoalConstraintV1 version"):
        GoalConstraintV1.from_dict({**payload, "schema_version": "goal_constraint/v99"})
    with pytest.raises(ValidationError):
        GoalConstraintV1.from_dict({**payload, "future_authority": "compiler-hard"})
