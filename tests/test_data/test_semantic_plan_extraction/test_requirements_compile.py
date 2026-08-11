"""Tests for PGS-A02 goal-constraint compilation."""

from __future__ import annotations

import pytest

from slm_training.data.contract import GenerationRequest, RuntimeSymbol
from slm_training.data.progspec.goal_constraints import CompiledGoalConstraintSetV1
from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.progspec.synthesis_problem import (
    PackIdentityV1,
    RuntimeSymbolDeclarationV1,
    VerificationRequirementV1,
    VerifiedSynthesisProblemV1,
)
from slm_training.data.semantic_plan.requirements_compile import (
    COMPILER_VERSION,
    GoldTargetAccessError,
    IdentityMismatchError,
    compile_goal_constraints,
    compile_rule_table,
    digest_recipe,
    request_production_digest,
    source_to_authority_matrix,
)
from slm_training.data.semantic_plan.requirements_extract import extract_prompt_requirements
from slm_training.dsl.pack import get_pack


def _problem(**overrides: object) -> VerifiedSynthesisProblemV1:
    defaults: dict[str, object] = {
        "problem_id": "p_compile",
        "pack_identity": PackIdentityV1(pack_id="openui"),
        "runtime_symbols": (),
        "requirements": PromptSemanticRequirementsV1(facts=()),
    }
    defaults.update(overrides)
    return VerifiedSynthesisProblemV1(**defaults)


def _request(**overrides: object) -> GenerationRequest:
    defaults: dict[str, object] = {"prompt": "Build a form with a Button"}
    defaults.update(overrides)
    return GenerationRequest(**defaults)


def _aligned_requirements(
    request: GenerationRequest,
    requirements: PromptSemanticRequirementsV1,
) -> PromptSemanticRequirementsV1:
    return requirements.model_copy(
        update={"prompt_context_hash": request_production_digest(request)}
    )


def _compile(
    *,
    problem: VerifiedSynthesisProblemV1 | None = None,
    request: GenerationRequest | None = None,
    pack_id: str = "openui",
    **kwargs: object,
) -> CompiledGoalConstraintSetV1:
    req = request or _request()
    prob = problem or _problem()
    return compile_goal_constraints(prob, req, get_pack(pack_id), **kwargs)


def test_exact_slot_inventory_compiles_as_hard_constraint() -> None:
    request = _request(
        prompt="Fill slots.",
        slot_contract=(":slot_0", ":slot_1"),
    )
    compiled = _compile(request=request)
    by_id = {item.constraint_id: item for item in compiled.constraints}
    constraint = by_id["hard.slot_inventory_exact"]
    assert constraint.kind == "slot_inventory_exact"
    assert constraint.parameters["slots"] == [":slot_0", ":slot_1"]
    assert constraint.authority_tier == "compiler-hard"
    assert constraint.completeness == "EXACT"
    assert constraint.may_prune is False
    assert constraint.constraint_id in compiled.hard_constraint_ids


def test_runtime_symbols_compile_as_hard_constraints() -> None:
    request = _request(
        prompt="Use symbols.",
        runtime_symbols=(
            RuntimeSymbol(surface="$state", role="state"),
            RuntimeSymbol(surface=":slot_0", role="external_entity"),
        ),
    )
    problem = _problem(
        runtime_symbols=(
            RuntimeSymbolDeclarationV1(surface="$state", role="state"),
            RuntimeSymbolDeclarationV1(surface=":slot_0", role="external_entity"),
        )
    )
    compiled = _compile(problem=problem, request=request)
    surfaces = {
        item.parameters["surface"]
        for item in compiled.constraints
        if item.kind == "runtime_symbol_accounted"
    }
    assert surfaces == {"$state", ":slot_0"}
    assert all(item.may_prune is False for item in compiled.constraints if item.kind == "runtime_symbol_accounted")


def test_output_kind_and_category_compile_as_hard_constraints() -> None:
    request = _request(
        prompt="Render UI.",
        output_kind="statement",
        output_category="dashboard.card",
    )
    compiled = _compile(request=request)
    kinds = {item.kind: item for item in compiled.constraints}
    assert kinds["output_kind_equals"].parameters == {"output_kind": "statement"}
    assert kinds["output_category_equals"].parameters == {"output_category": "dashboard.card"}


def test_advisory_component_prose_stays_advisory_and_non_pruning() -> None:
    request = _request(prompt="Build a form with a Button")
    requirements = _aligned_requirements(
        request, extract_prompt_requirements(request, canonicalize=False)
    )
    compiled = _compile(
        request=request,
        problem=_problem(requirements=requirements),
    )
    advisory = [
        item
        for item in compiled.constraints
        if item.constraint_id.startswith("compiled.sgs004.prompt_component_required")
        and item.parameters.get("component") == "Button"
    ]
    assert len(advisory) == 1
    assert advisory[0].kind == "component_count_at_least"
    assert advisory[0].parameters["component"] == "Button"
    assert advisory[0].authority_tier == "advisory-learned"
    assert advisory[0].may_prune is False
    assert advisory[0].constraint_id in compiled.advisory_constraint_ids


def test_unknown_statement_recorded_as_uncompiled() -> None:
    fact = RequirementFact(
        fact_id="custom.unknown.v1:1",
        factor_family="style_layout",
        disposition="required",
        statement="make it look modern and polished",
        confidence=0.4,
        provenance="predicted",
        authority="advisory-learned",
    )
    compiled = _compile(
        problem=_problem(requirements=PromptSemanticRequirementsV1(facts=(fact,))),
    )
    assert "custom.unknown.v1:1" in compiled.uncompiled_source_ids
    assert not any(
        constraint.source_id.endswith("custom.unknown.v1:1")
        for constraint in compiled.constraints
    )


def test_evaluation_fixture_stays_evaluation_only() -> None:
    fact = RequirementFact(
        fact_id="eval.button.required",
        factor_family="inventory",
        disposition="required",
        statement="prompt requires component Button",
        confidence=1.0,
        provenance="retrieved",
        authority="evaluation-only",
    )
    compiled = _compile(
        problem=_problem(requirements=PromptSemanticRequirementsV1(facts=(fact,))),
    )
    compiled_fact = next(
        item for item in compiled.constraints if item.constraint_id == "compiled.eval.button.required"
    )
    assert compiled_fact.authority_tier == "evaluation-only"
    assert compiled_fact.source_kind == "evaluation_fixture"
    assert compiled_fact.may_prune is False
    assert compiled_fact.constraint_id in compiled.evaluation_constraint_ids
    assert compiled_fact.constraint_id not in compiled.hard_constraint_ids


def test_unresolved_either_or_keeps_ambiguity_group() -> None:
    request = _request(prompt="Use either a Button or an Input")
    requirements = _aligned_requirements(
        request, extract_prompt_requirements(request, canonicalize=False)
    )
    compiled = _compile(
        request=request,
        problem=_problem(requirements=requirements),
    )
    assert len(compiled.ambiguity_groups) == 1
    group = compiled.ambiguity_groups[0]
    assert len(group.member_constraint_ids) == 2
    members = [
        item
        for item in compiled.constraints
        if item.constraint_id in group.member_constraint_ids
    ]
    assert {item.parameters["component"] for item in members} == {"Button", "Input"}
    assert all(item.parameters.get("alternative") is True for item in members)


def test_identity_mismatch_fails_closed() -> None:
    request = _request()
    problem = _problem(
        pack_identity=PackIdentityV1(pack_id="symbolic_regression"),
        requirements=PromptSemanticRequirementsV1(
            prompt_context_hash="deadbeef" * 8,
        ),
    )
    with pytest.raises(IdentityMismatchError, match="pack_id"):
        _compile(problem=problem, request=request, pack_id="openui")

    problem_hash = _problem(
        requirements=PromptSemanticRequirementsV1(
            prompt_context_hash="deadbeef" * 8,
        )
    )
    with pytest.raises(IdentityMismatchError, match="prompt_context_hash"):
        _compile(problem=problem_hash, request=request)


def test_attempted_gold_leakage_rejected() -> None:
    request = _request()
    problem = _problem(provenance={"gold_ast": {"typeName": "Button"}})
    with pytest.raises(GoldTargetAccessError, match="gold/target"):
        _compile(problem=problem, request=request)
    with pytest.raises(GoldTargetAccessError, match="gold/target"):
        compile_goal_constraints(problem, request, get_pack("openui"), target_ast={})


def test_reordering_requirements_is_digest_stable() -> None:
    request = _request(
        prompt="Fill slots.",
        slot_contract=(":slot_1", ":slot_0"),
        runtime_symbols=(RuntimeSymbol(surface=":slot_1", role="external_entity"),),
    )
    problem = _problem(
        runtime_symbols=(
            RuntimeSymbolDeclarationV1(surface=":slot_1", role="external_entity"),
        ),
        verification_requirements=(
            VerificationRequirementV1(requirement_id="b", kind="gate", gate="G3"),
            VerificationRequirementV1(requirement_id="a", kind="gate", gate="G2"),
        ),
    )
    first = _compile(problem=problem, request=request)
    second = _compile(problem=problem, request=request)
    assert first.digest == second.digest


def test_changing_structured_input_changes_digest() -> None:
    base = _compile(
        request=_request(slot_contract=(":slot_0",)),
        problem=_problem(
            runtime_symbols=(
                RuntimeSymbolDeclarationV1(surface=":slot_0", role="external_entity"),
            )
        ),
    )
    changed = _compile(
        request=_request(slot_contract=(":slot_0", ":slot_1")),
        problem=_problem(
            runtime_symbols=(
                RuntimeSymbolDeclarationV1(surface=":slot_0", role="external_entity"),
            )
        ),
    )
    assert changed.digest != base.digest


def test_compiler_metadata_exports() -> None:
    table = compile_rule_table()
    assert table["compiler_version"] == COMPILER_VERSION
    assert "prompt_component_required" in table["rules"]
    matrix = source_to_authority_matrix()
    assert matrix["prompt_requirement"]["may_prune"] is False
    assert matrix["pack_contract"]["may_prune"] is True
    assert "request_production_digest" in digest_recipe()


def test_bound_identities_match_inputs() -> None:
    request = _request(output_category="card")
    problem = _problem()
    compiled = _compile(problem=problem, request=request)
    assert compiled.problem_digest == problem.digest
    assert compiled.request_digest == request_production_digest(request)
    assert compiled.compiler_version == COMPILER_VERSION
