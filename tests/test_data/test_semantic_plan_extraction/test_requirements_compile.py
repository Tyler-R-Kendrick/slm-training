"""Tests for PGS-A02 goal-constraint compilation and PGS-A03 evaluation."""

from __future__ import annotations

import hashlib

import pytest

from slm_training.data.contract import GenerationRequest, RuntimeSymbol
from slm_training.data.progspec.goal_constraints import (
    CompiledGoalConstraintSetV1,
    GoalConstraintAmbiguityGroup,
    GoalConstraintEvaluationV1,
    GoalConstraintV1,
)
from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import (
    PlanBinding,
    PlanIdentity,
    PlanSymbol,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.data.progspec.synthesis_problem import (
    PackIdentityV1,
    RuntimeSymbolDeclarationV1,
    VerificationRequirementV1,
    VerifiedSynthesisProblemV1,
)
from slm_training.data.semantic_plan.requirements_compile import (
    COMPILER_VERSION,
    EVALUATOR_VERSION,
    GoldTargetAccessError,
    IdentityMismatchError,
    compile_goal_constraints,
    compile_rule_table,
    deferred_constraint_kinds,
    digest_recipe,
    evaluate_goal_constraints,
    evaluation_state_decision_table,
    evaluator_digest,
    evaluator_identity,
    request_production_digest,
    source_to_authority_matrix,
    supported_constraint_kinds,
)
from slm_training.data.semantic_plan.requirements_extract import extract_prompt_requirements
from slm_training.dsl.language_contract import contract_id
from slm_training.dsl.pack import get_pack
from slm_training.dsl.placeholders import extract_placeholders


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _evaluate(
    *,
    source: str,
    program_spec: ProgramSpec,
    plan: SemanticPlanV1,
    constraints: CompiledGoalConstraintSetV1,
) -> tuple[GoalConstraintEvaluationV1, ...]:
    return evaluate_goal_constraints(
        source=source,
        program_spec=program_spec,
        plan=plan,
        constraints=constraints,
    )


def _plan(**overrides: object) -> SemanticPlanV1:
    defaults: dict[str, object] = {
        "identity": PlanIdentity(pack_id="openui", provenance="predicted"),
    }
    defaults.update(overrides)
    return SemanticPlanV1(**defaults)


def _constraint_set(*constraints: GoalConstraintV1) -> CompiledGoalConstraintSetV1:
    hard = [item.constraint_id for item in constraints if item.authority_tier in {"compiler-hard", "verifier-hard"}]
    advisory = [
        item.constraint_id
        for item in constraints
        if item.authority_tier in {"advisory-learned", "oracle-diagnostic"}
    ]
    evaluation = [item.constraint_id for item in constraints if item.authority_tier == "evaluation-only"]
    payload = CompiledGoalConstraintSetV1(
        problem_digest="a" * 64,
        request_digest="b" * 64,
        pack_identity_digest="c" * 64,
        compiler_version=COMPILER_VERSION,
        constraints=constraints,
        hard_constraint_ids=tuple(hard),
        advisory_constraint_ids=tuple(advisory),
        evaluation_constraint_ids=tuple(evaluation),
    ).to_dict(include_digest=True)
    return CompiledGoalConstraintSetV1.from_dict(payload)


def _program_spec(openui: str, **overrides: object) -> ProgramSpec:
    spec_id = str(overrides.pop("id", "spec_eval"))
    ast = overrides.pop("ast", None)
    if ast is None:
        placeholders = extract_placeholders(openui)
        ast = {
            "type": "element",
            "typeName": "TextContent",
            "statementId": "root",
            "props": {"text": placeholders[0] if placeholders else ":copy.value"},
        }
    defaults: dict[str, object] = {
        "ast": ast,
        "canonical_openui": openui,
        "facts": {},
        "contract_id": contract_id(),
        "program_family_id": "pf1",
        "lineage_id": "ln1",
        "split_group_id": "sg1",
    }
    defaults.update(overrides)
    return ProgramSpec(id=spec_id, **defaults)  # type: ignore[arg-type]


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


def test_evaluate_slot_inventory_exact_pass_and_fail() -> None:
    source = 'root = Stack([title])\ntitle = TextContent(":hello.text")'
    spec = _program_spec(source)
    plan = _plan()
    passing = _constraint_set(
        GoalConstraintV1(
            constraint_id="hard.slot_inventory_exact",
            kind="slot_inventory_exact",
            parameters={"slots": [":hello.text"]},
            source_kind="generation_request",
            source_id="generation_request.slot_contract",
            source_digest=_digest("slot-pass"),
            authority_tier="compiler-hard",
            completeness="EXACT",
            may_prune=False,
        )
    )
    results = _evaluate(source=source, program_spec=spec, plan=plan, constraints=passing)
    assert results[0].status == "PASS"
    assert results[0].reason_code == "slot_inventory_exact_match"

    failing = _constraint_set(
        GoalConstraintV1(
            constraint_id="hard.slot_inventory_exact",
            kind="slot_inventory_exact",
            parameters={"slots": [":hello.text", ":missing.slot"]},
            source_kind="generation_request",
            source_id="generation_request.slot_contract",
            source_digest=_digest("slot-fail"),
            authority_tier="compiler-hard",
            completeness="EXACT",
            may_prune=False,
        )
    )
    fail_results = _evaluate(source=source, program_spec=spec, plan=plan, constraints=failing)
    assert fail_results[0].status == "FAIL"
    assert fail_results[0].reason_code == "slot_inventory_mismatch"


def test_evaluate_runtime_symbol_accounted() -> None:
    source = '$state = ":slot_0"\nroot = TextContent($state)'
    spec = _program_spec(
        source,
        ast={
            "type": "element",
            "typeName": "TextContent",
            "statementId": "root",
            "props": {"text": {"type": "ref", "name": "$state"}},
            "zdefs": [
                {
                    "type": "element",
                    "typeName": "State",
                    "statementId": "$state",
                    "props": {"value": ":slot_0"},
                }
            ],
        },
    )
    plan = _plan()
    constraints = _constraint_set(
        GoalConstraintV1(
            constraint_id="hard.runtime_symbol.state",
            kind="runtime_symbol_accounted",
            parameters={"surface": "$state", "role": "state"},
            source_kind="generation_request",
            source_id="generation_request.runtime_symbols:$state",
            source_digest=_digest("runtime-state"),
            authority_tier="compiler-hard",
            completeness="EXACT",
            may_prune=False,
        ),
        GoalConstraintV1(
            constraint_id="hard.runtime_symbol.slot",
            kind="runtime_symbol_accounted",
            parameters={"surface": ":slot_0", "role": "external_entity"},
            source_kind="generation_request",
            source_id="generation_request.runtime_symbols::slot_0",
            source_digest=_digest("runtime-slot"),
            authority_tier="compiler-hard",
            completeness="EXACT",
            may_prune=False,
        ),
    )
    results = {
        item.constraint_id: item
        for item in _evaluate(source=source, program_spec=spec, plan=plan, constraints=constraints)
    }
    assert results["hard.runtime_symbol.state"].status == "PASS"
    assert results["hard.runtime_symbol.slot"].status == "PASS"


def test_evaluate_component_inventory_complete_vs_unavailable() -> None:
    constraint = GoalConstraintV1(
        constraint_id="adv.button",
        kind="component_count_at_least",
        parameters={"component": "Button", "min_count": 1},
        source_kind="prompt_requirement",
        source_id="compiled:fact",
        source_digest=_digest("component-count"),
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
    )
    constraints = _constraint_set(constraint)
    source = 'root = Button(":cta.label")'
    spec = _program_spec(source)
    complete_plan = _plan(
        role_slots=(
            RoleSlot(role_id="role_0000", component_family="Button"),
        )
    )
    complete = _evaluate(source=source, program_spec=spec, plan=complete_plan, constraints=constraints)[0]
    assert complete.status == "PASS"

    incomplete_plan = _plan(
        role_slots=(
            RoleSlot(role_id="role_0000", component_family=None),
        )
    )
    unknown = _evaluate(source=source, program_spec=spec, plan=incomplete_plan, constraints=constraints)[0]
    assert unknown.status == "UNKNOWN"
    assert unknown.reason_code == "component_inventory_unavailable"


def test_evaluate_component_absent_requires_complete_inventory() -> None:
    constraint = GoalConstraintV1(
        constraint_id="adv.no_button",
        kind="component_absent",
        parameters={"component": "Button"},
        source_kind="prompt_requirement",
        source_id="compiled:forbidden",
        source_digest=_digest("component-absent"),
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
    )
    constraints = _constraint_set(constraint)
    source = 'root = TextContent(":copy.value")'
    spec = _program_spec(source)
    complete_plan = _plan(
        role_slots=(
            RoleSlot(role_id="role_0000", component_family="TextContent"),
        )
    )
    assert _evaluate(source=source, program_spec=spec, plan=complete_plan, constraints=constraints)[0].status == "PASS"

    incomplete_plan = _plan(role_slots=(RoleSlot(role_id="role_0000", component_family=None),))
    unavailable = _evaluate(source=source, program_spec=spec, plan=incomplete_plan, constraints=constraints)[0]
    assert unavailable.status == "UNKNOWN"


def test_evaluate_binding_resolved_unresolved_unavailable() -> None:
    constraint = GoalConstraintV1(
        constraint_id="bind.role",
        kind="binding_resolved",
        parameters={"role_slot_id": "role_0000"},
        source_kind="pack_contract",
        source_id="pack/bind",
        source_digest=_digest("binding"),
        authority_tier="compiler-hard",
        completeness="HEURISTIC",
        may_prune=False,
    )
    constraints = _constraint_set(constraint)
    source = 'root = TextContent(":copy.value")'
    spec = _program_spec(source)
    resolved_plan = _plan(
        role_slots=(RoleSlot(role_id="role_0000", component_family="TextContent"),),
        symbols=(PlanSymbol(symbol_id="sym_0000", semantic_role="text"),),
        bindings=(PlanBinding(role_slot_id="role_0000", candidate_symbols=("sym_0000",)),),
    )
    assert _evaluate(source=source, program_spec=spec, plan=resolved_plan, constraints=constraints)[0].status == "PASS"

    unresolved_plan = _plan(
        role_slots=(RoleSlot(role_id="role_0000", component_family="TextContent"),),
        bindings=(PlanBinding(role_slot_id="role_0000", candidate_symbols=()),),
    )
    assert _evaluate(source=source, program_spec=spec, plan=unresolved_plan, constraints=constraints)[0].status == "FAIL"

    unavailable_plan = _plan()
    assert _evaluate(source=source, program_spec=spec, plan=unavailable_plan, constraints=constraints)[0].status == "UNKNOWN"


def test_evaluate_topology_known_and_unknown() -> None:
    constraint = GoalConstraintV1(
        constraint_id="topo.contains",
        kind="topology_relation_present",
        parameters={
            "parent_role_id": "role_parent",
            "child_role_id": "role_child",
            "relation": "contains",
        },
        source_kind="pack_contract",
        source_id="pack/topo",
        source_digest=_digest("topology"),
        authority_tier="compiler-hard",
        completeness="HEURISTIC",
        may_prune=False,
    )
    constraints = _constraint_set(constraint)
    source = 'root = TextContent(":copy.value")'
    spec = _program_spec(source)
    known_plan = _plan(
        topology=PlanTopology(
            parent_relation_candidates=(
                {
                    "parent_role_id": "role_parent",
                    "child_role_id": "role_child",
                    "relation": "contains",
                },
            )
        )
    )
    assert _evaluate(source=source, program_spec=spec, plan=known_plan, constraints=constraints)[0].status == "PASS"
    assert _evaluate(source=source, program_spec=spec, plan=_plan(), constraints=constraints)[0].status == "UNKNOWN"


def test_evaluate_ambiguity_forces_unknown() -> None:
    left = GoalConstraintV1(
        constraint_id="alt.button",
        kind="component_count_at_least",
        parameters={"component": "Button", "min_count": 0, "alternative": True},
        source_kind="prompt_requirement",
        source_id="compiled:button",
        source_digest=_digest("ambiguity-button"),
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
    )
    right = GoalConstraintV1(
        constraint_id="alt.input",
        kind="component_count_at_least",
        parameters={"component": "Input", "min_count": 0, "alternative": True},
        source_kind="prompt_requirement",
        source_id="compiled:input",
        source_digest=_digest("ambiguity-input"),
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
    )
    payload = CompiledGoalConstraintSetV1(
        problem_digest="a" * 64,
        request_digest="b" * 64,
        pack_identity_digest="c" * 64,
        compiler_version=COMPILER_VERSION,
        constraints=(left, right),
        hard_constraint_ids=(),
        advisory_constraint_ids=(left.constraint_id, right.constraint_id),
        evaluation_constraint_ids=(),
        ambiguity_groups=(
            GoalConstraintAmbiguityGroup(
                group_id="ambiguity.0000",
                member_constraint_ids=(left.constraint_id, right.constraint_id),
            ),
        ),
    ).to_dict(include_digest=True)
    constraints = CompiledGoalConstraintSetV1.from_dict(payload)
    source = 'root = Button(":cta.label")'
    spec = _program_spec(source)
    plan = _plan(role_slots=(RoleSlot(role_id="role_0000", component_family="Button"),))
    results = _evaluate(source=source, program_spec=spec, plan=plan, constraints=constraints)
    assert all(item.status == "UNKNOWN" for item in results)
    assert all(item.reason_code == "ambiguity_unresolved" for item in results)


def test_evaluate_result_order_is_canonical() -> None:
    z = GoalConstraintV1(
        constraint_id="z.last",
        kind="required_gate_passes",
        parameters={"gate": "G3"},
        source_kind="verification_requirement",
        source_id="verification:z",
        source_digest=_digest("order-gate"),
        authority_tier="verifier-hard",
        completeness="EXACT",
        may_prune=True,
    )
    a = GoalConstraintV1(
        constraint_id="a.first",
        kind="slot_inventory_exact",
        parameters={"slots": []},
        source_kind="generation_request",
        source_id="generation_request.slot_contract",
        source_digest=_digest("order-slot"),
        authority_tier="compiler-hard",
        completeness="EXACT",
        may_prune=False,
    )
    constraints = _constraint_set(a, z)
    source = 'root = TextContent(":copy.value")'
    spec = _program_spec(source)
    plan = _plan()
    ids = [item.constraint_id for item in _evaluate(source=source, program_spec=spec, plan=plan, constraints=constraints)]
    assert ids == ["a.first", "z.last"]


def test_evaluate_redacts_raw_content_from_detail() -> None:
    with pytest.raises(ValueError, match="raw prompt/program content"):
        GoalConstraintEvaluationV1(
            constraint_id="c",
            status="FAIL",
            authority_tier="advisory-learned",
            may_prune=False,
            detail='missing slot :slot_0 in root = Button(":slot_0")',
        )


def test_evaluate_preserves_advisory_authority() -> None:
    constraint = GoalConstraintV1(
        constraint_id="adv.button",
        kind="component_count_at_least",
        parameters={"component": "Button", "min_count": 1},
        source_kind="prompt_requirement",
        source_id="compiled:fact",
        source_digest=_digest("advisory"),
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
    )
    constraints = _constraint_set(constraint)
    source = 'root = Button(":cta.label")'
    spec = _program_spec(source)
    plan = _plan(role_slots=(RoleSlot(role_id="role_0000", component_family="Button"),))
    result = _evaluate(source=source, program_spec=spec, plan=plan, constraints=constraints)[0]
    assert result.authority_tier == "advisory-learned"
    assert result.may_prune is False
    assert result.completeness_achieved == "HEURISTIC"


def test_evaluate_required_gate_passes_is_deferred_unknown() -> None:
    constraint = GoalConstraintV1(
        constraint_id="hard.verification.g3",
        kind="required_gate_passes",
        parameters={"gate": "G3"},
        source_kind="verification_requirement",
        source_id="verification:g3",
        source_digest=_digest("gate-deferred"),
        authority_tier="verifier-hard",
        completeness="EXACT",
        may_prune=True,
    )
    constraints = _constraint_set(constraint)
    source = 'root = TextContent(":copy.value")'
    spec = _program_spec(source)
    result = _evaluate(source=source, program_spec=spec, plan=_plan(), constraints=constraints)[0]
    assert result.status == "UNKNOWN"
    assert result.reason_code == "deferred_gate_verifier"


def test_evaluate_output_kind_not_applicable_without_candidate_identity() -> None:
    constraint = GoalConstraintV1(
        constraint_id="hard.output_kind_equals",
        kind="output_kind_equals",
        parameters={"output_kind": "statement"},
        source_kind="generation_request",
        source_id="generation_request.output_kind",
        source_digest=_digest("output-kind"),
        authority_tier="compiler-hard",
        completeness="EXACT",
        may_prune=False,
    )
    constraints = _constraint_set(constraint)
    source = 'root = TextContent(":copy.value")'
    spec = _program_spec(source)
    result = _evaluate(source=source, program_spec=spec, plan=_plan(), constraints=constraints)[0]
    assert result.status == "NOT_APPLICABLE"


def test_evaluate_is_byte_deterministic() -> None:
    request = _request(
        prompt="Fill slots.",
        slot_contract=(":slot_0",),
    )
    compiled = _compile(request=request)
    source = 'root = TextContent(":slot_0")'
    spec = _program_spec(source)
    plan = _plan(role_slots=(RoleSlot(role_id="role_0000", component_family="TextContent"),))
    first = _evaluate(source=source, program_spec=spec, plan=plan, constraints=compiled)
    second = _evaluate(source=source, program_spec=spec, plan=plan, constraints=compiled)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]


def test_evaluator_metadata_exports() -> None:
    assert EVALUATOR_VERSION.startswith("pgs-a03")
    assert "required_gate_passes" in deferred_constraint_kinds()
    assert "slot_inventory_exact" in supported_constraint_kinds()
    assert set(evaluation_state_decision_table()) >= {"PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"}
    assert evaluator_identity()["evaluator_digest"] == evaluator_digest()
    assert "evaluator_digest" in digest_recipe()
