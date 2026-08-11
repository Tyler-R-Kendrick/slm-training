"""Tests for OpenUIGoalVerifier terminal tri-state semantics (PGS-B02 / SLM-498)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import re

import pytest

from slm_training.data.contract import GenerationRequest
from slm_training.data.progspec.goal_constraints import (
    CompiledGoalConstraintSetV1,
    GoalConstraintV1,
)
from slm_training.data.progspec.synthesis_problem import PackIdentityV1
from slm_training.data.semantic_plan.requirements_compile import (
    COMPILER_VERSION,
    compile_goal_constraints,
    request_production_digest,
    source_digest,
)
from slm_training.data.progspec.synthesis_problem import VerifiedSynthesisProblemV1
from slm_training.data.progspec.prompt_requirements import PromptSemanticRequirementsV1
from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.verify.stack import VerificationContext
from slm_training.dsl import lang_core
from slm_training.dsl.language_contract import contract_id
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.dsl.solver.goal_support import (
    EvaluatorIdentityV1,
    GoalVerifierProfileV1,
    compute_pack_identity_digest,
    redact_bounded_string,
)
from slm_training.dsl.solver.openui_support import (
    GOAL_SUPPORT_PROFILE_PREFIX,
    GoalTerminalEvidenceTrace,
    OpenUIGoalVerifier,
    OpenUIWellFormedVerifier,
    current_openui_pack_identity,
    goal_verifier_invocation_map,
    goal_verifier_status_table,
)
from slm_training.dsl.solver.support import VerifyOutcome, VerifyStatus
from slm_training.dsl.pack import get_pack


pytestmark = []  # structural probe applied per-test via fixture


class _AcceptingStructuralVerifier(OpenUIWellFormedVerifier):
    """Test double: accept well-formed ``root =`` programs without lang-core."""

    def verify(self, program: str) -> VerifyOutcome:
        stripped = program.strip()
        if stripped.startswith("root =") and "broken" not in stripped:
            return VerifyOutcome(VerifyStatus.ACCEPT)
        if not stripped:
            return VerifyOutcome(VerifyStatus.REJECT, detail="parse_error")
        return VerifyOutcome(VerifyStatus.REJECT, detail="parse_error")


def _ast_for_openui(openui: str) -> dict[str, object]:
    placeholders = extract_placeholders(openui)
    match = re.match(r"root\s*=\s*(\w+)", openui.strip())
    type_name = match.group(1) if match else "TextContent"
    prop = "text" if type_name == "TextContent" else "label"
    return {
        "type": "element",
        "typeName": type_name,
        "statementId": "root",
        "props": {prop: placeholders[0] if placeholders else ":copy.value"},
    }


def _manual_program_spec(
    *, program_id: str, openui: str, facts: dict[str, object] | None = None
) -> ProgramSpec:
    return ProgramSpec(
        id=program_id,
        ast=_ast_for_openui(openui),
        canonical_openui=openui.strip(),
        facts=facts or {},
        contract_id=contract_id(),
        program_family_id=program_id,
        lineage_id=program_id,
        split_group_id=program_id,
    )


def _manual_emit_record(
    spec: ProgramSpec,
    *,
    prompt: str,
    task: str,
    openui: str | None = None,
) -> ExampleRecord:
    target = openui or spec.canonical_openui
    return ExampleRecord(
        id=f"{spec.id}_{task}",
        prompt=prompt,
        openui=target,
        placeholders=list(extract_placeholders(target)),
        split=spec.split,
        source="test_fixture",
        meta={"contract_id": spec.contract_id},
    )


@pytest.fixture(autouse=True)
def _patch_bridge_free_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    from slm_training.data.verify.stack import Gate, GateResult, GateStatus, evaluate_gate as real_evaluate_gate

    def _stub_evaluate_gate(gate, record, context=None):
        if gate in {Gate.INDEPENDENT_JUDGE, Gate.HUMAN_AUDIT}:
            return real_evaluate_gate(gate, record, context)
        return GateResult(gate=gate, status=GateStatus.PASS)

    monkeypatch.setattr(
        ProgramSpec,
        "from_openui",
        lambda **kwargs: _manual_program_spec(
            program_id=str(kwargs["id"]),
            openui=str(kwargs["openui"]),
            facts=dict(kwargs.get("facts") or {}),
        ),
    )
    monkeypatch.setattr(
        "slm_training.dsl.solver.openui_support.emit_record",
        _manual_emit_record,
    )
    monkeypatch.setattr(
        "slm_training.dsl.solver.openui_support.evaluate_gate",
        _stub_evaluate_gate,
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _pack_identity(**overrides: object) -> PackIdentityV1:
    defaults: dict[str, object] = current_openui_pack_identity().model_dump()
    defaults.update(overrides)
    return PackIdentityV1(**defaults)


def _test_problem() -> VerifiedSynthesisProblemV1:
    return VerifiedSynthesisProblemV1(
        problem_id="openui-goal-verifier-test",
        pack_identity=_pack_identity(),
    )


def _test_request() -> GenerationRequest:
    return GenerationRequest(prompt="Build a Button", slot_contract=(":slot_0",))


def _constraint(**overrides: object) -> GoalConstraintV1:
    defaults: dict[str, object] = {
        "constraint_id": "c_slot_button",
        "kind": "slot_present",
        "parameters": {"role_id": "primary_action"},
        "source_kind": "pack_contract",
        "source_id": "pack/openui",
        "source_digest": _digest("source"),
        "authority_tier": "compiler-hard",
        "completeness": "EXACT",
        "may_prune": True,
    }
    defaults.update(overrides)
    return GoalConstraintV1(**defaults)


def _compiled_set(**overrides: object) -> CompiledGoalConstraintSetV1:
    hard = _constraint(
        constraint_id="hard.slot_inventory_exact",
        kind="slot_inventory_exact",
        parameters={"slots": [":slot_0"]},
        source_kind="generation_request",
        source_id="generation_request.slot_contract",
        source_digest=source_digest({"slots": [":slot_0"]}),
        may_prune=True,
    )
    advisory = _constraint(
        constraint_id="adv.hint",
        kind="component_count_at_least",
        parameters={"component": "Chart", "min_count": 1},
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
        source_kind="prompt_requirement",
    )
    defaults: dict[str, object] = {
        "problem_digest": _test_problem().digest,
        "request_digest": request_production_digest(
            _test_request()
        ),
        "pack_identity_digest": compute_pack_identity_digest(_pack_identity()),
        "compiler_version": COMPILER_VERSION,
        "constraints": (hard, advisory),
        "hard_constraint_ids": (hard.constraint_id,),
        "advisory_constraint_ids": (advisory.constraint_id,),
        "evaluation_constraint_ids": (),
    }
    defaults.update(overrides)
    payload = CompiledGoalConstraintSetV1(**defaults).to_dict(include_digest=True)
    return CompiledGoalConstraintSetV1.from_dict(payload)


def _profile_for(constraint_set: CompiledGoalConstraintSetV1, **overrides: object) -> GoalVerifierProfileV1:
    pack = _pack_identity()
    defaults: dict[str, object] = {
        "profile_id": "profile/prod",
        "mode": "production_exact",
        "problem_digest": constraint_set.problem_digest,
        "constraint_set_digest": constraint_set.digest,
        "pack_identity": pack,
        "pack_identity_digest": compute_pack_identity_digest(pack),
        "required_constraint_ids": constraint_set.hard_constraint_ids,
        "required_gates": (),
        "required_evaluators": (),
        "metric_identities": (),
        "authority_tier": "compiler-hard",
    }
    defaults.update(overrides)
    payload = GoalVerifierProfileV1(**defaults).to_dict(include_digest=True)
    return GoalVerifierProfileV1.from_dict(payload)


def _verifier(**overrides: object) -> OpenUIGoalVerifier:
    request = overrides.pop("request", _test_request())
    problem = overrides.pop("problem", _test_problem())
    constraint_set = overrides.pop(
        "constraint_set",
        _compiled_set(request_digest=request_production_digest(request)),
    )
    profile = overrides.pop("profile", _profile_for(constraint_set))
    structural_verifier = overrides.pop("structural_verifier", None)
    return OpenUIGoalVerifier(
        profile=profile,
        constraint_set=constraint_set,
        problem=problem,
        request=request,
        pack_identity=profile.pack_identity,
        structural_verifier=structural_verifier,
        **overrides,
    )


def test_reference_tables_and_profile_string_format() -> None:
    status_table = goal_verifier_status_table()
    assert status_table["mandatory_exact_fail"] == "REJECT"
    assert status_table["mandatory_skip_or_unknown"] == "UNAVAILABLE"
    invocation = goal_verifier_invocation_map()
    assert invocation["constraints"] == "evaluate_goal_constraints"
    assert invocation["gates"] == "evaluate_gate"

    verifier = _verifier()
    assert verifier.profile.startswith(f"{GOAL_SUPPORT_PROFILE_PREFIX}/")
    assert len(verifier.profile.split("/")[-1]) == 64


def test_production_profile_rejects_substitute_structural_verifier() -> None:
    with pytest.raises(ValueError, match="canonical OpenUI structural verifier"):
        _verifier(structural_verifier=_AcceptingStructuralVerifier())


def test_well_formed_but_exact_goal_fail_returns_reject() -> None:
    source = 'root = TextContent(":copy.value")'
    assert _AcceptingStructuralVerifier().verify(source).status is VerifyStatus.ACCEPT
    verifier = _verifier()
    outcome = verifier.verify(source)
    assert outcome.status is VerifyStatus.REJECT
    evidence = verifier.last_trace.latest()
    assert evidence is not None
    assert evidence.structural_status == "ACCEPT"
    assert evidence.overall_status == "REJECT"
    assert any(atom.source_kind == "constraint" for atom in evidence.mandatory_failure_atoms)


def test_fully_satisfying_terminal_returns_accept() -> None:
    source = 'root = Button(":slot_0")'
    verifier = _verifier()
    outcome = verifier.verify(source)
    assert outcome.status is VerifyStatus.ACCEPT
    evidence = verifier.last_trace.latest()
    assert evidence is not None
    assert evidence.overall_status == "ACCEPT"
    assert evidence.structural_status == "ACCEPT"
    assert not evidence.mandatory_failure_atoms
    assert not evidence.mandatory_unknown_atoms


def test_mandatory_skipped_gate_returns_unavailable() -> None:
    source = 'root = Button(":slot_0")'
    constraint_set = _compiled_set(
        constraints=(),
        hard_constraint_ids=(),
        advisory_constraint_ids=(),
        evaluation_constraint_ids=(),
    )
    profile = _profile_for(
        constraint_set,
        mode="evaluation_oracle",
        authority_tier="evaluation-only",
        required_constraint_ids=(),
        required_gates=("G11",),
    )
    verifier = _verifier(
        profile=profile,
        constraint_set=constraint_set,
        verification_context=VerificationContext(independent_judge_passed=None),
    )
    outcome = verifier.verify(source)
    assert outcome.status is VerifyStatus.UNAVAILABLE
    evidence = verifier.last_trace.latest()
    assert evidence is not None
    assert evidence.overall_status == "UNAVAILABLE"
    assert any(item.gate_id == "G11" for item in evidence.required_gate_results)
    assert any(atom.source_kind == "gate" for atom in evidence.mandatory_unknown_atoms)


@pytest.mark.parametrize(
    ("gate_id", "context"),
    (
        ("G11", VerificationContext(independent_judge_passed=False)),
        ("G12", VerificationContext(human_audit_passed=False)),
    ),
)
def test_heuristic_gate_rejection_has_no_exact_failure_atom(
    gate_id: str, context: VerificationContext
) -> None:
    constraint_set = _compiled_set(
        constraints=(),
        hard_constraint_ids=(),
        advisory_constraint_ids=(),
        evaluation_constraint_ids=(),
    )
    profile = _profile_for(
        constraint_set,
        mode="evaluation_oracle",
        authority_tier="evaluation-only",
        required_constraint_ids=(),
        required_gates=(gate_id,),
    )
    verifier = _verifier(
        profile=profile,
        constraint_set=constraint_set,
        verification_context=context,
    )
    outcome = verifier.verify('root = Button(":slot_0")')
    evidence = verifier.terminal_evidence[-1]
    assert outcome.status is VerifyStatus.REJECT
    assert evidence.overall_status == "REJECT"
    assert not evidence.mandatory_failure_atoms


def test_production_profile_rejects_uncertified_required_evaluator() -> None:
    constraint_set = _compiled_set()
    profile = _profile_for(
        constraint_set,
        required_evaluators=(
            EvaluatorIdentityV1(
                evaluator_id="meaningful_program/v2",
                version="v2",
                implementation_hash="c" * 64,
            ),
        ),
    )
    with pytest.raises(ValueError, match="no certified exact evaluator registry"):
        _verifier(
            profile=profile,
            constraint_set=constraint_set,
        )


def test_evaluation_evaluator_version_mismatch_is_unavailable() -> None:
    constraint_set = _compiled_set(
        constraints=(),
        hard_constraint_ids=(),
        advisory_constraint_ids=(),
        evaluation_constraint_ids=(),
    )
    profile = _profile_for(
        constraint_set,
        mode="evaluation_oracle",
        authority_tier="evaluation-only",
        required_constraint_ids=(),
        required_evaluators=(
            EvaluatorIdentityV1(
                evaluator_id="meaningful_program/v2",
                version="not-the-live-version",
                implementation_hash="0" * 64,
            ),
        ),
    )
    outcome = _verifier(profile=profile, constraint_set=constraint_set).verify(
        'root = Button(":slot_0")'
    )
    assert outcome.status is VerifyStatus.UNAVAILABLE


def test_evaluation_profile_cannot_emit_compiler_hard_failure_atoms() -> None:
    eval_constraint = _constraint(
        constraint_id="eval.metric",
        kind="slot_inventory_exact",
        parameters={"slots": [":slot_0"]},
        authority_tier="evaluation-only",
        may_prune=False,
        source_kind="evaluation_fixture",
    )
    constraint_set = _compiled_set(
        constraints=(eval_constraint,),
        hard_constraint_ids=(),
        advisory_constraint_ids=(),
        evaluation_constraint_ids=(eval_constraint.constraint_id,),
    )
    profile = _profile_for(
        constraint_set,
        mode="evaluation_oracle",
        authority_tier="evaluation-only",
        required_constraint_ids=(eval_constraint.constraint_id,),
        required_gates=(),
        required_evaluators=(
            EvaluatorIdentityV1(
                evaluator_id="meaningful_program/v2",
                version="v2",
                implementation_hash="c" * 64,
            ),
        ),
    )
    source = 'root = TextContent(":copy.value")'
    verifier = _verifier(profile=profile, constraint_set=constraint_set)
    outcome = verifier.verify(source)
    assert outcome.status is VerifyStatus.REJECT
    evidence = verifier.last_trace.latest()
    assert evidence is not None
    assert profile.authority_tier == "evaluation-only"
    assert not any(atom.source_kind == "evaluator" for atom in evidence.mandatory_failure_atoms)


def test_profile_mismatch_fails_closed_at_construction() -> None:
    constraint_set = _compiled_set()
    profile = _profile_for(constraint_set, constraint_set_digest="f" * 64)
    with pytest.raises(ValueError, match="constraint_set_digest is stale"):
        _verifier(profile=profile, constraint_set=constraint_set)


def test_stale_request_fails_closed_at_construction() -> None:
    constraint_set = _compiled_set(
        request_digest=request_production_digest(GenerationRequest(prompt="request A"))
    )
    with pytest.raises(ValueError, match="live request"):
        _verifier(
            request=GenerationRequest(prompt="request B"),
            constraint_set=constraint_set,
            profile=_profile_for(constraint_set),
        )


def test_self_asserted_hard_constraint_fails_source_replay() -> None:
    problem = _test_problem()
    request = _test_request()
    compiled = compile_goal_constraints(problem, request, get_pack("openui"))
    constraints = tuple(
        GoalConstraintV1(
            **{
                **item.model_dump(),
                "parameters": {"output_kind": "screen"},
                "source_digest": source_digest({"output_kind": "screen"}),
            }
        )
        if item.constraint_id == "hard.output_kind_equals"
        else item
        for item in compiled.constraints
    )
    payload = compiled.model_dump(exclude={"digest"})
    payload["constraints"] = constraints
    tampered = CompiledGoalConstraintSetV1.from_dict(
        CompiledGoalConstraintSetV1(**payload).to_dict(include_digest=True)
    )
    profile = _profile_for(tampered)
    with pytest.raises(ValueError, match="does not replay from live sources"):
        _verifier(
            problem=problem,
            request=request,
            constraint_set=tampered,
            profile=profile,
        )


def test_stale_pack_identity_fails_closed_at_construction() -> None:
    constraint_set = _compiled_set()
    profile = _profile_for(constraint_set)
    with pytest.raises(ValueError, match="pack identity"):
        OpenUIGoalVerifier(
            profile=profile,
            constraint_set=constraint_set,
            problem=_test_problem(),
            request=_test_request(),
            pack_identity=_pack_identity(tokenizer_id="stale"),
        )


def test_same_id_noncanonical_pack_fails_closed_at_construction() -> None:
    constraint_set = _compiled_set()
    profile = _profile_for(constraint_set)
    with pytest.raises(ValueError, match="canonical live pack"):
        OpenUIGoalVerifier(
            profile=profile,
            constraint_set=constraint_set,
            problem=_test_problem(),
            request=_test_request(),
            pack_identity=_pack_identity(),
            pack=replace(get_pack("openui")),
        )


def test_mandatory_not_applicable_is_unavailable() -> None:
    constraint = _constraint(
        constraint_id="ordinary-customer-literal",
        kind="output_category_equals",
        parameters={"output_category": "dashboard.card"},
        source_kind="evaluation_fixture",
        source_id="evaluation.output_category",
        authority_tier="evaluation-only",
        may_prune=False,
    )
    constraint_set = _compiled_set(
        constraints=(constraint,),
        hard_constraint_ids=(),
        advisory_constraint_ids=(),
        evaluation_constraint_ids=(constraint.constraint_id,),
    )
    verifier = _verifier(
        constraint_set=constraint_set,
        profile=_profile_for(
            constraint_set,
            mode="evaluation_oracle",
            authority_tier="evaluation-only",
            required_constraint_ids=(constraint.constraint_id,),
        ),
    )
    assert verifier.verify('root = Button(":slot_0")').status is VerifyStatus.UNAVAILABLE
    assert any(
        atom.source_kind == "constraint"
        for atom in verifier.terminal_evidence[-1].mandatory_unknown_atoms
    )
    assert "ordinary-customer-literal" not in json.dumps(
        verifier.terminal_evidence[-1].to_dict()
    )


def test_structural_timeout_is_unavailable() -> None:
    verifier = _verifier()
    verifier._structural.verify = lambda _program: (_ for _ in ()).throw(TimeoutError())
    assert verifier.verify('root = Button(":slot_0")').status is VerifyStatus.UNAVAILABLE


def test_constraint_evaluation_timeout_is_unavailable(monkeypatch) -> None:
    verifier = _verifier()

    def timeout(**_kwargs):
        raise TimeoutError

    monkeypatch.setattr(
        "slm_training.dsl.solver.openui_support.evaluate_goal_constraints", timeout
    )
    assert verifier.verify('root = Button(":slot_0")').status is VerifyStatus.UNAVAILABLE


@pytest.mark.parametrize("gate_id", ("G1", "G2", "G8"))
@pytest.mark.parametrize("error", (TimeoutError("deadline"), RuntimeError("unavailable")))
def test_required_gate_backend_unavailability_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    gate_id: str,
    error: Exception,
) -> None:
    from slm_training.data.verify import stack
    from slm_training.dsl.solver import openui_support

    def raise_error(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(openui_support, "evaluate_gate", stack.evaluate_gate)
    if gate_id == "G1":
        backend = type("UnavailableBackend", (), {"validate": raise_error})()
        monkeypatch.setattr(stack, "_grammar_backend", lambda: backend)
    else:
        monkeypatch.setattr(stack, "validate", raise_error)

    constraint_set = _compiled_set()
    profile = _profile_for(constraint_set, required_gates=(gate_id,))
    verifier = _verifier(profile=profile, constraint_set=constraint_set)
    assert verifier.verify('root = Button(":slot_0")').status is VerifyStatus.UNAVAILABLE
    evidence = verifier.terminal_evidence[-1]
    assert evidence.required_gate_results[0].status == "UNAVAILABLE"
    assert any(atom.source_kind == "gate" for atom in evidence.mandatory_unknown_atoms)


@pytest.mark.parametrize("gate_id", ("G1", "G2", "G8"))
def test_required_gate_backend_candidate_failure_remains_reject(
    monkeypatch: pytest.MonkeyPatch,
    gate_id: str,
) -> None:
    from slm_training.data.verify import stack
    from slm_training.dsl.solver import openui_support

    error: Exception = (
        ValueError("invalid grammar") if gate_id == "G1" else stack.ParseError("invalid program")
    )

    def raise_error(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(openui_support, "evaluate_gate", stack.evaluate_gate)
    if gate_id == "G1":
        backend = type("RejectingBackend", (), {"validate": raise_error})()
        monkeypatch.setattr(stack, "_grammar_backend", lambda: backend)
    else:
        monkeypatch.setattr(stack, "validate", raise_error)

    constraint_set = _compiled_set()
    profile = _profile_for(constraint_set, required_gates=(gate_id,))
    verifier = _verifier(profile=profile, constraint_set=constraint_set)
    assert verifier.verify('root = Button(":slot_0")').status is VerifyStatus.REJECT
    evidence = verifier.terminal_evidence[-1]
    assert evidence.required_gate_results[0].status == "REJECT"


def test_verify_is_deterministic() -> None:
    source = 'root = Button(":slot_0")'
    verifier = _verifier()
    first = verifier.verify(source)
    second = verifier.verify(source)
    assert first.status == second.status
    assert first.detail == second.detail
    assert verifier.last_trace.latest().compute_digest() == verifier.last_trace.latest().compute_digest()
    assert len(verifier.terminal_evidence) == 2


def test_no_cross_query_trace_contamination() -> None:
    first_verifier = _verifier()
    second_verifier = _verifier()
    first_outcome = first_verifier.verify('root = Button(":slot_0")')
    first_digest = first_verifier.last_trace.latest().evidence_digest
    second_outcome = second_verifier.verify('root = TextContent(":copy.value")')
    second_digest = second_verifier.last_trace.latest().evidence_digest
    assert first_outcome.status is VerifyStatus.ACCEPT
    assert second_outcome.status is VerifyStatus.REJECT
    assert first_digest != second_digest
    assert isinstance(first_verifier.last_trace, GoalTerminalEvidenceTrace)
    assert len(first_verifier.terminal_evidence) == 1
    assert len(second_verifier.terminal_evidence) == 1


def test_secret_shaped_content_not_in_outcome_or_evidence() -> None:
    secret = "token=hf_" + ("x" * 24)
    request = GenerationRequest(
        prompt=f"Use secret {secret}", slot_contract=(":slot_0",)
    )
    verifier = _verifier(request=request)
    outcome = verifier.verify('root = Button(":slot_0")')
    assert "hf_" not in outcome.detail
    payload = verifier.last_trace.latest().to_dict()
    serialized = json.dumps(payload)
    assert "hf_" not in serialized
    assert "[REDACTED]" in redact_bounded_string(secret)


def test_gate_detail_is_replaced_by_closed_reason_code(monkeypatch) -> None:
    from slm_training.data.verify.stack import GateResult, GateStatus

    constraint_set = _compiled_set(
        constraints=(),
        hard_constraint_ids=(),
        advisory_constraint_ids=(),
        evaluation_constraint_ids=(),
    )
    profile = _profile_for(
        constraint_set,
        mode="evaluation_oracle",
        authority_tier="evaluation-only",
        required_constraint_ids=(),
        required_gates=("G0",),
    )
    monkeypatch.setattr(
        "slm_training.dsl.solver.openui_support.evaluate_gate",
        lambda gate, *_args: GateResult(
            gate=gate,
            status=GateStatus.FAIL,
            detail="ordinary-user-literal-must-not-persist",
        ),
    )
    verifier = _verifier(profile=profile, constraint_set=constraint_set)
    verifier.verify('root = Button(":slot_0")')
    payload = json.dumps(verifier.terminal_evidence[-1].to_dict())
    assert "ordinary-user-literal" not in payload
    assert "gate_fail" in payload


def test_compile_and_verify_round_trip() -> None:
    request = GenerationRequest(prompt="Build a Button")
    problem = VerifiedSynthesisProblemV1(
        problem_id="p1",
        pack_identity=_pack_identity(),
        requirements=PromptSemanticRequirementsV1(facts=()),
    )
    pack = get_pack("openui")
    compiled = compile_goal_constraints(problem, request, pack)
    profile = _profile_for(
        compiled,
        required_constraint_ids=compiled.hard_constraint_ids,
        required_gates=("G0", "G1", "G2", "G3"),
    )
    verifier = OpenUIGoalVerifier(
        profile=profile,
        constraint_set=compiled,
        problem=problem,
        request=request,
        pack_identity=problem.pack_identity,
    )
    accept = verifier.verify('root = Button(":slot_0")')
    assert accept.status is VerifyStatus.ACCEPT


def test_request_output_kind_is_compared_with_candidate_kind() -> None:
    request = GenerationRequest(prompt="Return one statement", output_kind="statement")
    problem = VerifiedSynthesisProblemV1(
        problem_id="output-kind-mismatch",
        pack_identity=_pack_identity(),
    )
    compiled = compile_goal_constraints(problem, request, get_pack("openui"))
    profile = _profile_for(compiled)
    verifier = OpenUIGoalVerifier(
        profile=profile,
        constraint_set=compiled,
        problem=problem,
        request=request,
        pack_identity=problem.pack_identity,
    )
    assert verifier.verify('root = Button(":slot_0")').status is VerifyStatus.REJECT


def test_unobserved_output_category_is_unavailable() -> None:
    request = GenerationRequest(
        prompt="Build a dashboard",
        output_category="dashboard.card",
    )
    problem = VerifiedSynthesisProblemV1(
        problem_id="output-category-unobserved",
        pack_identity=_pack_identity(),
    )
    compiled = compile_goal_constraints(problem, request, get_pack("openui"))
    profile = _profile_for(compiled)
    verifier = OpenUIGoalVerifier(
        profile=profile,
        constraint_set=compiled,
        problem=problem,
        request=request,
        pack_identity=problem.pack_identity,
    )
    assert verifier.verify('root = Button(":slot_0")').status is VerifyStatus.UNAVAILABLE


@pytest.mark.skipif(
    not lang_core.bridge_available(),
    reason="lang-core bridge required for live structural integration",
)
def test_live_structural_verifier_integration_when_bridge_available() -> None:
    source = 'root = Button(":slot_0")'
    live = OpenUIWellFormedVerifier().verify(source)
    if live.status is not VerifyStatus.ACCEPT:
        pytest.skip(f"lang-core validate unavailable at runtime: {live.detail}")
    verifier = _verifier(structural_verifier=OpenUIWellFormedVerifier())
    assert verifier.verify(source).status is VerifyStatus.ACCEPT
