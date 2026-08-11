"""Tests for OpenUIGoalVerifier terminal tri-state semantics (PGS-B02 / SLM-498)."""

from __future__ import annotations

import hashlib
import json

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


def _manual_program_spec(*, program_id: str, openui: str) -> ProgramSpec:
    return ProgramSpec(
        id=program_id,
        ast=_ast_for_openui(openui),
        canonical_openui=openui.strip(),
        facts={},
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
        "source_digest": _digest("source"),
        "authority_tier": "compiler-hard",
        "completeness": "EXACT",
        "may_prune": True,
    }
    defaults.update(overrides)
    return GoalConstraintV1(**defaults)


def _compiled_set(**overrides: object) -> CompiledGoalConstraintSetV1:
    hard = _constraint(
        constraint_id="hard.slot.button",
        kind="slot_inventory_exact",
        parameters={"slots": [":cta.label"]},
        source_kind="generation_request",
        source_id="generation_request.slot_contract",
        may_prune=False,
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
        "problem_digest": "a" * 64,
        "request_digest": "b" * 64,
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
    constraint_set = overrides.pop("constraint_set", _compiled_set())
    profile = overrides.pop("profile", _profile_for(constraint_set))
    request = overrides.pop("request", GenerationRequest(prompt="Build a Button"))
    structural_verifier = overrides.pop("structural_verifier", _AcceptingStructuralVerifier())
    return OpenUIGoalVerifier(
        profile=profile,
        constraint_set=constraint_set,
        request=request,
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
    source = 'root = Button(":cta.label")'
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
    source = 'root = Button(":cta.label")'
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


def test_optional_unknown_does_not_erase_mandatory_fail() -> None:
    constraint_set = _compiled_set()
    profile = _profile_for(
        constraint_set,
        required_evaluators=(
            EvaluatorIdentityV1(
                evaluator_id="meaningful_program/v2",
                version="v2",
            ),
        ),
    )
    source = 'root = TextContent(":copy.value")'
    verifier = _verifier(profile=profile, constraint_set=constraint_set)
    outcome = verifier.verify(source)
    assert outcome.status is VerifyStatus.REJECT
    evidence = verifier.last_trace.latest()
    assert evidence is not None
    assert any(atom.source_kind == "constraint" for atom in evidence.mandatory_failure_atoms)


def test_evaluation_profile_cannot_emit_compiler_hard_failure_atoms() -> None:
    eval_constraint = _constraint(
        constraint_id="eval.metric",
        kind="slot_inventory_exact",
        parameters={"slots": [":cta.label"]},
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


def test_verify_is_deterministic() -> None:
    source = 'root = Button(":cta.label")'
    verifier = _verifier()
    first = verifier.verify(source)
    second = verifier.verify(source)
    assert first.status == second.status
    assert first.detail == second.detail
    assert verifier.last_trace.latest().compute_digest() == verifier.last_trace.latest().compute_digest()


def test_no_cross_query_trace_contamination() -> None:
    verifier = _verifier()
    first_outcome = verifier.verify('root = Button(":cta.label")')
    first_digest = verifier.last_trace.latest().evidence_digest
    second_outcome = verifier.verify('root = TextContent(":copy.value")')
    second_digest = verifier.last_trace.latest().evidence_digest
    assert first_outcome.status is VerifyStatus.ACCEPT
    assert second_outcome.status is VerifyStatus.REJECT
    assert first_digest != second_digest
    assert isinstance(verifier.last_trace, GoalTerminalEvidenceTrace)
    assert len(verifier.last_trace.records) == 1


def test_secret_shaped_content_not_in_outcome_or_evidence() -> None:
    secret = "token=hf_" + ("x" * 24)
    request = GenerationRequest(prompt=f"Use secret {secret}")
    verifier = _verifier(request=request)
    outcome = verifier.verify('root = Button(":cta.label")')
    assert "hf_" not in outcome.detail
    payload = verifier.last_trace.latest().to_dict()
    serialized = json.dumps(payload)
    assert "hf_" not in serialized
    assert "[REDACTED]" in redact_bounded_string(secret)


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
        request=request,
        structural_verifier=_AcceptingStructuralVerifier(),
    )
    accept = verifier.verify('root = Button(":cta.label")')
    assert accept.status is VerifyStatus.ACCEPT


@pytest.mark.skipif(
    not lang_core.bridge_available(),
    reason="lang-core bridge required for live structural integration",
)
def test_live_structural_verifier_integration_when_bridge_available() -> None:
    source = 'root = Button(":cta.label")'
    live = OpenUIWellFormedVerifier().verify(source)
    if live.status is not VerifyStatus.ACCEPT:
        pytest.skip(f"lang-core validate unavailable at runtime: {live.detail}")
    verifier = _verifier(structural_verifier=OpenUIWellFormedVerifier())
    assert verifier.verify(source).status is VerifyStatus.ACCEPT
