"""Cross-contract schema-version compatibility (SGS-010 / SLM-445, PGS-I02 / SLM-513).

One table-driven reader-behavior check over every serialized contract this
initiative introduced. Per-contract semantics live with their own tests; this
file certifies the *shared* compatibility policy so a new contract cannot land
with a version field nobody enforces:

* current version round-trips losslessly,
* a newer/unknown version is rejected (never coerced or reinterpreted),
* a missing version is rejected (absence is not "assume current"),
* the static registry in ``scripts.verify_ownership_map`` lists the contract.

Historical artifacts stay readable exactly because a semantics-changing
migration must mint a new version identity (and therefore a new row here)
rather than redefine an existing one in place.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from scripts.verify_ownership_map import SERIALIZED_CONTRACTS
from slm_training.data.progspec.goal_constraints import (
    CompiledGoalConstraintSetV1,
    GoalConstraintEvaluationV1,
    GoalConstraintV1,
)
from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
)
from slm_training.data.progspec.synthesis_problem import (
    PackIdentityV1,
    VerifiedSynthesisProblemV1,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.dsl.solver.goal_support import (
    GoalSupportResultV1,
    GoalTerminalEvidenceV1,
    GoalVerifierProfileV1,
    compute_pack_identity_digest,
)
from slm_training.dsl.solver.goal_support_domain_adequacy import (
    GoalActionEvidenceV1,
    GoalDomainActionPartitionsV1,
    GoalDomainAdequacyReportV1,
    GoalDomainAdequacyStatsV1,
)
from slm_training.dsl.solver.goal_support_obstruction import (
    DomainObstructionCoreV1,
    ObstructionCoreStatsV1,
    TerminalFailureSetV1,
)
from slm_training.evals.semantic_failure import (
    VerifierWitnessV1,
    build_verifier_witness,
    trace_semantic_failure,
)
from slm_training.harnesses.experiments.compute_state_controller import (
    ComputeStateFeaturesV1,
    features_from_decode_record,
)
from slm_training.models.decode_stats import (
    DecodeStats,
    DecodeStatsRecordV1,
    build_decode_stats_record,
)

Case = tuple[str, str, Callable[[], dict[str, Any]], Callable[[dict[str, Any]], Any]]


def _requirements_payload() -> dict[str, Any]:
    return PromptSemanticRequirementsV1(prompt_context_hash="abc").to_dict()


def _synthesis_payload() -> dict[str, Any]:
    return VerifiedSynthesisProblemV1(
        problem_id="p0", pack_identity=PackIdentityV1(pack_id="openui")
    ).to_dict()


def _witness_payload() -> dict[str, Any]:
    record = ExampleRecord(
        id="x",
        prompt="Build a Button. Placeholders: :cta.label",
        openui='root = Button(":cta.label")',
        split="smoke",
        source="fixture",
    )
    return build_verifier_witness(
        trace_semantic_failure("root = Stack([])", record)
    ).to_dict()


def _decode_record_payload() -> dict[str, Any]:
    return build_decode_stats_record(
        DecodeStats(), measurement_stage="steady_state"
    ).to_dict()


def _compute_state_features_payload() -> dict[str, Any]:
    record = build_decode_stats_record(DecodeStats(), measurement_stage="steady_state")
    features = features_from_decode_record(record)
    return ComputeStateFeaturesV1.from_dict(features.to_dict()).to_dict()


def _goal_constraint_payload() -> dict[str, Any]:
    constraint = GoalConstraintV1(
        constraint_id="c_slot_button",
        kind="slot_present",
        parameters={"role_id": "primary_action"},
        source_kind="pack_contract",
        source_id="pack/openui",
        source_digest="abc123",
        authority_tier="compiler-hard",
        completeness="EXACT",
        may_prune=True,
    )
    return GoalConstraintV1.from_dict(constraint.to_dict()).to_dict()


def _compiled_goal_constraint_set_payload() -> dict[str, Any]:
    constraint = GoalConstraintV1(
        constraint_id="c_slot_button",
        kind="slot_present",
        parameters={"role_id": "primary_action"},
        source_kind="pack_contract",
        source_id="pack/openui",
        source_digest="abc123",
        authority_tier="compiler-hard",
        completeness="EXACT",
        may_prune=True,
    )
    payload = CompiledGoalConstraintSetV1(
        problem_digest="a" * 64,
        request_digest="b" * 64,
        pack_identity_digest="c" * 64,
        compiler_version="compiler/v1",
        constraints=(constraint,),
        hard_constraint_ids=(constraint.constraint_id,),
        advisory_constraint_ids=(),
        evaluation_constraint_ids=(),
    ).to_dict(include_digest=True)
    return CompiledGoalConstraintSetV1.from_dict(payload).to_dict(include_digest=True)


def _goal_constraint_evaluation_payload() -> dict[str, Any]:
    evaluation = GoalConstraintEvaluationV1(
        constraint_id="c_slot_button",
        status="UNKNOWN",
        detail="fixture",
    )
    return GoalConstraintEvaluationV1.from_dict(evaluation.to_dict()).to_dict()


def _goal_verifier_profile_payload() -> dict[str, Any]:
    pack = PackIdentityV1(
        pack_id="openui",
        contract_version=2,
        grammar_sha256="a" * 64,
        tokenizer_id="tok/v1",
        canonicalizer_id="canon/v1",
        artifact_schema="openui/v1",
        artifact_sha256="b" * 64,
    )
    constraint = GoalConstraintV1(
        constraint_id="c_slot_button",
        kind="slot_present",
        parameters={"role_id": "primary_action"},
        source_kind="pack_contract",
        source_id="pack/openui",
        source_digest="abc123",
        authority_tier="compiler-hard",
        completeness="EXACT",
        may_prune=True,
    )
    constraint_set = CompiledGoalConstraintSetV1.from_dict(
        CompiledGoalConstraintSetV1(
            problem_digest="a" * 64,
            request_digest="b" * 64,
            pack_identity_digest=compute_pack_identity_digest(pack),
            compiler_version="compiler/v1",
            constraints=(constraint,),
            hard_constraint_ids=(constraint.constraint_id,),
            advisory_constraint_ids=(),
            evaluation_constraint_ids=(),
        ).to_dict(include_digest=True)
    )
    profile = GoalVerifierProfileV1(
        profile_id="profile/prod",
        mode="production_exact",
        problem_digest=constraint_set.problem_digest,
        constraint_set_digest=constraint_set.digest,
        pack_identity=pack,
        pack_identity_digest=compute_pack_identity_digest(pack),
        required_constraint_ids=(constraint.constraint_id,),
        required_gates=("G0",),
        required_evaluators=(),
        metric_identities=(),
        authority_tier="compiler-hard",
    )
    payload = profile.to_dict(include_digest=True)
    return GoalVerifierProfileV1.from_dict(payload).to_dict(include_digest=True)


def _goal_terminal_evidence_payload() -> dict[str, Any]:
    profile = GoalVerifierProfileV1.from_dict(_goal_verifier_profile_payload())
    evidence = GoalTerminalEvidenceV1(
        profile_digest=profile.digest,
        program_digest="1" * 64,
        canonical_program_digest="2" * 64,
        program_spec_digest="3" * 64,
        semantic_plan_digest="4" * 64,
        constraint_evaluations=(
            GoalConstraintEvaluationV1(constraint_id="c_slot_button", status="PASS"),
        ),
        required_gate_results=(),
        required_evaluator_results=(),
        mandatory_failure_atoms=(),
        mandatory_unknown_atoms=(),
        structural_status="ACCEPT",
        overall_status="ACCEPT",
    )
    payload = evidence.to_dict(include_digest=True)
    return GoalTerminalEvidenceV1.from_dict(payload).to_dict(include_digest=True)


def _goal_support_result_payload() -> dict[str, Any]:
    sidecar = GoalSupportResultV1(
        base_certificate_digest="a" * 64,
        base_verdict="supported",
        profile_digest="b" * 64,
        authority_tier="compiler-hard",
    )
    payload = sidecar.to_dict(include_digest=True)
    return GoalSupportResultV1.from_dict(payload).to_dict(include_digest=True)


def _domain_obstruction_core_payload() -> dict[str, Any]:
    core = DomainObstructionCoreV1(
        mode="minimum_cardinality_exact",
        core_atoms=("atom1",),
        terminal_failure_sets=(
            TerminalFailureSetV1(
                terminal_evidence_digest="a" * 64,
                exact_mandatory_failure_atom_ids=("atom1",),
            ),
        ),
        profile_digest="b" * 64,
        state_fingerprint="state",
        bounds_digest="c" * 64,
        base_certificate_digest="d" * 64,
        terminal_set_digest="e" * 64,
        atom_vocabulary_digest="f" * 64,
        stats=ObstructionCoreStatsV1(atom_universe_size=1, terminal_count=1),
    )
    payload = core.to_dict(include_digest=True)
    return DomainObstructionCoreV1.from_dict(payload).to_dict(include_digest=True)


def _goal_action_evidence_payload() -> dict[str, Any]:
    action_id = "a" * 64
    evidence = GoalActionEvidenceV1(
        state_fingerprint="state",
        legal_set_fingerprint="b" * 64,
        profile_digest="c" * 64,
        hole_id={"namespace": "word", "path": ["0", "ROOT"], "kind": "next"},
        action_id=action_id,
        partition="supported",
        base_certificate_digest="d" * 64,
        goal_sidecar_digest="e" * 64,
        authority_tier="compiler-hard",
        replay_ok=True,
    )
    payload = evidence.to_dict(include_digest=True)
    return GoalActionEvidenceV1.from_dict(payload).to_dict(include_digest=True)


def _goal_domain_adequacy_report_payload() -> dict[str, Any]:
    action_id = "a" * 64
    report = GoalDomainAdequacyReportV1(
        classification="coverage_unknown",
        state_fingerprint="state",
        legal_set_fingerprint="b" * 64,
        selected_action_id=action_id,
        profile_digest="c" * 64,
        problem_id="word:ROOT",
        pack_id="fixture-word",
        constraint_version="v1",
        bounds_digest="d" * 64,
        hole_id={"namespace": "word", "path": ["0", "ROOT"], "kind": "next"},
        partitions=GoalDomainActionPartitionsV1(unobserved=(action_id,)),
        evidence_digests=(),
        stats=GoalDomainAdequacyStatsV1(
            action_count=1,
            queried_action_count=0,
            terminal_count=0,
            query_count=0,
            verifier_calls=0,
            expanded_nodes=0,
            backtracks=0,
            stop_reasons=("cap_exhausted",),
        ),
        exact_action_cap=8,
        cap_applied=False,
    )
    payload = report.to_dict(include_digest=True)
    return GoalDomainAdequacyReportV1.from_dict(payload).to_dict(include_digest=True)


CASES: tuple[Case, ...] = (
    (
        "PromptSemanticRequirementsV1",
        "requirements_version",
        _requirements_payload,
        PromptSemanticRequirementsV1.from_dict,
    ),
    (
        "VerifiedSynthesisProblemV1",
        "schema_version",
        _synthesis_payload,
        VerifiedSynthesisProblemV1.from_dict,
    ),
    (
        "VerifierWitnessV1",
        "schema_version",
        _witness_payload,
        VerifierWitnessV1.from_dict,
    ),
    (
        "DecodeStatsRecordV1",
        "schema_version",
        _decode_record_payload,
        DecodeStatsRecordV1.from_dict,
    ),
    (
        "ComputeStateFeaturesV1",
        "schema_version",
        _compute_state_features_payload,
        ComputeStateFeaturesV1.from_dict,
    ),
    (
        "GoalConstraintV1",
        "schema_version",
        _goal_constraint_payload,
        GoalConstraintV1.from_dict,
    ),
    (
        "CompiledGoalConstraintSetV1",
        "schema_version",
        _compiled_goal_constraint_set_payload,
        CompiledGoalConstraintSetV1.from_dict,
    ),
    (
        "GoalConstraintEvaluationV1",
        "schema_version",
        _goal_constraint_evaluation_payload,
        GoalConstraintEvaluationV1.from_dict,
    ),
    (
        "GoalVerifierProfileV1",
        "schema_version",
        _goal_verifier_profile_payload,
        GoalVerifierProfileV1.from_dict,
    ),
    (
        "GoalTerminalEvidenceV1",
        "schema_version",
        _goal_terminal_evidence_payload,
        GoalTerminalEvidenceV1.from_dict,
    ),
    (
        "GoalSupportResultV1",
        "schema_version",
        _goal_support_result_payload,
        GoalSupportResultV1.from_dict,
    ),
    (
        "DomainObstructionCoreV1",
        "schema_version",
        _domain_obstruction_core_payload,
        DomainObstructionCoreV1.from_dict,
    ),
    (
        "GoalActionEvidenceV1",
        "schema_version",
        _goal_action_evidence_payload,
        GoalActionEvidenceV1.from_dict,
    ),
    (
        "GoalDomainAdequacyReportV1",
        "schema_version",
        _goal_domain_adequacy_report_payload,
        GoalDomainAdequacyReportV1.from_dict,
    ),
)

_IDS = [case[0] for case in CASES]


@pytest.mark.parametrize("name,field,build,read", CASES, ids=_IDS)
def test_current_version_round_trips(
    name: str, field: str, build: Callable[[], dict[str, Any]], read: Callable[..., Any]
) -> None:
    payload = build()
    assert payload[field], f"{name} serialized without a {field}"
    restored = read(payload)
    assert restored.to_dict() == payload or restored.to_dict(include_digest=True) == payload


@pytest.mark.parametrize("name,field,build,read", CASES, ids=_IDS)
def test_future_version_is_rejected_not_coerced(
    name: str, field: str, build: Callable[[], dict[str, Any]], read: Callable[..., Any]
) -> None:
    payload = build()
    payload[field] = f"{payload[field]}-from-the-future"
    with pytest.raises((ValueError, TypeError)):
        read(payload)


@pytest.mark.parametrize("name,field,build,read", CASES, ids=_IDS)
def test_missing_version_is_rejected(
    name: str, field: str, build: Callable[[], dict[str, Any]], read: Callable[..., Any]
) -> None:
    payload = build()
    del payload[field]
    with pytest.raises((ValueError, TypeError, KeyError)):
        read(payload)


def test_every_case_is_registered_for_static_certification() -> None:
    # The CI-side static check and this runtime check must cover the same set;
    # a contract in one but not the other is exactly the gap SGS-010 closes.
    assert {case[0] for case in CASES} == {c.contract_id for c in SERIALIZED_CONTRACTS}
