from __future__ import annotations

from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProofV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    NodeRef,
    OperatorApplicationV1,
    OperatorArgumentSlotV1,
    OperatorRejectionV1,
    PreconditionV1,
    RefKind,
    RoleRef,
    SymbolRef,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


def _operator(**changes: object) -> AstOperatorV1:
    value = AstOperatorV1(
        operator_id="openui.add_child",
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("parent", RefKind.NODE, BindingPhase.STATE),
            OperatorArgumentSlotV1("role", RefKind.ROLE, BindingPhase.APPLICATION),
        ),
        preconditions=(
            PreconditionV1("openui.parent_accepts_role", ("parent", "role")),
        ),
        effect_signature=(
            EffectDeltaKind.TOPOLOGY,
            EffectDeltaKind.CARDINALITY,
        ),
        inverse_operator_id="openui.remove_node",
        commutes_with=("openui.set_property", "openui.bind_symbol"),
        idempotent=False,
        locality="subtree",
        cost=1.0,
    )
    return replace(value, **changes)


def _effect() -> ActionEffectV1:
    request_id = "req-1"
    role = RoleRef(request_id, "r1")
    node = NodeRef(request_id, "n1")
    return ActionEffectV1(
        produced_roles=(role,),
        produced_binders=(SymbolRef(request_id, "s1"),),
        cardinality_deltas=(EffectDeltaV1(EffectDeltaKind.CARDINALITY, role, 0, 1),),
        topology_deltas=(
            EffectDeltaV1(
                EffectDeltaKind.TOPOLOGY,
                node,
                {"children": []},
                {"children": ["opaque-child"]},
            ),
        ),
        compiler_coverage=CompilerCoverage.EXACT,
        estimated_completion_cost=2.0,
    )


def _provenance() -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.compiler",
        compiler_version="2026.07",
        source_artifact_digest=_A,
        request_id="req-1",
    )


def test_equivalent_declarations_fingerprint_identically() -> None:
    first = _operator()
    second = _operator(
        commutes_with=("openui.bind_symbol", "openui.set_property"),
        preconditions=tuple(reversed(first.preconditions)),
        effect_signature=tuple(reversed(first.effect_signature)),
    )
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_declaration_fingerprint_ignores_no_model_or_display_surface() -> None:
    fields = _operator().__dataclass_fields__
    assert "embedding" not in fields
    assert "token_id" not in fields
    assert "display_name" not in fields
    assert _operator(cost=2.0).fingerprint != _operator().fingerprint


def test_every_argument_declares_resolvable_type_and_phase() -> None:
    operator = _operator()
    assert all(slot.ref_kind for slot in operator.argument_slots)
    assert all(slot.binding_phase for slot in operator.argument_slots)
    with pytest.raises(ValueError, match="unknown slots"):
        _operator(preconditions=(PreconditionV1("openui.exists", ("missing",)),))


def test_opaque_refs_are_request_local_and_have_no_display_field() -> None:
    ref = NodeRef("req-1", "n1")
    assert ref.to_dict() == {
        "kind": "node",
        "request_id": "req-1",
        "opaque_id": "n1",
    }
    with pytest.raises(ValueError, match="opaque"):
        NodeRef("req-1", "user visible name")


def test_action_effect_requires_typed_delta_buckets() -> None:
    effect = _effect()
    assert effect.compiler_coverage is CompilerCoverage.EXACT
    assert effect.fingerprint == _effect().fingerprint
    with pytest.raises(ValueError, match="mismatched kind"):
        ActionEffectV1(
            scope_deltas=effect.topology_deltas,
        )
    with pytest.raises(ValueError, match="one request"):
        ActionEffectV1(
            consumed_roles=(RoleRef("req-1", "r1"),),
            produced_roles=(RoleRef("req-2", "r2"),),
        )


def test_successful_application_records_deterministic_identity_and_provenance() -> None:
    operator = _operator()
    effect = _effect()
    proof = ApplicationProofV1(
        proof_kind="compiler.replay",
        checks=("ast.valid", "effect.matches"),
        compiler_result_digest=_B,
        effect_fingerprint=effect.fingerprint,
    )
    application = OperatorApplicationV1(
        operator_fingerprint=operator.fingerprint,
        arguments=(
            BoundArgumentV1("parent", NodeRef("req-1", "n1")),
            BoundArgumentV1("role", RoleRef("req-1", "r1")),
        ),
        before_state_digest=_A,
        before_ast_digest=_B,
        after_state_digest=_C,
        after_ast_digest=_D,
        effect=effect,
        proof=proof,
        provenance=_provenance(),
    )
    reordered = replace(application, arguments=tuple(reversed(application.arguments)))
    assert application.succeeded
    assert application.application_id == reordered.application_id
    assert application.to_dict()["provenance"]["compiler_id"] == "openui.compiler"


def test_proof_must_bind_the_recorded_effect() -> None:
    effect = _effect()
    with pytest.raises(ValueError, match="does not match"):
        OperatorApplicationV1(
            operator_fingerprint=_operator().fingerprint,
            arguments=(),
            before_state_digest=_A,
            before_ast_digest=_B,
            after_state_digest=_C,
            after_ast_digest=_D,
            effect=effect,
            proof=ApplicationProofV1(
                proof_kind="compiler.replay",
                checks=("ast.valid",),
                compiler_result_digest=_A,
                effect_fingerprint=_B,
            ),
            provenance=_provenance(),
        )


def test_rejection_is_deterministic_and_cannot_claim_after_state() -> None:
    rejection = OperatorRejectionV1(
        code="precondition.failed",
        failed_precondition="openui.parent_accepts_role",
        compiler_result_digest=_C,
    )
    application = OperatorApplicationV1(
        operator_fingerprint=_operator().fingerprint,
        arguments=(),
        before_state_digest=_A,
        before_ast_digest=_B,
        rejection=rejection,
        provenance=_provenance(),
    )
    assert not application.succeeded
    assert application.application_id == replace(application).application_id
    with pytest.raises(ValueError, match="cannot claim"):
        replace(application, after_state_digest=_D)


def test_application_rejects_cross_request_refs() -> None:
    with pytest.raises(ValueError, match="application request"):
        OperatorApplicationV1(
            operator_fingerprint=_operator().fingerprint,
            arguments=(BoundArgumentV1("parent", NodeRef("other-request", "n1")),),
            before_state_digest=_A,
            before_ast_digest=_B,
            rejection=OperatorRejectionV1(
                code="precondition.failed",
                failed_precondition=None,
                compiler_result_digest=_C,
            ),
            provenance=_provenance(),
        )
