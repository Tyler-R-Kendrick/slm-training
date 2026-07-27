"""DSH3-11 (SLM-379): OperatorFrameV1 derivation, provenance, conflicts."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from slm_training.dsl.operator_frame import (
    OperatorFactV1,
    UnsupportedOperatorFrameError,
    derive_operator_frame,
    frame_conflicts,
)
from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    ReferenceTableV1,
    RegisteredOperatorV1,
    RoleRef,
    SymbolRef,
    build_reference_table,
)
from slm_training.dsl.pack import get_pack

SOURCE = 'root = TextContent(":hero.title")'
OPERATOR_ID = "openui.fixture_frame"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _descriptors() -> tuple[ReferenceDescriptorV1, ...]:
    return (
        ReferenceDescriptorV1(
            ref_kind=RefKind.VALUE,
            semantic_fingerprint=_sha("semantic-value-0"),
            value_type="openui.string",
        ),
        ReferenceDescriptorV1(
            ref_kind=RefKind.VALUE,
            semantic_fingerprint=_sha("semantic-value-1"),
            value_type="openui.string",
        ),
        ReferenceDescriptorV1(
            ref_kind=RefKind.ROLE,
            semantic_fingerprint=_sha("semantic-role-0"),
            value_type="openui.role",
        ),
        ReferenceDescriptorV1(
            ref_kind=RefKind.SYMBOL,
            semantic_fingerprint=_sha("semantic-symbol-0"),
            value_type="openui.binder",
        ),
    )


def _fixture(*, coverage: CompilerCoverage = CompilerCoverage.EXACT):
    base_pack = get_pack("openui")
    state = OperatorStateV1.from_source(base_pack, SOURCE)
    table = build_reference_table(
        request_id="request-1",
        state_digest=state.state_digest,
        branch_digest=_sha("branch"),
        descriptors=_descriptors(),
        seed=7,
    )
    refs = {entry.ref.KIND: entry.ref for entry in table.entries}
    value_refs = sorted(
        (entry.ref for entry in table.entries if entry.ref.KIND is RefKind.VALUE),
        key=lambda ref: ref.opaque_id,
    )

    def execute(operator_state, arguments):
        return OperatorMutationV1(
            source=operator_state.source.replace(":hero.title", ":hero.body"),
            effect=ActionEffectV1(
                property_deltas=(
                    EffectDeltaV1(
                        kind=EffectDeltaKind.PROPERTY,
                        target=arguments[0].value,
                        before="draft",
                        after="published",
                    ),
                ),
                produced_roles=(RoleRef("request-1", refs[RefKind.ROLE].opaque_id),),
                consumed_binders=(
                    SymbolRef("request-1", refs[RefKind.SYMBOL].opaque_id),
                ),
                compiler_coverage=coverage,
            ),
        )

    declaration = AstOperatorV1(
        operator_id=OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(EffectDeltaKind.PROPERTY,),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(base_pack, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(SOURCE),
        request_id="request-1",
    )
    return pack, library, state, table, provenance, declaration, value_refs


def _apply(fixture, ref):
    pack, library, state, table, provenance, declaration, _ = fixture
    result = library.apply(
        pack,
        state,
        OPERATOR_ID,
        (BoundArgumentV1("value", ref),),
        provenance,
    )
    assert result.succeeded and result.state is not None
    return result


def test_frame_derivation_is_deterministic_with_provenance() -> None:
    fixture = _fixture()
    _, _, state, table, _, declaration, value_refs = fixture
    application = _apply(fixture, value_refs[0]).application
    first = derive_operator_frame(
        declaration, application.arguments, state, application.effect, table
    )
    second = derive_operator_frame(
        declaration, application.arguments, state, application.effect, table
    )
    assert first.fingerprint == second.fingerprint
    assert first.equivalent_to(second)
    assert first.to_dict() == second.to_dict()
    for fact in (
        *first.required_edit_facts,
        *first.forbidden_edit_facts,
        *first.state_reference_facts,
    ):
        assert fact.provenance.source in {"effect", "argument", "reference_table"}
        assert fact.provenance.descriptor_fingerprint
    delta_facts = [
        f for f in first.required_edit_facts if f.fact_kind == "effect_delta"
    ]
    assert delta_facts[0].provenance.delta_kind == "property"
    assert delta_facts[0].value == {"before": "draft", "after": "published"}
    arg_facts = [
        f for f in first.state_reference_facts if f.provenance.slot_id is not None
    ]
    assert [f.provenance.slot_id for f in arg_facts] == ["value"]


def test_required_and_forbidden_facts_come_from_action_effect() -> None:
    fixture = _fixture()
    _, _, state, table, _, declaration, value_refs = fixture
    application = _apply(fixture, value_refs[0]).application
    frame = derive_operator_frame(
        declaration, application.arguments, state, application.effect, table
    )
    kinds = {fact.fact_kind for fact in frame.required_edit_facts}
    assert kinds == {"effect_delta", "produced_role", "consumed_binder"}
    touched = {
        (fact.ref.KIND, fact.ref.opaque_id)
        for fact in (*frame.required_edit_facts, *frame.state_reference_facts)
    }
    # EXACT coverage: every untouched table entry is forbidden to change.
    assert frame.compiler_coverage == "exact"
    forbidden = {
        (fact.ref.KIND, fact.ref.opaque_id) for fact in frame.forbidden_edit_facts
    }
    all_entries = {(entry.ref.KIND, entry.ref.opaque_id) for entry in table.entries}
    assert forbidden == all_entries - touched
    assert frame_conflicts(frame) == ()


def test_bounded_coverage_yields_no_forbidden_overclaim() -> None:
    fixture = _fixture(coverage=CompilerCoverage.BOUNDED)
    _, _, state, table, _, declaration, value_refs = fixture
    application = _apply(fixture, value_refs[0]).application
    frame = derive_operator_frame(
        declaration, application.arguments, state, application.effect, table
    )
    assert frame.compiler_coverage == "bounded"
    assert frame.forbidden_edit_facts == ()
    assert frame.required_edit_facts  # required facts still derive


def test_frame_conflicts_flags_required_and_forbidden_overlap() -> None:
    fixture = _fixture()
    _, _, state, table, _, declaration, value_refs = fixture
    application = _apply(fixture, value_refs[0]).application
    frame = derive_operator_frame(
        declaration, application.arguments, state, application.effect, table
    )
    delta_fact = next(
        f for f in frame.required_edit_facts if f.fact_kind == "effect_delta"
    )
    mutated = replace(
        frame,
        forbidden_edit_facts=(
            *frame.forbidden_edit_facts,
            OperatorFactV1(
                category="forbidden",
                fact_kind="untouched_reference",
                ref=delta_fact.ref,
                value=None,
                provenance=delta_fact.provenance,
            ),
        ),
    )
    conflicts = frame_conflicts(mutated)
    assert len(conflicts) == 1
    assert "both required to change and forbidden" in conflicts[0].reason


def test_stale_reference_table_fails_closed() -> None:
    fixture = _fixture()
    _, _, state, table, _, declaration, value_refs = fixture
    application = _apply(fixture, value_refs[0]).application
    stale = ReferenceTableV1(
        request_id=table.request_id,
        state_digest=_sha("other-state"),
        branch_digest=table.branch_digest,
        entries=table.entries,
    )
    with pytest.raises(UnsupportedOperatorFrameError, match="stale"):
        derive_operator_frame(
            declaration, application.arguments, state, application.effect, stale
        )


def test_undescribed_reference_fails_closed() -> None:
    fixture = _fixture()
    _, _, state, table, _, declaration, value_refs = fixture
    application = _apply(fixture, value_refs[0]).application
    effect = application.effect
    assert effect is not None
    stranger = EffectDeltaV1(
        kind=EffectDeltaKind.PROPERTY,
        target=RoleRef("request-1", "zz-not-in-table"),
        before="a",
        after="b",
    )
    mutated = replace(effect, property_deltas=(*effect.property_deltas, stranger))
    with pytest.raises(UnsupportedOperatorFrameError, match="absent from the typed"):
        derive_operator_frame(declaration, application.arguments, state, mutated, table)
