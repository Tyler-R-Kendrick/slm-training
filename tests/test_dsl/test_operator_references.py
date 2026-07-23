from __future__ import annotations

import hashlib
import json

from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    CompilerFact,
    EffectDeltaKind,
    EffectDeltaV1,
    IndexRef,
    NodeRef,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    ReferenceEntryV1,
    ReferenceResolutionError,
    ReferenceTableV1,
    RoleRef,
    RuntimeSymbolDescriptorV1,
    RuntimeSymbolRole,
    RegisteredOperatorV1,
    branch_fingerprint,
    branch_local_disambiguator,
    build_reference_table,
    ordered_parent_digest,
    persistent_node_fingerprint,
)
from slm_training.dsl.pack import get_pack

_STATE = "a" * 64
_ROOT = "b" * 64
_NONCE = "c" * 64


def _branch(nonce: str = _NONCE) -> str:
    return branch_fingerprint(_ROOT, nonce)


def _node_descriptor(
    component: str,
    collision_index: int,
    *,
    branch: str | None = None,
) -> ReferenceDescriptorV1:
    branch = branch or _branch()
    structural = persistent_node_fingerprint(
        {"component": component, "properties": ("children",)},
        parent_fingerprint=None,
        branch_disambiguator=branch_local_disambiguator(
            branch,
            persistent_node_fingerprint(
                {"component": component},
                parent_fingerprint=None,
                branch_disambiguator="d" * 64,
            ),
            collision_index,
        ),
    )
    return ReferenceDescriptorV1(
        ref_kind=RefKind.NODE,
        semantic_fingerprint=structural,
        value_type=f"openui.{component.lower()}",
        compiler_facts=(CompilerFact.NODE_VISIBLE, CompilerFact.NODE_MUTABLE),
    )


def _table(seed: int = 1) -> ReferenceTableV1:
    return build_reference_table(
        request_id="req-1",
        state_digest=_STATE,
        branch_digest=_branch(),
        descriptors=(
            _node_descriptor("Stack", 0),
            _node_descriptor("Card", 0),
            _node_descriptor("TextContent", 0),
        ),
        seed=seed,
    )


def test_persistent_fingerprint_uses_canonical_structure_not_binder_name() -> None:
    branch = _branch()
    disambiguator = branch_local_disambiguator(branch, "e" * 64, 0)
    first = persistent_node_fingerprint(
        {"component": "TextContent", "slot_kind": "content"},
        parent_fingerprint=None,
        branch_disambiguator=disambiguator,
    )
    renamed = persistent_node_fingerprint(
        {"slot_kind": "content", "component": "TextContent"},
        parent_fingerprint=None,
        branch_disambiguator=disambiguator,
    )
    assert first == renamed


def test_ref_id_and_candidate_permutation_preserve_resolution_and_result() -> None:
    first = _table(seed=1)
    permuted = first.permuted(seed=99)
    descriptor = first.entries[0].descriptor
    first_ref = next(
        entry.ref for entry in first.entries if entry.descriptor == descriptor
    )
    permuted_ref = next(
        entry.ref for entry in permuted.entries if entry.descriptor == descriptor
    )

    first_resolved = first.resolve(
        first_ref,
        state_digest=_STATE,
        branch_digest=_branch(),
        expected_kind=RefKind.NODE,
    )
    permuted_resolved = permuted.resolve(
        permuted_ref,
        state_digest=_STATE,
        branch_digest=_branch(),
        expected_kind=RefKind.NODE,
    )

    assert first_ref.opaque_id != permuted_ref.opaque_id
    assert [entry.ref.opaque_id for entry in first.entries] != [
        entry.ref.opaque_id for entry in permuted.entries
    ]
    assert first_resolved == permuted_resolved
    result = lambda value: json.dumps(  # noqa: E731 - compact fixture compiler
        {"selected_node": value.semantic_fingerprint}, sort_keys=True
    )
    assert result(first_resolved) == result(permuted_resolved)


def test_ref_permutation_preserves_pack_legal_application_result() -> None:
    source = 'root = TextContent(":hero.title")'
    base_state = OperatorStateV1.from_source(get_pack("openui"), source)
    table = build_reference_table(
        request_id="req-1",
        state_digest=base_state.state_digest,
        branch_digest=_branch(),
        descriptors=(_node_descriptor("TextContent", 0),),
        seed=1,
    )
    permuted = table.permuted(seed=2)

    declaration = AstOperatorV1(
        operator_id="openui.fixture_select_node",
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("node", RefKind.NODE, BindingPhase.STATE),
        ),
        preconditions=(),
        effect_signature=(EffectDeltaKind.PROPERTY,),
        locality="node",
        cost=1.0,
    )

    def execute_with(bound_table: ReferenceTableV1):
        def execute(
            state: OperatorStateV1, arguments: tuple[BoundArgumentV1, ...]
        ) -> OperatorMutationV1:
            ref = arguments[0].value
            bound_table.resolve(
                ref,
                state_digest=state.state_digest,
                branch_digest=_branch(),
                expected_kind=RefKind.NODE,
            )
            return OperatorMutationV1(
                source=state.source.replace(":hero.title", ":hero.body"),
                effect=ActionEffectV1(
                    property_deltas=(
                        EffectDeltaV1(
                            EffectDeltaKind.PROPERTY,
                            ref,
                            ":hero.title",
                            ":hero.body",
                        ),
                    ),
                    compiler_coverage=CompilerCoverage.EXACT,
                ),
            )

        return execute

    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.compiler",
        compiler_version="2026.07",
        source_artifact_digest=hashlib.sha256(source.encode()).hexdigest(),
        request_id="req-1",
    )
    results = []
    for bound_table in (table, permuted):
        library = OperatorLibraryV1(
            (RegisteredOperatorV1(declaration, execute_with(bound_table)),)
        )
        pack = replace(get_pack("openui"), operator_library=library)
        state = OperatorStateV1.from_source(pack, source)
        ref = bound_table.entries[0].ref
        result = library.apply(
            pack,
            state,
            declaration.operator_id,
            (BoundArgumentV1("node", ref),),
            provenance,
        )
        assert result.succeeded
        assert result.state is not None
        results.append(result.state.source)
    assert results[0] == results[1]


def test_allocation_is_independent_of_descriptor_input_order() -> None:
    descriptors = (
        _node_descriptor("Stack", 0),
        _node_descriptor("Card", 0),
        _node_descriptor("TextContent", 0),
    )
    first = build_reference_table(
        request_id="req-1",
        state_digest=_STATE,
        branch_digest=_branch(),
        descriptors=descriptors,
        seed=11,
    )
    reversed_input = build_reference_table(
        request_id="req-1",
        state_digest=_STATE,
        branch_digest=_branch(),
        descriptors=tuple(reversed(descriptors)),
        seed=11,
    )
    assert first == reversed_input


@pytest.mark.parametrize(
    ("change", "code"),
    (
        ({"state_digest": "f" * 64}, "ref.stale_state"),
        ({"branch_digest": branch_fingerprint(_ROOT, "1" * 64)}, "ref.cross_branch"),
    ),
)
def test_stale_and_cross_branch_refs_have_stable_codes(
    change: dict[str, str], code: str
) -> None:
    table = _table()
    ref = table.entries[0].ref
    kwargs = {
        "state_digest": _STATE,
        "branch_digest": _branch(),
        "expected_kind": RefKind.NODE,
        **change,
    }
    with pytest.raises(ReferenceResolutionError) as raised:
        table.resolve(ref, **kwargs)
    assert raised.value.code == code


def test_missing_cross_request_and_type_incompatible_refs_fail_closed() -> None:
    table = _table()
    with pytest.raises(ReferenceResolutionError) as raised:
        table.resolve(
            NodeRef("req-1", "missing"),
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=RefKind.NODE,
        )
    assert raised.value.code == "ref.missing"

    with pytest.raises(ReferenceResolutionError) as raised:
        table.resolve(
            NodeRef("other-request", "missing"),
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=RefKind.NODE,
        )
    assert raised.value.code == "ref.cross_request"

    with pytest.raises(ReferenceResolutionError) as raised:
        table.resolve(
            table.entries[0].ref,
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=RefKind.ROLE,
        )
    assert raised.value.code == "ref.type_incompatible"


def test_duplicate_refs_are_rejected_at_table_boundary() -> None:
    table = _table()
    with pytest.raises(ReferenceResolutionError) as raised:
        ReferenceTableV1(
            request_id=table.request_id,
            state_digest=table.state_digest,
            branch_digest=table.branch_digest,
            entries=(table.entries[0], table.entries[0]),
        )
    assert raised.value.code == "ref.duplicate"


def test_index_ref_is_relative_to_current_parent_order() -> None:
    parent = "2" * 64
    children = ("3" * 64, "4" * 64)
    order = ordered_parent_digest(parent, children)
    descriptor = ReferenceDescriptorV1(
        ref_kind=RefKind.INDEX,
        semantic_fingerprint="5" * 64,
        value_type="openui.child_index",
        parent_fingerprint=parent,
        parent_order_digest=order,
        position=1,
        compiler_facts=(CompilerFact.INDEX_ORDERED_PARENT,),
    )
    table = build_reference_table(
        request_id="req-1",
        state_digest=_STATE,
        branch_digest=_branch(),
        descriptors=(descriptor,),
        seed=7,
    )
    ref = table.entries[0].ref
    assert isinstance(ref, IndexRef)
    assert (
        table.resolve(
            ref,
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=RefKind.INDEX,
            current_parent_order_digest=order,
        ).position
        == 1
    )
    changed_order = ordered_parent_digest(parent, tuple(reversed(children)))
    with pytest.raises(ReferenceResolutionError) as raised:
        table.resolve(
            ref,
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=RefKind.INDEX,
            current_parent_order_digest=changed_order,
        )
    assert raised.value.code == "ref.stale_index"


def test_descriptor_table_roundtrip_contains_no_user_surface_or_address() -> None:
    descriptor = ReferenceDescriptorV1(
        ref_kind=RefKind.SYMBOL,
        semantic_fingerprint="6" * 64,
        value_type="openui.alpha_binder",
        compiler_facts=(CompilerFact.SYMBOL_IN_SCOPE,),
    )
    with pytest.raises(ValueError, match="kind"):
        ReferenceEntryV1(RoleRef("req-1", "r1"), descriptor)

    table = build_reference_table(
        request_id="req-1",
        state_digest=_STATE,
        branch_digest=_branch(),
        descriptors=(descriptor,),
        seed=3,
    )
    symbol = RuntimeSymbolDescriptorV1(
        symbol_fingerprint="7" * 64,
        ref_fingerprint=descriptor.fingerprint,
        symbol_role=RuntimeSymbolRole.ALPHA_BINDER,
        semantic_role="content_slot",
    )
    table = ReferenceTableV1(
        request_id=table.request_id,
        state_digest=table.state_digest,
        branch_digest=table.branch_digest,
        entries=table.entries,
        runtime_symbols=(symbol,),
    )
    payload = table.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    assert "surface" not in encoded
    assert "display" not in encoded
    assert "address" not in encoded
    assert ReferenceTableV1.from_dict(payload) == table
    assert ReferenceTableV1.from_dict(payload).fingerprint == table.fingerprint
    bad_descriptor = descriptor.to_dict()
    bad_descriptor["compiler_facts"] = ["user.alice"]
    with pytest.raises(ValueError):
        ReferenceDescriptorV1.from_dict(bad_descriptor)
    bad_symbol = symbol.to_dict()
    bad_symbol["symbol_role"] = "user_role"
    with pytest.raises(ValueError):
        RuntimeSymbolDescriptorV1.from_dict(bad_symbol)
