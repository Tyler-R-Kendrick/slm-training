from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Callable

from slm_training.dsl.operators import (
    ApplicationProvenanceV1,
    BoundArgumentV1,
    OperatorStateV1,
    RefKind,
    SelectorFact,
    SelectorKind,
    branch_fingerprint,
    build_openui_local_operator_context,
    build_selector_descriptor,
)
from slm_training.dsl.operators.bulk import (
    MAP_SET_PROPERTY,
    attach_bulk_selector,
    build_openui_bulk_operator_library,
)
from slm_training.dsl.operators.contracts import OperatorRef
from slm_training.dsl.operators.local import LiteralValueV1, NodeLocationV1, RoleLocationV1
from slm_training.dsl.pack import get_pack
from slm_training.dsl.production_codec import parse_statement_bindings

# Three sibling ``Stack`` nodes, each wrapping one ``TextContent`` leaf. Stack's
# "direction" property (enum "row"/"column") is a genuine structural role that
# survives D2 canonicalization (unlike TextContent's "size", which the D2
# codec treats as a pure style literal and strips on every round trip — see
# ``slm_training.data.structure.strip_style_literals``), so it is the right
# scalar property to exercise a real, persisting bulk rewrite.
SOURCE = (
    'root = Card([Stack([TextContent(":item.zero")]), '
    'Stack([TextContent(":item.one")]), Stack([TextContent(":item.two")])])'
)
ROLE_NAME = "direction"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _provenance(source: str, request_id: str = "request-1") -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.local_operator_compiler",
        compiler_version="v1",
        source_artifact_digest=_sha(source),
        request_id=request_id,
    )


def _fixture(
    source: str = SOURCE,
    *,
    values: tuple[object, ...] = ("row", "column"),
    request_id: str = "request-1",
    seed: int = 7,
):
    base_pack = get_pack("openui")
    state = OperatorStateV1.from_source(base_pack, source)
    branch = branch_fingerprint(state.state_digest, "a" * 64)
    context = build_openui_local_operator_context(
        base_pack,
        state,
        request_id=request_id,
        branch_digest=branch,
        seed=seed,
        templates=(),
        values=values,
    )
    return base_pack, state, context


def _ref(context, kind: RefKind, matches: Callable[[object], bool]) -> OperatorRef:
    for ref in context.references(kind):
        if matches(context.payload(ref)):
            return ref
    raise AssertionError(f"missing {kind.value} reference")


def _node_at(context, path: tuple[str | int, ...]) -> OperatorRef:
    return _ref(
        context,
        RefKind.NODE,
        lambda payload: isinstance(payload, NodeLocationV1) and payload.path == path,
    )


def _descriptor(context, ref: OperatorRef):
    return next(
        entry.descriptor
        for entry in context.reference_table.entries
        if entry.ref == ref
    )


def _role(context, node_ref: OperatorRef, name: str) -> OperatorRef:
    owner = _descriptor(context, node_ref).semantic_fingerprint
    return _ref(
        context,
        RefKind.ROLE,
        lambda payload: (
            isinstance(payload, RoleLocationV1)
            and payload.node_fingerprint == owner
            and payload.property_name == name
        ),
    )


def _literal(context, kind: RefKind, value: object) -> OperatorRef:
    return _ref(
        context,
        kind,
        lambda payload: isinstance(payload, LiteralValueV1) and payload.value == value,
    )


def _stacks(context) -> tuple[OperatorRef, ...]:
    """The three sibling ``Stack`` nodes under ``root``."""
    return tuple(
        _node_at(context, ("root", "props", "children", index)) for index in range(3)
    )


def _text_leaf(context, stack_index: int) -> OperatorRef:
    """The ``TextContent`` leaf inside one ``Stack`` — has no "direction" role."""
    return _node_at(
        context,
        ("root", "props", "children", stack_index, "props", "children", 0),
    )


def _selector_over(
    context,
    node_refs: tuple[OperatorRef, ...],
    *,
    seed: int = 100,
    scope: str = "1" * 64,
):
    descriptor = build_selector_descriptor(
        table=context.reference_table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=scope,
        matching_refs=node_refs,
        max_fanout=8,
        compiler_facts=(SelectorFact.EXACT_FINITE,),
    )
    return attach_bulk_selector(context, descriptor, seed=seed)


def _apply(library, pack, state, operator_id: str, *bindings):
    return library.apply(
        pack,
        state,
        operator_id,
        tuple(BoundArgumentV1(slot, ref) for slot, ref in bindings),
        _provenance(state.source),
    )


def test_bulk_apply_across_multiple_targets_produces_one_delta_per_target() -> None:
    pack, state, context = _fixture()
    stacks = _stacks(context)
    context, selector_ref = _selector_over(context, stacks)
    library = build_openui_bulk_operator_library(context)
    pack = replace(pack, operator_library=library)
    role_ref = _role(context, stacks[0], ROLE_NAME)
    value_ref = _literal(context, RefKind.VALUE, "row")

    result = _apply(
        library,
        pack,
        state,
        MAP_SET_PROPERTY,
        ("selector", selector_ref),
        ("role", role_ref),
        ("value", value_ref),
    )
    assert result.succeeded and result.state is not None
    assert result.application.effect is not None
    assert len(result.application.effect.property_deltas) == 3
    assert result.application.effect.compiler_coverage.value == "exact"

    parsed = parse_statement_bindings(result.state.source)
    directions = [
        child["props"]["direction"]
        for child in parsed["root"]["props"]["children"]
    ]
    assert directions == ["row", "row", "row"]
    assert pack.oracle(result.state.source).ok


def test_bulk_apply_rejects_atomically_when_one_target_is_incompatible() -> None:
    pack, state, context = _fixture()
    stacks = _stacks(context)
    # One of the two selected targets is a ``TextContent`` leaf, which has no
    # "direction" role at all: the whole bulk apply must reject atomically,
    # with nothing mutated, rather than applying to the compatible member.
    text_leaf = _text_leaf(context, 0)
    context, selector_ref = _selector_over(context, (text_leaf, stacks[1]))
    library = build_openui_bulk_operator_library(context)
    pack = replace(pack, operator_library=library)
    role_ref = _role(context, stacks[0], ROLE_NAME)
    value_ref = _literal(context, RefKind.VALUE, "row")

    before_source = state.source
    result = _apply(
        library,
        pack,
        state,
        MAP_SET_PROPERTY,
        ("selector", selector_ref),
        ("role", role_ref),
        ("value", value_ref),
    )
    assert not result.succeeded
    assert result.state is None
    assert result.application.rejection is not None
    assert result.application.rejection.code == "bulk.incompatible_role_member"
    assert state.source == before_source

    # Confirm the pack-authorized source truly never changed: re-derive a
    # fresh state from the same source and check it round-trips identically.
    fresh = OperatorStateV1.from_source(pack, before_source)
    assert fresh.source == before_source


def test_zero_target_selector_is_rejected_before_mutation() -> None:
    pack, state, context = _fixture()
    context, selector_ref = _selector_over(context, ())
    library = build_openui_bulk_operator_library(context)
    pack = replace(pack, operator_library=library)
    role_ref = _role(context, _stacks(context)[0], ROLE_NAME)
    value_ref = _literal(context, RefKind.VALUE, "row")

    result = _apply(
        library,
        pack,
        state,
        MAP_SET_PROPERTY,
        ("selector", selector_ref),
        ("role", role_ref),
        ("value", value_ref),
    )
    assert not result.succeeded
    assert result.state is None
    assert result.application.rejection is not None
    assert result.application.rejection.code == "bulk.no_targets"


def test_replay_reproduces_the_same_application_identity() -> None:
    pack, state, context = _fixture()
    stacks = _stacks(context)
    context, selector_ref = _selector_over(context, stacks)
    library = build_openui_bulk_operator_library(context)
    pack = replace(pack, operator_library=library)
    role_ref = _role(context, stacks[0], ROLE_NAME)
    value_ref = _literal(context, RefKind.VALUE, "row")

    result = _apply(
        library,
        pack,
        state,
        MAP_SET_PROPERTY,
        ("selector", selector_ref),
        ("role", role_ref),
        ("value", value_ref),
    )
    assert result.succeeded

    replayed = library.replay(pack, state, result.application)
    assert replayed.application.application_id == result.application.application_id
    assert replayed.state == result.state


def test_selector_construction_order_carries_no_semantics_for_the_effect() -> None:
    pack, state, context = _fixture()
    stacks = _stacks(context)

    forward_context, forward_ref = _selector_over(context, stacks, seed=11)
    backward_context, backward_ref = _selector_over(
        context, tuple(reversed(stacks)), seed=12
    )

    forward_library = build_openui_bulk_operator_library(forward_context)
    backward_library = build_openui_bulk_operator_library(backward_context)
    forward_pack = replace(pack, operator_library=forward_library)
    backward_pack = replace(pack, operator_library=backward_library)

    role_ref = _role(context, stacks[0], ROLE_NAME)
    value_ref = _literal(context, RefKind.VALUE, "row")

    forward_result = _apply(
        forward_library,
        forward_pack,
        state,
        MAP_SET_PROPERTY,
        ("selector", forward_ref),
        ("role", role_ref),
        ("value", value_ref),
    )
    backward_result = _apply(
        backward_library,
        backward_pack,
        state,
        MAP_SET_PROPERTY,
        ("selector", backward_ref),
        ("role", role_ref),
        ("value", value_ref),
    )
    assert forward_result.succeeded and backward_result.succeeded
    assert forward_result.state == backward_result.state
    assert (
        forward_result.application.effect.fingerprint
        == backward_result.application.effect.fingerprint
    )


def test_bulk_apply_rejects_when_every_target_already_has_the_value() -> None:
    pack, state, context = _fixture(values=("row",))
    stacks = _stacks(context)
    context, selector_ref = _selector_over(context, stacks)
    library = build_openui_bulk_operator_library(context)
    pack = replace(pack, operator_library=library)
    role_ref = _role(context, stacks[0], ROLE_NAME)
    value_ref = _literal(context, RefKind.VALUE, "row")

    first = _apply(
        library,
        pack,
        state,
        MAP_SET_PROPERTY,
        ("selector", selector_ref),
        ("role", role_ref),
        ("value", value_ref),
    )
    assert first.succeeded and first.state is not None

    pack2, state2, context2 = _fixture(source=first.state.source, values=("row",))
    stacks2 = _stacks(context2)
    context2, selector_ref2 = _selector_over(context2, stacks2, seed=13)
    library2 = build_openui_bulk_operator_library(context2)
    pack2 = replace(pack2, operator_library=library2)
    role_ref2 = _role(context2, stacks2[0], ROLE_NAME)
    value_ref2 = _literal(context2, RefKind.VALUE, "row")
    second = _apply(
        library2,
        pack2,
        state2,
        MAP_SET_PROPERTY,
        ("selector", selector_ref2),
        ("role", role_ref2),
        ("value", value_ref2),
    )
    assert not second.succeeded
    assert second.application.rejection is not None
    assert second.application.rejection.code == "bulk.no_change_member"


def test_declaration_is_registered_with_exact_coverage_and_selector_slot() -> None:
    _, _, context = _fixture()
    library = build_openui_bulk_operator_library(context)
    declaration = library.lookup(MAP_SET_PROPERTY)
    assert declaration.effect_signature == (declaration.effect_signature[0],)
    slot_kinds = {slot.slot_id: slot.ref_kind for slot in declaration.argument_slots}
    assert slot_kinds == {
        "selector": RefKind.SELECTOR,
        "role": RefKind.ROLE,
        "value": RefKind.VALUE,
    }
