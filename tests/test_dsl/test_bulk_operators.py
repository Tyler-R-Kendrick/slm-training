"""Tests for the DSH5-02 atomic bulk set-property operator.

``openui.map_set_property(selector, role, value)`` is the first consumer of
the DSH5-01 ``SelectorRef`` contract: apply one schema-valid property update
to every node an exact selector names, atomically, with a fresh post-commit
reference table and a diagnostic-only primitive-lowering equivalence oracle.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import replace
from typing import Mapping

import pytest

from slm_training.dsl.operators import (
    MAP_SET_PROPERTY,
    MAP_SET_PROPERTY_MAX_FANOUT,
    ApplicationProvenanceV1,
    BoundArgumentV1,
    NodeRef,
    OperatorStateV1,
    RefKind,
    ReferenceResolutionError,
    SelectorKind,
    SelectorRef,
    attach_bulk_selector,
    branch_fingerprint,
    build_bulk_selector,
    build_openui_bulk_operator_library,
    build_openui_local_operator_context,
    build_selector_descriptor,
    lower_map_set_property_to_primitives,
)
from slm_training.dsl.operators.local import NodeLocationV1, RoleLocationV1
from slm_training.dsl.operators.registry import OperatorRejectedError
from slm_training.dsl.pack import get_pack

_SCOPE = "1" * 64


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(source: str, request_id: str = "request-1") -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.bulk_operator_compiler",
        compiler_version="v1",
        source_artifact_digest=_sha(source),
        request_id=request_id,
    )


def _stack_source(n: int, *, preset: Mapping[int, str] | None = None) -> str:
    preset = preset or {}
    parts = []
    for i in range(n):
        child = f'Stack([TextContent(":item{i}")]'
        if i in preset:
            child += f', "{preset[i]}"'
        child += ")"
        parts.append(child)
    return f'root = Card([{", ".join(parts)}], "clear")'


def _mixed_source(n_stacks: int) -> str:
    """N Stacks (expose ``direction``) plus one bare TextContent leaf (does not)."""
    parts = [f'Stack([TextContent(":item{i}")])' for i in range(n_stacks)]
    parts.append('TextContent(":leaf")')
    return f'root = Card([{", ".join(parts)}], "clear")'


def _variant_divergence_source() -> str:
    """A root Card and a nested Callout: both expose ``variant``, disjoint enums."""
    return (
        'root = Card([Callout("info", ":a.title", ":a.desc")], "clear")'
    )


def _fixture(
    source: str,
    *,
    values: tuple[object, ...] = (),
    request_id: str = "request-1",
    seed: int = 7,
):
    pack = get_pack("openui")
    state = OperatorStateV1.from_source(pack, source)
    branch = branch_fingerprint(state.state_digest, "a" * 64)
    context = build_openui_local_operator_context(
        pack,
        state,
        request_id=request_id,
        branch_digest=branch,
        seed=seed,
        values=values,
    )
    return pack, state, context


def _descriptor_for(context, ref):
    return next(
        entry.descriptor for entry in context.reference_table.entries if entry.ref == ref
    )


def _root_children_refs(context) -> tuple[NodeRef, ...]:
    """Every direct child of ``root``, ordered by its position in the tree."""
    located = []
    for ref in context.references(RefKind.NODE):
        payload = context.payload(ref)
        if (
            isinstance(payload, NodeLocationV1)
            and payload.path[:3] == ("root", "props", "children")
            and len(payload.path) == 4
        ):
            located.append((payload.path[3], ref))
    return tuple(ref for _, ref in sorted(located))


def _root_ref(context) -> NodeRef:
    return next(
        ref
        for ref in context.references(RefKind.NODE)
        if isinstance(context.payload(ref), NodeLocationV1)
        and context.payload(ref).path == ("root",)
    )


def _role_ref(context, node_ref, property_name: str):
    owner = _descriptor_for(context, node_ref).semantic_fingerprint
    return next(
        ref
        for ref in context.references(RefKind.ROLE)
        if isinstance(context.payload(ref), RoleLocationV1)
        and context.payload(ref).node_fingerprint == owner
        and context.payload(ref).property_name == property_name
    )


def _value_ref(context, value):
    return next(
        ref for ref in context.references(RefKind.VALUE) if context.payload(ref).value == value
    )


def _with_bulk_selector(
    context,
    target_refs: tuple[NodeRef, ...],
    *,
    selector_kind: SelectorKind = SelectorKind.COMPONENT_TYPE_IN_SCOPE,
    scope: str = _SCOPE,
    max_fanout: int = 8,
    seed: int = 100,
):
    return build_bulk_selector(
        context,
        selector_kind=selector_kind,
        scope_fingerprint=scope,
        matching_refs=target_refs,
        max_fanout=max_fanout,
        seed=seed,
    )


def _library_and_pack(base_pack, context):
    library = build_openui_bulk_operator_library(context)
    return library, replace(base_pack, operator_library=library)


def _apply_bulk(library, pack, state, selector_ref, role_ref, value_ref, request_id="request-1"):
    return library.apply(
        pack,
        state,
        MAP_SET_PROPERTY,
        (
            BoundArgumentV1("selector", selector_ref),
            BoundArgumentV1("role", role_ref),
            BoundArgumentV1("value", value_ref),
        ),
        _provenance(state.source, request_id),
    )


# --------------------------------------------------------------------------
# Declaration / composition
# --------------------------------------------------------------------------


def test_declaration_has_selector_role_value_slots_and_property_effect() -> None:
    _, _, context = _fixture(_stack_source(1), values=("row",))
    library = build_openui_bulk_operator_library(context)
    declaration = library.lookup(MAP_SET_PROPERTY)
    slots = {slot.slot_id: slot for slot in declaration.argument_slots}
    assert set(slots) == {"selector", "role", "value"}
    assert slots["selector"].ref_kind is RefKind.SELECTOR
    assert slots["role"].ref_kind is RefKind.ROLE
    assert slots["value"].ref_kind is RefKind.VALUE
    assert declaration.effect_signature == (
        declaration.effect_signature[0],
    )  # PROPERTY only, declared once
    assert declaration.idempotent is True


def test_composed_library_keeps_every_local_primitive_plus_bulk_operator() -> None:
    _, _, context = _fixture(_stack_source(1), values=("row",))
    library = build_openui_bulk_operator_library(context)
    operator_ids = {declaration.operator_id for declaration in library.declarations}
    assert MAP_SET_PROPERTY in operator_ids
    assert "openui.set_property" in operator_ids
    assert "openui.add_child" in operator_ids


# --------------------------------------------------------------------------
# Success matrix: 1 / 2 / 4 / 8 homogeneous targets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_homogeneous_targets_update_every_node_atomically(count: int) -> None:
    pack, state, context = _fixture(_stack_source(8), values=("row",))
    children = _root_children_refs(context)
    targets = children[:count]
    context, selector_ref = _with_bulk_selector(context, targets)
    library, pack_with_lib = _library_and_pack(pack, context)

    anchor_role = _role_ref(context, targets[0], "direction")
    value_ref = _value_ref(context, "row")
    result = _apply_bulk(library, pack_with_lib, state, selector_ref, anchor_role, value_ref)

    assert result.succeeded and result.state is not None
    assert pack.oracle(result.state.source).ok
    assert result.application.effect is not None
    assert len(result.application.effect.property_deltas) == count
    # Every selected node's own text placeholder still present with direction set.
    for i in range(count):
        assert f'TextContent(":item{i}")], "row"' in result.state.source
    for i in range(count, 8):
        assert f'TextContent(":item{i}")])' in result.state.source


# --------------------------------------------------------------------------
# Incompatible selections must reject atomically, with no state change
# --------------------------------------------------------------------------


def test_incompatible_selected_node_missing_role_rejects_atomically() -> None:
    pack, state, context = _fixture(_mixed_source(2), values=("row",))
    children = _root_children_refs(context)  # 2 stacks + 1 bare TextContent leaf
    assert len(children) == 3
    context, selector_ref = _with_bulk_selector(context, children)
    library, pack_with_lib = _library_and_pack(pack, context)

    anchor_role = _role_ref(context, children[0], "direction")
    value_ref = _value_ref(context, "row")
    result = _apply_bulk(library, pack_with_lib, state, selector_ref, anchor_role, value_ref)

    assert not result.succeeded
    assert result.state is None
    assert result.application.rejection is not None
    assert result.application.rejection.code == "local.unsupported_role"
    # No partial commit: the original source is untouched (state is unmodified,
    # and the rejected application carries no after-state at all).
    assert result.application.after_state_digest is None
    assert state.source == _mixed_source(2)


def test_incompatible_selected_node_value_fails_target_schema_rejects_atomically() -> None:
    pack, state, context = _fixture(_variant_divergence_source(), values=("sunk",))
    root = _root_ref(context)
    callout = next(
        ref
        for ref in context.references(RefKind.NODE)
        if isinstance(context.payload(ref), NodeLocationV1)
        and context.payload(ref).path == ("root", "props", "children", 0)
    )
    targets = (root, callout)
    context, selector_ref = _with_bulk_selector(context, targets)
    library, pack_with_lib = _library_and_pack(pack, context)

    anchor_role = _role_ref(context, root, "variant")
    value_ref = _value_ref(context, "sunk")  # valid for Card, not for Callout
    result = _apply_bulk(library, pack_with_lib, state, selector_ref, anchor_role, value_ref)

    assert not result.succeeded
    assert result.application.rejection is not None
    assert result.application.rejection.code == "local.property_value_invalid"
    assert state.source == _variant_divergence_source()


def test_already_satisfied_subset_rejects_as_no_change_not_partial_commit() -> None:
    source = _stack_source(3, preset={0: "row"})
    pack, state, context = _fixture(source, values=("row",))
    children = _root_children_refs(context)
    context, selector_ref = _with_bulk_selector(context, children)
    library, pack_with_lib = _library_and_pack(pack, context)

    anchor_role = _role_ref(context, children[1], "direction")
    value_ref = _value_ref(context, "row")
    result = _apply_bulk(library, pack_with_lib, state, selector_ref, anchor_role, value_ref)

    assert not result.succeeded
    assert result.application.rejection is not None
    assert result.application.rejection.code == "local.no_change"
    assert state.source == source


def test_empty_selector_rejects_with_no_explicit_no_op_policy() -> None:
    pack, state, context = _fixture(_stack_source(2), values=("row",))
    context, selector_ref = _with_bulk_selector(context, ())
    library, pack_with_lib = _library_and_pack(pack, context)

    children = _root_children_refs(context)
    anchor_role = _role_ref(context, children[0], "direction")
    value_ref = _value_ref(context, "row")
    result = _apply_bulk(library, pack_with_lib, state, selector_ref, anchor_role, value_ref)

    assert not result.succeeded
    assert result.application.rejection is not None
    assert result.application.rejection.code == "local.empty_selector"


def test_selector_over_non_node_refs_rejects() -> None:
    pack, state, context = _fixture(_stack_source(2), values=("row",))
    children = _root_children_refs(context)
    role_refs = tuple(context.references(RefKind.ROLE))[:1]
    context, selector_ref = _with_bulk_selector(context, role_refs)
    library, pack_with_lib = _library_and_pack(pack, context)

    anchor_role = _role_ref(context, children[0], "direction")
    value_ref = _value_ref(context, "row")
    result = _apply_bulk(library, pack_with_lib, state, selector_ref, anchor_role, value_ref)

    assert not result.succeeded
    assert result.application.rejection is not None
    assert result.application.rejection.code == "local.selector_not_node_kind"


def test_role_anchor_must_belong_to_a_selected_node() -> None:
    pack, state, context = _fixture(_mixed_source(2), values=("row",))
    children = _root_children_refs(context)
    stacks = children[:2]
    leaf = children[2]
    # Selector only covers the two stacks; anchor role comes from the leaf,
    # which is not a member of the selection.
    context, selector_ref = _with_bulk_selector(context, stacks)
    library, pack_with_lib = _library_and_pack(pack, context)

    # The leaf's own "text" role is a legitimate RoleRef in this state, but its
    # owner node was never a member of the selection.
    anchor_role = _role_ref(context, leaf, "text")
    value_ref = _value_ref(context, "row")

    result = _apply_bulk(library, pack_with_lib, state, selector_ref, anchor_role, value_ref)
    assert not result.succeeded
    assert result.application.rejection is not None
    assert result.application.rejection.code == "local.role_not_selected"


# --------------------------------------------------------------------------
# Max fanout boundary
# --------------------------------------------------------------------------


def test_max_fanout_boundary_at_and_over_the_bound() -> None:
    pack, state, context = _fixture(_stack_source(9), values=("row",))
    children = _root_children_refs(context)
    assert len(children) == 9

    at_bound = children[:MAP_SET_PROPERTY_MAX_FANOUT]
    assert len(at_bound) == 8
    context_ok, selector_ok = _with_bulk_selector(
        context, at_bound, max_fanout=20, seed=1
    )
    library_ok, pack_ok = _library_and_pack(pack, context_ok)
    role_ok = _role_ref(context_ok, at_bound[0], "direction")
    value_ok = _value_ref(context_ok, "row")
    ok_result = _apply_bulk(library_ok, pack_ok, state, selector_ok, role_ok, value_ok)
    assert ok_result.succeeded

    over_bound = children  # all 9
    context_over, selector_over = _with_bulk_selector(
        context, over_bound, max_fanout=20, seed=2
    )
    library_over, pack_over = _library_and_pack(pack, context_over)
    role_over = _role_ref(context_over, over_bound[0], "direction")
    value_over = _value_ref(context_over, "row")
    over_result = _apply_bulk(
        library_over, pack_over, state, selector_over, role_over, value_over
    )
    assert not over_result.succeeded
    assert over_result.application.rejection is not None
    assert over_result.application.rejection.code == "local.selector_fanout_exceeded"


# --------------------------------------------------------------------------
# Nested scopes bind mutation to exact selector membership
# --------------------------------------------------------------------------


def test_nested_scope_selectors_bound_mutation_to_exact_members() -> None:
    pack, state, context = _fixture(_stack_source(4), values=("row",))
    children = _root_children_refs(context)
    outer_scope = _SCOPE
    inner_scope = "3" * 64

    context, outer_ref = _with_bulk_selector(
        context, children, scope=outer_scope, seed=10
    )
    context, inner_ref = build_bulk_selector(
        context,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=inner_scope,
        matching_refs=children[:1],
        max_fanout=8,
        seed=11,
    )
    assert len(context.reference_table.selectors) == 2
    library, pack_with_lib = _library_and_pack(pack, context)

    # Re-resolve the (reallocated-by-attach) selector refs by content, mirroring
    # DSH5-01's own nested-scope precedent — attaching a second selector
    # reallocates every selector's opaque id.
    outer_descriptor = next(
        entry.descriptor
        for entry in context.reference_table.selectors
        if entry.descriptor.scope_fingerprint == outer_scope
    )
    inner_descriptor = next(
        entry.descriptor
        for entry in context.reference_table.selectors
        if entry.descriptor.scope_fingerprint == inner_scope
    )
    inner_live_ref = next(
        entry.ref
        for entry in context.reference_table.selectors
        if entry.descriptor == inner_descriptor
    )
    # Attaching the inner selector reallocates every selector's opaque id
    # (the same permutation-safe rule DSH5-01 entries already follow), so the
    # earlier ``outer_ref`` is now stale — but ``inner_ref`` was computed
    # against the resulting (post-reallocation) table by ``build_bulk_selector``
    # itself, so it is still live immediately after creation.
    assert not any(entry.ref == outer_ref for entry in context.reference_table.selectors)
    assert inner_ref == inner_live_ref
    role_ref = _role_ref(context, children[0], "direction")
    value_ref = _value_ref(context, "row")

    result = _apply_bulk(library, pack_with_lib, state, inner_live_ref, role_ref, value_ref)
    assert result.succeeded and result.state is not None
    assert result.application.effect is not None
    assert len(result.application.effect.property_deltas) == 1
    assert outer_descriptor.cardinality == 4
    assert inner_descriptor.cardinality == 1


# --------------------------------------------------------------------------
# Target iteration / primitive-argument order carries no semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order", list(itertools.permutations(range(3)))
)
def test_target_iteration_order_does_not_affect_application_identity(
    order: tuple[int, ...],
) -> None:
    pack, state, context = _fixture(_stack_source(3), values=("row",))
    children = _root_children_refs(context)
    permuted_targets = tuple(children[i] for i in order)

    context, selector_ref = _with_bulk_selector(context, permuted_targets, seed=42)
    library, pack_with_lib = _library_and_pack(pack, context)
    role_ref = _role_ref(context, children[0], "direction")
    value_ref = _value_ref(context, "row")

    result = _apply_bulk(library, pack_with_lib, state, selector_ref, role_ref, value_ref)
    assert result.succeeded and result.state is not None
    # Compare against a freshly-built canonical (sorted-order) run for the
    # same fixture/seed: build_selector_descriptor already sorts targets by
    # fingerprint regardless of matching_refs order (DSH5-01), so every
    # permutation must reach the same selector -> same application identity.
    canonical_context, canonical_selector = _with_bulk_selector(
        _fixture(_stack_source(3), values=("row",))[2], children, seed=42
    )
    canonical_library, canonical_pack = _library_and_pack(pack, canonical_context)
    canonical_role = _role_ref(canonical_context, children[0], "direction")
    canonical_value = _value_ref(canonical_context, "row")
    canonical_result = _apply_bulk(
        canonical_library,
        canonical_pack,
        state,
        canonical_selector,
        canonical_role,
        canonical_value,
    )
    assert canonical_result.succeeded and canonical_result.state is not None
    assert result.application.application_id == canonical_result.application.application_id
    assert result.state.source == canonical_result.state.source


def test_duplicate_matching_refs_are_structurally_unrepresentable() -> None:
    _, _, context = _fixture(_stack_source(2), values=("row",))
    children = _root_children_refs(context)
    with pytest.raises(ReferenceResolutionError) as raised:
        build_selector_descriptor(
            table=context.reference_table,
            selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            scope_fingerprint=_SCOPE,
            matching_refs=(children[0], children[0]),
            max_fanout=8,
        )
    assert raised.value.code == "selector.duplicate_target"


def test_stale_state_and_unknown_selector_fail_closed_at_context_resolution() -> None:
    pack, state, context = _fixture(_stack_source(2), values=("row",))
    children = _root_children_refs(context)
    context, selector_ref = _with_bulk_selector(context, children)

    with pytest.raises(OperatorRejectedError) as stale:
        context.resolve_selector(selector_ref, replace(state, state_digest="f" * 64))
    assert stale.value.code == "selector.stale_state"

    with pytest.raises(OperatorRejectedError) as missing:
        context.resolve_selector(SelectorRef("request-1", "not-real"), state)
    assert missing.value.code == "selector.missing"

    with pytest.raises(OperatorRejectedError) as wrong_type:
        context.resolve_selector(children[0], state)  # a NodeRef, not a SelectorRef
    assert wrong_type.value.code == "selector.type_incompatible"


# --------------------------------------------------------------------------
# Replay and fresh post-commit continuation
# --------------------------------------------------------------------------


def test_replay_reconstructs_identical_state_and_application_id() -> None:
    pack, state, context = _fixture(_stack_source(3), values=("row",))
    children = _root_children_refs(context)
    context, selector_ref = _with_bulk_selector(context, children)
    library, pack_with_lib = _library_and_pack(pack, context)
    role_ref = _role_ref(context, children[0], "direction")
    value_ref = _value_ref(context, "row")

    result = _apply_bulk(library, pack_with_lib, state, selector_ref, role_ref, value_ref)
    assert result.succeeded

    replayed = library.replay(pack_with_lib, state, result.application)
    assert replayed.application.application_id == result.application.application_id
    assert replayed.state == result.state


def test_post_commit_state_supports_fresh_reference_table_continuation() -> None:
    pack, state, context = _fixture(_stack_source(2), values=("row",))
    children = _root_children_refs(context)
    context, selector_ref = _with_bulk_selector(context, children)
    library, pack_with_lib = _library_and_pack(pack, context)
    role_ref = _role_ref(context, children[0], "direction")
    value_ref = _value_ref(context, "row")

    result = _apply_bulk(library, pack_with_lib, state, selector_ref, role_ref, value_ref)
    assert result.succeeded and result.state is not None

    fresh_branch = branch_fingerprint(result.state.state_digest, "d" * 64)
    fresh_context = build_openui_local_operator_context(
        pack,
        result.state,
        request_id="request-2",
        branch_digest=fresh_branch,
        seed=99,
    )
    assert fresh_context.reference_table.state_digest == result.state.state_digest
    fresh_children = _root_children_refs(fresh_context)
    assert len(fresh_children) == 2
    for ref in fresh_children:
        assert '"row"' in result.state.source


# --------------------------------------------------------------------------
# Deterministic primitive lowering (diagnostics only)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 4])
def test_primitive_lowering_matches_bulk_atomic_result(count: int) -> None:
    pack, state, context = _fixture(_stack_source(4), values=("row",))
    children = _root_children_refs(context)
    targets = children[:count]
    target_paths = tuple(context.payload(ref).path for ref in targets)

    context, selector_ref = _with_bulk_selector(context, targets)
    library, pack_with_lib = _library_and_pack(pack, context)
    role_ref = _role_ref(context, targets[0], "direction")
    value_ref = _value_ref(context, "row")

    bulk_result = _apply_bulk(library, pack_with_lib, state, selector_ref, role_ref, value_ref)
    assert bulk_result.succeeded and bulk_result.state is not None

    lowered_state = lower_map_set_property_to_primitives(
        pack,
        state,
        target_paths=target_paths,
        property_name="direction",
        value="row",
        request_id="request-1",
        branch_nonce="b" * 64,
        seed=1,
    )
    assert lowered_state.source == bulk_result.state.source
    assert lowered_state.state_digest == bulk_result.state.state_digest


def test_attach_bulk_selector_preserves_existing_entry_refs() -> None:
    pack, state, context = _fixture(_stack_source(2), values=("row",))
    children = _root_children_refs(context)
    node_ref = children[0]
    descriptor = build_selector_descriptor(
        table=context.reference_table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=children,
        max_fanout=8,
    )
    new_context = attach_bulk_selector(context, descriptor, seed=5)
    assert new_context.reference_table.entries == context.reference_table.entries
    assert (
        new_context.resolve(node_ref, state, RefKind.NODE)
        == context.resolve(node_ref, state, RefKind.NODE)
    )
