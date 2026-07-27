"""Exact atomic bulk set-property operator over ``SelectorRefV1`` (DSH5-02).

``openui.map_set_property(selector, role, value)`` is the first consumer of
the DSH5-01 selector contract: it applies one schema-valid property update to
every node an exact, finite, pack-authorized ``SelectorRef`` names, atomically,
or rejects the whole action with no state change. It is declared in the AST
operator registry like every other compiler-owned operator — never as a
tokenizer special token — and reuses the DSH3-04 local-operator machinery
(``OpenUILocalOperatorContextV1``, its schema/role/value helpers, and the
``SET_PROPERTY`` primitive) rather than building a parallel execution path.

Two extra composition helpers (``build_bulk_selector`` / ``attach_bulk_selector``)
thread a selector through an existing ``OpenUILocalOperatorContextV1`` using
exactly the DSH5-01 builder functions — no bulk-specific selector semantics
are introduced. ``lower_map_set_property_to_primitives`` is a diagnostic-only
equivalence oracle (never a production dispatch path) that applies the same
edit as N sequential ``SET_PROPERTY`` primitives so tests can assert the two
paths produce byte-identical canonical state.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from typing import Any

from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    OperatorRef,
    PreconditionV1,
    RefKind,
    RoleRef,
    SelectorRef,
    ValueRef,
)
from slm_training.dsl.operators.local import (
    NodeLocationV1,
    OpenUILocalOperatorContextV1,
    RoleLocationV1,
    SET_PROPERTY,
    _argument,
    _effect_digest,
    _matches_scalar_schema,
    _mutation,
    _ordered_properties,
    _property_schema,
    _required_property,
    _resolve_literal,
    _resolve_node,
    build_openui_local_operator_context,
    build_openui_local_operator_library,
    openui_local_registered_operators,
)
from slm_training.dsl.operators.references import (
    SelectorDescriptorV1,
    SelectorFact,
    SelectorKind,
    branch_fingerprint,
)
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RegisteredOperatorV1,
)
from slm_training.dsl.operators.selectors import attach_selector, build_selector
from slm_training.dsl.pack import DslPack
from slm_training.dsl.production_codec import parse_statement_bindings

MAP_SET_PROPERTY = "openui.map_set_property"

#: Operator-level bulk fanout policy, independent of (and possibly tighter
#: than) any given selector's own ``max_fanout``. Matches the DSH5-02 test
#: matrix's largest fanout point (8 targets), so the boundary is exercised
#: directly rather than through an arbitrary unrelated number.
MAP_SET_PROPERTY_MAX_FANOUT = 8


def _role_ref_for_node(
    context: OpenUILocalOperatorContextV1, node_fingerprint: str, property_name: str
) -> RoleRef:
    """Find a target node's own role reference for ``property_name``.

    ``context`` already carries one ``RoleRef`` per (node, schema property)
    pair built at context-construction time (DSH3-04); this is the reverse
    lookup from a node's persistent fingerprint back to that ref, used only
    to name the per-target effect-delta target. A missing entry means the
    node's own component schema never declared this property at all.
    """
    for ref in context.references(RefKind.ROLE):
        payload = context.payload(ref)
        if (
            isinstance(payload, RoleLocationV1)
            and payload.node_fingerprint == node_fingerprint
            and payload.property_name == property_name
        ):
            assert isinstance(ref, RoleRef)
            return ref
    raise OperatorRejectedError("local.unsupported_role", "schema.role")


def _declarations() -> tuple[AstOperatorV1, ...]:
    return (
        AstOperatorV1(
            operator_id=MAP_SET_PROPERTY,
            version="v1",
            domain="openui.ast",
            codomain="openui.ast",
            argument_slots=(
                OperatorArgumentSlotV1(
                    "selector", RefKind.SELECTOR, BindingPhase.STATE
                ),
                OperatorArgumentSlotV1("role", RefKind.ROLE, BindingPhase.STATE),
                OperatorArgumentSlotV1(
                    "value", RefKind.VALUE, BindingPhase.APPLICATION
                ),
            ),
            preconditions=(
                PreconditionV1("selector.non_empty", ("selector",)),
                PreconditionV1("schema.bulk_property", ("selector", "role", "value")),
            ),
            effect_signature=(EffectDeltaKind.PROPERTY,),
            locality="selector.property",
            cost=1.0,
            idempotent=True,
        ),
    )


def _executor(operator_id: str, context: OpenUILocalOperatorContextV1):
    def execute(
        state: OperatorStateV1, arguments: tuple[BoundArgumentV1, ...]
    ) -> OperatorMutationV1:
        if operator_id != MAP_SET_PROPERTY:
            raise OperatorRejectedError("operator.unsupported")

        bindings = parse_statement_bindings(state.source, dsl="openui")
        selector_ref = _argument(arguments, "selector", SelectorRef)
        role_ref = _argument(arguments, "role", RoleRef)
        value_ref = _argument(arguments, "value", ValueRef)

        _, target_refs = context.resolve_selector(selector_ref, state)
        if not target_refs:
            raise OperatorRejectedError("local.empty_selector", "selector.non_empty")
        if any(ref.KIND is not RefKind.NODE for ref in target_refs):
            raise OperatorRejectedError(
                "local.selector_not_node_kind", "selector.node_targets"
            )
        if len(target_refs) > MAP_SET_PROPERTY_MAX_FANOUT:
            raise OperatorRejectedError(
                "local.selector_fanout_exceeded", "selector.max_fanout"
            )

        _, role_payload = context.resolve(role_ref, state, RefKind.ROLE)
        if not isinstance(role_payload, RoleLocationV1):
            raise OperatorRejectedError("ref.type_incompatible", "reference.payload")
        property_name = role_payload.property_name
        value = _resolve_literal(context, state, value_ref, RefKind.VALUE)

        resolved = tuple(
            _resolve_node(context, state, ref, bindings) for ref in target_refs
        )
        owner_fingerprints = {
            node_descriptor.semantic_fingerprint
            for node_descriptor, _, _ in resolved
        }
        if role_payload.node_fingerprint not in owner_fingerprints:
            raise OperatorRejectedError(
                "local.role_not_selected", "selector.role_owner"
            )

        # Pass 1 + 2 are fused: every target is validated and mutated in one
        # walk, but atomicity does not depend on that fusion. ``bindings`` is
        # a scratch parse of ``state.source`` local to this call; ``state``
        # itself is a frozen dataclass no branch below can touch. A raise
        # anywhere in this loop discards ``bindings`` before ``_mutation``
        # ever runs, so the owning ``OperatorLibraryV1._execute`` always sees
        # either a fully-applied mutation or none at all — no target update
        # is ever partially committed.
        property_deltas: list[EffectDeltaV1] = []
        for node_descriptor, _location, node in resolved:
            prop_schema = _property_schema(
                context.schema_defs, node, property_name
            )
            if not _matches_scalar_schema(value, prop_schema):
                raise OperatorRejectedError(
                    "local.property_value_invalid", "schema.property"
                )
            props = node.setdefault("props", {})
            before = props.get(property_name)
            if before == value:
                raise OperatorRejectedError("local.no_change", "property.changed")
            ordered_properties = _ordered_properties(context.schema_defs, node)
            role_index = ordered_properties.index(property_name)
            if any(
                name not in props
                and not _required_property(context.schema_defs, node, name)
                for name in ordered_properties[:role_index]
            ):
                raise OperatorRejectedError(
                    "local.unsupported_pack_semantics",
                    "canonical.positional_property",
                )
            target_role_ref = _role_ref_for_node(
                context, node_descriptor.semantic_fingerprint, property_name
            )
            props[property_name] = copy.deepcopy(value)
            property_deltas.append(
                EffectDeltaV1(
                    EffectDeltaKind.PROPERTY,
                    target_role_ref,
                    _effect_digest(before),
                    _effect_digest(value),
                )
            )

        return _mutation(
            bindings,
            ActionEffectV1(
                property_deltas=tuple(property_deltas),
                compiler_coverage=CompilerCoverage.EXACT,
                estimated_completion_cost=float(len(property_deltas)),
            ),
            state.source,
        )

    return execute


def openui_bulk_registered_operators(
    context: OpenUILocalOperatorContextV1,
) -> tuple[RegisteredOperatorV1, ...]:
    """Return the bulk-only entries for composition into a larger pack library."""
    return tuple(
        RegisteredOperatorV1(declaration, _executor(declaration.operator_id, context))
        for declaration in _declarations()
    )


def build_openui_bulk_operator_library(
    context: OpenUILocalOperatorContextV1,
) -> OperatorLibraryV1:
    """Compose the DSH3-04 local primitives with ``openui.map_set_property``."""
    return OperatorLibraryV1(
        (
            *openui_local_registered_operators(context),
            *openui_bulk_registered_operators(context),
        )
    )


def build_bulk_selector(
    context: OpenUILocalOperatorContextV1,
    *,
    selector_kind: SelectorKind,
    scope_fingerprint: str,
    matching_refs: tuple[OperatorRef, ...],
    max_fanout: int,
    seed: int,
    compiler_facts: tuple[SelectorFact, ...] = (),
) -> tuple[OpenUILocalOperatorContextV1, SelectorRef]:
    """Build one exact DSH5-01 selector over ``context`` and attach it.

    Thin composition of :func:`slm_training.dsl.operators.selectors.build_selector`
    with a local operator context's own reference table — introduces no new
    selector semantics, only wires an existing context to consume one.
    """
    new_table, ref = build_selector(
        context.reference_table,
        selector_kind=selector_kind,
        scope_fingerprint=scope_fingerprint,
        matching_refs=matching_refs,
        max_fanout=max_fanout,
        seed=seed,
        compiler_facts=compiler_facts,
    )
    return replace(context, reference_table=new_table), ref


def attach_bulk_selector(
    context: OpenUILocalOperatorContextV1,
    descriptor: SelectorDescriptorV1,
    *,
    seed: int,
) -> OpenUILocalOperatorContextV1:
    """Attach an already-built selector descriptor to ``context``."""
    return replace(
        context,
        reference_table=attach_selector(context.reference_table, descriptor, seed=seed),
    )


def lower_map_set_property_to_primitives(
    pack: DslPack,
    initial_state: OperatorStateV1,
    *,
    target_paths: tuple[tuple[Any, ...], ...],
    property_name: str,
    value: Any,
    request_id: str,
    branch_nonce: str,
    seed: int,
) -> OperatorStateV1:
    """Diagnostic-only equivalence oracle: repeated single-target ``SET_PROPERTY``.

    Never a production dispatch path — ``MAP_SET_PROPERTY`` always applies in
    one atomic pack-authorized commit. This exists solely so tests can assert
    the bulk result is byte-identical to what N sequential primitive
    ``SET_PROPERTY`` applications would produce (the DSH5-02 primitive-
    lowering equivalence requirement). Each step rebuilds a fresh context
    because ``NodeRef``/``RoleRef`` identity is state-bound and a property
    edit changes the edited node's own persistent fingerprint (it is part of
    the node's canonical structure); the node is instead re-located by its
    stable structural path, which a property edit never moves.
    """
    state = initial_state
    for path in sorted(target_paths):
        branch = branch_fingerprint(state.state_digest, branch_nonce)
        context = build_openui_local_operator_context(
            pack,
            state,
            request_id=request_id,
            branch_digest=branch,
            seed=seed,
            values=(value,),
        )
        library = build_openui_local_operator_library(context)
        pack_with_library = replace(pack, operator_library=library)
        node_ref = next(
            ref
            for ref in context.references(RefKind.NODE)
            if isinstance(context.payload(ref), NodeLocationV1)
            and context.payload(ref).path == path  # type: ignore[union-attr]
        )
        node_descriptor = next(
            entry.descriptor
            for entry in context.reference_table.entries
            if entry.ref == node_ref
        )
        role_ref = _role_ref_for_node(
            context, node_descriptor.semantic_fingerprint, property_name
        )
        value_ref = next(
            ref
            for ref in context.references(RefKind.VALUE)
            if context.payload(ref).value == value  # type: ignore[union-attr]
        )
        provenance = ApplicationProvenanceV1(
            pack_id=pack.pack_id,
            compiler_id="openui.bulk_operator_lowering",
            compiler_version="v1",
            source_artifact_digest=hashlib.sha256(state.source.encode()).hexdigest(),
            request_id=request_id,
        )
        result = library.apply(
            pack_with_library,
            state,
            SET_PROPERTY,
            (
                BoundArgumentV1("node", node_ref),
                BoundArgumentV1("role", role_ref),
                BoundArgumentV1("value", value_ref),
            ),
            provenance,
        )
        if result.state is None:
            code = (
                result.application.rejection.code
                if result.application.rejection is not None
                else "operator.executor_rejected"
            )
            raise OperatorRejectedError(code)
        state = result.state
    return state


__all__ = [
    "MAP_SET_PROPERTY",
    "MAP_SET_PROPERTY_MAX_FANOUT",
    "attach_bulk_selector",
    "build_bulk_selector",
    "build_openui_bulk_operator_library",
    "lower_map_set_property_to_primitives",
    "openui_bulk_registered_operators",
]
