"""Exact atomic bulk operators over SelectorRefV1 targets (DSH5-02).

``openui.map_set_property`` is the first bulk operator to consume an
already-built :class:`~slm_training.dsl.operators.contracts.SelectorRef`
(DSH5-01): it sets one scalar property uniformly across every node the
selector names, enumerating only when *every* selected node already has a
schema-compatible role for that property, and applying atomically — either
every target is rewritten or none is.

This module deliberately does not build selectors itself (no OpenUI-pack
scope/role extraction lives here — that stays DSH5-01's own deferred scope,
per ``docs/design/dsh5-01-selector-refs.md``). It only consumes a
``SelectorRef`` already committed onto the owning
:class:`~slm_training.dsl.operators.local.OpenUILocalOperatorContextV1`'s
reference table, exactly the "future pack-integration change" DSH5-01
anticipated for its next M1 issue.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    NodeRef,
    OperatorArgumentSlotV1,
    PreconditionV1,
    RefKind,
    RoleRef,
    SelectorRef,
    ValueRef,
)
from slm_training.dsl.operators.local import (
    OpenUILocalOperatorContextV1,
    RoleLocationV1,
    _argument,
    _effect_digest,
    _matches_scalar_schema,
    _mutation,
    _ordered_properties,
    _property_schema,
    _required_property,
    _resolve_literal,
    _resolve_node,
    openui_local_registered_operators,
)
from slm_training.dsl.operators.references import (
    ReferenceDescriptorV1,
    ReferenceResolutionError,
    SelectorDescriptorV1,
)
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RegisteredOperatorV1,
)
from slm_training.dsl.operators.selectors import SelectorContextV1, attach_selector
from slm_training.dsl.production_codec import parse_statement_bindings

MAP_SET_PROPERTY = "openui.map_set_property"


def attach_bulk_selector(
    context: OpenUILocalOperatorContextV1,
    descriptor: SelectorDescriptorV1,
    *,
    seed: int,
) -> tuple[OpenUILocalOperatorContextV1, SelectorRef]:
    """Attach one committed selector to a local operator context's table.

    Thin composition of :func:`selectors.attach_selector` (the permutation-safe
    allocation path every other reference kind already goes through) with the
    ``OpenUILocalOperatorContextV1`` this module's operator executes against.
    Building the descriptor itself (deciding *which* nodes match) stays the
    caller's job — see the module docstring.
    """
    new_table = attach_selector(context.reference_table, descriptor, seed=seed)
    ref = next(
        entry.ref for entry in new_table.selectors if entry.descriptor == descriptor
    )
    return replace(context, reference_table=new_table), ref


def _resolve_selector_targets(
    context: OpenUILocalOperatorContextV1,
    state: OperatorStateV1,
    ref: SelectorRef,
) -> tuple[NodeRef, ...]:
    """Resolve a bound ``SelectorRef`` to its exact, canonically ordered members.

    Reuses ``ReferenceTableV1.resolve_selector`` and
    ``SelectorContextV1.resolve_members`` unchanged — no hand-rolled selector
    membership lookup. The returned order is whatever ``resolve_members``
    returns, which is already deterministic (sorted by descriptor fingerprint,
    independent of the order the selector's targets were originally supplied
    in — see ``test_target_order_carries_no_semantics``), so no further
    sorting is needed here.
    """
    table = context.reference_table
    entry = next((item for item in table.selectors if item.ref == ref), None)
    if entry is None:
        raise OperatorRejectedError("selector.missing", "reference.resolve")
    try:
        descriptor = table.resolve_selector(
            ref,
            state_digest=state.state_digest,
            branch_digest=context.branch_digest,
            expected_kind=entry.descriptor.selector_kind,
            current_scope_fingerprint=entry.descriptor.scope_fingerprint,
            current_target_fingerprints=entry.descriptor.target_fingerprints,
        )
        members = SelectorContextV1(table).resolve_members(descriptor)
    except ReferenceResolutionError as exc:
        raise OperatorRejectedError(exc.code, "reference.resolve") from exc
    for member in members:
        if not isinstance(member, NodeRef):
            raise OperatorRejectedError("ref.type_incompatible", "selector.member_kind")
    return members  # type: ignore[return-value]


def _role_ref_for_node(
    context: OpenUILocalOperatorContextV1,
    node_descriptor: ReferenceDescriptorV1,
    property_name: str,
) -> RoleRef:
    """Find the node-scoped ``RoleRef`` a target already owns for one property.

    Every node's every schema-declared property is enumerated as its own
    ``RoleRef`` entry when the context is built (see
    ``build_openui_local_operator_context``), so this is a lookup over
    already-committed table entries — never a freshly minted reference.
    """
    for ref in context.references(RefKind.ROLE):
        payload = context.payload(ref)
        if (
            isinstance(payload, RoleLocationV1)
            and payload.node_fingerprint == node_descriptor.semantic_fingerprint
            and payload.property_name == property_name
        ):
            assert isinstance(ref, RoleRef)
            return ref
    raise OperatorRejectedError("bulk.incompatible_role_member", "schema.role")


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
                PreconditionV1("schema.bulk_property", ("selector", "role", "value")),
            ),
            effect_signature=(EffectDeltaKind.PROPERTY,),
            locality="selector.property",
            cost=1.0,
            idempotent=True,
        ),
    )


def _executor(context: OpenUILocalOperatorContextV1):
    def execute(
        state: OperatorStateV1, arguments: tuple[BoundArgumentV1, ...]
    ) -> OperatorMutationV1:
        bindings = parse_statement_bindings(state.source, dsl="openui")
        selector_ref = _argument(arguments, "selector", SelectorRef)
        role_ref = _argument(arguments, "role", RoleRef)
        value_ref = _argument(arguments, "value", ValueRef)

        targets = _resolve_selector_targets(context, state, selector_ref)
        if not targets:
            raise OperatorRejectedError("bulk.no_targets", "selector.non_empty")

        _, role_payload = context.resolve(role_ref, state, RefKind.ROLE)
        if not isinstance(role_payload, RoleLocationV1):
            raise OperatorRejectedError("ref.type_incompatible", "reference.payload")
        property_name = role_payload.property_name

        value = _resolve_literal(context, state, value_ref, RefKind.VALUE)

        # Validation pass: every target must be schema-compatible before any
        # node is touched. Nothing in ``bindings`` is mutated here.
        validated: list[tuple[Any, RoleRef, Any, Any]] = []
        for target_ref in targets:
            node_descriptor, _, node = _resolve_node(
                context, state, target_ref, bindings
            )
            target_role_ref = _role_ref_for_node(
                context, node_descriptor, property_name
            )
            prop_schema = _property_schema(
                context.schema_defs, node, property_name
            )
            if not _matches_scalar_schema(value, prop_schema):
                raise OperatorRejectedError(
                    "bulk.incompatible_role_member", "schema.property"
                )
            props = dict(node.get("props") or {})
            before = props.get(property_name)
            if before == value:
                raise OperatorRejectedError(
                    "bulk.no_change_member", "property.changed"
                )
            ordered_properties = _ordered_properties(context.schema_defs, node)
            role_index = ordered_properties.index(property_name)
            if any(
                name not in props
                and not _required_property(context.schema_defs, node, name)
                for name in ordered_properties[:role_index]
            ):
                raise OperatorRejectedError(
                    "bulk.incompatible_role_member",
                    "canonical.positional_property",
                )
            validated.append((node, target_role_ref, before, value))

        # Apply pass: only reachable once every target above has passed.
        for node, _, _, after in validated:
            props = node.setdefault("props", {})
            props[property_name] = after

        return _mutation(
            bindings,
            ActionEffectV1(
                property_deltas=tuple(
                    EffectDeltaV1(
                        EffectDeltaKind.PROPERTY,
                        target_role_ref,
                        _effect_digest(before),
                        _effect_digest(after),
                    )
                    for _, target_role_ref, before, after in validated
                ),
                compiler_coverage=CompilerCoverage.EXACT,
                estimated_completion_cost=float(len(validated)),
            ),
            state.source,
        )

    return execute


def build_openui_bulk_operator_library(
    context: OpenUILocalOperatorContextV1,
) -> OperatorLibraryV1:
    return OperatorLibraryV1(
        (
            *openui_local_registered_operators(context),
            *openui_bulk_registered_operators(context),
        )
    )


def openui_bulk_registered_operators(
    context: OpenUILocalOperatorContextV1,
) -> tuple[RegisteredOperatorV1, ...]:
    """Return the bulk-only entries for composition into a larger pack library."""
    return tuple(
        RegisteredOperatorV1(declaration, _executor(context))
        for declaration in _declarations()
    )


__all__ = [
    "MAP_SET_PROPERTY",
    "attach_bulk_selector",
    "build_openui_bulk_operator_library",
    "openui_bulk_registered_operators",
]
