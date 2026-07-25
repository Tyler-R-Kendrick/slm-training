"""Exact topology and template-alias OpenUI operators (DSH3-05)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

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
    OperatorRef,
    PreconditionV1,
    RefKind,
    RoleRef,
    TemplateRef,
    _canonical_json,
    _fingerprint,
    _require_digest,
    _require_identifier,
)
from slm_training.dsl.operators.local import (
    IndexLocationV1,
    LiteralValueV1,
    NodeLocationV1,
    OpenUILocalOperatorContextV1,
    _allowed_component,
    _argument,
    _at,
    _effect_digest,
    _element,
    _mutation,
    _parent,
    _property_schema,
    _replace_at,
    _resolve_node,
    _resolve_role,
    _value_fingerprint,
    build_openui_local_operator_context,
    openui_local_registered_operators,
)
from slm_training.dsl.operators.references import ordered_parent_digest
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RegisteredOperatorV1,
)
from slm_training.dsl.pack import DslPack
from slm_training.dsl.production_codec import parse_statement_bindings

MOVE_NODE = "openui.move_node"
REPARENT_NODE = "openui.reparent_node"
WRAP_NODE = "openui.wrap_node"
UNWRAP_NODE = "openui.unwrap_node"
DUPLICATE_SUBTREE = "openui.duplicate_subtree"
EXPAND_TEMPLATE = "openui.expand_template"
CONTRACT_SUBTREE = "openui.contract_subtree"

TEMPLATE_LOWERING_ID = "openui.production_codec.statement_bindings"


def _freeze_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_tree(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_tree(item) for item in value)
    return value


def _thaw_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_tree(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class OpenUITemplateAliasV1:
    """Pack/compiler-owned exact expanded ↔ contracted subtree alias."""

    pack_id: str
    expanded: Mapping[str, Any]
    contracted: Mapping[str, Any]
    source_artifact_digest: str
    child_role: str | None = None
    lowering_id: str = TEMPLATE_LOWERING_ID
    schema: str = "openui_template_alias/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.pack_id, "pack_id")
        _require_identifier(self.lowering_id, "lowering_id")
        _require_digest(self.source_artifact_digest, "source_artifact_digest")
        if self.child_role is not None:
            _require_identifier(self.child_role, "child_role")
        if not _element(self.expanded) or not _element(self.contracted):
            raise ValueError("template aliases require element subtrees")
        _canonical_json(self.expanded)
        _canonical_json(self.contracted)
        if self.expanded == self.contracted:
            raise ValueError("template expansion and contraction must differ")
        object.__setattr__(self, "expanded", _freeze_tree(self.expanded))
        object.__setattr__(self, "contracted", _freeze_tree(self.contracted))

    @property
    def expanded_value(self) -> dict[str, Any]:
        return _thaw_tree(self.expanded)

    @property
    def contracted_value(self) -> dict[str, Any]:
        return _thaw_tree(self.contracted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pack_id": self.pack_id,
            "expanded": self.expanded_value,
            "contracted": self.contracted_value,
            "source_artifact_digest": self.source_artifact_digest,
            "child_role": self.child_role,
            "lowering_id": self.lowering_id,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def evidence(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pack_id": self.pack_id,
            "template_fingerprint": self.fingerprint,
            "source_artifact_digest": self.source_artifact_digest,
            "lowering_id": self.lowering_id,
        }


@dataclass(frozen=True)
class OpenUITopologyOperatorContextV1:
    local: OpenUILocalOperatorContextV1
    _aliases: Mapping[str, OpenUITemplateAliasV1]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_aliases", MappingProxyType(dict(self._aliases)))

    @property
    def has_template_aliases(self) -> bool:
        return bool(self._aliases)

    def template_alias(
        self,
        ref: OperatorRef,
        state: OperatorStateV1,
    ) -> tuple[OpenUITemplateAliasV1, Mapping[str, Any]]:
        descriptor, payload = self.local.resolve(ref, state, RefKind.TEMPLATE)
        alias = self._aliases.get(descriptor.fingerprint)
        if alias is None or not isinstance(payload, LiteralValueV1):
            raise OperatorRejectedError("template.unsupported", "template.alias")
        expanded = alias.expanded_value
        if payload.value != expanded:
            raise OperatorRejectedError(
                "template.provenance_invalid", "template.payload"
            )
        return alias, expanded


def _contains_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("type") == "ref":
            return True
        return any(_contains_reference(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_reference(item) for item in value)
    return False


def _capture_free(value: Any) -> None:
    if _contains_reference(value):
        raise OperatorRejectedError(
            "topology.capture_unsupported", "scope.capture_free"
        )


def _inline_parent(
    bindings: Mapping[str, Any], location: NodeLocationV1
) -> tuple[list[Any], int]:
    if location.path == ("root",):
        raise OperatorRejectedError("topology.root_move", "node.non_root")
    if len(location.path) == 1:
        raise OperatorRejectedError("topology.unsupported", "binding.reference_graph")
    parent, key = _parent(bindings, location.path)
    if not isinstance(parent, list) or not isinstance(key, int):
        raise OperatorRejectedError("topology.unsupported", "node.inline_child")
    return parent, key


def _destination_children(
    context: OpenUITopologyOperatorContextV1,
    state: OperatorStateV1,
    bindings: Mapping[str, Any],
    parent_ref: OperatorRef,
    role_ref: OperatorRef,
) -> tuple[Any, Any, list[Any], Mapping[str, Any]]:
    parent_descriptor, _, parent = _resolve_node(
        context.local, state, parent_ref, bindings
    )
    role = _resolve_role(context.local, state, role_ref, parent_descriptor)
    prop_schema = _property_schema(
        context.local.schema_defs, parent, role.property_name
    )
    children = dict(parent.get("props") or {}).get(role.property_name)
    if prop_schema.get("type") != "array" or not isinstance(children, list):
        raise OperatorRejectedError(
            "topology.incompatible_cardinality", "schema.child_role"
        )
    return parent_descriptor, role, children, prop_schema


def _insertion_position(
    context: OpenUITopologyOperatorContextV1,
    state: OperatorStateV1,
    parent_descriptor: Any,
    role: Any,
    children: list[Any],
    arguments: tuple[BoundArgumentV1, ...],
) -> int:
    index_ref = next(
        (argument.value for argument in arguments if argument.slot_id == "index"),
        None,
    )
    if index_ref is None:
        return len(children)
    order_digest = ordered_parent_digest(
        parent_descriptor.semantic_fingerprint,
        tuple(_value_fingerprint(child) for child in children),
    )
    index_descriptor, payload = context.local.resolve(
        index_ref,
        state,
        RefKind.INDEX,
        current_parent_order_digest=order_digest,
    )
    if (
        not isinstance(payload, IndexLocationV1)
        or payload.node_fingerprint != parent_descriptor.semantic_fingerprint
        or payload.property_name != role.property_name
    ):
        raise OperatorRejectedError("topology.index_role_mismatch", "index.owner")
    position = index_descriptor.position
    if position is None or not 0 <= position <= len(children):
        raise OperatorRejectedError("topology.invalid_index", "index.range")
    return position


def _topology_effect(
    target: OperatorRef,
    before: Any,
    after: Any,
    *,
    cost: float,
    cardinality: tuple[EffectDeltaV1, ...] = (),
    explicit: bool = False,
) -> ActionEffectV1:
    return ActionEffectV1(
        cardinality_deltas=cardinality,
        topology_deltas=(
            EffectDeltaV1(
                EffectDeltaKind.TOPOLOGY,
                target,
                before if explicit else _effect_digest(before),
                after if explicit else _effect_digest(after),
            ),
        ),
        compiler_coverage=CompilerCoverage.EXACT,
        estimated_completion_cost=cost,
    )


def _move_or_duplicate(
    context: OpenUITopologyOperatorContextV1,
    state: OperatorStateV1,
    arguments: tuple[BoundArgumentV1, ...],
    *,
    duplicate: bool,
) -> OperatorMutationV1:
    bindings = parse_statement_bindings(state.source, dsl="openui")
    node_ref = _argument(arguments, "node", NodeRef)
    parent_ref = _argument(arguments, "new_parent", NodeRef)
    role_ref = _argument(arguments, "role", RoleRef)
    _, location, node = _resolve_node(context.local, state, node_ref, bindings)
    source_children, source_index = _inline_parent(bindings, location)
    parent_descriptor, role, destination, prop_schema = _destination_children(
        context, state, bindings, parent_ref, role_ref
    )
    parent_location = context.local.payload(parent_ref)
    if not isinstance(parent_location, NodeLocationV1):
        raise OperatorRejectedError("ref.type_incompatible", "reference.payload")
    if (
        not duplicate
        and len(parent_location.path) >= len(location.path)
        and parent_location.path[: len(location.path)] == location.path
    ):
        raise OperatorRejectedError("topology.cycle", "topology.acyclic")
    _capture_free(node)
    component = str(node.get("typeName") or "")
    if not _allowed_component(prop_schema, component):
        raise OperatorRejectedError(
            "topology.incompatible_child", "schema.compatible_child"
        )
    position = _insertion_position(
        context, state, parent_descriptor, role, destination, arguments
    )
    moving_within_role = source_children is destination
    before_source_count = len(source_children)
    before_destination_count = len(destination)
    subtree = copy.deepcopy(node)
    if not duplicate:
        source_children.pop(source_index)
        if moving_within_role and source_index < position:
            position -= 1
    destination.insert(position, subtree)
    if not duplicate and moving_within_role and position == source_index:
        raise OperatorRejectedError("topology.no_change", "topology.changed")
    cardinality: list[EffectDeltaV1] = []
    if duplicate or not moving_within_role:
        cardinality.append(
            EffectDeltaV1(
                EffectDeltaKind.CARDINALITY,
                role_ref,
                before_destination_count,
                len(destination),
            )
        )
    if not duplicate and not moving_within_role:
        cardinality.append(
            EffectDeltaV1(
                EffectDeltaKind.CARDINALITY,
                node_ref,
                before_source_count,
                len(source_children),
            )
        )
    action = "duplicate" if duplicate else "move"
    return _mutation(
        bindings,
        _topology_effect(
            node_ref,
            (action, location.path, source_index),
            (action, parent_location.path, role.property_name, position),
            cost=2.0 if duplicate else 1.5,
            cardinality=tuple(cardinality),
        ),
        state.source,
    )


def _replace_with_template(
    context: OpenUITopologyOperatorContextV1,
    state: OperatorStateV1,
    arguments: tuple[BoundArgumentV1, ...],
    *,
    expand: bool,
) -> OperatorMutationV1:
    bindings = parse_statement_bindings(state.source, dsl="openui")
    node_ref = _argument(arguments, "node", NodeRef)
    template_ref = _argument(arguments, "template", TemplateRef)
    _, location, node = _resolve_node(context.local, state, node_ref, bindings)
    alias, expanded = context.template_alias(template_ref, state)
    expected = alias.contracted_value if expand else alias.expanded_value
    replacement = expanded if expand else alias.contracted_value
    if node != expected:
        raise OperatorRejectedError("template.mismatch", "template.exact_subtree")
    if len(location.path) > 1:
        parent, key = _parent(bindings, location.path)
        if isinstance(parent, list) and isinstance(key, int):
            owner = _at(bindings, location.path[:-3])
            prop_schema = _property_schema(
                context.local.schema_defs, owner, str(location.path[-2])
            )
            if not _allowed_component(
                prop_schema, str(replacement.get("typeName") or "")
            ):
                raise OperatorRejectedError(
                    "topology.incompatible_child", "schema.compatible_child"
                )
    _capture_free(replacement)
    _replace_at(bindings, location.path, replacement)
    evidence = alias.evidence()
    evidence["operation"] = "expand" if expand else "contract"
    return _mutation(
        bindings,
        _topology_effect(
            node_ref,
            {"subtree": _effect_digest(node), "template": evidence},
            {"subtree": _effect_digest(replacement), "template": evidence},
            cost=2.0,
            explicit=True,
        ),
        state.source,
    )


def _declarations() -> tuple[AstOperatorV1, ...]:
    node = lambda slot: OperatorArgumentSlotV1(  # noqa: E731
        slot, RefKind.NODE, BindingPhase.STATE
    )
    role = lambda slot: OperatorArgumentSlotV1(  # noqa: E731
        slot, RefKind.ROLE, BindingPhase.STATE
    )
    index = lambda: OperatorArgumentSlotV1(  # noqa: E731
        "index", RefKind.INDEX, BindingPhase.STATE, required=False
    )
    template = lambda: OperatorArgumentSlotV1(  # noqa: E731
        "template", RefKind.TEMPLATE, BindingPhase.APPLICATION
    )
    common = {
        "version": "v1",
        "domain": "openui.ast",
        "codomain": "openui.ast",
    }
    move_slots = (node("node"), node("new_parent"), role("role"), index())
    return (
        AstOperatorV1(
            operator_id=MOVE_NODE,
            argument_slots=move_slots,
            preconditions=(
                PreconditionV1("topology.acyclic", ("node", "new_parent")),
                PreconditionV1("scope.capture_free", ("node",)),
            ),
            effect_signature=(EffectDeltaKind.CARDINALITY, EffectDeltaKind.TOPOLOGY),
            locality="subtree.two_parents",
            cost=1.5,
            inverse_operator_id=MOVE_NODE,
            **common,
        ),
        AstOperatorV1(
            operator_id=REPARENT_NODE,
            argument_slots=move_slots,
            preconditions=(
                PreconditionV1("topology.acyclic", ("node", "new_parent")),
                PreconditionV1("scope.capture_free", ("node",)),
            ),
            effect_signature=(EffectDeltaKind.CARDINALITY, EffectDeltaKind.TOPOLOGY),
            locality="subtree.two_parents",
            cost=1.5,
            inverse_operator_id=REPARENT_NODE,
            **common,
        ),
        AstOperatorV1(
            operator_id=WRAP_NODE,
            argument_slots=(node("node"), template()),
            preconditions=(
                PreconditionV1("template.wrapper", ("node", "template")),
                PreconditionV1("scope.capture_free", ("node",)),
            ),
            effect_signature=(EffectDeltaKind.TOPOLOGY,),
            locality="subtree.parent",
            cost=1.5,
            inverse_operator_id=UNWRAP_NODE,
            **common,
        ),
        AstOperatorV1(
            operator_id=UNWRAP_NODE,
            argument_slots=(node("node"),),
            preconditions=(PreconditionV1("topology.single_child_wrapper", ("node",)),),
            effect_signature=(EffectDeltaKind.TOPOLOGY,),
            locality="subtree.parent",
            cost=1.0,
            inverse_operator_id=WRAP_NODE,
            **common,
        ),
        AstOperatorV1(
            operator_id=DUPLICATE_SUBTREE,
            argument_slots=move_slots,
            preconditions=(
                PreconditionV1("scope.capture_free", ("node",)),
                PreconditionV1(
                    "schema.compatible_child", ("node", "new_parent", "role")
                ),
            ),
            effect_signature=(EffectDeltaKind.CARDINALITY, EffectDeltaKind.TOPOLOGY),
            locality="subtree.destination_parent",
            cost=2.0,
            **common,
        ),
        AstOperatorV1(
            operator_id=EXPAND_TEMPLATE,
            argument_slots=(node("node"), template()),
            preconditions=(
                PreconditionV1("template.exact_contracted", ("node", "template")),
            ),
            effect_signature=(EffectDeltaKind.TOPOLOGY,),
            locality="subtree",
            cost=2.0,
            inverse_operator_id=CONTRACT_SUBTREE,
            **common,
        ),
        AstOperatorV1(
            operator_id=CONTRACT_SUBTREE,
            argument_slots=(node("node"), template()),
            preconditions=(
                PreconditionV1("template.exact_expanded", ("node", "template")),
            ),
            effect_signature=(EffectDeltaKind.TOPOLOGY,),
            locality="subtree",
            cost=2.0,
            inverse_operator_id=EXPAND_TEMPLATE,
            **common,
        ),
    )


def _executor(operator_id: str, context: OpenUITopologyOperatorContextV1):
    def execute(
        state: OperatorStateV1, arguments: tuple[BoundArgumentV1, ...]
    ) -> OperatorMutationV1:
        if operator_id in {MOVE_NODE, REPARENT_NODE}:
            return _move_or_duplicate(context, state, arguments, duplicate=False)
        if operator_id == DUPLICATE_SUBTREE:
            return _move_or_duplicate(context, state, arguments, duplicate=True)
        if operator_id in {EXPAND_TEMPLATE, CONTRACT_SUBTREE}:
            return _replace_with_template(
                context,
                state,
                arguments,
                expand=operator_id == EXPAND_TEMPLATE,
            )
        if operator_id == WRAP_NODE:
            bindings = parse_statement_bindings(state.source, dsl="openui")
            node_ref = _argument(arguments, "node", NodeRef)
            template_ref = _argument(arguments, "template", TemplateRef)
            _, location, node = _resolve_node(context.local, state, node_ref, bindings)
            alias, wrapper = context.template_alias(template_ref, state)
            if alias.child_role is None:
                raise OperatorRejectedError(
                    "template.unsupported", "template.child_role"
                )
            _capture_free(node)
            _capture_free(wrapper)
            prop_schema = _property_schema(
                context.local.schema_defs, wrapper, alias.child_role
            )
            children = dict(wrapper.get("props") or {}).get(alias.child_role)
            if (
                prop_schema.get("type") != "array"
                or not isinstance(children, list)
                or children
            ):
                raise OperatorRejectedError(
                    "topology.incompatible_cardinality", "template.empty_child_role"
                )
            if not _allowed_component(prop_schema, str(node.get("typeName") or "")):
                raise OperatorRejectedError(
                    "topology.incompatible_child", "schema.compatible_child"
                )
            children.append(copy.deepcopy(node))
            _replace_at(bindings, location.path, wrapper)
            evidence = alias.evidence()
            evidence["operation"] = "wrap"
            return _mutation(
                bindings,
                _topology_effect(
                    node_ref,
                    {"subtree": _effect_digest(node), "template": evidence},
                    {"subtree": _effect_digest(wrapper), "template": evidence},
                    cost=1.5,
                    explicit=True,
                ),
                state.source,
            )
        if operator_id == UNWRAP_NODE:
            bindings = parse_statement_bindings(state.source, dsl="openui")
            node_ref = _argument(arguments, "node", NodeRef)
            _, location, wrapper = _resolve_node(
                context.local, state, node_ref, bindings
            )
            _capture_free(wrapper)
            children = dict(wrapper.get("props") or {}).get("children")
            try:
                child_schema = _property_schema(
                    context.local.schema_defs, wrapper, "children"
                )
            except OperatorRejectedError as exc:
                raise OperatorRejectedError(
                    "topology.unsupported", "topology.children_role"
                ) from exc
            if (
                child_schema.get("type") != "array"
                or not isinstance(children, list)
                or len(children) != 1
            ):
                raise OperatorRejectedError(
                    "topology.incompatible_cardinality",
                    "topology.single_child_wrapper",
                )
            child = copy.deepcopy(children[0])
            if not _element(child):
                raise OperatorRejectedError(
                    "topology.unsupported", "topology.element_child"
                )
            _replace_at(bindings, location.path, child)
            return _mutation(
                bindings,
                _topology_effect(
                    node_ref,
                    ("unwrap", _effect_digest(wrapper)),
                    ("unwrap", _effect_digest(child)),
                    cost=1.0,
                ),
                state.source,
            )
        raise OperatorRejectedError("operator.unsupported")

    return execute


def build_openui_topology_operator_context(
    pack: DslPack,
    state: OperatorStateV1,
    *,
    request_id: str,
    branch_digest: str,
    seed: int,
    template_aliases: tuple[OpenUITemplateAliasV1, ...] = (),
    values: tuple[Any, ...] = (),
) -> OpenUITopologyOperatorContextV1:
    """Build one exact topology context from explicit pack-owned aliases."""
    if any(alias.pack_id != pack.pack_id for alias in template_aliases):
        raise OperatorRejectedError("template.provenance_invalid", "template.pack")
    if any(alias.lowering_id != TEMPLATE_LOWERING_ID for alias in template_aliases):
        raise OperatorRejectedError("template.provenance_invalid", "template.lowering")
    fingerprints = tuple(alias.fingerprint for alias in template_aliases)
    if len(set(fingerprints)) != len(fingerprints):
        raise OperatorRejectedError("template.duplicate", "template.identity")
    expanded = tuple(alias.expanded_value for alias in template_aliases)
    if len({_value_fingerprint(value) for value in expanded}) != len(expanded):
        raise OperatorRejectedError("template.duplicate", "template.expansion")
    local = build_openui_local_operator_context(
        pack,
        state,
        request_id=request_id,
        branch_digest=branch_digest,
        seed=seed,
        templates=expanded,
        values=values,
    )
    aliases: dict[str, OpenUITemplateAliasV1] = {}
    for entry in local.reference_table.entries:
        if entry.descriptor.ref_kind is not RefKind.TEMPLATE:
            continue
        payload = local.payload(entry.ref)
        if not isinstance(payload, LiteralValueV1):
            continue
        alias = next(
            (
                candidate
                for candidate in template_aliases
                if candidate.expanded_value == payload.value
            ),
            None,
        )
        if alias is not None:
            aliases[entry.descriptor.fingerprint] = alias
    return OpenUITopologyOperatorContextV1(local, aliases)


def build_openui_topology_operator_library(
    context: OpenUITopologyOperatorContextV1,
) -> OperatorLibraryV1:
    template_operators = {
        WRAP_NODE,
        EXPAND_TEMPLATE,
        CONTRACT_SUBTREE,
    }
    entries = (
        *openui_local_registered_operators(context.local),
        *(
            RegisteredOperatorV1(
                declaration, _executor(declaration.operator_id, context)
            )
            for declaration in _declarations()
            if context.has_template_aliases
            or declaration.operator_id not in template_operators
        ),
    )
    return OperatorLibraryV1(tuple(entries))


__all__ = [
    "CONTRACT_SUBTREE",
    "DUPLICATE_SUBTREE",
    "EXPAND_TEMPLATE",
    "MOVE_NODE",
    "REPARENT_NODE",
    "TEMPLATE_LOWERING_ID",
    "UNWRAP_NODE",
    "WRAP_NODE",
    "OpenUITemplateAliasV1",
    "OpenUITopologyOperatorContextV1",
    "build_openui_topology_operator_context",
    "build_openui_topology_operator_library",
]
