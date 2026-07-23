"""Exact pack-owned local operators for statement-structural OpenUI (DSH3-04)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

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
    ValueRef,
    _fingerprint,
)
from slm_training.dsl.operators.references import (
    CompilerFact,
    ReferenceDescriptorV1,
    ReferenceResolutionError,
    ReferenceTableV1,
    branch_local_disambiguator,
    build_reference_table,
    ordered_parent_digest,
    persistent_node_fingerprint,
)
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RegisteredOperatorV1,
)
from slm_training.dsl.pack import DslPack
from slm_training.dsl.production_codec import (
    emit_statement_bindings,
    parse_statement_bindings,
)

ADD_CHILD = "openui.add_child"
REMOVE_NODE = "openui.remove_node"
REPLACE_NODE = "openui.replace_node"
SET_PROPERTY = "openui.set_property"
UNSET_PROPERTY = "openui.unset_property"
REORDER_CHILDREN = "openui.reorder_children"

PathStep = str | int
AstPath = tuple[PathStep, ...]


@dataclass(frozen=True)
class NodeLocationV1:
    path: AstPath


@dataclass(frozen=True)
class RoleLocationV1:
    node_fingerprint: str
    property_name: str


@dataclass(frozen=True)
class IndexLocationV1:
    node_fingerprint: str
    property_name: str
    position: int


@dataclass(frozen=True)
class LiteralValueV1:
    value: Any


@dataclass(frozen=True)
class OpenUILocalOperatorContextV1:
    """Compiler-private payloads behind one serialized opaque reference table."""

    reference_table: ReferenceTableV1
    branch_digest: str
    schema_defs: Mapping[str, Any]
    _payloads: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_defs", _freeze(self.schema_defs))
        object.__setattr__(self, "_payloads", MappingProxyType(dict(self._payloads)))

    def references(self, kind: RefKind) -> tuple[OperatorRef, ...]:
        return tuple(
            entry.ref
            for entry in self.reference_table.entries
            if entry.descriptor.ref_kind is kind
        )

    def payload(self, ref: OperatorRef) -> object:
        for entry in self.reference_table.entries:
            if entry.ref == ref:
                return copy.deepcopy(self._payloads[entry.descriptor.fingerprint])
        raise ReferenceResolutionError("ref.missing")

    def resolve(
        self,
        ref: OperatorRef,
        state: OperatorStateV1,
        expected_kind: RefKind,
        *,
        current_parent_order_digest: str | None = None,
    ) -> tuple[ReferenceDescriptorV1, object]:
        try:
            descriptor = self.reference_table.resolve(
                ref,
                state_digest=state.state_digest,
                branch_digest=self.branch_digest,
                expected_kind=expected_kind,
                current_parent_order_digest=current_parent_order_digest,
            )
        except ReferenceResolutionError as exc:
            raise OperatorRejectedError(exc.code, "reference.resolve") from exc
        return descriptor, copy.deepcopy(self._payloads[descriptor.fingerprint])


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _semantic_structure(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("type") == "ref":
            return {"type": "ref"}
        return {key: _semantic_structure(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_semantic_structure(item) for item in value]
    return value


def _value_fingerprint(value: Any) -> str:
    return _fingerprint(
        {"schema": "openui_local_value/v1", "value": _semantic_structure(value)}
    )


def _element(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "element"


def _walk_nodes(
    value: Any, path: AstPath
) -> tuple[tuple[AstPath, Mapping[str, Any]], ...]:
    found: list[tuple[AstPath, Mapping[str, Any]]] = []
    if _element(value):
        found.append((path, value))
        for prop, child in sorted(dict(value.get("props") or {}).items()):
            found.extend(_walk_nodes(child, (*path, "props", prop)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_nodes(child, (*path, index)))
    return tuple(found)


def _at(root: Any, path: AstPath) -> Any:
    value = root
    for step in path:
        value = value[step]
    return value


def _parent(root: Any, path: AstPath) -> tuple[Any, PathStep]:
    if not path:
        raise OperatorRejectedError("local.root_required", "node.non_root")
    return _at(root, path[:-1]), path[-1]


def _replace_at(root: Any, path: AstPath, value: Any) -> None:
    parent, key = _parent(root, path)
    parent[key] = copy.deepcopy(value)


def _component_schema(
    schema_defs: Mapping[str, Any], node: Mapping[str, Any]
) -> Mapping[str, Any]:
    component = str(node.get("typeName") or "")
    schema = schema_defs.get(component)
    if not isinstance(schema, Mapping):
        raise OperatorRejectedError(
            "local.unsupported_pack_semantics", "schema.component"
        )
    return schema


def _property_schema(
    schema_defs: Mapping[str, Any], node: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    component = _component_schema(schema_defs, node)
    props = component.get("properties")
    schema = props.get(name) if isinstance(props, Mapping) else None
    if not isinstance(schema, Mapping):
        raise OperatorRejectedError("local.unsupported_role", "schema.role")
    return schema


def _required_property(
    schema_defs: Mapping[str, Any], node: Mapping[str, Any], name: str
) -> bool:
    required = _component_schema(schema_defs, node).get("required", ())
    return name in required if isinstance(required, Sequence) else False


def _ordered_properties(
    schema_defs: Mapping[str, Any], node: Mapping[str, Any]
) -> tuple[str, ...]:
    properties = _component_schema(schema_defs, node).get("properties", {})
    return tuple(properties) if isinstance(properties, Mapping) else ()


def _allowed_component(property_schema: Mapping[str, Any], component: str) -> bool:
    item_schema = (
        property_schema.get("items")
        if property_schema.get("type") == "array"
        else property_schema
    )
    if not isinstance(item_schema, Mapping) or not item_schema:
        return True
    options = item_schema.get("anyOf", (item_schema,))
    if not isinstance(options, Sequence):
        return True
    refs = {
        str(option.get("$ref", "")).rsplit("/", 1)[-1]
        for option in options
        if isinstance(option, Mapping) and option.get("$ref")
    }
    return not refs or component in refs


def _matches_scalar_schema(value: Any, schema: Mapping[str, Any]) -> bool:
    if "enum" in schema and value not in schema["enum"]:
        return False
    expected = schema.get("type")
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def ast_diff_paths(before: Any, after: Any, path: AstPath = ()) -> tuple[AstPath, ...]:
    """Return the minimal deterministic AST paths whose values differ."""
    if type(before) is not type(after):
        return (path,)
    if isinstance(before, dict):
        paths: list[AstPath] = []
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                paths.append((*path, key))
            else:
                paths.extend(ast_diff_paths(before[key], after[key], (*path, key)))
        return tuple(paths)
    if isinstance(before, list):
        if len(before) != len(after):
            return (path,)
        paths = []
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            paths.extend(ast_diff_paths(left, right, (*path, index)))
        return tuple(paths)
    return () if before == after else (path,)


def _argument(
    arguments: tuple[BoundArgumentV1, ...], slot_id: str, expected: type[OperatorRef]
) -> OperatorRef:
    value = next(
        argument.value for argument in arguments if argument.slot_id == slot_id
    )
    if not isinstance(value, expected):
        raise OperatorRejectedError("ref.type_incompatible", "argument.type")
    return value


def _resolve_node(
    context: OpenUILocalOperatorContextV1,
    state: OperatorStateV1,
    ref: OperatorRef,
    bindings: Mapping[str, Any],
) -> tuple[ReferenceDescriptorV1, NodeLocationV1, Mapping[str, Any]]:
    descriptor, payload = context.resolve(ref, state, RefKind.NODE)
    if not isinstance(payload, NodeLocationV1):
        raise OperatorRejectedError("ref.type_incompatible", "reference.payload")
    node = _at(bindings, payload.path)
    if not _element(node):
        raise OperatorRejectedError("ref.stale_state", "node.element")
    return descriptor, payload, node


def _resolve_role(
    context: OpenUILocalOperatorContextV1,
    state: OperatorStateV1,
    ref: OperatorRef,
    node_descriptor: ReferenceDescriptorV1,
) -> RoleLocationV1:
    _, payload = context.resolve(ref, state, RefKind.ROLE)
    if not isinstance(payload, RoleLocationV1):
        raise OperatorRejectedError("ref.type_incompatible", "reference.payload")
    if payload.node_fingerprint != node_descriptor.semantic_fingerprint:
        raise OperatorRejectedError("local.role_mismatch", "role.owner")
    return payload


def _resolve_literal(
    context: OpenUILocalOperatorContextV1,
    state: OperatorStateV1,
    ref: OperatorRef,
    kind: RefKind,
) -> Any:
    _, payload = context.resolve(ref, state, kind)
    if not isinstance(payload, LiteralValueV1):
        raise OperatorRejectedError("ref.type_incompatible", "reference.payload")
    return copy.deepcopy(payload.value)


def _effect_digest(value: Any) -> str:
    return _value_fingerprint(value)


def _mutation(
    bindings: dict[str, Any], effect: ActionEffectV1, before_source: str
) -> OperatorMutationV1:
    source = emit_statement_bindings(bindings, dsl="openui")
    if source == before_source:
        raise OperatorRejectedError(
            "local.unsupported_pack_semantics", "canonical.rewrite_visible"
        )
    return OperatorMutationV1(source=source, effect=effect)


def _declarations() -> tuple[AstOperatorV1, ...]:
    node = lambda slot: OperatorArgumentSlotV1(  # noqa: E731
        slot, RefKind.NODE, BindingPhase.STATE
    )
    role = lambda slot: OperatorArgumentSlotV1(  # noqa: E731
        slot, RefKind.ROLE, BindingPhase.STATE
    )
    template = lambda slot: OperatorArgumentSlotV1(  # noqa: E731
        slot, RefKind.TEMPLATE, BindingPhase.APPLICATION
    )
    value = lambda slot: OperatorArgumentSlotV1(  # noqa: E731
        slot, RefKind.VALUE, BindingPhase.APPLICATION
    )
    index = lambda slot: OperatorArgumentSlotV1(  # noqa: E731
        slot, RefKind.INDEX, BindingPhase.STATE, required=False
    )
    common = {
        "version": "v1",
        "domain": "openui.ast",
        "codomain": "openui.ast",
        "cost": 1.0,
    }
    return (
        AstOperatorV1(
            operator_id=ADD_CHILD,
            argument_slots=(
                node("parent"),
                role("role"),
                template("child"),
                index("index"),
            ),
            preconditions=(
                PreconditionV1("schema.child_role", ("parent", "role", "child")),
            ),
            effect_signature=(EffectDeltaKind.CARDINALITY, EffectDeltaKind.TOPOLOGY),
            locality="parent.role",
            inverse_operator_id=REMOVE_NODE,
            **common,
        ),
        AstOperatorV1(
            operator_id=REMOVE_NODE,
            argument_slots=(node("node"),),
            preconditions=(PreconditionV1("node.non_root", ("node",)),),
            effect_signature=(EffectDeltaKind.CARDINALITY, EffectDeltaKind.TOPOLOGY),
            locality="parent.role",
            inverse_operator_id=ADD_CHILD,
            **common,
        ),
        AstOperatorV1(
            operator_id=REPLACE_NODE,
            argument_slots=(node("node"), template("replacement")),
            preconditions=(
                PreconditionV1(
                    "schema.compatible_replacement", ("node", "replacement")
                ),
            ),
            effect_signature=(EffectDeltaKind.TOPOLOGY,),
            locality="node",
            idempotent=True,
            **common,
        ),
        AstOperatorV1(
            operator_id=SET_PROPERTY,
            argument_slots=(node("node"), role("role"), value("value")),
            preconditions=(
                PreconditionV1("schema.property", ("node", "role", "value")),
            ),
            effect_signature=(EffectDeltaKind.PROPERTY,),
            locality="node.property",
            inverse_operator_id=UNSET_PROPERTY,
            idempotent=True,
            **common,
        ),
        AstOperatorV1(
            operator_id=UNSET_PROPERTY,
            argument_slots=(node("node"), role("role")),
            preconditions=(
                PreconditionV1("schema.optional_property", ("node", "role")),
            ),
            effect_signature=(EffectDeltaKind.PROPERTY,),
            locality="node.property",
            inverse_operator_id=SET_PROPERTY,
            idempotent=True,
            **common,
        ),
        AstOperatorV1(
            operator_id=REORDER_CHILDREN,
            argument_slots=(node("parent"), role("role"), value("order")),
            preconditions=(
                PreconditionV1("schema.child_order", ("parent", "role", "order")),
            ),
            effect_signature=(EffectDeltaKind.TOPOLOGY,),
            locality="parent.role",
            inverse_operator_id=REORDER_CHILDREN,
            **common,
        ),
    )


def _executor(operator_id: str, context: OpenUILocalOperatorContextV1):
    def execute(
        state: OperatorStateV1, arguments: tuple[BoundArgumentV1, ...]
    ) -> OperatorMutationV1:
        bindings = parse_statement_bindings(state.source, dsl="openui")
        if operator_id == ADD_CHILD:
            parent_ref = _argument(arguments, "parent", NodeRef)
            role_ref = _argument(arguments, "role", RoleRef)
            child_ref = _argument(arguments, "child", TemplateRef)
            parent_desc, _, parent = _resolve_node(context, state, parent_ref, bindings)
            role = _resolve_role(context, state, role_ref, parent_desc)
            prop_schema = _property_schema(
                context.schema_defs, parent, role.property_name
            )
            children = dict(parent.get("props") or {}).get(role.property_name)
            if prop_schema.get("type") != "array" or not isinstance(children, list):
                raise OperatorRejectedError(
                    "local.child_role_required", "schema.child_role"
                )
            child = _resolve_literal(context, state, child_ref, RefKind.TEMPLATE)
            if not _element(child) or not _allowed_component(
                prop_schema, str(child.get("typeName") or "")
            ):
                raise OperatorRejectedError(
                    "local.incompatible_replacement", "schema.child_role"
                )
            position = len(children)
            index_arg = next(
                (item.value for item in arguments if item.slot_id == "index"), None
            )
            if index_arg is not None:
                child_fingerprints = tuple(
                    _value_fingerprint(item) for item in children
                )
                order_digest = ordered_parent_digest(
                    parent_desc.semantic_fingerprint, child_fingerprints
                )
                index_desc, index_payload = context.resolve(
                    index_arg,
                    state,
                    RefKind.INDEX,
                    current_parent_order_digest=order_digest,
                )
                if (
                    not isinstance(index_payload, IndexLocationV1)
                    or index_payload.node_fingerprint
                    != parent_desc.semantic_fingerprint
                    or index_payload.property_name != role.property_name
                ):
                    raise OperatorRejectedError(
                        "local.index_role_mismatch", "index.owner"
                    )
                position = (
                    index_desc.position if index_desc.position is not None else -1
                )
            if position < 0 or position > len(children):
                raise OperatorRejectedError("local.invalid_index", "index.range")
            before_count = len(children)
            children.insert(position, child)
            return _mutation(
                bindings,
                ActionEffectV1(
                    cardinality_deltas=(
                        EffectDeltaV1(
                            EffectDeltaKind.CARDINALITY,
                            role_ref,
                            before_count,
                            len(children),
                        ),
                    ),
                    topology_deltas=(
                        EffectDeltaV1(
                            EffectDeltaKind.TOPOLOGY,
                            parent_ref,
                            _effect_digest(("insert", position, before_count)),
                            _effect_digest(("insert", position, len(children))),
                        ),
                    ),
                    compiler_coverage=CompilerCoverage.EXACT,
                    estimated_completion_cost=1.0,
                ),
                state.source,
            )

        if operator_id == REMOVE_NODE:
            node_ref = _argument(arguments, "node", NodeRef)
            _, location, _ = _resolve_node(context, state, node_ref, bindings)
            if location.path == ("root",):
                raise OperatorRejectedError("local.root_deletion", "node.non_root")
            if len(location.path) == 1:
                raise OperatorRejectedError(
                    "local.unsupported_pack_semantics", "binding.reference_graph"
                )
            parent, key = _parent(bindings, location.path)
            if isinstance(parent, list) and isinstance(key, int):
                before_count = len(parent)
                parent.pop(key)
                return _mutation(
                    bindings,
                    ActionEffectV1(
                        cardinality_deltas=(
                            EffectDeltaV1(
                                EffectDeltaKind.CARDINALITY,
                                node_ref,
                                before_count,
                                len(parent),
                            ),
                        ),
                        topology_deltas=(
                            EffectDeltaV1(
                                EffectDeltaKind.TOPOLOGY,
                                node_ref,
                                _effect_digest(("present", key)),
                                _effect_digest(("removed", key)),
                            ),
                        ),
                        compiler_coverage=CompilerCoverage.EXACT,
                        estimated_completion_cost=1.0,
                    ),
                    state.source,
                )
            raise OperatorRejectedError(
                "local.unsupported_pack_semantics", "node.parent_role"
            )

        if operator_id == REPLACE_NODE:
            node_ref = _argument(arguments, "node", NodeRef)
            replacement_ref = _argument(arguments, "replacement", TemplateRef)
            _, location, old_node = _resolve_node(context, state, node_ref, bindings)
            replacement = _resolve_literal(
                context, state, replacement_ref, RefKind.TEMPLATE
            )
            if not _element(replacement):
                raise OperatorRejectedError(
                    "local.incompatible_replacement", "schema.element"
                )
            if len(location.path) > 1:
                parent, key = _parent(bindings, location.path)
                if isinstance(parent, list) and isinstance(key, int):
                    owner_path = location.path[:-3]
                    role_name = str(location.path[-2])
                    owner = _at(bindings, owner_path)
                    prop_schema = _property_schema(
                        context.schema_defs, owner, role_name
                    )
                    if not _allowed_component(
                        prop_schema, str(replacement.get("typeName") or "")
                    ):
                        raise OperatorRejectedError(
                            "local.incompatible_replacement",
                            "schema.compatible_replacement",
                        )
            _replace_at(bindings, location.path, replacement)
            return _mutation(
                bindings,
                ActionEffectV1(
                    topology_deltas=(
                        EffectDeltaV1(
                            EffectDeltaKind.TOPOLOGY,
                            node_ref,
                            _effect_digest(old_node),
                            _effect_digest(replacement),
                        ),
                    ),
                    compiler_coverage=CompilerCoverage.EXACT,
                    estimated_completion_cost=1.0,
                ),
                state.source,
            )

        if operator_id in {SET_PROPERTY, UNSET_PROPERTY}:
            node_ref = _argument(arguments, "node", NodeRef)
            role_ref = _argument(arguments, "role", RoleRef)
            node_desc, _, node = _resolve_node(context, state, node_ref, bindings)
            role = _resolve_role(context, state, role_ref, node_desc)
            prop_schema = _property_schema(
                context.schema_defs, node, role.property_name
            )
            props = node.setdefault("props", {})
            before = props.get(role.property_name)
            ordered_properties = _ordered_properties(context.schema_defs, node)
            role_index = ordered_properties.index(role.property_name)
            if operator_id == UNSET_PROPERTY:
                if _required_property(context.schema_defs, node, role.property_name):
                    raise OperatorRejectedError(
                        "local.required_property_removal", "schema.optional_property"
                    )
                if role.property_name not in props:
                    raise OperatorRejectedError(
                        "local.property_missing", "property.present"
                    )
                if any(name in props for name in ordered_properties[role_index + 1 :]):
                    raise OperatorRejectedError(
                        "local.unsupported_pack_semantics",
                        "canonical.positional_property",
                    )
                del props[role.property_name]
                after = None
            else:
                value_ref = _argument(arguments, "value", ValueRef)
                after = _resolve_literal(context, state, value_ref, RefKind.VALUE)
                if not _matches_scalar_schema(after, prop_schema):
                    raise OperatorRejectedError(
                        "local.property_value_invalid", "schema.property"
                    )
                if before == after:
                    raise OperatorRejectedError("local.no_change", "property.changed")
                if any(
                    name not in props
                    and not _required_property(context.schema_defs, node, name)
                    for name in ordered_properties[:role_index]
                ):
                    raise OperatorRejectedError(
                        "local.unsupported_pack_semantics",
                        "canonical.positional_property",
                    )
                props[role.property_name] = after
            return _mutation(
                bindings,
                ActionEffectV1(
                    property_deltas=(
                        EffectDeltaV1(
                            EffectDeltaKind.PROPERTY,
                            role_ref,
                            _effect_digest(before),
                            _effect_digest(after),
                        ),
                    ),
                    compiler_coverage=CompilerCoverage.EXACT,
                    estimated_completion_cost=1.0,
                ),
                state.source,
            )

        if operator_id == REORDER_CHILDREN:
            parent_ref = _argument(arguments, "parent", NodeRef)
            role_ref = _argument(arguments, "role", RoleRef)
            order_ref = _argument(arguments, "order", ValueRef)
            parent_desc, _, parent = _resolve_node(context, state, parent_ref, bindings)
            role = _resolve_role(context, state, role_ref, parent_desc)
            prop_schema = _property_schema(
                context.schema_defs, parent, role.property_name
            )
            children = dict(parent.get("props") or {}).get(role.property_name)
            if prop_schema.get("type") != "array" or not isinstance(children, list):
                raise OperatorRejectedError(
                    "local.child_role_required", "schema.child_order"
                )
            order = _resolve_literal(context, state, order_ref, RefKind.VALUE)
            if (
                not isinstance(order, (tuple, list))
                or any(not isinstance(index, int) for index in order)
                or sorted(order) != list(range(len(children)))
            ):
                raise OperatorRejectedError("local.invalid_order", "order.permutation")
            if list(order) == list(range(len(children))):
                raise OperatorRejectedError("local.no_change", "order.changed")
            before = tuple(_value_fingerprint(child) for child in children)
            children[:] = [children[index] for index in order]
            after = tuple(_value_fingerprint(child) for child in children)
            return _mutation(
                bindings,
                ActionEffectV1(
                    topology_deltas=(
                        EffectDeltaV1(
                            EffectDeltaKind.TOPOLOGY,
                            parent_ref,
                            _effect_digest(before),
                            _effect_digest(after),
                        ),
                    ),
                    compiler_coverage=CompilerCoverage.EXACT,
                    estimated_completion_cost=1.0,
                ),
                state.source,
            )

        raise OperatorRejectedError("operator.unsupported")

    return execute


def build_openui_local_operator_context(
    pack: DslPack,
    state: OperatorStateV1,
    *,
    request_id: str,
    branch_digest: str,
    seed: int,
    templates: tuple[Any, ...] = (),
    values: tuple[Any, ...] = (),
) -> OpenUILocalOperatorContextV1:
    """Compile one state into opaque typed refs and private executable payloads."""
    if pack.pack_id != "openui":
        raise OperatorRejectedError("local.unsupported_pack_semantics", "pack.openui")
    if state.pack_id != pack.pack_id:
        raise OperatorRejectedError("ref.type_incompatible", "state.pack")
    if OperatorStateV1.from_source(pack, state.source) != state:
        raise OperatorRejectedError("ref.stale_state", "state.identity")
    bindings = parse_statement_bindings(state.source, dsl="openui")
    schema = pack.backend.library_schema()
    schema_defs = schema.get("$defs")
    if not isinstance(schema_defs, Mapping):
        raise OperatorRejectedError(
            "local.unsupported_pack_semantics", "schema.definitions"
        )

    descriptors: list[ReferenceDescriptorV1] = []
    payloads: dict[str, object] = {}
    collision_counts: dict[tuple[str | None, str], int] = {}
    node_fingerprints: dict[AstPath, str] = {}
    node_descriptors: list[tuple[ReferenceDescriptorV1, NodeLocationV1, Any]] = []
    for binding in sorted(bindings):
        for path, node in _walk_nodes(bindings[binding], (binding,)):
            structure_fp = _value_fingerprint(node)
            parent_candidates = [
                (candidate, fingerprint)
                for candidate, fingerprint in node_fingerprints.items()
                if len(candidate) < len(path) and path[: len(candidate)] == candidate
            ]
            parent_fp = (
                max(parent_candidates, key=lambda item: len(item[0]))[1]
                if parent_candidates
                else None
            )
            collision_key = (parent_fp, structure_fp)
            collision_index = collision_counts.get(collision_key, 0)
            collision_counts[collision_key] = collision_index + 1
            disambiguator = branch_local_disambiguator(
                branch_digest, structure_fp, collision_index
            )
            node_fp = persistent_node_fingerprint(
                _semantic_structure(node),
                parent_fingerprint=parent_fp,
                branch_disambiguator=disambiguator,
            )
            node_fingerprints[path] = node_fp
            descriptor = ReferenceDescriptorV1(
                RefKind.NODE,
                node_fp,
                "openui.element",
                (CompilerFact.NODE_VISIBLE, CompilerFact.NODE_MUTABLE),
                parent_fingerprint=parent_fp,
            )
            location = NodeLocationV1(path)
            descriptors.append(descriptor)
            payloads[descriptor.fingerprint] = location
            node_descriptors.append((descriptor, location, node))

    for node_descriptor, _, node in node_descriptors:
        component = _component_schema(schema_defs, node)
        properties = component.get("properties", {})
        if not isinstance(properties, Mapping):
            continue
        for property_name in sorted(properties):
            semantic_fp = _fingerprint(
                {
                    "schema": "openui_role/v1",
                    "node": node_descriptor.semantic_fingerprint,
                    "property": property_name,
                }
            )
            descriptor = ReferenceDescriptorV1(
                RefKind.ROLE,
                semantic_fp,
                "openui.property_role",
                (CompilerFact.ROLE_AVAILABLE,),
                parent_fingerprint=node_descriptor.semantic_fingerprint,
            )
            descriptors.append(descriptor)
            payloads[descriptor.fingerprint] = RoleLocationV1(
                node_descriptor.semantic_fingerprint, property_name
            )
        for property_name, children in dict(node.get("props") or {}).items():
            if not isinstance(children, list):
                continue
            child_fingerprints = tuple(_value_fingerprint(child) for child in children)
            order_digest = ordered_parent_digest(
                node_descriptor.semantic_fingerprint, child_fingerprints
            )
            for position in range(len(children) + 1):
                descriptor = ReferenceDescriptorV1(
                    RefKind.INDEX,
                    _fingerprint(
                        {
                            "schema": "openui_index/v1",
                            "parent": node_descriptor.semantic_fingerprint,
                            "role": property_name,
                            "order": order_digest,
                            "position": position,
                        }
                    ),
                    "openui.insertion_index",
                    (CompilerFact.INDEX_ORDERED_PARENT,),
                    parent_fingerprint=node_descriptor.semantic_fingerprint,
                    parent_order_digest=order_digest,
                    position=position,
                )
                descriptors.append(descriptor)
                payloads[descriptor.fingerprint] = IndexLocationV1(
                    node_descriptor.semantic_fingerprint,
                    property_name,
                    position,
                )

    for kind, items, value_type, fact in (
        (
            RefKind.TEMPLATE,
            templates,
            "openui.template",
            CompilerFact.TEMPLATE_AVAILABLE,
        ),
        (RefKind.VALUE, values, "openui.value", CompilerFact.VALUE_VISIBLE),
    ):
        for item in items:
            descriptor = ReferenceDescriptorV1(
                kind,
                _fingerprint(
                    {
                        "schema": f"openui_{kind.value}/v1",
                        "value": _semantic_structure(item),
                    }
                ),
                value_type,
                (fact,),
            )
            if descriptor.fingerprint in payloads:
                continue
            descriptors.append(descriptor)
            payloads[descriptor.fingerprint] = LiteralValueV1(copy.deepcopy(item))

    reference_table = build_reference_table(
        request_id=request_id,
        state_digest=state.state_digest,
        branch_digest=branch_digest,
        descriptors=tuple(descriptors),
        seed=seed,
    )
    return OpenUILocalOperatorContextV1(
        reference_table=reference_table,
        branch_digest=branch_digest,
        schema_defs=schema_defs,
        _payloads=payloads,
    )


def build_openui_local_operator_library(
    context: OpenUILocalOperatorContextV1,
) -> OperatorLibraryV1:
    return OperatorLibraryV1(
        tuple(
            RegisteredOperatorV1(
                declaration, _executor(declaration.operator_id, context)
            )
            for declaration in _declarations()
        )
    )


__all__ = [
    "ADD_CHILD",
    "REMOVE_NODE",
    "REORDER_CHILDREN",
    "REPLACE_NODE",
    "SET_PROPERTY",
    "UNSET_PROPERTY",
    "IndexLocationV1",
    "LiteralValueV1",
    "NodeLocationV1",
    "OpenUILocalOperatorContextV1",
    "RoleLocationV1",
    "ast_diff_paths",
    "build_openui_local_operator_context",
    "build_openui_local_operator_library",
]
