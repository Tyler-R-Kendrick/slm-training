from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Callable

import pytest

from slm_training.dsl.operators import (
    CONTRACT_SUBTREE,
    DUPLICATE_SUBTREE,
    EXPAND_TEMPLATE,
    MOVE_NODE,
    REPARENT_NODE,
    UNWRAP_NODE,
    WRAP_NODE,
    ApplicationProvenanceV1,
    BoundArgumentV1,
    OpenUITemplateAliasV1,
    OperatorRejectedError,
    OperatorStateV1,
    RefKind,
    branch_fingerprint,
    build_openui_topology_operator_context,
    build_openui_topology_operator_library,
)
from slm_training.dsl.operators.contracts import OperatorRef
from slm_training.dsl.operators.local import (
    IndexLocationV1,
    NodeLocationV1,
    RoleLocationV1,
)
from slm_training.dsl.pack import get_pack
from slm_training.dsl.production_codec import (
    emit_statement_bindings,
    parse_statement_bindings,
)

MOVE_SOURCE = 'root = Card([Stack([TextContent(":source.value")]), Stack([])], "clear")'
WRAPPER = {
    "type": "element",
    "typeName": "Stack",
    "props": {"children": [], "direction": "column"},
}
COMPACT = {
    "type": "element",
    "typeName": "TextContent",
    "props": {"text": ":compact.value"},
}
EXPANDED = {
    "type": "element",
    "typeName": "Card",
    "props": {
        "children": [
            {
                "type": "element",
                "typeName": "TextContent",
                "props": {"text": ":expanded.value"},
            }
        ]
    },
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _alias(
    expanded=EXPANDED,
    contracted=COMPACT,
    *,
    child_role: str | None = None,
    digest: str = "c" * 64,
) -> OpenUITemplateAliasV1:
    return OpenUITemplateAliasV1(
        pack_id="openui",
        expanded=expanded,
        contracted=contracted,
        source_artifact_digest=digest,
        child_role=child_role,
    )


def _provenance(source: str) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.topology_operator_compiler",
        compiler_version="v1",
        source_artifact_digest=_sha(source),
        request_id="request-1",
    )


def _fixture(
    source: str = MOVE_SOURCE,
    *,
    aliases: tuple[OpenUITemplateAliasV1, ...] = (),
    values: tuple[object, ...] = (),
    seed: int = 11,
):
    base = get_pack("openui")
    state = OperatorStateV1.from_source(base, source)
    branch = branch_fingerprint(state.state_digest, "b" * 64)
    context = build_openui_topology_operator_context(
        base,
        state,
        request_id="request-1",
        branch_digest=branch,
        seed=seed,
        template_aliases=aliases,
        values=values,
    )
    library = build_openui_topology_operator_library(context)
    return replace(base, operator_library=library), state, context, library


def _ref(context, kind: RefKind, matches: Callable[[object], bool]) -> OperatorRef:
    for ref in context.local.references(kind):
        if matches(context.local.payload(ref)):
            return ref
    raise AssertionError(f"missing {kind.value} reference")


def _node(context, path: tuple[str | int, ...]) -> OperatorRef:
    return _ref(
        context,
        RefKind.NODE,
        lambda payload: isinstance(payload, NodeLocationV1) and payload.path == path,
    )


def _descriptor(context, ref: OperatorRef):
    return next(
        entry.descriptor
        for entry in context.local.reference_table.entries
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


def _index(context, node_ref: OperatorRef, role: str, position: int) -> OperatorRef:
    owner = _descriptor(context, node_ref).semantic_fingerprint
    return _ref(
        context,
        RefKind.INDEX,
        lambda payload: (
            isinstance(payload, IndexLocationV1)
            and payload.node_fingerprint == owner
            and payload.property_name == role
            and payload.position == position
        ),
    )


def _template(
    context, state: OperatorStateV1, alias: OpenUITemplateAliasV1
) -> OperatorRef:
    for ref in context.local.references(RefKind.TEMPLATE):
        try:
            resolved, _ = context.template_alias(ref, state)
        except Exception:  # noqa: BLE001 - candidate search only
            continue
        if resolved.fingerprint == alias.fingerprint:
            return ref
    raise AssertionError("missing template alias")


def _apply(library, pack, state, operator_id: str, *bindings):
    return library.apply(
        pack,
        state,
        operator_id,
        tuple(BoundArgumentV1(slot, ref) for slot, ref in bindings),
        _provenance(state.source),
    )


@pytest.mark.parametrize("operator_id", [MOVE_NODE, REPARENT_NODE])
def test_move_and_reparent_are_exact_aliases_and_inverse_capable(
    operator_id: str,
) -> None:
    pack, state, context, library = _fixture()
    source_node = _node(
        context, ("root", "props", "children", 0, "props", "children", 0)
    )
    destination = _node(context, ("root", "props", "children", 1))
    result = _apply(
        library,
        pack,
        state,
        operator_id,
        ("node", source_node),
        ("new_parent", destination),
        ("role", _role(context, destination, "children")),
        ("index", _index(context, destination, "children", 0)),
    )
    assert result.succeeded and result.state is not None
    assert pack.oracle(result.state.source).ok
    bindings = parse_statement_bindings(result.state.source)
    stacks = bindings["root"]["props"]["children"]
    assert stacks[0]["props"]["children"] == []
    assert stacks[1]["props"]["children"][0]["props"]["text"] == ":source.value"

    pack2, state2, context2, library2 = _fixture(result.state.source)
    moved = _node(context2, ("root", "props", "children", 1, "props", "children", 0))
    original_parent = _node(context2, ("root", "props", "children", 0))
    restored = _apply(
        library2,
        pack2,
        state2,
        operator_id,
        ("node", moved),
        ("new_parent", original_parent),
        ("role", _role(context2, original_parent, "children")),
        ("index", _index(context2, original_parent, "children", 0)),
    )
    assert restored.state is not None
    assert restored.state.source == state.source


def test_duplicate_subtree_preserves_source_and_uses_destination_index() -> None:
    pack, state, context, library = _fixture()
    source_node = _node(
        context, ("root", "props", "children", 0, "props", "children", 0)
    )
    destination = _node(context, ("root", "props", "children", 1))
    result = _apply(
        library,
        pack,
        state,
        DUPLICATE_SUBTREE,
        ("node", source_node),
        ("new_parent", destination),
        ("role", _role(context, destination, "children")),
        ("index", _index(context, destination, "children", 0)),
    )
    assert result.state is not None
    stacks = parse_statement_bindings(result.state.source)["root"]["props"]["children"]
    assert stacks[0]["props"]["children"][0]["props"]["text"] == ":source.value"
    assert stacks[1]["props"]["children"][0]["props"]["text"] == ":source.value"


def test_same_parent_move_uses_before_state_boundaries_and_rejects_noop() -> None:
    source = (
        'root = Card([TextContent(":item.one"), TextContent(":item.two"), '
        'TextContent(":item.three")])'
    )
    pack, state, context, library = _fixture(source)
    root = _node(context, ("root",))
    first = _node(context, ("root", "props", "children", 0))
    moved = _apply(
        library,
        pack,
        state,
        MOVE_NODE,
        ("node", first),
        ("new_parent", root),
        ("role", _role(context, root, "children")),
        ("index", _index(context, root, "children", 3)),
    )
    assert moved.state is not None
    children = parse_statement_bindings(moved.state.source)["root"]["props"]["children"]
    assert [child["props"]["text"] for child in children] == [
        ":item.two",
        ":item.three",
        ":item.one",
    ]

    noop = _apply(
        library,
        pack,
        state,
        MOVE_NODE,
        ("node", first),
        ("new_parent", root),
        ("role", _role(context, root, "children")),
        ("index", _index(context, root, "children", 1)),
    )
    assert noop.application.rejection.code == "topology.no_change"  # type: ignore[union-attr]


def test_cycle_and_capture_sensitive_moves_fail_closed() -> None:
    cycle_source = "root = Card([Stack([Stack([])])])"
    pack, state, context, library = _fixture(cycle_source)
    outer = _node(context, ("root", "props", "children", 0))
    inner = _node(context, ("root", "props", "children", 0, "props", "children", 0))
    cycle = _apply(
        library,
        pack,
        state,
        MOVE_NODE,
        ("node", outer),
        ("new_parent", inner),
        ("role", _role(context, inner, "children")),
    )
    assert cycle.application.rejection.code == "topology.cycle"  # type: ignore[union-attr]

    capture_source = (
        'root = Card([Stack([text]), Stack([])])\ntext = TextContent(":source.value")'
    )
    pack2, state2, context2, library2 = _fixture(capture_source)
    captured = _node(context2, ("root", "props", "children", 0))
    destination = _node(context2, ("root", "props", "children", 1))
    capture = _apply(
        library2,
        pack2,
        state2,
        MOVE_NODE,
        ("node", captured),
        ("new_parent", destination),
        ("role", _role(context2, destination, "children")),
    )
    assert capture.application.rejection.code == "topology.capture_unsupported"  # type: ignore[union-attr]


def test_wrap_then_unwrap_restores_canonical_identity() -> None:
    alias = _alias(
        expanded=WRAPPER,
        contracted=COMPACT,
        child_role="children",
        digest="d" * 64,
    )
    source = 'root = TextContent(":source.value")'
    pack, state, context, library = _fixture(source, aliases=(alias,))
    root = _node(context, ("root",))
    wrapped = _apply(
        library,
        pack,
        state,
        WRAP_NODE,
        ("node", root),
        ("template", _template(context, state, alias)),
    )
    assert wrapped.succeeded and wrapped.state is not None
    assert 'Stack([TextContent(":source.value")], "column")' in wrapped.state.source
    effect = wrapped.application.effect
    assert effect is not None
    assert (
        effect.topology_deltas[0].after["template"]["template_fingerprint"]
        == alias.fingerprint
    )

    pack2, state2, context2, library2 = _fixture(wrapped.state.source, aliases=(alias,))
    root2 = _node(context2, ("root",))
    unwrapped = _apply(library2, pack2, state2, UNWRAP_NODE, ("node", root2))
    assert unwrapped.state is not None
    assert unwrapped.state.source == state.source


def test_unwrap_rejects_non_single_child_cardinality() -> None:
    source = 'root = Stack([TextContent(":one.value"), TextContent(":two.value")])'
    pack, state, context, library = _fixture(source)
    result = _apply(
        library,
        pack,
        state,
        UNWRAP_NODE,
        ("node", _node(context, ("root",))),
    )
    assert result.application.rejection.code == "topology.incompatible_cardinality"  # type: ignore[union-attr]


def test_expand_then_contract_uses_exact_alias_and_explicit_provenance() -> None:
    alias = _alias()
    source = 'root = TextContent(":compact.value")'
    pack, state, context, library = _fixture(source, aliases=(alias,))
    expanded = _apply(
        library,
        pack,
        state,
        EXPAND_TEMPLATE,
        ("node", _node(context, ("root",))),
        ("template", _template(context, state, alias)),
    )
    assert expanded.succeeded and expanded.state is not None
    assert "Card" in expanded.state.source
    assert expanded.state.source == pack.canonicalize(
        emit_statement_bindings({"root": EXPANDED}, dsl="openui")
    )
    delta = expanded.application.effect.topology_deltas[0]  # type: ignore[union-attr]
    assert delta.after["template"] == {
        **alias.evidence(),
        "operation": "expand",
    }

    pack2, state2, context2, library2 = _fixture(
        expanded.state.source, aliases=(alias,)
    )
    contracted = _apply(
        library2,
        pack2,
        state2,
        CONTRACT_SUBTREE,
        ("node", _node(context2, ("root",))),
        ("template", _template(context2, state2, alias)),
    )
    assert contracted.state is not None
    assert contracted.state.source == state.source


def test_template_mismatch_and_missing_capability_fail_stably() -> None:
    alias = _alias()
    source = 'root = TextContent(":other.value")'
    pack, state, context, library = _fixture(source, aliases=(alias,))
    mismatch = _apply(
        library,
        pack,
        state,
        EXPAND_TEMPLATE,
        ("node", _node(context, ("root",))),
        ("template", _template(context, state, alias)),
    )
    assert mismatch.application.rejection.code == "template.mismatch"  # type: ignore[union-attr]

    pack2, state2, _, library2 = _fixture(source)
    unsupported = library2.apply(
        pack2,
        state2,
        EXPAND_TEMPLATE,
        (),
        _provenance(state2.source),
    )
    assert unsupported.application.rejection.code == "operator.unsupported"  # type: ignore[union-attr]


def test_template_alias_is_immutable_and_input_defensively_copied() -> None:
    expanded = {
        "type": "element",
        "typeName": "Stack",
        "props": {"children": []},
    }
    alias = _alias(expanded=expanded)
    expanded["props"]["children"].append(COMPACT)
    assert alias.expanded_value["props"]["children"] == []
    with pytest.raises(TypeError):
        alias.expanded["typeName"] = "Card"  # type: ignore[index]


def test_missing_pack_and_template_capabilities_fail_closed() -> None:
    toy = get_pack("toy-layout")
    forged = OperatorStateV1(
        pack_id="toy-layout",
        source='root = text(":hero.title")',
        state_digest="e" * 64,
        ast_digest="f" * 64,
    )
    with pytest.raises(OperatorRejectedError) as pack_error:
        build_openui_topology_operator_context(
            toy,
            forged,
            request_id="request-1",
            branch_digest="a" * 64,
            seed=1,
        )
    assert pack_error.value.code == "local.unsupported_pack_semantics"

    base = get_pack("openui")
    state = OperatorStateV1.from_source(base, 'root = TextContent(":compact.value")')
    bad_pack_alias = OpenUITemplateAliasV1(
        pack_id="other",
        expanded=EXPANDED,
        contracted=COMPACT,
        source_artifact_digest="a" * 64,
    )
    with pytest.raises(OperatorRejectedError) as provenance_error:
        build_openui_topology_operator_context(
            base,
            state,
            request_id="request-1",
            branch_digest="b" * 64,
            seed=1,
            template_aliases=(bad_pack_alias,),
        )
    assert provenance_error.value.code == "template.provenance_invalid"


def test_declared_topology_costs_locality_and_inverse_inventory() -> None:
    _, _, _, library = _fixture(aliases=(_alias(),))
    declarations = {
        declaration.operator_id: declaration for declaration in library.declarations
    }
    assert declarations[MOVE_NODE].locality == "subtree.two_parents"
    assert declarations[MOVE_NODE].inverse_operator_id == MOVE_NODE
    assert declarations[WRAP_NODE].inverse_operator_id == UNWRAP_NODE
    assert declarations[EXPAND_TEMPLATE].inverse_operator_id == CONTRACT_SUBTREE
    assert declarations[DUPLICATE_SUBTREE].cost > declarations[UNWRAP_NODE].cost


def test_topology_library_composes_core_local_inventory() -> None:
    _, _, _, library = _fixture(aliases=(_alias(),))
    ids = {declaration.operator_id for declaration in library.declarations}
    assert {
        MOVE_NODE,
        REPARENT_NODE,
        WRAP_NODE,
        UNWRAP_NODE,
        DUPLICATE_SUBTREE,
        EXPAND_TEMPLATE,
        CONTRACT_SUBTREE,
        "openui.add_child",
        "openui.remove_node",
    } <= ids
    assert len(ids) == 13
