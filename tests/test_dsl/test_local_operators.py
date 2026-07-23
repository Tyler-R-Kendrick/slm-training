from __future__ import annotations

import hashlib
import itertools
from dataclasses import replace
from typing import Callable

import pytest

from slm_training.dsl.operators import (
    ADD_CHILD,
    REMOVE_NODE,
    REORDER_CHILDREN,
    REPLACE_NODE,
    SET_PROPERTY,
    UNSET_PROPERTY,
    ApplicationProvenanceV1,
    BoundArgumentV1,
    OperatorStateV1,
    RefKind,
    ast_diff_paths,
    branch_fingerprint,
    build_openui_local_operator_context,
    build_openui_local_operator_library,
)
from slm_training.dsl.operators.contracts import OperatorRef
from slm_training.dsl.operators.local import (
    IndexLocationV1,
    LiteralValueV1,
    NodeLocationV1,
    RoleLocationV1,
)
from slm_training.dsl.pack import get_pack
from slm_training.dsl.production_codec import parse_statement_bindings

SOURCE = 'root = Card([TextContent(":hero.title"), TextContent(":hero.body")], "clear")'
TITLE = {"type": "element", "typeName": "TextContent", "props": {"text": ":new.title"}}
BAD = {"type": "element", "typeName": "Bogus", "props": {}}


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
    templates: tuple[object, ...] = (TITLE, BAD),
    values: tuple[object, ...] = (
        "large",
        "clear",
        "column",
        "l",
        (1, 0),
        (0, 0),
    ),
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
        templates=templates,
        values=values,
    )
    library = build_openui_local_operator_library(context)
    return replace(base_pack, operator_library=library), state, context, library


def _ref(context, kind: RefKind, matches: Callable[[object], bool]) -> OperatorRef:
    for ref in context.references(kind):
        if matches(context.payload(ref)):
            return ref
    raise AssertionError(f"missing {kind.value} reference")


def _root(context) -> OperatorRef:
    return _ref(
        context,
        RefKind.NODE,
        lambda payload: (
            isinstance(payload, NodeLocationV1) and payload.path == ("root",)
        ),
    )


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


def _apply(library, pack, state, operator_id: str, *bindings):
    return library.apply(
        pack,
        state,
        operator_id,
        tuple(BoundArgumentV1(slot, ref) for slot, ref in bindings),
        _provenance(state.source),
    )


def test_core_inventory_is_small_exact_and_declared() -> None:
    _, _, _, library = _fixture()
    declarations = {item.operator_id: item for item in library.declarations}
    assert set(declarations) == {
        ADD_CHILD,
        REMOVE_NODE,
        REPLACE_NODE,
        SET_PROPERTY,
        UNSET_PROPERTY,
        REORDER_CHILDREN,
    }
    assert declarations[ADD_CHILD].inverse_operator_id == REMOVE_NODE
    assert declarations[SET_PROPERTY].inverse_operator_id == UNSET_PROPERTY
    assert all(item.cost == 1.0 for item in declarations.values())


def test_set_then_unset_optional_property_restores_canonical_identity() -> None:
    source = 'root = Card([TextContent(":hero.title")])'
    pack, state, context, library = _fixture(source, values=("clear",))
    root = _root(context)
    variant = _role(context, root, "variant")
    value = _literal(context, RefKind.VALUE, "clear")
    added = _apply(
        library,
        pack,
        state,
        SET_PROPERTY,
        ("node", root),
        ("role", variant),
        ("value", value),
    )
    assert added.succeeded and added.state is not None
    assert pack.oracle(added.state.source).ok

    pack2, state2, context2, library2 = _fixture(added.state.source, values=())
    root2 = _root(context2)
    removed = _apply(
        library2,
        pack2,
        state2,
        UNSET_PROPERTY,
        ("node", root2),
        ("role", _role(context2, root2, "variant")),
    )
    assert removed.succeeded and removed.state is not None
    assert removed.state.source == state.source


def test_add_then_remove_inline_child_restores_canonical_identity() -> None:
    source = 'root = Card([TextContent(":hero.body")])'
    pack, state, context, library = _fixture(source, templates=(TITLE,), values=())
    root = _root(context)
    added = _apply(
        library,
        pack,
        state,
        ADD_CHILD,
        ("parent", root),
        ("role", _role(context, root, "children")),
        ("child", _literal(context, RefKind.TEMPLATE, TITLE)),
        ("index", _index(context, root, "children", 1)),
    )
    assert added.succeeded and added.state is not None
    assert (
        len(parse_statement_bindings(added.state.source)["root"]["props"]["children"])
        == 2
    )

    pack2, state2, context2, library2 = _fixture(
        added.state.source, templates=(), values=()
    )
    child = _node_at(context2, ("root", "props", "children", 1))
    removed = _apply(library2, pack2, state2, REMOVE_NODE, ("node", child))
    assert removed.succeeded and removed.state is not None
    assert removed.state.source == state.source


def test_add_child_honors_state_bound_insertion_index() -> None:
    pack, state, context, library = _fixture(
        'root = Card([TextContent(":hero.body")])',
        templates=(TITLE,),
        values=(),
    )
    root = _root(context)
    result = _apply(
        library,
        pack,
        state,
        ADD_CHILD,
        ("parent", root),
        ("role", _role(context, root, "children")),
        ("child", _literal(context, RefKind.TEMPLATE, TITLE)),
        ("index", _index(context, root, "children", 0)),
    )
    assert result.state is not None
    children = parse_statement_bindings(result.state.source)["root"]["props"][
        "children"
    ]
    assert [child["props"]["text"] for child in children] == [
        ":new.title",
        ":hero.body",
    ]


@pytest.mark.parametrize(
    ("child_count", "position"),
    [
        (child_count, position)
        for child_count in range(3)
        for position in range(child_count + 1)
    ],
)
def test_add_child_exhausts_small_state_insertion_positions(
    child_count: int, position: int
) -> None:
    children = ", ".join(
        f'TextContent(":item.{name}")' for name in ("zero", "one")[:child_count]
    )
    source = f"root = Card([{children}])"
    pack, state, context, library = _fixture(source, templates=(TITLE,), values=())
    root = _root(context)
    result = _apply(
        library,
        pack,
        state,
        ADD_CHILD,
        ("parent", root),
        ("role", _role(context, root, "children")),
        ("child", _literal(context, RefKind.TEMPLATE, TITLE)),
        ("index", _index(context, root, "children", position)),
    )
    assert result.succeeded and result.state is not None
    actual = parse_statement_bindings(result.state.source)["root"]["props"]["children"]
    assert len(actual) == child_count + 1
    assert actual[position]["props"]["text"] == ":new.title"


@pytest.mark.parametrize(
    "order",
    [order for order in itertools.permutations(range(3)) if order != tuple(range(3))],
)
def test_reorder_exhausts_three_child_permutations(order: tuple[int, ...]) -> None:
    source = (
        'root = Card([TextContent(":item.zero"), TextContent(":item.one"), '
        'TextContent(":item.two")])'
    )
    pack, state, context, library = _fixture(source, values=(order,))
    root = _root(context)
    result = _apply(
        library,
        pack,
        state,
        REORDER_CHILDREN,
        ("parent", root),
        ("role", _role(context, root, "children")),
        ("order", _literal(context, RefKind.VALUE, order)),
    )
    assert result.succeeded and result.state is not None
    children = parse_statement_bindings(result.state.source)["root"]["props"][
        "children"
    ]
    assert [child["props"]["text"] for child in children] == [
        f":item.{('zero', 'one', 'two')[index]}" for index in order
    ]


def test_reorder_is_exact_and_invalid_order_has_stable_code() -> None:
    pack, state, context, library = _fixture()
    root = _root(context)
    role = _role(context, root, "children")
    order = _literal(context, RefKind.VALUE, (1, 0))
    result = _apply(
        library,
        pack,
        state,
        REORDER_CHILDREN,
        ("parent", root),
        ("role", role),
        ("order", order),
    )
    assert result.succeeded and result.state is not None
    children = parse_statement_bindings(result.state.source)["root"]["props"][
        "children"
    ]
    assert children[0]["props"]["text"] == ":hero.body"

    invalid = _apply(
        library,
        pack,
        state,
        REORDER_CHILDREN,
        ("parent", root),
        ("role", role),
        ("order", _literal(context, RefKind.VALUE, (0, 0))),
    )
    assert invalid.application.rejection is not None
    assert invalid.application.rejection.code == "local.invalid_order"


def test_root_required_property_and_incompatible_replacement_reject_stably() -> None:
    pack, state, context, library = _fixture()
    root = _root(context)
    root_delete = _apply(library, pack, state, REMOVE_NODE, ("node", root))
    required_unset = _apply(
        library,
        pack,
        state,
        UNSET_PROPERTY,
        ("node", root),
        ("role", _role(context, root, "children")),
    )
    child = _node_at(context, ("root", "props", "children", 0))
    incompatible = _apply(
        library,
        pack,
        state,
        REPLACE_NODE,
        ("node", child),
        ("replacement", _literal(context, RefKind.TEMPLATE, BAD)),
    )
    assert root_delete.application.rejection.code == "local.root_deletion"  # type: ignore[union-attr]
    assert (
        required_unset.application.rejection.code == "local.required_property_removal"
    )  # type: ignore[union-attr]
    assert incompatible.application.rejection.code == "local.incompatible_replacement"  # type: ignore[union-attr]


def test_compatible_replacement_is_pure_local_and_pack_valid() -> None:
    replacement = {
        "type": "element",
        "typeName": "CardHeader",
        "props": {"title": ":new.title"},
    }
    pack, state, context, library = _fixture(templates=(replacement,), values=())
    child = _node_at(context, ("root", "props", "children", 0))
    before = state
    dry = library.dry_run(
        pack,
        state,
        REPLACE_NODE,
        (
            BoundArgumentV1("node", child),
            BoundArgumentV1(
                "replacement",
                _literal(context, RefKind.TEMPLATE, replacement),
            ),
        ),
        _provenance(state.source),
    )
    result = _apply(
        library,
        pack,
        state,
        REPLACE_NODE,
        ("node", child),
        ("replacement", _literal(context, RefKind.TEMPLATE, replacement)),
    )
    assert result.succeeded and result.state is not None
    assert result.application.application_id == dry.application_id
    assert state == before
    assert pack.oracle(result.state.source).ok
    assert "CardHeader" in result.state.source


def test_stale_context_and_cross_node_role_reject_before_mutation() -> None:
    pack, state, context, library = _fixture()
    root = _root(context)
    child = _node_at(context, ("root", "props", "children", 0))
    wrong_role = _role(context, child, "size")
    mismatch = _apply(
        library,
        pack,
        state,
        SET_PROPERTY,
        ("node", root),
        ("role", wrong_role),
        ("value", _literal(context, RefKind.VALUE, "l")),
    )
    assert mismatch.application.rejection.code == "local.role_mismatch"  # type: ignore[union-attr]

    changed = _apply(
        library,
        pack,
        state,
        REORDER_CHILDREN,
        ("parent", root),
        ("role", _role(context, root, "children")),
        ("order", _literal(context, RefKind.VALUE, (1, 0))),
    )
    assert changed.state is not None
    stale = library.apply(
        pack,
        changed.state,
        SET_PROPERTY,
        (
            BoundArgumentV1("node", root),
            BoundArgumentV1("role", _role(context, root, "gap")),
            BoundArgumentV1("value", _literal(context, RefKind.VALUE, "l")),
        ),
        _provenance(changed.state.source),
    )
    assert stale.application.rejection.code == "ref.stale_state"  # type: ignore[union-attr]


def test_positional_property_holes_fail_closed_before_pack_authority() -> None:
    source = 'root = Card([TextContent(":hero.title")])'
    pack, state, context, library = _fixture(source, values=("column",))
    root = _root(context)
    result = _apply(
        library,
        pack,
        state,
        SET_PROPERTY,
        ("node", root),
        ("role", _role(context, root, "direction")),
        ("value", _literal(context, RefKind.VALUE, "column")),
    )
    assert result.application.rejection is not None
    assert result.application.rejection.code == "local.unsupported_pack_semantics"
    assert (
        result.application.rejection.failed_precondition
        == "canonical.positional_property"
    )


def test_declared_locality_matches_minimal_ast_diff() -> None:
    pack, state, context, library = _fixture()
    before = parse_statement_bindings(state.source)
    root = _root(context)
    result = _apply(
        library,
        pack,
        state,
        SET_PROPERTY,
        ("node", root),
        ("role", _role(context, root, "direction")),
        ("value", _literal(context, RefKind.VALUE, "column")),
    )
    assert result.state is not None
    after = parse_statement_bindings(result.state.source)
    assert ast_diff_paths(before, after) == (("root", "props", "direction"),)
    assert library.lookup(SET_PROPERTY).locality == "node.property"


def test_reference_permutation_does_not_change_application_result() -> None:
    left_pack, left_state, left_context, left_library = _fixture(seed=3)
    right_pack, right_state, right_context, right_library = _fixture(seed=19)
    left_root = _root(left_context)
    right_root = _root(right_context)
    left = _apply(
        left_library,
        left_pack,
        left_state,
        SET_PROPERTY,
        ("node", left_root),
        ("role", _role(left_context, left_root, "direction")),
        ("value", _literal(left_context, RefKind.VALUE, "column")),
    )
    right = _apply(
        right_library,
        right_pack,
        right_state,
        SET_PROPERTY,
        ("node", right_root),
        ("role", _role(right_context, right_root, "direction")),
        ("value", _literal(right_context, RefKind.VALUE, "column")),
    )
    assert left.state == right.state


def test_nested_node_identity_binds_structural_parent() -> None:
    source = (
        'root = Card([Stack([TextContent(":same.value")]), '
        'Stack([TextContent(":same.value")])])'
    )
    _, _, context, _ = _fixture(source, templates=(), values=())
    first_parent = _node_at(context, ("root", "props", "children", 0))
    second_parent = _node_at(context, ("root", "props", "children", 1))
    first_child = _node_at(
        context, ("root", "props", "children", 0, "props", "children", 0)
    )
    second_child = _node_at(
        context, ("root", "props", "children", 1, "props", "children", 0)
    )
    first_descriptor = _descriptor(context, first_child)
    second_descriptor = _descriptor(context, second_child)
    assert (
        first_descriptor.parent_fingerprint
        == _descriptor(context, first_parent).semantic_fingerprint
    )
    assert (
        second_descriptor.parent_fingerprint
        == _descriptor(context, second_parent).semantic_fingerprint
    )
    assert (
        first_descriptor.semantic_fingerprint != second_descriptor.semantic_fingerprint
    )


def test_compiler_private_literal_payload_is_defensively_copied() -> None:
    _, _, context, _ = _fixture(templates=(TITLE,), values=())
    template = _literal(context, RefKind.TEMPLATE, TITLE)
    exposed = context.payload(template)
    assert isinstance(exposed, LiteralValueV1)
    exposed.value["props"]["text"] = ":mutated.value"
    fresh = context.payload(template)
    assert isinstance(fresh, LiteralValueV1)
    assert fresh.value["props"]["text"] == ":new.title"


@pytest.mark.parametrize(
    "operator_id",
    [
        ADD_CHILD,
        REMOVE_NODE,
        REPLACE_NODE,
        SET_PROPERTY,
        UNSET_PROPERTY,
        REORDER_CHILDREN,
    ],
)
def test_every_operator_declaration_has_exact_compiler_coverage(
    operator_id: str,
) -> None:
    _, _, _, library = _fixture()
    declaration = library.lookup(operator_id)
    assert declaration.effect_signature
    assert declaration.locality in {"node", "node.property", "parent.role"}
