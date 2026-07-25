"""SLM-358 (DSH1-06): typed answer composition with scope-safe AST combinators."""

from __future__ import annotations

import ast
import hashlib
import inspect

import pytest

from slm_training.dsl.canonicalize import canonicalize
from slm_training.dsl.parser import validate
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.train_data import answer_composition as ac
from slm_training.harnesses.train_data.artifact_graph import (
    ArtifactGraphStore,
    ArtifactNodeV1,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

FIXTURE_A = (
    'root = Stack([header, cta], "column")\n'
    'header = TextContent(":title")\n'
    'cta = Button(":cta", "primary")\n'
)
FIXTURE_B = 'root = Card([body])\nbody = TextContent(":hero.body")'
FIXTURE_C = 'root = Stack([x], "row")\nx = TextContent(":a")'
FIXTURE_D = 'root = Stack([y], "row")\ny = TextContent(":b")'
FIXTURE_V05 = (
    'root = TextContent(":title")\n'
    'q0 = Query("tool", {arg: $x}, {default: []}, 15)'
)


@pytest.fixture(scope="module")
def answer_a():
    return ac.admit_answer(FIXTURE_A, root_family_id="fam-a")


@pytest.fixture(scope="module")
def answer_b():
    return ac.admit_answer(FIXTURE_B, root_family_id="fam-a")


def _assert_admitted(result: ac.CompositionResultV1) -> None:
    assert result.ok, result.rejection
    assert result.rejection is None
    assert result.composite_source and result.canonical_source
    # Every admitted composite parses, validates, and round-trips canonically.
    validate(result.composite_source)
    assert canonicalize(result.composite_source) == result.canonical_source
    assert canonicalize(result.canonical_source) == result.canonical_source
    assert result.after_ast_digest == hashlib.sha256(
        result.canonical_source.encode()
    ).hexdigest()
    assert result.before_ast_digest
    assert result.lineage_id
    assert result.combinator_version == "v1"
    assert result.proof and result.proof["official_parser"]


# --------------------------------------------------------------------------- #
# All five combinators on fixtures
# --------------------------------------------------------------------------- #


def test_append_statement_round_trips(answer_a):
    result = ac.apply_combinator(
        "APPEND_STATEMENT", answer_a,
        binder="footer", rhs=ac.parse_rhs('TextContent(":foot")'),
    )
    _assert_admitted(result)
    assert result.proof["appended_binder"] == "footer"
    assert "TextContent" in result.canonical_source


def test_insert_child_round_trips_and_preserves_existing_children(answer_a):
    result = ac.apply_combinator(
        "INSERT_CHILD", answer_a,
        target="root", child=ac.parse_rhs('TextContent(":foot")'),
    )
    _assert_admitted(result)
    assert result.proof["children_cardinality"] == 3
    # Original children are preserved, never dropped or reordered.
    assert result.canonical_source.startswith("root = Stack([v0, v1, ")


def test_make_list_round_trips(answer_a):
    result = ac.apply_combinator(
        "MAKE_LIST", answer_a, binder="items",
        items=(ac.parse_rhs('TextContent(":a")'), ac.parse_rhs('TextContent(":b")')),
    )
    _assert_admitted(result)
    assert result.proof["cardinality"] == 2
    assert "[TextContent" in result.canonical_source


def test_wrap_node_root_round_trips(answer_a):
    result = ac.apply_combinator(
        "WRAP_NODE", answer_a, target="root", wrapper_type="Card",
    )
    _assert_admitted(result)
    inner = result.proof["inner_binder"]
    assert inner != "root"
    assert "root = Card([" in result.composite_source
    # The original subtree moved intact under the fresh inner binder.
    assert "Stack([v0, v1]" in result.composite_source


def test_compose_document_round_trips(answer_a, answer_b):
    result = ac.apply_combinator("COMPOSE_DOCUMENT", (answer_a, answer_b))
    _assert_admitted(result)
    assert result.proof["root_arity"] == 2
    assert result.split == answer_a.split
    assert result.root_family_id == "fam-a"


# --------------------------------------------------------------------------- #
# Deterministic alpha-renaming with atomic reference update
# --------------------------------------------------------------------------- #


def test_binder_collision_alpha_renamed_deterministically_and_atomically():
    left = ac.admit_answer(FIXTURE_C)
    right = ac.admit_answer(FIXTURE_D)
    first = ac.apply_combinator("COMPOSE_DOCUMENT", (left, right))
    second = ac.apply_combinator("COMPOSE_DOCUMENT", (left, right))
    _assert_admitted(first)
    assert first.to_dict() == second.to_dict(), "renaming must be deterministic"
    rename_map = first.proof["alpha_rename_map"]
    # Both sources bind `root` and `v0`; the second collides and is renamed.
    assert rename_map["root"] != "root"
    assert rename_map["v0"] != "v0"
    composite_bindings = ac._parse_bindings(first.composite_source)
    merged = composite_bindings
    renamed_root = rename_map["root"]
    renamed_v0 = rename_map["v0"]
    assert renamed_root in merged and renamed_v0 in merged
    # Atomicity: the renamed root references the renamed binder, and no stale
    # reference to a colliding name survives anywhere in the merged scope.
    refs = {
        ref
        for name, rhs in merged.items()
        for ref in ac._expr_refs(rhs)
    }
    assert renamed_v0 in refs
    stale = {
        name
        for name in merged
        if name in rename_map.values()
        and any(
            ref in rename_map and rename_map[ref] != ref
            for ref in ac._expr_refs(merged[name])
            if ref in rename_map
        )
    }
    assert not stale
    # Lineage records both sources.
    assert set(first.source_answer_ids) == {left.answer_id, right.answer_id}


# --------------------------------------------------------------------------- #
# No string concatenation of program text (source audit)
# --------------------------------------------------------------------------- #


def test_module_never_concatenates_program_text():
    """Audit: composition operates on parsed ASTs, never spliced source text."""
    module_source = inspect.getsource(ac)
    tree = ast.parse(module_source)

    def references_program_text(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr in {
                "source", "surface", "program_text",
            }:
                return True
            if isinstance(sub, ast.Name) and sub.id in {
                "source", "left_source", "right_source", "program_text",
            }:
                return True
        return False

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if references_program_text(node.left) or references_program_text(node.right):
                violations.append(f"line {node.lineno}: '+' over program text")
        if isinstance(node, ast.JoinedStr) and references_program_text(node):
            violations.append(f"line {node.lineno}: f-string over program text")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and node.args
            and references_program_text(node.args[0])
        ):
            violations.append(f"line {node.lineno}: str.join over program text")
    assert not violations, violations


# --------------------------------------------------------------------------- #
# Fail-closed: incompatible roots / splits, illegal intermediates, contracts
# --------------------------------------------------------------------------- #


def _rejection(result: ac.CompositionResultV1, code: ac.CompositionCode) -> None:
    assert not result.ok
    assert result.composite_source is None
    assert result.after_ast_digest is None
    assert result.rejection is not None
    assert result.rejection.code == code
    assert result.rejection.reason


def test_incompatible_typed_root_fails_closed(answer_a):
    typed = ac.admit_answer(FIXTURE_V05)
    result = ac.apply_combinator("COMPOSE_DOCUMENT", (answer_a, typed))
    _rejection(result, ac.CompositionCode.INCOMPATIBLE_ROOTS)


def test_cross_split_composition_fails_closed(answer_a):
    other = ac.admit_answer(FIXTURE_B, split="test", root_family_id="fam-b")
    result = ac.apply_combinator("COMPOSE_DOCUMENT", (answer_a, other))
    _rejection(result, ac.CompositionCode.CROSS_SPLIT)
    assert "split" in result.rejection.reason


def test_insert_child_into_leaf_is_illegal_intermediate(answer_a):
    canonical_bindings = answer_a.bindings()
    leaf = next(
        name
        for name, rhs in canonical_bindings.items()
        if isinstance(rhs, dict)
        and rhs.get("type") == "element"
        and rhs.get("typeName") == "TextContent"
    )
    result = ac.apply_combinator(
        "INSERT_CHILD", answer_a, target=leaf,
        child=ac.parse_rhs('TextContent(":foot")'),
    )
    _rejection(result, ac.CompositionCode.ILLEGAL_INTERMEDIATE)
    assert "illegal intermediate" in result.rejection.reason


def test_wrap_node_silent_parent_meaning_change_rejected(answer_a):
    """Wrapping a leaf still referenced by root would change root's meaning."""
    result = ac.apply_combinator(
        "WRAP_NODE", answer_a, target="v0", wrapper_type="Card",
    )
    _rejection(result, ac.CompositionCode.ILLEGAL_INTERMEDIATE)
    assert "stop rule" in result.rejection.reason


def test_pre_and_postcondition_violations_fail_closed(answer_a):
    _rejection(
        ac.apply_combinator("MAKE_LIST", answer_a, binder="items", items=()),
        ac.CompositionCode.CARDINALITY,
    )
    _rejection(
        ac.apply_combinator(
            "APPEND_STATEMENT", answer_a, binder=":bad",
            rhs=ac.parse_rhs('TextContent(":x")'),
        ),
        ac.CompositionCode.PRECONDITION,
    )
    _rejection(
        ac.apply_combinator(
            "APPEND_STATEMENT", answer_a, binder="dangling",
            rhs=ac.parse_rhs("missing_binder"),
        ),
        ac.CompositionCode.UNRESOLVED_REF,
    )
    _rejection(
        ac.apply_combinator("COMPOSE_DOCUMENT", (answer_a,)),
        ac.CompositionCode.CARDINALITY,
    )
    _rejection(
        ac.apply_combinator(
            "INSERT_CHILD", answer_a, target="ghost",
            child=ac.parse_rhs('TextContent(":x")'),
        ),
        ac.CompositionCode.INVALID_TARGET,
    )
    with pytest.raises(KeyError, match="unknown combinator"):
        ac.apply_combinator("SPLICE_TEXT", answer_a)  # type: ignore[arg-type]


def test_append_binder_collision_renamed_deterministically(answer_a):
    result = ac.apply_combinator(
        "APPEND_STATEMENT", answer_a, binder="v0",
        rhs=ac.parse_rhs('TextContent(":foot")'),
    )
    _assert_admitted(result)
    appended = result.proof["appended_binder"]
    assert appended != "v0" and appended.startswith("v0__")
    assert result.proof["alpha_rename_map"] == {"v0": appended}


# --------------------------------------------------------------------------- #
# Content-addressed lineage persistence (artifact graph)
# --------------------------------------------------------------------------- #


def _train_root() -> str:
    policy = RootFamilySplitPolicyV1()
    for index in range(10_000):
        root = f"compose-family-{index}"
        if policy.assign(root) == "train":
            return root
    raise AssertionError("no train root")


def _source_node(answer: ac.AnswerV1, root: str) -> ArtifactNodeV1:
    return ArtifactNodeV1(
        artifact_id=answer.answer_id,
        artifact_type="answer",
        root_family_id=root,
        split_group_id=root,
        split=answer.split,
        parent_ids=(),
        surface_sha256=hashlib.sha256(answer.source.encode()).hexdigest(),
        alpha_sha256=answer.canonical_digest,
        canonical_ast_sha256=answer.canonical_digest,
        near_template_sha256=content_sha(sorted(answer.canonical_source.split())),
        payload={"openui": answer.canonical_source},
    )


def test_lineage_content_addressed_and_deterministic(tmp_path):
    root = _train_root()
    left = ac.admit_answer(FIXTURE_C, root_family_id=root)
    right = ac.admit_answer(FIXTURE_D, root_family_id=root)
    store = ArtifactGraphStore(tmp_path)
    store.append(_source_node(left, root))
    store.append(_source_node(right, root))

    result = ac.apply_combinator("COMPOSE_DOCUMENT", (left, right))
    _assert_admitted(result)
    lineage_id = ac.persist_lineage(result, store)
    assert lineage_id == result.lineage_id
    # Deterministic: an identical composition yields the identical lineage id.
    again = ac.apply_combinator("COMPOSE_DOCUMENT", (left, right))
    assert again.lineage_id == lineage_id
    ac.persist_lineage(again, store)  # idempotent write

    nodes = store.load_nodes()
    node = nodes[lineage_id]
    assert node.artifact_type == ac.ARTIFACT_TYPE
    assert set(node.parent_ids) == {left.answer_id, right.answer_id}
    assert node.payload["combinator"] == "COMPOSE_DOCUMENT"
    assert node.payload["combinator_version"] == "v1"
    assert node.payload["before_ast_digest"] == result.before_ast_digest
    assert node.payload["after_ast_digest"] == result.after_ast_digest
    assert node.canonical_ast_sha256 == result.after_ast_digest
    assert sorted(store.ancestors(lineage_id)) == sorted(result.source_answer_ids)


def test_rejected_composition_has_no_lineage(tmp_path, answer_a):
    other = ac.admit_answer(FIXTURE_B, split="test", root_family_id="fam-b")
    result = ac.apply_combinator("COMPOSE_DOCUMENT", (answer_a, other))
    assert result.lineage_id is None
    with pytest.raises(ValueError, match="no lineage"):
        ac.persist_lineage(result, ArtifactGraphStore(tmp_path))
