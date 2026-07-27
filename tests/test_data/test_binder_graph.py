"""Tests for slm_training.data.progspec.binder_graph (AP-026 / SLM-320)."""

from __future__ import annotations

from slm_training.data.progspec import (
    BinderGraphV1,
    DependencyKind,
    ProgramSpec,
    derive_binder_graph,
    derive_capsule_graph,
)
from slm_training.dsl.language_contract import contract_id

# Sibling statement definitions are nested under a top-level "zdefs" key
# (rather than inline where they're referenced) because the shared node/edge
# walk resolves a name's target node from whichever STATEMENT contract's
# scope_id sorts last among the contracts that (transitively) contain that
# name -- and the "root" statement's own contract always transitively claims
# every nested name (see capsules.derive_node_edge_walk). "zdefs" sorts after
# "root" lexically, so each specific definition's own contract -- not root's
# aggregate -- wins the binder's target-node mapping.
def _definition(statement_id: str, props: dict) -> dict:
    return {
        "type": "element",
        "typeName": "Widget",
        "statementId": statement_id,
        "props": props,
    }


def _spec(spec_id: str, ast: dict, canonical_openui: str) -> ProgramSpec:
    return ProgramSpec(
        id=spec_id,
        ast=ast,
        canonical_openui=canonical_openui,
        facts={},
        contract_id=contract_id(),
        program_family_id=f"{spec_id}_family",
        lineage_id=f"{spec_id}_lineage",
        split_group_id=f"{spec_id}_group",
    )


def _resolved_ast(cta_name: str = "cta") -> dict:
    return {
        "type": "element",
        "typeName": "Stack",
        "statementId": "root",
        "props": {"children": [{"type": "ref", "name": cta_name}]},
        "zdefs": [_definition(cta_name, {"label": ":cta.label"})],
    }


def test_unresolved_reference_is_collected_not_raised() -> None:
    spec = _spec(
        "unresolved",
        {
            "type": "element",
            "typeName": "Stack",
            "statementId": "root",
            "props": {"children": [{"type": "ref", "name": "undefined_binder"}]},
        },
        "root = Stack([undefined_binder])",
    )
    graph = derive_binder_graph(spec)
    assert [u.name for u in graph.unresolved] == ["undefined_binder"]

    result = graph.check_g3()
    assert result.ok is False
    assert result.all_resolved is False
    assert result.single_root is True


def test_matches_capsule_graph_when_fully_resolved() -> None:
    spec = _spec(
        "resolved", _resolved_ast(), 'root = Stack([cta])\ncta = Button(":cta.label")'
    )

    binder_graph = derive_binder_graph(spec)
    capsule_graph = derive_capsule_graph(spec)

    assert binder_graph.nodes == capsule_graph.nodes
    assert binder_graph.edges == capsule_graph.edges
    assert binder_graph.unresolved == ()

    result = binder_graph.check_g3()
    assert result.ok is True
    assert result.cycle is None
    assert result.unreachable_node_ids == ()


def test_reference_cycle_is_detected() -> None:
    spec = _spec(
        "cyclic",
        {
            "type": "element",
            "typeName": "Stack",
            "statementId": "root",
            "props": {"children": [{"type": "ref", "name": "a"}]},
            "zdefs": [
                _definition("a", {"child": {"type": "ref", "name": "b"}}),
                _definition("b", {"child": {"type": "ref", "name": "a"}}),
            ],
        },
        "root = Stack([a])\na = Widget(b)\nb = Widget(a)",
    )
    graph = derive_binder_graph(spec)
    result = graph.check_g3()
    assert result.ok is False
    assert result.acyclic is False
    assert result.cycle is not None
    assert len(result.cycle) >= 2
    assert result.cycle[0] == result.cycle[-1]


def test_unreachable_binder_is_reported() -> None:
    # "orphan" is defined but never referenced from anywhere -> unreachable.
    spec = _spec(
        "unreachable",
        {
            "type": "element",
            "typeName": "Stack",
            "statementId": "root",
            "props": {"children": []},
            "zdefs": [_definition("orphan", {})],
        },
        "root = Stack([])\norphan = Widget()",
    )
    graph = derive_binder_graph(spec)
    result = graph.check_g3()
    assert result.ok is False
    assert result.reachable is False
    assert result.unreachable_node_ids


def test_alpha_renaming_binder_names_preserves_fingerprint() -> None:
    spec_a = _spec(
        "alpha_a", _resolved_ast("cta"), 'root = Stack([cta])\ncta = Button(":cta.label")'
    )
    spec_b = _spec(
        "alpha_b",
        _resolved_ast("submitButton"),
        'root = Stack([submitButton])\nsubmitButton = Button(":cta.label")',
    )

    graph_a = derive_binder_graph(spec_a)
    graph_b = derive_binder_graph(spec_b)

    # Sanity: the two graphs really do differ in raw (non-canonicalized) form.
    assert graph_a.definitions() != graph_b.definitions()
    assert graph_a.spec_id != graph_b.spec_id

    assert graph_a.alpha_fingerprint() == graph_b.alpha_fingerprint()


def test_alpha_fingerprint_differs_for_structurally_different_graphs() -> None:
    spec_a = _spec(
        "shape_a", _resolved_ast("cta"), 'root = Stack([cta])\ncta = Button(":cta.label")'
    )
    spec_b = _spec(
        "shape_b",
        {
            "type": "element",
            "typeName": "Stack",
            "statementId": "root",
            "props": {"children": []},
        },
        "root = Stack([])",
    )

    graph_a = derive_binder_graph(spec_a)
    graph_b = derive_binder_graph(spec_b)
    assert graph_a.alpha_fingerprint() != graph_b.alpha_fingerprint()


def test_binder_graph_round_trips() -> None:
    spec = _spec(
        "roundtrip", _resolved_ast(), 'root = Stack([cta])\ncta = Button(":cta.label")'
    )
    graph = derive_binder_graph(spec)

    assert BinderGraphV1.from_dict(graph.to_dict()) == graph


def test_external_slot_stays_an_edge_not_unresolved() -> None:
    spec = _spec(
        "external", _resolved_ast(), 'root = Stack([cta])\ncta = Button(":cta.label")'
    )
    graph = derive_binder_graph(spec)
    external_edges = [e for e in graph.edges if e.kind == DependencyKind.EXTERNAL]
    assert external_edges
    assert graph.unresolved == ()
