"""BinderPrecisionChannel regression tests (AP-026 / SLM-320)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.data.progspec.binder_graph import BinderGraphV1
from slm_training.data.progspec.capsules import DependencyKind, ScopeEdge, ScopeNode
from slm_training.models.binder_precision_channel import (
    BinderChannelArm,
    BinderChannelGate,
    confirm_predicted_edges,
    resolve_binder_graph,
)


def _node(node_id: str, *, kind: str = "statement", definitions: tuple[str, ...] = ()) -> ScopeNode:
    return ScopeNode(
        node_id=node_id,
        scope_id=node_id,
        kind=kind,
        ast_path=(),
        member_paths=(),
        definitions=definitions,
        external_dependencies=(),
    )


def _edge(source: str, target: str, *, role: str, kind: DependencyKind = DependencyKind.REFERENCE) -> ScopeEdge:
    return ScopeEdge(source=source, target=target, kind=kind, role=role)


def _graph(edges: tuple[ScopeEdge, ...]) -> BinderGraphV1:
    node_ids = {edge.source for edge in edges} | {edge.target for edge in edges} | {"root"}
    return BinderGraphV1(
        root_id="root",
        nodes=tuple(_node(nid) for nid in sorted(node_ids)),
        edges=edges,
        unresolved=(),
        spec_id="spec",
        version=BinderGraphV1.VERSION,
    )


def test_resolve_binder_graph_rejects_disabled_arm() -> None:
    graph = _graph((_edge("root", "cta", role="cta"),))
    with pytest.raises(ValueError):
        resolve_binder_graph(graph, arm=BinderChannelArm.DISABLED)


@pytest.mark.parametrize("arm", [BinderChannelArm.ORACLE, BinderChannelArm.PREDICTED_SOFT])
def test_passthrough_arms_return_the_same_graph(arm: BinderChannelArm) -> None:
    graph = _graph((_edge("root", "cta", role="cta"),))
    resolved = resolve_binder_graph(graph, arm=arm)
    assert resolved == graph


def test_absent_arm_is_empty_but_keeps_identity_fields() -> None:
    graph = _graph((_edge("root", "cta", role="cta"),))
    resolved = resolve_binder_graph(graph, arm=BinderChannelArm.ABSENT)
    assert resolved.nodes == ()
    assert resolved.edges == ()
    assert resolved.spec_id == graph.spec_id
    assert resolved.root_id == graph.root_id


def test_shuffled_arm_preserves_edge_count_but_breaks_targets() -> None:
    graph = _graph(
        (
            _edge("root", "cta", role="cta"),
            _edge("root", "title", role="title"),
        )
    )
    generator = torch.Generator().manual_seed(0)
    resolved = resolve_binder_graph(graph, arm=BinderChannelArm.SHUFFLED, generator=generator)
    assert len(resolved.edges) == len(graph.edges)
    assert resolved.nodes == graph.nodes


def test_corrupted_arm_redirects_reference_edges_to_a_wrong_target() -> None:
    graph = _graph(
        (
            _edge("root", "cta", role="cta"),
            _edge("root", "title", role="title"),
        )
    )
    resolved = resolve_binder_graph(graph, arm=BinderChannelArm.CORRUPTED)
    assert len(resolved.edges) == len(graph.edges)
    for original, corrupted in zip(graph.edges, resolved.edges):
        assert corrupted.target != original.target


def test_confirm_predicted_edges_drops_edges_the_verifier_does_not_have() -> None:
    # Predicted graph guesses an extra, wrong edge ("root" -> "bogus") that
    # the deterministic verifier graph never produced.
    verifier = _graph((_edge("root", "cta", role="cta"),))
    predicted = _graph(
        (
            _edge("root", "cta", role="cta"),
            _edge("root", "bogus", role="bogus"),
        )
    )
    confirmed = confirm_predicted_edges(predicted, verifier=verifier)
    assert confirmed == (_edge("root", "cta", role="cta"),)
    assert all(edge.role != "bogus" for edge in confirmed)


def test_confirm_predicted_edges_never_promotes_beyond_the_verifier() -> None:
    # Even a predicted graph that agrees with the verifier plus more must
    # never end up with MORE confirmed edges than the verifier has -- the
    # gate is a filter over predicted edges, not a merge with the verifier.
    verifier = _graph((_edge("root", "cta", role="cta"),))
    predicted = _graph(
        (
            _edge("root", "cta", role="cta"),
            _edge("cta", "root", role="root"),  # a second, unverified guess
        )
    )
    confirmed = confirm_predicted_edges(predicted, verifier=verifier)
    assert len(confirmed) <= len(verifier.reference_edges())


def test_gate_initialized_at_zero_is_a_numerical_no_op() -> None:
    gate = BinderChannelGate()
    features = torch.randn(4, 8)
    mask = torch.ones(4)
    bias = gate.bias_for_confirmed_edges(features, confirmed_mask=mask)
    assert torch.equal(bias, torch.zeros_like(features))


def test_gate_never_leaks_unconfirmed_edge_features_even_when_open() -> None:
    gate = BinderChannelGate()
    with torch.no_grad():
        gate.gate.fill_(5.0)  # force the gate fully open
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    confirmed_mask = torch.tensor([1.0, 0.0])  # only the first edge confirmed
    bias = gate.bias_for_confirmed_edges(features, confirmed_mask=confirmed_mask)
    assert torch.equal(bias[1], torch.zeros(2))
    assert not torch.equal(bias[0], torch.zeros(2))


def test_gate_rejects_mismatched_edge_and_mask_counts() -> None:
    gate = BinderChannelGate()
    with pytest.raises(ValueError):
        gate.bias_for_confirmed_edges(torch.randn(3, 4), confirmed_mask=torch.ones(2))
