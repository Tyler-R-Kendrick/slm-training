"""Regression tests for the capsule-owned topology adapter (SLM-67)."""
from __future__ import annotations

from dataclasses import dataclass, field

from slm_training.dsl.solver import (
    TopologyAction,
    TopologyAdapterConfig,
    TopologyEdit,
    derive_topology_holes,
    legal_topology_productions,
)
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
)


@dataclass
class _FakeCodec:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    mask_id: int = 3
    unk_id: int = 4
    id_to_production: dict[int, str] = field(default_factory=lambda: {
        0: "<pad>",
        1: "<bos>",
        2: "<eos>",
        3: "<mask>",
        4: "<unk>",
        5: "+Stack",
        6: "+Button",
        7: "r=",
        8: "$=",
        9: "[",
        10: "]",
        11: "!v0.5",
        12: "!lexical",
    })


@dataclass
class _FakeNode:
    node_id: int
    node_type: str
    production_id: int
    slot_id: int = 0
    parent_id: int = -1
    depth: int = 0
    sibling_index: int = 0
    children: list[_FakeNode] = field(default_factory=list)
    active: bool = False


def test_legal_productions_are_torch_free():
    codec = _FakeCodec()
    pids = legal_topology_productions(codec, "document")
    assert all(isinstance(pid, int) for pid in pids)
    # Document-legal tokens include <bos>, !v0.5, and fragment markers.
    assert set(pids) == {1, 11, 12}

    statement_pids = legal_topology_productions(codec, "statement")
    assert set(statement_pids) == {7, 8}


def test_derive_holes_returns_complete_edit_tuples():
    codec = _FakeCodec()
    root = _FakeNode(
        node_id=0,
        node_type="document",
        production_id=codec.bos_id,
        active=True,
        children=[
            _FakeNode(
                node_id=1,
                node_type="statement",
                production_id=7,
                active=True,
                parent_id=0,
                depth=1,
            )
        ],
    )
    config = TopologyAdapterConfig(topology_max_nodes=8, topology_max_active=8)
    holes = derive_topology_holes(root, codec, config)
    assert holes
    for hole in holes:
        assert hole.domain.values
        for value in hole.domain.values:
            assert isinstance(value, DomainValue)
            assert value.tag == "topology_edit"
            assert len(value.payload) == 4
            # The adapter round-trips through main's canonical DomainValue.
            assert TopologyEdit.from_value(value).to_value() == value


def test_hole_domains_are_finite_and_unique():
    codec = _FakeCodec()
    root = _FakeNode(
        node_id=0,
        node_type="document",
        production_id=codec.bos_id,
        active=True,
    )
    config = TopologyAdapterConfig(topology_max_nodes=8, topology_max_active=8)
    holes = derive_topology_holes(root, codec, config, slot_inventory=[":a", ":b"])
    assert len(holes) == 1
    domain = holes[0].domain
    assert len(domain.values) <= 4 + 2 * 9 * 3  # 4 structural + pids*arity*slots
    assert len(set(domain.values)) == len(domain.values)


def test_topology_edit_round_trips_through_domain_value():
    edit = TopologyEdit(
        action=TopologyAction.EXPAND,
        production_id=42,
        arity=2,
        slot_id=1,
    )
    value = edit.to_value()
    assert isinstance(value, DomainValue)
    assert value.tag == "topology_edit"
    recovered = TopologyEdit.from_value(value)
    assert recovered == edit


def test_finite_domain_state_fingerprints_are_stable():
    hole_id = HoleId(namespace="topology", path=(0,), kind="document")
    values = (
        TopologyEdit(
            action=TopologyAction.KEEP, production_id=1, arity=0, slot_id=0
        ).to_value(),
        TopologyEdit(
            action=TopologyAction.EXPAND, production_id=5, arity=1, slot_id=0
        ).to_value(),
    )
    domain = HoleDomain(hole_id=hole_id, values=values)
    state = FiniteDomainState(
        problem_id="topology-fingerprint",
        pack_id="vss2",
        constraint_version="v1",
        bounds=SolverBounds(64, 32, 8, 16, 20),
        holes=(domain,),
    )
    assert state.fingerprint
    restored = FiniteDomainState.from_dict(state.to_dict())
    assert restored.fingerprint == state.fingerprint


def test_importing_adapter_does_not_import_torch():
    import importlib
    import sys

    # Snapshot current torch presence, then fresh-import the adapter module.
    had_torch = "torch" in sys.modules
    if "slm_training.dsl.solver.topology_adapter" in sys.modules:
        del sys.modules["slm_training.dsl.solver.topology_adapter"]
    importlib.import_module("slm_training.dsl.solver.topology_adapter")
    assert ("torch" in sys.modules) == had_torch
