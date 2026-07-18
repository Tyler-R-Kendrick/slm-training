"""Regression tests for the topology finite-domain solver seam (VSS3-03)."""

from __future__ import annotations

from dataclasses import dataclass, field

from slm_training.dsl.production_codec import ProductionCodec
from slm_training.dsl.solver.state import SolverBounds
from slm_training.dsl.solver.topology_adapter import TopologyAdapterConfig
from slm_training.dsl.solver.topology_solver import topology_solver_prune

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'


@dataclass
class _FakeNode:
    node_id: int
    node_type: str
    production_id: int
    slot_id: int = 0
    parent_id: int = -1
    depth: int = 0
    sibling_index: int = 0
    children: list["_FakeNode"] = field(default_factory=list)
    active: bool = False


def test_topology_solver_prune_runs_and_is_monotone() -> None:
    """Closure may only remove candidates; it never invents a new one."""
    codec = ProductionCodec.build([HERO])
    root = _FakeNode(
        node_id=0,
        node_type="document",
        production_id=codec.bos_id,
        active=True,
        children=[
            _FakeNode(
                node_id=1,
                node_type="statement",
                production_id=codec.production_to_id["="],
                active=True,
                parent_id=0,
                depth=1,
            )
        ],
    )
    adapter_config = TopologyAdapterConfig(
        topology_max_nodes=8,
        topology_max_active=8,
        topology_max_depth=4,
    )
    bounds = SolverBounds(
        max_tokens=256,
        max_nodes=8,
        max_depth=4,
        max_backtracks=2,
        max_verifier_calls=4,
    )
    survivors, result = topology_solver_prune(
        root,
        codec,
        adapter_config,
        slot_inventory=[":hero.title"],
        output_kind="document",
        bounds=bounds,
        max_queries=4,
    )
    # Collect the original complete edit domain.
    from slm_training.dsl.solver.topology_adapter import TopologyEdit

    original: set[tuple[int, str, int, int, int]] = set()
    from slm_training.dsl.solver.topology_adapter import derive_topology_holes

    for hole in derive_topology_holes(
        root, codec, adapter_config, slot_inventory=[":hero.title"]
    ):
        for value in hole.domain.values:
            edit = TopologyEdit.from_value(value)
            original.add(
                (hole.node_id, edit.action.name, edit.production_id, edit.arity, edit.slot_id)
            )
    assert survivors <= original
    assert result.counters.support_queries >= 0
