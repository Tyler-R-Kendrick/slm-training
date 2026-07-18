"""Torch-free adapter between GrammarDiffusionModel topology and finite domains."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

from slm_training.dsl.solver.state import DomainValue, HoleDomain, HoleId


class TopologyAction(IntEnum):
    """Bounded topology edit actions (mirrors grammar_diffusion.TopologyAction)."""

    EXPAND = 0
    KEEP = 1
    DELETE = 2
    CONTRACT = 3
    STOP = 4


NODE_TYPES = ("document", "statement", "expression", "component", "list", "leaf")
NODE_TYPE_ID = {name: index for index, name in enumerate(NODE_TYPES)}
V05_MARKERS = {"r=", "$=", "q=", "m=", "a=", "="}
FRAGMENT_MARKERS = {"!lexical", "!expression", "!statement"}
FRAGMENT_CHUNK = "!fragment_chunk"


@runtime_checkable
class TopologyNodeLike(Protocol):
    """Torch-free view of a topology node."""

    node_id: int
    node_type: str
    production_id: int
    slot_id: int
    parent_id: int
    depth: int
    sibling_index: int
    children: list[Any]
    active: bool


@dataclass(frozen=True)
class TopologyEdit:
    """One complete edit tuple for a topology hole."""

    action: TopologyAction
    production_id: int
    arity: int
    slot_id: int

    def to_value(self) -> DomainValue:
        """Encode as a canonical ``topology_edit`` domain value on main's API."""
        return DomainValue.create(
            "topology_edit",
            [self.action.name, self.production_id, self.arity, self.slot_id],
        )

    @classmethod
    def from_value(cls, value: DomainValue) -> TopologyEdit:
        """Decode a ``topology_edit`` domain value produced by ``to_value``."""
        if value.tag != "topology_edit":
            raise ValueError(
                f"expected a topology_edit domain value, got {value.tag!r}"
            )
        action_name, production_id, arity, slot_id = value.payload
        return cls(
            action=TopologyAction[action_name],
            production_id=int(production_id),
            arity=int(arity),
            slot_id=int(slot_id),
        )


@dataclass(frozen=True)
class TopologyHole:
    """A semantic hole backed by one topology node."""

    hole_id: HoleId
    node_id: int
    node_type: str
    domain: HoleDomain

    def to_dict(self) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id.to_dict(),
            "node_id": self.node_id,
            "node_type": self.node_type,
            "domain": self.domain.to_dict(),
        }


@dataclass(frozen=True)
class TopologyAdapterConfig:
    """Bounded frame for topology-domain enumeration."""

    topology_max_nodes: int = 256
    topology_max_active: int = 64
    topology_max_arity: int = 8
    topology_max_depth: int = 32
    topology_bounded_buffer: bool = True
    topology_global_sync_interval: int = 1

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, int) and value < 0:
                raise ValueError(f"{name} must be non-negative")


def _node_type(token: str) -> str:
    if token == FRAGMENT_CHUNK:
        return "list"
    if token.startswith("+"):
        return "component"
    if token == "[":
        return "list"
    return "leaf"


def legal_topology_productions(
    codec: Any,
    node_type: str,
    *,
    leaf_only: bool = False,
) -> list[int]:
    """Return production ids legal for a topology node type.

    Mirrors GrammarDiffusionModel._legal_ids without importing torch.
    """
    pad_id = getattr(codec, "pad_id", -1)
    eos_id = getattr(codec, "eos_id", -1)
    mask_id = getattr(codec, "mask_id", -1)
    specials = {pad_id, eos_id, mask_id}
    result: list[int] = []
    id_to_production = getattr(codec, "id_to_production", {})
    for pid, token in id_to_production.items():
        if pid in specials or token in {"-", "]", ";"}:
            continue
        if node_type == "document" and token not in {
            "<bos>",
            "!v0.5",
            *FRAGMENT_MARKERS,
        }:
            continue
        if node_type == "statement" and token not in V05_MARKERS:
            continue
        if node_type == "expression" and token in (
            V05_MARKERS | FRAGMENT_MARKERS | {"!v0.5", "<bos>"}
        ):
            continue
        if leaf_only and _node_type(token) != "leaf":
            continue
        result.append(pid)
    if not result:
        unk_id = getattr(codec, "unk_id", mask_id)
        result.append(unk_id)
    return result


def _flatten(root: TopologyNodeLike) -> list[TopologyNodeLike]:
    out: list[TopologyNodeLike] = []

    def visit(node: TopologyNodeLike) -> None:
        out.append(node)
        for child in node.children:
            visit(child)

    visit(root)
    return out


def _selected_nodes(
    root: TopologyNodeLike,
    config: TopologyAdapterConfig,
    phase: int = 0,
) -> list[TopologyNodeLike]:
    """Bounded active-node buffer matching the model's selection."""
    nodes = _flatten(root)[: config.topology_max_nodes]
    if (
        not config.topology_bounded_buffer
        or phase % max(1, config.topology_global_sync_interval) == 0
    ):
        return nodes
    active = [node for node in nodes if node.active][: config.topology_max_active]
    wanted = {node.node_id for node in active}
    by_id = {node.node_id: node for node in nodes}
    for node in active:
        current = node
        while current.parent_id in by_id:
            current = by_id[current.parent_id]
            wanted.add(current.node_id)
            for sibling in current.children:
                wanted.add(sibling.node_id)
    return [node for node in nodes if node.node_id in wanted][
        : config.topology_max_nodes
    ]


def derive_topology_holes(
    root: TopologyNodeLike,
    codec: Any,
    config: TopologyAdapterConfig,
    slot_inventory: list[str] | None = None,
    *,
    phase: int = 0,
) -> list[TopologyHole]:
    """Map active topology nodes to finite semantic holes with edit tuples."""
    slots = slot_inventory or []
    nodes = _selected_nodes(root, config, phase=phase)
    holes: list[TopologyHole] = []
    for node in nodes:
        if not node.active:
            continue
        if node.depth >= config.topology_max_depth:
            continue
        values: list[DomainValue] = []

        # Complete edit tuples for structural actions.
        for action in (TopologyAction.KEEP, TopologyAction.DELETE, TopologyAction.STOP):
            values.append(
                TopologyEdit(
                    action=action,
                    production_id=node.production_id,
                    arity=len(node.children),
                    slot_id=node.slot_id,
                ).to_value()
            )

        # EXPAND actions: enumerate legal complete edits (production, arity, slot).
        if node.node_type != "leaf":
            legal_pids = legal_topology_productions(codec, node.node_type)
            max_arity = min(config.topology_max_arity, 8)
            for pid in legal_pids:
                token = codec.id_to_production.get(pid, "")
                is_container = token.startswith("+") or token in {
                    "[",
                    FRAGMENT_CHUNK,
                }
                arity_range = range(max_arity + 1) if is_container else range(1)
                for arity in arity_range:
                    for slot_id in ([0, *range(1, len(slots) + 1)]) if slots else [0]:
                        values.append(
                            TopologyEdit(
                                action=TopologyAction.EXPAND,
                                production_id=pid,
                                arity=arity,
                                slot_id=slot_id,
                            ).to_value()
                        )

        hole_id = HoleId(
            namespace="topology",
            path=(node.node_id,),
            kind=node.node_type,
        )
        holes.append(
            TopologyHole(
                hole_id=hole_id,
                node_id=node.node_id,
                node_type=node.node_type,
                domain=HoleDomain(
                    hole_id=hole_id,
                    values=tuple(dict.fromkeys(values)),
                    metadata=(
                        ("depth", node.depth),
                        ("production_id", node.production_id),
                        ("slot_count", len(slots)),
                    ),
                ),
            )
        )
    return holes
