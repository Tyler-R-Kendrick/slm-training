"""Torch-free topology expander/verifier for exact closure integration (VSS3-03).

This module couples the finite-domain solver machinery to the grammar-diffusion
topology tree without importing torch.  It is deliberately a *decode-time* seam:
exact closure prunes the hard edit domain before the model (or an optional energy
ranker) orders the survivors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleId,
    SolverBounds,
)
from slm_training.dsl.solver.support import (
    ExpandStatus,
    ExpandStep,
    VerifyOutcome,
    VerifyStatus,
)
from slm_training.dsl.solver.topology_adapter import (
    FRAGMENT_CHUNK,
    TopologyAdapterConfig,
    TopologyEdit,
    TopologyNodeLike,
    _node_type,
    derive_topology_state,
)


@dataclass
class SolverTopologyNode:
    """Mutable torch-free topology node used inside the solver expander."""

    node_id: int
    node_type: str
    production_id: int
    slot_id: int = 0
    parent_id: int = -1
    depth: int = 0
    sibling_index: int = 0
    children: list["SolverTopologyNode"] = field(default_factory=list)
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "production_id": self.production_id,
            "slot_id": self.slot_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "sibling_index": self.sibling_index,
            "children": [child.to_dict() for child in self.children],
            "active": self.active,
        }


def _copy_tree(node: SolverTopologyNode) -> SolverTopologyNode:
    """Deep-copy a topology tree while preserving original node ids."""
    copy = SolverTopologyNode(
        node_id=node.node_id,
        node_type=node.node_type,
        production_id=node.production_id,
        slot_id=node.slot_id,
        parent_id=node.parent_id,
        depth=node.depth,
        sibling_index=node.sibling_index,
        active=node.active,
    )
    for child in node.children:
        child_copy = _copy_tree(child)
        child_copy.parent_id = copy.node_id
        copy.children.append(child_copy)
    return copy


def _max_node_id(root: SolverTopologyNode) -> int:
    best = root.node_id
    for child in root.children:
        best = max(best, _max_node_id(child))
    return best


def _child_type(parent_node_type: str, token: str, output_kind: str) -> str:
    """Mirror GrammarDiffusionModel._decode_one child typing."""
    if parent_node_type == "document":
        return "leaf" if output_kind != "document" else "statement"
    return "expression"


def _find_node(root: SolverTopologyNode, node_id: int) -> SolverTopologyNode | None:
    if root.node_id == node_id:
        return root
    for child in root.children:
        found = _find_node(child, node_id)
        if found is not None:
            return found
    return None


def _apply_edit(
    root: SolverTopologyNode,
    node_id: int,
    edit: TopologyEdit,
    codec: Any,
    output_kind: str,
) -> SolverTopologyNode | None:
    """Apply one TopologyEdit to a copied tree and return the new root.

    Returns ``None`` for structurally illegal edits (e.g. deleting the root).
    """
    node = _find_node(root, node_id)
    if node is None:
        return None

    action = edit.action
    if action.name == "DELETE":
        if node.parent_id < 0:
            return None
        parent = None

        def find_parent(n: SolverTopologyNode) -> SolverTopologyNode | None:
            for child in n.children:
                if child.node_id == node.node_id:
                    return n
                found = find_parent(child)
                if found is not None:
                    return found
            return None

        parent = find_parent(root)
        if parent is None:
            return None
        parent.children = [
            child for child in parent.children if child.node_id != node.node_id
        ]
        return root

    # KEEP, STOP, and EXPAND all resolve the node and possibly create children.
    if action.name == "EXPAND":
        node.production_id = edit.production_id
        node.slot_id = edit.slot_id
        token = codec.id_to_production.get(edit.production_id, "")
        if node.node_type == "expression":
            if output_kind != "document" and token == FRAGMENT_CHUNK:
                node.node_type = "list"
            elif output_kind != "document":
                node.node_type = "leaf"
            else:
                node.node_type = _node_type(token)
    elif action.name in {"KEEP", "STOP"}:
        # Domain tuple already carries the node's current production/slot.
        node.slot_id = edit.slot_id
    else:
        return None

    node.active = False
    node.children = []
    next_id = _max_node_id(root) + 1
    for child_index in range(edit.arity):
        token = codec.id_to_production.get(node.production_id, "")
        child_type = _child_type(node.node_type, token, output_kind)
        child = SolverTopologyNode(
            node_id=next_id,
            node_type=child_type,
            production_id=codec.mask_id,
            parent_id=node.node_id,
            depth=node.depth + 1,
            sibling_index=child_index,
            active=True,
        )
        node.children.append(child)
        next_id += 1
    return root


def _serialize_topology(
    codec: Any,
    root: SolverTopologyNode,
) -> tuple[list[int], list[int]]:
    """Torch-free mirror of grammar_diffusion._serialize_topology."""
    production_ids: list[int] = [codec.bos_id]
    slot_ids: list[int] = [codec.slot_none_id]
    root_token = codec.id_to_production.get(root.production_id, "")
    is_v05 = root_token == "!v0.5"
    is_fragment = root_token in {"!lexical", "!expression", "!statement"}
    if is_v05 or is_fragment:
        production_ids.append(root.production_id)
        slot_ids.append(root.slot_id)

    def emit(node: SolverTopologyNode) -> None:
        production_ids.append(node.production_id)
        slot_ids.append(node.slot_id)
        token = codec.id_to_production.get(node.production_id, "")
        for child in node.children:
            emit(child)
        if token.startswith("+"):
            production_ids.append(codec.production_to_id["-"])
            slot_ids.append(codec.slot_none_id)
        elif token == "[":
            production_ids.append(codec.production_to_id["]"])
            slot_ids.append(codec.slot_none_id)

    for statement in root.children:
        if is_fragment:

            def emit_fragment(node: SolverTopologyNode) -> None:
                token = codec.id_to_production.get(node.production_id, "")
                if token != FRAGMENT_CHUNK:
                    production_ids.append(node.production_id)
                    slot_ids.append(node.slot_id)
                for child in node.children:
                    emit_fragment(child)

            emit_fragment(statement)
        else:
            emit(statement)
            if is_v05:
                production_ids.append(codec.production_to_id[";"])
                slot_ids.append(codec.slot_none_id)
    return production_ids, slot_ids


class TopologyVerifier:
    """Validates a structurally-solved topology program with the DSL parser."""

    def __init__(self, codec: Any, output_kind: str = "document") -> None:
        self._codec = codec
        self._output_kind = output_kind

    @property
    def profile(self) -> str:
        return f"topology/{self._output_kind}"

    def verify(self, program: str) -> VerifyOutcome:
        from slm_training.dsl.parser import validate_output

        try:
            validate_output(program, self._output_kind)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            return VerifyOutcome(
                status=VerifyStatus.REJECT,
                detail=str(exc)[:200],
            )
        return VerifyOutcome(status=VerifyStatus.ACCEPT, detail="ok")


class TopologyProblemExpander:
    """Deterministic bounded expansion of topology edit choices."""

    def __init__(
        self,
        codec: Any,
        adapter_config: TopologyAdapterConfig,
        slot_inventory: list[str],
        output_kind: str,
        bounds: SolverBounds,
    ) -> None:
        self._codec = codec
        self._adapter_config = adapter_config
        self._slot_inventory = slot_inventory
        self._output_kind = output_kind
        self._bounds = bounds

    @property
    def problem_id(self) -> str:
        return "topology"

    @property
    def pack_id(self) -> str:
        return "openui"

    @property
    def constraint_version(self) -> str:
        return "v1"

    @property
    def bounds(self) -> SolverBounds:
        return self._bounds

    def successor(
        self, state: FiniteDomainState, hole_id: HoleId, value: DomainValue
    ) -> ExpandStep:
        try:
            edit = TopologyEdit.from_value(value)
        except (KeyError, ValueError, TypeError) as exc:
            return ExpandStep(
                status=ExpandStatus.DEAD,
                detail=f"bad topology edit value: {exc}",
            )

        if not state.holes:
            return ExpandStep(
                status=ExpandStatus.DEAD,
                detail="no holes in state",
            )

        # Convert the first (and only) topology hole path entry to a node id.
        if not hole_id.path or not isinstance(hole_id.path[0], int):
            return ExpandStep(
                status=ExpandStatus.DEAD,
                detail="topology hole path must start with an int node_id",
            )
        node_id = hole_id.path[0]

        # The active tree is not stored in the state, so we reconstruct it from
        # the hole metadata we attached in derive_topology_holes.  Each hole's
        # metadata records the original node_id and depth.
        root = _reconstruct_tree_from_holes(state)
        if root is None:
            return ExpandStep(
                status=ExpandStatus.INCOMPLETE,
                detail="topology state does not carry tree reconstruction metadata",
            )

        edited = _apply_edit(root, node_id, edit, self._codec, self._output_kind)
        if edited is None:
            return ExpandStep(
                status=ExpandStatus.DEAD,
                detail=f"edit {edit} could not be applied to node {node_id}",
            )

        next_state = derive_topology_state(
            edited,  # type: ignore[arg-type]
            self._codec,
            self._adapter_config,
            slot_inventory=self._slot_inventory,
            problem_id=self.problem_id,
            pack_id=self.pack_id,
            constraint_version=self.constraint_version,
            bounds=self._bounds,
            phase=0,
        )
        if next_state.is_bottom:
            return ExpandStep(
                status=ExpandStatus.DEAD,
                detail="derived topology state is bottom",
            )
        if next_state.is_structurally_solved:
            production_ids, slot_ids = _serialize_topology(self._codec, edited)
            text = self._codec.decode(
                production_ids, slot_ids, self._slot_inventory
            ).strip()
            return ExpandStep(
                status=ExpandStatus.TERMINAL,
                program=text,
                detail="structurally_solved",
            )
        return ExpandStep(
            status=ExpandStatus.CONTINUE,
            next_state=next_state,
            detail="continue",
        )


def _reconstruct_tree_from_holes(
    state: FiniteDomainState,
) -> SolverTopologyNode | None:
    """Reconstruct a minimal SolverTopologyNode tree from hole metadata.

    Each hole was produced by ``derive_topology_holes`` and its metadata contains
    ``depth``, ``parent_id`` and ``production_id`` for the owning node.  We build
    a tree by wiring each node to its parent; the root is the node with
    ``parent_id < 0``.
    """
    if not state.holes:
        return None
    nodes: dict[int, SolverTopologyNode] = {}
    parent_of: dict[int, int] = {}
    for hole in state.holes:
        path = hole.hole_id.path
        if not path or not isinstance(path[0], int):
            return None
        node_id = path[0]
        if node_id in nodes:
            continue
        depth = 0
        production_id = 0
        parent_id = -1
        for key, value in hole.metadata:
            if key == "depth":
                depth = int(value) if isinstance(value, int) else 0
            elif key == "parent_id":
                parent_id = int(value) if isinstance(value, int) else -1
            elif key == "production_id":
                production_id = int(value) if isinstance(value, int) else 0
        parent_of[node_id] = parent_id
        nodes[node_id] = SolverTopologyNode(
            node_id=node_id,
            node_type=hole.hole_id.kind,
            production_id=production_id,
            parent_id=parent_id,
            depth=depth,
            active=True,
        )
    if not nodes:
        return None
    roots = [n for n in nodes.values() if n.parent_id < 0]
    if len(roots) != 1:
        return None
    root = roots[0]
    # Wire children under their parents in sibling order by depth/node_id.
    for node in sorted(nodes.values(), key=lambda n: (n.depth, n.node_id)):
        if node is root:
            continue
        parent = nodes.get(node.parent_id)
        if parent is None:
            return None
        node.sibling_index = len(parent.children)
        parent.children.append(node)
    return root


def build_topology_support_provider(
    codec: Any,
    adapter_config: TopologyAdapterConfig,
    slot_inventory: list[str],
    output_kind: str,
    bounds: SolverBounds,
) -> Any:
    """Return a SupportProvider for topology exact closure."""
    from slm_training.dsl.solver.closure import EnumerativeSupportProvider

    expander = TopologyProblemExpander(
        codec, adapter_config, slot_inventory, output_kind, bounds
    )
    verifier = TopologyVerifier(codec, output_kind)
    return EnumerativeSupportProvider(expander, verifier)


def topology_solver_prune(
    root: TopologyNodeLike,
    codec: Any,
    adapter_config: TopologyAdapterConfig,
    slot_inventory: list[str],
    output_kind: str,
    bounds: SolverBounds,
    *,
    cache: dict[str, Any] | None = None,
    certificate_store: dict[str, Any] | None = None,
    max_queries: int | None = None,
) -> tuple[set[tuple[int, str, int, int, int]], Any]:
    """Run exact closure over the current topology tree and return live edits.

    Returns a set keyed by ``(node_id, action.name, production_id, arity, slot_id)``
    plus the raw ``ClosureResult`` for tracing.
    """
    from slm_training.dsl.solver.closure import exact_closure

    state = derive_topology_state(
        root,
        codec,
        adapter_config,
        slot_inventory=slot_inventory,
        problem_id="topology",
        pack_id="openui",
        constraint_version="v1",
        bounds=bounds,
        phase=0,
    )
    provider = build_topology_support_provider(
        codec, adapter_config, slot_inventory, output_kind, bounds
    )
    result = exact_closure(
        state,
        provider,
        cache=cache,
        certificate_store=certificate_store,
        max_queries=max_queries,
    )
    survivors: set[tuple[int, str, int, int, int]] = set()
    for hole in result.state.holes:
        node_id = hole.hole_id.path[0]
        if not isinstance(node_id, int):
            continue
        for value in hole.values:
            try:
                edit = TopologyEdit.from_value(value)
            except (KeyError, ValueError, TypeError):
                continue
            survivors.add(
                (node_id, edit.action.name, edit.production_id, edit.arity, edit.slot_id)
            )
    return survivors, result
