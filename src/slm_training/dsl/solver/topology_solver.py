"""Torch-free topology expander/verifier for exact closure integration (VSS3-03).

This module couples the finite-domain solver machinery to the grammar-diffusion
topology tree without importing torch.  It is deliberately a *decode-time* seam:
exact closure prunes the hard edit domain before the model (or an optional energy
ranker) orders the survivors.
"""

from __future__ import annotations

import hashlib
import json
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
    _flatten,
    _node_type,
    derive_topology_state,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


def _copy_topology_node(node: TopologyNodeLike) -> SolverTopologyNode:
    """Deep-copy any TopologyNodeLike into a mutable SolverTopologyNode."""
    copy = SolverTopologyNode(
        node_id=node.node_id,
        node_type=node.node_type,
        production_id=node.production_id,
        slot_id=node.slot_id,
        parent_id=node.parent_id,
        depth=node.depth,
        sibling_index=node.sibling_index,
        active=bool(getattr(node, "active", False)),
    )
    for child in node.children:
        child_copy = _copy_topology_node(child)
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

    def __init__(
        self, codec: Any, output_kind: str = "document", slot_inventory: list[str] | None = None
    ) -> None:
        self._codec = codec
        self._output_kind = output_kind
        self._slot_inventory = slot_inventory or []

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


# --------------------------------------------------------------------------- #
# Synthetic capsule-aware solve (VSS3-03 wiring)
#
# This is a deliberately simple coordinator integration: each active topology
# node becomes its own independent verification capsule, solved in dependency
# order by ``solve_capsule_graph``.  It exercises the capsule coordinator and
# SCC-joint plumbing without requiring a full ProgramSpec-to-topology mapping.
# Future work will replace the synthetic graph with one derived from the
# request's ProgramSpec and will implement real pack capsule slots.
# --------------------------------------------------------------------------- #


def _derive_synthetic_capsule_graph(
    root: TopologyNodeLike,
) -> Any:
    """Build a single joint capsule over all active topology nodes.

    This exercises the SCC-joint coordinator path: every active node is solved
    together rather than independently, so the search sees cross-node effects.
    """
    from slm_training.data.progspec.capsules import (
        CapsuleGraph,
        ScopeNode,
        VerificationCapsule,
    )

    active_ids = [
        node.node_id for node in _flatten(root) if getattr(node, "active", False)
    ]
    if not active_ids:
        active_ids = [root.node_id]
    nodes = tuple(
        ScopeNode(
            node_id=f"node_{node_id}",
            scope_id=None,
            kind="topology",
            ast_path=(node_id,),
            member_paths=(),
            definitions=(),
            external_dependencies=(),
        )
        for node_id in active_ids
    )
    node_id_strs = tuple(f"node_{node_id}" for node_id in active_ids)
    capsule = VerificationCapsule(
        capsule_id="capsule_topology",
        node_ids=node_id_strs,
        entry_node_id=node_id_strs[0],
        external_dependencies=(),
    )
    return CapsuleGraph(
        root_id=node_id_strs[0],
        nodes=nodes,
        edges=(),
        capsules=(capsule,),
        spec_id="topology",
        version=CapsuleGraph.VERSION,
    )


def _state_for_capsule(
    root: TopologyNodeLike,
    node_ids: tuple[int, ...],
    codec: Any,
    adapter_config: TopologyAdapterConfig,
    slot_inventory: list[str],
    bounds: SolverBounds,
) -> FiniteDomainState:
    """Derive a finite-domain state containing holes owned by ``node_ids``."""
    full_state = derive_topology_state(
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
    holes = tuple(
        hole
        for hole in full_state.holes
        if hole.hole_id.path
        and isinstance(hole.hole_id.path[0], int)
        and hole.hole_id.path[0] in node_ids
    )
    return FiniteDomainState(
        problem_id=full_state.problem_id,
        pack_id=full_state.pack_id,
        constraint_version=full_state.constraint_version,
        bounds=full_state.bounds,
        holes=holes,
    )


class TopologyCapsuleProblemBuilder:
    """Pack-style builder that maps a synthetic capsule to one node's state."""

    def __init__(
        self,
        root: TopologyNodeLike,
        codec: Any,
        adapter_config: TopologyAdapterConfig,
        slot_inventory: list[str],
        output_kind: str,
    ) -> None:
        self._root = root
        self._codec = codec
        self._adapter_config = adapter_config
        self._slot_inventory = slot_inventory
        self._output_kind = output_kind

    def build_problem(
        self,
        capsule: Any,
        predecessor_summaries: Any,
        external_inputs: Any,
        bounds: SolverBounds,
    ) -> Any:
        from slm_training.dsl.solver.capsule_solver import CapsuleProblem

        node_ids = tuple(
            int(nid.split("_", 1)[1]) for nid in capsule.node_ids
        )
        state = _state_for_capsule(
            self._root,
            node_ids,
            self._codec,
            self._adapter_config,
            self._slot_inventory,
            bounds,
        )
        return CapsuleProblem(
            capsule=capsule,
            state=state,
            predecessor_summaries=predecessor_summaries,
            external_inputs=external_inputs,
        )


class TopologyCapsuleSummaryExtractor:
    """Exact empty-boundary summary for a solved synthetic capsule."""

    def extract_summary(self, capsule: Any, state: FiniteDomainState) -> Any:
        from slm_training.dsl.solver.capsule_solver import CapsuleInterfaceSummary

        payload = _canonical_json(
            {
                "capsule_id": capsule.capsule_id,
                "state_fingerprint": state.fingerprint,
                "conservative": False,
            }
        )
        return CapsuleInterfaceSummary(
            capsule_id=capsule.capsule_id,
            input_bindings=(),
            output_bindings=(),
            slots=(),
            preconditions=(),
            postconditions=(),
            effects=(),
            exceptions=(),
            captures=(),
            conservative=False,
            fingerprint=_sha256(payload),
        )


class TopologyCapsuleMaterializer:
    """Apply solved capsule decisions to a copy of the tree and serialize."""

    def __init__(
        self,
        root: TopologyNodeLike,
        codec: Any,
        slot_inventory: list[str],
        output_kind: str,
    ) -> None:
        self._root = root
        self._codec = codec
        self._slot_inventory = slot_inventory
        self._output_kind = output_kind

    def __call__(self, capsule_results: Any) -> str | None:
        tree = _copy_topology_node(self._root)
        for result in capsule_results:
            if result.status != "solved" or result.search_result is None:
                continue
            for decision in result.search_result.decisions:
                node_id = decision.hole_id.path[0]
                if not isinstance(node_id, int):
                    continue
                try:
                    edit = TopologyEdit.from_value(decision.chosen)
                except (KeyError, ValueError, TypeError):
                    continue
                _apply_edit(tree, node_id, edit, self._codec, self._output_kind)
        production_ids, slot_ids = _serialize_topology(self._codec, tree)
        return self._codec.decode(
            production_ids, slot_ids, self._slot_inventory
        ).strip()


class TopologyCapsuleTerminalChecker:
    """Terminal checker used by the search controller inside each capsule."""

    def __init__(
        self, codec: Any, output_kind: str, slot_inventory: list[str] | None = None
    ) -> None:
        self._verifier = TopologyVerifier(codec, output_kind, slot_inventory)

    def check(self, state: FiniteDomainState) -> Any:
        from slm_training.dsl.solver.controller import TerminalOutcome

        if not state.is_structurally_solved:
            return TerminalOutcome(
                accepted=False,
                detail="capsule state is not structurally solved",
            )
        # Materialize the structurally solved state.
        tree = _reconstruct_tree_from_holes(state)
        if tree is None:
            return TerminalOutcome(
                accepted=False,
                detail="cannot reconstruct tree from solved state",
            )
        production_ids, slot_ids = _serialize_topology(self._verifier._codec, tree)
        source = self._verifier._codec.decode(
            production_ids, slot_ids, self._verifier._slot_inventory
        ).strip()
        outcome = self._verifier.verify(source)
        return TerminalOutcome(
            accepted=outcome.status is VerifyStatus.ACCEPT,
            source=source,
            detail=outcome.detail,
        )


class TopologyCapsuleGlobalVerifier:
    """Whole-program verifier used as the capsule coordinator global oracle."""

    def __init__(self, codec: Any, output_kind: str) -> None:
        self._verifier = TopologyVerifier(codec, output_kind)

    def verify(self, source: str | None) -> Any:
        from slm_training.dsl.solver.controller import TerminalOutcome

        if source is None:
            return TerminalOutcome(
                accepted=False,
                detail="capsule materializer produced no source",
            )
        outcome = self._verifier.verify(source)
        return TerminalOutcome(
            accepted=outcome.status is VerifyStatus.ACCEPT,
            detail=outcome.detail,
        )


def topology_capsule_solver_solve(
    root: TopologyNodeLike,
    codec: Any,
    adapter_config: TopologyAdapterConfig,
    slot_inventory: list[str],
    output_kind: str,
    bounds: SolverBounds,
    *,
    ranker: Any | None = None,
    cache: dict[str, Any] | None = None,
    certificate_store: dict[str, Any] | None = None,
) -> tuple[str | None, Any]:
    """Run the synthetic capsule coordinator over active topology nodes.

    Returns the assembled program text (or ``None``) and the raw
    ``CapsuleSolveResult`` for tracing.
    """
    from slm_training.dsl.solver.capsule_solver import solve_capsule_graph

    graph = _derive_synthetic_capsule_graph(root)
    provider = build_topology_support_provider(
        codec, adapter_config, slot_inventory, output_kind, bounds
    )
    builder = TopologyCapsuleProblemBuilder(
        root, codec, adapter_config, slot_inventory, output_kind
    )
    extractor = TopologyCapsuleSummaryExtractor()
    materializer = TopologyCapsuleMaterializer(
        root, codec, slot_inventory, output_kind
    )
    terminal_checker = TopologyCapsuleTerminalChecker(
        codec, output_kind, slot_inventory
    )
    global_verifier = TopologyCapsuleGlobalVerifier(codec, output_kind)

    result = solve_capsule_graph(
        graph,
        builder=builder,
        provider=provider,
        terminal_checker=terminal_checker,
        summary_extractor=extractor,
        materializer=materializer,
        global_verifier=global_verifier.verify,
        ranker=ranker,
        bounds=bounds,
        cache=cache,
        certificate_store=certificate_store,
    )
    return result.assembled_source, result
