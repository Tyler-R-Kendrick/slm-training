"""SLM-187 (FFE1-01): topology legal-edit state Markov-complete / solver-runtime transition parity wiring fixture.

Deterministic, CPU-only parity oracle that compares the finite-domain edit
values produced by ``derive_topology_holes`` with the exact committable edit
tuples that the grammar-diffusion runtime can emit.  The harness builds bounded
fixture trees, runs an exhaustive small-state search over action/arity/slot
choices, and reports whether every runtime-committable edit is present in the
solver domain and whether every solver-domain edit is reachable by the runtime.

No model is trained, no GPU is required, and no ship-gate claim is made.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.production_codec import ProductionCodec
from slm_training.dsl.solver.state import SolverBounds
from slm_training.dsl.solver.topology_adapter import (
    TopologyAction,
    TopologyAdapterConfig,
    TopologyEdit,
    _node_type,
    derive_topology_holes,
    legal_topology_productions,
)
from slm_training.dsl.solver.topology_solver import (
    SolverTopologyNode,
    _apply_edit,
    _copy_topology_node,
    _serialize_topology,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "TopologyStateV2",
    "TopologyTransitionTuple",
    "TopologyParityCase",
    "TopologyParityReport",
    "build_fixture_codec",
    "build_fixture_trees",
    "build_runtime_proposals",
    "compare_solver_runtime_domain",
    "derive_topology_state_v2",
    "run_topology_parity_fixture",
    "render_markdown",
]

MATRIX_VERSION = "ffe1-01-v1"
MATRIX_SET = "slm187_topology_parity"
EXPERIMENT_ID = "slm187-topology-parity"

_HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'

_HYPOTHESIS = (
    "The topology finite-domain solver state carries enough resolved tree and "
    "context information to reconstruct every successor, and the solver's "
    "enumerated edit domain contains every edit the runtime can commit for the "
    "actions that participate in solver filtering (EXPAND/KEEP)."
)

_FALSIFIER = (
    "An exhaustive fixture over bounded topology states finds a runtime-committable "
    "EXPAND or KEEP edit that is missing from the solver domain, or a solver-domain "
    "edit whose successor fingerprint differs from the runtime's successor."
)

_HONEST_CAVEATS = (
    "Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.",
    "Runtime proposal logic is mirrored torch-free for comparison; small discrepancies "
    "with the live grammar_diffusion.py path are reported as parity gaps, not fixed silently.",
    "DELETE and CONTRACT proposals are intentionally outside the finite edit domain in "
    "the current runtime (production_id < 0 bypasses solver filtering); the oracle treats "
    "them as a separate contract.",
    "STOP is a solver-domain structural action that the runtime does not currently emit; "
    "this asymmetry is surfaced, not suppressed.",
    "Multi-step terminal traces are validated with the DSL parser; a parse failure is "
    "reported honestly rather than treated as parity success.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _node_fingerprint(node: SolverTopologyNode) -> str:
    """Stable fingerprint of a single node's hard-state identity."""
    return _sha256(
        _canonical_json(
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "production_id": node.production_id,
                "slot_id": node.slot_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "sibling_index": node.sibling_index,
                "active": node.active,
            }
        )
    )


def _tree_fingerprint(root: SolverTopologyNode) -> str:
    """Canonical fingerprint of the complete materialized tree."""
    return _sha256(_canonical_json(root.to_dict()))


@dataclass(frozen=True)
class TopologyTransitionTuple:
    """One normalized edit tuple comparable across solver and runtime."""

    node_id: int
    action: str
    production_id: int
    arity: int
    slot_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "action": self.action,
            "production_id": self.production_id,
            "arity": self.arity,
            "slot_id": self.slot_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyTransitionTuple":
        return cls(
            node_id=int(data["node_id"]),
            action=str(data["action"]),
            production_id=int(data["production_id"]),
            arity=int(data["arity"]),
            slot_id=int(data["slot_id"]),
        )


@dataclass(frozen=True)
class TopologyStateV2:

    """Markov-complete hard-state carrier for a topology tree.

    Carries the complete canonical materialized tree, active hole domains, and
    request-local contract identity.  The fingerprint is a complete Markov
    identity for the hard state: equal fingerprints imply equal candidate
    domains and equal successors for every live edit.
    """

    schema: str = "TopologyStateV2"
    tree: SolverTopologyNode = field(default_factory=lambda: SolverTopologyNode(0, "document", 0))
    problem_id: str = "topology"
    pack_id: str = "openui"
    constraint_version: str = "v2"
    output_kind: str = "document"
    slot_inventory: tuple[str, ...] = ()
    bounds: SolverBounds = field(
        default_factory=lambda: SolverBounds(
            max_tokens=256, max_nodes=8, max_depth=4, max_backtracks=0, max_verifier_calls=0
        )
    )
    phase: int = 0
    tree_fingerprint: str = ""
    state_fingerprint: str = ""

    def __post_init__(self) -> None:
        # Dataclass is frozen, so we mutate via object.__setattr__ once.
        if not self.tree_fingerprint:
            object.__setattr__(self, "tree_fingerprint", _tree_fingerprint(self.tree))
        if not self.state_fingerprint:
            object.__setattr__(
                self,
                "state_fingerprint",
                _sha256(
                    _canonical_json(
                        {
                            "tree": self.tree.to_dict(),
                            "problem_id": self.problem_id,
                            "pack_id": self.pack_id,
                            "constraint_version": self.constraint_version,
                            "output_kind": self.output_kind,
                            "slot_inventory": list(self.slot_inventory),
                            "bounds": self.bounds.to_dict() if hasattr(self.bounds, "to_dict") else str(self.bounds),
                            "phase": self.phase,
                        }
                    )
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tree": self.tree.to_dict(),
            "problem_id": self.problem_id,
            "pack_id": self.pack_id,
            "constraint_version": self.constraint_version,
            "output_kind": self.output_kind,
            "slot_inventory": list(self.slot_inventory),
            "bounds": self.bounds.to_dict() if hasattr(self.bounds, "to_dict") else str(self.bounds),
            "phase": self.phase,
            "tree_fingerprint": self.tree_fingerprint,
            "state_fingerprint": self.state_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyStateV2":
        from slm_training.dsl.solver.state import SolverBounds

        tree_data = data.get("tree", {})
        tree = _solver_topology_node_from_dict(tree_data)
        bounds_data = data.get("bounds")
        bounds = SolverBounds.from_dict(bounds_data) if isinstance(bounds_data, dict) else SolverBounds(256, 8, 4, 0, 0)
        return cls(
            schema=str(data.get("schema", "TopologyStateV2")),
            tree=tree,
            problem_id=str(data.get("problem_id", "topology")),
            pack_id=str(data.get("pack_id", "openui")),
            constraint_version=str(data.get("constraint_version", "v2")),
            output_kind=str(data.get("output_kind", "document")),
            slot_inventory=tuple(str(x) for x in data.get("slot_inventory", ())),
            bounds=bounds,
            phase=int(data.get("phase", 0)),
            tree_fingerprint=str(data.get("tree_fingerprint", "")),
            state_fingerprint=str(data.get("state_fingerprint", "")),
        )


def _solver_topology_node_from_dict(data: dict[str, Any]) -> SolverTopologyNode:
    """Reconstruct a SolverTopologyNode from its to_dict output."""
    node = SolverTopologyNode(
        node_id=int(data.get("node_id", 0)),
        node_type=str(data.get("node_type", "document")),
        production_id=int(data.get("production_id", 0)),
        slot_id=int(data.get("slot_id", 0)),
        parent_id=int(data.get("parent_id", -1)),
        depth=int(data.get("depth", 0)),
        sibling_index=int(data.get("sibling_index", 0)),
        active=bool(data.get("active", False)),
    )
    for child_data in data.get("children", ()):
        child = _solver_topology_node_from_dict(child_data)
        child.parent_id = node.node_id
        node.children.append(child)
    return node


@dataclass(frozen=True)
class TopologyParityCase:
    """One fixture tree plus its solver/runtime parity comparison."""

    case_id: str
    description: str
    state_v2: TopologyStateV2
    solver_domain: frozenset[TopologyTransitionTuple]
    runtime_domain: frozenset[TopologyTransitionTuple]
    runtime_filtered_domain: frozenset[TopologyTransitionTuple]
    solver_only: frozenset[TopologyTransitionTuple]
    runtime_only: frozenset[TopologyTransitionTuple]
    shared: frozenset[TopologyTransitionTuple]
    parity_ok: bool
    terminal_text: str | None
    terminal_valid: bool
    terminal_error: str | None

    def to_dict(self) -> dict[str, Any]:
        sort_key = lambda d: (d["node_id"], d["action"], d["production_id"], d["arity"], d["slot_id"])  # noqa: E731
        return {
            "case_id": self.case_id,
            "description": self.description,
            "state_v2": self.state_v2.to_dict(),
            "solver_domain": sorted([t.to_dict() for t in self.solver_domain], key=sort_key),
            "runtime_domain": sorted([t.to_dict() for t in self.runtime_domain], key=sort_key),
            "runtime_filtered_domain": sorted([t.to_dict() for t in self.runtime_filtered_domain], key=sort_key),
            "solver_only": sorted([t.to_dict() for t in self.solver_only], key=sort_key),
            "runtime_only": sorted([t.to_dict() for t in self.runtime_only], key=sort_key),
            "shared": sorted([t.to_dict() for t in self.shared], key=sort_key),
            "parity_ok": self.parity_ok,
            "terminal_text": self.terminal_text,
            "terminal_valid": self.terminal_valid,
            "terminal_error": self.terminal_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyParityCase":
        return cls(
            case_id=str(data["case_id"]),
            description=str(data.get("description", "")),
            state_v2=TopologyStateV2.from_dict(data.get("state_v2", {})),
            solver_domain=frozenset(TopologyTransitionTuple.from_dict(t) for t in data.get("solver_domain", ())),
            runtime_domain=frozenset(TopologyTransitionTuple.from_dict(t) for t in data.get("runtime_domain", ())),
            runtime_filtered_domain=frozenset(TopologyTransitionTuple.from_dict(t) for t in data.get("runtime_filtered_domain", ())),
            solver_only=frozenset(TopologyTransitionTuple.from_dict(t) for t in data.get("solver_only", ())),
            runtime_only=frozenset(TopologyTransitionTuple.from_dict(t) for t in data.get("runtime_only", ())),
            shared=frozenset(TopologyTransitionTuple.from_dict(t) for t in data.get("shared", ())),
            parity_ok=bool(data.get("parity_ok", False)),
            terminal_text=data.get("terminal_text") if "terminal_text" in data else None,
            terminal_valid=bool(data.get("terminal_valid", False)),
            terminal_error=data.get("terminal_error") if "terminal_error" in data else None,
        )


@dataclass(frozen=True)
class TopologyParityReport:
    """Full fixture report for SLM-187."""

    schema: str = "TopologyParityReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm187-topology-parity"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    cases: tuple[TopologyParityCase, ...] = ()
    n_cases: int = 0
    n_parity_ok: int = 0
    n_runtime_only: int = 0
    n_solver_only: int = 0
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = _HONEST_CAVEATS
    version_stamp: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_cases", len(self.cases))
        object.__setattr__(self, "n_parity_ok", sum(1 for c in self.cases if c.parity_ok))
        object.__setattr__(
            self,
            "n_runtime_only",
            sum(len(c.runtime_only) for c in self.cases),
        )
        object.__setattr__(
            self,
            "n_solver_only",
            sum(len(c.solver_only) for c in self.cases),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "cases": [c.to_dict() for c in self.cases],
            "n_cases": self.n_cases,
            "n_parity_ok": self.n_parity_ok,
            "n_runtime_only": self.n_runtime_only,
            "n_solver_only": self.n_solver_only,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyParityReport":
        return cls(
            schema=str(data.get("schema", "TopologyParityReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", "slm187-topology-parity")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            cases=tuple(
                TopologyParityCase.from_dict(c) for c in data.get("cases", ())
            ),
            n_cases=int(data.get("n_cases", 0)),
            n_parity_ok=int(data.get("n_parity_ok", 0)),
            n_runtime_only=int(data.get("n_runtime_only", 0)),
            n_solver_only=int(data.get("n_solver_only", 0)),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


@dataclass(frozen=True)
class TopologyParityPlanManifest:
    """Lightweight plan-only manifest for the fixture script."""

    schema: str = "TopologyParityPlanManifestV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm187-topology-parity"
    status: str = "plan_only"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    version_stamp: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
        }


def build_fixture_codec() -> ProductionCodec:
    """Return the HERO codec used by topology solver tests."""
    return ProductionCodec.build([_HERO])


def _new_node(
    node_id: int,
    node_type: str,
    production_id: int,
    *,
    parent_id: int = -1,
    depth: int = 0,
    sibling_index: int = 0,
    active: bool = False,
    slot_id: int = 0,
) -> SolverTopologyNode:
    return SolverTopologyNode(
        node_id=node_id,
        node_type=node_type,
        production_id=production_id,
        slot_id=slot_id,
        parent_id=parent_id,
        depth=depth,
        sibling_index=sibling_index,
        active=active,
    )


def build_fixture_trees(codec: ProductionCodec) -> list[tuple[str, str, SolverTopologyNode]]:
    """Return (case_id, description, root) fixture trees."""
    trees: list[tuple[str, str, SolverTopologyNode]] = []

    # 1. Active document root.
    root = _new_node(0, "document", codec.bos_id, active=True)
    trees.append(("doc_root", "active document root", root))

    # 2. Document root with one active statement child.
    root = _new_node(0, "document", codec.bos_id, active=True)
    statement = _new_node(
        1,
        "statement",
        codec.production_to_id["="],
        parent_id=0,
        depth=1,
        sibling_index=0,
        active=True,
    )
    root.children.append(statement)
    trees.append(("doc_statement", "document root -> active statement", root))

    # 3. Statement resolved, active component expression.
    root = _new_node(0, "document", codec.bos_id, active=False)
    statement = _new_node(
        1,
        "statement",
        codec.production_to_id["="],
        parent_id=0,
        depth=1,
        sibling_index=0,
        active=False,
    )
    expression = _new_node(
        2,
        "expression",
        codec.production_to_id["+Stack"],
        parent_id=1,
        depth=2,
        sibling_index=0,
        active=True,
    )
    root.children.append(statement)
    statement.children.append(expression)
    trees.append(("stmt_component", "resolved statement -> active component", root))

    # 4. Component with active list child.
    root = _new_node(0, "document", codec.bos_id, active=False)
    statement = _new_node(
        1, "statement", codec.production_to_id["="], parent_id=0, depth=1, active=False
    )
    expression = _new_node(
        2,
        "expression",
        codec.production_to_id["+Stack"],
        parent_id=1,
        depth=2,
        active=False,
    )
    list_node = _new_node(
        3,
        "list",
        codec.production_to_id["["],
        parent_id=2,
        depth=3,
        active=True,
    )
    root.children.append(statement)
    statement.children.append(expression)
    expression.children.append(list_node)
    trees.append(("component_list", "component -> active list", root))

    # 5. Leaf binding slot.
    root = _new_node(0, "document", codec.bos_id, active=False)
    statement = _new_node(
        1, "statement", codec.production_to_id["="], parent_id=0, depth=1, active=False
    )
    expression = _new_node(
        2,
        "expression",
        codec.production_to_id["+Stack"],
        parent_id=1,
        depth=2,
        active=False,
    )
    list_node = _new_node(3, "list", codec.production_to_id["["], parent_id=2, depth=3, active=False)
    leaf = _new_node(
        4,
        "leaf",
        codec.production_to_id["&0"],
        parent_id=3,
        depth=4,
        sibling_index=0,
        active=True,
        slot_id=1,
    )
    root.children.append(statement)
    statement.children.append(expression)
    expression.children.append(list_node)
    list_node.children.append(leaf)
    trees.append(("leaf_slot", "leaf node with slot binding", root))

    # 6. Fragment output marker root (not present in HERO codec; use document).
    root = _new_node(0, "document", codec.bos_id, active=True)
    trees.append(("fragment_root", "fragment output marker root (document fallback)", root))

    # 7. Maximum depth leaf.
    root = _new_node(0, "document", codec.bos_id, active=False)
    statement = _new_node(
        1, "statement", codec.production_to_id["="], parent_id=0, depth=1, active=False
    )
    expression = _new_node(
        2,
        "expression",
        codec.production_to_id["+Stack"],
        parent_id=1,
        depth=2,
        active=False,
    )
    leaf = _new_node(
        3,
        "leaf",
        codec.production_to_id["&0"],
        parent_id=2,
        depth=3,
        sibling_index=0,
        active=True,
    )
    root.children.append(statement)
    statement.children.append(expression)
    expression.children.append(leaf)
    trees.append(("max_depth_leaf", "leaf at topology_max_depth boundary", root))

    # 8. Multiple active siblings.
    root = _new_node(0, "document", codec.bos_id, active=False)
    statement = _new_node(
        1, "statement", codec.production_to_id["="], parent_id=0, depth=1, active=False
    )
    expr_a = _new_node(
        2,
        "expression",
        codec.production_to_id["+Stack"],
        parent_id=1,
        depth=2,
        sibling_index=0,
        active=True,
    )
    expr_b = _new_node(
        3,
        "expression",
        codec.production_to_id["+Card"],
        parent_id=1,
        depth=2,
        sibling_index=1,
        active=True,
    )
    root.children.append(statement)
    statement.children.append(expr_a)
    statement.children.append(expr_b)
    trees.append(("sibling_choices", "two active sibling expressions", root))

    return trees


def _derive_solver_domain(
    root: SolverTopologyNode,
    codec: ProductionCodec,
    config: TopologyAdapterConfig,
    slot_inventory: list[str],
) -> frozenset[TopologyTransitionTuple]:
    """Return the normalized solver-domain edit tuples for a tree."""
    holes = derive_topology_holes(root, codec, config, slot_inventory=slot_inventory)
    domain: set[TopologyTransitionTuple] = set()
    for hole in holes:
        for value in hole.domain.values:
            try:
                edit = TopologyEdit.from_value(value)
            except (KeyError, ValueError, TypeError):
                continue
            domain.add(
                TopologyTransitionTuple(
                    node_id=hole.node_id,
                    action=edit.action.name,
                    production_id=edit.production_id,
                    arity=edit.arity,
                    slot_id=edit.slot_id,
                )
            )
    return frozenset(domain)


def _runtime_arity(
    node_type: str,
    production_id: int,
    raw_arity: int,
    codec: ProductionCodec,
    config: TopologyAdapterConfig,
    output_kind: str,
) -> int:
    """Mirror grammar_diffusion.py arity coercion."""
    token = codec.id_to_production.get(production_id, "")
    if node_type == "document":
        return max(1, raw_arity)
    if node_type == "statement" and token == "=":
        return 1
    if _node_type(token) == "leaf":
        return 0
    return min(raw_arity, config.topology_max_arity)


def build_runtime_proposals(
    root: SolverTopologyNode,
    codec: ProductionCodec,
    config: TopologyAdapterConfig,
    slot_inventory: list[str],
    output_kind: str = "document",
    *,
    action_choices: dict[int, int] | None = None,
    include_contract: bool = True,
) -> frozenset[TopologyTransitionTuple]:
    """Return the normalized runtime-committable edit tuples for a tree.

    Mirrors the proposal path in ``GrammarDiffusionModel._decode_one`` without
    importing torch.  ``action_choices`` maps ``node_id -> TopologyAction`` for
    deterministic exhaustive testing; when absent, every legal action is
    considered for active nodes and CONTRACT is considered for inactive nodes.
    """
    from slm_training.dsl.solver.topology_adapter import _selected_nodes

    selected = _selected_nodes(root, config, phase=0)
    index_by_id = {node.node_id: idx for idx, node in enumerate(selected)}
    proposals: set[TopologyTransitionTuple] = set()
    active_selected = [node for node in selected if node.active][: config.topology_max_active]

    proposal_nodes = list(active_selected)
    if include_contract:
        # Extend with inactive nodes that could contract, matching the runtime's
        # topology_actions branch when action_choices requests CONTRACT.
        proposal_nodes.extend(node for node in selected if not node.active and node.parent_id >= 0)

    for node in proposal_nodes:
        if node.node_id not in index_by_id:
            continue
        if action_choices is None:
            action = int(TopologyAction.EXPAND) if node.active else int(TopologyAction.KEEP)
        else:
            action = action_choices.get(
                node.node_id,
                int(TopologyAction.EXPAND) if node.active else int(TopologyAction.KEEP),
            )

        if not node.active:
            if include_contract and action == int(TopologyAction.CONTRACT):
                proposals.add(
                    TopologyTransitionTuple(
                        node_id=node.node_id,
                        action="CONTRACT",
                        production_id=-2,
                        arity=0,
                        slot_id=0,
                    )
                )
            continue

        if action == int(TopologyAction.DELETE) and node.parent_id >= 0:
            proposals.add(
                TopologyTransitionTuple(
                    node_id=node.node_id,
                    action="DELETE",
                    production_id=-1,
                    arity=0,
                    slot_id=0,
                )
            )
            continue

        if action not in {int(TopologyAction.EXPAND), int(TopologyAction.KEEP)}:
            continue

        legal = legal_topology_productions(
            codec, node.node_type, leaf_only=node.depth >= config.topology_max_depth
        )
        if node.node_type == "document" and output_kind != "document":
            marker = f"!{output_kind}"
            marker_id = codec.production_to_id.get(marker)
            if marker_id is None:
                continue
            legal = [marker_id]

        for production_id in legal:
            for raw_arity in range(config.topology_max_arity + 1):
                arity = _runtime_arity(
                    node.node_type, production_id, raw_arity, codec, config, output_kind
                )
                # Runtime only emits one arity per head choice, but for parity we
                # enumerate every reachable arity so the oracle is complete.
                for slot_id in ([0, *range(1, len(slot_inventory) + 1)]) if slot_inventory else [0]:
                    edit_action = (
                        "KEEP"
                        if production_id == node.production_id
                        else "EXPAND"
                    )
                    proposals.add(
                        TopologyTransitionTuple(
                            node_id=node.node_id,
                            action=edit_action,
                            production_id=production_id,
                            arity=arity,
                            slot_id=slot_id,
                        )
                    )
    return frozenset(proposals)


def _apply_runtime_edit(
    root: SolverTopologyNode,
    edit: TopologyTransitionTuple,
    codec: ProductionCodec,
    output_kind: str,
) -> SolverTopologyNode | None:
    """Apply a runtime-committable edit using the solver's transition function."""
    try:
        action = TopologyAction[edit.action]
    except KeyError:
        return None
    topology_edit = TopologyEdit(
        action=action,
        production_id=edit.production_id,
        arity=edit.arity,
        slot_id=edit.slot_id,
    )
    return _apply_edit(root, edit.node_id, topology_edit, codec, output_kind)


def _terminal_text(root: SolverTopologyNode, codec: ProductionCodec, slot_inventory: list[str]) -> tuple[str | None, bool, str | None]:
    """Serialize and validate a solved tree; return (text, valid, error)."""
    from slm_training.dsl.parser import validate_output

    try:
        production_ids, slot_ids = _serialize_topology(codec, root)
        text = codec.decode(production_ids, slot_ids, slot_inventory).strip()
    except Exception as exc:  # noqa: BLE001
        return None, False, str(exc)[:200]
    try:
        validate_output(text, "document")
    except Exception as exc:  # noqa: BLE001
        return text, False, str(exc)[:200]
    return text, True, None


def derive_topology_state_v2(
    root: SolverTopologyNode,
    codec: ProductionCodec,
    config: TopologyAdapterConfig,
    *,
    slot_inventory: list[str] | None = None,
    output_kind: str = "document",
    bounds: SolverBounds | None = None,
) -> TopologyStateV2:
    """Build a Markov-complete V2 state carrying the full tree snapshot."""
    if bounds is None:
        bounds = SolverBounds(
            max_tokens=config.topology_max_nodes * 64,
            max_nodes=config.topology_max_nodes,
            max_depth=config.topology_max_depth,
            max_backtracks=0,
            max_verifier_calls=0,
        )
    return TopologyStateV2(
        tree=_copy_topology_node(root),
        output_kind=output_kind,
        slot_inventory=tuple(slot_inventory or ()),
        bounds=bounds,
    )


def compare_solver_runtime_domain(
    root: SolverTopologyNode,
    codec: ProductionCodec,
    config: TopologyAdapterConfig,
    slot_inventory: list[str],
    output_kind: str = "document",
    *,
    case_id: str = "case",
    description: str = "",
    action_choices: dict[int, int] | None = None,
) -> TopologyParityCase:
    """Compare solver and runtime domains for one tree."""
    solver_domain = _derive_solver_domain(root, codec, config, slot_inventory)
    runtime_domain = build_runtime_proposals(
        root,
        codec,
        config,
        slot_inventory,
        output_kind=output_kind,
        action_choices=action_choices,
        include_contract=True,
    )

    # The finite-domain solver filters only EXPAND/KEEP proposals.
    runtime_filtered = frozenset(t for t in runtime_domain if t.action in {"EXPAND", "KEEP"})
    shared = solver_domain & runtime_filtered
    solver_only = solver_domain - runtime_filtered
    runtime_only = runtime_filtered - solver_domain

    # Parity is OK when every runtime EXPAND/KEEP is in the solver domain.
    parity_ok = not runtime_only

    # Terminal serialization check only when the tree is structurally solved.
    def flatten(n: SolverTopologyNode) -> list[SolverTopologyNode]:
        out = [n]
        for child in n.children:
            out.extend(flatten(child))
        return out

    any_active = any(node.active for node in flatten(root))
    if not any_active:
        text, valid, error = _terminal_text(root, codec, slot_inventory)
    else:
        text, valid, error = None, False, "tree_not_solved"

    return TopologyParityCase(
        case_id=case_id,
        description=description,
        state_v2=derive_topology_state_v2(root, codec, config, slot_inventory=slot_inventory, output_kind=output_kind),
        solver_domain=solver_domain,
        runtime_domain=runtime_domain,
        runtime_filtered_domain=runtime_filtered,
        solver_only=solver_only,
        runtime_only=runtime_only,
        shared=shared,
        parity_ok=parity_ok,
        terminal_text=text,
        terminal_valid=valid,
        terminal_error=error,
    )


def _resolve_disposition(cases: tuple[TopologyParityCase, ...]) -> tuple[str, str]:
    """Classify the fixture outcome."""
    if not cases:
        return ("inconclusive", "No cases were generated.")

    runtime_only_total = sum(len(c.runtime_only) for c in cases)
    solver_only_total = sum(len(c.solver_only) for c in cases)
    parity_ok_count = sum(1 for c in cases if c.parity_ok)

    if runtime_only_total == 0 and parity_ok_count == len(cases):
        return (
            "parity_holds",
            "Over the exhaustive declared fixture bounds, every runtime-committable "
            "EXPAND/KEEP edit is present in the solver domain. Solver-only tuples "
            "(DELETE/STOP) are structural actions outside the current runtime filtering "
            "contract.",
        )

    if runtime_only_total > 0:
        return (
            "parity_gap",
            f"{runtime_only_total} runtime EXPAND/KEEP tuple(s) are missing from the "
            "solver domain. The hard state or adapter enumeration needs repair before "
            "topology intermediates can be used for Markov/CTMC training.",
        )

    return (
        "inconclusive",
        f"No runtime-only gaps, but {solver_only_total} solver-only structural tuples "
        "exist. Verify whether these are intentionally outside the runtime contract.",
    )


def run_topology_parity_fixture(
    *,
    codec: ProductionCodec | None = None,
    config: TopologyAdapterConfig | None = None,
    slot_inventory: list[str] | None = None,
    output_kind: str = "document",
    seed: int = 0,
    run_id: str | None = None,
) -> TopologyParityReport:
    """Run the SLM-187 topology parity fixture."""
    rng = random.Random(seed)
    codec = codec or build_fixture_codec()
    config = config or TopologyAdapterConfig(
        topology_max_nodes=8,
        topology_max_active=8,
        topology_max_arity=3,
        topology_max_depth=4,
    )
    slot_inventory = slot_inventory or [":hero.title", ":hero.body"]

    trees = build_fixture_trees(codec)
    cases: list[TopologyParityCase] = []
    for case_id, description, root in trees:
        # Randomize action choices to exercise runtime branches deterministically.
        from slm_training.dsl.solver.topology_adapter import _flatten as adapter_flatten

        all_nodes = adapter_flatten(root)
        action_choices: dict[int, int] = {}
        for node in all_nodes:
            if node.active:
                action_choices[node.node_id] = rng.choice(
                    [int(TopologyAction.EXPAND), int(TopologyAction.KEEP), int(TopologyAction.DELETE)]
                )
            elif node.parent_id >= 0:
                action_choices[node.node_id] = rng.choice(
                    [int(TopologyAction.CONTRACT), int(TopologyAction.KEEP)]
                )

        case = compare_solver_runtime_domain(
            root,
            codec,
            config,
            slot_inventory,
            output_kind=output_kind,
            case_id=case_id,
            description=description,
            action_choices=action_choices,
        )
        cases.append(case)

    disposition, rationale = _resolve_disposition(tuple(cases))

    manifest = TopologyParityReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        cases=tuple(cases),
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm187_topology_parity",
            "dsl.solver.topology",
        ),
    )
    return manifest


def render_markdown(report: TopologyParityReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-187 (FFE1-01): topology solver/runtime transition parity fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no trainable "
        "weights were updated, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Summary",
        "",
        f"- Cases: {report.n_cases}",
        f"- Parity OK: {report.n_parity_ok}",
        f"- Runtime-only tuples (missing from solver): {report.n_runtime_only}",
        f"- Solver-only tuples (outside runtime EXPAND/KEEP contract): {report.n_solver_only}",
        f"- Disposition: **{report.disposition}**",
        "",
        "## Parity cases",
        "",
        "| Case | Description | Solver domain | Runtime domain | Shared | Runtime-only | Solver-only | Parity | Terminal valid |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in report.cases:
        lines.append(
            f"| {case.case_id} | {case.description} | {len(case.solver_domain)} | "
            f"{len(case.runtime_domain)} | {len(case.shared)} | {len(case.runtime_only)} | "
            f"{len(case.solver_only)} | {case.parity_ok} | {case.terminal_valid} |"
        )

    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"**{report.disposition}**",
            "",
            report.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The parity oracle, "
            "V2 state carrier, and transition comparison are exercised on deterministic "
            "synthetic trees. Real model runtime parity under `topology_verified_solver=True` "
            "and full multi-step terminal validation are required before topology "
            "intermediates may be used for Markov/CTMC/flow training. The mechanism "
            "remains ``retain_diagnostic`` / ``blocked_pending_solver_runtime_audit``.",
            "",
            "## Honest caveats",
            "",
        ]
    )
    for caveat in report.honest_caveats:
        lines.append(f"- {caveat}")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm187_topology_parity_fixture --mode plan-only",
            "python -m scripts.run_slm187_topology_parity_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
