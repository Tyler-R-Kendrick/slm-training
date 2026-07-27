"""BinderGraphV1: definitions, uses, directed reference edges, scopes, and
unresolved placeholders for a ProgramSpec's binder/reference structure
(AP-026 / SLM-320).

This is the structural (oracle) layer: it wraps the same node/edge walk
:func:`~slm_training.data.progspec.capsules.derive_capsule_graph` already
uses (``ScopeNode``/``ScopeEdge``/``DependencyKind``), but never raises on a
forward or undefined reference -- it collects one as data
(:class:`UnresolvedPlaceholder`) instead, so a program that isn't fully
resolved yet is still representable rather than a hard failure. Whether a
*predicted* binder graph may condition decoding lives one layer up in
:mod:`slm_training.models.binder_precision_channel`, which only ever treats
graph structure as a ranking signal -- legality is decided by
:meth:`BinderGraphV1.check_g3`, matching the deterministic G3 reference-graph
gate contract in ``docs/design/verifier-stack.md`` ("one root, resolved
binders, reachability, no cycles") and decode-invariant I3 ("ranking is a
lever; legality is not").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from slm_training.data.progspec.capsules import (
    DependencyKind,
    ScopeEdge,
    ScopeNode,
    UnresolvedReference,
    derive_node_edge_walk,
)
from slm_training.data.progspec.schema import ProgramSpec

__all__ = [
    "BinderGraphG3Result",
    "BinderGraphV1",
    "UnresolvedPlaceholder",
    "derive_binder_graph",
]


@dataclass(frozen=True)
class UnresolvedPlaceholder:
    """A reference occurrence with no defining binder in this program.

    Distinct from a ``:slot.path`` template placeholder (an intentional,
    externally-filled input, modeled as an ``EXTERNAL`` edge on
    :class:`~slm_training.data.progspec.capsules.ScopeNode`): this is a named
    binder reference (``{"type": "ref", "name": ...}``) that never resolves
    against any statement definition in the program -- a forward reference,
    a typo, or a binder pruned out from under a still-live use.
    """

    node_id: str
    name: str
    ast_path: tuple[str | int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "ast_path": list(self.ast_path),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UnresolvedPlaceholder:
        return cls(
            node_id=str(data["node_id"]),
            name=str(data["name"]),
            ast_path=tuple(data.get("ast_path") or ()),
        )

    @classmethod
    def _from_walk(cls, ref: UnresolvedReference) -> UnresolvedPlaceholder:
        return cls(node_id=ref.node_id, name=ref.name, ast_path=ref.ast_path)


@dataclass(frozen=True)
class BinderGraphG3Result:
    """G3 ("reference graph": one root, resolved binders, reachability, no
    cycles -- ``docs/design/verifier-stack.md``) evaluated over a
    :class:`BinderGraphV1` structure, independent of the string-level
    ``Gate.REFERENCES`` check in ``data/verify/stack.py`` (which operates on
    flat ``name = expression`` source text rather than the parsed AST)."""

    ok: bool
    single_root: bool
    all_resolved: bool
    reachable: bool
    acyclic: bool
    unreachable_node_ids: tuple[str, ...]
    cycle: tuple[str, ...] | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["unreachable_node_ids"] = list(self.unreachable_node_ids)
        data["cycle"] = list(self.cycle) if self.cycle is not None else None
        return data


@dataclass(frozen=True)
class BinderGraphV1:
    """Binder definitions/uses/edges/scopes plus unresolved placeholders.

    ``nodes``/``edges`` reuse the exact ``ScopeNode``/``ScopeEdge``/
    ``DependencyKind`` vocabulary already produced by
    :func:`~slm_training.data.progspec.capsules.derive_capsule_graph`, so a
    program with no unresolved references yields a graph whose ``nodes`` and
    ``edges`` are identical to the corresponding ``CapsuleGraph`` (see
    ``tests/test_data/test_binder_graph.py::test_matches_capsule_graph_when_fully_resolved``).
    """

    root_id: str
    nodes: tuple[ScopeNode, ...]
    edges: tuple[ScopeEdge, ...]
    unresolved: tuple[UnresolvedPlaceholder, ...]
    spec_id: str
    version: str

    VERSION = "binder-graph-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "unresolved": [u.to_dict() for u in self.unresolved],
            "spec_id": self.spec_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BinderGraphV1:
        return cls(
            root_id=str(data["root_id"]),
            nodes=tuple(ScopeNode.from_dict(d) for d in data.get("nodes", [])),
            edges=tuple(ScopeEdge.from_dict(d) for d in data.get("edges", [])),
            unresolved=tuple(
                UnresolvedPlaceholder.from_dict(d) for d in data.get("unresolved", [])
            ),
            spec_id=str(data.get("spec_id", "")),
            version=str(data.get("version", cls.VERSION)),
        )

    def definitions(self) -> tuple[str, ...]:
        """All binder names defined by some node, sorted."""
        return tuple(
            sorted({name for node in self.nodes for name in node.definitions})
        )

    def reference_edges(self) -> tuple[ScopeEdge, ...]:
        return tuple(edge for edge in self.edges if edge.kind == DependencyKind.REFERENCE)

    def check_g3(self) -> BinderGraphG3Result:
        """Evaluate the G3 reference-graph contract over this structure.

        A predicted/learned :class:`BinderGraphV1` (see
        ``models/binder_precision_channel.py``) is only ever eligible to
        condition decoding when this reports ``ok=True`` -- a gated
        connector must never treat an unconfirmed learned edge as legal.

        Reachability is undirected connectivity from the root position: the
        shared node/edge walk this graph is built from (see
        ``capsules.derive_node_edge_walk``) directs a REFERENCE edge from
        the *referencing* occurrence toward the statement that defines it,
        not from the program's entry point outward, so a node with any edge
        touching it -- in either direction -- is "part of the program", and
        a node with none is a genuine orphan. The walk also never emits a
        ROOT_OUTPUT edge (``"root"`` is filtered out of every definitions
        set, so ``definition_to_node["root"]`` never exists), leaving the
        synthetic document-root node and the statement node for the
        program's own ``root = ...`` assignment structurally unconnected
        even though both sit at ``ast_path == ()`` -- the walk's entry
        position is seeded with both rather than just the synthetic node.
        """
        root_nodes = tuple(node.node_id for node in self.nodes if node.kind == "root")
        single_root = len(root_nodes) == 1
        entry_ids = tuple(node.node_id for node in self.nodes if node.ast_path == ())
        all_resolved = not self.unresolved

        reachable_ids: set[str] = set()
        if single_root:
            adjacency: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
            for edge in self.edges:
                if edge.source in adjacency and edge.target in adjacency:
                    adjacency[edge.source].append(edge.target)
                    adjacency[edge.target].append(edge.source)
            stack = list(entry_ids)
            while stack:
                current = stack.pop()
                if current in reachable_ids:
                    continue
                reachable_ids.add(current)
                stack.extend(adjacency.get(current, ()))
        unreachable = tuple(
            sorted(node.node_id for node in self.nodes if node.node_id not in reachable_ids)
        )
        reachable = not unreachable

        cycle = self._find_cycle()
        acyclic = cycle is None

        ok = single_root and all_resolved and reachable and acyclic
        return BinderGraphG3Result(
            ok=ok,
            single_root=single_root,
            all_resolved=all_resolved,
            reachable=reachable,
            acyclic=acyclic,
            unreachable_node_ids=unreachable,
            cycle=cycle,
        )

    def _find_cycle(self) -> tuple[str, ...] | None:
        """Return one reference-edge cycle (node-id sequence) if present."""
        adjacency: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for edge in self.reference_edges():
            if edge.source in adjacency:
                adjacency[edge.source].append(edge.target)

        visiting: list[str] = []
        visiting_set: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> tuple[str, ...] | None:
            if node_id in visiting_set:
                start = visiting.index(node_id)
                return tuple(visiting[start:]) + (node_id,)
            if node_id in visited:
                return None
            visiting.append(node_id)
            visiting_set.add(node_id)
            for target in sorted(adjacency.get(node_id, ())):
                found = visit(target)
                if found is not None:
                    return found
            visiting.pop()
            visiting_set.discard(node_id)
            visited.add(node_id)
            return None

        for node_id in sorted(adjacency):
            if node_id not in visited:
                found = visit(node_id)
                if found is not None:
                    return found
        return None

    def alpha_fingerprint(self) -> str:
        """A structural fingerprint invariant to binder surface spelling.

        Renames every binder name to a canonical positional id (definition
        order over sorted node ids), then hashes node kinds/scope-shape and
        edge kinds/positions -- never the surface names themselves. Two
        graphs that differ only by consistently renaming their binders
        produce the same fingerprint; this mirrors
        ``slm_training.dsl.scope_env.ScopeEnv.fingerprint``'s alpha-rename
        convention (surface spelling is realization data, not identity).

        Node identity is canonicalized by ``ast_path`` sort order (a purely
        structural position, independent of ``spec_id`` or binder spelling)
        rather than by the raw ``node_id`` string, since ``node_id`` embeds
        ``spec_id`` and would otherwise make the fingerprint spec-specific.
        """
        ordered_nodes = sorted(self.nodes, key=lambda n: (n.ast_path, n.kind))
        node_rename = {
            node.node_id: f"node_{index}" for index, node in enumerate(ordered_nodes)
        }
        binder_rename: dict[str, str] = {}
        for node in ordered_nodes:
            for name in sorted(node.definitions):
                binder_rename.setdefault(name, f"binder_{len(binder_rename)}")

        def canonical_role(edge: ScopeEdge) -> str:
            # Only REFERENCE roles carry a binder name; other kinds' roles
            # (external slot text, the literal "root" marker) aren't binder
            # identity and are dropped rather than leaked into the hash.
            if edge.kind == DependencyKind.REFERENCE:
                return binder_rename.get(edge.role, edge.role)
            return ""

        node_shapes = [
            {
                "kind": node.kind,
                "ast_depth": len(node.ast_path),
                "definition_count": len(node.definitions),
                "external_dependency_count": len(node.external_dependencies),
            }
            for node in ordered_nodes
        ]
        edge_shapes = sorted(
            (
                node_rename.get(edge.source, edge.source),
                node_rename.get(edge.target, edge.target),
                edge.kind.value,
                canonical_role(edge),
            )
            for edge in self.edges
        )
        payload = json.dumps(
            {"nodes": node_shapes, "edges": edge_shapes},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def derive_binder_graph(spec: ProgramSpec) -> BinderGraphV1:
    """Build a :class:`BinderGraphV1` from a ``ProgramSpec``.

    Unlike :func:`~slm_training.data.progspec.capsules.derive_capsule_graph`,
    a forward or undefined reference is never a hard failure -- it becomes an
    :class:`UnresolvedPlaceholder` entry, and :meth:`BinderGraphV1.check_g3`
    reports ``all_resolved=False`` for it instead.
    """
    walk = derive_node_edge_walk(spec, raise_on_unresolved=False)
    unresolved = tuple(
        sorted(
            (UnresolvedPlaceholder._from_walk(ref) for ref in walk.unresolved),
            key=lambda placeholder: (placeholder.node_id, placeholder.name),
        )
    )
    return BinderGraphV1(
        root_id=walk.root_id,
        nodes=walk.nodes,
        edges=walk.edges,
        unresolved=unresolved,
        spec_id=spec.id,
        version=BinderGraphV1.VERSION,
    )
