"""Dependency-closed verification capsules derived from ProgramSpec scopes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.scopes import ScopeKind, derive_scope_contracts


class DependencyKind(str, Enum):
    """Edge semantics in the capsule dependency graph."""

    DEFINES = "defines"
    REFERENCE = "reference"
    CONTAINMENT = "containment"
    ROOT_OUTPUT = "root_output"
    EFFECT = "effect"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ScopeNode:
    """One non-overlapping semantic unit in the capsule graph."""

    node_id: str
    scope_id: str | None
    kind: str
    ast_path: tuple[str | int, ...]
    member_paths: tuple[tuple[str | int, ...], ...]
    definitions: tuple[str, ...]
    external_dependencies: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ast_path"] = list(self.ast_path)
        data["member_paths"] = [list(p) for p in self.member_paths]
        data["definitions"] = list(self.definitions)
        data["external_dependencies"] = list(self.external_dependencies)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScopeNode:
        return cls(
            node_id=str(data["node_id"]),
            scope_id=str(data["scope_id"]) if data.get("scope_id") else None,
            kind=str(data["kind"]),
            ast_path=tuple(data.get("ast_path") or ()),
            member_paths=tuple(
                tuple(p) for p in (data.get("member_paths") or ())
            ),
            definitions=tuple(str(v) for v in data.get("definitions") or ()),
            external_dependencies=tuple(
                str(v) for v in data.get("external_dependencies") or ()
            ),
        )


@dataclass(frozen=True)
class ScopeEdge:
    """Directed dependency between two scope nodes."""

    source: str
    target: str
    kind: DependencyKind
    role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScopeEdge:
        return cls(
            source=str(data["source"]),
            target=str(data["target"]),
            kind=DependencyKind(str(data["kind"])),
            role=str(data.get("role", "")),
        )


@dataclass(frozen=True)
class VerificationCapsule:
    """One SCC-derived dependency-closed capsule."""

    capsule_id: str
    node_ids: tuple[str, ...]
    entry_node_id: str
    external_dependencies: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "node_ids": list(self.node_ids),
            "entry_node_id": self.entry_node_id,
            "external_dependencies": list(self.external_dependencies),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VerificationCapsule:
        return cls(
            capsule_id=str(data["capsule_id"]),
            node_ids=tuple(str(v) for v in data.get("node_ids") or ()),
            entry_node_id=str(data["entry_node_id"]),
            external_dependencies=tuple(
                str(v) for v in data.get("external_dependencies") or ()
            ),
        )


@dataclass(frozen=True)
class CapsuleGraph:
    """Deterministic dependency graph and SCC-derived capsules."""

    root_id: str
    nodes: tuple[ScopeNode, ...]
    edges: tuple[ScopeEdge, ...]
    capsules: tuple[VerificationCapsule, ...]
    spec_id: str
    version: str

    VERSION = "vss2-01-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "capsules": [c.to_dict() for c in self.capsules],
            "spec_id": self.spec_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapsuleGraph:
        return cls(
            root_id=str(data["root_id"]),
            nodes=tuple(ScopeNode.from_dict(d) for d in data.get("nodes", [])),
            edges=tuple(ScopeEdge.from_dict(d) for d in data.get("edges", [])),
            capsules=tuple(
                VerificationCapsule.from_dict(d) for d in data.get("capsules", [])
            ),
            spec_id=str(data.get("spec_id", "")),
            version=str(data.get("version", cls.VERSION)),
        )


def derive_capsule_graph(spec: ProgramSpec) -> CapsuleGraph:
    """Build a dependency-closed verification-capsule graph from a ProgramSpec.

    Statement scopes become graph nodes. Nested component-call and child-list
    scopes are attached as member paths to the nearest containing statement or
    the synthetic root. External slot/template inputs become EXTERNAL edges to
    the root interface node. Reference edges follow binder definitions; forward
    references (uses with no defining statement) raise ValueError.
    """
    contracts = derive_scope_contracts(spec)
    root_id = f"{spec.id}:root"

    statements = [c for c in contracts if c.kind == ScopeKind.STATEMENT]
    non_statements = [c for c in contracts if c.kind != ScopeKind.STATEMENT]

    # Map each statement to a node.
    nodes: dict[str, ScopeNode] = {}
    definition_to_node: dict[str, str] = {}
    path_to_node: dict[tuple[str | int, ...], str] = {}

    for contract in statements:
        node_id = f"{contract.scope_id}:node"
        nodes[node_id] = ScopeNode(
            node_id=node_id,
            scope_id=contract.scope_id,
            kind="statement",
            ast_path=contract.ast_path,
            member_paths=(),
            definitions=contract.definitions,
            external_dependencies=(),
        )
        path_to_node[contract.ast_path] = node_id
        for name in contract.definitions:
            definition_to_node[name] = node_id

    # Synthetic root node.
    nodes[root_id] = ScopeNode(
        node_id=root_id,
        scope_id=None,
        kind="root",
        ast_path=(),
        member_paths=(),
        definitions=(),
        external_dependencies=(),
    )
    path_to_node[()] = root_id

    def nearest_node(path: tuple[str | int, ...]) -> str:
        """Find the nearest containing statement/root by ast_path prefix."""
        for length in range(len(path), -1, -1):
            candidate = path[:length]
            if candidate in path_to_node:
                return path_to_node[candidate]
        return root_id

    # Attach non-statement contracts as member paths on the nearest node.
    for contract in non_statements:
        container_id = nearest_node(contract.ast_path)
        container = nodes[container_id]
        member_paths = (*container.member_paths, contract.ast_path)
        nodes[container_id] = ScopeNode(
            node_id=container.node_id,
            scope_id=container.scope_id,
            kind=container.kind,
            ast_path=container.ast_path,
            member_paths=member_paths,
            definitions=container.definitions,
            external_dependencies=container.external_dependencies,
        )

    edges: list[ScopeEdge] = []
    external_by_node: dict[str, set[str]] = {node_id: set() for node_id in nodes}

    def _current_node(path: tuple[str | int, ...]) -> str:
        for length in range(len(path), -1, -1):
            candidate = path[:length]
            if candidate in path_to_node:
                return path_to_node[candidate]
        return root_id

    def _walk(value: Any, path: tuple[str | int, ...]) -> None:
        current_id = _current_node(path)
        if isinstance(value, dict):
            statement_id = value.get("statementId")
            if isinstance(statement_id, str) and statement_id in definition_to_node:
                target_id = definition_to_node[statement_id]
                if target_id != current_id:
                    edges.append(
                        ScopeEdge(
                            source=current_id,
                            target=target_id,
                            kind=DependencyKind.REFERENCE,
                            role=statement_id,
                        )
                    )
                # Cross into the referenced statement's own scope.
                current_id = target_id
            elif value.get("type") == "ref":
                ref_name = value.get("name")
                if isinstance(ref_name, str):
                    if ref_name in definition_to_node:
                        target_id = definition_to_node[ref_name]
                        if target_id != current_id:
                            edges.append(
                                ScopeEdge(
                                    source=current_id,
                                    target=target_id,
                                    kind=DependencyKind.REFERENCE,
                                    role=ref_name,
                                )
                            )
                    else:
                        raise ValueError(
                            f"forward reference or undefined binder {ref_name!r} in {spec.id}"
                        )
            for key, child in value.items():
                _walk(child, (*path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                _walk(child, (*path, index))
        elif isinstance(value, str):
            if value.startswith(":"):
                edges.append(
                    ScopeEdge(
                        source=current_id,
                        target=root_id,
                        kind=DependencyKind.EXTERNAL,
                        role=value,
                    )
                )
                external_by_node[current_id].add(value)

    _walk(spec.ast, ())

    # Propagate collected external dependencies back onto node objects.
    for node_id, node in nodes.items():
        deps = external_by_node.get(node_id, set())
        if deps:
            nodes[node_id] = ScopeNode(
                node_id=node.node_id,
                scope_id=node.scope_id,
                kind=node.kind,
                ast_path=node.ast_path,
                member_paths=node.member_paths,
                definitions=node.definitions,
                external_dependencies=tuple(sorted(deps)),
            )

    # Root-output edge from root to the statement that defines root, if any.
    if "root" in definition_to_node:
        edges.append(
            ScopeEdge(
                source=root_id,
                target=definition_to_node["root"],
                kind=DependencyKind.ROOT_OUTPUT,
                role="root",
            )
        )

    node_list = tuple(sorted(nodes.values(), key=lambda n: n.node_id))
    edge_list = tuple(sorted(edges, key=lambda e: (e.source, e.target, e.kind.value, e.role)))

    # Compute SCCs over statement nodes using reference edges.
    sccs = _tarjan_sccs({n.node_id: n for n in node_list}, edge_list)

    capsules: list[VerificationCapsule] = []
    for index, component in enumerate(sccs):
        node_ids = tuple(sorted(component))
        entry = node_ids[0]
        external = sorted(
            {
                dep
                for nid in node_ids
                for dep in nodes[nid].external_dependencies
            }
        )
        capsules.append(
            VerificationCapsule(
                capsule_id=f"{spec.id}:capsule:{index}",
                node_ids=node_ids,
                entry_node_id=entry,
                external_dependencies=tuple(external),
            )
        )

    return CapsuleGraph(
        root_id=root_id,
        nodes=node_list,
        edges=edge_list,
        capsules=tuple(capsules),
        spec_id=spec.id,
        version=CapsuleGraph.VERSION,
    )


def _tarjan_sccs(
    nodes: dict[str, ScopeNode],
    edges: tuple[ScopeEdge, ...],
) -> list[list[str]]:
    """Return SCCs in reverse topological order (Tarjan)."""
    adjacency: dict[str, list[str]] = {nid: [] for nid in nodes}
    for edge in edges:
        if edge.kind == DependencyKind.REFERENCE and edge.source in adjacency:
            adjacency[edge.source].append(edge.target)

    index_counter = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(node_id: str) -> None:
        nonlocal index_counter
        indices[node_id] = index_counter
        lowlinks[node_id] = index_counter
        index_counter += 1
        stack.append(node_id)
        on_stack.add(node_id)

        for target in adjacency[node_id]:
            if target not in indices:
                strongconnect(target)
                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[target])
            elif target in on_stack:
                lowlinks[node_id] = min(lowlinks[node_id], indices[target])

        if lowlinks[node_id] == indices[node_id]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == node_id:
                    break
            sccs.append(component)

    for node_id in sorted(nodes):
        if node_id not in indices:
            strongconnect(node_id)

    return sccs
