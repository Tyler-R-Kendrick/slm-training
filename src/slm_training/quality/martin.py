"""Robert C. Martin's package metrics and the three coupling principles.

From *Agile Software Development: Principles, Patterns, and Practices* and
*Clean Architecture*:

* **Ca** -- afferent coupling: components that depend on this one.
* **Ce** -- efferent coupling: components this one depends on.
* **I**  -- instability, ``Ce / (Ca + Ce)``. 0 is maximally stable (many
  dependents, no dependencies); 1 is maximally unstable.
* **A**  -- abstractness, ``abstract classes / all classes``.
* **D**  -- distance from the main sequence, ``|A + I - 1|``. The main sequence
  is the line from (0, 1) -- stable *and* abstract -- to (1, 0) -- unstable and
  concrete. Components far off it are misbuilt.

* **ADP** (Acyclic Dependencies) -- the component graph must be a DAG.
* **SDP** (Stable Dependencies) -- depend in the direction of stability;
  ``I(source) >= I(target)``.
* **SAP** (Stable Abstractions) -- a stable component must be abstract, or it
  becomes rigid. Violations land in the *zone of pain* (stable and concrete) or
  the *zone of uselessness* (unstable and abstract).

See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from slm_training.quality.imports import ModuleInfo, parse_modules, resolve_edges

#: Distance from the main sequence past which a component is reported.
#: Martin treats D as a normalised 0..1 distance; 0.7 admits the ordinary
#: concrete-leaf case while still flagging genuinely misplaced components.
MAX_DISTANCE = 0.7

#: A component with at least this many dependents is "stable enough" that being
#: concrete is a real risk (zone of pain) rather than a rounding artefact.
ZONE_OF_PAIN_MIN_DEPENDENTS = 5


@dataclass(frozen=True)
class ComponentMetrics:
    """Martin metrics for one component."""

    name: str
    modules: int
    lines: int
    classes: int
    abstract_classes: int
    afferent: int
    efferent: int

    @property
    def instability(self) -> float:
        total = self.afferent + self.efferent
        return self.efferent / total if total else 0.0

    @property
    def abstractness(self) -> float:
        return self.abstract_classes / self.classes if self.classes else 0.0

    @property
    def distance(self) -> float:
        return abs(self.abstractness + self.instability - 1.0)

    @property
    def zone(self) -> str:
        """Which failure zone the component sits in, if any."""

        if self.distance <= MAX_DISTANCE:
            return "main-sequence"
        if self.instability < 0.5 and self.abstractness < 0.5:
            return "pain"
        return "uselessness"


def build(*, root: Path) -> tuple[list[ComponentMetrics], dict[str, set[str]]]:
    """Compute per-component metrics and the component dependency graph."""

    modules = parse_modules(root=root)
    edges = resolve_edges(modules)
    afferent: dict[str, set[str]] = {name: set() for name in edges}
    for source, targets in edges.items():
        for target in targets:
            afferent.setdefault(target, set()).add(source)

    grouped: dict[str, list[ModuleInfo]] = {}
    for module in modules:
        grouped.setdefault(module.component, []).append(module)

    metrics = [
        ComponentMetrics(
            name=name,
            modules=len(members),
            lines=_line_total(members, root=root),
            classes=sum(item.classes for item in members),
            abstract_classes=sum(item.abstract_classes for item in members),
            afferent=len(afferent.get(name, set())),
            efferent=len(edges.get(name, set())),
        )
        for name, members in sorted(grouped.items())
    ]
    return metrics, edges


def _line_total(members: list[ModuleInfo], *, root: Path) -> int:
    total = 0
    for member in members:
        try:
            total += len((root / member.path).read_text(encoding="utf-8").splitlines())
        except OSError:
            continue
    return total


@dataclass
class _TarjanState:
    """Bookkeeping for one iterative Tarjan traversal."""

    edges: Mapping[str, set[str]]
    index: dict[str, int] = field(default_factory=dict)
    low: dict[str, int] = field(default_factory=dict)
    on_stack: set[str] = field(default_factory=set)
    stack: list[str] = field(default_factory=list)
    counter: int = 0

    def children(self, node: str) -> list[str]:
        return sorted(self.edges.get(node, ()))

    def enter(self, node: str) -> None:
        """Assign ``node`` its discovery index and put it on the SCC stack."""

        self.index[node] = self.low[node] = self.counter
        self.counter += 1
        self.stack.append(node)
        self.on_stack.add(node)

    def close(self, node: str) -> list[str] | None:
        """Pop the SCC rooted at ``node``, or ``None`` if it is not a cycle.

        A single component is only a cycle when it imports itself.
        """

        if self.low[node] != self.index[node]:
            return None
        group: list[str] = []
        while True:
            member = self.stack.pop()
            self.on_stack.discard(member)
            group.append(member)
            if member == node:
                break
        if len(group) > 1 or node in self.edges.get(node, set()):
            return sorted(group)
        return None


def _descend(
    state: _TarjanState,
    work: list[tuple[str, list[str]]],
    node: str,
    child: str,
) -> None:
    """Take one edge ``node -> child``, pushing unvisited children onto ``work``."""

    if child not in state.index:
        state.enter(child)
        work.append((child, state.children(child)))
    elif child in state.on_stack:
        state.low[node] = min(state.low[node], state.index[child])


def _visit(state: _TarjanState, root: str) -> list[list[str]]:
    """Every cycle reachable from ``root``, by iterative depth-first search."""

    found: list[list[str]] = []
    state.enter(root)
    work: list[tuple[str, list[str]]] = [(root, state.children(root))]
    while work:
        node, pending = work[-1]
        if pending:
            _descend(state, work, node, pending.pop(0))
            continue
        work.pop()
        if work:
            parent = work[-1][0]
            state.low[parent] = min(state.low[parent], state.low[node])
        group = state.close(node)
        if group is not None:
            found.append(group)
    return found


def cycles(edges: Mapping[str, set[str]]) -> list[list[str]]:
    """Every dependency cycle among components (ADP violations).

    Tarjan's strongly-connected-components algorithm, iterative so a deep
    component graph cannot exhaust the recursion limit.
    """

    state = _TarjanState(edges=edges)
    found: list[list[str]] = []
    for root in sorted(edges):
        if root not in state.index:
            found.extend(_visit(state, root))
    return sorted(found)


def stable_dependency_violations(
    metrics: list[ComponentMetrics], edges: Mapping[str, set[str]]
) -> list[str]:
    """Edges that point from a stable component to a less stable one (SDP)."""

    by_name = {item.name: item for item in metrics}
    violations: list[str] = []
    for source in sorted(edges):
        for target in sorted(edges[source]):
            head, tail = by_name.get(source), by_name.get(target)
            if head is None or tail is None:
                continue
            if head.instability < tail.instability:
                violations.append(
                    f"{source} (I={head.instability:.2f}) -> "
                    f"{target} (I={tail.instability:.2f})"
                )
    return violations


def stable_abstraction_violations(metrics: list[ComponentMetrics]) -> list[str]:
    """Components sitting too far from the main sequence (SAP)."""

    violations: list[str] = []
    for item in sorted(metrics, key=lambda entry: -entry.distance):
        if item.zone == "main-sequence":
            continue
        if item.zone == "pain" and item.afferent < ZONE_OF_PAIN_MIN_DEPENDENTS:
            continue
        violations.append(
            f"{item.name}: zone of {item.zone} "
            f"(D={item.distance:.2f}, I={item.instability:.2f}, "
            f"A={item.abstractness:.2f}, Ca={item.afferent}, Ce={item.efferent})"
        )
    return violations
