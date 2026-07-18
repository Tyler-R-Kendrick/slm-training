"""Task-confusability graph and counterexample-guided neural state quotient.

CAP1-03 (SLM-83): compute which exact compiler states may safely share one
neural representation under a declared task distortion, while preserving the
compiler states themselves unchanged.

This module is Torch-free. It operates on state fingerprints and aligned
action records; it never merges compiler states or changes legality.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

PolicyMetric = Literal["js", "tv", "cross_entropy_regret", "topk_regret"]
Horizon = Literal["next_action", "bounded_completion", "terminal"]


@dataclass(frozen=True)
class TaskDistortionSpec:
    """Versioned distortion specification for state confusability."""

    spec_id: str
    action_alignment: str = "production_family"
    policy_metric: PolicyMetric = "cross_entropy_regret"
    policy_tolerance: float = 0.1
    value_weight: float = 0.0
    execution_weight: float = 0.0
    semantic_fingerprint_weight: float = 0.0
    average_tolerance: float | None = None
    cvar_alpha: float | None = None
    cvar_tolerance: float | None = None
    hard_forbidden_confusions: tuple[str, ...] = ()
    horizon: Horizon = "next_action"

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "action_alignment": self.action_alignment,
            "policy_metric": self.policy_metric,
            "policy_tolerance": self.policy_tolerance,
            "value_weight": self.value_weight,
            "execution_weight": self.execution_weight,
            "semantic_fingerprint_weight": self.semantic_fingerprint_weight,
            "average_tolerance": self.average_tolerance,
            "cvar_alpha": self.cvar_alpha,
            "cvar_tolerance": self.cvar_tolerance,
            "hard_forbidden_confusions": list(self.hard_forbidden_confusions),
            "horizon": self.horizon,
        }


@dataclass(frozen=True)
class AlignedActionRecord:
    """One observed decision at an exact state with aligned actions."""

    state_fingerprint: str
    action_id: str
    aligned_family: str
    probability: float | None = None
    value: float | None = None
    semantic_fingerprint: str | None = None


@dataclass(frozen=True)
class StateProfile:
    """Aggregated profile of one exact state from one or more traces."""

    fingerprint: str
    visit_count: int
    action_distribution: dict[str, float]
    aligned_families: set[str] = field(default_factory=set)
    value_estimate: float | None = None
    semantic_fingerprint: str | None = None


@dataclass
class ConfusabilityGraph:
    """Undirected graph over exact state fingerprints."""

    vertices: set[str]
    edges: set[frozenset[str]]
    edge_reasons: dict[frozenset[str], str] = field(default_factory=dict)

    def adjacency(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {v: set() for v in self.vertices}
        for a, b in (tuple(e) for e in self.edges if len(e) == 2):
            adj[a].add(b)
            adj[b].add(a)
        return adj


@dataclass
class ColoringResult:
    """Coloring witness plus exactness flag."""

    colors: dict[str, int]
    num_colors: int
    exact: bool
    lower_bound: int
    upper_bound: int
    algorithm: str

    def verify(self, graph: ConfusabilityGraph) -> list[str]:
        """Return violations of the coloring witness."""
        violations: list[str] = []
        for edge in graph.edges:
            a, b = tuple(edge)
            if self.colors.get(a) == self.colors.get(b):
                violations.append(f"edge ({a}, {b}) shares color {self.colors.get(a)}")
        return violations


@dataclass
class QuotientReport:
    """Aggregate report for a task quotient analysis."""

    spec: TaskDistortionSpec
    graph: ConfusabilityGraph
    coloring: ColoringResult
    state_count: int
    edge_count: int
    density: float
    class_size_histogram: dict[int, int]
    counterexamples: list[dict[str, Any]]
    capacity_feasibility: dict[tuple[int, int], bool]
    estimated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "state_count": self.state_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "coloring": {
                "num_colors": self.coloring.num_colors,
                "exact": self.coloring.exact,
                "lower_bound": self.coloring.lower_bound,
                "upper_bound": self.coloring.upper_bound,
                "algorithm": self.coloring.algorithm,
            },
            "class_size_histogram": self.class_size_histogram,
            "counterexamples": self.counterexamples,
            "capacity_feasibility": {
                f"K={k},d={d}": feasible
                for (k, d), feasible in self.capacity_feasibility.items()
            },
            "estimated": self.estimated,
        }


def align_action(action_id: str, alignment: str = "production_family") -> str:
    """Map an action id to its semantic family for alignment."""
    if alignment == "production_family":
        # production_codec prefixes: +component, ^direction, @slot, &ref, ~rel_ref, #lit
        if action_id.startswith("+"):
            return "component"
        if action_id.startswith("^"):
            return "direction"
        if action_id.startswith("@"):
            return "slot"
        if action_id.startswith("&") or action_id.startswith("~"):
            return "reference"
        if action_id.startswith("#"):
            return "literal"
        if action_id in {"=", "r=", "$=", "q=", "m=", "a=", ";", "[", "]", "(", ")", ",", "-"}:
            return "structural"
        return "other"
    if alignment == "exact":
        return action_id
    return action_id


def _kl_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """KL(p||q) over the union of supports."""
    total = 0.0
    for key in set(p) | set(q):
        pi = float(p.get(key, 0.0))
        qi = float(q.get(key, 0.0))
        if pi > 0.0:
            if qi <= 0.0:
                return float("inf")
            total += pi * math.log(pi / qi)
    return total


def _js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in set(p) | set(q)}
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _total_variation(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _cross_entropy_regret(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Regret of using q in place of p: CE(p,q) - H(p)."""
    ce = 0.0
    entropy = 0.0
    for key in set(p):
        pi = float(p[key])
        qi = float(q.get(key, 0.0))
        if pi > 0.0:
            if qi <= 0.0:
                return float("inf")
            ce -= pi * math.log(qi)
            entropy -= pi * math.log(pi)
    return ce - entropy


def _topk_regret(p: Mapping[str, float], q: Mapping[str, float], k: int = 3) -> float:
    p_top = set(sorted(p, key=p.get, reverse=True)[:k])  # type: ignore[arg-type]
    q_top = set(sorted(q, key=q.get, reverse=True)[:k])  # type: ignore[arg-type]
    return 1.0 - len(p_top & q_top) / k


def policy_distance(
    p: Mapping[str, float],
    q: Mapping[str, float],
    metric: PolicyMetric,
) -> float:
    """Distance between two aligned action distributions."""
    if metric == "js":
        return _js_divergence(p, q)
    if metric == "tv":
        return _total_variation(p, q)
    if metric == "cross_entropy_regret":
        return _cross_entropy_regret(p, q)
    return _topk_regret(p, q)


def build_state_profiles(
    records: Iterable[AlignedActionRecord],
) -> dict[str, StateProfile]:
    """Aggregate aligned action records into per-state profiles.

    When records carry a pre-computed ``probability`` it is used as a weight;
    otherwise each record contributes one unweighted observation.
    """
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    family_sets: dict[str, set[str]] = defaultdict(set)
    visits: dict[str, float] = defaultdict(float)
    semantic_votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for record in records:
        fp = record.state_fingerprint
        weight = record.probability if record.probability is not None else 1.0
        visits[fp] += weight
        counts[fp][record.aligned_family] += weight
        family_sets[fp].add(record.aligned_family)
        if record.semantic_fingerprint is not None:
            semantic_votes[fp][record.semantic_fingerprint] += weight
    profiles: dict[str, StateProfile] = {}
    for fp, action_counts in counts.items():
        total = sum(action_counts.values())
        dist = {action: cnt / total for action, cnt in action_counts.items()}
        semantic = None
        if semantic_votes[fp]:
            semantic = max(semantic_votes[fp], key=lambda k: (semantic_votes[fp][k], k))
        profiles[fp] = StateProfile(
            fingerprint=fp,
            visit_count=int(round(visits[fp])),
            action_distribution=dist,
            aligned_families=family_sets[fp],
            semantic_fingerprint=semantic,
        )
    return profiles


def build_confusability_graph(
    profiles: Mapping[str, StateProfile],
    spec: TaskDistortionSpec,
) -> ConfusabilityGraph:
    """Build a confusability graph from state profiles and a distortion spec."""
    vertices = set(profiles)
    edges: set[frozenset[str]] = set()
    reasons: dict[frozenset[str], str] = {}
    fingerprints = sorted(vertices)

    merge_tolerance = (
        spec.average_tolerance
        if spec.average_tolerance is not None
        else spec.policy_tolerance
    )
    for i, a in enumerate(fingerprints):
        for b in fingerprints[i + 1 :]:
            pa = profiles[a].action_distribution
            pb = profiles[b].action_distribution
            dist = policy_distance(pa, pb, spec.policy_metric)
            if dist > merge_tolerance:
                edge = frozenset({a, b})
                edges.add(edge)
                reasons[edge] = f"policy_distance={dist:.4g} > {merge_tolerance}"
            elif (
                spec.hard_forbidden_confusions
                and profiles[a].semantic_fingerprint is not None
                and profiles[b].semantic_fingerprint is not None
                and profiles[a].semantic_fingerprint == profiles[b].semantic_fingerprint
                and profiles[a].semantic_fingerprint in spec.hard_forbidden_confusions
            ):
                edge = frozenset({a, b})
                edges.add(edge)
                reasons[edge] = "hard_forbidden_confusion"

    return ConfusabilityGraph(vertices=vertices, edges=edges, edge_reasons=reasons)


def _greedy_coloring(adj: Mapping[str, set[str]], order: Sequence[str]) -> dict[str, int]:
    colors: dict[str, int] = {}
    for vertex in order:
        used = {colors[n] for n in adj.get(vertex, ()) if n in colors}
        color = 1
        while color in used:
            color += 1
        colors[vertex] = color
    return colors


def _dsatur_coloring(adj: Mapping[str, set[str]]) -> dict[str, int]:
    """DSATUR greedy coloring; deterministic for tied saturation."""
    vertices = sorted(adj)
    colors: dict[str, int] = {}
    saturations: dict[str, set[int]] = {v: set() for v in vertices}
    degrees = {v: len(adj[v]) for v in vertices}
    uncolored = set(vertices)

    # Start with highest-degree vertex.
    first = max(vertices, key=lambda v: (degrees[v], v))
    colors[first] = 1
    uncolored.remove(first)
    for neighbor in adj[first]:
        saturations[neighbor].add(1)

    while uncolored:
        # Pick vertex with highest saturation, then degree, then id.
        vertex = max(
            uncolored,
            key=lambda v: (len(saturations[v]), degrees[v], v),
        )
        used = {colors[n] for n in adj[vertex] if n in colors}
        color = 1
        while color in used:
            color += 1
        colors[vertex] = color
        uncolored.remove(vertex)
        for neighbor in adj[vertex]:
            saturations[neighbor].add(color)

    return colors


def _clique_lower_bound(adj: Mapping[str, set[str]]) -> int:
    """Simple greedy clique lower bound via vertex ordering."""
    vertices = sorted(adj)
    best = 1
    for start in vertices:
        clique = {start}
        candidates = set(adj[start]) - {start}
        while candidates:
            # Deterministically pick smallest-id candidate connected to all current.
            next_v = None
            for v in sorted(candidates):
                if all(v in adj[u] for u in clique):
                    next_v = v
                    break
            if next_v is None:
                break
            clique.add(next_v)
            candidates = candidates & adj[next_v]
        best = max(best, len(clique))
    return best


def color_graph(
    graph: ConfusabilityGraph,
    *,
    exact_max_vertices: int = 128,
) -> ColoringResult:
    """Color the confusability graph.

    For small graphs (<= exact_max_vertices) attempt exact coloring via
    branch-and-bound. Otherwise use DSATUR heuristic.
    """
    adj = graph.adjacency()
    vertices = sorted(graph.vertices)
    n = len(vertices)
    lower = _clique_lower_bound(adj)

    if n <= exact_max_vertices:
        best_colors, best_k = _exact_coloring(adj, vertices, lower)
        if best_k is not None:
            return ColoringResult(
                colors=best_colors,
                num_colors=best_k,
                exact=True,
                lower_bound=lower,
                upper_bound=best_k,
                algorithm="branch_and_bound",
            )

    colors = _dsatur_coloring(adj)
    k = max(colors.values()) if colors else 0
    return ColoringResult(
        colors=colors,
        num_colors=k,
        exact=False,
        lower_bound=lower,
        upper_bound=k,
        algorithm="dsatur",
    )


def _exact_coloring(
    adj: Mapping[str, set[str]],
    vertices: Sequence[str],
    lower_bound: int,
) -> tuple[dict[str, int], int | None]:
    """Branch-and-bound exact coloring; returns best coloring and chromatic number."""
    n = len(vertices)
    best_k = n
    best_colors: dict[str, int] = {v: i + 1 for i, v in enumerate(vertices)}

    order = sorted(vertices, key=lambda v: (-len(adj[v]), v))

    def backtrack(
        coloring: dict[str, int],
        used_max: int,
        position: int,
    ) -> None:
        nonlocal best_k, best_colors
        if position == n:
            if used_max < best_k:
                best_k = used_max
                best_colors = dict(coloring)
            return
        vertex = order[position]
        neighbors = adj[vertex]
        used = {coloring[n] for n in neighbors if n in coloring}
        # Try existing colors first, then new color up to best_k - 1.
        for color in range(1, best_k):
            if color in used:
                continue
            coloring[vertex] = color
            backtrack(coloring, max(used_max, color), position + 1)
            del coloring[vertex]
        # Prune: if even a new color cannot beat best_k, skip.
        if used_max + 1 < best_k:
            coloring[vertex] = used_max + 1
            backtrack(coloring, used_max + 1, position + 1)
            del coloring[vertex]

    coloring: dict[str, int] = {}
    backtrack(coloring, 0, 0)
    if best_k <= n:
        return best_colors, best_k
    return best_colors, None


def capacity_feasible(
    num_colors: int,
    capacities: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], bool]:
    """Check K^d >= num_colors for each (K, d) capacity pair."""
    return {(k, d): (k ** d) >= num_colors for k, d in capacities}


def refine_quotient(
    profiles: Mapping[str, StateProfile],
    spec: TaskDistortionSpec,
    coloring: ColoringResult,
    *,
    max_iterations: int = 4,
) -> tuple[ConfusabilityGraph, ColoringResult, list[dict[str, Any]]]:
    """Counterexample-guided refinement: split colors with excess pairwise regret."""
    graph = build_confusability_graph(profiles, spec)
    counterexamples: list[dict[str, Any]] = []
    current_coloring = coloring

    for iteration in range(max_iterations):
        violations: list[tuple[str, str, float, int]] = []
        color_groups: dict[int, set[str]] = defaultdict(set)
        for vertex, color in current_coloring.colors.items():
            color_groups[color].add(vertex)

        for color, group in color_groups.items():
            members = sorted(group)
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    pa = profiles[a].action_distribution
                    pb = profiles[b].action_distribution
                    dist = policy_distance(pa, pb, spec.policy_metric)
                    if dist > spec.policy_tolerance:
                        violations.append((a, b, dist, color))

        if not violations:
            break

        # Add edges for the worst violation and recolor.
        a, b, dist, color = max(violations, key=lambda x: x[2])
        edge = frozenset({a, b})
        graph.edges.add(edge)
        graph.edge_reasons[edge] = f"refinement_iter{iteration}: dist={dist:.4g}"
        counterexamples.append(
            {
                "iteration": iteration,
                "state_a": a,
                "state_b": b,
                "color": color,
                "policy_distance": dist,
                "reason": "within_color_excess_regret",
            }
        )
        current_coloring = color_graph(graph, exact_max_vertices=128)

    return graph, current_coloring, counterexamples


def analyze_task_quotient(
    records: Iterable[AlignedActionRecord],
    spec: TaskDistortionSpec,
    *,
    capacities: Sequence[tuple[int, int]] | None = None,
    exact_max_vertices: int = 128,
    refine: bool = True,
    max_refinement_iterations: int = 4,
) -> QuotientReport:
    """Build confusability graph, color it, and optionally refine."""
    profiles = build_state_profiles(records)
    graph = build_confusability_graph(profiles, spec)
    coloring = color_graph(graph, exact_max_vertices=exact_max_vertices)
    counterexamples: list[dict[str, Any]] = []

    if refine:
        graph, coloring, counterexamples = refine_quotient(
            profiles,
            spec,
            coloring,
            max_iterations=max_refinement_iterations,
        )

    colors = coloring.colors
    class_sizes: dict[int, int] = defaultdict(int)
    for vertex in graph.vertices:
        class_sizes[colors.get(vertex, 0)] += 1

    state_count = len(graph.vertices)
    edge_count = len(graph.edges)
    density = (
        (2 * edge_count) / (state_count * (state_count - 1))
        if state_count > 1
        else 0.0
    )

    caps = capacities or [(2, 4), (3, 4), (4, 4), (8, 3)]
    feasible = capacity_feasible(coloring.num_colors, caps)

    return QuotientReport(
        spec=spec,
        graph=graph,
        coloring=coloring,
        state_count=state_count,
        edge_count=edge_count,
        density=density,
        class_size_histogram=dict(class_sizes),
        counterexamples=counterexamples,
        capacity_feasibility=feasible,
        estimated=True,
    )
