"""Quotient-state diffusion graph diagnostics (CAP4-05, SLM-99).

Torch-free utilities that measure whether a declared diffusion process over
canonical quotient states is connected, mixed, and informationally well-behaved.
The graph is built from observed ``(state, action, next_state)`` transitions or
from synthetic kernels; all expensive quantities label themselves ``exact`` or
``estimated``.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

KernelName = Literal[
    "surface_token_mask",
    "independent_production_mask",
    "ast_subtree_mask",
    "typed_hole_mask",
    "quotient_random_walk",
    "posterior_weighted_walk",
]

EdgeType = Literal[
    "replace_with_hole",
    "restore_production",
    "change_production",
    "add_remove_binding",
    "abstract_refine_template",
    "reversible_remask",
    "kernel",
]


@dataclass(frozen=True)
class Transition:
    """One replayable transition in the quotient diffusion graph."""

    state: str
    action: str
    next_state: str
    weight: float = 1.0
    edge_type: EdgeType = "kernel"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "weight": self.weight,
            "edge_type": self.edge_type,
            "metadata": dict(self.metadata),
        }


@dataclass
class QuotientDiffusionGraph:
    """Directed graph over quotient diffusion states.

    Vertices are opaque state fingerprints. Edges carry an action label, an
    edge type, and a weight. The same pair of states may have multiple edges
    (one per action / type).
    """

    states: set[str] = field(default_factory=set)
    transitions: list[Transition] = field(default_factory=list)

    def add_transition(self, transition: Transition) -> None:
        self.states.add(transition.state)
        self.states.add(transition.next_state)
        self.transitions.append(transition)

    def transition_counts(self) -> dict[tuple[str, str], dict[str, float]]:
        """Return ``state -> action -> next_state -> total weight``."""
        counts: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for tr in self.transitions:
            counts[(tr.state, tr.action)][tr.next_state] += tr.weight
        return counts

    def transition_matrix(self) -> dict[str, dict[str, float]]:
        """Return row-stochastic ``state -> next_state`` probabilities."""
        out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for tr in self.transitions:
            out[tr.state][tr.next_state] += tr.weight
        for state, dist in out.items():
            total = sum(dist.values())
            if total > 0.0:
                for ns in dist:
                    dist[ns] /= total
        # States with no outgoing edges get a self-loop so the chain is lazy.
        for state in self.states:
            if state not in out or not out[state]:
                out[state] = {state: 1.0}
        return {k: dict(v) for k, v in out.items()}

    def adjacency_directed(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {s: set() for s in self.states}
        for tr in self.transitions:
            adj[tr.state].add(tr.next_state)
        return adj

    def adjacency_undirected(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {s: set() for s in self.states}
        for tr in self.transitions:
            adj[tr.state].add(tr.next_state)
            adj[tr.next_state].add(tr.state)
        return adj

    def weak_components(self) -> list[set[str]]:
        return _connected_components(self.adjacency_undirected())

    def strong_components(self) -> list[set[str]]:
        return _tarjan(self.adjacency_directed())

    def is_strongly_connected(self) -> bool:
        if len(self.states) <= 1:
            return True
        return len(self.strong_components()) == 1

    def shortest_path_lengths(self, *, directed: bool | None = None) -> dict[str, dict[str, int]]:
        """All-pairs shortest path lengths.

        If ``directed`` is None, use directed when the graph is strongly
        connected, otherwise fall back to undirected distances.
        """
        if directed is None:
            directed = self.is_strongly_connected()
        adj = self.adjacency_directed() if directed else self.adjacency_undirected()
        return _all_pairs_shortest_paths(adj)

    def diameter(self) -> dict[str, Any]:
        """Exact diameter when feasible; otherwise estimated."""
        n = len(self.states)
        if n == 0:
            return {"value": 0, "exact": True, "kind": "empty"}
        if n <= 200:
            sp = self.shortest_path_lengths()
            finite = [
                d for lengths in sp.values() for d in lengths.values() if d < float("inf")
            ]
            if not finite:
                return {"value": float("inf"), "exact": True, "kind": "disconnected"}
            exact = max(finite)
            return {"value": exact, "exact": True, "kind": "directed"}
        # Estimated: sample up to 50 source vertices.
        sources = list(self.states)[:50]
        adj = self.adjacency_undirected()
        best = 0
        for src in sources:
            best = max(best, max(_bfs_distances(adj, src).values(), default=0))
        return {"value": best, "exact": False, "kind": "estimated_undirected"}

    def average_path_length(self) -> dict[str, Any]:
        sp = self.shortest_path_lengths()
        finite = [
            d for lengths in sp.values() for d in lengths.values() if d < float("inf")
        ]
        if not finite:
            return {"value": float("inf"), "exact": True}
        return {"value": sum(finite) / len(finite), "exact": len(self.states) <= 200}

    def stationary_distribution(self) -> dict[str, Any]:
        """Stationary distribution of the row-stochastic transition kernel.

        For strongly connected graphs the answer is unique. For reducible
        graphs a per-component stationary vector is returned with zeros
        outside the component.
        """
        matrix = self.transition_matrix()
        states = sorted(self.states)
        idx = {s: i for i, s in enumerate(states)}
        n = len(states)
        if n == 0:
            return {"distribution": {}, "exact": True}

        components = self.strong_components()
        if len(components) == 1:
            P = _matrix_from_dict(matrix, states)
            pi = _stationary_vector(P)
            return {
                "distribution": {s: float(pi[idx[s]]) for s in states},
                "exact": True,
            }

        # Reducible: return one stationary vector per component, padded with 0.
        per_component: list[dict[str, float]] = []
        for comp in components:
            sub_states = sorted(comp)
            sub_idx = {s: i for i, s in enumerate(sub_states)}
            sub_matrix = {
                s: {ns: matrix[s][ns] for ns in sub_states if ns in matrix.get(s, {})}
                for s in sub_states
            }
            P = _matrix_from_dict(sub_matrix, sub_states)
            pi = _stationary_vector(P)
            full = {s: 0.0 for s in states}
            for s in sub_states:
                full[s] = float(pi[sub_idx[s]])
            per_component.append(full)
        return {
            "distribution": per_component[0] if per_component else {},
            "per_component": per_component,
            "exact": True,
            "reducible": True,
        }

    def spectral_gap(self) -> dict[str, Any]:
        matrix = self.transition_matrix()
        states = sorted(self.states)
        if len(states) <= 200:
            P = _matrix_from_dict(matrix, states)
            eigenvalues = np.linalg.eigvals(P)
            moduli = sorted(np.abs(eigenvalues), reverse=True)
            if len(moduli) < 2:
                return {"value": 1.0, "exact": True}
            gap = 1.0 - float(moduli[1])
            return {"value": gap, "exact": True}
        # Estimated: power iteration estimate of the second eigenvalue modulus.
        val = _power_second_eigenvalue(matrix, states)
        return {"value": 1.0 - val, "exact": False}

    def conductance(self, *, exact_max_n: int = 12) -> dict[str, Any]:
        """Bottleneck conductance.

        Exact by subset enumeration for graphs up to ``exact_max_n`` vertices;
        otherwise estimated by random subset sampling.
        """
        stationary = self.stationary_distribution()
        pi = stationary["distribution"]
        if not pi:
            return {"value": 0.0, "exact": True}
        matrix = self.transition_matrix()
        states = sorted(self.states)
        n = len(states)
        if n <= exact_max_n:
            best = _exact_conductance(states, pi, matrix)
            return {"value": best, "exact": True}
        best = _estimated_conductance(states, pi, matrix, trials=min(2000, 50 * n))
        return {"value": best, "exact": False}

    def mixing_time_bound(self, eps: float = 0.25) -> dict[str, Any]:
        gap = self.spectral_gap()
        stationary = self.stationary_distribution()
        pi = stationary["distribution"]
        if not pi or gap["value"] <= 0.0:
            return {"value": float("inf"), "eps": eps, "exact": gap["exact"]}
        pi_min = min(v for v in pi.values() if v > 0.0)
        bound = math.log(1.0 / (eps * pi_min)) / gap["value"]
        return {"value": bound, "eps": eps, "exact": gap["exact"]}

    def reversibility(self) -> dict[str, Any]:
        stationary = self.stationary_distribution()
        pi = stationary["distribution"]
        matrix = self.transition_matrix()
        violations = 0
        total = 0
        tol = 1e-6
        for i, pi_i in pi.items():
            for j, pij in matrix.get(i, {}).items():
                total += 1
                lhs = pi_i * pij
                rhs = pi.get(j, 0.0) * matrix.get(j, {}).get(i, 0.0)
                if abs(lhs - rhs) > tol * max(abs(lhs), abs(rhs), 1e-12):
                    violations += 1
        reversible = violations == 0 and self.is_strongly_connected()
        return {
            "reversible": reversible,
            "violations": violations,
            "checked_edges": total,
            "exact": stationary["exact"],
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "vertex_count": len(self.states),
            "edge_count": len(self.transitions),
            "weak_components": [sorted(c) for c in self.weak_components()],
            "strong_components": [sorted(c) for c in self.strong_components()],
            "strongly_connected": self.is_strongly_connected(),
            "diameter": self.diameter(),
            "average_path_length": self.average_path_length(),
            "stationary_distribution": self.stationary_distribution(),
            "spectral_gap": self.spectral_gap(),
            "conductance": self.conductance(),
            "mixing_time_bound": self.mixing_time_bound(),
            "reversibility": self.reversibility(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": sorted(self.states),
            "transitions": [t.to_dict() for t in self.transitions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuotientDiffusionGraph:
        graph = cls()
        for tr in data.get("transitions", ()):
            graph.add_transition(Transition(**tr))
        return graph

    @classmethod
    def from_traces(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        action_key: str = "selected_action_id",
        state_key: str = "state_fingerprint",
        group_keys: tuple[str, ...] = ("run_id", "example_id", "seed"),
    ) -> QuotientDiffusionGraph:
        """Build a transition graph from ordered GrammarDecisionTrace dicts.

        Transitions are inferred between consecutive records that share the same
        ``group_keys`` and have a non-null selected action.
        """
        graph = cls()
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            key = tuple(rec.get(k, "") for k in group_keys)
            grouped[key].append(dict(rec))
        for key, seq in grouped.items():
            seq.sort(key=lambda r: r.get("decision_index", 0))
            for prev, cur in zip(seq, seq[1:]):
                action = prev.get(action_key)
                if action is None:
                    continue
                graph.add_transition(
                    Transition(
                        state=str(prev[state_key]),
                        action=str(action),
                        next_state=str(cur[state_key]),
                        weight=1.0,
                        edge_type="kernel",
                        metadata={"group": key},
                    )
                )
        return graph


# --------------------------------------------------------------------------- #
# Graph algorithms
# --------------------------------------------------------------------------- #


def _connected_components(adj: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for start in adj:
        if start in seen:
            continue
        stack = [start]
        comp: set[str] = set()
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.add(v)
            for u in adj.get(v, ()):
                if u not in seen:
                    stack.append(u)
        components.append(comp)
    return components


def _tarjan(adj: dict[str, set[str]]) -> list[set[str]]:
    """Tarjan's strongly connected components algorithm."""
    index_counter = [0]
    stack: list[str] = []
    lowlinks: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: set[str] = set()
    components: list[set[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], index[w])
        if lowlinks[v] == index[v]:
            comp: set[str] = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.add(w)
                if w == v:
                    break
            components.append(comp)

    for v in adj:
        if v not in index:
            strongconnect(v)
    return components


def _bfs_distances(adj: dict[str, set[str]], source: str) -> dict[str, int]:
    dist: dict[str, int] = {v: float("inf") for v in adj}  # type: ignore[dict-item]
    dist[source] = 0
    queue = [source]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        for u in adj.get(v, ()):
            if dist[u] == float("inf"):
                dist[u] = dist[v] + 1
                queue.append(u)
    return dist


def _all_pairs_shortest_paths(
    adj: dict[str, set[str]],
) -> dict[str, dict[str, int]]:
    return {v: _bfs_distances(adj, v) for v in adj}


def _matrix_from_dict(
    matrix: Mapping[str, Mapping[str, float]], states: Sequence[str]
) -> np.ndarray:
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    P = np.zeros((n, n), dtype=float)
    for s, row in matrix.items():
        i = idx[s]
        for ns, p in row.items():
            P[i, idx[ns]] = p
    return P


def _stationary_vector(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of row-stochastic ``P`` with eigenvalue 1."""
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # Find eigenvalue closest to 1.
    closest = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = np.real(eigenvectors[:, closest])
    pi = np.abs(pi)
    total = pi.sum()
    if total > 0.0:
        pi = pi / total
    else:
        pi = np.ones(len(pi)) / len(pi)
    return pi


def _power_second_eigenvalue(
    matrix: Mapping[str, Mapping[str, float]], states: Sequence[str], steps: int = 80
) -> float:
    """Rough estimate of the second-largest eigenvalue modulus via power iteration."""
    n = len(states)
    P = _matrix_from_dict(matrix, states)
    # Start with a random vector orthogonal to the uniform vector.
    v = np.random.RandomState(0).randn(n)
    v = v - v.mean()
    for _ in range(steps):
        v = P.T @ v
        norm = np.linalg.norm(v)
        if norm > 0.0:
            v = v / norm
    return float(abs(v @ (P.T @ v)))


def _exact_conductance(
    states: Sequence[str],
    pi: Mapping[str, float],
    matrix: Mapping[str, Mapping[str, float]],
) -> float:
    """Cheeger constant of a directed chain with stationary distribution ``pi``."""
    total_pi = sum(pi.values())
    if total_pi <= 0.0:
        return 0.0
    n = len(states)
    best = float("inf")
    # Enumerate non-empty proper subsets using bit masks.
    for mask in range(1, 1 << (n - 1)):
        s_set = {states[i] for i in range(n) if mask & (1 << i)}
        pi_s = sum(pi.get(v, 0.0) for v in s_set)
        if pi_s <= 0.0 or pi_s >= total_pi:
            continue
        cap = 0.0
        for i in s_set:
            for j, pij in matrix.get(i, {}).items():
                if j not in s_set:
                    cap += pi.get(i, 0.0) * pij
        conductance = cap / min(pi_s, total_pi - pi_s)
        if conductance < best:
            best = conductance
    return 0.0 if best == float("inf") else best


def _estimated_conductance(
    states: Sequence[str],
    pi: Mapping[str, float],
    matrix: Mapping[str, Mapping[str, float]],
    trials: int = 2000,
) -> float:
    total_pi = sum(pi.values())
    if total_pi <= 0.0:
        return 0.0
    rng = random.Random(0)
    best = float("inf")
    for _ in range(trials):
        subset = {s for s in states if rng.random() < 0.5}
        if not subset or len(subset) == len(states):
            continue
        pi_s = sum(pi.get(v, 0.0) for v in subset)
        if pi_s <= 0.0 or pi_s >= total_pi:
            continue
        cap = 0.0
        for i in subset:
            for j, pij in matrix.get(i, {}).items():
                if j not in subset:
                    cap += pi.get(i, 0.0) * pij
        conductance = cap / min(pi_s, total_pi - pi_s)
        if conductance < best:
            best = conductance
    return 0.0 if best == float("inf") else best


# --------------------------------------------------------------------------- #
# Diffusion kernels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelSpec:
    """Descriptive contract for one corruption / transition kernel."""

    name: KernelName
    transition_probs: dict[str, dict[str, float]] = field(repr=False)
    schedule: str = "uniform"
    terminal_distribution: str = "uniform_over_states"
    invalid_allowed: bool = False
    reversible: bool = False
    exact_posterior: bool = True
    compute_cost: str = "O(1) per step"
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalize(self) -> None:
        object.__setattr__(self, "_normalized", True)

    def sample(self, state: str, rng: random.Random) -> str:
        dist = self.transition_probs.get(state, {})
        if not dist:
            return state
        items = list(dist.items())
        weights = [w for _, w in items]
        total = sum(weights)
        if total <= 0.0:
            return state
        threshold = rng.random() * total
        cumulative = 0.0
        for nxt, w in items:
            cumulative += w
            if cumulative >= threshold:
                return nxt
        return items[-1][0]

    def information_loss_bits(self, state: str, target_state: str) -> float:
        """Naive log-inverse transition probability as a proxy for information loss."""
        p = self.transition_probs.get(state, {}).get(target_state, 0.0)
        if p <= 0.0:
            return float("inf")
        return -math.log2(p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schedule": self.schedule,
            "terminal_distribution": self.terminal_distribution,
            "invalid_allowed": self.invalid_allowed,
            "reversible": self.reversible,
            "exact_posterior": self.exact_posterior,
            "compute_cost": self.compute_cost,
            "metadata": dict(self.metadata),
        }


def _normalize_probs(
    probs: Mapping[str, float],
) -> dict[str, float]:
    total = sum(p for p in probs.values() if p >= 0.0)
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in probs.items() if v >= 0.0}


def _masking_kernel(
    states: Sequence[str],
    mask_groups: Mapping[str, Sequence[str]],
    *,
    name: KernelName,
    rng_seed: int = 0,
    schedule: str = "uniform",
    reversible: bool = False,
    compute_cost: str = "O(mask_positions) per step",
    metadata: dict[str, Any] | None = None,
) -> KernelSpec:
    """Build a kernel where each state can mask one of several positions/groups.

    ``mask_groups[state]`` lists the neighbour states reached by masking each
    group.  Probability is uniform over groups.
    """
    probs: dict[str, dict[str, float]] = {}
    for s in states:
        groups = mask_groups.get(s, [])
        if not groups:
            probs[s] = {s: 1.0}
            continue
        row: dict[str, float] = defaultdict(float)
        for nxt in groups:
            row[nxt] += 1.0 / len(groups)
        probs[s] = _normalize_probs(row)
    return KernelSpec(
        name=name,
        transition_probs=probs,
        schedule=schedule,
        reversible=reversible,
        compute_cost=compute_cost,
        metadata=metadata or {},
    )


def build_surface_token_kernel(
    states: Sequence[str], token_positions: Mapping[str, Sequence[str]], **kwargs: Any
) -> KernelSpec:
    return _masking_kernel(
        states,
        token_positions,
        name="surface_token_mask",
        schedule="linear",
        compute_cost="O(surface tokens) per step",
        metadata={"mask_kind": "surface_token"},
        **kwargs,
    )


def build_production_mask_kernel(
    states: Sequence[str], production_options: Mapping[str, Sequence[str]], **kwargs: Any
) -> KernelSpec:
    return _masking_kernel(
        states,
        production_options,
        name="independent_production_mask",
        schedule="linear",
        compute_cost="O(legal productions) per step",
        metadata={"mask_kind": "production"},
        **kwargs,
    )


def build_ast_subtree_kernel(
    states: Sequence[str], subtree_options: Mapping[str, Sequence[str]], **kwargs: Any
) -> KernelSpec:
    return _masking_kernel(
        states,
        subtree_options,
        name="ast_subtree_mask",
        schedule="cosine",
        compute_cost="O(subtree size) per step",
        metadata={"mask_kind": "ast_subtree"},
        **kwargs,
    )


def build_typed_hole_kernel(
    states: Sequence[str], hole_options: Mapping[str, Sequence[str]], **kwargs: Any
) -> KernelSpec:
    return _masking_kernel(
        states,
        hole_options,
        name="typed_hole_mask",
        schedule="information_balanced",
        compute_cost="O(typed holes) per step",
        metadata={"mask_kind": "typed_hole"},
        **kwargs,
    )


def build_quotient_random_walk_kernel(
    graph: QuotientDiffusionGraph, *, rng_seed: int = 0
) -> KernelSpec:
    matrix = graph.transition_matrix()
    return KernelSpec(
        name="quotient_random_walk",
        transition_probs={s: _normalize_probs(matrix.get(s, {s: 1.0})) for s in graph.states},
        schedule="uniform",
        terminal_distribution="stationary",
        reversible=graph.reversibility()["reversible"],
        exact_posterior=True,
        compute_cost="O(out_degree) per step",
        metadata={"vertex_count": len(graph.states), "edge_count": len(graph.transitions)},
    )


def build_posterior_weighted_kernel(
    states: Sequence[str],
    target_states: Sequence[str],
    *,
    distance_fn: Mapping[tuple[str, str], float] | None = None,
    temperature: float = 1.0,
    rng_seed: int = 0,
) -> KernelSpec:
    """Transition probability decays with semantic distance to a target set."""
    rng = random.Random(rng_seed)
    probs: dict[str, dict[str, float]] = {}
    for s in states:
        weights: dict[str, float] = {}
        for t in target_states:
            if s == t:
                weights[t] = 1.0
            else:
                d = (
                    distance_fn.get((s, t), rng.random() + 0.1)
                    if distance_fn
                    else rng.random() + 0.1
                )
                weights[t] = math.exp(-d / max(temperature, 1e-9))
        probs[s] = _normalize_probs(weights)
    return KernelSpec(
        name="posterior_weighted_walk",
        transition_probs=probs,
        schedule="posterior",
        terminal_distribution="target_set",
        reversible=False,
        exact_posterior=False,
        compute_cost="O(|targets|) per step",
        metadata={"temperature": temperature, "target_count": len(target_states)},
    )


# --------------------------------------------------------------------------- #
# Information schedule
# --------------------------------------------------------------------------- #


@dataclass
class InformationSchedulePoint:
    timestep: int
    mean_entropy_bits: float | None
    mean_support_size: float | None
    record_count: int
    information_remaining: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "mean_entropy_bits": self.mean_entropy_bits,
            "mean_support_size": self.mean_support_size,
            "record_count": self.record_count,
            "information_remaining": self.information_remaining,
        }


def information_schedule(
    records: Sequence[Mapping[str, Any]],
    *,
    timestep_key: str = "diffusion_timestep",
    entropy_key: str = "posterior_entropy_bits",
    support_key: str = "completion_support_size_exact",
) -> list[InformationSchedulePoint]:
    """Aggregate per-timestep entropy and posterior support."""
    by_t: dict[int, list[tuple[float | None, int | None]]] = defaultdict(list)
    for rec in records:
        t = rec.get(timestep_key)
        if t is None:
            continue
        by_t[int(t)].append((rec.get(entropy_key), rec.get(support_key)))

    points: list[InformationSchedulePoint] = []
    for t in sorted(by_t):
        entropies = [e for e, _ in by_t[t] if e is not None]
        supports = [s for _, s in by_t[t] if s is not None]
        point = InformationSchedulePoint(
            timestep=t,
            mean_entropy_bits=sum(entropies) / len(entropies) if entropies else None,
            mean_support_size=sum(supports) / len(supports) if supports else None,
            record_count=len(by_t[t]),
        )
        points.append(point)

    if points and points[0].mean_entropy_bits not in (None, 0.0):
        h0 = points[0].mean_entropy_bits
        for p in points:
            if p.mean_entropy_bits is not None:
                p.information_remaining = p.mean_entropy_bits / h0
    return points


def recommend_information_balanced_schedule(
    points: Sequence[InformationSchedulePoint],
    n_steps: int,
) -> list[float]:
    """Return target cumulative information fractions for equal loss per step.

    This is a wiring-level recommendation: it linearly interpolates the observed
    schedule and inverts it so that each step removes roughly the same amount of
    conditional information.
    """
    if not points or n_steps <= 0:
        return []
    values = [p.information_remaining for p in points if p.information_remaining is not None]
    if len(values) < 2:
        return [1.0 - i / n_steps for i in range(n_steps)]
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 - i / n_steps for i in range(n_steps)]
    targets: list[float] = []
    for i in range(n_steps):
        frac = i / (n_steps - 1) if n_steps > 1 else 0.0
        # Invert: we want cumulative information remaining after i steps.
        targets.append(hi - frac * (hi - lo))
    return targets


# --------------------------------------------------------------------------- #
# Kernel comparison at matched information loss
# --------------------------------------------------------------------------- #


@dataclass
class MatchedKernelComparison:
    """Compare kernels after equalizing their per-step information loss."""

    kernels: list[KernelSpec]
    target_loss_bits: float
    matched_steps: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_loss_bits": self.target_loss_bits,
            "matched_steps": self.matched_steps,
            "kernels": [k.to_dict() for k in self.kernels],
            "notes": list(self.notes),
        }


def compare_kernels_at_matched_loss(
    kernels: Sequence[KernelSpec],
    states: Sequence[str],
    target_loss_bits: float,
) -> MatchedKernelComparison:
    """Estimate how many steps each kernel needs to remove ``target_loss_bits``.

    Uses the average per-step information-loss proxy across states.
    """
    matched: dict[str, float] = {}
    notes: list[str] = []
    for kernel in kernels:
        losses: list[float] = []
        for s in states:
            row = kernel.transition_probs.get(s, {})
            if not row:
                continue
            avg_loss = sum(
                -math.log2(max(p, 1e-12)) * p for p in row.values() if p > 0.0
            )
            losses.append(avg_loss)
        avg = sum(losses) / len(losses) if losses else 0.0
        if avg <= 0.0:
            matched[kernel.name] = float("inf")
            notes.append(f"{kernel.name}: zero per-step loss")
        else:
            matched[kernel.name] = target_loss_bits / avg
            notes.append(
                f"{kernel.name}: {avg:.4f} bits/step -> {matched[kernel.name]:.2f} steps"
            )
    return MatchedKernelComparison(
        kernels=list(kernels),
        target_loss_bits=target_loss_bits,
        matched_steps=matched,
        notes=notes,
    )
