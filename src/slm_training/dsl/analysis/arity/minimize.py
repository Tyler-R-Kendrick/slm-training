"""Iterative acyclic minimisation of the arith-sketch prefix trie (CAP0-02).

The prefix trie of a *finite* language is an acyclic DFA, so its minimal DFA is
also acyclic and can be found by a single bottom-up pass: process nodes in
reverse-topological (deepest-first) order and key each on
``(terminal_status, tuple(sorted (action, child_class)))``. Equivalent futures
collapse to one class. The pass is iterative (an explicit ordered sweep, no
recursion) so arbitrarily deep tries never risk a stack overflow.

Downstream metrics (legal branching histogram, forced-move fraction, minimal
completion length) are computed over the resulting minimal DFA via monotone
fixpoint relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slm_training.dsl.analysis.arity.state_graph import Trie


@dataclass
class MinimizedDFA:
    """Minimal acyclic DFA: classes, transitions, and accepting mask."""

    class_count: int
    accepting: list[bool]
    edges: list[dict[str, int]]
    class_of_node: dict[int, int]
    start_class: int
    work: dict[str, int] = field(default_factory=dict)


def minimize(trie: Trie) -> MinimizedDFA:
    """Bottom-up acyclic minimisation of ``trie`` (single reverse-topo sweep)."""
    # Deepest first; ties by node id keep the sweep deterministic. Every node's
    # children are strictly deeper, so their classes are assigned before it.
    order = sorted(trie.nodes, key=lambda node: (-node.depth, node.node_id))

    class_of_node: dict[int, int] = {}
    signature_to_class: dict[tuple[object, ...], int] = {}
    representative: list[int] = []
    accepting: list[bool] = []
    sweeps = 0

    for node in order:
        sweeps += 1
        child_signature = tuple(
            sorted(
                (action, class_of_node[child_id])
                for action, child_id in node.edges.items()
            )
        )
        signature = (node.accepting, child_signature)
        class_id = signature_to_class.get(signature)
        if class_id is None:
            class_id = len(representative)
            signature_to_class[signature] = class_id
            representative.append(node.node_id)
            accepting.append(node.accepting)
        class_of_node[node.node_id] = class_id

    nodes_by_id = {node.node_id: node for node in trie.nodes}
    edges: list[dict[str, int]] = []
    for rep_id in representative:
        rep = nodes_by_id[rep_id]
        edges.append(
            {
                action: class_of_node[child_id]
                for action, child_id in sorted(rep.edges.items())
            }
        )

    return MinimizedDFA(
        class_count=len(representative),
        accepting=accepting,
        edges=edges,
        class_of_node=class_of_node,
        start_class=class_of_node[trie.root.node_id],
        work={"minimize_node_sweeps": sweeps, "minimize_classes": len(representative)},
    )


def branching_histogram(dfa: MinimizedDFA) -> dict[int, int]:
    """Histogram of legal next-action out-degree over minimal states."""
    histogram: dict[int, int] = {}
    for state_edges in dfa.edges:
        degree = len(state_edges)
        histogram[degree] = histogram.get(degree, 0) + 1
    return histogram


def max_local_branching(dfa: MinimizedDFA) -> int:
    """Largest legal next-action out-degree over minimal states."""
    return max((len(state_edges) for state_edges in dfa.edges), default=0)


def forced_visit_fraction(dfa: MinimizedDFA) -> dict[str, object]:
    """Fraction of decision states whose only legal move is forced.

    Denominator: states with at least one outgoing action (decision points).
    Numerator: those with exactly one outgoing action and not accepting — a
    truly forced move (no "stop here" alternative). Reported as an exact
    integer ratio (division-free) plus a float for readability.
    """
    decision = 0
    forced = 0
    for class_id, state_edges in enumerate(dfa.edges):
        out_degree = len(state_edges)
        if out_degree == 0:
            continue
        decision += 1
        if out_degree == 1 and not dfa.accepting[class_id]:
            forced += 1
    value = (forced / decision) if decision else 0.0
    return {"numerator": forced, "denominator": decision, "value": value}


def minimal_completion_lengths(dfa: MinimizedDFA) -> dict[int, int]:
    """Minimal actions-to-acceptance per minimal state, as a histogram.

    Monotone fixpoint relaxation over the acyclic minimal DFA: accepting states
    complete in 0, others in ``1 + min`` over successors. Mirrors the
    ``minimal_completion_length`` decode precedent, computed statically here.
    """
    completion: list[int | None] = [
        0 if is_accepting else None for is_accepting in dfa.accepting
    ]
    changed = True
    while changed:
        changed = False
        for class_id, state_edges in enumerate(dfa.edges):
            if dfa.accepting[class_id]:
                continue
            best: int | None = None
            for child in state_edges.values():
                child_cost = completion[child]
                if child_cost is None:
                    continue
                candidate = child_cost + 1
                best = candidate if best is None else min(best, candidate)
            if best is not None and best != completion[class_id]:
                completion[class_id] = best
                changed = True

    histogram: dict[int, int] = {}
    for cost in completion:
        if cost is None:
            raise ValueError("state cannot reach acceptance; language not finite")
        histogram[cost] = histogram.get(cost, 0) + 1
    return histogram
