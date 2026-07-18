"""Deterministic bounded enumeration -> prefix trie for arith-sketch (CAP0-02).

The bounded fixture ``bounded-expr`` is the finite language of canonical
straight-line arith-sketch programs under an :class:`AnalysisBounds` frame:

* at most ``max_live_bindings + 1`` statements, the last bound to ``root``;
* each statement RHS is a canonical expression whose node count fits the total
  ``max_ast_nodes`` budget (and optional ``max_ast_depth``);
* de Bruijn refs point only to the ``min(i, max_live_bindings)`` most-recent
  prior binders (the scope window);
* **liveness**: every non-root binder is referenced by a later statement (no
  dead bindings);
* **type validity**: the materialised source is accepted by
  ``arith-sketch`` ``backend.validate`` — so a ``root`` that resolves to a bare
  atom is rejected *before counting*.

Enumeration is fully deterministic (statement count ascending, then size tuples,
then :func:`itertools.product` over per-size candidate lists in a fixed order).
The distinct valid canonical programs are linearised to preorder action
sequences and folded into a prefix trie; a coarse *frontier x scope* structural
signature is attached to every trie node for the raw-state count.

Nothing here imports torch (or anything that does); the package is Torch-free by
construction.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from slm_training.dsl.analysis.arity.canonical import (
    CanonicalProgram,
    Expr,
    NUMBER_CLASS,
    expr_depth,
    is_type_valid,
    lit,
    materialize,
    op,
    program_actions,
    ref,
    ref_deltas,
)

# Registry of committed fixtures: name -> (operators, template_classes, types).
FIXTURES: dict[str, dict[str, tuple[str, ...]]] = {
    "bounded-expr": {
        "operators": ("+", "-", "*", "/"),
        "template_classes": (NUMBER_CLASS,),
        "result_types": ("number",),
    },
}


@dataclass(frozen=True)
class EnumerationBounds:
    """Minimal bound view the enumerator needs (mirrors report.AnalysisBounds)."""

    max_ast_nodes: int
    max_ast_depth: int | None
    max_live_bindings: int
    operators: tuple[str, ...]
    template_classes: tuple[str, ...]


def _exprs_of_size(
    size: int,
    live: int,
    ops: tuple[str, ...],
    classes: tuple[str, ...],
    cache: dict[tuple[int, int], list[Expr]],
) -> list[Expr]:
    """All canonical exprs with exactly ``size`` nodes, deterministic order."""
    key = (size, live)
    if key in cache:
        return cache[key]
    out: list[Expr] = []
    if size == 1:
        out.extend(lit(cls) for cls in classes)
        out.extend(ref(delta) for delta in range(1, live + 1))
    elif size % 2 == 1:
        for symbol in ops:
            for left_size in range(1, size - 1, 2):
                right_size = size - 1 - left_size
                for left in _exprs_of_size(left_size, live, ops, classes, cache):
                    for right in _exprs_of_size(right_size, live, ops, classes, cache):
                        out.append(op(symbol, left, right))
    cache[key] = out
    return out


def _size_buckets(
    max_nodes: int,
    max_depth: int | None,
    live: int,
    ops: tuple[str, ...],
    classes: tuple[str, ...],
    cache: dict[tuple[int, int], list[Expr]],
) -> dict[int, list[Expr]]:
    """Candidate exprs grouped by node count (respecting an optional depth cap)."""
    buckets: dict[int, list[Expr]] = {}
    for size in range(1, max_nodes + 1):
        exprs = _exprs_of_size(size, live, ops, classes, cache)
        if max_depth is not None:
            exprs = [expr for expr in exprs if expr_depth(expr) <= max_depth]
        if exprs:
            buckets[size] = exprs
    return buckets


def _size_tuples(
    available: list[list[int]], budget: int
) -> list[tuple[int, ...]]:
    """Size tuples (one per statement) with total nodes <= ``budget``."""
    results: list[tuple[int, ...]] = []

    def walk(index: int, chosen: list[int], used: int) -> None:
        if index == len(available):
            results.append(tuple(chosen))
            return
        for size in available[index]:
            if used + size > budget:
                continue
            chosen.append(size)
            walk(index + 1, chosen, used + size)
            chosen.pop()

    walk(0, [], 0)
    return results


def _is_live(program: CanonicalProgram) -> bool:
    """Whether every non-root binder is referenced by a later statement."""
    referenced: set[int] = set()
    for index, stmt in enumerate(program.statements):
        for delta in ref_deltas(stmt):
            referenced.add(index - delta)
    return all(j in referenced for j in range(program.num_statements - 1))


@dataclass
class EnumerationResult:
    """Deterministic enumeration output plus honest work counters."""

    programs: tuple[CanonicalProgram, ...]
    complete: bool
    work: dict[str, int] = field(default_factory=dict)


def enumerate_programs(
    bounds: EnumerationBounds, *, max_programs: int = 1_000_000
) -> EnumerationResult:
    """Enumerate all valid canonical programs of the bounded fixture.

    Returns programs deduped by canonical fingerprint in first-seen order and a
    ``complete`` flag (``False`` iff the ``max_programs`` safety cap was hit, so
    a caller can fail closed on an incomplete analysis).
    """
    ops = bounds.operators
    classes = bounds.template_classes
    max_nodes = bounds.max_ast_nodes
    max_depth = bounds.max_ast_depth
    max_live = bounds.max_live_bindings
    max_stmts = max_live + 1
    cache: dict[tuple[int, int], list[Expr]] = {}

    seen: set[str] = set()
    programs: list[CanonicalProgram] = []
    work = {
        "candidates_enumerated": 0,
        "candidates_not_live": 0,
        "validate_calls": 0,
        "validate_rejected": 0,
        "dedup_collisions": 0,
    }
    complete = True

    for num_stmts in range(1, max_stmts + 1):
        buckets = [
            _size_buckets(
                max_nodes, max_depth, min(i, max_live), ops, classes, cache
            )
            for i in range(num_stmts)
        ]
        available = [sorted(bucket) for bucket in buckets]
        if any(not sizes for sizes in available):
            continue
        for size_tuple in _size_tuples(available, max_nodes):
            lists = [buckets[i][size_tuple[i]] for i in range(num_stmts)]
            for combo in itertools.product(*lists):
                work["candidates_enumerated"] += 1
                program = CanonicalProgram(tuple(combo))
                if not _is_live(program):
                    work["candidates_not_live"] += 1
                    continue
                work["validate_calls"] += 1
                if not is_type_valid(materialize(program)):
                    work["validate_rejected"] += 1
                    continue
                fingerprint = program.fingerprint()
                if fingerprint in seen:
                    work["dedup_collisions"] += 1
                    continue
                seen.add(fingerprint)
                programs.append(program)
                if len(programs) > max_programs:
                    complete = False
                    return EnumerationResult(tuple(programs), complete, work)

    return EnumerationResult(tuple(programs), complete, work)


# --- prefix trie -----------------------------------------------------------

@dataclass
class TrieNode:
    """One prefix state of the deterministic action automaton."""

    node_id: int
    prefix: tuple[str, ...]
    depth: int
    completed: int
    pending_height: int
    accepting: bool = False
    edges: dict[str, int] = field(default_factory=dict)

    @property
    def expected_type(self) -> str:
        return "expr" if self.pending_height > 0 else "stmt_or_end"

    def frontier(self) -> tuple[str, ...]:
        return ("expr",) * self.pending_height

    def scope_signature(self, max_live: int) -> int:
        return min(self.completed, max_live)


@dataclass
class Trie:
    """Prefix trie over the action alphabet with per-node parse configs."""

    nodes: list[TrieNode]
    max_live: int

    @property
    def root(self) -> TrieNode:
        return self.nodes[0]

    def action_alphabet(self) -> set[str]:
        alphabet: set[str] = set()
        for node in self.nodes:
            alphabet.update(node.edges)
        return alphabet


def _step_config(node: TrieNode, action: str) -> tuple[int, int]:
    """(completed, pending_height) after taking ``action`` from ``node``."""
    completed = node.completed
    height = node.pending_height
    if action == "=":
        return completed, height + 1
    if action.startswith("o:"):
        return completed, height + 1
    # atom (literal ``#…`` or ref ``~…``): fills one pending obligation.
    height -= 1
    if height == 0:
        completed += 1
    return completed, height


def build_trie(programs: tuple[CanonicalProgram, ...], max_live: int) -> Trie:
    """Fold the programs' action sequences into a prefix trie with configs."""
    root = TrieNode(
        node_id=0, prefix=(), depth=0, completed=0, pending_height=0
    )
    nodes: list[TrieNode] = [root]
    index: dict[tuple[str, ...], TrieNode] = {(): root}

    for program in programs:
        actions = program_actions(program)
        node = root
        for step, action in enumerate(actions, start=1):
            child_id = node.edges.get(action)
            if child_id is None:
                completed, height = _step_config(node, action)
                child = TrieNode(
                    node_id=len(nodes),
                    prefix=node.prefix + (action,),
                    depth=step,
                    completed=completed,
                    pending_height=height,
                )
                nodes.append(child)
                index[child.prefix] = child
                node.edges[action] = child.node_id
                node = child
            else:
                node = nodes[child_id]
        node.accepting = True

    return Trie(nodes=nodes, max_live=max_live)


def structural_signatures(trie: Trie) -> set[tuple[Any, ...]]:
    """Distinct coarse *frontier x scope* signatures over all trie nodes.

    This is the ``raw_state_count`` quotient: it deliberately ignores the exact
    action prefix and the remaining node budget, keeping only
    ``(frontier, scope, expected_type, template_state)``. It is the local analog
    of the external CAP0-01 "raw frontier x scope" notion and is **not** claimed
    to reproduce that source-reported value.
    """
    signatures: set[tuple[Any, ...]] = set()
    for node in trie.nodes:
        signatures.add(
            (
                node.frontier(),
                node.scope_signature(trie.max_live),
                node.expected_type,
            )
        )
    return signatures


def scope_signature_values(trie: Trie) -> set[int]:
    """Distinct scope-window sizes realised across the trie."""
    return {node.scope_signature(trie.max_live) for node in trie.nodes}
