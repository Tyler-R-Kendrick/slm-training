"""SLM-308 (LAR2-02): bounded distance oracle over canonical AST fingerprints.

Ported by SLM-434 (LAR0-07) from the never-merged SLM-308 fork (commit
``ae5448c5``) onto main's evolved ``tree_edit_diffusion``/``slm299_edit_reachability``
(the extended action space grew to 12 actions with SLM-425's SET_PROPERTY and
SLM-426's pack-derived component inventory, but the four SLM-299 primitives
this module calls — ``_canonical_key``, ``_check_invariants``,
``_enumerate_children``, ``_normalize_inventory`` — kept identical signatures,
so the oracle logic below is unchanged from the fork).

Replaces mutation-count value labels with an exact/bounded/UNKNOWN cost-to-go
signal for the X22 tree-edit model. The oracle runs a **reverse BFS from the
target state** over alpha-invariant canonical fingerprints (``_canonical_key``
from SLM-299) using the extended action space's transitions. Because every
extended action is invertible inside the space, the transition graph is
undirected: enumerating forward children of the target is exactly enumerating
its reverse neighborhood, so BFS depth from the target is the true shortest
edit distance for every state it reaches.

Three labels, and only three:

- ``EXACT(d)`` — proven shortest distance ``d``: the BFS reached the state and
  BFS layering certifies no shorter path exists.
- ``BOUNDED(lo, hi)`` — proof of bounds only, with a machine-readable reason.
  ``lo`` comes from fully explored BFS layers (the state was not within the
  deepest fully explored depth); ``hi`` comes from a witness path length
  (e.g. the mutation chain that produced a training state) when one is
  supplied. An invariant proof yields ``BOUNDED(None, None)`` — the target is
  unreachable under the exact action set, which is a *proof*, never a budget
  artifact.
- ``UNKNOWN`` — node budget exhausted with a live frontier. UNKNOWN is
  explicit and is never conflated with far or unreachable; downstream
  consumers (value loss) must exclude it, never coerce it.

Results are cached keyed by ``(action_schema_version, grammar_version,
inventory_hash, target_ast_hash, state_ast_hash, max_depth, witness)``. The
action schema version is the SLM-299 component version (it watches the edit
space); there is no explicit grammar backend version, so the grammar version
is a sha256 fragment of ``dsl/grammars/openui.lark`` — a grammar or action
change is a guaranteed cache miss.

The oracle is **training-time only**: it reads the gold target and must never
be reachable from any inference/decode path.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from slm_training.harness_core.versioning import component_version
from slm_training.harnesses.experiments.slm299_edit_reachability import (
    _canonical_key,
    _check_invariants,
    _enumerate_children,
    _normalize_inventory,
)
from slm_training.models.tree_edit_diffusion import Statement, TreeEditSpace

__all__ = [
    "DistanceKind",
    "DistanceLabel",
    "action_schema_version",
    "cache_stats",
    "clear_caches",
    "distance_to_target",
    "effective_distance",
    "grammar_version",
]

EXPERIMENT_ID = "slm308-distance-value"

_GRAMMAR_PATH = (
    Path(__file__).resolve().parents[2] / "dsl" / "grammars" / "openui.lark"
)

# Reason codes.
REASON_REACHED = "reached"
REASON_DEPTH_BOUND = "depth_bound"  # fully explored to max_depth; not found
REASON_FRONTIER_EXHAUSTED = "frontier_exhausted"  # frontier died inside bound
REASON_BUDGET = "budget"  # node budget exhausted with a live frontier
REASON_INVARIANT_PREFIX = "invariant:"  # + slm299 invariant reason code


class DistanceKind(str, Enum):
    """Proof status of one state→target distance query."""

    EXACT = "EXACT"
    BOUNDED = "BOUNDED"
    UNKNOWN = "UNKNOWN"


@dataclass
class DistanceLabel:
    """Distance proof for one (state, target) pair.

    ``distance`` is set only for EXACT. ``lo``/``hi`` are set for BOUNDED when
    the corresponding bound is proven (``hi=None`` means no finite upper bound
    was proven — e.g. an invariant-fired unreachable pair). ``nodes_expanded``
    is the oracle's own search cost for the BFS that decided this label.
    """

    kind: DistanceKind
    reason: str
    distance: int | None = None
    lo: int | None = None
    hi: int | None = None
    nodes_expanded: int = 0
    cache_hit: bool = False

    def value_target(self, max_depth: int) -> float | None:
        """Normalized cost-to-go value target in [0, 1] (higher = closer).

        EXACT ``d`` maps to ``1 - d/max_depth``; BOUNDED with both finite
        bounds maps to the flagged midpoint; BOUNDED without finite bounds and
        UNKNOWN return None — the state is excluded from value loss, never
        coerced.
        """
        eff = effective_distance(self)
        if eff is None:
            return None
        return max(0.0, 1.0 - min(eff, float(max_depth)) / float(max_depth))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "distance": self.distance,
            "lo": self.lo,
            "hi": self.hi,
            "nodes_expanded": self.nodes_expanded,
            "cache_hit": self.cache_hit,
        }


def effective_distance(label: DistanceLabel) -> float | None:
    """Comparable distance estimate: exact ``d`` or the flagged bound midpoint.

    None when the label carries no usable finite estimate (UNKNOWN, or
    invariant-proven unreachable with no witness). Callers must treat None as
    "exclude", never as far/unreachable.
    """
    if label.kind is DistanceKind.EXACT:
        return float(label.distance)
    if label.kind is DistanceKind.BOUNDED and label.lo is not None and label.hi is not None:
        return (label.lo + label.hi) / 2.0
    return None


def action_schema_version() -> str:
    """Version of the action space the oracle enumerates (SLM-299 component)."""
    return component_version("harness.experiments.slm299_edit_reachability")


def grammar_version() -> str:
    """Grammar identity: sha256 fragment of the Lark grammar the parser uses."""
    try:
        digest = hashlib.sha256(_GRAMMAR_PATH.read_bytes()).hexdigest()
    except OSError:
        return "unknown"
    return f"sha256:{digest[:16]}"


def _ast_hash(statements: list[Statement]) -> str:
    return hashlib.sha256(_canonical_key(statements).encode("utf-8")).hexdigest()[:16]


def _inventory_hash(inventory: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(inventory).encode("utf-8")).hexdigest()[:12]


# --- caches -----------------------------------------------------------------


@dataclass
class _TargetMap:
    """BFS distance map from one target, plus what the search proved."""

    dist: dict[str, int] = field(default_factory=dict)
    deepest_full_depth: int = -1  # every state at this depth was explored
    budget_exhausted: bool = False
    frontier_died: bool = False  # frontier provably emptied before max_depth
    nodes_expanded: int = 0


# target map: (action_ver, grammar_ver, inventory_hash, target_ast_hash, max_depth)
_TARGET_MAPS: dict[tuple[str, ...], _TargetMap] = {}
# label cache: target-map key + (state_ast_hash, witness)
_LABELS: dict[tuple, DistanceLabel] = {}
_STATS = {"label_hits": 0, "label_misses": 0, "map_builds": 0}


def cache_stats() -> dict[str, int | float]:
    return {
        **_STATS,
        "target_maps": len(_TARGET_MAPS),
        "labels": len(_LABELS),
        "map_nodes": sum(m.nodes_expanded for m in _TARGET_MAPS.values()),
    }


def clear_caches() -> None:
    _TARGET_MAPS.clear()
    _LABELS.clear()
    for key in _STATS:
        _STATS[key] = 0


def _build_target_map(
    target: list[Statement],
    inventory: list[str],
    space: TreeEditSpace,
    *,
    max_depth: int,
    node_budget: int,
) -> _TargetMap:
    """Layered BFS from the target over canonical fingerprints.

    Transitions are the real extended-space children via SLM-299's
    ``_enumerate_children`` applied through ``TreeEditSpace.apply`` — the
    analyzed space is the deployed space by construction. Because every action
    is invertible in-space, BFS depth from the target is the exact shortest
    distance for every reached state.
    """
    result = _TargetMap()
    target_key = _canonical_key(target)
    result.dist[target_key] = 0
    frontier: deque[list[Statement]] = deque([target])
    depth = 0
    while frontier and depth < max_depth:
        # Expand one full BFS layer; only fully explored layers raise
        # ``deepest_full_depth`` (a partial layer proves no lower bound).
        layer_size = len(frontier)
        layer_complete = True
        for _ in range(layer_size):
            if result.nodes_expanded >= node_budget:
                result.budget_exhausted = True
                layer_complete = False
                break
            statements = frontier.popleft()
            result.nodes_expanded += 1
            # Visited pruning is a pure search-efficiency hook: states already
            # keyed are rejected pre-validation inside ``apply``; it never
            # changes which distinct states are reachable.
            visited = set(result.dist)
            for child, _action in _enumerate_children(
                space, statements, inventory, mode="extended", visited=visited
            ):
                key = _canonical_key(child)
                if key in result.dist:
                    continue
                result.dist[key] = depth + 1
                frontier.append(child)
        if not layer_complete:
            break
        result.deepest_full_depth = depth
        depth += 1
    if not frontier and not result.budget_exhausted:
        # Frontier provably died: every reachable state within the
        # MAX_STMTS-bounded space was enumerated, so unreached states are
        # beyond any depth the space connects — still reported via bounds.
        result.frontier_died = True
    return result


def distance_to_target(
    statements: list[Statement],
    target_statements: list[Statement],
    *,
    space: TreeEditSpace,
    inventory: Sequence[str] = (),
    max_depth: int = 8,
    node_budget: int = 600,
    upper_bound_witness: int | None = None,
) -> DistanceLabel:
    """Distance proof from ``statements`` to ``target_statements``.

    ``upper_bound_witness`` is a proven path length (e.g. the mutation chain
    that produced a training state from the target); it only ever tightens
    ``hi``, never ``lo``. UNKNOWN is returned on budget exhaustion and is
    never conflated with far or unreachable.
    """
    versions = (action_schema_version(), grammar_version())
    inv = _normalize_inventory(inventory)
    inv_hash = _inventory_hash(inv)
    state_hash = _ast_hash(statements)
    target_hash = _ast_hash(target_statements)
    map_key = (*versions, inv_hash, target_hash, str(max_depth))
    label_key = (map_key, state_hash, upper_bound_witness, node_budget)

    cached = _LABELS.get(label_key)
    if cached is not None:
        _STATS["label_hits"] += 1
        return DistanceLabel(
            kind=cached.kind,
            reason=cached.reason,
            distance=cached.distance,
            lo=cached.lo,
            hi=cached.hi,
            nodes_expanded=cached.nodes_expanded,
            cache_hit=True,
        )
    _STATS["label_misses"] += 1

    if state_hash == target_hash:
        label = DistanceLabel(
            kind=DistanceKind.EXACT, reason=REASON_REACHED, distance=0
        )
        _LABELS[label_key] = label
        return label

    # Structural impossibility proofs over the EXACT extended action set
    # (container_add is a real capability of the extended space).
    fired = _check_invariants(
        target_statements,
        statements,
        inv,
        space,
        frozenset({"container_add"}),
        extended=True,
    )
    if fired is not None:
        label = DistanceLabel(
            kind=DistanceKind.BOUNDED,
            reason=f"{REASON_INVARIANT_PREFIX}{fired}",
            lo=None,
            hi=None,
        )
        _LABELS[label_key] = label
        return label

    target_map = _TARGET_MAPS.get(map_key)
    if target_map is None:
        _STATS["map_builds"] += 1
        target_map = _build_target_map(
            target_statements,
            inv,
            space,
            max_depth=max_depth,
            node_budget=node_budget,
        )
        _TARGET_MAPS[map_key] = target_map

    state_key = _canonical_key(statements)
    exact = target_map.dist.get(state_key)
    if exact is not None:
        label = DistanceLabel(
            kind=DistanceKind.EXACT,
            reason=REASON_REACHED,
            distance=exact,
            nodes_expanded=target_map.nodes_expanded,
        )
    elif target_map.budget_exhausted:
        # Live frontier when the budget cut the search: nothing is proven
        # about this state beyond the fully explored layers, and a witness
        # alone never upgrades an UNKNOWN to bounded-exact.
        if upper_bound_witness is not None and target_map.deepest_full_depth >= 0:
            label = DistanceLabel(
                kind=DistanceKind.BOUNDED,
                reason=REASON_BUDGET,
                lo=target_map.deepest_full_depth + 2,
                hi=upper_bound_witness,
                nodes_expanded=target_map.nodes_expanded,
            )
        else:
            label = DistanceLabel(
                kind=DistanceKind.UNKNOWN,
                reason=REASON_BUDGET,
                nodes_expanded=target_map.nodes_expanded,
            )
    else:
        if target_map.frontier_died:
            # Complete proof: the target's connected component was fully
            # enumerated and the state is not in it — unreachable, with the
            # same status as an invariant proof (no finite bounds).
            label = DistanceLabel(
                kind=DistanceKind.BOUNDED,
                reason=REASON_FRONTIER_EXHAUSTED,
                lo=None,
                hi=None,
                nodes_expanded=target_map.nodes_expanded,
            )
        else:
            # Depth bound cut a live frontier: distances up to max_depth were
            # enumerated, so this state's distance is provably > max_depth.
            label = DistanceLabel(
                kind=DistanceKind.BOUNDED,
                reason=REASON_DEPTH_BOUND,
                lo=max_depth + 1,
                hi=upper_bound_witness,
                nodes_expanded=target_map.nodes_expanded,
            )
    _LABELS[label_key] = label
    return label


def timed_distance_to_target(**kwargs) -> tuple[DistanceLabel, float]:
    """``distance_to_target`` plus wall-clock milliseconds (oracle cost evidence)."""
    start = time.perf_counter()
    label = distance_to_target(**kwargs)
    return label, (time.perf_counter() - start) * 1000.0
