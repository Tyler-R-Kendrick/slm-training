"""Conflict-slice localized remask vs full remask and suffix rollback (EFS2-03).

This module provides the schema, slicers, and repair-policy executor for a
preregistered comparison of localized conflict-slice repair against baseline
repair policies.  It is wiring-grade code: it does not modify the live decode
loop and makes no ship-quality claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.versioning import build_version_stamp


CompletenessClass = Literal["EXACT", "SOUND_OVERAPPROX", "HEURISTIC"]
RepairPolicyName = Literal[
    "none",
    "suffix_rollback",
    "full_remask",
    "conflict_slice",
    "conflict_slice_expanded",
]
ConflictStage = Literal[
    "grammar",
    "schema",
    "binding",
    "slot_contract",
    "component_inventory",
    "whole_program",
]


@dataclass(frozen=True)
class TopologyNode:
    """Simplified topology node used for conflict-slice fixtures."""

    node_id: int
    node_type: str
    parent_id: int | None = None
    children: tuple["TopologyNode", ...] = ()
    active: bool = True
    protected: bool = False
    certified: bool = False
    decision_level: int = 0

    def clone(self) -> "TopologyNode":
        return TopologyNode(
            node_id=self.node_id,
            node_type=self.node_type,
            parent_id=self.parent_id,
            children=tuple(child.clone() for child in self.children),
            active=self.active,
            protected=self.protected,
            certified=self.certified,
            decision_level=self.decision_level,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "parent_id": self.parent_id,
            "children": [child.to_dict() for child in self.children],
            "active": self.active,
            "protected": self.protected,
            "certified": self.certified,
            "decision_level": self.decision_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyNode":
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            parent_id=data.get("parent_id"),
            children=tuple(
                cls.from_dict(child) for child in data.get("children", [])
            ),
            active=data.get("active", True),
            protected=data.get("protected", False),
            certified=data.get("certified", False),
            decision_level=data.get("decision_level", 0),
        )


@dataclass(frozen=True)
class ConflictSliceV1:
    """Replayable conflict slice used to authorize localized remasking."""

    conflict_id: str
    stage: ConflictStage
    reason_code: str
    failing_node_ids: tuple[int, ...]
    dependency_frontier: tuple[int, ...]
    protected_node_ids: tuple[int, ...]
    completeness_class: CompletenessClass
    original_state_fingerprint: str
    source_provenance: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "failing_node_ids": list(self.failing_node_ids),
            "dependency_frontier": list(self.dependency_frontier),
            "protected_node_ids": list(self.protected_node_ids),
            "completeness_class": self.completeness_class,
            "original_state_fingerprint": self.original_state_fingerprint,
            "source_provenance": self.source_provenance,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictSliceV1":
        return cls(
            conflict_id=data["conflict_id"],
            stage=data["stage"],
            reason_code=data["reason_code"],
            failing_node_ids=tuple(data.get("failing_node_ids", [])),
            dependency_frontier=tuple(data.get("dependency_frontier", [])),
            protected_node_ids=tuple(data.get("protected_node_ids", [])),
            completeness_class=data["completeness_class"],
            original_state_fingerprint=data["original_state_fingerprint"],
            source_provenance=data.get("source_provenance", ""),
            notes=data.get("notes", ""),
        )

    def can_authorize_repair(self) -> bool:
        """Only EXACT or SOUND_OVERAPPROX slices may drive primary localized repair."""
        return self.completeness_class in ("EXACT", "SOUND_OVERAPPROX")


@dataclass(frozen=True)
class RepairTrace:
    """Replayable record of one repair attempt."""

    trace_id: str
    conflict_id: str
    policy: RepairPolicyName
    seed: int
    original_tree: TopologyNode
    repaired_tree: TopologyNode
    remasked_node_ids: tuple[int, ...]
    protected_mutations: int
    budget_forwards: int
    budget_verifier_calls: int
    recovered: bool
    repeated_conflict: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "conflict_id": self.conflict_id,
            "policy": self.policy,
            "seed": self.seed,
            "original_tree": self.original_tree.to_dict(),
            "repaired_tree": self.repaired_tree.to_dict(),
            "remasked_node_ids": list(self.remasked_node_ids),
            "protected_mutations": self.protected_mutations,
            "budget_forwards": self.budget_forwards,
            "budget_verifier_calls": self.budget_verifier_calls,
            "recovered": self.recovered,
            "repeated_conflict": self.repeated_conflict,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RepairOutcome:
    """Aggregate outcome for one policy on one conflict fixture."""

    conflict_id: str
    policy: RepairPolicyName
    seeds: tuple[int, ...]
    traces: tuple[RepairTrace, ...]
    recovery_rate: float
    mean_remasked_nodes: float
    mean_preserved_nodes: float
    mean_forwards: float
    mean_verifier_calls: float
    protected_mutations_total: int
    repeated_conflict_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "policy": self.policy,
            "seeds": list(self.seeds),
            "recovery_rate": self.recovery_rate,
            "mean_remasked_nodes": self.mean_remasked_nodes,
            "mean_preserved_nodes": self.mean_preserved_nodes,
            "mean_forwards": self.mean_forwards,
            "mean_verifier_calls": self.mean_verifier_calls,
            "protected_mutations_total": self.protected_mutations_total,
            "repeated_conflict_rate": self.repeated_conflict_rate,
            "traces": [t.to_dict() for t in self.traces],
        }


def _tree_fingerprint(root: TopologyNode) -> str:
    """Stable fingerprint of a topology tree."""
    payload = json.dumps(root.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _walk(root: TopologyNode):
    yield root
    for child in root.children:
        yield from _walk(child)


def _find_node(root: TopologyNode, node_id: int) -> TopologyNode | None:
    for node in _walk(root):
        if node.node_id == node_id:
            return node
    return None


def _node_count(root: TopologyNode) -> int:
    return sum(1 for _ in _walk(root))


def _active_node_count(root: TopologyNode) -> int:
    return sum(1 for node in _walk(root) if node.active)


def _with_replaced_node(
    root: TopologyNode,
    target_id: int,
    replacement: TopologyNode,
) -> TopologyNode:
    """Return a new tree with ``target_id`` replaced by ``replacement``."""
    if root.node_id == target_id:
        return replacement
    return TopologyNode(
        node_id=root.node_id,
        node_type=root.node_type,
        parent_id=root.parent_id,
        children=tuple(
            _with_replaced_node(child, target_id, replacement)
            for child in root.children
        ),
        active=root.active,
        protected=root.protected,
        certified=root.certified,
        decision_level=root.decision_level,
    )


def _remask_node(node: TopologyNode) -> TopologyNode:
    """Remask a node: clear children and mark active."""
    return TopologyNode(
        node_id=node.node_id,
        node_type="MASK",
        parent_id=node.parent_id,
        children=(),
        active=True,
        protected=node.protected,
        certified=False,
        decision_level=node.decision_level,
    )


def _collect_subtree_ids(root: TopologyNode) -> frozenset[int]:
    return frozenset(node.node_id for node in _walk(root))


def _nodes_at_or_after_level(root: TopologyNode, level: int) -> frozenset[int]:
    """Return IDs of nodes whose decision level is >= ``level``."""
    return frozenset(
        node.node_id for node in _walk(root) if node.decision_level >= level
    )


def _apply_remask_set(
    root: TopologyNode,
    remask_ids: frozenset[int],
    protected_ids: frozenset[int],
) -> tuple[TopologyNode, int, int]:
    """Apply remasking to ``remask_ids`` while refusing to touch protected nodes.

    Returns (repaired_tree, remasked_count, protected_mutations).
    """
    if not remask_ids:
        return root.clone(), 0, 0

    protected_mutations = 0

    def remap(node: TopologyNode) -> TopologyNode:
        nonlocal protected_mutations
        if node.node_id in remask_ids:
            if node.protected or node.node_id in protected_ids:
                protected_mutations += 1
                # Leave protected node untouched.
                return node.clone()
            new_node = _remask_node(node)
            # Recursively remap children of the original node, but since we are
            # clearing children in the new node we drop the subtree.
            return new_node
        return TopologyNode(
            node_id=node.node_id,
            node_type=node.node_type,
            parent_id=node.parent_id,
            children=tuple(remap(child) for child in node.children),
            active=node.active,
            protected=node.protected,
            certified=node.certified,
            decision_level=node.decision_level,
        )

    repaired = remap(root)
    remasked_count = len(remask_ids - protected_ids)
    return repaired, remasked_count, protected_mutations


def apply_repair_policy(
    root: TopologyNode,
    slice_: ConflictSliceV1,
    policy: RepairPolicyName,
    *,
    seed: int = 0,
    budget_forwards: int = 64,
    budget_verifier_calls: int = 16,
    max_remask_nodes: int | None = None,
) -> RepairTrace:
    """Apply one repair policy to a conflict and return a replayable trace.

    The implementation is intentionally synthetic: it models the structural
    effect of each policy on the simplified topology tree and records budget
    accounting.  It does not run a model or verifier.

    Localized slice policies on a ``HEURISTIC`` slice are refused: the trace
    records ``recovered=False`` and ``repeated_conflict=True`` instead of
    raising, so campaign comparisons can still include them.
    """
    original_tree = root.clone()
    original_fp = _tree_fingerprint(original_tree)
    if original_fp != slice_.original_state_fingerprint:
        raise ValueError(
            f"state fingerprint mismatch: tree={original_fp} "
            f"slice={slice_.original_state_fingerprint}"
        )

    protected_ids = frozenset(slice_.protected_node_ids)
    remask_ids: frozenset[int] = frozenset()
    refused = False

    if policy == "none":
        pass
    elif policy == "suffix_rollback":
        max_level = max(
            (
                node.decision_level
                for node in _walk(original_tree)
                if node.node_id in slice_.failing_node_ids
            ),
            default=0,
        )
        remask_ids = _nodes_at_or_after_level(original_tree, max_level)
    elif policy == "full_remask":
        remask_ids = frozenset(
            node.node_id
            for node in _walk(original_tree)
            if node.active and not node.protected and not node.certified
        )
    elif policy in ("conflict_slice", "conflict_slice_expanded"):
        if not slice_.can_authorize_repair():
            refused = True
        elif policy == "conflict_slice":
            remask_ids = frozenset(slice_.failing_node_ids) | frozenset(
                slice_.dependency_frontier
            )
        else:
            base = frozenset(slice_.failing_node_ids) | frozenset(
                slice_.dependency_frontier
            )
            # One-hop expansion: include parents of remask nodes unless protected.
            expanded: set[int] = set(base)
            for node in _walk(original_tree):
                if node.node_id in base and node.parent_id is not None:
                    expanded.add(node.parent_id)
            remask_ids = frozenset(expanded)
    else:
        raise ValueError(f"unknown repair policy: {policy}")

    if max_remask_nodes is not None and len(remask_ids) > max_remask_nodes:
        # Truncate to budget; for determinism, sort by node_id.
        remask_ids = frozenset(sorted(remask_ids)[:max_remask_nodes])

    remask_ids = remask_ids - protected_ids

    repaired, remasked_count, protected_mutations = _apply_remask_set(
        original_tree, remask_ids, protected_ids
    )

    # Synthetic recovery signal: localized repair is more likely to recover when
    # the slice is exact and the policy touches the true failing nodes.
    recovered = (
        not refused
        and slice_.completeness_class == "EXACT"
        and policy in ("conflict_slice", "conflict_slice_expanded")
        and protected_mutations == 0
        and remasked_count > 0
    )
    repeated_conflict = (
        refused
        or (
            policy in ("conflict_slice", "conflict_slice_expanded")
            and slice_.completeness_class == "HEURISTIC"
        )
    )

    trace_id = f"{slice_.conflict_id}-{policy}-s{seed}"
    notes = f"completeness={slice_.completeness_class}"
    if refused:
        notes += " refused=heuristic_slice"
    return RepairTrace(
        trace_id=trace_id,
        conflict_id=slice_.conflict_id,
        policy=policy,
        seed=seed,
        original_tree=original_tree,
        repaired_tree=repaired,
        remasked_node_ids=tuple(sorted(remask_ids)),
        protected_mutations=protected_mutations,
        budget_forwards=budget_forwards,
        budget_verifier_calls=budget_verifier_calls,
        recovered=recovered,
        repeated_conflict=repeated_conflict,
        notes=notes,
    )


def compare_repair_policies(
    root: TopologyNode,
    slice_: ConflictSliceV1,
    policies: tuple[RepairPolicyName, ...] = (
        "none",
        "suffix_rollback",
        "full_remask",
        "conflict_slice",
        "conflict_slice_expanded",
    ),
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    budget_forwards: int = 64,
    budget_verifier_calls: int = 16,
    max_remask_nodes: int | None = None,
) -> dict[RepairPolicyName, RepairOutcome]:
    """Run every policy over ``seeds`` and aggregate outcomes."""
    outcomes: dict[RepairPolicyName, RepairOutcome] = {}
    for policy in policies:
        traces: list[RepairTrace] = []
        for seed in seeds:
            trace = apply_repair_policy(
                root,
                slice_,
                policy,
                seed=seed,
                budget_forwards=budget_forwards,
                budget_verifier_calls=budget_verifier_calls,
                max_remask_nodes=max_remask_nodes,
            )
            traces.append(trace)

        recovery_rate = sum(1 for t in traces if t.recovered) / len(traces)
        mean_remasked = sum(len(t.remasked_node_ids) for t in traces) / len(traces)
        original_count = _node_count(root)
        preserved_counts = [
            original_count - len(t.remasked_node_ids) for t in traces
        ]
        mean_preserved = sum(preserved_counts) / len(preserved_counts)
        mean_forwards = sum(t.budget_forwards for t in traces) / len(traces)
        mean_verifier = sum(t.budget_verifier_calls for t in traces) / len(traces)
        protected_total = sum(t.protected_mutations for t in traces)
        repeated_rate = sum(1 for t in traces if t.repeated_conflict) / len(traces)

        outcomes[policy] = RepairOutcome(
            conflict_id=slice_.conflict_id,
            policy=policy,
            seeds=seeds,
            traces=tuple(traces),
            recovery_rate=recovery_rate,
            mean_remasked_nodes=mean_remasked,
            mean_preserved_nodes=mean_preserved,
            mean_forwards=mean_forwards,
            mean_verifier_calls=mean_verifier,
            protected_mutations_total=protected_total,
            repeated_conflict_rate=repeated_rate,
        )
    return outcomes


def save_outcomes(
    outcomes: dict[RepairPolicyName, RepairOutcome],
    path: Path | str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "efs2-03-repair-outcomes/v1",
        "version_stamp": build_version_stamp("harness.experiments"),
        "outcomes": {policy: outcome.to_dict() for policy, outcome in outcomes.items()},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
