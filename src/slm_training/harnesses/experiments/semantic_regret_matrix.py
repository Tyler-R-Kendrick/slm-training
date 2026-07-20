"""SLM-143 SPV0-03: bounded completion enumeration and semantic regret decomposition.

Wiring/fixture harness only. The toy finite graph lets us compute exact regret
terms in a deterministic setting; real OpenUI completion enumeration requires a
full grammar/search backend and a trained model. No ship claim, no production
promotion, no GPU run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanIdentity,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.versioning import UNKNOWN

__all__ = [
    "SEMANTIC_REGRET_SCHEMA",
    "BoundedCompletionState",
    "CompletionSnapshot",
    "RegretMetrics",
    "RegretReport",
    "SemanticRegretMatrixReport",
    "compiler_choice_adapter",
    "compute_regret_from_trace",
    "enumerate_bounded_completions",
    "plan_regret_delta",
    "selector_candidate_adapter",
    "x22_adapter",
]

SEMANTIC_REGRET_SCHEMA = "SemanticRegretReportV1"


@dataclass(frozen=True)
class BoundedCompletionState:
    """One node in a toy finite completion graph."""

    state_id: str
    # action_id -> (next_state_id or None for terminal, value, accepted, prune_cause)
    actions: Mapping[
        str, tuple[str | None, float, bool, str | None]
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for action_id, spec in self.actions.items():
            if not isinstance(spec, tuple) or len(spec) != 4:
                raise ValueError(
                    f"action {action_id!r} must be a 4-tuple "
                    "(next_state_id, value, accepted, prune_cause)"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "actions": {
                action_id: {
                    "next_state_id": spec[0],
                    "value": spec[1],
                    "accepted": spec[2],
                    "prune_cause": spec[3],
                }
                for action_id, spec in self.actions.items()
            },
        }


@dataclass(frozen=True)
class CompletionSnapshot:
    """Result of a bounded enumeration over the completion graph."""

    start_state_id: str
    terminal_values: tuple[float, ...]
    accepted_values: tuple[float, ...]
    pruned_values: tuple[float, ...]
    coverage_complete: bool
    nodes_visited: int
    bound_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_state_id": self.start_state_id,
            "terminal_values": list(self.terminal_values),
            "accepted_values": list(self.accepted_values),
            "pruned_values": list(self.pruned_values),
            "coverage_complete": self.coverage_complete,
            "nodes_visited": self.nodes_visited,
            "bound_hit": self.bound_hit,
        }


@dataclass(frozen=True)
class RegretMetrics:
    """Per-state regret decomposition recorded during a policy trace."""

    state_id: str
    legal_actions: tuple[str, ...]
    chosen_action: str | None
    oracle_best_value: float | None
    representation_regret: float | str  # 0.0 if reachable, UNKNOWN otherwise
    candidate_coverage: bool
    acceptable_action_rank: int | str  # 1-based rank, UNKNOWN if no accepted action
    local_regret: float | str  # value(best_accepted) - value(chosen)
    pruning_regret: float | str  # best pruned completion - best retained completion
    global_rank_regret: float | str  # best accepted value - chosen value
    plan_regret: float | str  # populated by plan_regret_delta
    unknown: bool

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["legal_actions"] = list(self.legal_actions)
        return data


@dataclass(frozen=True)
class RegretReport:
    """Regret decomposition along a single policy trace."""

    start_state_id: str
    policy_actions: tuple[str, ...]
    max_nodes: int
    snapshots: tuple[RegretMetrics, ...]
    completion_snapshot: CompletionSnapshot
    plan_factor_meta: dict[str, Any] = field(default_factory=dict)
    unknown: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["policy_actions"] = list(self.policy_actions)
        data["snapshots"] = [s.to_dict() for s in self.snapshots]
        data["completion_snapshot"] = self.completion_snapshot.to_dict()
        return data


@dataclass(frozen=True)
class SemanticRegretMatrixReport:
    """Container for one or more regret arms plus metadata."""

    schema: str = SEMANTIC_REGRET_SCHEMA
    claim_class: str = "wiring"
    status: str = "fixture"
    arms: Mapping[str, RegretReport] = field(default_factory=dict)
    plan_deltas: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    timestamp: str | None = None
    version_stamp: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["arms"] = {
            name: report.to_dict() for name, report in self.arms.items()
        }
        data["plan_deltas"] = {
            name: dict(delta) for name, delta in self.plan_deltas.items()
        }
        return data


def enumerate_bounded_completions(
    start_state_id: str,
    graph: Mapping[str, BoundedCompletionState],
    *,
    max_nodes: int = 256,
) -> CompletionSnapshot:
    """Enumerate reachable terminal completions up to ``max_nodes``.

    Performs a deterministic breadth-first walk. Returns terminal values,
    accepted terminal values, pruned terminal values, and whether the graph was
    fully covered before the bound.
    """
    if start_state_id not in graph:
        return CompletionSnapshot(
            start_state_id=start_state_id,
            terminal_values=(),
            accepted_values=(),
            pruned_values=(),
            coverage_complete=False,
            nodes_visited=0,
            bound_hit=False,
        )

    terminal_values: list[float] = []
    accepted_values: list[float] = []
    pruned_values: list[float] = []
    visited: set[str] = set()
    queue: list[str] = [start_state_id]

    while queue and len(visited) < max_nodes:
        state_id = queue.pop(0)
        if state_id in visited or state_id is None:
            continue
        visited.add(state_id)
        state = graph.get(state_id)
        if state is None:
            continue
        for action_id, (next_state_id, value, accepted, prune_cause) in state.actions.items():
            if next_state_id is None:
                terminal_values.append(value)
                if accepted:
                    accepted_values.append(value)
                if prune_cause is not None:
                    pruned_values.append(value)
            elif next_state_id not in visited:
                queue.append(next_state_id)

    bound_hit = bool(queue)

    return CompletionSnapshot(
        start_state_id=start_state_id,
        terminal_values=tuple(sorted(terminal_values)),
        accepted_values=tuple(sorted(accepted_values)),
        pruned_values=tuple(sorted(pruned_values)),
        coverage_complete=not bound_hit,
        nodes_visited=len(visited),
        bound_hit=bound_hit,
    )


def _best_reachable_value(
    start_state_id: str,
    graph: Mapping[str, BoundedCompletionState],
    *,
    max_nodes: int = 256,
) -> float | None:
    snapshot = enumerate_bounded_completions(
        start_state_id, graph, max_nodes=max_nodes
    )
    if not snapshot.terminal_values:
        return None
    return max(snapshot.terminal_values)


def compute_regret_from_trace(
    start_state_id: str,
    policy_actions: Sequence[str],
    graph: Mapping[str, BoundedCompletionState],
    *,
    max_nodes: int = 256,
    oracle_best_value: float | None = None,
) -> RegretReport:
    """Follow ``policy_actions`` and record per-state regret terms.

    The policy is a sequence of action ids starting at ``start_state_id``. The
    trace stops when the action is terminal, unknown, or the sequence ends.
    """
    snapshots: list[RegretMetrics] = []
    current_state_id: str | None = start_state_id
    unknown = False

    start_snapshot = enumerate_bounded_completions(
        start_state_id, graph, max_nodes=max_nodes
    )
    global_best_accepted_value = (
        max(start_snapshot.accepted_values)
        if start_snapshot.accepted_values
        else None
    )

    # If caller did not supply the oracle best value, derive it from the graph.
    if oracle_best_value is None:
        oracle_best_value = (
            max(start_snapshot.terminal_values)
            if start_snapshot.terminal_values
            else None
        )

    for chosen_action in policy_actions:
        if current_state_id is None:
            break
        state = graph.get(current_state_id)
        if state is None:
            unknown = True
            break

        legal_actions = tuple(sorted(state.actions.keys()))
        action_spec = state.actions.get(chosen_action)
        chosen_value: float | None = None
        chosen_accepted = False
        if action_spec is not None:
            next_state_id, immediate_value, chosen_accepted, _ = action_spec
            if next_state_id is None:
                chosen_value = immediate_value
            else:
                # Effective value of a non-terminal action is the best accepted
                # completion reachable from the next state.
                next_snapshot = enumerate_bounded_completions(
                    next_state_id, graph, max_nodes=max_nodes
                )
                if next_snapshot.accepted_values:
                    chosen_value = max(next_snapshot.accepted_values)
                elif next_snapshot.terminal_values:
                    chosen_value = max(next_snapshot.terminal_values)
                else:
                    chosen_value = immediate_value

        # Enumerate the reachable completions from this state.
        local_snapshot = enumerate_bounded_completions(
            current_state_id, graph, max_nodes=max_nodes
        )
        local_best = (
            max(local_snapshot.terminal_values)
            if local_snapshot.terminal_values
            else None
        )
        local_best_accepted = (
            max(local_snapshot.accepted_values)
            if local_snapshot.accepted_values
            else None
        )

        # Representation regret: is the global oracle best reachable from here?
        representation_regret: float | str
        if local_best is None:
            representation_regret = UNKNOWN
        elif oracle_best_value is not None and local_best >= oracle_best_value:
            representation_regret = 0.0
        elif oracle_best_value is None:
            representation_regret = UNKNOWN
        else:
            # The oracle best is not reachable from this state.
            representation_regret = UNKNOWN

        candidate_coverage = bool(local_snapshot.accepted_values)

        # Acceptable-action rank: position of the best accepted action when all
        # legal actions are ranked by value (used as a model-score proxy in the
        # fixture). Rank 1 means the best accepted action is the top-scoring
        # action at this state.
        acceptable_action_rank: int | str
        if local_best_accepted is not None:
            better_count = sum(
                1
                for action_id, (next_state_id, value, accepted, prune_cause)
                in state.actions.items()
                if value > local_best_accepted
            )
            acceptable_action_rank = better_count + 1
        else:
            acceptable_action_rank = UNKNOWN

        # Local regret: zero when the chosen action is itself accepted; otherwise
        # the gap to the best accepted action reachable from this state.
        chosen_accepted = action_spec is not None and action_spec[2]
        if chosen_accepted:
            local_regret = 0.0
        elif local_best_accepted is not None and chosen_value is not None:
            local_regret = local_best_accepted - chosen_value
        else:
            local_regret = UNKNOWN

        # Pruning regret: best pruned completion minus best retained completion.
        retained_values: list[float] = []
        pruned_completion_values: list[float] = []
        for action_id, (next_state_id, value, accepted, prune_cause) in state.actions.items():
            pool = pruned_completion_values if prune_cause is not None else retained_values
            if next_state_id is None:
                pool.append(value)
            else:
                pool.extend(
                    enumerate_bounded_completions(
                        next_state_id, graph, max_nodes=max_nodes
                    ).terminal_values
                )

        if not pruned_completion_values:
            pruning_regret = 0.0
        elif retained_values:
            pruning_regret = max(pruned_completion_values) - max(retained_values)
        else:
            pruning_regret = max(pruned_completion_values)

        # Global rank regret: globally best accepted value minus chosen value.
        if global_best_accepted_value is not None and chosen_value is not None:
            global_rank_regret = global_best_accepted_value - chosen_value
        else:
            global_rank_regret = UNKNOWN

        snapshots.append(
            RegretMetrics(
                state_id=current_state_id,
                legal_actions=legal_actions,
                chosen_action=chosen_action if action_spec is not None else None,
                oracle_best_value=oracle_best_value,
                representation_regret=representation_regret,
                candidate_coverage=candidate_coverage,
                acceptable_action_rank=acceptable_action_rank,
                local_regret=local_regret,
                pruning_regret=pruning_regret,
                global_rank_regret=global_rank_regret,
                plan_regret=0.0,
                unknown=local_snapshot.bound_hit,
            )
        )

        # Advance the trace.
        if action_spec is None:
            current_state_id = None
            unknown = True
            break
        next_state_id, *_ = action_spec
        current_state_id = next_state_id
        if local_snapshot.bound_hit:
            unknown = True

    completion_snapshot = enumerate_bounded_completions(
        start_state_id, graph, max_nodes=max_nodes
    )

    return RegretReport(
        start_state_id=start_state_id,
        policy_actions=tuple(policy_actions),
        max_nodes=max_nodes,
        snapshots=tuple(snapshots),
        completion_snapshot=completion_snapshot,
        unknown=unknown or completion_snapshot.bound_hit,
    )


def plan_regret_delta(
    baseline_report: RegretReport,
    oracle_report: RegretReport,
) -> dict[str, Any]:
    """Return per-factor and per-snapshot deltas when an oracle plan is substituted.

    The fixture uses the same completion graph for both reports, so the only
    planned-in delta is the difference in policy traces. Real plan-regret
    decomposition will use oracle substitution per factor
    (``PlanOracleSubstitutor``) against a live model.
    """
    per_snapshot: list[dict[str, Any]] = []
    for baseline, oracle in zip(baseline_report.snapshots, oracle_report.snapshots):
        def _delta(b: float | str, o: float | str) -> float | str:
            if isinstance(b, str) or isinstance(o, str):
                return UNKNOWN
            return round(o - b, 9)

        per_snapshot.append(
            {
                "state_id": baseline.state_id,
                "local_regret_delta": _delta(baseline.local_regret, oracle.local_regret),
                "global_rank_regret_delta": _delta(
                    baseline.global_rank_regret, oracle.global_rank_regret
                ),
                "representation_regret_delta": _delta(
                    baseline.representation_regret, oracle.representation_regret
                ),
                "pruning_regret_delta": _delta(
                    baseline.pruning_regret, oracle.pruning_regret
                ),
                "plan_regret_delta": _delta(baseline.plan_regret, oracle.plan_regret),
            }
        )

    return {
        "baseline_policy": list(baseline_report.policy_actions),
        "oracle_policy": list(oracle_report.policy_actions),
        "per_snapshot": per_snapshot,
        "summary": {
            "n_snapshots": len(per_snapshot),
            "baseline_unknown": baseline_report.unknown,
            "oracle_unknown": oracle_report.unknown,
        },
    }


def compiler_choice_adapter() -> dict[str, Any]:
    """Placeholder adapter for compiler-choice regret decomposition."""
    return {
        "status": UNKNOWN,
        "reason": "compiler-choice regret decomposition not implemented in fixture",
    }


def x22_adapter() -> dict[str, Any]:
    """Placeholder adapter for X22 tree-edit regret decomposition."""
    return {
        "status": UNKNOWN,
        "reason": "x22 tree-edit regret decomposition not implemented in fixture",
    }


def selector_candidate_adapter() -> dict[str, Any]:
    """Placeholder adapter for selector candidate-set regret decomposition."""
    return {
        "status": UNKNOWN,
        "reason": "selector candidate-set regret decomposition not implemented in fixture",
    }


def make_semantic_plan_fixture(
    *,
    provenance: str,
    archetype_id: str,
    role_ids: Sequence[str] = (),
) -> SemanticPlanV1:
    """Build a tiny SemanticPlanV1 for plan-factor substitution wiring tests."""
    return SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="fixture_spv0_03",
            provenance=provenance,  # type: ignore[arg-type]
        ),
        archetype=PlanArchetype(
            id=archetype_id,
            distribution={archetype_id: 1.0},
            confidence=0.9,
        ),
        role_slots=tuple(
            RoleSlot(role_id=role_id, component_family="fixture")
            for role_id in role_ids
        ),
    )


def build_fixture_graph() -> dict[str, BoundedCompletionState]:
    """Return a deterministic toy graph with known exact regrets.

    Graph structure::

        start
          ├── accept_good (terminal, value=1.0, accepted=True)
          ├── accept_ok   (terminal, value=0.6, accepted=True)
          ├── prune_high  (terminal, value=1.5, accepted=False, prune_cause="budget")
          └── continue    -> mid
                                ├── target (terminal, value=2.0, accepted=True)
                                └── dead_end (terminal, value=0.0, accepted=False)

    The global oracle best reachable from ``start`` is 2.0 via ``continue`` ->
    ``mid`` -> ``target``. ``prune_high`` has a higher immediate value than the
    accepted actions but is pruned (``prune_cause="budget"``), so it must not be
    counted as a scoring regret.
    """
    return {
        "start": BoundedCompletionState(
            state_id="start",
            actions={
                "accept_good": (None, 1.0, True, None),
                "accept_ok": (None, 0.6, True, None),
                "prune_high": (None, 1.5, False, "budget"),
                "continue": ("mid", 0.0, False, None),
            },
        ),
        "mid": BoundedCompletionState(
            state_id="mid",
            actions={
                "target": (None, 2.0, True, None),
                "dead_end": (None, 0.0, False, None),
            },
        ),
    }


def build_unreachable_graph() -> dict[str, BoundedCompletionState]:
    """Graph where the oracle-best target is not reachable from ``start``."""
    return {
        "start": BoundedCompletionState(
            state_id="start",
            actions={
                "accept_low": (None, 0.5, True, None),
                "goto_mid": ("mid", 0.0, False, None),
            },
        ),
        "mid": BoundedCompletionState(
            state_id="mid",
            actions={
                "accept_mid": (None, 0.8, True, None),
            },
        ),
        "unreachable": BoundedCompletionState(
            state_id="unreachable",
            actions={
                "oracle_best": (None, 2.0, True, None),
            },
        ),
    }
