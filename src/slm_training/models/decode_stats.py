"""Per-phase decode latency instrumentation for generate / LTR paths."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass
class DecodeStats:
    """Accumulated wall-clock timings and counters for one generate call."""

    denoiser_ms: float = 0.0
    dfa_sync_ms: float = 0.0
    stream_check_ms: float = 0.0
    detok_ms: float = 0.0
    context_ms: float = 0.0
    finalize_ms: float = 0.0
    pick_ms: float = 0.0
    total_ms: float = 0.0
    forwards_count: int = 0
    probes_count: int = 0
    dfa_sync_count: int = 0
    tokens_emitted: int = 0
    attempts: int = 1
    accepted_run_tokens: int = 0  # P3 multi-token accepts beyond the first
    canvas_tokens: int = 0
    unconstrained_retries: int = 0  # grammar decode fell back to unfiltered retry
    backbone_ms: float = 0.0
    projection_ms: float = 0.0
    compiler_ms: float = 0.0
    trie_ms: float = 0.0
    compiler_candidates: int = 0
    component_plan_applications: int = 0
    component_plan_choice_changes: int = 0
    slot_component_applications: int = 0
    slot_component_choice_changes: int = 0
    visible_reference_applications: int = 0
    visible_reference_choice_changes: int = 0
    root_reference_arity_applications: int = 0
    root_reference_arity_choice_changes: int = 0
    component_edge_applications: int = 0
    component_edge_choice_changes: int = 0
    binder_component_plan_applications: int = 0
    binder_component_plan_choice_changes: int = 0
    binder_topology_applications: int = 0
    binder_topology_choice_changes: int = 0
    binder_arity_applications: int = 0
    binder_arity_choice_changes: int = 0
    forced_spans: int = 0
    forced_tokens: int = 0
    choice_state_cache_hits: int = 0
    choice_state_cache_misses: int = 0
    choice_candidates_considered: int = 0
    choice_vocab_candidates_avoided: int = 0
    choice_completion_cache_hits: int = 0
    choice_completion_cache_misses: int = 0
    trie_nodes: int = 0
    restricted_projections: int = 0
    full_projections: int = 0
    compiler_fallbacks: int = 0
    seeded_fallbacks: int = 0
    compiler_lattice_states: int = 0
    compiler_lattice_candidates: int = 0
    compiler_lattice_bottoms: int = 0
    compiler_lattice_rollbacks: int = 0
    compiler_lattice_nogoods: int = 0
    compiler_lattice_nogood_hits: int = 0
    compiler_lattice_trajectory_triggers: int = 0
    compiler_lattice_trajectories: int = 0
    compiler_lattice_unique_proposals: int = 0
    compiler_lattice_recurrences: int = 0
    compiler_lattice_stagnation_triggers: int = 0
    compiler_lattice_bottom_triggers: int = 0
    compiler_lattice_always_triggers: int = 0
    compiler_lattice_abstentions: int = 0
    compiler_lattice_budget_exhaustions: int = 0
    compiler_lattice_false_hard_eliminations: int = 0
    compiler_lattice_max_rollback_depth: int = 0
    compiler_lattice_valid_trajectories: int = 0
    compiler_lattice_unique_valid_asts: int = 0
    compiler_lattice_verifier_calls: int = 0
    compiler_lattice_invalid_selected_over_valid: int = 0
    compiler_lattice_selector_regret: float = 0.0
    compiler_lattice_last_signature: str = ""
    compiler_lattice_termination_reason: str = ""
    template_fastpath_count: int = 0
    template_fallback_count: int = 0
    root_invariant_bypass_count: int = 0
    dynamic_mask_applications: int = 0
    dynamic_candidates_before: int = 0
    dynamic_candidates_after: int = 0
    # A2 (ASAp): constraint-violating (position, token) mass removals recorded.
    asap_penalties: int = 0
    asap_positions: int = 0
    constraint_graph_edges: int = 0
    completion_bound_known: int = 0
    completion_bound_unknown: int = 0
    # VSS1-04 (SLM-64): verified-solver decode work metrics. Zero on every
    # historical/default path (solver disabled); solver wall time is separated
    # from denoiser_ms/projection_ms. Names are stable and documented in
    # docs/design/telemetry.md.
    solver_ms: float = 0.0
    solver_enabled: int = 0
    solver_closure_passes: int = 0
    solver_support_queries: int = 0
    solver_support_cache_hits: int = 0
    solver_supported: int = 0
    solver_unsupported: int = 0
    solver_unknown: int = 0
    solver_certified_removed: int = 0
    solver_decisions: int = 0
    solver_backtracks: int = 0
    solver_nogoods: int = 0
    solver_expanded_nodes: int = 0
    solver_verifier_calls: int = 0
    solver_certificate_replay_failures: int = 0
    solver_terminal_status: str = ""
    constrained_dead_ends: int = 0
    constrained_dead_end_last_position: int = -1
    constrained_dead_end_forced_rank: int = -1
    constrained_last_legal_candidates: int = -1
    constrained_dead_end_candidate_count: int = 0
    constrained_dead_end_traces: list[dict[str, object]] = field(default_factory=list)
    # Bounded prefix/choice evidence for diagnosing the first bad constrained
    # decision without emitting an unbounded trace for long canvases.
    constrained_selection_traces: list[dict[str, object]] = field(default_factory=list)
    newline_commit_traces: list[dict[str, object]] = field(default_factory=list)

    def add_ms(self, field_name: str, ms: float) -> None:
        setattr(self, field_name, float(getattr(self, field_name)) + float(ms))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def merge(self, other: "DecodeStats") -> None:
        for key, value in other.as_dict().items():
            if key == "attempts":
                self.attempts = max(self.attempts, int(value))
                continue
            cur = getattr(self, key)
            if isinstance(cur, (int, float)) and isinstance(value, (int, float)):
                setattr(self, key, cur + value)


@contextmanager
def timed_ms(stats: DecodeStats | None, field_name: str) -> Iterator[None]:
    """Accumulate wall time into ``stats.<field_name>`` when stats is set."""
    if stats is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.add_ms(field_name, (time.perf_counter() - t0) * 1000.0)


# Thread-local-ish active stats for helpers that cannot take an explicit arg.
_ACTIVE: DecodeStats | None = None


def get_active_stats() -> DecodeStats | None:
    return _ACTIVE


def set_active_stats(stats: DecodeStats | None) -> DecodeStats | None:
    global _ACTIVE
    prev = _ACTIVE
    _ACTIVE = stats
    return prev


@contextmanager
def collect_decode_stats(stats: DecodeStats | None = None) -> Iterator[DecodeStats]:
    """Activate a DecodeStats collector for nested grammar/decode helpers."""
    bucket = stats if stats is not None else DecodeStats()
    prev = set_active_stats(bucket)
    t0 = time.perf_counter()
    try:
        yield bucket
    finally:
        bucket.total_ms += (time.perf_counter() - t0) * 1000.0
        set_active_stats(prev)


def aggregate_stats(rows: list[DecodeStats]) -> dict[str, Any]:
    """Mean / sum summary across multiple generate calls."""
    if not rows:
        return {}
    keys = [
        "denoiser_ms",
        "dfa_sync_ms",
        "stream_check_ms",
        "detok_ms",
        "context_ms",
        "finalize_ms",
        "pick_ms",
        "total_ms",
        "forwards_count",
        "probes_count",
        "dfa_sync_count",
        "tokens_emitted",
        "accepted_run_tokens",
        "canvas_tokens",
        "unconstrained_retries",
        "backbone_ms",
        "projection_ms",
        "compiler_ms",
        "trie_ms",
        "compiler_candidates",
        "component_plan_applications",
        "component_plan_choice_changes",
        "slot_component_applications",
        "slot_component_choice_changes",
        "visible_reference_applications",
        "visible_reference_choice_changes",
        "root_reference_arity_applications",
        "root_reference_arity_choice_changes",
        "component_edge_applications",
        "component_edge_choice_changes",
        "binder_component_plan_applications",
        "binder_component_plan_choice_changes",
        "binder_topology_applications",
        "binder_topology_choice_changes",
        "binder_arity_applications",
        "binder_arity_choice_changes",
        "forced_spans",
        "forced_tokens",
        "choice_state_cache_hits",
        "choice_state_cache_misses",
        "choice_candidates_considered",
        "choice_vocab_candidates_avoided",
        "choice_completion_cache_hits",
        "choice_completion_cache_misses",
        "trie_nodes",
        "restricted_projections",
        "full_projections",
        "compiler_fallbacks",
        "seeded_fallbacks",
        "compiler_lattice_states",
        "compiler_lattice_candidates",
        "compiler_lattice_bottoms",
        "compiler_lattice_rollbacks",
        "compiler_lattice_nogoods",
        "compiler_lattice_nogood_hits",
        "compiler_lattice_trajectory_triggers",
        "compiler_lattice_trajectories",
        "compiler_lattice_unique_proposals",
        "compiler_lattice_recurrences",
        "compiler_lattice_stagnation_triggers",
        "compiler_lattice_bottom_triggers",
        "compiler_lattice_always_triggers",
        "compiler_lattice_abstentions",
        "compiler_lattice_budget_exhaustions",
        "compiler_lattice_false_hard_eliminations",
        "compiler_lattice_max_rollback_depth",
        "compiler_lattice_valid_trajectories",
        "compiler_lattice_unique_valid_asts",
        "compiler_lattice_verifier_calls",
        "compiler_lattice_invalid_selected_over_valid",
        "compiler_lattice_selector_regret",
        "constrained_dead_ends",
        "constrained_dead_end_last_position",
        "constrained_dead_end_forced_rank",
        "constrained_last_legal_candidates",
        "constrained_dead_end_candidate_count",
        "template_fastpath_count",
        "template_fallback_count",
        "root_invariant_bypass_count",
        "dynamic_mask_applications",
        "dynamic_candidates_before",
        "dynamic_candidates_after",
        "asap_penalties",
        "asap_positions",
        "constraint_graph_edges",
        "completion_bound_known",
        "completion_bound_unknown",
        "solver_ms",
        "solver_enabled",
        "solver_closure_passes",
        "solver_support_queries",
        "solver_support_cache_hits",
        "solver_supported",
        "solver_unsupported",
        "solver_unknown",
        "solver_certified_removed",
        "solver_decisions",
        "solver_backtracks",
        "solver_nogoods",
        "solver_expanded_nodes",
        "solver_verifier_calls",
        "solver_certificate_replay_failures",
    ]
    out: dict[str, Any] = {"n": len(rows)}
    # Timings always report (a 0ms phase is a measurement); feature counters
    # report only when they fired — a disabled feature's counter is noise, so
    # it is omitted and named in counters_omitted_zero (self-describing: the
    # counter was measured at zero, not unmeasured).
    omitted_zero: list[str] = []
    for key in keys:
        vals = [float(getattr(r, key)) for r in rows]
        total = sum(vals)
        if total == 0.0 and not key.endswith("_ms"):
            omitted_zero.append(key)
            continue
        out[f"{key}_sum"] = round(total, 3)
        out[f"{key}_mean"] = round(total / len(vals), 3)
    out["counters_omitted_zero"] = sorted(omitted_zero)
    totals = sorted(float(r.total_ms) for r in rows)

    def _nearest_rank(fraction: float) -> float | None:
        if not totals:
            return None
        # Nearest-rank percentiles keep p95 >= p50 for tiny benchmark samples.
        index = max(0, min(len(totals) - 1, math.ceil(fraction * len(totals)) - 1))
        return totals[index]

    out["total_ms_p50"] = _nearest_rank(0.50)
    out["total_ms_p95"] = _nearest_rank(0.95)
    out["constrained_dead_end_traces"] = [
        trace for row in rows for trace in row.constrained_dead_end_traces
    ]
    out["constrained_selection_traces"] = [
        trace for row in rows for trace in row.constrained_selection_traces
    ]
    out["newline_commit_traces"] = [
        trace for row in rows for trace in row.newline_commit_traces
    ]
    return out


__all__ = [
    "DecodeStats",
    "aggregate_stats",
    "collect_decode_stats",
    "get_active_stats",
    "set_active_stats",
    "timed_ms",
]
