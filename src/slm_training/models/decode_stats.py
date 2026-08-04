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
    denoiser_rows_evaluated: int = 0
    ambiguous_rows_forwarded: int = 0
    forced_row_tokens_without_forward: int = 0
    all_forced_steps_without_forward: int = 0
    semantic_singleton_bypasses: int = 0
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
    compiler_prefill_batches: int = 0
    compiler_prefill_states: int = 0
    compiler_prefill_tokens: int = 0
    # SLM-293: inventory is a live factor bias, so causal-use controls need
    # the same eligible-position/argmax accounting as the other factor heads.
    component_inventory_applications: int = 0
    component_inventory_choice_changes: int = 0
    component_plan_applications: int = 0
    component_plan_choice_changes: int = 0
    semantic_plan_applications: int = 0
    semantic_plan_choice_changes: int = 0
    semantic_plan_binding_applications: int = 0
    semantic_plan_binding_choice_changes: int = 0
    semantic_plan_root_applications: int = 0
    semantic_plan_root_choice_changes: int = 0
    slot_component_applications: int = 0
    slot_component_choice_changes: int = 0
    slot_coverage_close_applications: int = 0
    slot_coverage_close_choice_changes: int = 0
    # E627: root-cause instrumentation for required_slot_margin_decode_weight
    # (E626) — counts how often the still-missing-required-slot floor fires
    # and how often it actually flips the position's argmax candidate.
    required_slot_margin_applications: int = 0
    required_slot_margin_choice_changes: int = 0
    visible_reference_applications: int = 0
    visible_reference_choice_changes: int = 0
    root_reference_arity_applications: int = 0
    root_reference_arity_choice_changes: int = 0
    root_reference_identity_applications: int = 0
    root_reference_identity_choice_changes: int = 0
    slot_alias_unique_applications: int = 0
    slot_alias_unique_choice_changes: int = 0
    required_slot_array_completion_applications: int = 0
    required_slot_array_completion_choice_changes: int = 0
    required_slot_root_completion_applications: int = 0
    required_slot_root_completion_choice_changes: int = 0
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
    # I3 deterministic speculative ranking over the symbol table: how often the
    # corpus scorer was consulted, how often it was confident enough to commit
    # without a forward, and how many tokens that bought.
    speculative_rank_evaluations: int = 0
    speculative_rank_commits: int = 0
    speculative_rank_tokens: int = 0
    speculative_rank_declined: int = 0
    # I4 symbol-table prefill scheduling: how many forwards the planner shaped,
    # how many rows it dropped from them, and how much canvas it did not read.
    scheduled_prefills: int = 0
    scheduled_rows_skipped: int = 0
    scheduled_prefill_tokens_saved: int = 0
    schedule_checkpoint_hits: int = 0
    choice_state_cache_hits: int = 0
    choice_state_cache_misses: int = 0
    choice_candidates_considered: int = 0
    choice_vocab_candidates_avoided: int = 0
    choice_completion_cache_hits: int = 0
    choice_completion_cache_misses: int = 0
    choice_schema_intern_hits: int = 0
    choice_schema_intern_misses: int = 0
    choice_distance_cache_hits: int = 0
    choice_distance_cache_misses: int = 0
    choice_distance_cache_evictions: int = 0
    choice_distance_cache_peak_entries: int = 0
    choice_clone_count: int = 0
    choice_frame_cow_copies: int = 0
    # Packed grammar-completion kernel.  These counters are request-local
    # deltas folded from CompletionSession snapshots; they must never be
    # assigned from a cumulative snapshot or repeated domain queries would
    # double count prior work.
    completion_session_starts: int = 0
    completion_state_intern_hits: int = 0
    completion_state_intern_misses: int = 0
    completion_unique_states: int = 0
    completion_edges_built: int = 0
    completion_transition_cache_hits: int = 0
    completion_transition_cache_misses: int = 0
    completion_domain_cache_hits: int = 0
    completion_domain_cache_misses: int = 0
    completion_reachability_cache_hits: int = 0
    completion_reachability_cache_misses: int = 0
    completion_witness_states_expanded: int = 0
    completion_forced_closure_hits: int = 0
    completion_forced_closure_tokens: int = 0
    completion_direct_terminal_feeds: int = 0
    completion_full_sync_fallbacks: int = 0
    completion_full_prefix_lex_bytes: int = 0
    completion_parser_forks: int = 0
    completion_candidate_engine_allocations: int = 0
    completion_scope_reference_scans_avoided: int = 0
    completion_branch_memo_hits: int = 0
    completion_branch_memo_misses: int = 0
    completion_shared_domain_hits: int = 0
    completion_shared_domain_misses: int = 0
    # Incremental DFA engine lifetime counters (engine.py `stats`), folded once
    # per request-local engine via `collect_engine_stats`. Zero on paths that
    # never fold an engine.
    dfa_full_syncs: int = 0
    dfa_incremental_advances: int = 0
    dfa_copy_probes: int = 0
    dfa_copy_probe_fallbacks: int = 0
    dfa_engine_sync_ms: float = 0.0
    dfa_full_sync_fallbacks: int = 0
    dfa_full_prefix_lex_bytes: int = 0
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
    certified_fallbacks: int = 0
    root_invariant_bypass_count: int = 0
    dynamic_mask_applications: int = 0
    dynamic_candidates_before: int = 0
    dynamic_candidates_after: int = 0
    # A2 (ASAp): constraint-violating (position, token) mass removals recorded.
    asap_penalties: int = 0
    # HX1 blast-radius (advisory): admit probes on canvases that still hold
    # committed tokens AFTER the first hole — the configuration admit_fill
    # cannot validate (left-prefix over-approximation; see residual_support).
    admit_probe_canvases: int = 0
    admit_probe_committed_suffix: int = 0
    # L-D: parallel block-step commits reverted because the joint canvas was
    # PROVEN uncompletable by multi_region_support (never on unknown/budget).
    block_joint_rejections: int = 0
    # HV-A: the completion-domain filter drops a candidate for BOTH a
    # certified UNSUPPORTED witness and a budget-exhausted UNKNOWN one
    # (dsl/pack.py). Only the first is a soundness-preserving rejection;
    # the second silently narrows the legal domain. Split so the two are
    # distinguishable, plus how many witnesses were built and kept.
    witness_pruned_unsupported: int = 0
    witness_pruned_unknown: int = 0
    witness_materialized: int = 0
    witness_kept: int = 0
    # A domain query ending with exactly ONE proven candidate while >=1
    # UNKNOWN (budget-exhausted) candidate was dropped: the survivor then
    # looks deterministically forced to exact_forced_token_id and bypasses
    # the model (I2), though the alternative was unproven, not impossible.
    witness_false_singleton_risk: int = 0
    # L-D/HX4: joint verdicts that came back without proof authority (node
    # budget exhausted). Commits are kept — counted so the fail-open share of
    # the joint validator is observable instead of silent.
    block_joint_unknowns: int = 0
    # HX4 hybrid unmask scheduler (unmask_mode="hybrid"): positions committed
    # through the block-scheduled span lane vs the frontier (positionwise) lane,
    # and span-only reverts taken before the all-or-nothing fallback.
    hybrid_span_commits: int = 0
    hybrid_frontier_commits: int = 0
    hybrid_span_reverts: int = 0
    # E20: masked slot positions seeded from the slot-contract template.
    template_slot_positions: int = 0
    asap_positions: int = 0
    constraint_graph_edges: int = 0
    completion_bound_known: int = 0
    completion_bound_unknown: int = 0
    # SLM-176: retrieve-then-rerank shortlist decision traces (default empty).
    action_shortlist_traces: list[dict[str, object]] = field(default_factory=list)
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

# Cooperative decode wall (monotonic deadline). SIGALRM alone can be swallowed by
# bare ``except Exception`` in the LTR/MaskGIT path after a one-shot timer fire;
# checking this deadline at loop heads makes the wall hard.
_DECODE_DEADLINE_MONO: float | None = None


def set_decode_deadline(seconds: float | None) -> None:
    """Arm a cooperative wall-clock deadline ``seconds`` from now (or clear)."""
    global _DECODE_DEADLINE_MONO
    if seconds is None or float(seconds) <= 0:
        _DECODE_DEADLINE_MONO = None
        return
    _DECODE_DEADLINE_MONO = time.monotonic() + float(seconds)


def clear_decode_deadline() -> None:
    """Disarm the cooperative decode deadline."""
    global _DECODE_DEADLINE_MONO
    _DECODE_DEADLINE_MONO = None


def decode_deadline_remaining() -> float | None:
    """Seconds remaining before deadline, or None if unarmed."""
    deadline = _DECODE_DEADLINE_MONO
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def check_decode_deadline() -> None:
    """Raise ``TimeoutError`` when the cooperative decode wall has elapsed."""
    deadline = _DECODE_DEADLINE_MONO
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("decode deadline exceeded")


def get_active_stats() -> DecodeStats | None:
    return _ACTIVE


_COMPLETION_COUNTER_FIELDS = {
    "session_starts": "completion_session_starts",
    "state_intern_hits": "completion_state_intern_hits",
    "state_intern_misses": "completion_state_intern_misses",
    "unique_states": "completion_unique_states",
    "edges_built": "completion_edges_built",
    "transition_cache_hits": "completion_transition_cache_hits",
    "transition_cache_misses": "completion_transition_cache_misses",
    "domain_cache_hits": "completion_domain_cache_hits",
    "domain_cache_misses": "completion_domain_cache_misses",
    "reachability_cache_hits": "completion_reachability_cache_hits",
    "reachability_cache_misses": "completion_reachability_cache_misses",
    "witness_states_expanded": "completion_witness_states_expanded",
    "forced_closure_hits": "completion_forced_closure_hits",
    "forced_closure_tokens": "completion_forced_closure_tokens",
    "direct_terminal_feeds": "completion_direct_terminal_feeds",
    "full_sync_fallbacks": "completion_full_sync_fallbacks",
    "full_prefix_lex_bytes": "completion_full_prefix_lex_bytes",
    "parser_forks": "completion_parser_forks",
    "candidate_engine_allocations": "completion_candidate_engine_allocations",
    "scope_reference_scans_avoided": "completion_scope_reference_scans_avoided",
    "branch_memo_hits": "completion_branch_memo_hits",
    "branch_memo_misses": "completion_branch_memo_misses",
}


def collect_completion_session_delta(
    session: Any,
    previous: dict[str, int] | None = None,
    *,
    stats: DecodeStats | None = None,
) -> dict[str, int]:
    """Fold new request-local completion work into a decode collector."""
    current = {
        str(key): int(value)
        for key, value in dict(session.stats()).items()
        if isinstance(value, (int, float))
    }
    bucket = stats if stats is not None else get_active_stats()
    if bucket is not None:
        before = previous or {}
        for counter, field_name in _COMPLETION_COUNTER_FIELDS.items():
            delta = current.get(counter, 0) - int(before.get(counter, 0))
            if delta > 0:
                setattr(bucket, field_name, int(getattr(bucket, field_name)) + delta)
    return current


_ENGINE_COUNTER_FIELDS = {
    "full_syncs": "dfa_full_syncs",
    "incremental_advances": "dfa_incremental_advances",
    "copy_probes": "dfa_copy_probes",
    "copy_probe_fallbacks": "dfa_copy_probe_fallbacks",
    "sync_ms": "dfa_engine_sync_ms",
    "full_sync_fallbacks": "dfa_full_sync_fallbacks",
    "full_prefix_lex_bytes": "dfa_full_prefix_lex_bytes",
}


def collect_engine_stats(engine: Any, stats: DecodeStats | None = None) -> None:
    """Fold a request-local incremental engine's lifetime counters once.

    Engines are allocated per request/state (`engine_for_dsl` returns a fresh
    instance), so folding the lifetime snapshot exactly once when the engine
    goes out of scope needs no delta bookkeeping.
    """
    bucket = stats if stats is not None else get_active_stats()
    if bucket is None:
        return
    try:
        current = dict(engine.stats)
    except Exception:  # noqa: BLE001
        return
    for counter, field_name in _ENGINE_COUNTER_FIELDS.items():
        value = current.get(counter, 0)
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        if field_name == "dfa_engine_sync_ms":
            bucket.dfa_engine_sync_ms += float(value)
        else:
            setattr(bucket, field_name, int(getattr(bucket, field_name)) + int(value))


# Coarse phase spans used for the advisory unattributed-time metric. These are
# top-level phases of a generate call; nested sliver charges (a pick inside the
# repair phase, dfa syncs inside a pick) can overlap by a few percent, so the
# derived fraction is a dark-cost detector, not precision accounting.
ATTRIBUTED_PHASE_FIELDS = (
    "context_ms",
    "denoiser_ms",
    "pick_ms",
    "compiler_ms",
    "solver_ms",
    "trie_ms",
    "finalize_ms",
    "dfa_sync_ms",
    "stream_check_ms",
    "detok_ms",
)


def attributed_time_summary(rows: list["DecodeStats"]) -> dict[str, float]:
    """Advisory coverage of total_ms by the coarse phase spans.

    `attributed_fraction` near 1.0 means the phase spans explain the wall
    time; a low value means a dominant cost has no span (the slm304 failure
    mode: build_completion_forest was 74% of eval cost and invisible to
    telemetry). Overlap can push the raw sum past total; the fraction is
    clamped to 1.0 and `unattributed_ms` to 0.
    """
    total = sum(float(getattr(r, "total_ms", 0.0)) for r in rows)
    attributed = sum(
        float(getattr(r, name, 0.0)) for r in rows for name in ATTRIBUTED_PHASE_FIELDS
    )
    if total <= 0.0:
        return {}
    return {
        "attributed_ms_sum": round(min(attributed, total), 3),
        "unattributed_ms_sum": round(max(0.0, total - attributed), 3),
        "attributed_fraction": round(min(1.0, attributed / total), 4),
    }


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
    except BaseException as exc:
        if getattr(exc, "decode_stats", None) is None:
            setattr(exc, "decode_stats", bucket)
        raise
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
        "denoiser_rows_evaluated",
        "ambiguous_rows_forwarded",
        "forced_row_tokens_without_forward",
        "all_forced_steps_without_forward",
        "semantic_singleton_bypasses",
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
        "compiler_prefill_batches",
        "compiler_prefill_states",
        "compiler_prefill_tokens",
        "component_inventory_applications",
        "component_inventory_choice_changes",
        "component_plan_applications",
        "component_plan_choice_changes",
        "semantic_plan_applications",
        "semantic_plan_choice_changes",
        "semantic_plan_binding_applications",
        "semantic_plan_binding_choice_changes",
        "semantic_plan_root_applications",
        "semantic_plan_root_choice_changes",
        "slot_component_applications",
        "slot_component_choice_changes",
        "slot_coverage_close_applications",
        "slot_coverage_close_choice_changes",
        "required_slot_margin_applications",
        "required_slot_margin_choice_changes",
        "visible_reference_applications",
        "visible_reference_choice_changes",
        "root_reference_arity_applications",
        "root_reference_arity_choice_changes",
        "root_reference_identity_applications",
        "root_reference_identity_choice_changes",
        "slot_alias_unique_applications",
        "slot_alias_unique_choice_changes",
        "required_slot_array_completion_applications",
        "required_slot_array_completion_choice_changes",
        "required_slot_root_completion_applications",
        "required_slot_root_completion_choice_changes",
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
        "speculative_rank_evaluations",
        "speculative_rank_commits",
        "speculative_rank_tokens",
        "speculative_rank_declined",
        "scheduled_prefills",
        "scheduled_rows_skipped",
        "scheduled_prefill_tokens_saved",
        "schedule_checkpoint_hits",
        "choice_state_cache_hits",
        "choice_state_cache_misses",
        "choice_candidates_considered",
        "choice_vocab_candidates_avoided",
        "choice_completion_cache_hits",
        "choice_completion_cache_misses",
        "choice_schema_intern_hits",
        "choice_schema_intern_misses",
        "choice_distance_cache_hits",
        "choice_distance_cache_misses",
        "choice_distance_cache_evictions",
        "choice_distance_cache_peak_entries",
        "choice_clone_count",
        "choice_frame_cow_copies",
        "completion_session_starts",
        "completion_state_intern_hits",
        "completion_state_intern_misses",
        "completion_unique_states",
        "completion_edges_built",
        "completion_transition_cache_hits",
        "completion_transition_cache_misses",
        "completion_domain_cache_hits",
        "completion_domain_cache_misses",
        "completion_reachability_cache_hits",
        "completion_reachability_cache_misses",
        "completion_witness_states_expanded",
        "completion_forced_closure_hits",
        "completion_forced_closure_tokens",
        "completion_direct_terminal_feeds",
        "completion_full_sync_fallbacks",
        "completion_full_prefix_lex_bytes",
        "completion_parser_forks",
        "completion_candidate_engine_allocations",
        "completion_scope_reference_scans_avoided",
        "completion_branch_memo_hits",
        "completion_branch_memo_misses",
        "completion_shared_domain_hits",
        "completion_shared_domain_misses",
        "dfa_full_syncs",
        "dfa_incremental_advances",
        "dfa_copy_probes",
        "dfa_copy_probe_fallbacks",
        "dfa_engine_sync_ms",
        "dfa_full_sync_fallbacks",
        "dfa_full_prefix_lex_bytes",
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
        "certified_fallbacks",
        "root_invariant_bypass_count",
        "dynamic_mask_applications",
        "dynamic_candidates_before",
        "dynamic_candidates_after",
        "asap_penalties",
        "admit_probe_canvases",
        "admit_probe_committed_suffix",
        "block_joint_rejections",
        "witness_pruned_unsupported",
        "witness_pruned_unknown",
        "witness_materialized",
        "witness_kept",
        "witness_false_singleton_risk",
        "block_joint_unknowns",
        "hybrid_span_commits",
        "hybrid_frontier_commits",
        "hybrid_span_reverts",
        "template_slot_positions",
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
    out.update(attributed_time_summary(rows))
    return out


__all__ = [
    "ATTRIBUTED_PHASE_FIELDS",
    "DecodeStats",
    "aggregate_stats",
    "attributed_time_summary",
    "collect_completion_session_delta",
    "collect_decode_stats",
    "collect_engine_stats",
    "get_active_stats",
    "set_active_stats",
    "timed_ms",
]
