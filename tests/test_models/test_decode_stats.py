from slm_training.models.decode_stats import (
    DecodeStats,
    aggregate_stats,
    collect_decode_stats,
    collect_completion_session_delta,
)
from slm_training.harnesses.model_build.eval_runner import _nearest_rank


def test_small_sample_percentiles_are_monotonic() -> None:
    rows = [DecodeStats(total_ms=value) for value in (10.0, 20.0)]
    summary = aggregate_stats(rows)
    assert summary["total_ms_p50"] == 10.0
    assert summary["total_ms_p95"] == 20.0


def test_percentiles_use_nearest_rank() -> None:
    rows = [DecodeStats(total_ms=value) for value in (10.0, 20.0, 30.0, 40.0)]
    summary = aggregate_stats(rows)
    assert summary["total_ms_p50"] == 20.0
    assert summary["total_ms_p95"] == 40.0


def test_eval_percentile_matches_decode_percentile() -> None:
    values = [10.0, 20.0]
    assert _nearest_rank(values, 0.50) == 10.0
    assert _nearest_rank(values, 0.95) == 20.0


def test_decode_stats_merge_counts_unconstrained_retries() -> None:
    total = DecodeStats(unconstrained_retries=1)
    total.merge(DecodeStats(unconstrained_retries=2))
    assert total.unconstrained_retries == 3


def test_decode_stats_attaches_live_bucket_to_exception() -> None:
    try:
        with collect_decode_stats() as stats:
            stats.completion_witness_states_expanded = 7
            raise KeyboardInterrupt
    except KeyboardInterrupt as exc:
        assert exc.decode_stats is stats  # type: ignore[attr-defined]
        assert exc.decode_stats.completion_witness_states_expanded == 7  # type: ignore[attr-defined]
    else:
        raise AssertionError("expected KeyboardInterrupt")


def test_decode_stats_aggregates_certified_fallbacks() -> None:
    summary = aggregate_stats([DecodeStats(certified_fallbacks=2)])
    assert summary["certified_fallbacks_sum"] == 2.0


def test_decode_stats_aggregates_row_bypass_counts() -> None:
    stats = DecodeStats(
        denoiser_rows_evaluated=2,
        ambiguous_rows_forwarded=2,
        forced_row_tokens_without_forward=3,
        all_forced_steps_without_forward=1,
    )
    summary = aggregate_stats([stats])
    assert summary["denoiser_rows_evaluated_sum"] == 2.0
    assert summary["ambiguous_rows_forwarded_sum"] == 2.0
    assert summary["forced_row_tokens_without_forward_sum"] == 3.0
    assert summary["all_forced_steps_without_forward_sum"] == 1.0


def test_decode_stats_tracks_constrained_dead_ends() -> None:
    stats = DecodeStats(constrained_dead_ends=2)
    assert stats.as_dict()["constrained_dead_ends"] == 2
    assert aggregate_stats([stats])["constrained_dead_ends_sum"] == 2.0


def test_decode_stats_aggregates_choice_state_cache_counts() -> None:
    stats = DecodeStats(
        choice_state_cache_hits=7,
        choice_state_cache_misses=2,
        choice_candidates_considered=31,
        choice_vocab_candidates_avoided=1200,
        choice_completion_cache_hits=29,
        choice_completion_cache_misses=3,
        choice_schema_intern_hits=11,
        choice_distance_cache_hits=13,
        choice_distance_cache_evictions=2,
        choice_distance_cache_peak_entries=64,
        choice_clone_count=17,
        choice_frame_cow_copies=5,
    )
    summary = aggregate_stats([stats])
    assert summary["choice_state_cache_hits_sum"] == 7.0
    assert summary["choice_state_cache_misses_sum"] == 2.0
    assert summary["choice_candidates_considered_sum"] == 31.0
    assert summary["choice_vocab_candidates_avoided_sum"] == 1200.0
    assert summary["choice_completion_cache_hits_sum"] == 29.0
    assert summary["choice_completion_cache_misses_sum"] == 3.0
    assert summary["choice_schema_intern_hits_sum"] == 11.0
    assert summary["choice_distance_cache_hits_sum"] == 13.0
    assert summary["choice_distance_cache_evictions_sum"] == 2.0
    assert summary["choice_distance_cache_peak_entries_sum"] == 64.0
    assert summary["choice_clone_count_sum"] == 17.0
    assert summary["choice_frame_cow_copies_sum"] == 5.0


def test_completion_session_snapshots_are_folded_as_deltas() -> None:
    class Session:
        counters = {"session_starts": 1, "unique_states": 2, "edges_built": 1}

        def stats(self):
            return dict(self.counters)

    session = Session()
    stats = DecodeStats()
    snapshot = collect_completion_session_delta(session, stats=stats)
    session.counters.update(unique_states=3, edges_built=4)
    collect_completion_session_delta(session, snapshot, stats=stats)

    assert stats.completion_session_starts == 1
    assert stats.completion_unique_states == 3
    assert stats.completion_edges_built == 4
    summary = aggregate_stats([stats])
    assert summary["completion_unique_states_sum"] == 3.0
    assert summary["completion_edges_built_sum"] == 4.0


def test_decode_stats_aggregates_root_reference_arity_counts() -> None:
    stats = DecodeStats(
        root_reference_arity_applications=7,
        root_reference_arity_choice_changes=2,
    )
    summary = aggregate_stats([stats])
    assert summary["root_reference_arity_applications_sum"] == 7.0
    assert summary["root_reference_arity_choice_changes_sum"] == 2.0


def test_decode_stats_aggregates_component_inventory_choice_counts() -> None:
    stats = DecodeStats(
        component_inventory_applications=7,
        component_inventory_choice_changes=2,
    )
    summary = aggregate_stats([stats])
    assert summary["component_inventory_applications_sum"] == 7.0
    assert summary["component_inventory_choice_changes_sum"] == 2.0


def test_decode_stats_aggregates_required_slot_margin_counts() -> None:
    """E627: root-cause instrumentation counters for E626's required_slot_margin_decode_weight."""
    stats = DecodeStats(
        required_slot_margin_applications=5,
        required_slot_root_completion_applications=2,
        required_slot_root_completion_choice_changes=1,
        required_slot_margin_choice_changes=4,
    )
    summary = aggregate_stats([stats])
    assert summary["required_slot_margin_applications_sum"] == 5.0
    assert summary["required_slot_root_completion_applications_sum"] == 2.0
    assert summary["required_slot_root_completion_choice_changes_sum"] == 1.0
    assert summary["required_slot_margin_choice_changes_sum"] == 4.0


def test_decode_stats_aggregates_dead_end_traces() -> None:
    stats = DecodeStats(constrained_dead_end_traces=[{"position": 1}])
    assert aggregate_stats([stats])["constrained_dead_end_traces"] == [{"position": 1}]


def test_decode_stats_aggregates_bounded_selection_traces() -> None:
    stats = DecodeStats(constrained_selection_traces=[{"position": 2, "chosen_token": "="}])
    assert aggregate_stats([stats])["constrained_selection_traces"] == [
        {"position": 2, "chosen_token": "="}
    ]


def test_decode_stats_aggregates_slot_coverage_close_counts() -> None:
    stats = DecodeStats(
        slot_coverage_close_applications=3,
        slot_coverage_close_choice_changes=2,
    )
    summary = aggregate_stats([stats])
    assert summary["slot_coverage_close_applications_sum"] == 3.0
    assert summary["slot_coverage_close_choice_changes_sum"] == 2.0


def test_collect_engine_stats_folds_engine_counters() -> None:
    from slm_training.models.decode_stats import collect_engine_stats

    class _Engine:
        stats = {
            "full_syncs": 2,
            "incremental_advances": 5,
            "copy_probes": 3,
            "copy_probe_fallbacks": 1,
            "sync_ms": 1.5,
            "full_sync_fallbacks": 4,
            "full_prefix_lex_bytes": 128,
            "parser_forks": 7,  # engine-only key: not folded here
        }

    stats = DecodeStats()
    collect_engine_stats(_Engine(), stats)
    assert stats.dfa_full_syncs == 2
    assert stats.dfa_incremental_advances == 5
    assert stats.dfa_copy_probes == 3
    assert stats.dfa_copy_probe_fallbacks == 1
    assert stats.dfa_engine_sync_ms == 1.5
    assert stats.dfa_full_sync_fallbacks == 4
    assert stats.dfa_full_prefix_lex_bytes == 128

    summary = aggregate_stats([stats])
    assert summary["dfa_full_syncs_sum"] == 2.0
    assert summary["dfa_incremental_advances_mean"] == 5.0
    # Zero counters are omitted from the aggregate summary.
    empty = aggregate_stats([DecodeStats()])
    assert "dfa_full_syncs_sum" not in empty


def test_attributed_time_summary_flags_dark_cost() -> None:
    from slm_training.models.decode_stats import attributed_time_summary

    # 90% dark: only 100ms of 1000ms is covered by phase spans.
    dark = DecodeStats(total_ms=1000.0, denoiser_ms=60.0, pick_ms=40.0)
    out = attributed_time_summary([dark])
    assert out["attributed_ms_sum"] == 100.0
    assert out["unattributed_ms_sum"] == 900.0
    assert out["attributed_fraction"] == 0.1

    # Fully covered (overlap clamps at 1.0, never negative residual).
    covered = DecodeStats(
        total_ms=100.0, denoiser_ms=60.0, pick_ms=30.0, finalize_ms=30.0
    )
    out = attributed_time_summary([covered])
    assert out["attributed_fraction"] == 1.0
    assert out["unattributed_ms_sum"] == 0.0

    assert attributed_time_summary([]) == {}
    # Aggregate summary carries the coverage fields.
    agg = aggregate_stats([dark])
    assert agg["attributed_fraction"] == 0.1
