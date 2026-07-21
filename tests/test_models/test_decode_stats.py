from slm_training.models.decode_stats import DecodeStats, aggregate_stats
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
    )
    summary = aggregate_stats([stats])
    assert summary["choice_state_cache_hits_sum"] == 7.0
    assert summary["choice_state_cache_misses_sum"] == 2.0
    assert summary["choice_candidates_considered_sum"] == 31.0
    assert summary["choice_vocab_candidates_avoided_sum"] == 1200.0
    assert summary["choice_completion_cache_hits_sum"] == 29.0
    assert summary["choice_completion_cache_misses_sum"] == 3.0


def test_decode_stats_aggregates_root_reference_arity_counts() -> None:
    stats = DecodeStats(
        root_reference_arity_applications=7,
        root_reference_arity_choice_changes=2,
    )
    summary = aggregate_stats([stats])
    assert summary["root_reference_arity_applications_sum"] == 7.0
    assert summary["root_reference_arity_choice_changes_sum"] == 2.0


def test_decode_stats_aggregates_required_slot_margin_counts() -> None:
    """E627: root-cause instrumentation counters for E626's required_slot_margin_decode_weight."""
    stats = DecodeStats(
        required_slot_margin_applications=5,
        required_slot_margin_choice_changes=4,
    )
    summary = aggregate_stats([stats])
    assert summary["required_slot_margin_applications_sum"] == 5.0
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
