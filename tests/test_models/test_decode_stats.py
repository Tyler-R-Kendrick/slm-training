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


def test_decode_stats_tracks_constrained_dead_ends() -> None:
    stats = DecodeStats(constrained_dead_ends=2)
    assert stats.as_dict()["constrained_dead_ends"] == 2
    assert aggregate_stats([stats])["constrained_dead_ends_sum"] == 2.0


def test_decode_stats_aggregates_dead_end_traces() -> None:
    stats = DecodeStats(constrained_dead_end_traces=[{"position": 1}])
    assert aggregate_stats([stats])["constrained_dead_end_traces"] == [{"position": 1}]


def test_decode_stats_aggregates_bounded_selection_traces() -> None:
    stats = DecodeStats(constrained_selection_traces=[{"position": 2, "chosen_token": "="}])
    assert aggregate_stats([stats])["constrained_selection_traces"] == [
        {"position": 2, "chosen_token": "="}
    ]
