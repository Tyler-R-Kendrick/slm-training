"""Derived read-only decode ratios: undefined denominators stay ``None``."""

from __future__ import annotations

import pytest

from slm_training.models.decode_stats import (
    DERIVED_DECODE_RATIO_FIELDS,
    DecodeStats,
    aggregate_stats,
    proves_zero_neural_work,
)


def test_zero_denominators_report_none_never_zero() -> None:
    stats = DecodeStats()
    assert stats.committed_tokens == 0
    assert stats.forced_token_fraction is None
    assert stats.speculative_commit_fraction is None
    assert stats.speculative_token_fraction is None
    assert stats.forwards_per_committed_token is None
    payload = stats.as_dict()
    for name in DERIVED_DECODE_RATIO_FIELDS:
        assert name in payload
    assert payload["committed_tokens"] == 0
    for name in DERIVED_DECODE_RATIO_FIELDS[1:]:
        assert payload[name] is None, name


def test_forwards_with_no_commit_is_still_undefined() -> None:
    # A forward that committed nothing has no per-token cost: None, not inf/0.
    stats = DecodeStats(forwards_count=3)
    assert stats.forwards_per_committed_token is None
    assert stats.forced_token_fraction is None


def test_synthetic_counts_give_exact_ratios() -> None:
    stats = DecodeStats(
        tokens_emitted=40,
        forced_tokens=10,
        forwards_count=8,
        speculative_rank_evaluations=4,
        speculative_rank_commits=3,
        speculative_rank_tokens=6,
    )
    assert stats.committed_tokens == 40
    assert stats.forced_token_fraction == pytest.approx(0.25)
    assert stats.speculative_commit_fraction == pytest.approx(0.75)
    assert stats.speculative_token_fraction == pytest.approx(0.15)
    assert stats.forwards_per_committed_token == pytest.approx(0.2)
    payload = stats.as_dict()
    assert payload["committed_tokens"] == 40
    assert payload["forced_token_fraction"] == pytest.approx(0.25)
    assert payload["speculative_commit_fraction"] == pytest.approx(0.75)
    assert payload["speculative_token_fraction"] == pytest.approx(0.15)
    assert payload["forwards_per_committed_token"] == pytest.approx(0.2)


def test_full_deterministic_bypass_reads_as_zero_forwards_per_token() -> None:
    # I2: every token forced, no forward -- the ratio is a genuine 0.0, and the
    # canonical bypass proof still holds.
    stats = DecodeStats(tokens_emitted=12, forced_tokens=12)
    assert proves_zero_neural_work(stats)
    assert stats.forwards_per_committed_token == 0.0
    assert stats.forced_token_fraction == 1.0


def test_ratios_are_read_only_and_not_dataclass_fields() -> None:
    stats = DecodeStats(tokens_emitted=2, forced_tokens=1)
    for name in DERIVED_DECODE_RATIO_FIELDS:
        with pytest.raises(AttributeError):
            setattr(stats, name, 0.5)
    field_names = {item.name for item in stats.__dataclass_fields__.values()}
    assert not field_names & set(DERIVED_DECODE_RATIO_FIELDS)


def test_round_trip_drops_derived_keys_and_recomputes() -> None:
    stats = DecodeStats(tokens_emitted=4, forced_tokens=1, forwards_count=2)
    payload = stats.as_dict()
    payload["forced_token_fraction"] = 0.99  # a tampered derived value
    restored = DecodeStats.from_dict(payload)
    assert restored.tokens_emitted == 4
    assert restored.forced_token_fraction == pytest.approx(0.25)
    assert restored.forwards_per_committed_token == pytest.approx(0.5)


def test_merge_sums_raw_counters_and_recomputes_ratios() -> None:
    total = DecodeStats(tokens_emitted=10, forced_tokens=5, forwards_count=1)
    total.merge(DecodeStats(tokens_emitted=10, forced_tokens=0, forwards_count=3))
    assert total.committed_tokens == 20
    assert total.forced_token_fraction == pytest.approx(0.25)
    assert total.forwards_per_committed_token == pytest.approx(0.2)
    # Merging an empty row must not poison a defined ratio with None.
    total.merge(DecodeStats())
    assert total.forced_token_fraction == pytest.approx(0.25)


def test_aggregate_stats_unaffected_by_derived_keys() -> None:
    rows = [DecodeStats(tokens_emitted=4, forced_tokens=2), DecodeStats()]
    summary = aggregate_stats(rows)
    assert summary["tokens_emitted_sum"] == 4.0
    assert summary["forced_tokens_sum"] == 2.0
    for name in DERIVED_DECODE_RATIO_FIELDS:
        assert f"{name}_sum" not in summary
