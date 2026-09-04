"""Atomic unit tests for the paired-power evidence ledger and lease predicate.

These four functions in ``heal/fail_closed.py`` had no tests at all, which is
how a 61%-covered module hides its most consequential gap. They are not
peripheral:

* ``append_power_deltas`` / ``read_power_evidence`` / ``power_evidence_summary``
  accumulate the paired deltas whose SD the screening decision reports as
  ``unmeasured`` today. Until that SD exists every screen returns
  ``insufficient_evidence``, so a silent defect here keeps the loop advisory
  forever -- and it would be invisible, because "no measured SD yet" and "the
  accumulator is broken" look identical from the outside.
* ``lease_covers`` decides whether a dirty path is authorized. It is a
  fail-closed predicate: an expired lease, an unreadable one, or one that
  covers a *sibling* path must all answer no.

One behaviour per test, no shared state, no fixtures beyond ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.autoresearch.heal.fail_closed import (
    append_power_deltas,
    lease_covers,
    power_evidence_summary,
    read_power_evidence,
)

# ---------------------------------------------------------------------------
# append_power_deltas
# ---------------------------------------------------------------------------


def test_append_returns_the_total_count_not_the_appended_count(tmp_path: Path) -> None:
    """The return value is what a postcondition compares, so it must be the total."""
    path = tmp_path / "power.jsonl"

    assert (
        append_power_deltas(path, cycle="c1", metric="smoke.eval_nll", deltas=[0.1])
        == 1
    )
    assert (
        append_power_deltas(
            path, cycle="c2", metric="smoke.eval_nll", deltas=[0.2, 0.3]
        )
        == 3
    )


def test_append_creates_missing_parents(tmp_path: Path) -> None:
    path = tmp_path / "loops" / "loop-1" / "power.jsonl"

    assert append_power_deltas(path, cycle="c1", metric="m", deltas=[0.0]) == 1
    assert path.is_file()


def test_append_defaults_costs_to_zero_per_delta(tmp_path: Path) -> None:
    path = tmp_path / "power.jsonl"
    append_power_deltas(path, cycle="c1", metric="m", deltas=[0.1, 0.2])

    rows = read_power_evidence(path)
    assert [row["wall_seconds"] for row in rows] == [0.0, 0.0]


def test_append_pairs_each_cost_with_its_own_delta(tmp_path: Path) -> None:
    path = tmp_path / "power.jsonl"
    append_power_deltas(
        path, cycle="c1", metric="m", deltas=[0.1, 0.2], costs=[5.0, 7.0]
    )

    rows = read_power_evidence(path)
    assert [(row["delta"], row["wall_seconds"]) for row in rows] == [
        (0.1, 5.0),
        (0.2, 7.0),
    ]


def test_append_refuses_mismatched_costs_rather_than_truncating(tmp_path: Path) -> None:
    """``zip`` would silently drop the unpaired tail; the ledger must refuse."""
    path = tmp_path / "power.jsonl"

    with pytest.raises(ValueError):
        append_power_deltas(
            path, cycle="c1", metric="m", deltas=[0.1, 0.2], costs=[5.0]
        )


def test_append_of_nothing_leaves_the_count_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "power.jsonl"
    append_power_deltas(path, cycle="c1", metric="m", deltas=[0.1])

    assert append_power_deltas(path, cycle="c2", metric="m", deltas=[]) == 1


# ---------------------------------------------------------------------------
# read_power_evidence
# ---------------------------------------------------------------------------


def test_reading_a_missing_ledger_is_empty_not_an_error(tmp_path: Path) -> None:
    assert read_power_evidence(tmp_path / "absent.jsonl") == []


def test_a_malformed_row_is_skipped_and_the_rest_survive(tmp_path: Path) -> None:
    """An append interrupted mid-line must not discard the whole ledger."""
    path = tmp_path / "power.jsonl"
    path.write_text(
        '{"cycle": "c1", "metric": "m", "delta": 0.1}\n'
        "{not json\n"
        '{"cycle": "c2", "metric": "m", "delta": 0.2}\n',
        encoding="utf-8",
    )

    assert [row["delta"] for row in read_power_evidence(path)] == [0.1, 0.2]


def test_a_row_without_a_numeric_delta_is_not_evidence(tmp_path: Path) -> None:
    path = tmp_path / "power.jsonl"
    path.write_text(
        '{"cycle": "c1", "delta": "0.1"}\n'
        '{"cycle": "c2"}\n'
        '{"cycle": "c3", "delta": 0.3}\n',
        encoding="utf-8",
    )

    assert [row["delta"] for row in read_power_evidence(path)] == [0.3]


# ---------------------------------------------------------------------------
# power_evidence_summary
# ---------------------------------------------------------------------------


def test_an_empty_ledger_reports_no_sd_rather_than_zero(tmp_path: Path) -> None:
    """A fabricated 0.0 SD would read as a perfectly precise measurement."""
    summary = power_evidence_summary(tmp_path / "absent.jsonl")

    assert summary["paired_deltas"] == 0
    assert summary["sd"] is None
    assert summary["mean"] is None
    assert summary["ready"] is False


def test_a_single_observation_has_a_mean_but_no_sd(tmp_path: Path) -> None:
    path = tmp_path / "power.jsonl"
    append_power_deltas(path, cycle="c1", metric="m", deltas=[0.4])

    summary = power_evidence_summary(path)
    assert summary["mean"] == pytest.approx(0.4)
    assert summary["sd"] is None


def test_readiness_needs_both_enough_pairs_and_enough_cycles(tmp_path: Path) -> None:
    """Pairs from one cycle are not independent evidence, however many there are."""
    path = tmp_path / "power.jsonl"
    append_power_deltas(
        path, cycle="c1", metric="m", deltas=[0.1 * index for index in range(20)]
    )

    assert power_evidence_summary(path, min_pairs=10, min_cycles=2)["ready"] is False

    append_power_deltas(path, cycle="c2", metric="m", deltas=[0.5])
    assert power_evidence_summary(path, min_pairs=10, min_cycles=2)["ready"] is True


def test_the_summary_counts_distinct_cycles_not_rows(tmp_path: Path) -> None:
    path = tmp_path / "power.jsonl"
    for _ in range(3):
        append_power_deltas(path, cycle="c1", metric="m", deltas=[0.1])

    summary = power_evidence_summary(path)
    assert summary["paired_deltas"] == 3
    assert summary["cycles"] == 1


def test_mean_absolute_cost_averages_over_rows(tmp_path: Path) -> None:
    path = tmp_path / "power.jsonl"
    append_power_deltas(
        path, cycle="c1", metric="m", deltas=[0.1, 0.2], costs=[4.0, 6.0]
    )

    assert power_evidence_summary(path)["mean_abs_cost_seconds"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# lease_covers
# ---------------------------------------------------------------------------


def _lease(
    path: Path, *, expires_at: float, prefixes: list[str], key: str = "path_prefixes"
) -> Path:
    path.write_text(
        json.dumps({"expires_at": expires_at, key: prefixes}), encoding="utf-8"
    )
    return path


def test_a_missing_lease_authorizes_nothing(tmp_path: Path) -> None:
    assert lease_covers(tmp_path / "absent.json", "outputs/x", now=0.0) is False


def test_an_unparseable_lease_authorizes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    path.write_text("{ truncated", encoding="utf-8")

    assert lease_covers(path, "outputs/x", now=0.0) is False


def test_an_expired_lease_authorizes_nothing(tmp_path: Path) -> None:
    path = _lease(tmp_path / "lease.json", expires_at=100.0, prefixes=["outputs/"])

    assert lease_covers(path, "outputs/x", now=101.0) is False


def test_a_lease_expiring_exactly_now_is_already_expired(tmp_path: Path) -> None:
    """The boundary is closed against the lease: equal is expired."""
    path = _lease(tmp_path / "lease.json", expires_at=100.0, prefixes=["outputs/"])

    assert lease_covers(path, "outputs/x", now=100.0) is False
    assert lease_covers(path, "outputs/x", now=99.9) is True


def test_a_live_lease_covers_paths_under_its_prefix(tmp_path: Path) -> None:
    path = _lease(tmp_path / "lease.json", expires_at=100.0, prefixes=["outputs/runs"])

    assert lease_covers(path, "outputs/runs/a/b.json", now=0.0) is True


def test_a_prefix_covers_itself_exactly(tmp_path: Path) -> None:
    path = _lease(tmp_path / "lease.json", expires_at=100.0, prefixes=["outputs/runs"])

    assert lease_covers(path, "outputs/runs", now=0.0) is True


def test_a_prefix_never_covers_a_sibling_that_merely_shares_its_text(
    tmp_path: Path,
) -> None:
    """``outputs/runs`` must not authorize ``outputs/runs-backup``.

    A bare ``startswith`` would; the separator is what makes the predicate a
    path-prefix test rather than a string-prefix test.
    """
    path = _lease(tmp_path / "lease.json", expires_at=100.0, prefixes=["outputs/runs"])

    assert lease_covers(path, "outputs/runs-backup/x", now=0.0) is False


def test_a_windows_style_dirty_path_is_normalized_before_matching(
    tmp_path: Path,
) -> None:
    path = _lease(tmp_path / "lease.json", expires_at=100.0, prefixes=["outputs/runs"])

    assert lease_covers(path, "outputs\\runs\\a.json", now=0.0) is True


def test_the_legacy_paths_key_is_still_honored(tmp_path: Path) -> None:
    path = _lease(
        tmp_path / "lease.json",
        expires_at=100.0,
        prefixes=["outputs/runs"],
        key="paths",
    )

    assert lease_covers(path, "outputs/runs/a.json", now=0.0) is True


def test_a_lease_with_no_prefixes_covers_nothing(tmp_path: Path) -> None:
    path = _lease(tmp_path / "lease.json", expires_at=100.0, prefixes=[])

    assert lease_covers(path, "outputs/runs/a.json", now=0.0) is False
