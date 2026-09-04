"""The continuous loop's append-only ledgers, tested without a training cycle.

These writers previously lived inside the 18k-line runner and could only be
reached through it. Extracted into ``scripts/autotrain_ledgers.py`` and
``scripts/autotrain_provenance.py``, the durability properties that matter --
append-only, tolerant of a corrupt line, deterministic paths -- are checked
directly.

Contract: ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import autotrain_ledgers as ledgers
from scripts import autotrain_provenance as provenance


# --- ledger paths ----------------------------------------------------------


@pytest.mark.parametrize(
    ("helper", "filename"),
    [
        (ledgers.dynamic_thrash_arms_path, "dynamic_thrash_arms.jsonl"),
        (ledgers.hillclimb_iteration_path, "hillclimb_iterations.jsonl"),
        (ledgers.interesting_residuals_path, "interesting_residuals.jsonl"),
        (ledgers.slug_stats_path, "slug_stats.json"),
    ],
)
def test_ledger_paths_are_loop_scoped(helper, filename: str, tmp_path: Path) -> None:
    """Every ledger lives under its own loop id, so two loops never interleave."""

    assert helper(tmp_path, "L1") == tmp_path / "loops" / "L1" / filename
    assert helper(tmp_path, "L1") != helper(tmp_path, "L2")


# --- append-only behaviour -------------------------------------------------


def test_hillclimb_append_is_additive_and_returns_the_whole_ledger(
    tmp_path: Path,
) -> None:
    first = ledgers.append_hillclimb_iteration(tmp_path, "L", {"cycle": 1})
    second = ledgers.append_hillclimb_iteration(tmp_path, "L", {"cycle": 2})
    assert [row["cycle"] for row in first] == [1]
    assert [row["cycle"] for row in second] == [1, 2]


def test_hillclimb_append_creates_missing_parents(tmp_path: Path) -> None:
    """The loop directory may not exist on the first cycle."""

    ledgers.append_hillclimb_iteration(tmp_path / "deep" / "er", "L", {"cycle": 1})
    assert (tmp_path / "deep" / "er" / "loops" / "L").is_dir()


def test_a_corrupt_line_does_not_lose_the_rest_of_the_ledger(tmp_path: Path) -> None:
    """A half-written row from a killed cycle must not poison later reads."""

    path = ledgers.hillclimb_iteration_path(tmp_path, "L")
    path.parent.mkdir(parents=True)
    path.write_text('{"cycle": 1}\nnot json at all\n', encoding="utf-8")
    rows = ledgers.append_hillclimb_iteration(tmp_path, "L", {"cycle": 2})
    assert [row["cycle"] for row in rows] == [1, 2]


def test_non_object_rows_are_skipped(tmp_path: Path) -> None:
    path = ledgers.hillclimb_iteration_path(tmp_path, "L")
    path.parent.mkdir(parents=True)
    path.write_text('[1, 2, 3]\n"a string"\n', encoding="utf-8")
    assert ledgers.append_hillclimb_iteration(tmp_path, "L", {"cycle": 9}) == [
        {"cycle": 9}
    ]


def test_historical_reclassification_appends_one_json_line_per_event(
    tmp_path: Path,
) -> None:
    ledgers.append_historical_reclassification(tmp_path, "L", {"a": 1})
    ledgers.append_historical_reclassification(tmp_path, "L", {"b": 2})
    path = tmp_path / "loops" / "L" / "historical_reclassification.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"a": 1}, {"b": 2}]


def test_residual_observations_of_an_absent_ledger_is_empty(tmp_path: Path) -> None:
    assert ledgers.load_residual_observations(tmp_path, "L") == []


# --- provenance ------------------------------------------------------------


def test_candidate_checkpoint_is_none_without_a_candidate(tmp_path: Path) -> None:
    assert provenance.checkpoint_path_for_candidate(tmp_path, "c", None) is None
    assert provenance.checkpoint_path_for_candidate(tmp_path, "c", "") is None


def test_candidate_checkpoint_is_none_until_the_file_exists(tmp_path: Path) -> None:
    """A path that merely could exist is not evidence a checkpoint was written."""

    assert provenance.checkpoint_path_for_candidate(tmp_path, "camp", "cand") is None


def test_candidate_checkpoint_resolves_once_written(tmp_path: Path) -> None:
    ckpt = tmp_path / "camp" / "runs" / "cand" / "checkpoints" / "last.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")
    assert provenance.checkpoint_path_for_candidate(tmp_path, "camp", "cand") == ckpt


def test_version_registry_location_is_the_canonical_one() -> None:
    """The registry path is a contract; drifting it silently skips no-bump notes."""

    assert provenance.VERSION_REGISTRY_REL == (
        "src/slm_training/resources/versions.json"
    )
