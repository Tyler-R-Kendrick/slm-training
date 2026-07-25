"""SLM-303: census / hash-pin / sweep-pairing tests (pure functions only)."""

from __future__ import annotations

import pytest

from scripts.run_slm303_decode_budget_audit import (
    SWEEP_CHECKPOINTS,
    build_sweep_arms,
    classify_scoreboard,
    pair_cells,
    sha_matches_committed,
)

ACTUAL = "653e71b286964373eda30ff7fb28a836bf0514c6bf78cac9fb07a12223f8fa3a"


def test_sha_fragment_prefix_suffix_match() -> None:
    assert sha_matches_committed("653e71b2…f8fa3a", ACTUAL) is True
    assert sha_matches_committed("`653e71b2…f8fa3a`", ACTUAL) is True
    assert sha_matches_committed("deadbeef…f8fa3a", ACTUAL) is False
    assert sha_matches_committed("653e71b2…0000", ACTUAL) is False


def test_sha_full_match_and_mismatch() -> None:
    assert sha_matches_committed(ACTUAL, ACTUAL) is True
    assert sha_matches_committed("0" * 64, ACTUAL) is False


def test_sha_uncommitted_is_unverifiable_not_pass() -> None:
    assert sha_matches_committed(None, ACTUAL) is None
    assert sha_matches_committed("", ACTUAL) is None
    assert sha_matches_committed("not-a-sha", ACTUAL) is None


def _scoreboard(**overrides):
    base = {
        "n": 3,
        "decode_timeout_count": 0,
        "fallback_count": 0,
        "decode_canvas_cap": 256,
        "parse_rate": 0.0,
        "details": [],
    }
    base.update(overrides)
    return base


def test_classify_scoreboard_unmeasured_when_fallback_null() -> None:
    result = classify_scoreboard(_scoreboard(fallback_count=None))
    assert result["classification"] == "unmeasured"


def test_classify_scoreboard_timeout_interference() -> None:
    result = classify_scoreboard(_scoreboard(decode_timeout_count=3))
    assert result["classification"] == "runtime_timeout_interference"


def test_classify_scoreboard_fallback_interference() -> None:
    result = classify_scoreboard(_scoreboard(fallback_count=2))
    assert result["classification"] == "fallback_interference"


def test_classify_scoreboard_clean_zero_is_model_behavior() -> None:
    result = classify_scoreboard(_scoreboard())
    assert result["classification"] == "model_behavior"


def test_classify_scoreboard_missing_fields_unmeasured() -> None:
    assert classify_scoreboard({"n": 3})["classification"] == "unmeasured"


def test_classify_scoreboard_row_outcomes_from_details() -> None:
    board = classify_scoreboard(
        _scoreboard(
            details=[
                {"id": "a", "parse_ok": True, "prediction": "(root)"},
                {"id": "b", "parse_ok": False, "prediction": "junk"},
                {"id": "c", "parse_ok": False, "prediction": ""},
            ]
        )
    )
    outcomes = {r["id"]: r["outcome"] for r in board["row_outcomes"]}
    assert outcomes == {
        "a": "model_valid",
        "b": "model_invalid",
        "c": "model_abstain",
    }


def test_sweep_arms_differ_only_in_budget() -> None:
    arms = build_sweep_arms(256)
    assert set(arms) == {"baseline", "budget10x"}
    assert arms["baseline"]["canvas_cap"] == 256
    assert arms["budget10x"]["canvas_cap"] == 2560
    assert arms["budget10x"]["timeout_s"] == 10 * arms["baseline"]["timeout_s"]


def _cell(checkpoint: str, arm: str, record: str, outcome: str, sha: str = "abc"):
    return {
        "checkpoint": checkpoint,
        "arm": arm,
        "record_id": record,
        "outcome": outcome,
        "budget": {"timeout_s": 10.0, "canvas_cap": 256},
        "checkpoint_sha256": sha,
        "record_sha256": "rec",
        "seed": 0,
        "elapsed_ms": 5.0,
        "parse_ok": None,
    }


def test_pair_cells_pairs_and_detects_flips() -> None:
    cells = [
        _cell("ltr2", "baseline", "smoke_hero_01", "runtime_timeout"),
        _cell("ltr2", "budget10x", "smoke_hero_01", "model_invalid"),
    ]
    (pair,) = pair_cells(cells)
    assert pair["status"] == "paired"
    assert pair["outcome_baseline"] == "runtime_timeout"
    assert pair["outcome_10x"] == "model_invalid"


def test_pair_cells_unpaired_and_violation() -> None:
    unpaired = pair_cells([_cell("ltr2", "baseline", "smoke_hero_01", "x")])
    assert unpaired[0]["status"] == "unpaired"
    violation = pair_cells(
        [
            _cell("ltr2", "baseline", "smoke_hero_01", "a", sha="abc"),
            _cell("ltr2", "budget10x", "smoke_hero_01", "b", sha="def"),
        ]
    )
    assert violation[0]["status"] == "pair_violation"
    assert "checkpoint_sha256" in violation[0]["pair_violation_fields"]


def test_pairing_is_deterministic() -> None:
    cells = [
        _cell("ltr2", "budget10x", "smoke_button_01", "model_invalid"),
        _cell("ltr2", "baseline", "smoke_hero_01", "runtime_timeout"),
        _cell("ltr2", "baseline", "smoke_button_01", "runtime_timeout"),
        _cell("ltr2", "budget10x", "smoke_hero_01", "model_invalid"),
    ]
    assert pair_cells(cells) == pair_cells(list(reversed(cells)))


def test_sweep_checkpoints_are_the_two_remediated() -> None:
    assert set(SWEEP_CHECKPOINTS) == {"ltr2", "lexer_ltr2"}
    for spec in SWEEP_CHECKPOINTS.values():
        assert spec["recorded_decode_timeout_count"] == 3
        assert spec["recorded_canvas_cap"] in (128, 256)


@pytest.mark.parametrize("cap", [128, 256])
def test_arms_track_recorded_cap(cap: int) -> None:
    arms = build_sweep_arms(cap)
    assert arms["baseline"]["canvas_cap"] == cap
    assert arms["budget10x"]["canvas_cap"] == cap * 10
