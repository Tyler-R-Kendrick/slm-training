"""SLM-286 deterministic ship-gate evidence census tests."""

from __future__ import annotations

import pytest

from slm_training.harnesses.model_build.evidence_census import (
    _comparison_overlap,
    append_adjudications,
    extract_scoreboards,
    verify_adjudication_chain,
)


def test_extract_scoreboards_accepts_only_root_and_direct_results() -> None:
    suite = {"smoke": {"n": 20, "meaningful_program_rate": 1.0}}
    payload = {
        "suites": suite,
        "results": [
            {"suites": suite},
            {"nested": {"suites": suite}},
        ],
        "nested": {"suites": suite},
    }
    rows = extract_scoreboards(payload)
    assert [(pointer, suites) for pointer, suites, _ in rows] == [
        ("/suites", suite),
        ("/results/0/suites", suite),
    ]


def test_extract_scoreboards_preserves_missing_n_for_integrity_replay() -> None:
    rows = extract_scoreboards({"suites": {"smoke": {"n": None}}})
    assert rows[0][1]["smoke"]["n"] is None
    assert extract_scoreboards({"suites": {"unknown": {"n": 20}}}) == []


def test_extract_scoreboards_uses_canonical_nested_and_legacy_fallbacks() -> None:
    suite = {"smoke": {"n": 20, "meaningful_program_rate": 1.0}}
    nested = extract_scoreboards({"honest_evaluation": {"suites": suite}})
    assert nested[0][0] == "/honest_evaluation/suites"
    legacy = extract_scoreboards(
        {
            "suite": "smoke",
            "metrics": {
                "n": 20,
                "meaningful_program_rate": 1.0,
                "structural_similarity": 0.8,
            },
        }
    )
    assert legacy[0][0] == "/metrics"
    assert set(legacy[0][1]) == {"smoke"}


def _candidate(path: str, file_sha: str = "a", pointer: str = "/suites") -> dict:
    return {
        "source": {
            "path": path,
            "file_sha256": file_sha,
            "json_pointer": pointer,
            "scoreboard_sha256": file_sha,
        },
        "gate_replay_sha256": file_sha,
        "adjudicated_verdict": "inconclusive_until_powered",
    }


def test_append_adjudications_preserves_prefix_and_is_idempotent() -> None:
    first = append_adjudications([_candidate("b")])
    repeated = append_adjudications([_candidate("b")], first)
    extended = append_adjudications([_candidate("a"), _candidate("b")], first)
    assert repeated == first
    assert extended[: len(first)] == first
    assert extended[-1]["source"]["path"] == "a"
    verify_adjudication_chain(extended)


def test_append_adjudications_supersedes_changed_slot() -> None:
    first = append_adjudications([_candidate("a", "old")])
    second = append_adjudications([_candidate("a", "new")], first)
    assert second[-1]["supersedes_event_id"] == first[0]["event_id"]


def test_append_adjudications_normalizes_legacy_slot_pointers() -> None:
    first = append_adjudications([_candidate("a", "old", "")])
    second = append_adjudications([_candidate("a", "new", "/suites")], first)
    assert second[-1]["supersedes_event_id"] == first[0]["event_id"]

    result_first = append_adjudications([_candidate("b", "old", "/results/0")])
    result_second = append_adjudications(
        [_candidate("b", "new", "/results/0/suites")], result_first
    )
    assert result_second[-1]["supersedes_event_id"] == result_first[0]["event_id"]


def test_verify_adjudication_chain_rejects_tampering() -> None:
    ledger = append_adjudications([_candidate("a")])
    ledger[0]["adjudicated_verdict"] = "supported_negative"
    with pytest.raises(ValueError, match="event hash"):
        verify_adjudication_chain(ledger)


def test_comparison_overlap_requires_exact_count_provenance() -> None:
    def arm(low: float, high: float) -> dict:
        return {
            "rate_evidence": {
                "meaningful_program_rate": {
                    "numerator": 10,
                    "denominator": 20,
                    "interval": {"low": low, "high": high},
                }
            }
        }

    assert _comparison_overlap({"control": arm(0.2, 0.6), "candidate": arm(0.5, 0.8)})
    assert not _comparison_overlap(
        {"control": arm(0.2, 0.4), "candidate": arm(0.5, 0.8)}
    )
    assert _comparison_overlap({"control": {}, "candidate": {}}) is None
