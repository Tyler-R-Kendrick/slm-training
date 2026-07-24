"""Tests for SLM-297 unified final-program re-score (scripts/run_slm297_unified_rescore)."""

from __future__ import annotations

import json

import pytest

from scripts.run_slm297_unified_rescore import build_payload
from slm_training.harnesses.experiments.slm155_factorization_comparison import (
    FINAL_PROGRAM_OUTCOME_SCHEMA,
    FinalProgramOutcomeV1,
    build_transition_table,
    old_outcome_label,
    paired_validity_table,
)

_OUTCOME_KEYS = {
    "record_id",
    "arm_id",
    "family",
    "seed",
    "program_source",
    "program_note",
    "parse_ok",
    "contract_ok",
    "v1_verdict",
    "v1_reason_codes",
    "structural_similarity",
    "abstain_reason",
    "process_metrics",
    "schema",
}

_PROCESS_KEYS = {
    "accepted",
    "recovered",
    "cost_edits",
    "cost_forwards",
    "cost_verifier_calls",
}


@pytest.fixture(scope="module")
def payload() -> dict:
    return build_payload()


def _outcome(
    record_id: str,
    arm_id: str,
    family: str,
    seed: int,
    verdict: bool,
    *,
    accepted: bool = False,
    recovered: bool = False,
) -> FinalProgramOutcomeV1:
    return FinalProgramOutcomeV1(
        record_id=record_id,
        arm_id=arm_id,
        family=family,
        seed=seed,
        program_source='root = Stack([n0], "column")\nn0 = TextContent(":x")',
        program_note=None,
        parse_ok=True,
        contract_ok=True,
        v1_verdict=verdict,
        v1_reason_codes=(),
        structural_similarity=None,
        abstain_reason=None,
        process_metrics={
            "accepted": accepted,
            "recovered": recovered,
            "cost_edits": 0,
            "cost_forwards": 1,
            "cost_verifier_calls": 0,
        },
    )


def test_mixed_semantics_guard(payload: dict) -> None:
    """Every record has a program_source and one commensurate schema."""
    outcomes = payload["outcomes"]
    assert outcomes, "expected re-scored outcomes"
    arm_ids = {o["arm_id"] for o in outcomes}
    # All 12 SLM-155 arms must be present.
    assert len(arm_ids) == 12
    for outcome in outcomes:
        assert outcome["schema"] == FINAL_PROGRAM_OUTCOME_SCHEMA
        assert set(outcome.keys()) == _OUTCOME_KEYS
        assert outcome["program_source"], f"no program retained for {outcome}"
        # `recovered` may appear only inside process_metrics, never as a
        # top-level semantic field.
        assert set(outcome["process_metrics"].keys()) == _PROCESS_KEYS
        assert "recovered" not in set(outcome.keys()) - {"process_metrics"}
        # New semantic verdict is the v1 final-program verdict.
        assert isinstance(outcome["v1_verdict"], bool)


def test_transition_table_math() -> None:
    outcomes = [
        _outcome("r0", "AR-G", "ar", 0, True, accepted=True),
        _outcome("r1", "AR-G", "ar", 0, False, accepted=True),
        _outcome("r2", "AR-G", "ar", 0, False, accepted=False),
        _outcome("r3", "AR-G", "ar", 1, True, accepted=False),
    ]
    table = build_transition_table(outcomes)
    assert table == {
        "accepted": {"valid": 1, "invalid": 1},
        "rejected": {"valid": 1, "invalid": 1},
    }
    x22 = [
        _outcome("r0", "X-P", "x22", 0, True, recovered=True),
        _outcome("r1", "X-P", "x22", 0, False, recovered=False),
    ]
    assert build_transition_table(x22) == {
        "recovered": {"valid": 1, "invalid": 0},
        "not_recovered": {"valid": 0, "invalid": 1},
    }


def test_old_outcome_label_uses_process_metrics_only() -> None:
    o = _outcome("r", "H-1", "hybrid", 0, True, recovered=True, accepted=False)
    assert old_outcome_label(o) == "recovered"
    o2 = _outcome("r", "AR-G", "ar", 0, False, accepted=True)
    assert old_outcome_label(o2) == "accepted"


def test_paired_invalid_over_valid_visibility() -> None:
    """AR final valid + hybrid final invalid on the same id+seed must surface."""
    ar = [
        _outcome("d0", "AR-G", "ar", 0, True),
        _outcome("d1", "AR-G", "ar", 0, False),
        _outcome("d2", "AR-G", "ar", 0, True),
    ]
    hybrid = [
        _outcome("d0", "H-1", "hybrid", 0, False),  # damage: hybrid regressed
        _outcome("d1", "H-1", "hybrid", 0, True),  # repair: hybrid fixed
        _outcome("d2", "H-1", "hybrid", 0, True),
    ]
    counts = paired_validity_table(ar, hybrid)
    assert counts["a_valid_b_invalid"] == 1
    assert counts["a_invalid_b_valid"] == 1
    assert counts["both_valid"] == 1
    assert counts["both_invalid"] == 0
    assert counts["unpaired"] == 0


def test_paired_tables_present_for_campaign(payload: dict) -> None:
    paired = payload["paired_ar_vs_hybrid"]
    assert paired, "expected AR-vs-hybrid paired tables"
    for counts in paired.values():
        assert set(counts) == {
            "both_valid",
            "a_valid_b_invalid",
            "a_invalid_b_valid",
            "both_invalid",
            "unpaired",
        }


def test_determinism() -> None:
    def _strip(payload: dict) -> dict:
        data = dict(payload)
        data.pop("timestamp", None)
        stamp = dict(data.get("version_stamp", {}))
        stamp.pop("stamped_at", None)
        data["version_stamp"] = stamp
        return data

    first = _strip(build_payload())
    second = _strip(build_payload())
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(
        second, sort_keys=True, default=str
    )
