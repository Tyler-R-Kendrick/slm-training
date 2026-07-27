"""Tests for SLM-296 (AP-013) verifier-filtered teacher-admission wiring harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm296_verified_teacher_admission import (
    TEACHER_MODEL_FAMILY,
    check_family_independence,
    run_verified_teacher_admission,
)


def test_check_family_independence_none_when_no_judge() -> None:
    assert check_family_independence(TEACHER_MODEL_FAMILY, None) is None


def test_check_family_independence_false_for_same_family() -> None:
    assert check_family_independence(TEACHER_MODEL_FAMILY, TEACHER_MODEL_FAMILY) is False


def test_check_family_independence_true_for_different_family() -> None:
    assert check_family_independence(TEACHER_MODEL_FAMILY, "other_family_v1") is True


def test_campaign_is_deterministic_and_self_hashed() -> None:
    first = run_verified_teacher_admission()
    second = run_verified_teacher_admission()
    assert first == second
    assert first["manifest_hash"]


def test_campaign_is_honest_wiring_only() -> None:
    report = run_verified_teacher_admission()
    assert report["claim_class"] == "wiring"
    assert report["status"] == "yield_limitation_no_real_teacher_available"
    assert report["limitations"]


def test_teacher_source_never_reaches_gold_or_silver() -> None:
    """Structural enforcement of 'teacher self-judgment alone cannot assign
    Gold/Silver': every admitted row's tier must be Bronze or worse."""
    report = run_verified_teacher_admission()
    for row in report["rows"]:
        if row["locked_set_overlap"]:
            continue
        assert row["tier"] in {"BRONZE", "QUARANTINE"}
    assert report["counts"]["accepted_gold_or_silver"] == 0


def test_same_family_shadow_judge_is_fully_quarantined() -> None:
    report = run_verified_teacher_admission()
    control = report["family_independence_control"]
    assert control["same_family_independent"] is False
    assert control["different_family_independent"] is True
    assert control["shadow_judge_fully_quarantined"] is True
    assert control["shadow_judge_quarantined"] == control["shadow_judge_sample_size"]


def test_locked_set_overlap_rows_are_excluded_from_acceptance() -> None:
    report = run_verified_teacher_admission()
    counts = report["counts"]
    overlap_rows = [row for row in report["rows"] if row["locked_set_overlap"]]
    assert len(overlap_rows) == counts["locked_set_overlap_rejected"]
    for row in overlap_rows:
        assert row["tier"] == "excluded_locked_set_overlap"
    non_overlap_accepted = sum(
        1
        for row in report["rows"]
        if not row["locked_set_overlap"] and row["tier"] != "QUARANTINE"
    )
    assert non_overlap_accepted == counts["accepted"]


def test_accepted_roots_never_collide_with_locked_holdout() -> None:
    report = run_verified_teacher_admission()
    accepted_roots = {
        row["canonical_root_id"]
        for row in report["rows"]
        if not row["locked_set_overlap"] and row["tier"] != "QUARANTINE"
    }
    overlap_roots = {
        row["canonical_root_id"] for row in report["rows"] if row["locked_set_overlap"]
    }
    assert accepted_roots.isdisjoint(overlap_roots)
