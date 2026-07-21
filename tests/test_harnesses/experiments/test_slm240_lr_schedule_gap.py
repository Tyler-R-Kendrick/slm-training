"""Tests for the SLM-240 learning-rate schedule gap probe harness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.experiments.slm240_lr_schedule_gap import (
    EXPERIMENT_ID,
    MATRIX_SET,
    LrScheduleGapReport,
    render_markdown,
    run_lr_schedule_gap_probe,
)


def test_probe_runs_every_optimizer_seed_combination() -> None:
    report = run_lr_schedule_gap_probe(
        steps=3, n_records=4, optimizers=("adamw", "muon_hybrid"), seeds=(0, 1)
    )

    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.claim_class == "wiring"
    assert report.status == "fixture"
    assert len(report.arms) == 4  # 2 optimizers x 2 seeds

    seen = {(a.optimizer_name, a.seed) for a in report.arms}
    assert seen == {
        ("adamw", 0),
        ("adamw", 1),
        ("muon_hybrid", 0),
        ("muon_hybrid", 1),
    }
    for arm in report.arms:
        assert arm.steps_recorded == 3
        assert arm.finite_throughout

    assert report.disposition in {
        "unstable",
        "config_mismatch",
        "schedule_detected",
        "gap_confirmed_telemetry_partial",
        "gap_confirmed",
    }
    assert report.version_stamp.get("stamp_schema") == "version_stamp/v1"


def test_adamw_arm_lr_is_constant_and_matches_config() -> None:
    report = run_lr_schedule_gap_probe(
        steps=5, n_records=4, optimizers=("adamw",), seeds=(0,)
    )
    arm = report.arms[0]
    assert arm.optimizer_name == "adamw"
    assert arm.lr_constant
    assert arm.lr_matches_config
    assert arm.configured_lrs == {"adamw": pytest.approx(3e-4)}
    assert all(entry["lr"] == pytest.approx(3e-4) for entry in arm.first_snapshot)


def test_muon_hybrid_arm_has_two_distinct_stable_groups() -> None:
    report = run_lr_schedule_gap_probe(
        steps=5, n_records=4, optimizers=("muon_hybrid",), seeds=(0,)
    )
    arm = report.arms[0]
    assert arm.optimizer_name == "muon_hybrid"
    assert arm.lr_constant
    assert arm.lr_matches_config
    families = {entry["optimizer"] for entry in arm.first_snapshot}
    assert families == {"muon", "adamw"}
    lrs_by_family = {entry["optimizer"]: entry["lr"] for entry in arm.first_snapshot}
    assert lrs_by_family["muon"] == pytest.approx(5e-4)
    assert lrs_by_family["adamw"] == pytest.approx(1e-4)


def test_metrics_jsonl_does_not_log_lr_field() -> None:
    # This is the second half of the SLM-240 hypothesis: even the per-step
    # telemetry omits the applied lr.
    report = run_lr_schedule_gap_probe(
        steps=3, n_records=4, optimizers=("adamw",), seeds=(0,)
    )
    arm = report.arms[0]
    assert arm.metrics_logs_lr is False
    assert report.any_metrics_log_lr is False


def test_report_roundtrips_through_dict() -> None:
    report = run_lr_schedule_gap_probe(
        steps=2, n_records=4, optimizers=("adamw",), seeds=(0,)
    )
    payload = report.to_dict()
    restored = LrScheduleGapReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_lr_schedule_gap_probe(
        steps=2, n_records=4, optimizers=("adamw", "muon_hybrid"), seeds=(0,)
    )
    text = render_markdown(report)
    assert report.disposition in text
    assert (
        "| optimizer | seed | steps recorded | configured lrs | lr constant? "
        "| lr matches config? | metrics logs lr? | finite? |"
        in text
    )
    assert "No-go for any 'schedule already works' claim" in text
