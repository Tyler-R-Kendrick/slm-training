"""VSS4-03 campaign harness tests (SLM-76).

Torch-free. Verifies that the campaign report round-trips, that CPU-runnable
phases execute and pass hard gates, and that every blocked frontier phase names
the exact missing artifact or capability.
"""

from __future__ import annotations

import json

from slm_training.harnesses.experiments.vss4_campaign import (
    describe_vss4_campaign,
    render_markdown,
    run_vss4_campaign,
)


def test_campaign_report_is_json_stable():
    report = describe_vss4_campaign()
    a = report.to_json()
    b = report.to_json()
    assert a == b
    payload = json.loads(a)
    assert payload["campaign_id"] == "vss4-03"
    assert "artifact_lock" in payload
    assert "phases" in payload
    assert "blocked_phases" in payload
    assert "honesty_note" in payload


def test_artifact_lock_records_required_artifacts():
    report = describe_vss4_campaign()
    lock = report.artifact_lock
    assert lock.source_commit
    assert lock.python_version
    assert "vss4_01_benchmark" in lock.required_artifacts
    assert "vss4_02_matrix" in lock.required_artifacts
    assert lock.required_artifacts["twotower_ranker_checkpoint"]["status"] == "missing"
    assert lock.required_artifacts["cost_to_go_energy_checkpoint"]["status"] == "missing"
    assert lock.required_artifacts["surface_ar_checkpoint"]["status"] == "missing"


def test_fixture_phases_run_and_pass_gates():
    report = run_vss4_campaign()
    by_phase = {p.phase: p for p in report.phases}
    assert by_phase["phase_0_artifact_lock"].status == "ran"
    assert by_phase["phase_1_correctness_reference"].status == "ran"
    assert by_phase["phase_5_matched_matrix"].status == "ran"

    p1 = by_phase["phase_1_correctness_reference"]
    assert p1.evidence["passed"] is True

    p5 = by_phase["phase_5_matched_matrix"]
    assert p5.evidence["passed"] is True
    assert p5.evidence["gate_failure_count"] == 0
    assert "R0" in p5.evidence["rows_ran"]
    assert "R1" in p5.evidence["rows_ran"]


def test_frontier_phases_are_blocked_with_reasons():
    report = run_vss4_campaign()
    by_phase = {p.phase: p for p in report.phases}
    blocked = {
        "phase_2_on_policy_supervision",
        "phase_3_energy_training",
        "phase_4_surface_training",
        "phase_6_adversarial_ood",
    }
    assert set(report.blocked_phases) == blocked
    for name in blocked:
        assert by_phase[name].status == "blocked"
        assert by_phase[name].blocked_reason
        assert "requires" in by_phase[name].blocked_reason.lower()


def test_frontier_matrix_rows_are_blocked_not_silently_ran():
    report = run_vss4_campaign()
    p5 = next(p for p in report.phases if p.phase == "phase_5_matched_matrix")
    for row in p5.evidence["rows"]:
        if row["row_id"] in ("R2", "R3", "R4", "R5", "R6"):
            assert row["capability_status"] in ("blocked", "not_run")
            assert row["blocked_reason"]


def test_describe_does_not_run_benchmarks():
    report = describe_vss4_campaign()
    by_phase = {p.phase: p for p in report.phases}
    assert by_phase["phase_1_correctness_reference"].status == "blocked"
    assert "--describe mode" in by_phase["phase_1_correctness_reference"].blocked_reason
    assert by_phase["phase_5_matched_matrix"].status == "blocked"
    assert "--describe mode" in by_phase["phase_5_matched_matrix"].blocked_reason


def test_run_id_is_stable_across_identical_runs():
    assert describe_vss4_campaign().run_id == describe_vss4_campaign().run_id


def test_markdown_renders_caveat_and_blocked_phases():
    report = describe_vss4_campaign()
    md = render_markdown(report)
    assert "VSS4-03" in md
    assert report.honesty_note in md
    assert "Blocked frontier scope" in md
    assert "phase_3_energy_training" in md
    assert "phase_4_surface_training" in md
