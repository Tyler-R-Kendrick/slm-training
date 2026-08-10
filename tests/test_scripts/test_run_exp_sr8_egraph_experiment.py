"""RSP-004 (SLM-479) regression tests: the EXP-SR-8 e-graph experiment
runner script."""

from __future__ import annotations

import json

from scripts.run_exp_sr8_egraph_experiment import main, run_experiment


def _strip_timing(report: dict) -> dict:
    stripped = {k: v for k, v in report.items() if k != "version_stamp"}
    stripped["control_arm"] = {k: v for k, v in stripped["control_arm"].items() if k != "wall_ms"}
    stripped["candidate_arm"] = {k: v for k, v in stripped["candidate_arm"].items() if k != "wall_ms"}
    stripped["comparison"] = {
        k: v for k, v in stripped["comparison"].items() if k not in {"wall_ms_overhead"}
    }
    return stripped


def test_run_experiment_is_deterministic():
    report_a = run_experiment()
    report_b = run_experiment()
    # wall_ms is real wall-clock timing, not a deterministic output -- every
    # other field (unique forms, e-graph shape, verification, gates,
    # decision) must be byte-identical across runs.
    assert _strip_timing(report_a) == _strip_timing(report_b)


def test_run_experiment_reports_honest_matched_comparison():
    report = run_experiment()
    assert report["family_id"] == "exp-sr-8"
    assert report["claim_class"] == "fixture"
    assert report["fixture"]["n_raw_forms"] > 0
    assert report["control_arm"]["n_unique_forms"] == report["candidate_arm"]["n_unique_forms"]
    assert report["comparison"]["unique_forms_match"] is True
    assert report["comparison"]["pareto_frontier_matches"] is True


def test_every_rewrite_application_is_certified_and_reverified():
    report = run_experiment()
    assert report["verification"]["n_applications"] > 0
    assert report["verification"]["n_passed"] == report["verification"]["n_applications"]
    assert report["verification"]["egraph_certified_equivalence_rate"] == 1.0
    assert report["verification"]["failed_rule_ids"] == []


def test_kill_gate_does_not_fire_on_a_fully_certified_run():
    report = run_experiment()
    kill_gate = next(g for g in report["gates"] if g["gate_id"].endswith("_kill"))
    assert kill_gate["fired"] is False
    assert report["killed"] is False


def test_adoption_decision_stays_experimental_regardless_of_fixture_outcome():
    report = run_experiment()
    assert report["adoption_decision"] in {"not_adopted_stays_experimental", "killed_by_rollback_gate"}
    assert "default-off" in report["adoption_rationale"]


def test_cli_writes_durable_report(tmp_path):
    out_json = tmp_path / "exp-sr8.json"
    out_md = tmp_path / "exp-sr8.md"
    exit_code = main(["--out-json", str(out_json), "--out-md", str(out_md)])
    assert exit_code == 0
    report = json.loads(out_json.read_text())
    assert report["family_id"] == "exp-sr-8"
    assert "version_stamp" in report
    assert "Gates" in out_md.read_text()
