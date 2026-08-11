"""INTEG-06 / SLM-573: adversarial end-to-end acceptance matrix."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.formal.integ06_acceptance import (
    MATRIX_SCHEMA,
    REPORT_SCHEMA,
    assert_suite_passed,
    coverage_table,
    gate_coverage_by_scenario,
    load_matrix,
    run_suite,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = (
    ROOT
    / "src/slm_training/resources/formal/integ06_adversarial_matrix.v1.json"
)
RESULTS = ROOT / "docs/design/integ-06-adversarial-acceptance-results.json"


def test_matrix_covers_required_scenarios() -> None:
    matrix = load_matrix(root=ROOT)
    assert matrix["schema"] == MATRIX_SCHEMA
    assert MATRIX.is_file()
    covered = {entry["scenario"] for entry in matrix["adversarial_cases"]} | {
        entry["scenario"] for entry in matrix["positive_controls"]
    }
    assert covered == set(matrix["required_scenarios"])
    runners = {entry["runner"] for entry in matrix["adversarial_cases"]} | {
        entry["runner"] for entry in matrix["positive_controls"]
    }
    assert len(runners) == len(matrix["adversarial_cases"]) + len(
        matrix["positive_controls"]
    )


def test_suite_rejects_adversarial_and_keeps_positives() -> None:
    report = run_suite(root=ROOT)
    assert_suite_passed(report)
    assert report["schema"] == REPORT_SCHEMA
    assert report["passed"] is True
    assert report["no_failure_becomes_destructive_authority"] is True
    assert all(item["rejected"] and item["ok"] for item in report["adversarial_cases"])
    assert all(item["ok"] for item in report["positive_controls"])


def test_gate_coverage_table_rows() -> None:
    matrix = load_matrix(root=ROOT)
    report = run_suite(root=ROOT)
    rows = coverage_table(report)
    assert len(rows) == len(matrix["adversarial_cases"]) + len(
        matrix["positive_controls"]
    )
    by_scenario = gate_coverage_by_scenario(report)
    assert [row["scenario"] for row in by_scenario] == matrix["required_scenarios"]
    assert all(row["ok"] == "True" for row in by_scenario)
    assert all(row["adversarial_gate"] for row in by_scenario)


def test_committed_results_match_live_suite() -> None:
    report = run_suite(root=ROOT)
    assert_suite_passed(report)
    committed = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert committed["schema"] == REPORT_SCHEMA
    assert committed["passed"] is True
    assert committed["adversarial_count"] == report["adversarial_count"]
    assert [item["case_id"] for item in committed["adversarial_cases"]] == [
        item["case_id"] for item in report["adversarial_cases"]
    ]
    assert all(item["ok"] for item in committed["adversarial_cases"])
    assert all(item["ok"] for item in committed["positive_controls"])
