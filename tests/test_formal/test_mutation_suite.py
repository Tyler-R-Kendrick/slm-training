"""EVID-11 / SLM-559: formal-evidence mutation matrix + red-team suite."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.formal.mutation_suite import (
    MATRIX_SCHEMA,
    REPORT_SCHEMA,
    assert_suite_passed,
    coverage_table,
    load_matrix,
    run_suite,
    semantic_authority_requires_distinct_trust_domains,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = (
    ROOT
    / "src/slm_training/resources/formal/evid11_mutation_matrix.v1.json"
)
RESULTS = ROOT / "docs/design/formal-evidence-mutation-results.json"


def test_matrix_covers_required_families() -> None:
    matrix = load_matrix(root=ROOT)
    assert matrix["schema"] == MATRIX_SCHEMA
    assert MATRIX.is_file()
    covered = {entry["family"] for entry in matrix["mutations"]}
    assert covered == set(matrix["required_families"])
    runners = {entry["runner"] for entry in matrix["mutations"]}
    assert len(runners) == len(matrix["mutations"])


def test_suite_rejects_every_mutation_and_keeps_positives() -> None:
    report = run_suite(root=ROOT)
    assert_suite_passed(report)
    assert report["schema"] == REPORT_SCHEMA
    assert report["passed"] is True
    assert report["structural_consistency_alone_confers_authority"] is False
    assert all(item["rejected"] and item["ok"] for item in report["mutations"])
    assert all(item["ok"] for item in report["positive_controls"])


def test_shared_trust_domain_gate() -> None:
    assert not semantic_authority_requires_distinct_trust_domains(
        ("python_structural", "python_reference")
    )
    assert semantic_authority_requires_distinct_trust_domains(
        ("python_structural", "lean_kernel")
    )


def test_coverage_table_rows_match_matrix() -> None:
    matrix = load_matrix(root=ROOT)
    rows = coverage_table(run_suite(root=ROOT))
    assert len(rows) == len(matrix["mutations"])
    assert {row["mutation"] for row in rows} == {
        entry["id"] for entry in matrix["mutations"]
    }


def test_committed_results_match_live_suite() -> None:
    report = run_suite(root=ROOT)
    assert_suite_passed(report)
    committed = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert committed["schema"] == REPORT_SCHEMA
    assert committed["passed"] is True
    assert committed["mutation_count"] == report["mutation_count"]
    assert [item["case_id"] for item in committed["mutations"]] == [
        item["case_id"] for item in report["mutations"]
    ]
    assert all(item["ok"] for item in committed["mutations"])
    assert all(item["ok"] for item in committed["positive_controls"])
