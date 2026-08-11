"""INTEG-10 / SLM-576: bounded infrastructure release matrix."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.formal.integ10_release_matrix import (
    MATRIX_SCHEMA,
    REPORT_SCHEMA,
    assert_suite_passed,
    content_address,
    load_matrix,
    run_matrix,
    stable_report,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "src/slm_training/resources/formal/integ10_release_matrix.v1.json"
RESULTS = ROOT / "docs/design/integ-10-bounded-release-matrix-results.json"


def test_matrix_covers_required_categories() -> None:
    matrix = load_matrix(root=ROOT)
    assert matrix["schema"] == MATRIX_SCHEMA
    assert MATRIX.is_file()
    categories = {gate["category"] for gate in matrix["gates"]}
    assert "required" in categories
    assert "optional_tool" in categories
    assert "research_default_off" in categories
    ids = [gate["id"] for gate in matrix["gates"]]
    assert len(ids) == len(set(ids))
    assert "integ06_adversarial_acceptance" in ids
    assert "integ07_activation_preflight_recall" in ids
    assert "integ08_dominance_evidence" in ids
    assert "research_default_off_status" in ids


def test_check_mode_passes_and_emits_content_address() -> None:
    report = run_matrix(root=ROOT, execute=False)
    assert_suite_passed(report)
    assert report["schema"] == REPORT_SCHEMA
    assert report["passed"] is True
    assert report["execute"] is False
    assert report["evidence_content_address"].startswith("cas://sha256/")
    assert report["matrix_content_address"].startswith("cas://sha256/")
    assert report["no_optional_skip_is_success"] is True
    assert report["no_optional_skip_is_refutation"] is True
    assert report["infrastructure_release_needs_research_pilot"] is False
    assert report["empirical_performance_separate"] is True

    by_id = {row["gate_id"]: row for row in report["gates"]}
    assert by_id["lean_contracts_axioms"]["status"] == "delegated"
    assert by_id["integ06_adversarial_acceptance"]["status"] == "delegated"
    assert by_id["research_default_off_status"]["status"] == "research_default_off"
    optional = by_id["optional_lean_lake_probe"]
    assert optional["status"] in {
        "skipped_optional",
        "skipped_optional_timeout",
        "failed",
        "ok",
    }
    # Optional outcomes never count toward release.passed via required_failed.
    assert "optional_lean_lake_probe" not in report["required_failed"]
    for item in report["optional_status"]:
        assert item["counted_as_success"] is False
        assert item["counted_as_refutation"] is False


def test_gate_identity_table_links_every_gate() -> None:
    report = run_matrix(root=ROOT, execute=False)
    matrix = load_matrix(root=ROOT)
    table = report["gate_identity_table"]
    assert len(table) == len(matrix["gates"])
    for row in table:
        assert row["content_address"] == content_address(
            {
                "gate_id": row["gate_id"],
                "identities": row["identities"],
                "module": row["module"],
                "owner": row["owner"],
            }
        )


def test_committed_results_match_live_check_mode() -> None:
    report = run_matrix(root=ROOT, execute=False)
    assert_suite_passed(report)
    committed = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert committed["schema"] == REPORT_SCHEMA
    assert stable_report(committed) == stable_report(report)
