"""RSP-005 (SLM-480) regression tests: the EXP-SR-9 macro-library
experiment runner script."""

from __future__ import annotations

import json

from scripts.run_exp_sr9_macro_library_experiment import main, run_experiment


def _strip_nondeterministic(report: dict) -> dict:
    return {k: v for k, v in report.items() if k not in {"manifest_sha256", "catalogue_manifest_sha256"}}


def test_run_experiment_is_deterministic():
    report_a = run_experiment()
    report_b = run_experiment()
    assert _strip_nondeterministic(report_a) == _strip_nondeterministic(report_b)


def test_run_experiment_reports_honest_fixture_evidence():
    report = run_experiment()
    assert report["catalogue_id"] == "exp-sr-9"
    assert report["claim_class"] == "fixture"
    assert report["promotion"] is False
    assert report["n_train_records"] > 0
    assert report["n_test_records"] > 0
    assert set(report["arms"]) == {"no_macros", "frequency_macros", "mdl_macros"}
    # The no-macros control arm must always show exactly zero reduction.
    assert report["arms"]["no_macros"]["reduction_rate"] == 0.0


def test_leakage_audit_is_clean_and_expansion_equivalence_holds():
    report = run_experiment()
    assert report["leakage_audit"]["is_clean"] is True
    assert report["expansion_equivalence_all_verified"] is True


def test_cli_writes_durable_report(tmp_path):
    out_json = tmp_path / "exp-sr9.json"
    out_md = tmp_path / "exp-sr9.md"
    exit_code = main(["--out-json", str(out_json), "--out-md", str(out_md)])
    assert exit_code == 0
    report = json.loads(out_json.read_text())
    assert report["catalogue_id"] == "exp-sr-9"
    assert "Arms" in out_md.read_text()
