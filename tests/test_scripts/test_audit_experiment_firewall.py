"""Tests for the SLM-184 experiment firewall audit CLI."""

from __future__ import annotations

import json

from scripts.audit_experiment_firewall import main
from slm_training.harnesses.experiments.claim_manifest import (
    MATRIX_SET,
    MATRIX_VERSION,
    EXPERIMENT_ID,
    build_default_manifest,
    freeze_manifest,
)


def test_describe_prints_schema() -> None:
    assert main(["--mode", "describe"]) == 0


def test_fixture_writes_report(tmp_path) -> None:
    assert main(["--mode", "fixture", "--output-dir", str(tmp_path)]) == 0
    report_path = tmp_path / "slm184_claim_manifest_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["matrix_set"] == MATRIX_SET
    assert data["matrix_version"] == MATRIX_VERSION
    assert data["experiment_id"] == EXPERIMENT_ID
    assert data["first_confirmation"]["allowed"] is True
    assert data["second_confirmation"]["allowed"] is False
    assert data["dev_touch"]["allowed"] is True


def test_fixture_writes_design_docs(tmp_path) -> None:
    design_json = tmp_path / "iter-slm184-claim-manifest-20260720.json"
    design_md = tmp_path / "iter-slm184-claim-manifest-20260720.md"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--write-design-docs",
                "--design-json",
                str(design_json),
                "--design-md",
                str(design_md),
            ]
        )
        == 0
    )
    assert design_json.exists()
    assert design_md.exists()
    data = json.loads(design_json.read_text())
    assert data["status"] == "fixture"
    assert data["experiment_id"] == EXPERIMENT_ID


def test_check_dev_allowed(tmp_path) -> None:
    manifest = build_default_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    ledger_path = tmp_path / "ledger.json"
    assert (
        main(
            [
                "--mode",
                "check",
                "--manifest",
                str(manifest_path),
                "--ledger",
                str(ledger_path),
                "--suite-id",
                "smoke",
                "--suite-digest",
                "sha256:abc",
            ]
        )
        == 0
    )
    assert ledger_path.exists()
    ledger = json.loads(ledger_path.read_text())
    assert len(ledger["records"]) == 1
    assert ledger["records"][0]["touch_kind"] == "dev"


def test_check_confirmation_denied_without_frozen_manifest(tmp_path) -> None:
    manifest = build_default_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    ledger_path = tmp_path / "ledger.json"
    assert (
        main(
            [
                "--mode",
                "check",
                "--confirmation",
                "--manifest",
                str(manifest_path),
                "--ledger",
                str(ledger_path),
                "--suite-id",
                manifest.confirmation_suite_id,
                "--suite-digest",
                manifest.confirmation_suite_digest,
            ]
        )
        == 1
    )


def test_check_confirmation_allowed_once(tmp_path) -> None:
    manifest = build_default_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    freeze_manifest(manifest, tmp_path)
    ledger_path = tmp_path / "ledger.json"
    assert (
        main(
            [
                "--mode",
                "check",
                "--confirmation",
                "--manifest",
                str(manifest_path),
                "--ledger",
                str(ledger_path),
                "--suite-id",
                manifest.confirmation_suite_id,
                "--suite-digest",
                manifest.confirmation_suite_digest,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--mode",
                "check",
                "--confirmation",
                "--manifest",
                str(manifest_path),
                "--ledger",
                str(ledger_path),
                "--suite-id",
                manifest.confirmation_suite_id,
                "--suite-digest",
                manifest.confirmation_suite_digest,
            ]
        )
        == 1
    )


def test_audit_history_classifies_iters(tmp_path) -> None:
    iter_dir = tmp_path / "design"
    iter_dir.mkdir()
    clean = {
        "status": "confirmatory",
        "claim_class": "confirmatory",
        "suite_role": "confirmation",
        "confirmation_suite_id": "rico_held",
        "version_stamp": {"stamp_schema": "version_stamp/v1", "code_commit": "abc"},
    }
    dev = {"status": "dev", "claim_class": "wiring"}
    (iter_dir / "iter-clean.json").write_text(json.dumps(clean))
    (iter_dir / "iter-dev.json").write_text(json.dumps(dev))

    output = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"
    assert (
        main(
            [
                "--mode",
                "audit-history",
                "--iter-dir",
                str(iter_dir),
                "--output",
                str(output),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )
    data = json.loads(output.read_text())
    assert data["schema"] == "Slm184AuditHistoryV1"
    assert data["summary"]["clean_confirmation"] == 1
    assert data["summary"]["development_only"] == 1
    results = {r["path"]: r["classification"] for r in data["results"]}
    assert results["design/iter-clean.json"] == "clean_confirmation"
    assert results["design/iter-dev.json"] == "development_only"
    assert output_md.exists()
