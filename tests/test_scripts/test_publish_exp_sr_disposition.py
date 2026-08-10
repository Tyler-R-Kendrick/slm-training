"""RSP-009 (SLM-490) regression tests: the EXP-SR cross-experiment
disposition publisher."""

from __future__ import annotations

from slm_training.harnesses.experiments.mechanism_disposition_report import (
    Disposition,
    disposition_record_integrity_ok,
)
from scripts.publish_exp_sr_disposition import MECHANISM_SPECS, build_records, main


def test_every_catalogue_family_has_exactly_one_disposed_mechanism():
    family_ids = [spec.family_id for spec in MECHANISM_SPECS]
    assert family_ids == [f"exp-sr-{i}" for i in range(1, 13)]
    assert len(set(family_ids)) == 12


def test_no_fixture_or_scratch_evidence_is_ever_adopted():
    # The schema's own validator already fails closed on this, but pin it
    # here too so a future edit to MECHANISM_SPECS cannot silently violate
    # SLM-490's own rule ("Fixture-only evidence cannot adopt").
    for record in build_records():
        if record.evidence_class in {"fixture", "scratch"}:
            assert record.disposition not in {
                Disposition.ADOPT_PRIMARY,
                Disposition.ADOPT_OPTIONAL,
            }


def test_every_record_carries_its_source_linear_issues():
    for record in build_records():
        assert record.issues, f"{record.mechanism_id} has no linked issues"


def test_records_are_individually_hash_verified():
    for record in build_records():
        assert disposition_record_integrity_ok(record)


def test_pysr_srbench_adapter_is_blocked_not_silently_omitted():
    records = {r.mechanism_id: r for r in build_records()}
    assert records["pysr_srbench_adapter"].disposition is Disposition.BLOCKED


def test_cli_writes_a_hash_verified_report_and_check_mode_passes(tmp_path):
    out_json = tmp_path / "disposition.json"
    out_md = tmp_path / "disposition.md"
    exit_code = main(["--out-json", str(out_json), "--out-md", str(out_md)])
    assert exit_code == 0
    assert "Mechanism disposition table" in out_md.read_text()

    check_exit_code = main(["--out-json", str(out_json), "--out-md", str(out_md), "--check"])
    assert check_exit_code == 0
