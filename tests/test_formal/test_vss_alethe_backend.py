"""RESEARCH-07 / SLM-565: VSS Alethe/SMT pilot tests (default-off)."""

from __future__ import annotations

from pathlib import Path

from slm_training.formal.vss_alethe_backend import (
    check_alethe_certificate,
    default_theory_suite,
    exhaustive_replay_theory,
    generate_alethe_certificate,
    incomplete_reconstruction_fixture,
    mutation_rejection_suite,
    paired_agreement_report,
    pinned_toolchain,
    supports_theory,
    unsupported_theory_fixture,
)
from slm_training.harnesses.experiments import research_07_vss_alethe as r07
from slm_training.research_preregistry import experiment_by_key


def test_toolchain_pin_stable() -> None:
    pin = pinned_toolchain()
    assert pin["toolchain_id"] == "hermetic_python_alethe_stub_v1"
    assert pin["certificate_format"] == "alethe_pilot"
    assert "QF_BOOL" in pin["supported_smt_theories"]
    assert "QF_UF" in pin["supported_smt_theories"]
    assert len(pin["encoder_hash"]) == 64


def test_exhaustive_and_alethe_agree_on_suite() -> None:
    for problem in default_theory_suite():
        control = exhaustive_replay_theory(problem)
        cold, encoded = generate_alethe_certificate(problem)
        assert control.outcome == "unsat"
        assert cold.outcome == "unsat"
        assert cold.certificate is not None and encoded is not None
        warm = check_alethe_certificate(problem, encoded, cold.certificate)
        assert warm.outcome == "unsat"


def test_mutations_rejected() -> None:
    problem = default_theory_suite()[-1]
    report = mutation_rejection_suite(problem)
    assert report["ok"] is True
    assert report["rejection_rate"] == 1.0


def test_incomplete_and_unsupported_stay_unknown() -> None:
    incomplete = incomplete_reconstruction_fixture()
    assert supports_theory(incomplete) == "supported"
    assert exhaustive_replay_theory(incomplete).outcome == "unsat"
    cold, _ = generate_alethe_certificate(incomplete)
    assert cold.outcome == "unknown"
    assert "incomplete" in cold.reason

    unsupported = unsupported_theory_fixture()
    assert supports_theory(unsupported) == "unsupported"
    cold2, _ = generate_alethe_certificate(unsupported)
    assert cold2.outcome == "unknown"


def test_paired_report_decision_contract() -> None:
    report = paired_agreement_report(default_theory_suite())
    assert report["witness_disagreement_count"] == 0
    assert report["mutation_rejection_rate"] == 1.0
    assert report["reconstruction_failure_as_refutation_count"] == 0
    assert report["exact_agreement_rate_on_supported_theory_subset"] == 1.0
    assert report["decision"] == "accept"
    assert report["sat_overlap_agreement_rate"] == 1.0


def test_default_off_skips_without_enable(tmp_path: Path) -> None:
    result = r07.run_experiment(root=tmp_path, enabled=False)
    assert result["executed"] is False
    assert result["decision"] == "skipped_default_off"


def test_campaign_lock_roundtrip(tmp_path: Path) -> None:
    lock = r07.write_campaign_lock(tmp_path / "lock.json", root=tmp_path)
    restored = r07.load_campaign_lock(tmp_path / "lock.json")
    assert restored.manifest_sha256 == lock.manifest_sha256
    assert restored.verify_digest().manifest_sha256 == lock.manifest_sha256


def test_preregistry_row_points_at_evidence() -> None:
    row = experiment_by_key("RESEARCH-07")
    assert row["linear_id"] == "SLM-565"
    assert row["default_off"] is True
    assert row["evidence_path"] == r07.EVIDENCE_MD
    assert row["primary_metric"] == r07.PRIMARY_METRIC
    assert row["knobs"]["pilot"] == "vss_alethe_smt"
