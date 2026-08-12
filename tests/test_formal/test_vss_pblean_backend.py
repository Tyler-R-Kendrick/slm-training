"""RESEARCH-06 / SLM-564: VSS PBLean PB pilot tests (default-off)."""

from __future__ import annotations

from pathlib import Path

from slm_training.formal.encoding_adapter import BoundedProblemV1, LiteralV1
from slm_training.formal.vss_pblean_backend import (
    check_pblean_certificate,
    default_unsat_suite,
    encode_cnf_to_pb,
    exhaustive_replay,
    generate_pblean_certificate,
    mutation_rejection_suite,
    pad_unit_conflict_problem,
    paired_warm_trials,
    pinned_toolchain,
    supports_pblean_subset,
)
from slm_training.harnesses.experiments import research_06_vss_pblean as r06
from slm_training.research_preregistry import experiment_by_key


def test_toolchain_pin_stable() -> None:
    pin = pinned_toolchain()
    assert pin["toolchain_id"] == "hermetic_python_pblean_pilot_v1"
    assert pin["certificate_format"] == "pblean_pilot"
    assert pin["external_pblean"] == "optional_unavailable_fixture_stub"
    assert len(pin["encoder_hash"]) == 64


def test_exhaustive_and_pblean_agree_on_unsat_suite() -> None:
    for problem in default_unsat_suite():
        control = exhaustive_replay(problem)
        cold, encoded, pb = generate_pblean_certificate(problem)
        assert control.outcome == "unsat"
        assert cold.outcome == "unsat"
        assert cold.certificate is not None and encoded is not None and pb is not None
        warm = check_pblean_certificate(problem, encoded, pb, cold.certificate)
        assert warm.outcome == "unsat"
        assert encode_cnf_to_pb(encoded).digest() == pb.digest()


def test_mutations_rejected() -> None:
    problem = pad_unit_conflict_problem(problem_id="t/mut", pad_vars=2)
    report = mutation_rejection_suite(problem)
    assert report["ok"] is True
    assert report["rejection_rate"] == 1.0


def test_unsupported_and_unknown_preserve_candidates() -> None:
    unsupported = BoundedProblemV1(
        problem_id="t/card",
        domains={"x": (0, 1)},
        clauses=((LiteralV1("x", True),),),
        features=frozenset({"bool_domain", "clause", "cardinality"}),
    )
    assert supports_pblean_subset(unsupported) == "unsupported"
    assert exhaustive_replay(unsupported).outcome == "unknown"
    cold, _, _ = generate_pblean_certificate(unsupported)
    assert cold.outcome == "unknown"

    other_unsat = BoundedProblemV1(
        problem_id="t/xor_unsat",
        domains={"a": (0, 1), "b": (0, 1)},
        clauses=(
            (LiteralV1("a", True), LiteralV1("b", True)),
            (LiteralV1("a", True), LiteralV1("b", False)),
            (LiteralV1("a", False), LiteralV1("b", True)),
            (LiteralV1("a", False), LiteralV1("b", False)),
        ),
        features=frozenset({"bool_domain", "clause"}),
    )
    assert supports_pblean_subset(other_unsat) == "unknown"
    cold2, _, _ = generate_pblean_certificate(other_unsat)
    assert cold2.outcome == "unknown"


def test_paired_report_decision_contract() -> None:
    report = paired_warm_trials(default_unsat_suite(), warm_repeats=3)
    assert report["witness_disagreement_count"] == 0
    assert report["mutation_rejection_rate"] == 1.0
    assert report["supported_subset_coverage"] == 1.0
    assert report["exact_semantic_agreement_rate_on_supported_pb_subset"] == 1.0
    assert report["lrat_comparable_n"] == report["suite_n"]
    assert report["lrat_agreement_n"] == report["suite_n"]
    ratio = report["median_paired_warm_pblean_check_over_exhaustive_replay_time_ratio"]
    assert ratio is not None
    assert report["decision"] in {"accept", "reject"}
    if ratio < 1.0:
        assert report["decision"] == "accept"
    else:
        assert report["decision"] == "reject"


def test_default_off_skips_without_enable(tmp_path: Path) -> None:
    result = r06.run_experiment(root=tmp_path, enabled=False)
    assert result["executed"] is False
    assert result["decision"] == "skipped_default_off"


def test_campaign_lock_roundtrip(tmp_path: Path) -> None:
    lock = r06.write_campaign_lock(tmp_path / "lock.json", root=tmp_path)
    restored = r06.load_campaign_lock(tmp_path / "lock.json")
    assert restored.manifest_sha256 == lock.manifest_sha256
    assert restored.verify_digest().manifest_sha256 == lock.manifest_sha256


def test_preregistry_row_points_at_evidence() -> None:
    row = experiment_by_key("RESEARCH-06")
    assert row["linear_id"] == "SLM-564"
    assert row["default_off"] is True
    assert row["evidence_path"] == r06.EVIDENCE_MD
    assert row["primary_metric"] == (
        "median_paired_warm_pblean_check_over_exhaustive_replay_time_ratio"
    )
