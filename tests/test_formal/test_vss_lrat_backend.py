"""RESEARCH-05 / SLM-563: VSS LRAT SAT pilot tests (default-off)."""

from __future__ import annotations

from pathlib import Path

from slm_training.formal.encoding_adapter import BoundedProblemV1, LiteralV1
from slm_training.formal.vss_lrat_backend import (
    check_lrat_certificate,
    default_unsat_suite,
    exhaustive_replay,
    generate_lrat_certificate,
    mutation_rejection_suite,
    pad_unit_conflict_problem,
    paired_warm_trials,
    pinned_toolchain,
    supports_lrat_subset,
)
from slm_training.harnesses.experiments import research_05_vss_lrat as r05
from slm_training.research_preregistry import experiment_by_key


def test_toolchain_pin_stable() -> None:
    pin = pinned_toolchain()
    assert pin["toolchain_id"] == "hermetic_python_lrat_pilot_v1"
    assert pin["certificate_format"] == "lrat_pilot"
    assert len(pin["encoder_hash"]) == 64


def test_exhaustive_and_lrat_agree_on_unsat_suite() -> None:
    for problem in default_unsat_suite():
        control = exhaustive_replay(problem)
        cold, encoded = generate_lrat_certificate(problem)
        assert control.outcome == "unsat"
        assert cold.outcome == "unsat"
        assert cold.certificate is not None and encoded is not None
        warm = check_lrat_certificate(problem, encoded, cold.certificate)
        assert warm.outcome == "unsat"


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
    assert supports_lrat_subset(unsupported) == "unsupported"
    assert exhaustive_replay(unsupported).outcome == "unknown"
    cold, _ = generate_lrat_certificate(unsupported)
    assert cold.outcome == "unknown"

    # Unsat without unit conflict → unknown (no silent approx).
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
    assert supports_lrat_subset(other_unsat) == "unknown"
    cold2, _ = generate_lrat_certificate(other_unsat)
    assert cold2.outcome == "unknown"


def test_paired_report_decision_contract() -> None:
    report = paired_warm_trials(default_unsat_suite(), warm_repeats=3)
    assert report["witness_disagreement_count"] == 0
    assert report["mutation_rejection_rate"] == 1.0
    assert report["supported_subset_coverage"] == 1.0
    ratio = report["median_paired_warm_lrat_check_over_exhaustive_replay_time_ratio"]
    assert ratio is not None
    assert report["decision"] in {"accept", "reject"}
    if ratio < 1.0:
        assert report["decision"] == "accept"
    else:
        assert report["decision"] == "reject"


def test_default_off_skips_without_enable(tmp_path: Path) -> None:
    result = r05.run_experiment(root=tmp_path, enabled=False)
    assert result["executed"] is False
    assert result["decision"] == "skipped_default_off"


def test_campaign_lock_roundtrip(tmp_path: Path) -> None:
    lock = r05.write_campaign_lock(tmp_path / "lock.json", root=tmp_path)
    restored = r05.load_campaign_lock(tmp_path / "lock.json")
    assert restored.manifest_sha256 == lock.manifest_sha256
    assert restored.verify_digest().manifest_sha256 == lock.manifest_sha256


def test_preregistry_row_points_at_evidence() -> None:
    row = experiment_by_key("RESEARCH-05")
    assert row["linear_id"] == "SLM-563"
    assert row["default_off"] is True
    assert row["evidence_path"] == r05.EVIDENCE_MD
    assert row["primary_metric"] == (
        "median_paired_warm_lrat_check_over_exhaustive_replay_time_ratio"
    )
