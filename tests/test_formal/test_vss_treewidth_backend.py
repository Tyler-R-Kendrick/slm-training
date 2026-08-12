"""RESEARCH-09 / SLM-541: VSS treewidth / parameterized pilot tests (default-off)."""

from __future__ import annotations

from pathlib import Path

from slm_training.formal.vss_treewidth_backend import (
    bag_dp_solve,
    default_param_suite,
    exhaustive_replay,
    instrument_structural_params,
    mutation_rejection_suite,
    paired_param_report,
    pinned_toolchain,
    spearman_rank_correlation,
    supports_treewidth_subset,
    unsupported_high_tw_fixture,
)
from slm_training.harnesses.experiments import research_09_vss_treewidth as r09
from slm_training.research_preregistry import experiment_by_key


def test_toolchain_pin_stable() -> None:
    pin = pinned_toolchain()
    assert pin["toolchain_id"] == "hermetic_python_treewidth_proxy_v1"
    assert pin["backend_id"] == "vss_treewidth_param_pilot"
    assert pin["treatment_solver"] == "bag_dp_on_nice_elimination"


def test_structural_params_instrument_without_solver_change() -> None:
    problem = default_param_suite()[0]
    params = instrument_structural_params(problem)
    assert params.n_vars >= 2
    assert params.exact_treewidth is True
    assert params.treewidth is not None
    assert params.binder_depth >= 1
    assert params.residual_class_count >= 1
    # Production enumerator still decides the same instance.
    control = exhaustive_replay(problem)
    assert control.outcome in {"sat", "unsat"}


def test_enumerative_and_bag_dp_agree_on_suite() -> None:
    for problem in default_param_suite():
        assert supports_treewidth_subset(problem) == "supported"
        control = exhaustive_replay(problem)
        treatment = bag_dp_solve(problem)
        assert control.outcome == treatment.outcome
        assert treatment.outcome in {"sat", "unsat"}


def test_unsupported_stays_unknown() -> None:
    unsupported = unsupported_high_tw_fixture()
    assert supports_treewidth_subset(unsupported) == "unsupported"
    cold = bag_dp_solve(unsupported)
    assert cold.outcome == "unknown"


def test_mutations_rejected() -> None:
    problem = default_param_suite()[0]
    report = mutation_rejection_suite(problem)
    assert report["ok"] is True
    assert report["rejection_rate"] == 1.0


def test_spearman_monotonic() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    assert spearman_rank_correlation(xs, xs) == 1.0
    assert spearman_rank_correlation(xs, list(reversed(xs))) == -1.0


def test_paired_report_contract_fields() -> None:
    report = paired_param_report(default_param_suite())
    assert report["witness_disagreement_count"] == 0
    assert report["timeout_as_refutation_count"] == 0
    assert report["parameter_estimation_failures"] == 0
    assert report["mutation_rejection_rate"] == 1.0
    assert report["decision"] in {"accept", "reject"}
    assert "rank_correlation_of_predicted_vs_observed_solver_cost" in report


def test_default_off_skips_without_enable(tmp_path: Path) -> None:
    result = r09.run_experiment(root=tmp_path, enabled=False)
    assert result["executed"] is False
    assert result["decision"] == "skipped_default_off"


def test_campaign_lock_roundtrip(tmp_path: Path) -> None:
    lock = r09.write_campaign_lock(tmp_path / "lock.json", root=tmp_path)
    restored = r09.load_campaign_lock(tmp_path / "lock.json")
    assert restored.manifest_sha256 == lock.manifest_sha256
    assert restored.verify_digest().manifest_sha256 == lock.manifest_sha256


def test_preregistry_row_points_at_evidence() -> None:
    row = experiment_by_key("RESEARCH-09")
    assert row["linear_id"] == "SLM-541"
    assert row["default_off"] is True
    assert row["evidence_path"] == r09.EVIDENCE_MD
    assert row["primary_metric"] == r09.PRIMARY_METRIC
    assert row["knobs"]["pilot"] == "vss_treewidth"
    assert row["knobs"]["feature"] == "treewidth_proxy"
