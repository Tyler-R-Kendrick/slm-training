from __future__ import annotations

from slm_training.harnesses.experiments.slm194_candidate_proposals import (
    ARM_NAMES,
    run_candidate_proposal_matrix,
)


def test_matrix_covers_required_arms_k_grid_and_safety() -> None:
    report = run_candidate_proposal_matrix(steps=2, max_wall_minutes=2.8)
    assert tuple(report["arms"]) == ARM_NAMES
    assert report["k_grid"] == [1, 2, 4, 8, 16, "all"]
    assert report["confirmation"]["status"] == "not_touched"
    assert not report["checkpoint"]["written"]
    assert not report["common_candidate_interface"]["unknown_is_negative"]
    for arm, payload in report["arms"].items():
        if arm == "oracle_acceptable":
            continue
        for result in payload["k_results"].values():
            assert result["exact_final_output_parity"]
            assert result["invalid_over_valid_selections"] == 0
            assert not result["unknown_as_negative"]


def test_matrix_is_honest_when_joint_gate_does_not_clear() -> None:
    report = run_candidate_proposal_matrix(steps=2, max_wall_minutes=2.8)
    if not report["positive_claim_eligible"]:
        assert report["honest_verdict"] == "retain_exact_cached_enumeration"
    assert report["corpus"]["confirmation_rows"] == 0
    assert report["prerequisite_manifests"]["slm192_profile_sha256"]
    assert report["prerequisite_manifests"]["slm193_cache_sha256"]
