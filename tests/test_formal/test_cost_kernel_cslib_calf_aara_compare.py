"""RESEARCH-20 / SLM-552 backend tests."""

from pathlib import Path

from slm_training.formal import cost_kernel_cslib_calf_aara_compare as ck

ROOT = Path(__file__).resolve().parents[2]


def test_paired_report_rejects_external_formalisms() -> None:
    report = ck.paired_compare_report(root=ROOT)
    assert report["decision"] == "reject"
    assert report["reason"] == "external_formalism_proof_burden_without_transfer_gain"
    assert report["transferability_disagreement_rate_across_cost_formalisms"] > 0.0
    assert report["new_bound_classes_enabled"] == 0
    assert report["maintenance_cost_loc_burden_delta"] > 0


def test_local_kernel_faithful_on_eval() -> None:
    fixtures = [f for f in ck.default_fixtures(root=ROOT) if f.split == "eval"]
    for fixture in fixtures:
        row = ck.score_local_kernel(fixture)
        assert row["faithful_to_fixture"] is True
