"""Policy resolution cannot reinterpret the primary after campaign locking."""

import pytest

from tests.casefiles import case_values

from scripts.autotrain_measurement import locked_primary_failure
from slm_training.autoresearch.storage import CampaignStore
from tests.test_autoresearch.test_experiment_campaign import _manifest


def _locked(tmp_path):
    manifest = _manifest()
    store = CampaignStore(manifest.campaign_id, tmp_path)
    store.lock_experiment_campaign(manifest)
    return store


def test_matching_locked_primary_can_be_consumed(tmp_path):
    store = _locked(tmp_path)
    assert locked_primary_failure(store.root, "control", "candidate", "binder_reference_f1",
                                  {"direction": "increase", "minimum_effect": 0.01}) is None


@pytest.mark.parametrize("metric,direction,effect", case_values(__file__, "test_metric_direction_or_effect_drift_refused"))
def test_metric_direction_or_effect_drift_refused(tmp_path, metric, direction, effect):
    store = _locked(tmp_path)
    assert locked_primary_failure(store.root, "control", "candidate", metric,
                                  {"direction": direction, "minimum_effect": effect}) == (
        "measurement_incomplete:locked_primary_policy_mismatch"
    )


def test_real_driver_refuses_before_reading_or_calibrating_outcomes(tmp_path, monkeypatch):
    from scripts import run_autotrain_continuous as driver
    store = _locked(tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("incompatible measurement must not be consumed")

    monkeypatch.setattr(driver, "_run_metrics", unexpected)
    result = driver._classify_positive(camp_dir=store.root, primary_metric="binder_reference_f1",
                                       control_id="control", candidate_id="candidate")
    assert result["positive"] is False
    assert result["stack_layer"] is False
    assert result["measurement_complete"] is False
    assert result["scientific_status"] == "incomplete"


def test_existing_journal_without_pair_lock_does_not_authorize_comparison(tmp_path):
    store = CampaignStore("missing", tmp_path)
    store.append_event("initialized")
    assert locked_primary_failure(store.root, "control", "candidate", "binder_reference_f1",
                                  {"direction": "increase", "minimum_effect": 0.01})
