"""Scientific resource contracts use the same explicit grant as execution."""

import json
from types import SimpleNamespace

import pytest

from scripts import autoresearch, autotrain_cycle_prepare, autotrain_search
from scripts import run_autotrain_continuous as driver
from slm_training.autoresearch.schemas import CampaignBudget
from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ResourceGrant
from tests.test_autoresearch.test_harness import campaign, experiment, experiment_campaign, hypothesis_matrix


def configured_store(tmp_path, grant):
    configured = campaign().model_copy(update={"budget": CampaignBudget(continuation_grant=grant)})
    store = CampaignStore(configured.campaign_id, tmp_path)
    store.initialize(configured)
    return store


@pytest.mark.parametrize("owner", ["cycle", "search"])
def test_real_manifest_callers_preserve_campaign_grant(tmp_path, owner):
    grant = ResourceGrant(total_seconds=600, max_attempts=4)
    store = configured_store(tmp_path, grant)
    spec = experiment()
    if owner == "search":
        options = autotrain_search.manifest_options(
            driver, store=store, candidate=spec.model_dump(mode="json"),
            integration="a" * 40, policy=None, slug="test-grant",
        )
        manifest = options["manifest_templates"][spec.experiment_id]
    else:
        value = dict(replay_manifest_paths={}, screening_multi=False, candidate_eid=spec.experiment_id,
                     integration="a" * 40, role="screening", cycle_intent="screening", promotion_chunk_plan=None)
        path = autotrain_cycle_prepare._manifest_path(store, driver, value, spec)
        manifest = ExperimentCampaignV1.model_validate_json(path.read_text())
    assert manifest.budget.continuation_grant == grant
    assert manifest.budget.logical_seconds == 600


@pytest.mark.parametrize("manifest_grant", [None, ResourceGrant(total_seconds=601, max_attempts=4)])
def test_cmd_run_rejects_mismatched_locked_grant_before_compilation(tmp_path, monkeypatch, manifest_grant):
    store = configured_store(tmp_path, ResourceGrant(total_seconds=600, max_attempts=4))
    matrix = hypothesis_matrix()
    artifact = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=artifact.stem)
    selected = matrix.recommended_experiment_id
    manifest = experiment_campaign(experiment_id=selected).model_copy(
        update={"budget": CampaignBudget(continuation_grant=manifest_grant)})
    store.lock_experiment_campaign(manifest)
    monkeypatch.setattr(autoresearch, "compile_commands", lambda *a, **kw: pytest.fail("compiled mismatched grant"))
    args = SimpleNamespace(campaign_id=store.campaign_id, root=tmp_path, experiment=None,
                           execute=True, trackio=False, experiment_wall_seconds=30)
    with pytest.raises(ValueError, match="continuation grant.*successor"):
        autoresearch.cmd_run(args)
    assert not any(e["event_type"] == "experiment_started" for e in store.verify_event_chain())


def test_legacy_budget_golden_and_historical_wall():
    golden = '{"max_experiments":12,"max_gpu_hours":0.0,"max_wall_minutes":60.0}'
    budget = CampaignBudget.model_validate_json(golden)
    assert budget.model_dump_json() == golden and budget.logical_seconds == 180
    assert CampaignBudget.model_validate_json(budget.model_dump_json()) == budget


@pytest.mark.parametrize("field,value", [("total_seconds", float("nan")), ("total_seconds", float("inf")),
                                        ("max_attempts", True), ("max_attempts", 0),
                                        ("interrupt_seconds", 171), ("kill_grace_seconds", 11),
                                        ("total_seconds", 180)])
def test_explicit_grant_rejects_invalid_or_unfunded_contract(field, value):
    payload = {"continuation_grant": {"total_seconds": 600, field: value}}
    with pytest.raises(ValueError):
        CampaignBudget.model_validate_json(json.dumps(payload))
