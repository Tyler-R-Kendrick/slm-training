"""A campaign cannot execute using a different grant from its scientific lock."""

from types import SimpleNamespace

import pytest

from scripts import autoresearch
from slm_training.autoresearch.schemas import CampaignBudget
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ResourceGrant
from tests.test_autoresearch.test_harness import (
    campaign,
    experiment_campaign,
    hypothesis_matrix,
)


@pytest.mark.parametrize("explicit_path", [False, True])
@pytest.mark.parametrize("changed", [None, 601])
def test_mismatch_refused_before_compile_or_execution(
    tmp_path, monkeypatch, explicit_path, changed
):
    grant = ResourceGrant(total_seconds=600, max_attempts=4)
    configured = campaign().model_copy(
        update={"budget": CampaignBudget(continuation_grant=grant)}
    )
    store = CampaignStore(configured.campaign_id, tmp_path)
    store.initialize(configured)
    matrix = hypothesis_matrix()
    artifact = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=artifact.stem)
    other = (
        None if changed is None else grant.model_copy(update={"total_seconds": changed})
    )
    manifest = experiment_campaign(
        experiment_id=matrix.recommended_experiment_id
    ).model_copy(update={"budget": CampaignBudget(continuation_grant=other)})
    path = store.write_artifact("fixture_manifests", manifest)
    if not explicit_path:
        store.lock_experiment_campaign(manifest)
    monkeypatch.setattr(
        autoresearch,
        "compile_commands",
        lambda *a, **kw: pytest.fail("compiled mismatched grant"),
    )
    args = SimpleNamespace(
        campaign_id=store.campaign_id,
        root=tmp_path,
        experiment=None,
        campaign_manifest=path if explicit_path else None,
        execute=True,
        trackio=False,
        experiment_wall_seconds=30,
    )
    with pytest.raises(ValueError, match="continuation grant.*successor"):
        autoresearch.cmd_run(args)
    assert not any(
        e["event_type"] == "experiment_started" for e in store.verify_event_chain()
    )


def test_matching_grant_and_legacy_bytes_preserved():
    legacy = '{"max_experiments":12,"max_gpu_hours":0.0,"max_wall_minutes":60.0}'
    budget = CampaignBudget.model_validate_json(legacy)
    budget.require_matching_grant(CampaignBudget())
    assert budget.model_dump_json() == legacy
    assert budget.logical_seconds == budget.bounded_invocation_seconds() == 180
    assert budget.bounded_invocation_seconds(30) == 30
    grant = CampaignBudget(continuation_grant=ResourceGrant(total_seconds=600))
    grant.require_matching_grant(
        CampaignBudget.model_validate_json(grant.model_dump_json())
    )
    assert grant.logical_seconds == 600 and grant.bounded_invocation_seconds() == 180


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0, -1])
def test_invalid_dynamic_limit_rejected(invalid):
    with pytest.raises(ValueError, match="positive and finite"):
        CampaignBudget().bounded_invocation_seconds(invalid)
