"""Controller provenance, interruption, and version-governance regressions."""

import json
import shutil
from dataclasses import replace

import pytest

from tests.casefiles import case_values
from pydantic import ValidationError

from slm_training.autoresearch.heal.isolation_workspace import manifest_digest, tree_manifest
from slm_training.autoresearch.heal.recovery_dispatch import RepairRecipe
from slm_training.autoresearch.heal.repair_acceptance import proposal_patch_digest
from slm_training.autoresearch.heal.repair_contracts import RepairDispatchResult
from slm_training.autoresearch.heal.repair_governance import reconcile_version_overlay
from slm_training.autoresearch.heal.repair_jobs import logical_job_digest, resumable_proposal
from slm_training.autoresearch.heal.repair_release import source_verification_callback
from slm_training.autoresearch.heal.repair_scope import _next_version, apply_version_overlay
from slm_training.autoresearch.storage import CampaignStore

pytest_plugins = ("tests.test_autoresearch.test_repair_source_workspace",)


def _registry(workspace):
    path = workspace.base / "src/slm_training/resources/versions.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"components": {"fixture": {
        "version": "v1", "paths": ["fixture.py"], "history": [],
    }}}, indent=2, sort_keys=True) + "\n")
    target = workspace.candidate / path.relative_to(workspace.base)
    target.parent.mkdir(parents=True)
    shutil.copyfile(path, target)
    return target


def _result(request, proposal, workspace):
    proposal = proposal.model_copy(update={
        "request_digest": request.digest(),
        "tree_digest": manifest_digest(tree_manifest(workspace.candidate)),
        "patch_digest": proposal_patch_digest(workspace.base, workspace.candidate),
    })
    return RepairDispatchResult(status="waiting_verification", request_digest=request.digest(),
                                proposal=proposal, reason="proposal_received")


@pytest.mark.parametrize("version, expected", case_values(__file__, "test_controller_bumps_every_registry_version_shape"))
def test_controller_bumps_every_registry_version_shape(version, expected):
    assert _next_version(version) == expected


def test_controller_overlay_recovers_after_interruption(repair_inputs, monkeypatch):
    context, _, _, request, proposal, workspace = repair_inputs
    _registry(workspace)
    result = _result(request, proposal, workspace)
    journal = CampaignStore(context.campaign_id, context.root)
    write = journal.write_artifact

    def interrupt(kind, value):
        if kind == "repair_governance_overlays":
            raise KeyboardInterrupt
        return write(kind, value)

    monkeypatch.setattr(journal, "write_artifact", interrupt)
    with pytest.raises(KeyboardInterrupt):
        reconcile_version_overlay(request, result, workspace, journal)
    monkeypatch.setattr(journal, "write_artifact", write)
    governed = reconcile_version_overlay(request, result, workspace, journal)
    assert governed.proposal.tree_digest == manifest_digest(tree_manifest(workspace.candidate))


def test_worker_authored_overlay_has_no_controller_provenance(repair_inputs):
    context, _, _, request, proposal, workspace = repair_inputs
    _registry(workspace)
    assert apply_version_overlay(workspace.base, workspace.candidate,
                                 ("fixture.py", "tests/test_added.py"), request.digest())
    result = _result(request, proposal, workspace)
    with pytest.raises(ValueError, match="no journal provenance"):
        reconcile_version_overlay(request, result, workspace,
                                  CampaignStore(context.campaign_id, context.root))


def test_retained_proposal_outlives_coding_grant(repair_inputs):
    context, config, _, request, proposal, workspace = repair_inputs
    expired = config.model_copy(update={
        "grant": config.grant.model_copy(update={"expires_at": 1.0})
    })
    assert expired.proposal_config_digest() == config.proposal_config_digest()
    renewed = request.model_copy(update={
        "grant": request.grant.model_copy(update={"expires_at": request.grant.expires_at + 3600})
    })
    assert logical_job_digest(request, config.proposal_config_digest()) == logical_job_digest(
        renewed, config.proposal_config_digest()
    )
    request = request.model_copy(update={"grant": expired.grant})
    candidate = context.root / context.campaign_id / "repair_workspaces" / request.digest() / "candidate"
    candidate.parent.mkdir(parents=True)
    workspace.candidate.rename(candidate)
    workspace = replace(workspace, candidate=candidate)
    proposal = _result(request, proposal, workspace).proposal
    assert source_verification_callback(context, expired)(request, proposal, workspace) is not None


def test_pre_normalization_job_event_reuses_retained_proposal(repair_inputs):
    context, config, _, request, proposal, _ = repair_inputs
    journal = CampaignStore(context.campaign_id, context.root)
    journal.write_artifact("repair_requests", request)
    legacy_config = config.proposal_config_digest(include_expiry=True)
    journal.append_event("repair_job_attempt", detail={
        "job_digest": logical_job_digest(
            request, legacy_config, normalize_expiry=False
        ),
        "request_digest": request.digest(),
    })
    result = RepairDispatchResult(status="waiting_verification",
                                  request_digest=request.digest(), proposal=proposal,
                                  reason="proposal_received")
    artifact = journal.write_artifact("repair_dispatch", result)
    journal.append_event("repair_dispatch", artifact_sha256=artifact.stem,
                         detail={"request_digest": request.digest()})
    renewed = request.model_copy(update={
        "grant": request.grant.model_copy(update={"expires_at": request.grant.expires_at + 3600})
    })
    original, retained = resumable_proposal(
        journal, renewed, config.proposal_config_digest(),
        legacy_config_digest=legacy_config,
    )
    assert original == request and retained == result
    changed = config.model_copy(update={"runtime_roots": ("/different-runtime",)})
    _, retained = resumable_proposal(
        journal, renewed, changed.proposal_config_digest(),
        legacy_config_digest=changed.proposal_config_digest(include_expiry=True),
    )
    assert retained is None


def test_repair_recipe_never_grants_controller_registry(repair_inputs):
    recipe = repair_inputs[1].recipes["harness_code_failure"]
    values = {key: getattr(recipe, key) for key in RepairRecipe.model_fields
              if key != "allowed_paths"}
    with pytest.raises(ValidationError, match="controller-owned"):
        RepairRecipe(**values, allowed_paths=("src/slm_training/resources/versions.json",))
