"""Resumed repairs reuse authenticated verifier receipts without resetting budget."""

from types import SimpleNamespace

from scripts import autotrain_verification as owner
from slm_training.autoresearch.heal import repair_source_workspace
from slm_training.autoresearch.heal.repair_acceptance import SourceVerificationGate
from slm_training.autoresearch.heal.repair_release import source_verification_callback
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ResourceGrant


def test_resumed_repair_reuses_completed_successor_without_new_allowance(tmp_path, monkeypatch):
    context = SimpleNamespace(root=tmp_path, loop_id="resume")
    store = CampaignStore("runtime", tmp_path / "loops" / "resume")
    request = SimpleNamespace(digest=lambda: "a" * 64)
    proposal = SimpleNamespace(digest=lambda: "b" * 64)
    grant = ResourceGrant(interrupt_seconds=65, total_seconds=100, max_attempts=2)
    dependency = {
        "request_digest": request.digest(), "proposal_digest": proposal.digest(),
        "root": str(tmp_path / "source"), "state_dir": str(tmp_path / "cache"),
        "base_ref": "frozen-base", "verification_identity": "c" * 64,
        "activity_id": "source-verification", "grant": grant.model_dump(mode="json"),
        "wake": {"predicate": "complete_current_source_verification",
                 "source": "source_verification_completed", "identity_digest": "c" * 64},
    }
    artifact = store.write_artifact("source_verification_requests", dependency)
    store.append_event("source_verification_completed", experiment_id="repair",
                       detail={"dependency_digest": artifact.stem,
                               "request_digest": request.digest(),
                               "proposal_digest": proposal.digest()})
    monkeypatch.setattr(owner, "dependency_plan",
                        lambda value: {"identity": value["verification_identity"]})
    monkeypatch.setattr(owner, "authenticated_completion", lambda _: True)
    monkeypatch.setattr(SourceVerificationGate, "read", lambda self, workspace: {})
    monkeypatch.setattr(repair_source_workspace, "prepare_source_verification",
                        lambda *args: (_ for _ in ()).throw(AssertionError("fresh full grant")))

    gate = source_verification_callback(
        context, SimpleNamespace(source_verification_grant=grant),
    )(request, proposal, object())

    assert gate.identity == dependency["verification_identity"]
