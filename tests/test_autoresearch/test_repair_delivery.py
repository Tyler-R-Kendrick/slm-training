"""Producer/consumer binding fixtures; canonical gate execution is tested separately."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values

from slm_training.autoresearch.heal import repair_delivery as delivery
from slm_training.autoresearch.heal import repair_release as release
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest, private_snapshot, tree_manifest,
)
from slm_training.autoresearch.heal.repair_acceptance import patch_manifest_digest
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core import execution_release as core
from slm_training.harness_core.activity_contract import contract_digest
from tests.test_autoresearch.test_repair_release import publication  # noqa: F401


def test_verified_repair_can_publish_after_coding_grant_expires(request):
    repair_request, result, kwargs = request.getfixturevalue("publication")
    expired_request = repair_request.model_copy(update={
        "grant": repair_request.grant.model_copy(update={"expires_at": 1.0})
    })
    proposal = result.proposal.model_copy(update={"request_digest": expired_request.digest()})
    receipt = result.verification.model_copy(update={
        "request_digest": expired_request.digest(), "proposal_digest": proposal.digest(),
    })
    expired_result = result.model_copy(update={
        "request_digest": expired_request.digest(), "proposal": proposal, "verification": receipt,
    })
    kwargs["authenticated"] = lambda value: value is expired_result
    assert release.publish_verified_repair(
        expired_request, expired_result, **kwargs
    )["resume_activity_id"] == "original-eval"


@pytest.fixture
def accepted_source(tmp_path, monkeypatch, request):
    """Real journals, snapshots and release publisher; simulated verifier evidence."""
    fault = getattr(request, "param", None)
    request, result, kwargs = request.getfixturevalue("publication")
    original = tmp_path / "original"
    original.mkdir()
    (original / "fixture.py").write_text("answer = 41\n")
    ancestry = {"integration_commit": "b" * 40, "upstream_commit": "c" * 40,
                "code_dirty": True}
    predecessor = tmp_path / "predecessor"
    with monkeypatch.context() as patch:
        patch.setattr(core, "_checkout_provenance", lambda _: ancestry)
        marker = core.prepare_release(original, tmp_path / "base", predecessor,
                                      kwargs["destinations"][1])
    candidate = private_snapshot(tmp_path / "base", tmp_path / "accepted-candidate")
    (candidate / "fixture.py").chmod(0o644)
    (candidate / "fixture.py").write_text("answer = 42\n")
    (candidate / "tests").mkdir()
    (candidate / "tests/test_regression.py").write_text("def test_answer():\n    assert 6 * 7 == 42\n")
    if fault == "executable":
        (candidate / "fixture.py").chmod(0o755)
    elif fault in {"symlink", "delete"}:
        (candidate / "fixture.py").unlink()
        if fault == "symlink":
            (candidate / "fixture.py").symlink_to("tests/test_regression.py")
    before, after = tree_manifest(tmp_path / "base"), tree_manifest(candidate)
    request = request.model_copy(update={
        "allowed_paths": ("fixture.py", "tests/test_regression.py"),
        "blocker": request.blocker.model_copy(update={"source_digest": manifest_digest(before)}),
    })
    if fault == "ungranted":
        request = request.model_copy(update={"allowed_paths": ("fixture.py",)})
    proposal = result.proposal.model_copy(update={
        "request_digest": request.digest(), "tree_digest": manifest_digest(after),
        "patch_digest": patch_manifest_digest(before, after),
    })
    journal = CampaignStore(request.campaign_id, kwargs["destinations"][1])
    identity_inputs = {"request_digest": request.digest(), "proposal_digest": proposal.digest(),
                       "config_digest": "d" * 64}
    binding = {"runtime_roots": [], "changed_paths": sorted(after)}
    summary = {"identity": contract_digest(binding), "verification_complete": True}
    inputs = {"schema_version": "repair_source_verification_input/v1", **identity_inputs,
              "source_snapshot_digest": manifest_digest(before),
              "candidate_snapshot_digest": manifest_digest(after), "binding": binding,
              "base_ref": "e" * 40, "verification_identity": summary["identity"]}
    directory_identity = {**identity_inputs, "verification_identity": summary["identity"]}
    directory = journal.root / "source_verification" / contract_digest(directory_identity)
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(inputs))
    private_snapshot(candidate, directory / "root")
    source_inputs = journal.write_artifact("repair_source_verification_inputs", inputs)
    journal.append_event("repair_source_verification_prepared", artifact_sha256=source_inputs.stem)
    evidence = journal.write_artifact("repair_verification", {
        "schema_version": "source_and_predicate_verification/v1",
        "source_verification": summary,
        "source_snapshot_digest": manifest_digest(before),
        "candidate_snapshot_digest": manifest_digest(after),
    })
    receipt = result.verification.model_copy(update={
        "request_digest": request.digest(), "proposal_digest": proposal.digest(),
        "source_digest": manifest_digest(before), "release_digest": proposal.tree_digest,
        "evidence_digest": evidence.stem,
    })
    result = result.model_copy(update={"request_digest": request.digest(),
                                       "proposal": proposal, "verification": receipt})
    for kind, event, value in (("repair_requests", "repair_started", request),
                               ("repair_dispatch", "repair_dispatch", result)):
        artifact = journal.write_artifact(kind, value)
        journal.append_event(event, artifact_sha256=artifact.stem)

    def read_gate(gate, workspace):
        assert gate.identity == summary["identity"]
        assert tree_manifest(gate.root) == tree_manifest(workspace.candidate)
        return summary

    monkeypatch.setattr(delivery.SourceVerificationGate, "read", read_gate)
    monkeypatch.setattr(delivery, "_historical_gate", lambda gate, inputs:
                        read_gate(gate, SimpleNamespace(candidate=candidate)))
    kwargs.update(candidate=candidate, authenticated=lambda supplied: supplied is result,
                  destinations=release.ReleaseDestinations(*kwargs["destinations"],
                      (predecessor, marker["source_digest"])))
    return request, result, kwargs, journal


def test_accepted_source_roundtrip_and_idempotent_handoff(accepted_source):
    request, result, kwargs, journal = accepted_source
    handoff = release.publish_verified_repair(request, result, **kwargs)
    assert release.publish_verified_repair(request, result, **kwargs) == handoff
    subject = delivery.resolve_source_delivery(kwargs["runtime"].store, handoff["source_delivery"])
    assert subject["successor_source_digest"] == handoff["source_digest"]
    assert subject["successor_execution"] == handoff["successor_execution"]
    assert subject["successor_source_digest"] != kwargs["destinations"].verified_predecessor[1]
    assert subject["scope"]["patch_digest"] == result.proposal.patch_digest
    assert set(subject["scope"]["changes"]) == {"fixture.py", "tests", "tests/test_regression.py"}
    assert subject["artifacts"]["repair_requests"] == request.digest()
    assert subject["artifacts"]["repair_dispatch"] == result.digest()
    assert subject["campaign_store"] == str(journal.root)
    assert subject["source_verification"]["base_ref"] == "e" * 40
    assert "remote_base_ref" not in subject
    ancestry = core.runtime_git_provenance(Path(handoff["successor_execution"]))
    assert ancestry == subject["predecessor"]["git_provenance"]
    assert ancestry["code_dirty"] is True


@pytest.mark.parametrize("kind", case_values(__file__, "test_substituted_evidence_rejected_before_publication"))
def test_substituted_evidence_rejected_before_publication(accepted_source, kind):
    request, result, kwargs, journal = accepted_source
    artifact = next((journal.root / "artifacts" / kind).glob("*.json"))
    artifact.write_text('{"substituted":true}')
    with pytest.raises(ValueError, match="artifact_changed"):
        release.publish_verified_repair(request, result, **kwargs)
    assert not (kwargs["runtime"].store.root / "source_release_pointer.json").exists()


@pytest.mark.parametrize("fault", ["candidate", "predecessor", "gate", "manifest"])
def test_stale_source_rejected_before_publication(accepted_source, monkeypatch, fault):
    request, result, kwargs, journal = accepted_source
    if fault == "candidate":
        (kwargs["candidate"] / "extra.py").write_text("unauthorized = True\n")
    elif fault == "predecessor":
        kwargs["destinations"] = kwargs["destinations"]._replace(
            verified_predecessor=(kwargs["destinations"].verified_predecessor[0], "f" * 64))
    elif fault == "gate":
        monkeypatch.setattr(delivery.SourceVerificationGate, "read", lambda *args: None)
    else:
        next((journal.root / "source_verification").glob("*/manifest.json")).write_text("{}")
    with pytest.raises(ValueError):
        release.publish_verified_repair(request, result, **kwargs)
    assert not (kwargs["runtime"].store.root / "source_release_pointer.json").exists()


def test_orphan_subject_cannot_authorize_delivery(accepted_source, monkeypatch):
    request, result, kwargs, _ = accepted_source
    store = kwargs["runtime"].store
    append = store.append_event

    def fail_acceptance(kind, **values):
        if kind == "repair_release_accepted":
            raise OSError("crash before acceptance")
        return append(kind, **values)

    with monkeypatch.context() as patch:
        patch.setattr(store, "append_event", fail_acceptance)
        with pytest.raises(OSError):
            release.publish_verified_repair(request, result, **kwargs)
    artifact = next((store.root / "artifacts/verified_repair_source_delivery").glob("*.json"))
    payload = json.loads(artifact.read_text())
    wait = {"kind": "verified_repair_source", "artifact_sha256": artifact.stem,
            "publication_id": payload["publication_id"],
            "required_capability": "authorized_github_connector_delivery"}
    with pytest.raises(ValueError, match="acceptance_missing"):
        delivery.resolve_source_delivery(store, wait)
    handoff = release.publish_verified_repair(request, result, **kwargs)
    assert handoff["source_delivery"] == wait
    assert delivery.resolve_source_delivery(store, wait)["publication_id"] == payload["publication_id"]


@pytest.mark.parametrize("fault", ["wait", "snapshot", "subject", "source_evidence"])
def test_consumer_rejects_substitution(accepted_source, fault):
    request, result, kwargs, journal = accepted_source
    handoff = release.publish_verified_repair(request, result, **kwargs)
    store = kwargs["runtime"].store
    wait = dict(handoff["source_delivery"])
    if fault == "wait":
        wait["successor_source_digest"] = "a" * 64
    elif fault == "snapshot":
        subject = delivery.resolve_source_delivery(store, wait)
        (Path(subject["successor"]["verified_source"]) / "fixture.py").write_text("changed\n")
    elif fault == "subject":
        (store.root / "artifacts/verified_repair_source_delivery" / f"{wait['artifact_sha256']}.json").write_text("{}")
    else:
        next((journal.root / "artifacts/repair_dispatch").glob("*.json")).write_text("{}")
    with pytest.raises(ValueError):
        delivery.resolve_source_delivery(store, wait)


def test_resolution_is_bound_to_accepted_event_not_latest_pointer(accepted_source):
    request, result, kwargs, _ = accepted_source
    handoff = release.publish_verified_repair(request, result, **kwargs)
    store = kwargs["runtime"].store
    (store.root / "source_release_pointer.json").write_text('{"publication_id":"later"}')
    assert delivery.resolve_source_delivery(store, handoff["source_delivery"])["successor_source_digest"] == handoff["source_digest"]


def test_production_callback_carries_delivery_subject(accepted_source):
    request, result, kwargs, _ = accepted_source
    callback = release.verified_release_callback(
        kwargs["runtime"], kwargs["lease"], destinations=kwargs["destinations"][:2],
        verified_predecessor=kwargs["destinations"].verified_predecessor,
    )
    handoff = callback(request, result, kwargs["candidate"])
    assert delivery.resolve_source_delivery(kwargs["runtime"].store, handoff["source_delivery"])


@pytest.mark.parametrize("accepted_source", ["ungranted", "symlink", "delete"], indirect=True)
def test_receipt_cannot_widen_routine_scope(accepted_source):
    request, result, kwargs, _ = accepted_source
    with pytest.raises(ValueError, match="exact repair grant|delete or link"):
        release.publish_verified_repair(request, result, **kwargs)
    assert not (kwargs["runtime"].store.root / "source_release_pointer.json").exists()


@pytest.mark.parametrize("accepted_source", ["executable"], indirect=True)
def test_executable_mode_is_preserved_in_exact_scope(accepted_source):
    request, result, kwargs, _ = accepted_source
    handoff = release.publish_verified_repair(request, result, **kwargs)
    subject = delivery.resolve_source_delivery(kwargs["runtime"].store, handoff["source_delivery"])
    assert subject["scope"]["changes"]["fixture.py"]["after"][0] & 0o111
    assert subject["successor"]["files"]["fixture.py"][1] is True


def test_unrelated_current_pointer_cannot_become_predecessor(accepted_source):
    request, result, kwargs, _ = accepted_source
    pointer = kwargs["runtime"].store.root / "source_release_pointer.json"
    pointer.write_text(json.dumps({"publication_id": "other", "runtime_source_digest": "f" * 64}))
    kwargs["expected_previous"] = "other"
    with pytest.raises(ValueError, match="predecessor_publication_mismatch"):
        release.publish_verified_repair(request, result, **kwargs)
    assert not any(event["event_type"] == "repair_release_accepted"
                   for event in kwargs["runtime"].store.verify_event_chain())
