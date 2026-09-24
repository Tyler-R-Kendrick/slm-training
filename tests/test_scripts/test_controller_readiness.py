"""Controller request production, independent action locks and bootstrap replay."""
import pytest

from scripts.autotrain_controller_repair import _resolve_data_request, bootstrap_screening_request
from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.readiness_contract import load_locked_readiness_request
from slm_training.harnesses.train_data.readiness import lock_readiness_request
from tests.test_autoresearch.test_experiment_campaign import _manifest_payload
from tests.test_harnesses.train_data.readiness_fixtures import certified_screening_fixture, request_fixture, screening_context


def _arguments(root):
    cwd = certified_screening_fixture(root.parent / "source")
    return dict(cwd=cwd, root=root, train_version="wf_smoke_v2",
                eval_version="e938_role_safe_all_targets_smoke96_v2", minimum=6)


def test_unrelated_locked_action_does_not_suppress_request_creation(tmp_path):
    old = request_fixture(tmp_path)
    store = CampaignStore(old.campaign_id, tmp_path / "campaigns")
    manifest = ExperimentCampaignV1.model_validate(
        _manifest_payload(campaign_id=old.campaign_id, claim_class="fixture"))
    store.lock_experiment_campaign(manifest)
    lock_readiness_request(store, old, experiment_id=manifest.experiment_id)
    args = _arguments(tmp_path / "campaigns")
    args["screening_context"] = tuple(args.pop(key) for key in ("train_version", "eval_version", "minimum"))
    args.update(campaign_id=old.campaign_id, blocker={"data_action_id": "new-action"},
                screening=True, readiness_context=screening_context(args["cwd"]))
    new = _resolve_data_request(**args)
    assert new is not None and new.action_id == "new-action"
    assert _resolve_data_request(**args) == new
    assert load_locked_readiness_request(store, action_id=old.action_id) == old
    assert sum(event["event_type"] == "data_readiness_request_locked"
               for event in store.verify_event_chain()) == 2
    args["readiness_context"]["trainer_config"]["d_model"] *= 2
    assert _resolve_data_request(**args) is None
    assert load_locked_readiness_request(store, action_id="new-action") == new


def test_bootstrap_is_a_durable_readiness_lock_not_a_scientific_manifest(tmp_path):
    args = _arguments(tmp_path / "campaigns")
    args.update(loop_id="first-cycle", context=screening_context(args["cwd"]))
    first = bootstrap_screening_request(**args)
    assert first == bootstrap_screening_request(**args)
    store = CampaignStore(first["campaign_id"], args["root"])
    request = load_locked_readiness_request(store, action_id=first["data_action_id"])
    assert request.sha256 == first["data_readiness_request_sha256"]
    events = store.verify_event_chain()
    assert len(events) == 1 and events[0]["event_type"] == "data_readiness_request_locked"
    assert events[0].get("experiment_id") is None
    assert "manifest_sha256" not in events[0]["detail"]
    inputs = store.root / "artifacts/data_readiness_inputs" / (events[0]["detail"]["activity_input_sha256"] + ".json")
    inputs.write_text("{}")
    with pytest.raises(ValueError, match="bootstrap readiness input"):
        load_locked_readiness_request(store, action_id=first["data_action_id"])


def test_missing_context_records_scoped_capability_not_false_healing(tmp_path):
    args = _arguments(tmp_path / "campaigns")
    args["screening_context"] = tuple(args.pop(key) for key in ("train_version", "eval_version", "minimum"))
    args.update(campaign_id="missing", blocker={"data_action_id": "needs-context"}, screening=True)
    for _ in range(2):
        assert _resolve_data_request(**args) is None
    events = CampaignStore("missing", args["root"]).verify_event_chain()
    assert len(events) == 1
    assert events[0]["status"] == "waiting_capability"
    assert events[0]["detail"]["needed_capability"] == "locked_data_readiness_context"
    assert events[0]["detail"]["data_action_id"] == "needs-context"


def test_bootstrap_lock_rejects_request_with_different_controller_config(tmp_path):
    import json

    args = _arguments(tmp_path / "campaigns")
    args.update(loop_id="first-cycle", context=screening_context(args["cwd"]))
    first = bootstrap_screening_request(**args)
    store = CampaignStore(first["campaign_id"], args["root"])
    event = store.verify_event_chain()[0]
    inputs = json.loads((store.root / "artifacts/data_readiness_inputs" /
        (event["detail"]["activity_input_sha256"] + ".json")).read_text())
    request = load_locked_readiness_request(store)
    changed = request.model_copy(update={"trainer_config": {**request.trainer_config, "d_model": 32}})
    with pytest.raises(ValueError, match="bootstrap readiness context mismatch"):
        lock_readiness_request(store, changed, activity_inputs=inputs)


def test_bootstrap_recovers_initialized_activity_without_request_lock(tmp_path):
    args = _arguments(tmp_path / "campaigns")
    args.update(loop_id="first-cycle", context=screening_context(args["cwd"]))
    first = bootstrap_screening_request(**args)
    args["root"] = tmp_path / "restarted-campaigns"
    store = CampaignStore(first["campaign_id"], args["root"])
    store.append_event("activity_initialized", status="running")
    recovered = bootstrap_screening_request(**args)
    assert recovered["data_readiness_request_sha256"] == first["data_readiness_request_sha256"]
    assert load_locked_readiness_request(store).action_id == first["data_action_id"]


@pytest.mark.parametrize("configured", [False, True])
def test_one_pending_source_verification_stops_further_repair_dispatch(tmp_path, monkeypatch, configured):
    from scripts import autotrain_controller_repair as controller
    from types import SimpleNamespace

    calls = []
    if configured:
        monkeypatch.setattr(controller, "load_recovery_config", lambda _: SimpleNamespace(
            grant=None, source_verification_grant=None))
    def dispatch(pending, *args, **kwargs):
        calls.append(pending)
        callback = kwargs["source_verification"]
        if configured:
            assert callable(callback) and callback(None, None, None) is None
        else:
            assert callback is None
        return {"status": "waiting_verification", "verification_dependency": {
            "verification_identity": "a" * 64, "grant_required": True}}
    monkeypatch.setattr(controller, "dispatch_hard_pending", dispatch)
    request = {"lease": {"activity_id": "repair", "attempt_id": "attempt", "epoch": "epoch",
        "generation": 1, "token": "fence", "owner_identity": "controller", "expires_at": 1.0},
        "source_digest": "a" * 64, "environment_digest": "b" * 64,
        "parent_event": "parent", "campaign_id": "campaign", "hard_pending": [
            {"kind": "repair_harness", "blocker_code": "harness_code_failure", "reason": "first"},
            {"kind": "repair_harness", "blocker_code": "harness_code_failure", "reason": "second"}]}
    results = controller.dispatch_controller_repairs(request, cwd=tmp_path, root=tmp_path / "campaigns", loop_id="loop")
    assert len(results) == len(calls) == 1


def test_declared_parent_requires_every_actual_lineage_ancestor(tmp_path):
    from dataclasses import replace
    from slm_training.harness_core.lineage.records import DataSnapshot
    from slm_training.harness_core.lineage.store import LineageStore
    from slm_training.harnesses.train_data.readiness import build_screening_request
    from tests.test_lineage.test_lineage import run_manifest
    from tests.test_harnesses.train_data.readiness_fixtures import record, write_snapshot

    args = _arguments(tmp_path / "campaigns")
    context = screening_context(args["cwd"])
    old = write_snapshot(args["cwd"], "ancestor", [record(id="old-case", prompt="Old training")])
    store = LineageStore(args["cwd"] / "lineage")
    for run_id, ref, parents in (("ancestor-run", old.model_dump(), ()),
                                ("start-run", context["training_ancestors"][0], ("ancestor-run",))):
        snapshot = DataSnapshot(snapshot_id=run_id, sources=(ref["directory"],), records_sha=ref["records_sha256"],
                                record_count=1, target_token_count=1, created_at="2026-09-08T00:00:00Z")
        store.write_snapshot(snapshot)
        store.create_run(replace(run_manifest(run_id, parent_ids=parents), data_snapshot_sha=snapshot.sha))
    context.update(initialization="parent", lineage_root="lineage", starting_run_id="start-run")
    factory = dict(root=args["cwd"], campaign_id="factory", action_id="ancestry",
                   train_version=args["train_version"], eval_version=args["eval_version"],
                   minimum=6, context=context)
    with pytest.raises(ValueError, match="missing training ancestor"):
        build_screening_request(**factory)
    context["training_ancestors"].append(old.model_dump())
    request = build_screening_request(**factory)
    assert request.initialization == "parent" and request.starting_run_id == "start-run"
    assert len(request.training_ancestors) == 2


def test_first_cycle_bootstrap_dispatches_actual_builder_without_scientific_lock(tmp_path):
    import json
    from scripts.autotrain_controller_repair import dispatch_screening_rebuild

    args = _arguments(tmp_path / "campaigns")
    args.update(loop_id="first-cycle", minimum=2,
                readiness_context=screening_context(args["cwd"]), campaign_id=None)
    assert dispatch_screening_rebuild(**args) == "screening_successor_ready"
    assert dispatch_screening_rebuild(**args) is None
    campaigns = list(args["root"].glob("readiness-*"))
    assert len(campaigns) == 1
    events = CampaignStore(campaigns[0].name, args["root"]).verify_event_chain()
    assert not any(event["event_type"] == "experiment_campaign_locked" for event in events)
    ready = [event for event in events if event["event_type"] == "screening_successor_ready"]
    assert len(ready) == 1 and ready[0]["detail"]["measurement_complete"] is False
    evidence = json.loads((campaigns[0] / "artifacts/data_readiness" /
                           (ready[0]["artifact_sha256"] + ".json")).read_text())
    assert evidence["ready"] and evidence["usable_unique_cases"] == evidence["usable_unique_families"] == 2
    assert evidence["preparation"]["forwards"] == 0
