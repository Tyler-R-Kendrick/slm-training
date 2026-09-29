"""Real certified sampler -> immutable DataStore -> original readiness probe."""

import json

import pytest

from scripts.autotrain_data_builder import build_requested_successor
from slm_training.data.readiness_contract import DataReadinessRequest, evidence_digest, file_digest
from slm_training.dsl.schema import load_jsonl, write_jsonl
from slm_training.harnesses.train_data.readiness import check_readiness
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1
from tests.test_harnesses.train_data.readiness_fixtures import certified_screening_fixture, record, request_fixture, screening_context, write_snapshot


def _request(root):
    training = request_fixture(root, bad=False)
    original = write_snapshot(root, "screening", [record(
        id="existing", prompt="Separate the page", openui="root = Separator()",
        placeholders=[], split="smoke", meta={"root_parent_id": "existing-family"},
    )], kind="eval")
    split = RootFamilySplitPolicyV1()
    family = next(f"sample-family-{i}" for i in range(1000)
                  if split.assign(f"sample-family-{i}") == "validation")
    corpus = root / "corpus.jsonl"
    write_jsonl(corpus, [record(
        id="candidate", prompt="Submit the form", split="train",
        openui='root = Stack([action])\naction = Button(":slot_0")',
        meta={"root_parent_id": family},
    )])
    policy = {"source": "certified", "corpus": "corpus.jsonl",
              "corpus_sha256": file_digest(corpus), "seed": 0,
              "splits": ["validation"], "suite": "smoke"}
    return DataReadinessRequest(
        **{**training.model_dump(), "purpose": "screening_volume", "original": original,
           "scored_suites": [original], "training_ancestors": [training.original],
           "minimum_unique_cases": 2, "minimum_unique_families": 2,
           "sampling_policy_digest": evidence_digest(policy), "generation_knobs": policy}
    )


@pytest.mark.parametrize("through_playbook", [False, True])
def test_actual_sampler_publishes_only_ready_successor(tmp_path, through_playbook):
    request = _request(tmp_path)
    source = tmp_path / request.original.directory / "records.jsonl"
    before = source.read_bytes()
    assert not check_readiness(request, root=tmp_path)["ready"]
    if through_playbook:
        from slm_training.autoresearch.heal.playbooks import data_rebuild
        from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
        from slm_training.autoresearch.storage import CampaignStore
        from slm_training.harnesses.train_data.readiness import lock_readiness_request
        from tests.test_autoresearch.test_experiment_campaign import _manifest_payload

        store = CampaignStore(request.campaign_id, tmp_path / "campaigns")
        manifest = ExperimentCampaignV1.model_validate(
            _manifest_payload(campaign_id=request.campaign_id, claim_class="fixture"))
        store.lock_experiment_campaign(manifest)
        lock_readiness_request(store, request, experiment_id=manifest.experiment_id)
        blocker = {"kind": "rebuild_data", "reason": "screening volume"}
        receipt = data_rebuild.execute(blocker, cwd=tmp_path, root=tmp_path / "campaigns",
                                       loop_id="loop", campaign_id=request.campaign_id)
        assert receipt.outcome == "healed", receipt
        from slm_training.harnesses.train_data.readiness import snapshot_input
        candidate = snapshot_input(tmp_path, tmp_path / "outputs/data/eval/successor",
                                   exposure="public_regression")
        result = {"published": True, "candidate": candidate.model_dump(),
                  "evidence": check_readiness(request, root=tmp_path, candidate=candidate)}
    else:
        result = build_requested_successor(cwd=tmp_path, request=request)
    assert result["published"], result
    assert result["evidence"]["ready"]
    assert result["evidence"]["usable_unique_cases"] == 2
    assert result["evidence"]["usable_unique_families"] == 2
    destination = tmp_path / result["candidate"]["directory"]
    assert (destination / "records.jsonl").read_bytes().startswith(before)
    assert source.read_bytes() == before
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["suites"]["smoke"] == str(destination / "records.jsonl")
    assert len(load_jsonl(destination / "records.jsonl")) == 2
    quality = json.loads((destination / "quality_report.json").read_text())
    assert quality["suite_counts"] == {"smoke": 2}
    assert quality["error_count"] == 0
    assert quality["certified"]["readiness"]["ready"] is True
    assert quality["certified"]["sampling"]["selected"] == 1
    assert (destination / "screening_sample_size.json").is_file()
    assert not (tmp_path / "src").exists()


@pytest.mark.parametrize("fault", ["missing_ancestry", "changed_policy", "changed_corpus", "held_out"])
def test_no_guessing_sampling_authority_or_inputs(tmp_path, fault):
    request = _request(tmp_path)
    if fault == "missing_ancestry":
        request = request.model_copy(update={"training_ancestors": []})
    elif fault == "changed_policy":
        request = request.model_copy(update={"sampling_policy_digest": "0" * 64})
    elif fault == "changed_corpus":
        (tmp_path / "corpus.jsonl").write_text("{}\n")
    else:
        request = request.model_copy(update={"generation_knobs": {
            **request.generation_knobs, "splits": ["test"], "suite": "held_out"}})
    with pytest.raises(ValueError):
        build_requested_successor(cwd=tmp_path, request=request)
    assert not (tmp_path / "outputs/runs").exists()
    assert not (tmp_path / "outputs/data/eval/successor").exists()


def test_screening_blocker_locks_request_from_driver_inputs(tmp_path, monkeypatch):
    from scripts.autotrain_controller_repair import dispatch_screening_rebuild
    from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.data.readiness_contract import load_locked_readiness_request
    from tests.test_autoresearch.test_experiment_campaign import _manifest_payload
    from tests.test_scripts.test_run_autotrain_continuous import _screening_rebuild_handoff

    campaign_id = "screening-auto-request"
    root = tmp_path / "campaigns"
    handoff_path = _screening_rebuild_handoff(root, campaign_id)
    payload = json.loads(handoff_path.read_text())
    payload["actions"][0]["blocker_code"] = "screening_suite_volume"
    handoff_path.write_text(json.dumps(payload) + "\n")
    store = CampaignStore(campaign_id, root)
    manifest = ExperimentCampaignV1.model_validate(
        _manifest_payload(campaign_id=campaign_id, claim_class="fixture")
    )
    store.lock_experiment_campaign(manifest)
    monkeypatch.setattr("slm_training.autoresearch.heal.run_playbooks", lambda **_: ())
    cwd = certified_screening_fixture(tmp_path / "source")
    result = dispatch_screening_rebuild(
        cwd=cwd, root=root, loop_id="loop-1",
        campaign_id=campaign_id, train_version="wf_smoke_v2",
        eval_version="e938_role_safe_all_targets_smoke96_v2", minimum=6,
        readiness_context=screening_context(cwd),
    )
    assert result is None
    request = load_locked_readiness_request(store)
    assert request.action_id
    assert request.original.records == "suites/smoke/records.jsonl"
    assert request.training_ancestors[0].dataset_id == "wf_smoke_v2"


def test_insufficient_successor_retains_quality_failure(tmp_path):
    request = _request(tmp_path).model_copy(update={"minimum_unique_cases": 3})
    result = build_requested_successor(cwd=tmp_path, request=request)
    assert result["published"] is False
    assert not (tmp_path / "outputs/data/eval/successor").exists()
    paths = list((tmp_path / "outputs/runs").glob("screening-repair-*/successor/quality_report.json"))
    assert len(paths) == 1
    quality = json.loads(paths[0].read_text())
    assert quality["suite_counts"] == {"smoke": 2}
    assert quality["errors"] == ["insufficient_unique_cases"]
    assert quality["error_count"] == 1
    assert quality["certified"]["readiness"]["ready"] is False


@pytest.mark.parametrize("pending_action", [False, True])
def test_driver_dispatches_frozen_repair_without_receipt_count_progress(tmp_path, pending_action, monkeypatch):
    from scripts.run_autotrain_continuous import _self_heal_rebuild_screening_eval, _self_heal_rebuild_data
    from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
    from slm_training.autoresearch.storage import CampaignStore, autotrain_action_sha256
    from slm_training.harnesses.train_data.readiness import lock_readiness_request
    from tests.test_autoresearch.test_experiment_campaign import _manifest_payload

    request = _request(tmp_path)
    root = tmp_path / "campaigns"
    dispatch = _self_heal_rebuild_screening_eval
    if pending_action:
        from tests.test_scripts.test_run_autotrain_continuous import _screening_rebuild_handoff
        from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1

        path = _screening_rebuild_handoff(root, request.campaign_id)
        handoff = AutotrainCycleHandoffV1.model_validate_json(path.read_text())
        request = request.model_copy(update={"action_id": autotrain_action_sha256(handoff.actions[0])})
        dispatch = _self_heal_rebuild_data
    store = CampaignStore(request.campaign_id, root)
    manifest = ExperimentCampaignV1.model_validate(
        _manifest_payload(campaign_id=request.campaign_id, claim_class="fixture"))
    store.lock_experiment_campaign(manifest)
    lock_readiness_request(store, request, experiment_id=manifest.experiment_id)
    args = dict(cwd=tmp_path, root=root, loop_id="loop-1" if pending_action else "fixture",
                campaign_id=request.campaign_id)
    assert dispatch(**args) == ("rebuild_data" if pending_action else "screening_successor_ready")
    assert dispatch(**args) is None
    events = [row for row in store.verify_event_chain()
              if row["event_type"] == "screening_successor_ready"]
    assert len(events) == 1
    assert events[0]["detail"]["measurement_complete"] is False
    receipt_path = root / "loops" / args["loop_id"] / "action_receipts.jsonl"
    assert receipt_path.exists() is pending_action
    assert not (tmp_path / "src").exists()
    if pending_action:
        from scripts import run_autotrain_continuous as driver
        from scripts.autotrain_controller_repair import resolve_repaired_data

        resolved = resolve_repaired_data(cwd=tmp_path, root=root, loop_id="loop-1",
                                         train_version="original", eval_version="screening")
        assert resolved["eval_version"] == "successor"
        assert resolved["train_version"] == "original"
        with pytest.raises(ValueError, match="cross-lineage readiness"):
            resolve_repaired_data(cwd=tmp_path, root=root, loop_id="loop-1",
                                  train_version="independently-repaired-train", eval_version="screening")
        monkeypatch.chdir(tmp_path)
        assert driver._screening_suite_records("successor") == 2
        # Observe the actual owner input, not a second count implementation.
        observed = []
        count = driver._screening_suite_records
        monkeypatch.setattr(driver, "_screening_suite_records", lambda version=None:
                            observed.append(version) or count(version))
        driver._screening_n_report(eval_version=resolved["eval_version"])
        matrix = driver._matrix(campaign_id="next", evidence_snapshot_id="fixture",
            cites=["docs/a.md", "docs/b.md", "docs/c.md"], role_citations={},
            train_version=resolved["train_version"], eval_version=resolved["eval_version"],
            steps=8, cycle=1)
        assert observed == ["successor", "successor"]
        control = matrix["hypotheses"][0]["experiment"]["knobs"]
        assert control["train_version"] == "original" and control["eval_version"] == "successor"


def _accept_request(root, request):
    from scripts.autotrain_controller_repair import dispatch_data_actions
    from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1
    from slm_training.autoresearch.storage import CampaignStore, autotrain_action_sha256
    from slm_training.harnesses.train_data.readiness import lock_readiness_request
    from tests.test_autoresearch.test_experiment_campaign import _manifest_payload
    from tests.test_scripts.test_run_autotrain_continuous import _screening_rebuild_handoff

    campaigns = root / "campaigns"
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        _screening_rebuild_handoff(campaigns, request.campaign_id).read_text())
    request = request.model_copy(update={"action_id": autotrain_action_sha256(handoff.actions[0])})
    store = CampaignStore(request.campaign_id, campaigns)
    manifest = ExperimentCampaignV1.model_validate(
        _manifest_payload(campaign_id=request.campaign_id, claim_class="fixture"))
    store.lock_experiment_campaign(manifest)
    lock_readiness_request(store, request, experiment_id=manifest.experiment_id)
    assert dispatch_data_actions(cwd=root, root=campaigns, loop_id="loop-1",
                                 campaign_id=request.campaign_id) == "rebuild_data"
    return handoff


@pytest.mark.parametrize("fault", ["fork", "resolver", "original", "wrong_request"])
def test_successor_selection_rejects_ambiguity_and_identity_drift(tmp_path, monkeypatch, fault):
    from scripts.autotrain_controller_repair import resolve_repaired_data
    from slm_training.autoresearch.storage import CampaignStore

    request = _request(tmp_path)
    _accept_request(tmp_path, request)
    if fault == "fork":
        _accept_request(tmp_path, request.model_copy(update={"campaign_id": "fork", "successor_id": "fork"}))
    elif fault == "resolver":
        monkeypatch.setenv("SLM_DATA_ROOT", str(tmp_path / "different-store"))
    elif fault == "original":
        (tmp_path / request.original.directory / request.original.records).write_text("{broken\n")
    else:
        journal = CampaignStore("runtime", tmp_path / "campaigns/loops/loop-1")
        accepted = next(e for e in journal.verify_event_chain() if e["event_type"] == "data_successor_accepted")
        # Independently exercise the real event consumer with a mismatched request.
        from scripts.autotrain_controller_repair import _accepted_data_successor

        accepted["detail"]["request_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="selection identity"):
            _accepted_data_successor(accepted, root=tmp_path / "campaigns", loop_id="loop-1", cwd=tmp_path)
        return
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_repaired_data(cwd=tmp_path, root=tmp_path / "campaigns", loop_id="loop-1",
                              train_version="original", eval_version="screening")


def test_data_acceptance_reconciles_receipt_failure_without_rebuilding(tmp_path, monkeypatch):
    from scripts.autotrain_controller_repair import dispatch_data_actions, resolve_repaired_data
    from slm_training.autoresearch import storage

    request = _request(tmp_path)
    append = storage.append_autotrain_action_receipt
    monkeypatch.setattr(storage, "append_autotrain_action_receipt",
                        lambda *args: (_ for _ in ()).throw(OSError("receipt disk full")))
    with pytest.raises(OSError, match="receipt disk full"):
        _accept_request(tmp_path, request)
    destination = tmp_path / "outputs/data/eval/successor"
    committed = (destination / "records.jsonl").read_bytes()
    monkeypatch.setattr(storage, "append_autotrain_action_receipt", append)
    import slm_training.autoresearch.heal

    monkeypatch.setattr(slm_training.autoresearch.heal, "run_playbooks",
                        lambda **kwargs: pytest.fail("committed dataset was rebuilt"))
    root = tmp_path / "campaigns"
    assert dispatch_data_actions(cwd=tmp_path, root=root, loop_id="loop-1",
                                 campaign_id=request.campaign_id) == "rebuild_data"
    assert (destination / "records.jsonl").read_bytes() == committed
    journal = storage.CampaignStore("runtime", root / "loops/loop-1")
    assert sum(e["event_type"] == "data_successor_accepted" for e in journal.verify_event_chain()) == 1
    resolved = resolve_repaired_data(cwd=tmp_path, root=root, loop_id="loop-1",
                                     train_version="original", eval_version="screening")
    assert resolved["eval_version"] == "successor" and len(resolved["successions"]) == 1


@pytest.mark.parametrize("fault", ["legacy_file", "wrong_loop", "wrong_action"])
def test_data_receipt_migration_preserves_old_bytes_but_rejects_weak_proof(tmp_path, fault):
    from slm_training.autoresearch.schemas import AutotrainActionReceiptV1
    from slm_training.autoresearch.storage import append_autotrain_action_receipt, pending_autotrain_actions

    handoff = _accept_request(tmp_path, _request(tmp_path))
    root = tmp_path / "campaigns"
    path = root / "loops/loop-1/action_receipts.jsonl"
    payload = json.loads(path.read_text())
    if fault == "legacy_file":
        payload["evidence_uris"] = ["quality_report.json"]
        (root / handoff.campaign_id / "quality_report.json").write_text('{"ready":true}')
    elif fault == "wrong_loop":
        payload["loop_id"] = "foreign-loop"
    else:
        payload["action_sha256"] = "0" * 64
    # Emulate historical bytes, not an authorized current writer.
    path.write_text(json.dumps(payload) + "\n")
    before = path.read_bytes()
    assert pending_autotrain_actions(root, handoff)
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        append_autotrain_action_receipt(root, AutotrainActionReceiptV1.model_validate(payload))
    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", ["receipt", "acceptance"])
def test_data_acceptance_cannot_publish_after_lease_revocation(tmp_path, operation):
    from slm_training.autoresearch.heal.data_predicate_receipt import complete_data_action
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime, StaleLease
    from slm_training.autoresearch.schemas import AutotrainActionReceiptV1
    from slm_training.autoresearch.storage import CampaignStore, append_autotrain_action_receipt
    from slm_training.harness_core.checkpoint_publication import champion_publication_scope
    from tests.test_autoresearch.test_activity_runtime import spec

    handoff = _accept_request(tmp_path, _request(tmp_path))
    root = tmp_path / "campaigns"
    receipts = root / "loops/loop-1/action_receipts.jsonl"
    before = receipts.read_bytes()
    receipt = AutotrainActionReceiptV1.model_validate_json(before)
    with ActivityRuntime(CampaignStore("runtime", root / "loops/loop-1"), controller_clock=lambda: 100.) as runtime:
        runtime.register(spec())
        lease = runtime.claim_next(capabilities={"local_process"})
        with champion_publication_scope(runtime, lease, loop_dir=root / "loops/loop-1"):
            runtime.cancel(lease.activity_id, reason="revoked data publisher")
            with pytest.raises(StaleLease):
                if operation == "receipt":
                    append_autotrain_action_receipt(root, receipt)
                else:
                    complete_data_action(cwd=tmp_path, root=root, handoff=handoff, action_index=0)
    assert receipts.read_bytes() == before


@pytest.mark.parametrize("legacy_fence", [False, True])
def test_acceptance_reconciles_under_new_owner_without_rewriting_old_fence(tmp_path, monkeypatch, legacy_fence):
    from slm_training.autoresearch.heal.data_predicate_receipt import complete_data_action
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.checkpoint_publication import champion_publication_scope
    from tests.test_autoresearch.test_activity_runtime import spec

    append = CampaignStore.append_event
    def legacy_event(self, event_type, **kwargs):
        if legacy_fence and event_type == "data_successor_accepted":
            kwargs["detail"].pop("fence", None)
        return append(self, event_type, **kwargs)
    with monkeypatch.context() as old_writer:
        old_writer.setattr(CampaignStore, "append_event", legacy_event)
        handoff = _accept_request(tmp_path, _request(tmp_path))
    root = tmp_path / "campaigns"
    journal = CampaignStore("runtime", root / "loops/loop-1")
    accepted = [e for e in journal.verify_event_chain() if e["event_type"] == "data_successor_accepted"]
    with ActivityRuntime(journal, controller_clock=lambda: 100.) as runtime:
        runtime.register(spec())
        lease = runtime.claim_next(capabilities={"local_process"})
        with champion_publication_scope(runtime, lease, loop_dir=root / "loops/loop-1"):
            assert complete_data_action(cwd=tmp_path, root=root, handoff=handoff, action_index=0)
    assert [e for e in journal.verify_event_chain() if e["event_type"] == "data_successor_accepted"] == accepted
