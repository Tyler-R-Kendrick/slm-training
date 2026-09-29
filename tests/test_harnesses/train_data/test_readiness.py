"""Real admission, leakage, immutable correction and trainer differential tests."""
import pytest

from tests.casefiles import case_values

from slm_training.data.readiness_contract import DataReadinessRequest, LearnedTableInput, file_digest
from slm_training.data.readiness_lineage import leakage_findings
from slm_training.harnesses.train_data.readiness import check_readiness
from slm_training.harnesses.train_data.repair import publish_repair
from tests.test_harnesses.train_data.readiness_fixtures import (
    certified_screening_fixture, record, request_fixture, screening_context, stage_rows, write_snapshot,
)


@pytest.mark.parametrize("absolute_paths", [False, True])
def test_screening_request_factory_binds_published_inputs_and_sampling_authority(tmp_path, absolute_paths):
    from slm_training.harnesses.train_data.readiness import build_screening_request

    root = certified_screening_fixture(tmp_path, absolute_paths=absolute_paths)
    request = build_screening_request(
        root=root, campaign_id="screening-campaign", action_id="screening-action",
        train_version="wf_smoke_v2",
        eval_version="e938_role_safe_all_targets_smoke96_v2", minimum=6,
        context=screening_context(root),
    )
    assert request.original.records == "suites/smoke/records.jsonl"
    assert request.training_ancestors[0].dataset_id == "wf_smoke_v2"
    assert request.initialization == "scratch"
    assert request.lineage_root is None and request.starting_run_id is None
    assert len(request.scored_suites) == 5
    assert request.generation_knobs["source"] == "certified"
    assert request.trainer_config["context_backend"] == "scratch"


@pytest.mark.parametrize("fault", ["missing", "partial", "suite", "train", "parent", "table", "config", "ranker", "minimum"])
def test_screening_factory_refuses_guessed_or_incomplete_context(tmp_path, fault):
    from slm_training.harnesses.train_data.readiness import build_screening_request

    root = certified_screening_fixture(tmp_path)
    context = screening_context(root)
    minimum = 6
    if fault == "missing":
        context = None
    elif fault == "partial":
        context.pop("learned_tables")
    elif fault == "suite":
        context["scored_suites"] = context["scored_suites"][:1]
    elif fault == "train":
        context["training_ancestors"] = []
    elif fault == "parent":
        context.update(initialization="parent", lineage_root="missing-lineage", starting_run_id="missing-run")
    elif fault == "table":
        context["learned_tables"] = [{"path": "missing-table", "sha256": "0" * 64,
                                      "training_sources": context["training_ancestors"]}]
    elif fault == "config":
        context["trainer_config"] = {"not_a_trainer_setting": True}
    elif fault == "ranker":
        context["trainer_config"]["speculative_rank"] = "ngram"
    else:
        minimum = True
    with pytest.raises((OSError, TypeError, ValueError)):
        build_screening_request(root=root, campaign_id="factory", action_id="action",
            train_version="wf_smoke_v2", eval_version="e938_role_safe_all_targets_smoke96_v2",
            minimum=minimum, context=context)


def test_factory_preserves_explicit_config_extra_suites_and_table_provenance(tmp_path, monkeypatch):
    from slm_training.harnesses.train_data import readiness

    request = request_fixture(tmp_path, bad=False)
    original = request.scored_suites[0]
    extra = write_snapshot(tmp_path, "other-suite", [record(id="external")], kind="eval")
    table = tmp_path / "table.json"
    table.write_text("{}")
    context = {"trainer_config": request.trainer_config,
               "scored_suites": [original, extra], "training_ancestors": [request.original],
               "learned_tables": [LearnedTableInput(path="table.json", sha256=file_digest(table),
                    training_sources=[request.original])], "initialization": "scratch",
               "lineage_root": None, "starting_run_id": None,
               "preprocessing_identity": request.preprocessing_identity}
    monkeypatch.setattr(readiness, "_screening_snapshots", lambda *args:
                        (tmp_path / original.directory, original, request.original, [original]))
    monkeypatch.setattr(readiness, "_screening_sampling", lambda *args: ({"fixture": True}, "versions"))
    built = readiness.build_screening_request(root=tmp_path, campaign_id="factory", action_id="a",
        train_version="original", eval_version="public-eval", minimum=2, context=context)
    assert built.trainer_config["d_model"] == 16
    assert built.scored_suites == [original, extra]
    assert built.learned_tables[0].training_sources == [request.original]
    assert built.preprocessing_identity == request.preprocessing_identity


@pytest.mark.parametrize("wrong_source", [False, True])
def test_active_ranker_binds_declared_table_source(tmp_path, wrong_source):
    import json
    from slm_training.dsl.grammar.fastpath.speculative_rank import build_ngram_table
    from slm_training.harnesses.train_data.readiness import _check_ranker_sources

    request = request_fixture(tmp_path, bad=False)
    payload = build_ngram_table([[1, 2, 3]], order=2).to_dict()
    payload["source"] = {"dataset_id": request.original.dataset_id,
                         "records_sha256": "0" * 64 if wrong_source else request.original.records_sha256}
    path = tmp_path / "table.json"
    path.write_text(json.dumps(payload))
    table = LearnedTableInput(path="table.json", sha256=file_digest(path), training_sources=[request.original])
    request = request.model_copy(update={"learned_tables": [table], "trainer_config": {
        **request.trainer_config, "speculative_rank": "ngram", "speculative_rank_table": "table.json",
        "speculative_rank_margin": 0.0}})
    if wrong_source:
        with pytest.raises(ValueError, match="ranker training source identity"):
            _check_ranker_sources(tmp_path, request)
    else:
        _check_ranker_sources(tmp_path, request)


def test_readiness_rechecks_ranker_provenance_for_direct_requests(tmp_path):
    request = request_fixture(tmp_path, bad=False)
    request = request.model_copy(update={"trainer_config": {
        **request.trainer_config, "speculative_rank": "ngram"}})
    report = check_readiness(request, root=tmp_path)
    assert not report["ready"]
    assert "missing configured ranker learned-table provenance" in str(report["errors"])


def test_actual_trainer_fault_then_immutable_same_count_repair(tmp_path):
    request = request_fixture(tmp_path)
    before = check_readiness(request, root=tmp_path)
    assert not before["ready"] and before["rejected"]
    original = tmp_path / request.original.directory / "records.jsonl"
    old_bytes = original.read_bytes()
    result = publish_repair(request, root=tmp_path,
                            staged=stage_rows(tmp_path, request, [record()]))
    assert result["published"] and result["evidence"]["ready"], result
    evidence = result["evidence"]
    assert evidence["usable_unique_cases"] == evidence["usable_unique_families"] == 1
    assert evidence["preparation"]["records_prepared"] == 1
    assert evidence["preparation"]["forwards"] == 0
    assert original.read_bytes() == old_bytes
    assert evidence["input_family_support"] == {"train-family": 1}
    assert (tmp_path / result["candidate"]["directory"] / "data_readiness.json").exists()


def test_same_count_correction_does_not_satisfy_volume(tmp_path):
    request = request_fixture(tmp_path, minimum=2)
    result = publish_repair(request, root=tmp_path,
                            staged=stage_rows(tmp_path, request, [record()]))
    assert not result["published"]
    assert "insufficient_unique_cases" in result["evidence"]["errors"]
    assert result["evidence"]["rejected"] == []


@pytest.mark.parametrize("bad_kind", ["malformed", "duplicate", "wrong_target", "wrong_split"])
def test_invalid_or_unrelated_rows_cannot_heal(tmp_path, bad_kind):
    request = request_fixture(tmp_path)
    rows = [record()]
    if bad_kind == "duplicate":
        rows.append(record(id="renamed"))
    elif bad_kind == "wrong_target":
        rows = [record(openui='":slot_0"', target_kind="lexical")]
    elif bad_kind == "wrong_split":
        rows = [record(split="held_out")]
    stage = stage_rows(tmp_path, request, rows)
    if bad_kind == "malformed":
        with (stage / "records.jsonl").open("a") as stream:
            stream.write("{broken\n")
    result = publish_repair(request, root=tmp_path, staged=stage)
    assert not result["published"]
    assert result["evidence"]["rejected"]
    assert (stage / "readiness_rejected.jsonl").read_text()


@pytest.mark.parametrize("mode", ["root", "program", "prompt", "structure", "parent"])
def test_different_ids_and_hashes_do_not_establish_disjointness(mode):
    train = record()
    eval_row = record(id="new-id", prompt="Different request", openui="root = Stack([])",
                      placeholders=[], meta={"root_parent_id": "new-root"})
    if mode == "root":
        eval_row.meta = {"root_parent_id": "train-family"}
    elif mode == "parent":
        eval_row.meta = {"parent_id": "train-case"}
    elif mode == "program":
        eval_row.openui = train.openui
    elif mode == "prompt":
        eval_row.prompt = train.prompt
    else:
        eval_row.openui = train.openui.replace("c1", "c9")
    assert leakage_findings([train], {"scored": [eval_row]})


def test_ancestor_and_train_only_ngram_sources_are_scored(tmp_path):
    request = request_fixture(tmp_path, bad=False)
    leaked = record(id="old-id", prompt="Empty composition", openui="root = Stack([])",
                    placeholders=[], meta={"root_parent_id": "ancestor"})
    ancestor = write_snapshot(tmp_path, "ancestor", [leaked])
    with_ancestor = request.model_copy(update={"training_ancestors": [ancestor]})
    assert check_readiness(with_ancestor, root=tmp_path)["leakage"]
    table = tmp_path / "ngram.json"
    table.write_text("{}")
    learned = LearnedTableInput(path="ngram.json", sha256=file_digest(table),
                                training_sources=[ancestor])
    with_table = request.model_copy(update={"learned_tables": [learned]})
    assert check_readiness(with_table, root=tmp_path)["leakage"]


def test_public_suite_cannot_claim_sealed_and_missing_input_fails_closed(tmp_path):
    request = request_fixture(tmp_path, bad=False)
    suite = request.scored_suites[0]
    sealed = suite.model_copy(update={"exposure": "access_controlled"})
    report = check_readiness(request.model_copy(update={"scored_suites": [sealed]}), root=tmp_path)
    assert not report["ready"] and "sealed isolation" in str(report["errors"])
    (tmp_path / suite.directory / suite.records).unlink()
    assert not check_readiness(request, root=tmp_path)["ready"]


@pytest.mark.parametrize("field,value", case_values(__file__, "test_strict_request_rejects_invalid_contract"))
def test_strict_request_rejects_invalid_contract(tmp_path, field, value):
    raw = request_fixture(tmp_path).model_dump()
    raw[field] = value
    with pytest.raises(ValueError):
        DataReadinessRequest.model_validate(raw)


def test_admission_uses_real_trainer_owner(tmp_path, monkeypatch):
    from slm_training.data import record_admission
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
    from slm_training.harnesses.test_data.certified import _assert_certified_role_safe

    request = request_fixture(tmp_path, bad=False)
    def refuses(_):
        raise ValueError("new canonical preparation requirement")
    monkeypatch.setattr(record_admission, "assert_training_record", refuses)
    for consumer in (_assert_certified_role_safe,
                     lambda row: TwoTowerModel.from_records([row], TwoTowerConfig())):
        with pytest.raises(ValueError, match="new canonical"):
            consumer(record())
    # Readiness imports its guard at module load but calls the actual trainer
    # too: a future added trainer requirement cannot produce a ready report.
    assert not check_readiness(request, root=tmp_path)["ready"]


def test_publication_failure_preserves_original_and_no_successor(tmp_path, monkeypatch):
    from slm_training.data import publication
    request = request_fixture(tmp_path)
    original = (tmp_path / request.original.directory / "records.jsonl").read_bytes()
    def disk_full(_):
        raise OSError("simulated ENOSPC")
    monkeypatch.setattr(publication, "_fsync_directory", disk_full)
    with pytest.raises(OSError, match="ENOSPC"):
        publish_repair(request, root=tmp_path, staged=stage_rows(tmp_path, request, [record()]))
    assert not (tmp_path / "outputs/data/train/successor").exists()
    assert (tmp_path / request.original.directory / "records.jsonl").read_bytes() == original


def test_existing_successor_is_never_overwritten(tmp_path):
    request = request_fixture(tmp_path)
    stage = stage_rows(tmp_path, request, [record()])
    first = publish_repair(request, root=tmp_path, staged=stage)
    assert first["published"], first
    dest = tmp_path / first["candidate"]["directory"]
    previous = (dest / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        publish_repair(request, root=tmp_path, staged=stage)
    assert (dest / "manifest.json").read_bytes() == previous


@pytest.mark.parametrize("preprocessing", ["fixture-certified-v1", "préparation-符号-v1"])
def test_locked_request_producer_reaches_actual_repair_consumer(tmp_path, preprocessing):
    from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
    from slm_training.autoresearch.heal.playbooks.data_rebuild import load_readiness_request
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harnesses.train_data.readiness import lock_readiness_request
    from tests.test_autoresearch.test_experiment_campaign import _manifest_payload

    request = request_fixture(tmp_path).model_copy(update={"preprocessing_identity": preprocessing})
    store = CampaignStore(request.campaign_id, tmp_path / "campaigns")
    manifest = ExperimentCampaignV1.model_validate(
        _manifest_payload(campaign_id=request.campaign_id, claim_class="fixture"))
    with pytest.raises((FileNotFoundError, ValueError)):
        lock_readiness_request(store, request, experiment_id=manifest.experiment_id)
    store.lock_experiment_campaign(manifest)
    first = lock_readiness_request(store, request, experiment_id=manifest.experiment_id)
    assert first == lock_readiness_request(store, request, experiment_id=manifest.experiment_id)
    consumed = load_readiness_request(first, root=tmp_path / "campaigns",
                                      campaign_id=request.campaign_id)
    assert consumed == request
    assert consumed.sha256 == first["data_readiness_request_sha256"]
    # Actual pending actions carry no worker-authored JSON or latest-request file.
    assert load_readiness_request({}, root=tmp_path / "campaigns",
                                  campaign_id=request.campaign_id) == request
    events = store.verify_event_chain()
    assert sum(e["event_type"] == "data_readiness_request_locked" for e in events) == 1
    changed = request.model_copy(update={"minimum_unique_cases": 2})
    with pytest.raises(ValueError, match="idempotency"):
        lock_readiness_request(store, changed, experiment_id=manifest.experiment_id)
    with pytest.raises(ValueError, match="wrong campaign"):
        lock_readiness_request(CampaignStore("other", tmp_path), request,
                               experiment_id=manifest.experiment_id)


def test_actual_training_successor_dispatch_and_recovery_are_action_scoped(tmp_path):
    from scripts import run_autotrain_continuous as driver
    from scripts.autotrain_controller_repair import (
        _dispatch_data_request, resolve_repaired_data, verified_training_successors,
    )
    from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
    from slm_training.autoresearch.storage import CampaignStore, autotrain_action_sha256, pending_autotrain_actions
    from slm_training.harnesses.train_data.readiness import lock_readiness_request
    from tests.test_autoresearch.test_experiment_campaign import _manifest_payload
    from tests.test_scripts.test_run_autotrain_continuous import _screening_rebuild_handoff

    request = request_fixture(tmp_path)
    root = tmp_path / "campaigns"
    path = _screening_rebuild_handoff(root, request.campaign_id)
    handoff = driver.AutotrainCycleHandoffV1.model_validate_json(path.read_text())
    request = request.model_copy(update={"action_id": autotrain_action_sha256(handoff.actions[0])})
    store = CampaignStore(request.campaign_id, root)
    manifest = ExperimentCampaignV1.model_validate(
        _manifest_payload(campaign_id=request.campaign_id, claim_class="fixture"))
    store.lock_experiment_campaign(manifest)
    # Two real requests share a campaign: only the exact pending action may run.
    unrelated = request.model_copy(update={"action_id": "unrelated-action", "successor_id": "other"})
    lock_readiness_request(store, unrelated, experiment_id=manifest.experiment_id)
    lock_readiness_request(store, request, experiment_id=manifest.experiment_id)
    result = publish_repair(request, root=tmp_path,
                            staged=stage_rows(tmp_path, request, [record()]))
    assert result["published"] and result["evidence"]["preparation"]["forwards"] == 0
    args = dict(cwd=tmp_path, root=root, loop_id="loop-1", campaign_id=request.campaign_id)
    assert _dispatch_data_request(**args, blocker={"data_action_id": request.action_id}) == "data_successor_ready"
    # Readiness is a producer observation, not controller acceptance or selection.
    assert verified_training_successors(**args) == []
    assert driver._self_heal_rebuild_data(**args) == "rebuild_data"
    assert driver._self_heal_rebuild_data(**args) is None
    assert pending_autotrain_actions(root, handoff) == ()
    assert not (tmp_path / "outputs/data/train/other").exists()
    candidates = verified_training_successors(**args)
    assert [ref.dataset_id for ref in candidates] == ["successor"]
    assert not verified_training_successors(**{**args, "loop_id": "foreign-loop"})
    selection = dict(cwd=tmp_path, root=root, loop_id="loop-1",
                     train_version="original", eval_version="public-eval")
    resolved = resolve_repaired_data(**selection)
    assert resolved["train_version"] == "successor"
    assert resolved["eval_version"] == "public-eval"
    assert resolved["successions"][0]["request_sha256"] == request.sha256
    for intent in ("confirm", "promote", "retry_measurement"):
        frozen = resolve_repaired_data(**selection, intent=intent)
        assert frozen["train_version"] == "original" and not frozen["successions"]
    old_dynamic, old_loaded = driver._DYNAMIC_THRASH_ARMS, driver._DYNAMIC_THRASH_LOADED_FOR
    try:
        assert driver._recover_heal_resume_arm(
            root, "loop-1", cwd=tmp_path, predecessor_campaign_id=request.campaign_id)
        matches = [extras for slug, _, extras in driver._DYNAMIC_THRASH_ARMS
                   if slug == driver._HEAL_RESUME_SLUG]
        assert matches[0]["train_version"] == "successor"
    finally:
        driver._DYNAMIC_THRASH_ARMS, driver._DYNAMIC_THRASH_LOADED_FOR = old_dynamic, old_loaded
    # A historical ready event cannot hide current input corruption.
    (tmp_path / candidates[0].directory / candidates[0].records).write_text("{broken\n")
    assert verified_training_successors(**args) == []
    assert pending_autotrain_actions(root, handoff)  # Historical recovery is not current proof.
    with pytest.raises(ValueError, match="current data predicate"):
        resolve_repaired_data(**selection)


def test_lost_arm_recovery_refuses_two_accepted_successors(tmp_path, monkeypatch):
    from scripts import run_autotrain_continuous as driver
    from slm_training.autoresearch.storage import autotrain_action_sha256
    from tests.test_autoresearch.test_screening_data_repair import _accept_request
    from tests.test_scripts.test_run_autotrain_continuous import _screening_rebuild_handoff

    request = request_fixture(tmp_path)
    for campaign_id, version in (("first", "successor-a"), ("second", "successor-b")):
        path = _screening_rebuild_handoff(tmp_path / "campaigns", campaign_id)
        handoff = driver.AutotrainCycleHandoffV1.model_validate_json(path.read_text())
        selected = request.model_copy(update={"campaign_id": campaign_id, "successor_id": version,
                                             "action_id": autotrain_action_sha256(handoff.actions[0])})
        assert publish_repair(selected, root=tmp_path,
                              staged=stage_rows(tmp_path, selected, [record()]))["published"]
        _accept_request(tmp_path, selected)
    monkeypatch.setattr(driver, "_lineage_campaign_ids", lambda *_: ["first", "second"])
    monkeypatch.setattr(driver, "_DYNAMIC_THRASH_ARMS", [])
    monkeypatch.setattr(driver, "_DYNAMIC_THRASH_LOADED_FOR", None)
    with pytest.raises(ValueError, match="ambiguous accepted training successors"):
        driver._recover_heal_resume_arm(tmp_path / "campaigns", "loop-1", cwd=tmp_path,
                                        predecessor_campaign_id="second")
    assert not driver._DYNAMIC_THRASH_ARMS
    assert not driver._dynamic_thrash_arms_path(tmp_path / "campaigns", "loop-1").exists()
