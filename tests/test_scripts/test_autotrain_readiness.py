"""Actual compiler -> CLI resolver -> readiness activity, without model work."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from scripts.autotrain_readiness import readiness_context_from_commands
from slm_training.autoresearch.engine import compile_commands
from slm_training.autoresearch.schemas import ExperimentKnobs
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.readiness_contract import file_digest, load_locked_readiness_request
from slm_training.harnesses.train_data.readiness import bootstrap_screening_request
from tests.test_autoresearch.test_harness import campaign, experiment
from tests.test_harnesses.train_data.readiness_fixtures import certified_screening_fixture, record, write_snapshot


@pytest.fixture
def compiled(tmp_path, monkeypatch):
    certified_screening_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLM_DATA_ROOT", raising=False)
    source = Path(__file__).resolve().parents[2]
    monkeypatch.setenv(
        "PYTHONPATH",
        f"{source}:{source / 'src'}:{source / 'outputs/runs/autonomy-validation/dependencies'}",
    )
    knobs = ExperimentKnobs(steps=2, batch_size=2, lr=0.003, seed=7,
        context_backend="scratch", sync_checkpoints=False, train_version="wf_smoke_v2",
        eval_version="e938_role_safe_all_targets_smoke96_v2", eval_suites="smoke", eval_limit=6)
    commands = compile_commands(campaign(), experiment(knobs=knobs))
    return {"cwd": tmp_path, "train_command": next(cmd for cmd in commands if cmd[2] == "scripts.train_model"),
            "eval_commands": [cmd for cmd in commands if cmd[2] == "scripts.evaluate_model"]}


def test_actual_compiler_to_bootstrap_locks_config_and_full_suite_dependencies(compiled):
    from scripts.train_model import resolve_config
    from slm_training.harnesses.model_build.feature_flags import resolve
    from slm_training.harnesses.model_build.factory import _twotower_config_from_build

    context = readiness_context_from_commands(**compiled)
    resolved, _ = resolve(resolve_config(compiled["train_command"][3:]), phase="training")
    from pydantic_core import to_jsonable_python
    assert context["trainer_config"] == to_jsonable_python(asdict(_twotower_config_from_build(resolved)))
    assert len(context["scored_suites"]) == 5  # eval_limit is not the leakage scope.
    assert context["initialization"] == "scratch" and context["learned_tables"] == []
    assert context["training_ancestors"][0]["dataset_id"] == "wf_smoke_v2"
    root = compiled["cwd"] / "campaigns"
    locked = bootstrap_screening_request(cwd=compiled["cwd"], root=root, loop_id="bootstrap",
        train_version="wf_smoke_v2", eval_version="e938_role_safe_all_targets_smoke96_v2", minimum=6, context=context)
    store = CampaignStore(locked["campaign_id"], root)
    assert load_locked_readiness_request(store).trainer_config == context["trainer_config"]
    assert all(event.get("experiment_id") is None for event in store.verify_event_chain())
    assert not (compiled["cwd"] / "outputs/runs").exists()
    assert readiness_context_from_commands(**compiled) == context


def test_actual_recipe_changes_context_identity(compiled):
    old = readiness_context_from_commands(**compiled)
    command = compiled["train_command"]
    command[command.index("--lr") + 1] = "0.002"
    new = readiness_context_from_commands(**compiled)
    assert old["preprocessing_identity"] != new["preprocessing_identity"]


def test_zero_case_snapshot_can_lock_readiness_not_scientific_success(compiled):
    from slm_training.data.store import write_common_manifest

    directory = compiled["cwd"] / "outputs/data/eval/e938_role_safe_all_targets_smoke96_v2"
    # Disposable fixture only: republish its internally consistent empty input.
    (directory / "suites/smoke/records.jsonl").write_text("")
    write_common_manifest(directory, kind="eval", dataset_id=directory.name, immutable=False)
    context = readiness_context_from_commands(**compiled)
    assert len(context["scored_suites"]) == 5
    locked = bootstrap_screening_request(cwd=compiled["cwd"], root=compiled["cwd"] / "campaigns",
        loop_id="empty", train_version="wf_smoke_v2", eval_version=directory.name, minimum=6, context=context)
    request = load_locked_readiness_request(CampaignStore(locked["campaign_id"], compiled["cwd"] / "campaigns"))
    assert request.original.records_sha256 == file_digest(directory / "suites/smoke/records.jsonl")


@pytest.mark.parametrize("fault", ["no_eval", "wrong_command", "wrong_cwd", "parent_missing", "corrupt_suite"])
def test_missing_actual_inputs_fail_before_lock(compiled, fault):
    if fault == "no_eval":
        compiled["eval_commands"] = []
    elif fault == "wrong_command":
        compiled["train_command"][2] = "scripts.fake_train"
    elif fault == "wrong_cwd":
        compiled["cwd"] = compiled["cwd"] / "wrong"
    elif fault == "parent_missing":
        compiled["train_command"] += ["--initialize-from", "unverified.pt"]
    else:
        directory = compiled["cwd"] / "outputs/data/eval/e938_role_safe_all_targets_smoke96_v2"
        (directory / "suites/ood/records.jsonl").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        readiness_context_from_commands(**compiled)


def test_registered_checkpoint_ancestry_uses_dataset_fingerprint_not_raw_record_sha(compiled):
    from slm_training.harness_core.lineage.data_cycle import register_dataset_snapshot
    from slm_training.harness_core.lineage.store import LineageStore
    from slm_training.harness_core.checkpoint_bundle import stage_checkpoint_bundle
    from tests.test_lineage.test_lineage import run_manifest

    root = compiled["cwd"]
    ancestor = write_snapshot(root, "ancestor", [record(id="ancestor", prompt="Earlier task")])
    manifest_path = root / ancestor.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    from slm_training.harnesses.train_data.pipeline import _content_fingerprint
    manifest["content_fingerprint"] = _content_fingerprint([record(id="ancestor", prompt="Earlier task")])
    manifest_path.write_text(json.dumps(manifest))
    store = LineageStore(root / "lineage")
    snapshot, _, _ = register_dataset_snapshot(store, dataset_dir=root / ancestor.directory, kind="train")
    assert snapshot.records_sha != ancestor.records_sha256
    store.create_run(replace(run_manifest("parent"), data_snapshot_sha=snapshot.sha))
    checkpoint = root / "checkpoint/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"opaque checkpoint fixture; never loaded")
    checkpoint.with_suffix(".meta.json").write_text(json.dumps({"kind": "twotower", "output_contract_version": 2}))
    checkpoint.with_suffix(".tokenizer.json").write_text("{}")
    digest = stage_checkpoint_bundle(root / "bundled", checkpoint, {"role": "trial_cursor", "run_id": "parent"})
    path = root / "bundled/bundles" / digest / "last.pt"
    compiled["train_command"] += ["--initialize-from", str(path)]
    context = readiness_context_from_commands(**compiled, lineage_root="lineage", starting_run_id="parent")
    assert context["initialization"] == "parent"
    assert {ref["dataset_id"] for ref in context["training_ancestors"]} == {"ancestor", "wf_smoke_v2"}
    with pytest.raises(ValueError, match="checkpoint/run"):
        readiness_context_from_commands(**compiled, lineage_root="lineage", starting_run_id="unrelated")


@pytest.mark.parametrize("configured", [False, True])
def test_source_verification_is_wait_not_retry(configured):
    from scripts.autotrain_controller_repair import repair_payload_outcome
    from slm_training.harness_core.activity_contract import ActivityOutcome

    dependency = {"verification_identity": "a" * 64, "grant_required": True} if configured else {}
    outcome = repair_payload_outcome({"agent_repairs": [{"status": "waiting_verification",
                                                          "verification_dependency": dependency}]})
    assert outcome == (ActivityOutcome.DEPENDENCY if configured else ActivityOutcome.CAPABILITY)


def test_joint_actual_configs_each_prepare_once_and_reject_incompatible_parent(compiled):
    from scripts.autotrain_readiness import merge_readiness_contexts
    from slm_training.harnesses.train_data.readiness import build_screening_request, check_readiness

    first = readiness_context_from_commands(**compiled)
    compiled["train_command"] += ["--design-md-dropout", "0.2"]
    second = readiness_context_from_commands(**compiled)
    joint = merge_readiness_contexts([first, second, first])
    assert len(joint["additional_trainer_configs"]) == 1
    assert len(joint["scored_suites"]) == 5
    request = build_screening_request(root=compiled["cwd"], campaign_id="joint", action_id="joint",
        train_version="wf_smoke_v2", eval_version="e938_role_safe_all_targets_smoke96_v2", minimum=1, context=joint)
    report = check_readiness(request, root=compiled["cwd"])
    assert report["ready"], report["errors"]
    primary = report["preparation"]
    additional = primary["additional_preparations"]
    assert len(additional) == 1 and primary["forwards"] == additional[0]["forwards"] == 0
    assert primary["resolved_config_digest"] != additional[0]["resolved_config_digest"]
    with pytest.raises(ValueError, match="incompatible arm initialization"):
        merge_readiness_contexts([first, {**second, "starting_checkpoint_bundle": "a" * 64}])


def test_joint_readiness_never_ignores_second_unsupported_configuration(compiled):
    from scripts.autotrain_readiness import merge_readiness_contexts
    from slm_training.harnesses.train_data.readiness import build_screening_request, check_readiness

    first = readiness_context_from_commands(**compiled)
    compiled["train_command"] += ["--context-backend", "hf"]
    second = readiness_context_from_commands(**compiled)
    joint = merge_readiness_contexts([first, second])
    request = build_screening_request(root=compiled["cwd"], campaign_id="joint", action_id="joint",
        train_version="wf_smoke_v2", eval_version="e938_role_safe_all_targets_smoke96_v2", minimum=1, context=joint)
    report = check_readiness(request, root=compiled["cwd"])
    assert not report["ready"]
    assert "local readiness requires scratch backends" in str(report["errors"])


def test_legacy_single_config_request_digest_does_not_gain_obligations(tmp_path):
    from slm_training.data.readiness_contract import DataReadinessRequest, evidence_digest
    from tests.test_harnesses.train_data.readiness_fixtures import request_fixture

    old = request_fixture(tmp_path).model_dump(mode="json")
    old.pop("additional_trainer_configs")
    old.pop("starting_checkpoint_bundle")
    loaded = DataReadinessRequest.model_validate(old)
    assert loaded.sha256 == evidence_digest(old)
    assert loaded.additional_trainer_configs == []
    with pytest.raises(ValueError, match="explicit nonempty"):
        DataReadinessRequest.model_validate({**old, "additional_trainer_configs": [{}]})
    # Exercise the actual historical event reader, not only the model helper.
    from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
    from tests.test_autoresearch.test_experiment_campaign import _manifest_payload
    store = CampaignStore(loaded.campaign_id, tmp_path / "campaigns")
    manifest = ExperimentCampaignV1.model_validate(_manifest_payload(
        campaign_id=loaded.campaign_id, claim_class="fixture"))
    lock = store.lock_experiment_campaign(manifest)
    artifact = store.write_artifact("data_readiness_requests", old)
    store.append_event("data_readiness_request_locked", experiment_id=manifest.experiment_id,
        artifact_sha256=artifact.stem, detail={"request_sha256": evidence_digest(old),
                                             "manifest_sha256": lock.manifest_sha256})
    assert load_locked_readiness_request(store) == loaded


def test_actual_ranker_dependency_is_loaded_and_binds_its_train_source(compiled):
    from scripts.autotrain_readiness import _learned_inputs
    from scripts.train_model import resolve_config
    from slm_training.harnesses.model_build.factory import _twotower_config_from_build
    from slm_training.dsl.grammar.fastpath.speculative_rank import build_ngram_table
    from slm_training.data.readiness_contract import file_digest

    table = build_ngram_table([[1, 2, 3]]).to_dict()
    table["source"] = {"dataset_id": "wf_smoke_v2", "records_sha256": file_digest(
        compiled["cwd"] / "outputs/data/train/wf_smoke_v2/records.jsonl")}
    path = compiled["cwd"] / "ranker.json"
    path.write_text(json.dumps(table))
    # Ranker is currently exposed by ModelBuildConfig, not the train CLI.
    config = replace(resolve_config(compiled["train_command"][3:]),
                     speculative_rank="ngram", speculative_rank_table="ranker.json")
    trainer = asdict(_twotower_config_from_build(config))
    tables = _learned_inputs(compiled["cwd"], trainer)
    assert tables[0].sha256 == file_digest(path)
    assert tables[0].training_sources[0].dataset_id == "wf_smoke_v2"
    table["source"]["records_sha256"] = "0" * 64
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="ranker training source identity"):
        _learned_inputs(compiled["cwd"], trainer)


def _matrix_fixture(compiled):
    from scripts.train_model import resolve_config

    config = resolve_config(compiled["train_command"][3:])
    knobs = ExperimentKnobs(steps=config.steps, batch_size=2, context_backend="scratch",
        train_version="wf_smoke_v2", eval_version="e938_role_safe_all_targets_smoke96_v2",
        sync_checkpoints=False, eval_suites="smoke", eval_limit=6)
    spec = campaign()
    CampaignStore(spec.campaign_id, compiled["cwd"] / "campaigns").initialize(spec)
    arms = [experiment(experiment_id="control", knobs=knobs), experiment(experiment_id="candidate",
             knobs=knobs.model_copy(update={"design_md_dropout": 0.2}))]
    return {"campaign_id": spec.campaign_id, "hypotheses": [{"experiment": arm.model_dump(mode="json")} for arm in arms],
            "recommended_experiment_id": "candidate"}


@pytest.mark.parametrize("unsupported", [False, True])
def test_sufficient_counts_still_prepare_selected_matrix(compiled, monkeypatch, unsupported):
    from types import SimpleNamespace
    from scripts import run_autotrain_continuous as driver
    from scripts.autotrain_pending import resolve_screening_matrix

    matrix = _matrix_fixture(compiled)
    if unsupported:
        matrix["hypotheses"][1]["experiment"]["knobs"]["context_backend"] = "hf"
    version = "e938_role_safe_all_targets_smoke96_v2"
    resolved = {"eval_version": version, "successions": []}
    def unexpected(**kwargs):
        pytest.fail("an unchanged dataset must preserve the selected matrix")
    monkeypatch.setattr(driver, "_matrix", unexpected)
    result, pending = resolve_screening_matrix(matrix, {"eval_version": version}, None,
        context={"cwd": compiled["cwd"], "root": compiled["cwd"] / "campaigns",
            "loop_id": "healthy-counts", "minimum": 1, "fitted_candidates": 1,
            "resolved_data": resolved, "policy": SimpleNamespace(identity_dict=lambda: {})})
    assert result is matrix and resolved == {"eval_version": version, "successions": []}
    if unsupported:
        assert pending is not None and pending["measurement_complete"] is False
    else:
        assert pending is None


def test_matrix_helper_compiles_pair_and_accepts_one_revalidated_successor(compiled):
    from scripts.autotrain_readiness import resolve_matrix_readiness

    matrix = _matrix_fixture(compiled)
    root = compiled["cwd"] / "campaigns"
    args = dict(cwd=compiled["cwd"], root=root, loop_id="pair", minimum=2)
    first = resolve_matrix_readiness(matrix, **args)
    assert first["status"] == "ready", first
    assert first["eval_version"] != "e938_role_safe_all_targets_smoke96_v2"
    activity = CampaignStore(first["readiness_campaign_id"], root)
    request = load_locked_readiness_request(activity, wanted=first["request_digest"])
    assert len(request.additional_trainer_configs) == 1
    assert len(list(root.glob("readiness-*"))) == 1
    from slm_training.autoresearch.heal import load_heal_receipts
    attempts = load_heal_receipts(root, "pair")
    assert len(attempts) == 1 and attempts[0].outcome == "healed"
    assert resolve_matrix_readiness(matrix, **args) == first
    assert load_heal_receipts(root, "pair") == attempts
    destination = compiled["cwd"] / "outputs/data/eval" / first["eval_version"]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert set(manifest["suites"]) == {"smoke", "held_out", "adversarial", "ood", "rico_held"}
    # The rebuilt matrix resolves the successor's real full inputs and sampling authority.
    for arm in matrix["hypotheses"]:
        arm["experiment"]["knobs"]["eval_version"] = first["eval_version"]
    assert resolve_matrix_readiness(matrix, **args)["status"] == "ready"
    (destination / "records.jsonl").write_text("{}\n")
    refused = resolve_matrix_readiness(matrix, **args)
    assert refused["status"] == "waiting_dependency" and refused["failure_kind"] == "data_failure"
    assert refused["blocker"]["blocker_code"] == "data_readiness_context_failure"
    assert refused["blocker"]["kind"] == "repair_harness"  # No constructible request: execute diagnosis, not an inert builder.
    assert refused["wake"]["identity_digest"] == refused["request_digest"]


def test_matrix_context_failure_is_idempotent_durable_typed_wait(compiled):
    from scripts.autotrain_readiness import resolve_matrix_readiness

    matrix = _matrix_fixture(compiled)
    matrix["hypotheses"][1]["experiment"]["knobs"]["eval_version"] = "different"
    args = dict(cwd=compiled["cwd"], root=compiled["cwd"] / "campaigns", loop_id="pair", minimum=2)
    first = resolve_matrix_readiness(matrix, **args)
    assert first["status"] == "waiting_dependency" and len(first["request_digest"]) == 64
    assert first["failure_kind"] == "code_failure" and first["blocker"]["kind"] == "repair_harness"
    assert "one explicit shared evaluation" in first["reason"]
    assert resolve_matrix_readiness(matrix, **args) == first
    events = CampaignStore(matrix["campaign_id"], args["root"]).verify_event_chain()
    assert sum(event["event_type"] == "matrix_readiness_wait" for event in events) == 1
    assert not any(event["event_type"] == "experiment_finished" for event in events)


@pytest.mark.parametrize("failure", [KeyError("internal-bug"), RuntimeError("internal-bug"), ValueError("internal-bug")])
def test_internal_failure_is_code_repair_not_missing_capability(compiled, monkeypatch, failure):
    from scripts import autotrain_readiness as owner

    matrix = _matrix_fixture(compiled)
    def broken(*args, **kwargs):
        raise failure
    monkeypatch.setattr(owner, "_matrix_contexts", broken)
    result = owner.resolve_matrix_readiness(matrix, cwd=compiled["cwd"],
        root=compiled["cwd"] / "campaigns", loop_id="broken", minimum=2)
    assert result["status"] == "waiting_dependency" and result["failure_kind"] == "code_failure"
    assert result["blocker"]["executor"] == "configured_source_repair"
    assert result["blocker"]["original_reproducer"]["input_identity_digest"] == result["request_digest"]


def test_explicit_missing_context_remains_scoped_capability(compiled, monkeypatch):
    from scripts import autotrain_readiness as owner

    matrix = _matrix_fixture(compiled)
    def unavailable(*args, **kwargs):
        raise owner.ReadinessContextUnavailable("starting checkpoint lineage unavailable")
    monkeypatch.setattr(owner, "_matrix_contexts", unavailable)
    result = owner.resolve_matrix_readiness(matrix, cwd=compiled["cwd"],
        root=compiled["cwd"] / "campaigns", loop_id="unavailable", minimum=2)
    assert result["status"] == "waiting_capability"
    assert result["wake"]["predicate"] == "locked_data_readiness_context"


def test_exhausted_data_budget_requests_diagnosis_not_more_builder_attempts(compiled, monkeypatch):
    from scripts import autotrain_readiness as owner
    from scripts import autotrain_controller_repair as dispatcher

    matrix = _matrix_fixture(compiled)
    monkeypatch.setattr(owner, "_data_budget_exhausted", lambda *args: True)
    def unexpected(**kwargs):
        pytest.fail("an exhausted data budget dispatched another builder")
    monkeypatch.setattr(dispatcher, "dispatch_screening_rebuild", unexpected)
    result = owner.resolve_matrix_readiness(matrix, cwd=compiled["cwd"],
        root=compiled["cwd"] / "campaigns", loop_id="exhausted", minimum=2)
    assert result["status"] == "waiting_dependency" and result["failure_kind"] == "diagnosis_required"
    assert result["blocker"]["blocker_code"] == "screening_constraint_unknown"
    assert result["blocker"]["executor"] == "configured_source_repair"
