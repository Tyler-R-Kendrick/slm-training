"""Collector contracts, not a substitute for the real supervised demonstration."""

from types import SimpleNamespace

import pytest

from slm_training.harnesses.experiments.autonomous_learning.measurement_supervised_evidence import (
    collect_supervised,
    require_six_pairs,
)


def test_pair_metadata_is_not_miscounted_as_an_observation_array():
    paired = {
        "control": dict.fromkeys(range(6), 1.0),
        "candidate": dict.fromkeys(range(6), 0.9),
        "selection_sha256": "a" * 64,
    }
    require_six_pairs(paired, [])
    with pytest.raises(ValueError, match="six actual paired"):
        require_six_pairs(paired, ["identity_mismatch"])
    paired["candidate"].pop(5)
    with pytest.raises(ValueError, match="six actual paired"):
        require_six_pairs(paired, [])


@pytest.mark.parametrize("returncode", [False, True, 10, 1, None, "0"])
def test_collector_cannot_promote_missing_or_pending_supervisor_output(returncode):
    with pytest.raises(ValueError, match="actual same-campaign completion"):
        collect_supervised(
            SimpleNamespace(campaign_id="fixture"),
            {},
            {"returncode": returncode, "campaign_id": "fixture", "completion": {}},
        )


def test_public_supervisor_result_requires_current_pass_committed_output_and_source(tmp_path):
    import json
    from scripts.merge_verification_evidence import digest, file_digest
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.activity_contract import ActivitySpec, ActivityOutcome
    from slm_training.harnesses.experiments.autonomous_learning.measurement_supervised_evidence import completed_operation

    store = CampaignStore("runtime", tmp_path)
    with ActivityRuntime(store) as runtime:
        before = {event["event_id"] for event in store.verify_event_chain()}
        store.append_event("supervisor_pass", detail={"sequence": 1})
        runtime.register(ActivitySpec(activity_id="driver", family="fixture", kind="control",
                         source_digest="a" * 64, environment_digest="b" * 64,
                         input_digest="c" * 64, output_namespace="attempts/driver"))
        lease = runtime.claim_next(capabilities={"local_process"})
        directory = runtime.attempt_dir(lease)
        directory.mkdir(parents=True)
        request = {"operation": "driver", "source_digest": "a" * 64}
        payload = {"returncode": 0, "campaign_id": "fixture", "completion": {}}
        (directory / "request.json").write_text(json.dumps(request))
        result = directory / "result.json"
        result.write_text(json.dumps({"schema_version": "supervisor_operation/v1",
                         "operation": "driver", "request_digest": digest(request),
                         "payload": payload}))
        runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED,
                       outputs={"result.json": file_digest(result)}, spent_seconds=0)
        assert completed_operation(store, "fixture", before, "a" * 64) == payload
        assert completed_operation(store, "other-campaign", before, "a" * 64) is None
        assert completed_operation(store, "other-campaign", before, "d" * 64) is None
        with pytest.raises(ValueError, match="another source"):
            completed_operation(store, "fixture", before, "d" * 64)
        all_events = {event["event_id"] for event in store.verify_event_chain()}
        assert completed_operation(store, "fixture", all_events, "a" * 64) is None
        result.write_text("{}")
        with pytest.raises(ValueError, match="result changed"):
            completed_operation(store, "fixture", before, "a" * 64)


def test_supervised_result_binds_source_plan_operation_and_manifests(tmp_path, monkeypatch):
    import hashlib
    import json

    from slm_training.autoresearch.schemas import CampaignSpec
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.evals.measurement_identity import content_digest
    from slm_training.harnesses.experiments.autonomous_learning import measurement_supervised_evidence as owner

    store = CampaignStore("fixture", tmp_path)
    store.initialize(CampaignSpec(
        campaign_id="fixture", loop_id="fixture-loop", cycle_index=1,
        upstream_commit="a" * 40, integration_commit="b" * 40,
        objective="Bind supervised result to its execution", primary_metric="smoke.eval_nll",
    ))
    handoff = store.root / "cycle_handoff.json"
    handoff.write_text("immutable handoff")
    handoff_sha = hashlib.sha256(handoff.read_bytes()).hexdigest()
    completion = {"campaign_id": "fixture", "cycle": 1}
    operation = {"returncode": 0, "campaign_id": "fixture", "completion": completion,
                 "handoff_digest": handoff_sha}
    plan = {
        "execution_context": {"source": "c" * 64},
        "inputs": {"selection": {"count": 6}, "arms": {
            "control": {"trainable_parameters": 10},
            "candidate": {"trainable_parameters": 10},
        }},
        "arms": {"control": {"manifest_sha256": "d" * 64},
                 "candidate": {"manifest_sha256": "e" * 64}},
        "primary": {"metric": "smoke.eval_nll"},
    }
    (store.root / "sdlc_delivery.json").write_text(json.dumps({
        "measurement_complete": True, "primary_metric": "smoke.eval_nll",
    }))
    run_dirs = {}
    for name in plan["arms"]:
        run_dirs[name] = tmp_path / name
        run_dirs[name].mkdir()
        (run_dirs[name] / "scoreboard.json").write_text("{}")
        (run_dirs[name] / "loss_suites.json").write_text("{}")
    arms = {
        name: {"manifest_sha256": plan["arms"][name]["manifest_sha256"],
               "run_dir": run_dirs[name], "agentv_artifacts": []}
        for name in plan["arms"]
    }
    monkeypatch.setattr(owner, "completed_cycle_since", lambda *args: completion)
    monkeypatch.setattr(owner, "checked_arm", lambda _plan, name: arms[name])
    monkeypatch.setattr(owner, "read_paired_nll", lambda *_args: (
        {"control": dict.fromkeys(range(6), 1.0),
         "candidate": dict.fromkeys(range(6), 0.9)}, {"pairs": 6}, [],
    ))
    monkeypatch.setattr(owner, "measurement_is_complete", lambda _decision: True)
    monkeypatch.setattr(owner, "stage_receipts", lambda *_args: {"stage": {"seconds": 1}})
    monkeypatch.setattr(owner, "write_result_docs", lambda *_args: None)

    result = owner.collect_supervised(store, plan, operation)
    identity = result["execution_identity"]
    expected = {
        "source_digest": plan["execution_context"]["source"],
        "plan_sha256": content_digest(plan),
        "operation_sha256": content_digest(operation),
        "completion_sha256": content_digest(completion),
        "handoff_sha256": handoff_sha,
        "manifest_sha256s": {name: arm["manifest_sha256"] for name, arm in arms.items()},
    }
    assert result["source_digest"] == expected["source_digest"]
    assert {key: identity[key] for key in expected} == expected
    assert identity["identity_sha256"] == content_digest(expected)


def test_native_invocation_preserves_locked_grant_and_zero_exit_is_not_completion(tmp_path, monkeypatch):
    from slm_training.autoresearch.schemas import CampaignSpec
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core import bounded_process
    from slm_training.harness_core.activity_contract import ResourceGrant
    from slm_training.harnesses.experiments.autonomous_learning import measurement_supervised as native

    grant = ResourceGrant(total_seconds=1081, max_attempts=6)
    store = CampaignStore("fixture", tmp_path)
    store.initialize(CampaignSpec(campaign_id="fixture", loop_id="fixture-loop",
                     cycle_index=1, upstream_commit="a" * 40, integration_commit="b" * 40,
                     objective="Test native launch contract", primary_metric="smoke.eval_nll",
                     budget={"continuation_grant": grant}))
    plan = {"inputs": {"train_version": "retained"}, "execution_context": {"source": "a" * 64}}
    monkeypatch.setattr(native.bundle, "load_contract", lambda *_: plan)
    calls, budgets = [], []

    def child(command, **kwargs):
        calls.append(command)
        budgets.append(kwargs["interrupt_after_seconds"])
        return bounded_process.BoundedProcessResult(tuple(command),
            bounded_process.ProcessOutcome.COMPLETED, 0, "", "", 0.01)

    monkeypatch.setattr(bounded_process, "run_bounded_process", child)
    result = native.run(store)
    assert calls[0][1:3] == ["-m", "scripts.run_autotrain_supervisor"]
    assert "--operation-request" not in calls[0]
    assert ResourceGrant.model_validate_json(calls[0][-1]) == grant
    assert result["payload"] is None and result["returncode"] == 10
    assert not (store.root / "supervised_measurement_result.json").exists()
    from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS
    assert 0 < budgets[0] <= INTERRUPT_AFTER_SECONDS - 2 * KILL_GRACE_SECONDS


def test_native_resume_collects_committed_child_without_another_supervisor(tmp_path, monkeypatch):
    from slm_training.autoresearch.schemas import CampaignSpec
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core import bounded_process
    from slm_training.harnesses.experiments.autonomous_learning import measurement_supervised as native
    from slm_training.harnesses.experiments.autonomous_learning import measurement_supervised_evidence as owner

    store = CampaignStore("fixture", tmp_path)
    store.initialize(CampaignSpec(campaign_id="fixture", loop_id="loop", cycle_index=1,
        upstream_commit="a" * 40, integration_commit="b" * 40,
        objective="Recover a committed operation", primary_metric="smoke.eval_nll"))
    plan = {"execution_context": {"source": "a" * 64}}
    operation = {"returncode": 0, "campaign_id": "fixture", "completion": {}}
    observed = []
    monkeypatch.setattr(native.bundle, "load_contract", lambda *_: plan)
    monkeypatch.setattr(owner, "completed_operation", lambda journal, campaign, before, source:
        operation if campaign == "fixture" and not before and source == "a" * 64 else None)
    monkeypatch.setattr(owner, "collect_supervised", lambda *args: observed.append(args))
    monkeypatch.setattr(bounded_process, "run_bounded_process", lambda *a, **kw: pytest.fail("completed work restarted"))
    result = native.run(store)
    assert result["recovered"] and result["returncode"] == 0
    assert result["invocation_sha256"] is None and observed == [(store, plan, operation)]


def test_supervised_prepare_accepts_current_release_eval_version(tmp_path, monkeypatch):
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harnesses.experiments.autonomous_learning import measurement_supervised as native

    monkeypatch.setattr(native, "source_identity", lambda: {
        "components": {"harness.model_build.eval": "v107"}
    })
    store = CampaignStore("fixture", tmp_path)
    with pytest.raises(FileNotFoundError, match="unused-retained-plan"):
        native.prepare(store, "loop", tmp_path / "unused-retained-plan.json")


def test_collection_replays_durable_event_after_mirror_write_interruption(tmp_path, monkeypatch):
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harnesses.experiments.autonomous_learning import measurement_supervised_evidence as owner

    store = CampaignStore("fixture", tmp_path)
    original_write = owner._write
    monkeypatch.setattr(owner, "write_result_docs", lambda *a: None)

    def interrupted(*args):
        raise OSError("interrupted after durable event")

    monkeypatch.setattr(owner, "_write", interrupted)
    with pytest.raises(OSError):
        owner._persist_result(store, {"version_stamp": {"stamped_at": "first"}, "paired_count": 6})
    monkeypatch.setattr(owner, "_write", original_write)
    replay = owner._persist_result(store, {"version_stamp": {"stamped_at": "retry"}, "paired_count": 6})
    assert replay["version_stamp"]["stamped_at"] == "first"
    assert len(store.verify_event_chain()) == 1
    assert len(list((store.root / "artifacts/supervised_measurement_result").glob("*.json"))) == 1
    with pytest.raises(ValueError, match="committed supervised measurement changed"):
        owner._persist_result(store, {"version_stamp": {"stamped_at": "later"}, "paired_count": 5})


def test_relocated_retained_checkpoint_still_requires_original_content_hash(tmp_path, monkeypatch):
    import json
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.harness_core.checkpoint_bundle import stage_checkpoint_bundle
    from slm_training.harnesses.experiments.autonomous_learning import measurement_fixture as fixture
    from tests.test_harnesses.model_build.test_checkpoint_bundle import _checkpoint

    checkpoint = _checkpoint(tmp_path / "original")
    owner = tmp_path / "store"
    bundle = stage_checkpoint_bundle(owner, checkpoint, {})
    relocated = owner / "bundles" / bundle / "last.pt"
    rows = [ExampleRecord(id=f"case-{n}", prompt="fixture", openui="root = Stack([])")
            for n in range(6)]
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(fixture, "DataStore", lambda: SimpleNamespace(
        verify=lambda *_: SimpleNamespace(path=tmp_path)))
    monkeypatch.setattr(fixture, "load_suite_records", lambda *_: rows)
    evidence = {"selection": fixture.selected_identity(rows), "arms": {
        arm: {"checkpoint": "missing/original/last.pt", "checkpoint_sha256": fixture._sha(checkpoint),
              "trainable_parameters": 1} for arm in ("control", "candidate")}}
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    paths = dict.fromkeys(evidence["arms"], relocated)
    inputs = fixture._inputs(path, "train", "eval", checkpoint_paths=paths)
    assert inputs["arms"]["control"]["checkpoint"] == str(relocated)
    evidence["arms"]["control"]["checkpoint_sha256"] = "0" * 64
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        fixture._inputs(path, "train", "eval", checkpoint_paths=paths)


def test_documentation_projection_does_not_mutate_immutable_execution(tmp_path, monkeypatch):
    from slm_training.harness_core.execution_release import prepare_release, runtime_source_identity
    from slm_training.harnesses.experiments.autonomous_learning.measurement_fixture_evidence import write_result_docs

    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.py").write_text("# fixture source\n")
    execution, outputs = tmp_path / "execution", tmp_path / "outputs"
    manifest = prepare_release(source, tmp_path / "release", execution, outputs)
    monkeypatch.chdir(execution)
    result = {"decision": {"primary_metric": "smoke.eval_nll"},
              "arms": {"control": {"trainable_parameters": 1}}, "charged_child_seconds": 0}
    write_result_docs("fixture", result)
    assert (outputs / "docs/design/autonomy-measurement-fixture.json").is_file()
    assert (outputs / "docs/design/autonomy-measurement-fixture.md").is_file()
    assert not (execution / "docs").exists()
    assert runtime_source_identity(execution) == manifest["source_digest"]


@pytest.mark.parametrize("dirty", [False, True])
def test_frozen_provenance_reaches_native_refs_without_live_git(tmp_path, monkeypatch, dirty):
    from scripts import autoresearch
    from slm_training.harness_core import execution_release as release, versioning
    from slm_training.harnesses.experiments.autonomous_learning import measurement_bundle

    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.py").write_text("# fixture\n")
    provenance = {"integration_commit": "b" * 40, "upstream_commit": "a" * 40,
                  "code_dirty": dirty}
    monkeypatch.setattr(release, "_checkout_provenance", lambda _: provenance)
    execution = tmp_path / "execution"
    manifest = release.prepare_release(source, tmp_path / "release", execution, tmp_path / "outputs")
    monkeypatch.chdir(execution)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setattr(autoresearch, "ROOT", execution)

    def no_git(*args, **kwargs):
        pytest.fail("frozen provenance must not invoke live Git")

    monkeypatch.setattr(autoresearch, "_git", no_git)
    assert measurement_bundle.continuous_source_commits("b" * 40) == {
        "upstream_commit": "a" * 40, "integration_commit": "b" * 40}
    assert versioning._git_output(["rev-parse", "HEAD"]) == "b" * 40
    assert bool(versioning._git_output(["status", "--porcelain"])) is dirty
    if dirty:
        with pytest.raises(ValueError, match="clean tracked worktree"):
            autoresearch._validate_continuous_commits("a" * 40, "b" * 40)
    else:
        autoresearch._validate_continuous_commits("a" * 40, "b" * 40)
    with pytest.raises(ValueError, match="immutable release provenance"):
        autoresearch._validate_continuous_source_refs("c" * 40, "b" * 40)
    assert release.runtime_source_identity(execution) == manifest["source_digest"]
    (execution / "fixture.py").write_text("# drift\n")
    with pytest.raises(ValueError, match="execution_source_drift"):
        measurement_bundle.continuous_source_commits("b" * 40)
    assert versioning._git_output(["rev-parse", "HEAD"]) is None


def test_frozen_provenance_metadata_tamper_and_missing_refs_fail_closed(tmp_path, monkeypatch):
    import json
    from slm_training.harness_core import execution_release as release

    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.py").write_text("# fixture\n")
    monkeypatch.setattr(release, "_checkout_provenance", lambda _: None)
    execution = tmp_path / "execution"
    release.prepare_release(source, tmp_path / "release", execution, tmp_path / "outputs")
    with pytest.raises(ValueError, match="release_git_provenance_unavailable"):
        release.runtime_git_provenance(execution)
    marker = execution / release.MARKER
    manifest = json.loads(marker.read_text())
    manifest["git_provenance"] = {"integration_commit": "b" * 40,
                                  "upstream_commit": "a" * 40, "code_dirty": False}
    marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="immutable_release_changed"):
        release.runtime_git_provenance(execution)


@pytest.mark.parametrize("mutation", ["bytes", "added", "deleted", "executable", "hardlink", "external_link"])
def test_combined_source_provenance_rechecks_drift(tmp_path, monkeypatch, mutation):
    import os
    from slm_training.harness_core import execution_release as release

    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.py").write_text("original bytes\n")
    (source / "alias").symlink_to("fixture.py")
    provenance = {"integration_commit": "b" * 40, "upstream_commit": "a" * 40, "code_dirty": False}
    monkeypatch.setattr(release, "_checkout_provenance", lambda _: provenance)
    execution = tmp_path / "execution"
    manifest = release.prepare_release(source, tmp_path / "release", execution, tmp_path / "outputs")
    assert release.runtime_source_provenance(execution) == (manifest["source_digest"], provenance)
    target = execution / "fixture.py"
    info = target.stat()
    if mutation == "bytes":
        target.write_text("modified bytes\n")
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    elif mutation == "added":
        (execution / "new.py").write_text("new file")
    elif mutation == "deleted":
        target.unlink()
    elif mutation == "executable":
        target.chmod(0o755)
    elif mutation == "hardlink":
        os.link(target, tmp_path / "outside-hardlink")
    else:
        (execution / "alias").unlink()
        (execution / "alias").symlink_to(source / "fixture.py")
    with pytest.raises(ValueError, match="execution_source_drift|unsupported_source_file|external_source_link"):
        release.runtime_source_provenance(execution)
