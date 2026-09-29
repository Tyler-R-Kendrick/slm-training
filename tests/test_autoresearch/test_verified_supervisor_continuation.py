"""Real bounded verification -> ordinary fresh supervisor -> retained cursor.

Scientific children are deterministic protocol fixtures. Discovery/isolation and
Git provenance use declared fixture adapters, not a production verification claim.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import autoresearch, autotrain_cycle_prepare as prepare
from scripts.autoresearch_command_cursor import CommandCursor
from scripts.autotrain_cycle_context import CycleJournal
from scripts.autotrain_cycle_reconcile import start_attempts
from scripts.merge_verification_identity import digest, environment_identity
from slm_training.autoresearch import engine
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore, _sha
from slm_training.harness_core.activity_contract import ActivityOutcome, ActivitySpec, ResourceGrant, WakeCondition
from slm_training.harness_core import execution_release
from tests.test_autoresearch import test_locked_prereg_supervisor as locked
from tests.test_autoresearch.continuation_repair_fixture import workload_source, publish_repair
from tests.test_autoresearch.test_controller_workload_worker import _controller


def _setup(tmp_path, monkeypatch):
    controller, _ = _controller(tmp_path)
    science = tmp_path / "science"
    science.mkdir()
    (science / "rows").write_text("preserved\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "execution/src"))
    release = execution_release.prepare_release
    knobs = locked.ExperimentKnobs
    persist = locked.persist_pair

    def materialize(source, *args, **kwargs):
        workload_source(source, science)
        return release(source, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(execution_release, "prepare_release", materialize)
        patch.setattr(locked, "persist_pair", lambda store, pair: persist(
            store, {**pair, "hypothesis_id": "fixture-hypothesis"}))
        patch.setattr(locked, "ExperimentKnobs", lambda **kw: knobs(
            **kw, eval_partial_scoreboard=True, eval_max_records_this_run=2))
        patch.setattr(locked, "ResourceGrant", lambda **_: ResourceGrant(total_seconds=1200, max_attempts=8))
        plan, path, root, store, _ = locked._fixture(tmp_path, patch)
        selection = locked._selection(plan, path, root, store)
        value = prepare.prepare_recorded_cycle(Path(plan["source_path"]), root, SimpleNamespace(), selection)
    environment = environment_identity()
    controller_runtime = {k: v for k, v in environment.items() if k != "execution_environment_sha256"}
    binding = {"cwd": str(controller), "source_digest": execution_release.runtime_source_identity(controller),
               "runtime_digest": digest(controller_runtime)}
    args = SimpleNamespace(loop_id=plan["campaign_id"], root=root, train_version="fixture-train", steps=6,
        primary_metric="smoke.eval_nll", continuation_grant=store.load_campaign().budget.continuation_grant.model_dump_json(),
        locked_preregistration=path, locked_prereg_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    request = {"cwd": plan["source_path"], "root": str(root), "loop_id": args.loop_id,
        "source_digest": plan["source_digest"], "environment_digest": digest(environment),
        "repair_config": None, "repair_config_digest": None,
        "delivery_config": None, "delivery_config_digest": None,
        "operation": "driver", "driver_argv": prepare._driver_argv(args),
        "predecessor_campaign_id": store.campaign_id, "locked_diagnostic": True}
    return SimpleNamespace(directory=tmp_path, controller=controller, binding=binding, science=science,
        store=store, root=root, plan=plan, args=args, value=value, request=request,
        input_digest=_sha(value), base=tmp_path / "original")


def _seed_committed_train(f):
    eid = f.value["order"][0]
    arm = f.value["arms"][eid]
    spec = autoresearch.ExperimentSpec.model_validate_json(Path(f.value["by_id"][eid]).read_text())
    journal = CycleJournal(f.store, f.value)
    journal.start("arms", 30)
    start_attempts(journal, eid)
    with CommandCursor(f.store, spec, arm["commands"], arm["manifest_digest"],
        f.value["execution_identity"], f.value["total_seconds"], cwd=Path(f.plan["source_path"]),
        max_attempts=8) as cursor:
        cursor.start(30)

        def interrupt(partial):
            cursor.checkpoint(partial, 1)
            raise InterruptedError("fixture crash after committed training")

        with pytest.raises(InterruptedError, match="committed training"):
            outcome = engine.execute_commands(spec, arm["commands"], cwd=Path(f.plan["source_path"]),
                timeout_seconds=30, campaign_manifest_sha256=arm["manifest_digest"], stage_callback=interrupt)
            pytest.fail(str(outcome))
    assert (f.science / "trained").read_text() == "x"
    from scripts.autotrain_readiness_probe import probe_continuation
    from scripts.autotrain_cursor_reconcile import committed_prefix_reconcilable

    before = f.store.verify_event_chain()
    assert probe_continuation({"cwd": f.plan["source_path"], "continuation": {
        "campaign_id": f.store.campaign_id, "input_digest": f.input_digest,
        "blocked_state_digest": _sha(journal.state)}}, root=f.root)["ready"]
    assert f.store.verify_event_chain() == before
    assert not committed_prefix_reconcilable(f.store,
        {**f.value, "execution_identity": "forged"}, journal.state)
    return eid


def _original_operation(f, *, cursor_pending=False):
    journal = CampaignStore("runtime", f.root / "loops" / f.args.loop_id)
    with ActivityRuntime(journal) as runtime:
        runtime.register(ActivitySpec(activity_id="original-driver", family=f.args.loop_id, kind="control",
            source_digest=f.request["source_digest"], environment_digest=f.request["environment_digest"],
            input_digest=digest(f.request), output_namespace="attempts/original-driver",
            capabilities=("local_process", "controller_publication"),
            grant=f.store.load_campaign().budget.continuation_grant))
        lease = runtime.claim_next(activity_id="original-driver", capabilities={"local_process", "controller_publication"})
        result = runtime.run(lease, [sys.executable, "-I", "-c", "raise SystemExit(0)" if cursor_pending else "raise SystemExit(73)"], cwd=Path(f.plan["source_path"]))
        from slm_training.autoresearch.heal.operation_recovery import record_operation_failure

        if cursor_pending:
            from scripts.autotrain_pending import publish_pending
            from slm_training.autoresearch.heal.operation_recovery import record_driver_reconciliation

            pending = CycleJournal(f.store, f.value).pending(
                "driver_attempt_requires_reconciliation", capability=True)
            publish_pending(f.root, f.args.loop_id, pending)
            record_driver_reconciliation(runtime, lease, f.request, pending)
            directory = runtime.attempt_dir(lease)
            envelope_request = {**f.request, "lease": lease.model_dump(mode="json")}
            (directory / "request.json").write_text(json.dumps(envelope_request))
            output = directory / "result.json"
            output.write_text(json.dumps({"schema_version": "supervisor_operation/v1",
                "request_digest": digest(envelope_request), "operation": "driver",
                "payload": {"returncode": 10, "pending": pending, "campaign_id": f.store.campaign_id}}))
            runtime.finish(lease, outcome=ActivityOutcome.CAPABILITY,
                outputs={"result.json": hashlib.sha256(output.read_bytes()).hexdigest()},
                spent_seconds=result.duration_seconds, wake=WakeCondition.model_validate(pending["wake"]))
            return journal
        record_operation_failure(runtime, lease, f.request, result, outcome=ActivityOutcome.CODE_FAILURE)
        runtime.finish(lease, outcome=ActivityOutcome.CODE_FAILURE, spent_seconds=result.duration_seconds,
            wake=WakeCondition(predicate="original operation produces valid output",
                source="independent_repair_verification", identity_digest=digest(f.request)))
    return journal


def _supervisor(f, handoff, *, passes=0, repair_config=None):
    bootstrap = (
        "import runpy,sys\nsys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)]\n"
        "from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime\n"
        "run = ActivityRuntime.run\n"
        "def observed(self, *args, **kwargs):\n"
        "    result = run(self, *args, **kwargs)\n"
        "    print(result.stderr, file=sys.stderr)\n"
        "    return result\n"
        "ActivityRuntime.run = observed\n"
        "runpy.run_module('scripts.run_autotrain_supervisor',run_name='__main__')"
    )
    argv = [sys.executable, "-I", "-c", bootstrap, str(f.controller), str(f.controller / "src"),
        "--controller-root", str(f.controller), "--controller-source-digest", f.binding["source_digest"],
        "--controller-runtime-digest", f.binding["runtime_digest"], "--root", str(f.root),
        "--loop-id", f.args.loop_id, "--locked-preregistration", str(f.args.locked_preregistration),
        "--continuation-grant", f.args.continuation_grant, "--train-version", "fixture-train",
        "--steps", "6", "--primary-metric", "smoke.eval_nll", "--stop-after-pass", str(passes), "--no-playbooks", "--hard-backoff-seconds", "0.1"]
    if repair_config is not None:
        argv += ["--repair-config", str(repair_config)]
    result = subprocess.run(argv, cwd=handoff["successor_execution"], env=dict(os.environ),
                            capture_output=True, text=True, timeout=150, check=False)
    return result


def test_two_verified_repairs_resume_ordinary_supervisor_without_repeating_train(tmp_path, monkeypatch):
    f = _setup(tmp_path, monkeypatch)
    eid = _seed_committed_train(f)
    journal = _original_operation(f)
    checkpoint = (f.science / "checkpoint").read_bytes()
    inputs = list((f.store.root / "artifacts/command_cursor_inputs").glob("*.json"))
    before_inputs = {path.name: path.read_bytes() for path in inputs}
    original_budget = f.store.load_campaign().budget.model_dump(mode="json")
    activity = "original-driver"
    requests, handoffs = [], []
    for index in (1, 2):
        with ActivityRuntime(journal) as runtime:
            handoff = publish_repair(f, runtime, runtime.snapshot()[activity], index, monkeypatch)
        handoffs.append(handoff)
        result = _supervisor(f, handoff)
        assert result.returncode in {0, 2}, result.stdout + result.stderr
        plans = [e["detail"] for e in journal.verify_event_chain() if e["event_type"] == "operation_successor_planned"]
        request = plans[-1]["request"]
        requests.append(request)
        activity = request["successor_activity_id"]
        assert request["driver_argv"] == prepare._driver_argv(f.args)
        assert (f.science / "trained").read_text() == "x"
        assert (f.science / f"origin-{index}.json").exists(), result.stdout + result.stderr
        origin = json.loads((f.science / f"origin-{index}.json").read_text())
        assert origin["cwd"] == handoff["successor_execution"]
        assert origin["module"] == str(Path(handoff["successor_execution"]) / "scripts/evaluate_model.py")
        assert "--resume-run" in origin["argv"]
    assert (f.science / "rows").read_text() == "preserved\nrepair-1\nrepair-2\n"
    assert (f.science / "checkpoint").read_bytes() == checkpoint
    assert {path.name: path.read_bytes() for path in inputs} == before_inputs
    assert f.store.load_campaign().budget.model_dump(mode="json") == original_budget
    assert requests[1]["logical_continuation"]["prior_attempts"] > requests[0]["logical_continuation"]["prior_attempts"]
    assert requests[1]["resource_grant"]["total_seconds"] < requests[0]["resource_grant"]["total_seconds"]
    assert requests[1]["logical_continuation"]["scientific_replicate_increment"] == 0
    terminal = [json.loads((f.store.root / "artifacts/outcomes" /
        (e["artifact_sha256"] + ".json")).read_text()) for e in f.store.verify_event_chain()
        if e["event_type"] == "experiment_finished" and e["experiment_id"] == eid]
    assert terminal and all(o["status"] == "stopped" for o in terminal)
    assert all(o["error"] == "continuation_no_progress:repair_measurement_required" for o in terminal)

    # Adversarial fixture corruption: cutting an activation edge cannot launch.
    events_path = journal.root / "events.jsonl"
    original = events_path.read_text()
    lines = original.splitlines(keepends=True)
    cut = next(i for i, line in enumerate(lines)
               if json.loads(line)["event_type"] == "operation_controller_activation_verified")
    try:
        CampaignStore._replace_durable(events_path, "".join(lines[:cut] + lines[cut + 1:]))
        rejected = _supervisor(f, handoffs[-1])
        assert rejected.returncode != 0
        assert "chain" in rejected.stderr.lower()
        assert (f.science / "rows").read_text() == "preserved\nrepair-1\nrepair-2\n"
        assert (f.science / "trained").read_text() == "x"
    finally:
        CampaignStore._replace_durable(events_path, original)


def test_verified_controller_upgrade_resumes_unchanged_workload(tmp_path, monkeypatch):
    _unchanged_workload(_setup(tmp_path, monkeypatch))


def test_host_repair_config_resumes_retained_null_request_without_new_grant(tmp_path, monkeypatch):
    f = _setup(tmp_path, monkeypatch)
    path = tmp_path / "host-recovery.json"
    path.write_text(json.dumps({"schema_version": "recovery_config/v1", "verifier_release": "a" * 64}))
    _unchanged_workload(f, repair_config=path)


def _unchanged_workload(f, *, repair_config=None):
    from slm_training.autoresearch.runtime.activity_projection import ActivityProjection

    _seed_committed_train(f)
    journal = _original_operation(f, cursor_pending=True)
    original = ActivityProjection(journal).read()["original-driver"]
    source = execution_release.runtime_source_identity(Path(f.plan["source_path"]))
    request_digest = digest(f.request)
    budget = f.store.load_campaign().budget.model_dump(mode="json")
    result = _supervisor(f, {"successor_execution": f.plan["source_path"]}, passes=1, repair_config=repair_config)
    assert result.returncode == 10, result.stdout + result.stderr
    assert (f.science / "origin-0.json").exists(), result.stdout + result.stderr
    assert (f.science / "trained").read_text() == "x"
    assert (f.science / "checkpoint").read_bytes() == b"committed-six-updates"
    assert (f.science / "rows").read_text() == "preserved\nrepair-0\n"
    assert execution_release.runtime_source_identity(Path(f.plan["source_path"])) == source
    assert f.store.load_campaign().budget.model_dump(mode="json") == budget
    events = journal.verify_event_chain()
    launches = [e for e in events if e["event_type"] == "operation_controller_launch_verified"
                and e["experiment_id"] == "original-driver"]
    assert launches and launches[-1]["detail"]["request_digest"] == request_digest
    assert not any(e["event_type"] in {"repair_release_accepted", "operation_successor_planned"}
                   for e in events)
    assert any(e["event_type"] == "driver_pending_resolved" for e in events)
    states = ActivityProjection(journal).read()
    current = states["original-driver"]
    assert current.spec == original.spec
    assert current.attempts > original.attempts and current.charged_seconds >= original.charged_seconds
    assert not any("-driver-" in key for key in states if key != "original-driver")
    assert f.request["repair_config"] is None and f.request["repair_config_digest"] is None
    configuration = launches[-1]["detail"]["host_configuration"]
    assert configuration["repair_config"] == (str(repair_config) if repair_config else None)
    assert configuration["repair_config_digest"] == (hashlib.sha256(repair_config.read_bytes()).hexdigest() if repair_config else None)
