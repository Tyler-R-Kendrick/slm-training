"""Real publication/exec boundaries; verifier receipts are explicit fixtures."""

import json
import os
import sys
from pathlib import Path

import pytest

from scripts import autotrain_repair_activation as owner
from scripts import run_autotrain_supervisor as supervisor
from slm_training.autoresearch.heal import repair_release
from slm_training.harness_core.bounded_process import run_bounded_process
from tests.test_autoresearch import test_repair_release as publication_fixtures

publication = publication_fixtures.publication


def test_accepted_release_reaches_restart_before_inspection(publication):
    request, result, kwargs = publication
    handoff = repair_release.publish_verified_repair(request, result, **kwargs)
    runtime = kwargs["runtime"]
    with pytest.raises(owner.VerifiedRestart) as caught:
        owner.recover_release(
            runtime, {"cwd": str(kwargs["candidate"])}, sequence=1,
            log_event=lambda _: None, run_operation=lambda *_: pytest.fail("old imports ran"),
        )
    assert caught.value.handoff == handoff
    assert not any(e["event_type"] == "operation_successor_activated"
                   for e in runtime.store.verify_event_chain())


@pytest.mark.parametrize("fault", ["pointer", "source", "stop"])
def test_stale_or_cancelled_release_never_restarts(publication, fault):
    request, result, kwargs = publication
    handoff = repair_release.publish_verified_repair(request, result, **kwargs)
    runtime = kwargs["runtime"]
    if fault == "pointer":
        (runtime.store.root / "source_release_pointer.json").write_text("{}")
    elif fault == "source":
        (Path(handoff["successor_execution"]) / "fixture.py").write_text("answer = 0\n")
    else:
        runtime.cancel_event.set()
    def call():
        return owner.recover_release(
            runtime, {"cwd": str(kwargs["candidate"])}, sequence=1,
            log_event=lambda _: None, run_operation=lambda *_: pytest.fail("executed"),
        )
    if fault == "stop":
        call()
    else:
        with pytest.raises((KeyError, ValueError)):
            call()


def test_exec_replaces_imports_and_preserves_finite_options(tmp_path):
    """Real exec into a tiny disposable CLI, not a live coding-agent repair."""
    source = Path.cwd()
    execution = tmp_path / "successor"
    (execution / "scripts").mkdir(parents=True)
    (execution / "scripts/__init__.py").write_text("")
    (execution / "scripts/run_autotrain_supervisor.py").write_text(
        "import json,os,sys;print(json.dumps({'argv':sys.argv[1:],'pid':os.getpid(),"
        "'cwd':os.getcwd(),'pythonpath':os.environ['PYTHONPATH']}))\n"
    )
    code = (
        "import argparse,json,os;from pathlib import Path;"
        "from scripts.autotrain_repair_activation import restart_supervisor;"
        "print(json.dumps({'original_pid':os.getpid()}),flush=True);"
        f"restart_supervisor(argparse.Namespace(root=Path({str(tmp_path / 'outputs')!r}),"
        "loop_id='fixture',max_cycles=3,stop_after_pass=7,no_playbooks=True,"
        "operation_request=None,operation_output=None),"
        f"{{'successor_execution':{str(execution)!r},'restart_argv_prefix':['forbidden']}})"
    )
    result = run_bounded_process(
        (sys.executable, "-c", code), cwd=source,
        env={**os.environ, "PYTHONPATH": str(source / "src")},
        interrupt_after_seconds=10, kill_grace_seconds=10,
    )
    assert result.returncode == 0, result.stderr
    before, after = map(json.loads, result.stdout.splitlines())
    assert before["original_pid"] == after["pid"]
    assert after["cwd"] == str(execution)
    assert after["pythonpath"] == str(execution / "src")
    parsed = supervisor._build_parser().parse_args(after["argv"])
    assert parsed.stop_after_pass == 7 and parsed.max_cycles == 3
    assert parsed.no_playbooks and parsed.root == tmp_path / "outputs"


def test_main_releases_controller_before_exec(tmp_path, monkeypatch):
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.autoresearch.storage import CampaignStore

    root = tmp_path / "outputs"
    store = CampaignStore("runtime", root / "loops" / "fixture")
    handoff = {"successor_execution": "unused"}

    def accepted(*args, **kwargs):
        raise owner.VerifiedRestart(handoff)

    def exec_check(args, received):
        assert received == handoff and args.stop_after_pass == 1
        with ActivityRuntime(store):
            pass  # Would fail ControllerBusy if the old principal still owned it.
        raise SystemExit(19)

    monkeypatch.setattr(owner, "recover_release", accepted)
    monkeypatch.setattr(owner, "restart_supervisor", exec_check)
    monkeypatch.setattr(supervisor, "_source_identity", lambda _: "a" * 64)
    with pytest.raises(SystemExit) as exited:
        supervisor.main(["--loop-id", "fixture", "--root", str(root), "--max-cycles", "1"])
    assert exited.value.code == 19


@pytest.mark.parametrize("state", ["waiting_capability", "succeeded"])
def test_delivery_dependency_precedes_repaired_source_activation(publication, monkeypatch, state):
    from scripts import autotrain_supervision
    from slm_training.autoresearch.heal import repair_delivered

    request, result, kwargs = publication
    handoff = repair_release.publish_verified_repair(request, result, **kwargs)
    wait = {"kind": "verified_repair_source", "publication_id": handoff["publication_id"]}
    checked = {**handoff, "source_delivery": wait}
    delivered = {**checked, "successor_execution": str(kwargs["candidate"] / "delivered")}
    monkeypatch.setattr(owner, "verified_activation_handoff", lambda *_: checked)
    observed = []

    def deliver(runtime, common, waits, log):
        observed.append(waits)
        return [{"state": state, "reconciliation": {"delivered_activation": "committed-reference"}}]

    def resolve(store, reference):
        assert reference == "committed-reference" and state == "succeeded"
        return delivered

    monkeypatch.setattr(autotrain_supervision, "register_delivery_waits", deliver)
    monkeypatch.setattr(repair_delivered, "resolve_delivered_activation", resolve)
    arguments = dict(sequence=1, log_event=lambda _: None,
                     run_operation=lambda *_: pytest.fail("old source executed"))
    if state == "succeeded":
        with pytest.raises(owner.VerifiedRestart) as caught:
            owner.recover_release(kwargs["runtime"], {"cwd": str(kwargs["candidate"])}, **arguments)
        assert caught.value.handoff == delivered
    else:
        assert owner.recover_release(kwargs["runtime"], {"cwd": str(kwargs["candidate"])}, **arguments) == "waiting_delivery"
    assert observed == [[wait]]


def test_main_delivery_wait_exits_without_running_old_source(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, "recover_release", lambda *a, **kw: "waiting_delivery")
    monkeypatch.setattr(supervisor, "_source_identity", lambda _: "a" * 64)
    monkeypatch.setattr(supervisor, "_supervise", lambda *a: pytest.fail("old source inspected"))
    assert supervisor.main(["--loop-id", "fixture", "--root", str(tmp_path), "--max-cycles", "1"]) == 10


@pytest.mark.parametrize("limit", ["seconds", "attempts"])
def test_successor_chain_cannot_refill_root_grant(limit):
    """Exercise the actual planner across two repairs; runtime state is a fixture."""
    from types import SimpleNamespace
    from slm_training.autoresearch.heal.operation_recovery import (
        SuccessorGrantExhausted, _successor_plan,
    )
    from slm_training.harness_core.activity_contract import (
        ActivitySpec, ResourceGrant, contract_digest,
    )

    grant = ResourceGrant(interrupt_seconds=10, kill_grace_seconds=10,
                          total_seconds=65 if limit == "seconds" else 100,
                          max_attempts=10 if limit == "seconds" else 3)
    original = {"operation": "inspect", "replicate_id": "fixed-replicate"}
    spec = ActivitySpec(activity_id="root", family="fixture", kind="control",
                        source_digest="a" * 64, environment_digest="a" * 64,
                        input_digest=contract_digest(original), output_namespace="root",
                        grant=grant)
    for hop in range(3):
        state = SimpleNamespace(spec=spec, status="waiting_repair",
                                charged_seconds=20, attempts=1)
        runtime = SimpleNamespace(snapshot=lambda: {spec.activity_id: state})
        events = [{"event_type": "operation_repair_requested",
                   "experiment_id": spec.activity_id, "detail": {"request": original}}]
        checked = {"resume_activity_id": spec.activity_id, "publication_id": f"release-{hop}",
                   "source_digest": "b" * 64, "successor_execution": f"/release/{hop}"}
        if hop == 2:
            with pytest.raises(SuccessorGrantExhausted):
                _successor_plan(runtime, checked, events)
            break
        plan = _successor_plan(runtime, checked, events)
        original = plan["request"]
        spec = ActivitySpec.model_validate(plan["spec"])
        ancestry = original["logical_continuation"]
        assert ancestry["logical_resource_grant"] == grant.model_dump(mode="json")
        assert ancestry["prior_charged_seconds"] + spec.grant.total_seconds == grant.total_seconds
        assert ancestry["prior_attempts"] + spec.grant.max_attempts == grant.max_attempts
        assert ancestry["logical_activity_id"] == "root"
        assert ancestry["scientific_replicate_increment"] == 0
        assert original["replicate_id"] == "fixed-replicate"
