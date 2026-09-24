"""Independent supervisor replay oracle using real durable producer events."""

import hashlib
import json

import pytest

from tests.casefiles import case_values

from scripts import run_autotrain_supervisor as supervisor
from scripts.merge_verification_evidence import digest, environment_identity
from slm_training.autoresearch.runtime.activity_process import prepare_outputs
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore


def test_real_inspect_child_enters_publication_scope_before_healing(tmp_path):
    import sys
    from pathlib import Path
    from scripts.autotrain_supervisor_operations import _operation_state

    request = {"cwd": str(Path.cwd()), "root": str(tmp_path), "loop_id": "fixture",
               "operation": "inspect", "source_digest": "a" * 64, "environment_digest": digest(environment_identity())}
    child = '''
import sys
sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)]
from pathlib import Path
from types import SimpleNamespace
from scripts.autotrain_supervisor_operations import operation_main
from slm_training.harness_core.checkpoint_publication import has_champion_publication_scope
def heal(**kwargs):
    assert has_champion_publication_scope(), "pre-cycle heal lacks publication scope"
    return {"hard_pending": []}
def parked(**kwargs):
    assert has_champion_publication_scope(), "park recovery lacks publication scope"
    return None
continuous=SimpleNamespace(self_heal_unblock_loop=heal,
    _latest_cycle=lambda *_: (0,None), _check_regime_parked=parked)
raise SystemExit(operation_main(Path(sys.argv[1]),Path(sys.argv[2]),
    source_identity=lambda _: "a"*64, load_continuous=lambda: continuous,
    handle_hard_pending=lambda *a,**kw: {}, write_family_closures=lambda *_: None))
'''
    with ActivityRuntime(CampaignStore("runtime", tmp_path / "loops/fixture")) as runtime:
        state = _operation_state(runtime, request, 1, digest(request))
        lease = runtime.claim_next(activity_id=state.spec.activity_id,
                                   capabilities={"local_process", "controller_publication"})
        directory = runtime.attempt_dir(lease)
        directory.mkdir(parents=True, exist_ok=True)
        path, output = directory / "request.json", directory / "result.json"
        path.write_text(json.dumps({**request, "lease": lease.model_dump(mode="json")}))
        controller = Path(__file__).resolve().parents[2]
        result = runtime.run(lease, [sys.executable, "-c", child, str(controller),
                                    str(controller / "src"), str(path), str(output)], cwd=controller)
        assert result.returncode == 0, result.stderr
        assert "controller_publication" in state.spec.capabilities
    assert json.loads(output.read_text())["payload"]["report"] == {"hard_pending": []}


@pytest.mark.parametrize("wait_kind", ["document", "workspace"])
def test_delivery_capability_wait_does_not_starve_local_activity(tmp_path, wait_kind):
    import hashlib
    import sys
    from scripts.autotrain_supervision import register_delivery_waits
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.activity_contract import ActivitySpec, ActivityOutcome

    common = {"loop_id": "loop", "source_digest": "a" * 64, "environment_digest": "b" * 64}
    wait = {"campaign_id": "finished", "required_capability": "authorized_github_connector_delivery",
            "artifact_sha256": "c" * 64, "wake_source": "authorized_delivery_receipt"}
    if wait_kind == "workspace":
        from scripts.autotrain_controller_repair import preserve_workspace

        args = dict(cwd=tmp_path, root=tmp_path, loop_id="loop", reason="workspace_changes", paths=["human.py"])
        wait = preserve_workspace(**args)
        assert preserve_workspace(**args) == wait
    with ActivityRuntime(CampaignStore("runtime", tmp_path / "loops/loop")) as runtime:
        register_delivery_waits(runtime, common, [wait], lambda _: None)
        runtime.register(ActivitySpec(activity_id="healthy", family="independent", kind="control",
            source_digest="a" * 64, environment_digest="b" * 64, input_digest="d" * 64,
            output_namespace="healthy"))
        lease = runtime.claim_next(capabilities={"local_process"})
        assert lease.activity_id == "healthy"
        attempt = runtime.attempt_dir(lease)
        attempt.mkdir(parents=True, exist_ok=True)
        process = runtime.run(lease, [sys.executable, "-c",
            "from pathlib import Path; Path('proof.txt').write_text('completed independent work')"], cwd=attempt)
        assert process.returncode == 0
        runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED,
            outputs={"proof.txt": hashlib.sha256((attempt / "proof.txt").read_bytes()).hexdigest()},
            spent_seconds=process.duration_seconds)
        states = runtime.snapshot()
        assert states["healthy"].status == "succeeded"
        parked = next(state for key, state in states.items() if key.startswith(f"{wait_kind}-delivery-"))
        assert parked.status == "waiting_capability" and parked.attempts == 0


@pytest.mark.parametrize("argv", case_values(__file__, "test_controller_git_mutations_refuse_before_process_launch"))
def test_controller_git_mutations_refuse_before_process_launch(tmp_path, monkeypatch, argv):
    from scripts import run_autotrain_continuous as driver

    monkeypatch.setattr(driver, "run_bounded_process", lambda *a, **kw: pytest.fail("mutation launched"))
    with pytest.raises(PermissionError, match="controller Git is read-only"):
        driver._bounded_command(argv, cwd=tmp_path)


@pytest.mark.parametrize("drift", [None, "source", "environment"])
def test_supervisor_reuses_verified_before_terminal_event_after_restart(
    tmp_path, monkeypatch, drift,
):
    """Only crash injection is mocked; production replay/receipt owners are real."""
    from scripts import merge_verification_evidence as evidence

    monkeypatch.setattr(evidence, "source_identity", lambda _: "a" * 64)
    monkeypatch.setattr(evidence, "environment_identity", lambda: {"fixture_environment": True})
    request = {
        "operation": "closeout", "loop_id": "adversary",
        "source_digest": "a" * 64, "environment_digest": digest({"fixture_environment": True}),
        "cwd": str(tmp_path),
    }
    expected = {"events": [{"fixture": True}]}
    store = CampaignStore("supervisor-adversary", tmp_path / "store")

    class InjectedCrash(Exception):
        pass

    def crash_after_verified(runtime, lease, argv, **kwargs):
        directory = runtime.attempt_dir(lease)
        executed = json.loads((directory / "request.json").read_text())
        output = directory / "result.json"
        output.write_text(json.dumps({
            "schema_version": "supervisor_operation/v1",
            "operation": "closeout", "request_digest": digest(executed),
            "payload": expected,
        }))
        prepare_outputs(
            runtime, runtime.snapshot()[lease.activity_id], lease,
            {"result.json": hashlib.sha256(output.read_bytes()).hexdigest()}, 0.1,
        )
        raise InjectedCrash("after verified artifact, before finish event")

    with monkeypatch.context() as injection:
        injection.setattr(ActivityRuntime, "run", crash_after_verified)
        with ActivityRuntime(store) as runtime:
            with pytest.raises(InjectedCrash):
                supervisor._run_operation(runtime, request, sequence=0, log_event=lambda _: None)
    if drift == "source":
        monkeypatch.setattr(evidence, "source_identity", lambda _: "c" * 64)
    elif drift == "environment":
        monkeypatch.setattr(evidence, "environment_identity", lambda: {"fixture_environment": False})
    with ActivityRuntime(store) as runtime:
        if drift:
            with pytest.raises(ValueError, match=f"operation {drift} changed before replay"):
                supervisor._run_operation(runtime, request, sequence=0, log_event=lambda _: None)
        else:
            assert supervisor._run_operation(
                runtime, request, sequence=0, log_event=lambda _: None,
            ) == expected
    observations = [e for e in store.verify_event_chain()
                    if e["event_type"] == "activity_outputs_verified"]
    assert len(observations) == 1, "replay must not execute the completed effect twice"


def test_timeout_watchdog_counts_survive_supervisor_restarts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import autotrain_supervision as supervision
    from scripts.autotrain_supervisor_operations import interpret_operation_result
    from slm_training.autoresearch.heal.escalation import EscalationLedger
    from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome

    args = SimpleNamespace(root=tmp_path, loop_id="watchdog", max_cycles=1,
        stop_after_pass=None, hard_backoff_seconds=0, soft_backoff_seconds=0,
        train_version="fixture", steps=1, primary_metric="fixture", continuation_grant=None)
    common = {"root": str(tmp_path)}
    store = CampaignStore("runtime", tmp_path / "loops/watchdog")
    monkeypatch.setattr(supervision, "pre_cycle", lambda *args: {"campaign_id": "existing"})
    monkeypatch.setattr(supervision, "handle_pending", lambda *args: "run")
    monkeypatch.setattr(supervision, "post_cycle", lambda *args: 0)
    timeout = BoundedProcessResult(command=("fixture",), outcome=ProcessOutcome.TIMED_OUT,
        returncode=None, stdout="", stderr="", duration_seconds=1, timed_out=True)
    _, missing = interpret_operation_result(timeout, tmp_path / "absent.json", {"operation": "driver"})
    assert missing is None
    waits = []

    def invoke(payload):
        with ActivityRuntime(store) as runtime:
            monkeypatch.setattr(runtime.cancel_event, "wait", lambda seconds: waits.append(seconds))
            assert supervision.supervise(args, runtime, common,
                run_operation=lambda *args, **kwargs: payload,
                watchdog=supervision.watchdog_no_campaign) == 0

    for _ in range(5):
        invoke(missing)
    history = [row["detail"] for row in store.verify_event_chain()
               if row["event_type"] == "supervisor_driver_progress"]
    assert [row["passes_without_campaign"] for row in history] == [1, 2, 3, 4, 5]
    assert history[-1]["total_no_campaign_passes"] == 5
    assert waits[-1] == 30
    record, = EscalationLedger.load(tmp_path, "watchdog").records.values()
    assert record.status == "escalated" and "total_no_campaign_passes=5" in record.note

    invoke({"returncode": 0, "campaign_id": "existing", "completion": {"fixture": True}})
    invoke(missing)
    # A declared continuation preserves these operational counters across restart.
    invoke({"returncode": 10, "pending": {"schema_version": "driver_pending/v1",
        "outcome": "yielded", "reason": "fixture continuation", "measurement_complete": False,
        "wake": {"predicate": "resume cursor", "source": "fixture", "identity_digest": "a" * 64}}})
    history = [row["detail"] for row in store.verify_event_chain()
               if row["event_type"] == "supervisor_driver_progress"]
    assert len(history) == 7
    assert history[-2]["passes_without_campaign"] == 0
    assert history[-1]["passes_without_campaign"] == 1
    assert history[-1]["total_no_campaign_passes"] == 6


@pytest.mark.parametrize("boundary", ["before launch", "while running"])
def test_operation_rejects_measured_environment_drift(tmp_path, monkeypatch, boundary):
    """Identity-only fixture: publication fencing has separate real-process coverage."""
    from contextlib import nullcontext
    from scripts import autotrain_supervisor_operations as operations
    from scripts import merge_verification_evidence as evidence

    environment = {"fixture_environment": "original"}
    request = {"operation": "closeout", "root": str(tmp_path), "cwd": str(tmp_path),
        "loop_id": "fixture", "source_digest": "a" * 64,
        "environment_digest": digest(environment)}
    request_path, output_path = tmp_path / "request.json", tmp_path / "result.json"
    request_path.write_text(json.dumps(request))
    monkeypatch.setattr(evidence, "environment_identity", lambda: dict(environment))
    monkeypatch.setattr(operations, "operation_publication_scope", lambda *a: nullcontext())
    effects = []

    def closeout(log):
        effects.append("closeout")
        environment["fixture_environment"] = "changed"

    if boundary == "before launch":
        environment["fixture_environment"] = "changed"
    with pytest.raises(ValueError, match=f"operation environment changed {boundary}"):
        operations.operation_main(request_path, output_path, source_identity=lambda _: "a" * 64,
            load_continuous=lambda: None, handle_hard_pending=lambda *a: None,
            write_family_closures=closeout)
    assert effects == ([] if boundary == "before launch" else ["closeout"])
    assert not output_path.exists(), "drifted execution must not publish a success envelope"


@pytest.mark.parametrize("drift", ["source", "environment"])
def test_parent_discards_child_result_when_identity_drifts_before_commit(
    tmp_path, monkeypatch, drift,
):
    from pathlib import Path
    from slm_training.harness_core.bounded_process import (
        BoundedProcessResult,
        ProcessOutcome,
    )

    from scripts import run_autotrain_supervisor as supervisor
    from scripts.merge_verification_evidence import digest

    identity = {"source": "a" * 64}
    environment = {"fixture_environment": True}
    monkeypatch.setattr(supervisor, "_source_identity", lambda _: identity["source"])
    monkeypatch.setattr(
        "scripts.merge_verification_evidence.environment_identity",
        lambda: dict(environment),
    )
    request = {
        "operation": "closeout",
        "loop_id": "parent-race",
        "root": str(tmp_path),
        "cwd": str(tmp_path),
        "source_digest": "a" * 64,
        "environment_digest": digest(environment),
    }
    outputs = []

    def child_finishes_then_source_drifts(runtime, lease, argv, *, cwd, **kwargs):
        request_path = Path(argv[argv.index("--operation-request") + 1])
        output_path = Path(argv[argv.index("--operation-output") + 1])
        executed = json.loads(request_path.read_text())
        output_path.write_text(json.dumps({
            "schema_version": "supervisor_operation/v1",
            "operation": "closeout",
            "request_digest": digest(executed),
            "payload": {"report": {"completed": True}},
        }))
        outputs.append(output_path)
        if drift == "source":
            identity["source"] = "b" * 64
        else:
            environment["fixture_environment"] = False
        return BoundedProcessResult(
            command=tuple(argv),
            outcome=ProcessOutcome.COMPLETED,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(ActivityRuntime, "run", child_finishes_then_source_drifts)
    events = []
    store = CampaignStore("supervisor-parent-race", tmp_path / "store")
    with ActivityRuntime(store) as runtime:
        assert supervisor._run_operation(
            runtime, request, sequence=0, log_event=events.append,
        ) is None
        states = runtime.snapshot()

    assert len(outputs) == 1 and not outputs[0].exists()
    state, = states.values()
    assert state.status == "cancelled"
    assert not state.outputs
    assert any(event["event"] == "operation_identity_drift" for event in events)
    assert not any(
        row["event_type"] == "activity_outputs_verified"
        for row in store.verify_event_chain()
    )
