"""Imported-owner contracts; fixture execution is not model improvement evidence."""

from __future__ import annotations

import hashlib
import json
import sys

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.runtime.activity_runtime import (
    ActivityRuntime, ControllerBusy, StaleLease, process_identity,
)
from slm_training.harness_core.activity_contract import (
    ActivityEvent, ActivityLease, ActivityOutcome, ActivitySpec, ResourceCapacity,
    ResourceGrant, WakeCondition, contract_digest, reduce_activity,
)
from slm_training.autoresearch.storage import CampaignStore


def spec(name="test", **kwargs):
    return ActivitySpec(activity_id=name, family=name, kind="eval",
        source_digest="a" * 40, environment_digest="b" * 64, input_digest="c" * 64,
        output_namespace=f"runs/{name}", grant=ResourceGrant(
            cpu_slots=1, memory_mb=16, interrupt_seconds=1, kill_grace_seconds=.1,
            total_seconds=10, finalization_reserve_seconds=.5), **kwargs)


def publish(runtime, lease):
    output = runtime.attempt_dir(lease) / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"operational_fixture": true}\n')
    return {output.name: hashlib.sha256(output.read_bytes()).hexdigest()}


def test_replay_fence_outputs_and_epoch_recovery(tmp_path):
    store = CampaignStore("runtime", tmp_path)
    now = [100.]
    with ActivityRuntime(store, controller_clock=lambda: now[0]) as runtime:
        runtime.register(spec())
        lease = runtime.claim_next(capabilities={"local_process"})
        with pytest.raises(ControllerBusy):
            with ActivityRuntime(store):
                pass
        assert lease is not None
        with pytest.raises(ValueError, match="nonempty"):
            runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED, spent_seconds=.01)
        with pytest.raises(StaleLease):
            runtime.heartbeat(lease.model_copy(update={"token": "forged"}))
        runtime.heartbeat(lease)
        assert runtime.snapshot()["test"].last_progress_digest == ""
        first = runtime.snapshot()
        assert first == runtime.snapshot()
    with ActivityRuntime(store, controller_clock=lambda: now[0]) as runtime:
        with pytest.raises(StaleLease):
            with runtime.publication(lease):
                pytest.fail("stale owner published")
        state = runtime.snapshot()["test"]
        assert state.charged_seconds == pytest.approx(1.1)
        assert state.status == "waiting_retry"
        now[0] += 3
        replacement = runtime.claim_next(capabilities={"local_process"})
        assert replacement.epoch != lease.epoch
        assert replacement.generation == 2
        outputs = publish(runtime, replacement)
        runtime.finish(replacement, outcome=ActivityOutcome.SUCCEEDED, outputs=outputs, spent_seconds=.2)
        assert runtime.snapshot()["test"].charged_seconds == pytest.approx(1.3)
        assert runtime.claim_next(capabilities={"local_process"}) is None


def test_scoped_waits_fairness_and_no_retry_budget_reset(tmp_path):
    store = CampaignStore("fair", tmp_path)
    now = [100.]
    with ActivityRuntime(store, controller_clock=lambda: now[0]) as runtime:
        runtime.register(spec("remote", capabilities=("paid_unavailable",)))
        runtime.register(spec("healthy"))
        lease = runtime.claim_next(capabilities={"local_process"})
        assert lease.activity_id == "healthy"
        assert runtime.snapshot()["remote"].status == "waiting_capability"
        runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED, outputs=publish(runtime, lease), spent_seconds=.1)
        for index in range(3):
            lease = runtime.claim_next(capabilities={"paid_unavailable"})
            assert lease.activity_id == "remote"
            runtime.finish(lease, outcome=ActivityOutcome.RETRY, spent_seconds=.1)
            now[0] += 61
        state = runtime.snapshot()["remote"]
        assert state.status == "waiting_repair" and state.action == "diagnose_exhausted"
        with pytest.raises(ValueError, match="successor"):
            runtime.wake("remote", evidence=state.wake)
        assert runtime.claim_next(capabilities={"paid_unavailable"}) is None
        assert runtime.register(spec("remote", capabilities=("paid_unavailable",))).attempts == 3


@pytest.mark.parametrize("outcome,action", [
    (ActivityOutcome.WALL_BUDGET, "resume_evaluation"),
    (ActivityOutcome.SUITE_VOLUME, "rebuild_data"),
    (ActivityOutcome.MIXED_BUDGET, "calibrate_budget"),
    (ActivityOutcome.UNKNOWN_FAILURE, "diagnose"),
        (ActivityOutcome.CODE_FAILURE, "repair_harness"),
        (ActivityOutcome.FORMAL_INFRASTRUCTURE, "repair_formal"),
        (ActivityOutcome.FORMAL_CONTRADICTION, "review_hypothesis"),
        (ActivityOutcome.DELIVERY_FAILURE, "deliver_stack"),
])
def test_typed_cause_state_action_and_wake_agree(tmp_path, outcome, action):
    with ActivityRuntime(CampaignStore("causes", tmp_path)) as runtime:
        item = spec()
        runtime.register(item)
        lease = runtime.claim_next(capabilities={"local_process"})
        wake = WakeCondition(predicate="original_reproducer_passes", source="independent_verifier",
                             identity_digest=contract_digest(item))
        state = runtime.finish(lease, outcome=outcome, spent_seconds=.01, wake=wake)
        assert state.action == action and state.status == "waiting_repair"
        assert runtime.actions(capabilities={"local_process"})[0]["action"] == action
        with pytest.raises(ValueError):
            runtime.wake("test", evidence=wake.model_copy(update={"identity_digest": "0" * 64}))
        assert runtime.wake("test", evidence=wake).status == "runnable"


def test_reducer_rejects_unvalidated_lease_and_budget_payloads():
    item = spec()
    state = reduce_activity(None, ActivityEvent(operation="register", spec=item,
                           activity_id=item.activity_id, sequence=0, at=1))
    lease = ActivityLease(activity_id="wrong", attempt_id="attempt", epoch="epoch",
        generation=1, token="token", owner_identity=process_identity(), expires_at=10)
    with pytest.raises(ValueError, match="identity"):
        reduce_activity(state, ActivityEvent(operation="claim", activity_id="test", at=2, sequence=1, lease=lease))
    for field, value in (("spent_seconds", -1), ("at", float("nan")), ("spent_seconds", float("inf"))):
        with pytest.raises(ValidationError):
            ActivityEvent.model_validate({"operation": "finish", "activity_id": "test", "sequence": 2,
                                          "at": 3, field: value})
    with pytest.raises(ValidationError):
        ActivitySpec.model_validate({**item.model_dump(), "source_digest": "latest"})


def test_real_child_artifact_then_terminal_replay(tmp_path):
    store = CampaignStore("real", tmp_path)
    with ActivityRuntime(store) as runtime:
        runtime.register(spec())
        lease = runtime.claim_next(capabilities={"local_process"})
        path = runtime.attempt_dir(lease) / "result.json"
        result = runtime.run(lease, [sys.executable, "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{}')", str(path)], cwd=tmp_path)
        assert result.returncode == 0 and not result.timed_out
        runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED,
            outputs={"result.json": hashlib.sha256(path.read_bytes()).hexdigest()}, spent_seconds=result.duration_seconds)
    with ActivityRuntime(store) as resumed:
        assert resumed.snapshot()["test"].status == "succeeded"
        assert resumed.claim_next(capabilities={"local_process"}) is None


def test_control_child_waits_for_committed_process_registration(tmp_path, monkeypatch):
    import time

    marker = tmp_path / "registered"
    with ActivityRuntime(CampaignStore("registration", tmp_path)) as runtime:
        item = spec(capabilities=("local_process", "controller_publication"))
        runtime.register(item)
        lease = runtime.claim_next(capabilities=set(item.capabilities))
        heartbeat = runtime.heartbeat

        def delayed_registration(current, **kwargs):
            if "worker_pid" in kwargs:
                time.sleep(.15)
            result = heartbeat(current, **kwargs)
            marker.write_text("committed")
            return result

        monkeypatch.setattr(runtime, "heartbeat", delayed_registration)
        result = runtime.run(lease, [sys.executable, "-c",
            "from pathlib import Path; import sys; assert Path(sys.argv[1]).read_text() == 'committed'",
            str(marker)], cwd=tmp_path)
        assert result.returncode == 0 and not result.timed_out
        assert runtime.snapshot()["test"].worker_identity


def test_missing_and_escaping_output_cannot_complete(tmp_path):
    with ActivityRuntime(CampaignStore("escape", tmp_path)) as runtime:
        runtime.register(spec())
        lease = runtime.claim_next(capabilities={"local_process"})
        for outputs in ({"missing": "0" * 64}, {"../../outside": "0" * 64}):
            with pytest.raises(ValueError):
                runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED, outputs=outputs, spent_seconds=.01)
        assert runtime.snapshot()["test"].status == "running"


def test_dependencies_and_capacity_are_enforced(tmp_path):
    with ActivityRuntime(CampaignStore("capacity", tmp_path), capacity=ResourceCapacity(cpu_slots=1, memory_mb=16)) as runtime:
        runtime.register(spec("parent"))
        runtime.register(spec("child", dependencies=("parent",)))
        lease = runtime.claim_next(capabilities={"local_process"})
        assert lease.activity_id == "parent"
        assert runtime.claim_next(capabilities={"local_process"}) is None
        assert runtime.snapshot()["child"].status == "waiting_dependency"
        runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED, outputs=publish(runtime, lease), spent_seconds=.01)
        assert runtime.claim_next(capabilities={"local_process"}).activity_id == "child"


def test_targeted_claim_cancellation_and_current_status(tmp_path):
    runtime = ActivityRuntime(CampaignStore("targeted", tmp_path))
    with runtime:
        runtime.register(spec("first"))
        runtime.register(spec("second"))
        assert runtime.controller_status()["alive"]
        lease = runtime.claim_next(capabilities={"local_process"}, activity_id="second")
        assert lease.activity_id == "second"
        runtime.cancel_all(reason="explicit_user_stop")
        assert all(s.status == "cancelled" for s in runtime.snapshot().values())
        with pytest.raises(StaleLease):
            with runtime.publication(lease):
                pytest.fail("cancelled owner published")
    assert not runtime.controller_status()["alive"]


def test_crash_after_verified_output_recovers_without_new_attempt(tmp_path, monkeypatch):
    store = CampaignStore("recover_output", tmp_path)
    append = store.append_event
    with ActivityRuntime(store) as runtime:
        runtime.register(spec())
        lease = runtime.claim_next(capabilities={"local_process"})
        outputs = publish(runtime, lease)
        def crash(event_type, **kwargs):
            if event_type == "activity_transition" and kwargs["detail"]["operation"] == "finish":
                raise OSError("injected disk full after output verification")
            return append(event_type, **kwargs)
        monkeypatch.setattr(store, "append_event", crash)
        with pytest.raises(OSError):
            runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED, outputs=outputs, spent_seconds=.1)
    monkeypatch.setattr(store, "append_event", append)
    with ActivityRuntime(store) as runtime:
        state = runtime.snapshot()["test"]
        assert state.status == "succeeded" and state.attempts == 1
        assert state.outputs == outputs and state.charged_seconds == pytest.approx(.1)


def test_real_controller_process_crash_revokes_old_lease(tmp_path):
    from slm_training.harness_core.bounded_process import run_bounded_process
    item = spec().model_dump(mode="json")
    code = ("import os,json; from pathlib import Path; "
            "from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime; "
            "from slm_training.harness_core.activity_contract import ActivitySpec; "
            "from slm_training.autoresearch.storage import CampaignStore; "
            f"r=ActivityRuntime(CampaignStore('crash', {str(tmp_path)!r})); r.__enter__(); "
            f"r.register(ActivitySpec.model_validate(json.loads({json.dumps(item)!r}))); "
            "r.claim_next(capabilities={'local_process'}); os._exit(17)")
    result = run_bounded_process([sys.executable, "-c", code], interrupt_after_seconds=5, kill_grace_seconds=.1)
    assert result.returncode == 17
    with ActivityRuntime(CampaignStore("crash", tmp_path)) as runtime:
        state = runtime.snapshot()["test"]
        assert state.status == "waiting_retry" and state.attempts == 1
        assert state.charged_seconds == pytest.approx(1.1)


def test_configured_repair_factory_queues_actual_activity_on_finish_and_restart(tmp_path):
    def factory(state):
        return spec("repair-" + state.spec.activity_id).model_copy(update={
            "kind": "repair", "capabilities": ("approved_source_executor",),
            "input_digest": contract_digest(state.spec)})
    store = CampaignStore("repair_dispatch", tmp_path)
    with ActivityRuntime(store, repair_factory=factory) as runtime:
        runtime.register(spec())
        lease = runtime.claim_next(capabilities={"local_process"})
        runtime.finish(lease, outcome=ActivityOutcome.CODE_FAILURE, spent_seconds=.1,
            wake=WakeCondition(predicate="original_reproducer_passes", source="independent_verifier",
                               identity_digest=contract_digest(spec())))
        assert runtime.snapshot()["repair-test"].spec.kind == "repair"
        assert runtime.claim_next(capabilities={"local_process"}) is None
    with ActivityRuntime(store, repair_factory=factory) as runtime:
        assert len(runtime.snapshot()) == 2
        lease = runtime.claim_next(capabilities={"approved_source_executor"})
        assert lease.activity_id == "repair-test"
        assert runtime.snapshot()["test"].status == "waiting_repair"


def test_projection_cache_cannot_authorize_tampered_history(tmp_path):
    store = CampaignStore("tamper", tmp_path)
    with ActivityRuntime(store) as runtime:
        runtime.register(spec())
        runtime.snapshot()
        path = store.root / "events.jsonl"
        original = path.read_bytes()
        path.write_bytes(original.replace(b'"family":"test"', b'"family":"fake"'))
        with pytest.raises(RuntimeError, match="digest"):
            runtime.snapshot()
        path.write_bytes(original)
        assert runtime.snapshot()["test"].status == "runnable"


def test_model_copy_cannot_forge_negative_charges():
    event = ActivityEvent(operation="register", spec=spec(), activity_id="test", sequence=0, at=1)
    with pytest.raises(ValidationError):
        reduce_activity(None, event.model_copy(update={"spent_seconds": -1}))
