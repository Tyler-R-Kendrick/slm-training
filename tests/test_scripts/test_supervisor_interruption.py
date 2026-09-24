"""Interrupted wrappers retain the original operation, cursor and finite grant."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import autotrain_cycle_execution as execution
from scripts import autotrain_supervisor_operations as operations
from scripts.autotrain_pending import recover_driver_request
from slm_training.autoresearch.heal.operation_recovery import record_operation_failure
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ActivityOutcome, ResourceGrant, contract_digest
from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome
from scripts.merge_verification_evidence import digest


def _request(tmp_path, grant=None):
    grant = grant or ResourceGrant(interrupt_seconds=1, kill_grace_seconds=0,
                                   total_seconds=4, max_attempts=2)
    return dict(operation="driver", loop_id="test-loop", cwd=str(tmp_path),
                root=str(tmp_path / "campaigns"), source_digest="a" * 64,
                environment_digest="b" * 64, predecessor_campaign_id="before",
                driver_argv=["--continuation-grant", grant.model_dump_json()])


def _timeout(*, stalled=False):
    return BoundedProcessResult(command=("fixture",), outcome=ProcessOutcome.TIMED_OUT,
        returncode=-2, stdout="", stderr="", duration_seconds=1,
        timed_out=True, progress_stalled=stalled)


@pytest.mark.parametrize("stalled", [False, True])
def test_timeout_retries_same_operation_but_stall_requires_diagnosis(tmp_path, monkeypatch, stalled):
    request = _request(tmp_path)
    store = CampaignStore("runtime", tmp_path / "runtime")
    clock = [100.0]
    monkeypatch.setattr(operations, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    logs = []
    with ActivityRuntime(store, controller_clock=lambda: clock[0]) as runtime:
        def run(*args, **kwargs):
            clock[0] += 1
            return _timeout(stalled=stalled)
        monkeypatch.setattr(runtime, "run", run)
        assert operations.run_operation(runtime, request, sequence=1, log_event=logs.append) is None
        state, = runtime.snapshot().values()
        assert state.status == ("waiting_repair" if stalled else "waiting_retry")
        assert state.attempts == 1 and state.charged_seconds == 1
        assert state.outputs == {}
        assert logs[-1]["outcome"] == ("unknown_failure" if stalled else "retry")
        failure = next(e for e in store.verify_event_chain()
                       if e["event_type"] == "operation_repair_requested")
        assert failure["detail"]["pending"]["observed_outcome"] == (
            "unknown_failure" if stalled else "wall_budget")


@pytest.mark.parametrize("exhaustion", ["attempts", "seconds"])
def test_restart_and_new_predecessor_cannot_reset_exhausted_grant(tmp_path, monkeypatch, exhaustion):
    grant = ResourceGrant(interrupt_seconds=1, kill_grace_seconds=0,
        total_seconds=10 if exhaustion == "attempts" else 2,
        max_attempts=2 if exhaustion == "attempts" else 9)
    request = _request(tmp_path, grant)
    store = CampaignStore("runtime", tmp_path / "runtime")
    clock, calls, charges = [100.0], [], []
    monkeypatch.setattr(operations, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    for invocation in range(4):
        clock[0] += 10  # Retry backoff advances without sleeping.
        with ActivityRuntime(store, controller_clock=lambda: clock[0]) as runtime:
            def run(lease, *args, **kwargs):
                calls.append(lease.activity_id)
                clock[0] += 1
                return _timeout()
            monkeypatch.setattr(runtime, "run", run)
            observed = {**request, "predecessor_campaign_id": f"observed-{invocation}"}
            if invocation:
                assert recover_driver_request(runtime, observed)["predecessor_campaign_id"] == "observed-0"
            assert operations.run_operation(runtime, observed, sequence=invocation,
                                            log_event=lambda _: None) is None
            state, = runtime.snapshot().values()
            assert state.spec.grant == grant
            charges.append(state.charged_seconds)
    expected = 2 if exhaustion == "attempts" else 1
    assert len(calls) == expected and len(set(calls)) == 1
    assert state.attempts == expected and state.charged_seconds == expected
    assert state.status == "waiting_repair" and state.action == "diagnose_exhausted"
    assert charges == sorted(charges)


def _claim(runtime, request):
    state = operations._operation_state(runtime, request, 1, contract_digest(request))
    lease = runtime.claim_next(activity_id=state.spec.activity_id,
                               capabilities={"local_process", "controller_publication"})
    assert lease is not None
    return lease


def _interrupted(runtime, request, lease):
    record_operation_failure(runtime, lease, request, _timeout(), outcome=ActivityOutcome.WALL_BUDGET)
    runtime.finish(lease, outcome=ActivityOutcome.RETRY, spent_seconds=1)
    return {**request, "lease": lease.model_dump(mode="json")}


def test_active_interrupted_cycle_validates_identity_and_reuses_cursor(tmp_path, monkeypatch):
    from tests.test_autoresearch.test_driver_cycle_recovery import _fixture, _resume
    from scripts.autotrain_cycle_context import CycleJournal

    f = _fixture(tmp_path, monkeypatch, arms=2)
    request = _request(tmp_path)
    store = CampaignStore("runtime", f.root / "loops/test-loop")
    with ActivityRuntime(store) as runtime:
        request = _interrupted(runtime, request, _claim(runtime, request))
        before = CycleJournal(f.store, f.value).state
        assert execution.reconcile_interrupted_operation(
            request, store, f.cwd, f.root, "test-loop", f.store.campaign_id) is None
        assert CycleJournal(f.store, f.value).state == before and not f.calls
        with pytest.raises(ValueError, match="workspace/loop mismatch"):
            execution.reconcile_interrupted_operation(
                request, store, f.cwd, f.root, "wrong-loop", f.store.campaign_id)
    assert _resume(f)["campaign_id"] == f.store.campaign_id


@pytest.mark.parametrize("second_timeout", [False, True])
def test_completed_retirement_recovers_lost_envelope_without_reexecution(tmp_path, monkeypatch, second_timeout):
    from tests.test_autoresearch.test_driver_cycle_recovery import _fixture, _resume

    f = _fixture(tmp_path, monkeypatch, arms=2)
    request = _request(tmp_path, ResourceGrant(interrupt_seconds=1, kill_grace_seconds=0,
                                              total_seconds=10, max_attempts=3))
    store = CampaignStore("runtime", f.root / "loops/test-loop")
    clock = [100.0]
    with ActivityRuntime(store, controller_clock=lambda: clock[0]) as runtime:
        lease = _claim(runtime, request)
        assert _resume(f)["outcome"] == "yielded"
        assert _resume(f) == f.store.campaign_id
        interrupted = _interrupted(runtime, request, lease)
        if second_timeout:
            clock[0] += 10
            second_lease = _claim(runtime, request)
            assert second_lease.activity_id == lease.activity_id and second_lease.token != lease.token
            interrupted = _interrupted(runtime, request, second_lease)
        calls, finalized = list(f.calls), list(f.final_calls)
        def unexpected(*args, **kwargs):
            pytest.fail("completed interrupted campaign executed driver main again")
        driver = SimpleNamespace(main=unexpected, _latest_cycle=lambda *_: (1, f.store.campaign_id))
        for _ in range(2):
            result = execution.driver_operation(interrupted, driver, f.cwd, f.root, "test-loop")
            assert result["returncode"] == 0 and result["campaign_id"] == f.store.campaign_id
            assert result["handoff_digest"] == result["completion"]["outputs"]["cycle_handoff.json"]
        assert f.calls == calls and f.final_calls == finalized
        assert not list(store.root.glob("attempts/**/result.json"))


def test_missing_cursor_parks_instead_of_starting_a_new_campaign(tmp_path, monkeypatch):
    request = _request(tmp_path)
    root = tmp_path / "campaigns"
    store = CampaignStore("runtime", root / "loops/test-loop")
    with ActivityRuntime(store) as runtime:
        request = _interrupted(runtime, request, _claim(runtime, request))
        def unexpected(*args, **kwargs):
            pytest.fail("missing interrupted cursor started a fresh driver")
        driver = SimpleNamespace(main=unexpected, _latest_cycle=lambda *_: (0, None))
        result = execution.driver_operation(request, driver, tmp_path, root, "test-loop")
        assert result["returncode"] == 10 and result["campaign_id"] is None
        assert result["pending"]["measurement_complete"] is False
        assert result["pending"]["reason"] == "interrupted_driver_has_no_reconcilable_cycle"
        assert {p.name for p in root.iterdir()} == {"loops"}


def test_interrupted_cycle_restart_preserves_exhausted_logical_time(tmp_path, monkeypatch):
    from tests.test_autoresearch.test_driver_cycle_recovery import _fixture, _resume
    from scripts.autotrain_cycle_context import CycleJournal

    f = _fixture(tmp_path, monkeypatch)
    cursor = CycleJournal(f.store, f.value)
    cursor.state["spent_seconds"] = 175
    cursor.save()
    request = _request(tmp_path)
    store = CampaignStore("runtime", f.root / "loops/test-loop")
    with ActivityRuntime(store) as runtime:
        request = _interrupted(runtime, request, _claim(runtime, request))
    for _ in range(2):
        assert execution.reconcile_interrupted_operation(
            request, store, f.cwd, f.root, "test-loop", f.store.campaign_id) is None
        assert _resume(f)["reason"] == "driver_logical_grant_exhausted"
        assert CycleJournal(f.store, f.value).state["spent_seconds"] >= 175
    assert not f.calls and not f.final_calls


def test_expired_lease_after_child_output_retries_without_adopting_stale_result(
    tmp_path, monkeypatch
):
    request = {
        "operation": "inspect",
        "loop_id": "test-loop",
        "cwd": str(tmp_path),
        "root": str(tmp_path / "campaigns"),
        "source_digest": "a" * 64,
        "environment_digest": digest({"fixture_environment": True}),
    }
    store = CampaignStore("runtime", tmp_path / "runtime")
    clock, launches, logs = [100.0], [], []
    monkeypatch.setattr(
        operations, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    from scripts import merge_verification_evidence as evidence

    monkeypatch.setattr(evidence, "source_identity", lambda _: "a" * 64)
    monkeypatch.setattr(
        evidence, "environment_identity", lambda: {"fixture_environment": True}
    )
    with ActivityRuntime(store, controller_clock=lambda: clock[0]) as runtime:
        def run(lease, argv, **kwargs):
            launches.append(lease.attempt_id)
            execution_request = json.loads(Path(argv[-3]).read_text())
            payload = {"report": {"attempt": len(launches)}}
            Path(argv[-1]).write_text(json.dumps({
                "schema_version": "supervisor_operation/v1",
                "request_digest": digest(execution_request),
                "operation": "inspect",
                "payload": payload,
            }, sort_keys=True))
            if len(launches) == 1:
                # Child finished with output, but the parent's lease expires
                # before it can validate and commit that output.
                clock[0] = lease.expires_at + 0.01
            return BoundedProcessResult(
                tuple(argv), ProcessOutcome.COMPLETED, 0, "", "", 0.01
            )

        monkeypatch.setattr(runtime, "run", run)
        assert operations.run_operation(runtime, request, sequence=1, log_event=logs.append) is None
        state, = runtime.snapshot().values()
        assert state.status == "waiting_retry"
        assert state.outputs == {}
        stale = [e for e in store.verify_event_chain()
                 if e["event_type"] == "operation_stale_lease_reconciled"]
        assert len(stale) == 1
        assert stale[0]["detail"]["committed_receipt_replayed"] is False
        first_attempt = stale[0]["detail"]["attempt_id"]
        assert not any(
            e["event_type"] == "activity_outputs_verified"
            and e["detail"]["lease"]["attempt_id"] == first_attempt
            for e in store.verify_event_chain()
        )

        clock[0] = state.retry_at + 1
        payload = operations.run_operation(runtime, request, sequence=2, log_event=logs.append)
        assert payload == {"report": {"attempt": 2}}
        committed, = [e for e in store.verify_event_chain()
                      if e["event_type"] == "activity_outputs_verified"]
        assert committed["detail"]["lease"]["attempt_id"] == launches[1]
        assert runtime.snapshot()[state.spec.activity_id].status == "succeeded"

        # Existing committed-receipt replay must not execute the child again.
        assert operations.run_operation(runtime, request, sequence=1, log_event=logs.append) == payload
        assert len(launches) == 2
        finishes = [e for e in store.verify_event_chain()
                    if e["event_type"] == "activity_transition"
                    and e["experiment_id"] == state.spec.activity_id
                    and e["detail"]["operation"] in {"finish", "reconcile_finish"}
                    and e["detail"].get("outcome") == "succeeded"]
        assert len(finishes) == 1


def test_expired_lease_cannot_queue_source_verification(tmp_path):
    from scripts.autotrain_pending import verification_wait
    from slm_training.autoresearch.runtime.activity_runtime import StaleLease
    from slm_training.harness_core.activity_contract import ActivitySpec, ResourceGrant, WakeCondition

    clock = [100.0]
    store = CampaignStore("runtime", tmp_path / "runtime")
    with ActivityRuntime(store, controller_clock=lambda: clock[0]) as runtime:
        runtime.register(ActivitySpec(
            activity_id="repair-operation",
            family="repair",
            kind="control",
            source_digest="a" * 64,
            environment_digest="b" * 64,
            input_digest="c" * 64,
            output_namespace="attempts/repair-operation",
            capabilities=("local_process", "controller_publication"),
            grant=ResourceGrant(interrupt_seconds=1, kill_grace_seconds=0,
                                total_seconds=3, max_attempts=1),
        ))
        lease = runtime.claim_next(
            capabilities={"local_process", "controller_publication"},
            activity_id="repair-operation",
        )
        assert lease is not None
        clock[0] = lease.expires_at + 0.01
        dependency = {
            "schema_version": "repair_verification_dependency/v1",
            "wake": WakeCondition(
                predicate="verify proposal",
                source="independent_repair_verification",
                identity_digest="d" * 64,
            ).model_dump(mode="json"),
        }
        with pytest.raises(StaleLease):
            verification_wait(runtime, lease, {
                "agent_repairs": [{
                    "status": "waiting_verification",
                    "verification_dependency": dependency,
                }],
            })
        assert not any(
            row["event_type"] == "source_verification_requested"
            for row in store.verify_event_chain()
        )
