"""Runtime grant and source-verification successor regressions."""

from types import SimpleNamespace

from scripts import autotrain_verification as owner
from scripts.merge_verification_evidence import digest
from tests.test_scripts.test_autotrain_verification_successor import (
    _dependency,
    _materializer,
    _next_dependency,
    _park,
)
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ResourceGrant


def test_invalid_runtime_grant_observation_activates_config_successor(monkeypatch):
    from slm_training.autoresearch.heal import isolation

    event = {"experiment_id": "repair"}
    dependency = {"activity_id": "old-verifier"}
    runtime = SimpleNamespace(claim_next=lambda **_kwargs: object())
    events = []
    monkeypatch.setattr(owner, "_active_dependency", lambda *_args: (event, dependency))
    monkeypatch.setattr(owner, "dependency_plan", lambda _dependency: {"identity": "old"})
    monkeypatch.setattr(owner, "wake_repair", lambda *_args: False)
    monkeypatch.setattr(owner, "register_dependency", lambda *_args: None)
    monkeypatch.setattr(isolation, "probe_isolation", lambda: SimpleNamespace(available=True))
    monkeypatch.setattr(owner, "execute_release_attempt", lambda *_args: {
        "status": "invalid_evidence", "reason": "ambiguous_agentv_runtime_grant",
    })
    monkeypatch.setattr(owner, "_successor_after_drift", lambda *_args: (
        {"status": "successor_activated", "activity_id": "new-verifier"}, None,
    ))

    result = owner._run_source_verification(runtime, event, {}, events.append, object())

    assert result == {"status": "successor_activated", "activity_id": "new-verifier"}
    assert events == [{"event": "source_verification_successor", **result}]


def test_runtime_config_drift_rebinds_after_old_grant_failure(tmp_path, monkeypatch):
    from scripts import autotrain_verification_successor as successor_owner
    from slm_training.autoresearch.heal import recovery_dispatch

    old_root, new_root = tmp_path / "old", tmp_path / "new"
    dependency = {"runtime_roots": [str(old_root)]}
    common = {
        "repair_config": str(tmp_path / "recovery.json"),
        "repair_config_digest": "a" * 64,
        "loop_id": "loop",
    }
    config = SimpleNamespace(runtime_roots=(str(new_root),))
    monkeypatch.setattr(recovery_dispatch, "load_recovery_config", lambda *_a, **_kw: config)
    observed = []
    monkeypatch.setattr(successor_owner, "plan_successor",
                        lambda *args: observed.append(args) or {"status": "successor_activated"})
    error = ValueError("ambiguous_agentv_runtime_grant")

    result, original_error = owner._successor_after_drift(
        object(), {"event": "request"}, dependency, common, error,
    )

    assert result == {"status": "successor_activated"}
    assert original_error is error
    assert len(observed) == 1


def test_source_verification_scan_rotates_small_backlogs(monkeypatch):
    events = [
        {"event_type": "source_verification_requested", "experiment_id": str(i)}
        for i in range(10)
    ]
    states = {str(i): SimpleNamespace(status="waiting_dependency") for i in range(10)}
    runtime = SimpleNamespace(
        store=SimpleNamespace(verify_event_chain=lambda: events),
        snapshot=lambda: states,
    )
    seen = []
    monkeypatch.setattr(owner, "_attempt_count", lambda _store, _states, event: int(event["experiment_id"]) >= 2)
    monkeypatch.setattr(
        owner,
        "_run_source_verification",
        lambda _runtime, event, *_args: seen.append(event["experiment_id"]),
    )

    owner.drain_source_verification(runtime, {}, lambda _: None, cycle=1)
    assert seen == [str(i) for i in range(10)]
    seen.clear()
    owner.drain_source_verification(runtime, {}, lambda _: None, cycle=2)
    assert seen == [str(i) for i in range(2, 10)] + ["0", "1"]


def test_successor_rebinds_authenticated_runtime_grants(tmp_path, monkeypatch):
    from scripts import autotrain_verification_successor as successor_owner
    from slm_training.autoresearch.heal import recovery_dispatch

    predecessor, binding = _dependency(tmp_path, monkeypatch)
    event_runtime = tmp_path / "new-runtime"
    event_runtime.mkdir()
    (event_runtime / "runtime.json").write_text('{"runtime":"updated"}')
    roots = (str(event_runtime),)
    current = {**binding, "runtime_identity": owner.runtime_identity((event_runtime,))}
    gate = SimpleNamespace(identity=digest(current), binding=current)
    event = None
    grant = ResourceGrant(interrupt_seconds=65, total_seconds=200, max_attempts=3)
    monkeypatch.setattr(recovery_dispatch, "load_recovery_config", lambda *_a, **_kw:
                        SimpleNamespace(runtime_roots=roots, source_verification_grant=grant))
    monkeypatch.setattr(successor_owner, "_materialize", _materializer(
        tmp_path, gate, predecessor["request_digest"], predecessor["proposal_digest"], roots,
    ))
    monkeypatch.setattr(recovery_dispatch, "_verification_dependency",
                        lambda request, proposal, gate, grant, roots:
                        _next_dependency(predecessor, gate, grant, roots))
    with ActivityRuntime(CampaignStore("runtime", tmp_path / "loops" / "loop")) as runtime:
        event = _park(runtime, predecessor)
        bindings = iter((binding, current))
        monkeypatch.setattr(owner, "verification_binding", lambda *a, **kw: next(bindings))
        result = successor_owner.plan_successor(runtime, event, predecessor, {
            "repair_config": "/approved/recovery.json", "repair_config_digest": "1" * 64,
            "loop_id": "loop",
        })
        successor_event = next(row for row in reversed(runtime.store.verify_event_chain())
                               if row["event_type"] == "source_verification_requested"
                               and row["detail"]["dependency_digest"] != event["detail"]["dependency_digest"])
        successor = owner.load_dependency(runtime.store, successor_event)
        assert result["status"] == "successor_activated"
        assert successor["runtime_roots"] == list(roots)
        assert successor["request_digest"] == predecessor["request_digest"]
        assert successor["proposal_digest"] == predecessor["proposal_digest"]
        assert successor["verification_identity"] != predecessor["verification_identity"]
        assert successor["grant"]["total_seconds"] == predecessor["grant"]["total_seconds"]
        assert successor["grant"]["max_attempts"] == predecessor["grant"]["max_attempts"]
        assert runtime.snapshot()[predecessor["activity_id"]].status == "cancelled"
