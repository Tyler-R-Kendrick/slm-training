"""Driver tail crash, finite-grant and finalization continuation contracts."""

from types import SimpleNamespace
import time

import pytest

from scripts import autotrain_cycle_context as context
from scripts import autotrain_cycle_execution as execution
from scripts import autotrain_cycle_prepare as prepare
from scripts import autoresearch, run_autotrain_continuous as driver
from tests.test_autoresearch.test_driver_cycle_continuation import _fixture, _resume


def test_explicit_logical_grant_survives_actual_driver_reentry(tmp_path, monkeypatch):
    from tests.test_autoresearch import test_driver_cycle_continuation as fixture
    from slm_training.autoresearch.schemas import CampaignBudget
    from slm_training.harness_core.activity_contract import ResourceGrant
    from slm_training.levers import INTERRUPT_AFTER_SECONDS

    legacy = CampaignBudget()
    assert "continuation_grant" not in legacy.model_dump(mode="json")
    assert CampaignBudget(max_wall_minutes=60).logical_seconds == 180
    grant = ResourceGrant(total_seconds=600, max_attempts=32)
    budget = CampaignBudget(continuation_grant=grant)
    configured = fixture.campaign().model_copy(update={"budget": budget})
    monkeypatch.setattr(fixture, "campaign", lambda: configured)
    f = _fixture(tmp_path, monkeypatch, arms=2)
    assert f.value["total_seconds"] == 600
    assert autoresearch._bounded_campaign_seconds(configured) == 180
    assert _resume(f)["outcome"] == "yielded"
    assert _resume(f) == f.store.campaign_id
    assert all(float(cmd[cmd.index("--experiment-wall-seconds") + 1])
               <= INTERRUPT_AFTER_SECONDS for cmd in f.calls)
    state = context.CycleJournal(f.store, context.load_context(f.store)).state
    assert state["phase"] == "completed" and 0 < state["spent_seconds"] < 600
    inputs = list((f.store.root / "artifacts/command_cursor_inputs").glob("*.json"))
    import json
    assert inputs and all(json.loads(p.read_text())["total"] == 600 for p in inputs)


def test_logical_attempt_grant_cannot_reset_on_cursor_restart(tmp_path, monkeypatch):
    from tests.test_autoresearch import test_evaluation_continuation as fixture
    from scripts.autoresearch_command_cursor import ContinuationGrant
    from slm_training.autoresearch.storage import CampaignStore

    clock = [0.0]
    monkeypatch.setattr(fixture.continuation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    spec = fixture.experiment()
    store = CampaignStore(spec.campaign_id, tmp_path)
    calls = []

    def execute(spec, commands, **kwargs):
        calls.append(commands)
        clock[0] += 16
        return fixture.outcome("stopped", fixture.pending(commands[0]))

    grant = ContinuationGrant("release", 600, 1)
    first = fixture.run([fixture.EVAL], execute, tmp_path, store=store, grant=grant, spec=spec)
    assert first.error == "continuation_total_attempts_exhausted"
    assert not fixture.is_continuation_pending(first)
    assert fixture.run([fixture.EVAL], execute, tmp_path, store=store, grant=grant, spec=spec) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="immutable inputs changed"):
        fixture.run([fixture.EVAL], execute, tmp_path, store=store, spec=spec,
                    grant=ContinuationGrant("release", 600, 2))


def test_supervisor_forwards_and_enforces_declared_grant(tmp_path):
    from scripts.run_autotrain_supervisor import _build_parser
    from scripts.autotrain_supervision import _driver_argv
    from scripts.autotrain_supervisor_operations import _operation_state
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.activity_contract import ResourceGrant

    grant = ResourceGrant(total_seconds=600, max_attempts=32)
    args = _build_parser().parse_args(["--continuation-grant", grant.model_dump_json()])
    request = dict(operation="driver", loop_id=args.loop_id, source_digest="a" * 64,
                   environment_digest="b" * 64, driver_argv=_driver_argv(args))
    with ActivityRuntime(CampaignStore("runtime", tmp_path)) as runtime:
        state = _operation_state(runtime, request, 1, "c" * 64)
    assert state.spec.grant == grant
    assert state.spec.grant.interrupt_seconds == 170


def test_child_reconciliation_retains_charge_and_all_attempt_references(
    tmp_path, monkeypatch
):
    f = _fixture(tmp_path, monkeypatch)
    launch = driver._stage_command

    def crash(*args, **kwargs):
        launch(*args, **kwargs)
        raise SystemExit("after child")

    with monkeypatch.context() as scoped:
        scoped.setattr(driver, "_stage_command", crash)
        with pytest.raises(SystemExit, match="after child"):
            _resume(f)
    before = context.CycleJournal(f.store, f.value).state
    assert 0 < before["inflight"]["reserved_seconds"] < f.value["total_seconds"]
    assert _resume(f)["outcome"] == "yielded"
    after = context.CycleJournal(f.store, f.value).state
    assert after["spent_seconds"] >= before["spent_seconds"]
    assert after["index"] == 1 and (tmp_path / f"{f.ids[0]}-trained").read_text() == "x"
    returns = [
        e["detail"]
        for e in f.store.verify_event_chain()
        if e["event_type"] == "experiment_attempt_returned"
        and e["experiment_id"] == f.ids[0]
    ]
    assert all(
        row["reconciled"] and row["independent_sample_increment"] == 0
        for row in returns
    )
    assert len({row["shared_execution_id"] for row in returns}) == 1


@pytest.mark.parametrize("corrupt", [False, True])
def test_only_fresh_intact_publications_can_reconcile(tmp_path, monkeypatch, corrupt):
    f = _fixture(tmp_path, monkeypatch, arms=2)
    assert _resume(f)["outcome"] == "yielded"
    producer = driver._phase_a_delivery
    if not corrupt:
        producer(arm_exits={eid: 0 for eid in f.ids})  # Valid but BEFORE this attempt.

    def crash(**kwargs):
        if corrupt:
            producer(**kwargs)
        raise SystemExit("publication boundary")

    with monkeypatch.context() as scoped:
        scoped.setattr(driver, "_phase_a_delivery", crash)
        with pytest.raises(SystemExit, match="publication boundary"):
            _resume(f)
    launches = len(f.calls)
    if corrupt:
        event = next(
            e
            for e in reversed(f.store.verify_event_chain())
            if e["event_type"] == "cycle_delivery_published"
        )
        path = (
            f.store.root
            / "artifacts/cycle_deliveries"
            / f"{event['artifact_sha256']}.json"
        )
        path.write_text("{}")
        with pytest.raises(ValueError, match="artifact changed"):
            _resume(f)
    else:
        assert _resume(f)["reason"] == "driver_attempt_requires_reconciliation"
    assert len(f.calls) == launches


@pytest.mark.parametrize("corrupt", [False, True])
def test_same_campaign_completion_requires_current_retirement_and_outputs(
    tmp_path, monkeypatch, corrupt
):
    from slm_training.autoresearch.storage import CampaignStore

    f = _fixture(tmp_path, monkeypatch, arms=2)
    runtime = CampaignStore("runtime", f.root / "loops/test-loop")
    assert _resume(f)["outcome"] == "yielded"
    before = {event["event_id"] for event in runtime.verify_event_chain()}
    with pytest.raises(ValueError, match="current retirement"):
        execution.completed_cycle_since(
            f.cwd, f.root, "test-loop", f.store.campaign_id, before
        )
    assert _resume(f) == f.store.campaign_id
    if corrupt:
        (f.store.root / "cycle_handoff.json").write_text('{"campaign_id":"wrong"}')
        with pytest.raises(ValueError, match="current outputs"):
            execution.completed_cycle_since(
                f.cwd, f.root, "test-loop", f.store.campaign_id, before
            )
        return
    proof = execution.completed_cycle_since(
        f.cwd, f.root, "test-loop", f.store.campaign_id, before
    )
    assert proof["outputs"] and proof["input_digest"] == context._sha(f.value)
    after = {event["event_id"] for event in runtime.verify_event_chain()}
    with pytest.raises(ValueError, match="current retirement"):
        execution.completed_cycle_since(
            f.cwd, f.root, "test-loop", f.store.campaign_id, after
        )


def test_arm_uses_existing_bound_manifest_without_regeneration(tmp_path):
    from tests.test_autoresearch.test_driver_cycle_continuation import _store
    from tests.test_autoresearch.test_harness import (
        experiment_campaign,
        hypothesis_matrix,
    )

    store = _store(tmp_path)
    experiment = hypothesis_matrix().hypotheses[0].experiment
    manifest = experiment_campaign(experiment_id=experiment.experiment_id).model_copy(
        update={"locked_eval_manifest_sha256": "e" * 64}
    )
    store.lock_experiment_campaign(manifest)
    owner = SimpleNamespace(
        _manifest=lambda *args, **kwargs: pytest.fail("regenerated locked manifest")
    )
    path = prepare._manifest_path(
        store, owner, {"replay_manifest_paths": {}}, experiment
    )
    assert type(manifest).model_validate_json(path.read_text()) == manifest
    assert store.load_experiment_campaign(experiment.experiment_id).manifest == manifest
    path.write_text(
        experiment_campaign(experiment_id=experiment.experiment_id).model_dump_json()
    )
    with pytest.raises(ValueError, match="successor required"):
        prepare._manifest_path(store, owner, {"replay_manifest_paths": {}}, experiment)


def test_finalization_yields_and_resumes_without_repeating_delivery(
    tmp_path, monkeypatch
):
    f = _fixture(tmp_path, monkeypatch)
    assert _resume(f)["outcome"] == "yielded"
    clock = [time.monotonic()]
    monkeypatch.setattr(execution, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(context, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = driver._phase_a_delivery
    from scripts import autotrain_search

    credited = []
    record_credit = autotrain_search.record_cycle_credit

    def credit(journal):
        credited.append(journal.store.campaign_id)
        record_credit(journal)

    monkeypatch.setattr(autotrain_search, "record_cycle_credit", credit)

    def delivery(**kwargs):
        assert credited == [f.store.campaign_id]
        result = original(**kwargs)
        clock[0] += 60
        return result

    monkeypatch.setattr(driver, "_phase_a_delivery", delivery)
    second = execution.resume_cycle(
        f.cwd, f.root, "test-loop", driver, deadline=clock[0] + 60
    )
    assert second["outcome"] == "yielded" and f.final_calls == ["delivery"]
    assert context.CycleJournal(f.store, f.value).state["phase"] == "finalizing"
    assert _resume(f) == f.store.campaign_id
    assert f.final_calls == ["delivery", "handoff"] and len(f.calls) == 4
    assert credited == [f.store.campaign_id]


@pytest.mark.parametrize(
    "change",
    [
        {"index": True},
        {"phase": "completed"},
        {"final_index": -1},
        {"seen": ["hyp-0"]},
        {"spent_seconds": float("nan")},
    ],
)
def test_cursor_rejects_invalid_partition_and_numeric_state(
    tmp_path, monkeypatch, change
):
    f = _fixture(tmp_path, monkeypatch)
    journal = context.CycleJournal(f.store, f.value)
    journal.state.update(change)
    with pytest.raises(ValueError, match="invalid driver cursor"):
        journal.save()


def test_driver_grant_is_not_reset_by_another_invocation(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    journal = context.CycleJournal(f.store, f.value)
    journal.state["spent_seconds"] = 175
    journal.save()
    for _ in range(2):
        result = _resume(f)
        assert result["outcome"] == "capability"
        assert result["reason"] == "driver_logical_grant_exhausted"
    assert not f.calls
    assert context.CycleJournal(f.store, f.value).state["spent_seconds"] >= 175


def test_driver_attempt_count_does_not_exhaust_per_stage_grant(monkeypatch):
    """The command cursor, not the multi-stage driver, owns attempt limits."""
    journal = SimpleNamespace(
        state={"phase": "finalizing", "attempt": 999},
        remaining=600.0,
        store=SimpleNamespace(
            load_campaign=lambda: pytest.fail("driver must not reapply cursor grant")
        ),
    )
    monkeypatch.setattr(execution.time, "monotonic", lambda: 0.0)
    assert execution._budget_pending(journal, 120.0) is None


def test_lock_event_is_reconciled_from_existing_runtime_reference(
    tmp_path, monkeypatch
):
    f = _fixture(tmp_path, monkeypatch)
    original = f.store.verify_event_chain

    def missing_lock():
        return [e for e in original() if e["event_type"] != "driver_cycle_locked"]

    monkeypatch.setattr(f.store, "verify_event_chain", missing_lock)
    # Simulated append boundary; actual append owner verifies its idempotency key.
    assert context.load_context(f.store, context._sha(f.value)) == f.value
    assert len([e for e in original() if e["event_type"] == "driver_cycle_locked"]) == 1


def test_completed_state_artifact_reconciles_interrupted_event_append(
    tmp_path, monkeypatch
):
    f = _fixture(tmp_path, monkeypatch)
    journal = context.CycleJournal(f.store, f.value)
    journal.start("arms", 60)
    original = f.store.append_event

    def crash(kind, **kwargs):
        if kind == "driver_cycle_checkpoint":
            raise SystemExit("crash after durable state artifact")
        return original(kind, **kwargs)

    monkeypatch.setattr(f.store, "append_event", crash)
    with pytest.raises(SystemExit):
        journal.settle()
    monkeypatch.setattr(f.store, "append_event", original)
    restored = context.CycleJournal(f.store, f.value)
    assert restored.state["inflight"] is None and restored.state["attempt"] == 1


def test_no_progress_requests_scoped_repair_and_does_not_repeat(tmp_path, monkeypatch):
    from tests.test_autoresearch.test_evaluation_continuation import pending
    from slm_training.autoresearch.schemas import ExperimentOutcome

    f = _fixture(tmp_path, monkeypatch)
    assert _resume(f)["outcome"] == "yielded"
    calls = []

    def stalled(spec, commands, **kwargs):
        calls.append(commands)
        return ExperimentOutcome(
            experiment_id=spec.experiment_id,
            campaign_id=spec.campaign_id,
            campaign_manifest_sha256=kwargs["campaign_manifest_sha256"],
            status="stopped",
            stage_telemetry=(pending(commands[0], n=1),),
        )

    monkeypatch.setattr(autoresearch, "execute_commands", stalled)
    result = _resume(f)
    assert result["outcome"] == "capability" and "no_progress" in result["reason"]
    assert result["blocker"]["kind"] == "repair_harness" and not f.final_calls
    assert _resume(f)["outcome"] == "capability" and len(calls) == 1


def test_missing_finalization_outputs_never_finish_cycle(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    assert _resume(f)["outcome"] == "yielded"
    monkeypatch.setattr(driver, "_write_cycle_handoff", lambda **kwargs: None)
    with pytest.raises(FileNotFoundError):
        _resume(f)
    assert _resume(f)["outcome"] == "capability"
