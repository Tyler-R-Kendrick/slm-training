"""Interrupted partial evaluation: actual engine children and durable driver/cursor.

Fixture children speak the evaluation protocol; no model-quality claim is made.
"""

import json
from contextlib import contextmanager

import pytest

from scripts import autoresearch
from scripts.autoresearch_command_cursor import CommandCursor
from scripts.autotrain_cycle_context import CycleJournal
from tests.test_autoresearch.test_driver_cycle_continuation import _fixture, _resume


def _candidate_cursor(fixture):
    rows = [json.loads(path.read_text()) for path in
            (fixture.store.root / "artifacts/command_cursor_inputs").glob("*.json")]
    return next(row for row in rows
                if row["experiment"]["experiment_id"] == fixture.ids[1])


def _cursor_charge(store, eid):
    cursor_charged, reserved = 0.0, 0.0
    for event in store.verify_event_chain():
        if event["experiment_id"] != eid:
            continue
        kind, detail = event["event_type"], event["detail"]
        if kind == "command_cursor_started":
            reserved = detail["reserved_seconds"]
            cursor_charged += reserved
        elif kind == "command_cursor_committed":
            artifact = store.root / "artifacts/command_cursors" / f"{event['artifact_sha256']}.json"
            cursor_charged += json.loads(artifact.read_text())["spent_seconds"] - reserved
        elif kind == "command_cursor_overhead":
            cursor_charged += detail["spent_seconds"]
    return cursor_charged


@contextmanager
def _fenced(f):
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.checkpoint_publication import champion_publication_scope
    from slm_training.harness_core.activity_contract import ResourceGrant
    from tests.test_autoresearch.test_activity_runtime import spec

    loop = f.root / "loops/test-loop"
    with ActivityRuntime(CampaignStore("publication-test", loop)) as runtime:
        runtime.register(spec().model_copy(update={"grant": ResourceGrant()}))
        lease = runtime.claim_next(capabilities={"local_process"})
        with champion_publication_scope(runtime, lease, loop_dir=loop):
            yield runtime, lease


def test_interrupted_partial_cursor_resumes_same_campaign_without_repeating_prefix(
    tmp_path, monkeypatch
):
    f = _fixture(tmp_path, monkeypatch, arms=2)
    candidate = f.ids[1]
    executor = autoresearch.execute_commands

    def crash_resume(spec, commands, **kwargs):
        assert spec.experiment_id == candidate
        assert "--resume-run" in commands[0]
        raise SystemExit("resumed start interruption")

    assert _resume(f)["outcome"] == "yielded"
    with monkeypatch.context() as crash:
        crash.setattr(autoresearch, "execute_commands", crash_resume)
        with pytest.raises(SystemExit, match="interruption"):
            _resume(f)

    before = CycleJournal(f.store, f.value).state
    assert before["inflight"] is not None and before["index"] == 1
    inputs = _candidate_cursor(f)
    manifests = {eid: f.store.load_experiment_campaign(eid).manifest_sha256 for eid in f.ids}
    checkpoint_events = [e for e in f.store.verify_event_chain()
                         if e["experiment_id"] == candidate
                         and e["event_type"] == "command_cursor_checkpoint"]
    assert checkpoint_events
    assert (tmp_path / "partial").exists()
    assert not (tmp_path / "candidate-tail").exists()
    cursor_charged = _cursor_charge(f.store, candidate)
    assert cursor_charged > 0
    resumed = []

    def capture(spec, commands, **kwargs):
        resumed.extend(commands)
        return executor(spec, commands, **kwargs)

    monkeypatch.setattr(autoresearch, "execute_commands", capture)
    with _fenced(f):
        assert _resume(f) == f.store.campaign_id
    after = CycleJournal(f.store, f.value).state
    assert after["phase"] == "completed"
    assert after["spent_seconds"] >= before["spent_seconds"]
    assert _candidate_cursor(f) == inputs
    assert manifests == {eid: f.store.load_experiment_campaign(eid).manifest_sha256 for eid in f.ids}
    assert resumed and "--resume-run" in resumed[0]
    assert all((tmp_path / f"{eid}-trained").read_text() == "x" for eid in f.ids)
    assert (tmp_path / "candidate-tail").exists()
    assert f.final_calls == ["delivery", "handoff"]
    # Full prior command reservation remains charged; no manufactured sample.
    with CommandCursor(f.store, autoresearch.ExperimentSpec.model_validate(inputs["experiment"]),
                       inputs["commands"], inputs["manifest"], inputs["identity"],
                       inputs["total"], cwd=f.cwd) as cursor:
        assert not cursor.unresolved
        assert cursor.spent >= cursor_charged
    recovered = [e["detail"] for e in f.store.verify_event_chain()
                 if e["event_type"] == "experiment_attempt_returned"
                 and e["experiment_id"] == candidate and e["detail"].get("reconciled")]
    assert recovered and all(row["independent_sample_increment"] == 0 for row in recovered)


@pytest.mark.parametrize("guard", ["unleased", "revoked", "stale_identity", "nonresumable", "no_certified_pending"])
def test_interrupted_cursor_refuses_unfenced_or_unproven_recovery(tmp_path, monkeypatch, guard):
    from scripts.autotrain_cycle_reconcile import _after_inflight, _recover_arm
    from slm_training.autoresearch.runtime.activity_runtime import StaleLease

    f = _fixture(tmp_path, monkeypatch, arms=2)
    if guard not in {"nonresumable", "no_certified_pending"}:
        assert _resume(f)["outcome"] == "yielded"

    def crash(*args, **kwargs):
        raise SystemExit("before child completion")

    commit = CommandCursor.commit

    def crash_pending(cursor, outcome, position, spent):
        if cursor.experiment.experiment_id == f.ids[1] and outcome.status == "stopped":
            raise SystemExit("before child completion")
        return commit(cursor, outcome, position, spent)

    with monkeypatch.context() as scoped:
        if guard == "no_certified_pending":
            scoped.setattr(CommandCursor, "commit", crash_pending)
        else:
            scoped.setattr(autoresearch, "execute_commands", crash)
        with pytest.raises(SystemExit, match="before child"):
            _resume(f)
    journal = CycleJournal(f.store, f.value)
    before = list(f.store.verify_event_chain())
    initial_charge = journal.state["spent_seconds"]
    fresh = _after_inflight(journal)
    if guard == "unleased":
        assert not _recover_arm(journal, None, fresh)
    else:
        with _fenced(f) as (runtime, lease):
            if guard == "revoked":
                runtime.cancel(lease.activity_id, reason="adversarial stale publisher")
                with pytest.raises(StaleLease):
                    _recover_arm(journal, None, fresh)
            elif guard == "stale_identity":
                journal.value = {**journal.value, "execution_identity": "different-source"}
                with pytest.raises(ValueError, match="immutable inputs changed"):
                    _recover_arm(journal, None, fresh)
            else:
                assert not _recover_arm(journal, None, fresh)
    assert journal.state["spent_seconds"] == initial_charge
    assert journal.state["inflight"] is not None
    assert f.store.verify_event_chain() == before
    assert not (tmp_path / "candidate-tail").exists()


def test_scoped_deadline_is_readonly_nonextending_and_driver_yields_before_expiry(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    from slm_training.harness_core import checkpoint_publication as publication
    from slm_training.levers import HARNESS_FINALIZATION_RESERVE_SECONDS, KILL_GRACE_SECONDS

    f = _fixture(tmp_path, monkeypatch, arms=2)
    grant_before = f.store.load_campaign().budget.model_dump(mode="json")
    with _fenced(f) as (_, lease):
        mono = time.monotonic()
        remaining = KILL_GRACE_SECONDS + HARNESS_FINALIZATION_RESERVE_SECONDS + 5
        wall = lease.expires_at - remaining
        clock = [0.0]
        monkeypatch.setattr(publication, "time", SimpleNamespace(
            time=lambda: wall + clock[0], monotonic=lambda: mono + clock[0]))
        before = f.store.verify_event_chain()
        deadline = publication.controller_work_deadline(170, KILL_GRACE_SECONDS + HARNESS_FINALIZATION_RESERVE_SECONDS)
        clock[0] = 2
        assert publication.controller_work_deadline(170, KILL_GRACE_SECONDS + HARNESS_FINALIZATION_RESERVE_SECONDS) == pytest.approx(deadline)
        assert deadline == pytest.approx(mono + 5)
        assert f.store.verify_event_chain() == before
        result = _resume(f)
        assert result["reason"] == "driver_invocation_yielded"
        assert result["outcome"] == "yielded"
        assert wall + clock[0] < lease.expires_at
    assert not f.calls
    assert f.store.load_campaign().budget.model_dump(mode="json") == grant_before
    assert CycleJournal(f.store, f.value).state["inflight"] is None


def test_completed_prefix_checkpoint_resumes_first_explicit_eval_without_replay(tmp_path, monkeypatch):
    """Native NLL-before-decode boundary, using real protocol-fixture children."""
    from scripts import autotrain_cycle_prepare as prepare

    prepare_cycle = prepare.prepare_cycle

    def prepare_explicit(*args, **kwargs):
        compile_commands = autoresearch.compile_commands

        def compile_resume(*compile_args, **compile_kwargs):
            commands = compile_commands(*compile_args, **compile_kwargs)
            for command in commands:
                if "scripts.evaluate_model" in command:
                    command.append("--resume-run")
            return commands

        monkeypatch.setattr(autoresearch, "compile_commands", compile_resume)
        return prepare_cycle(*args, **kwargs)

    monkeypatch.setattr(prepare, "prepare_cycle", prepare_explicit)
    f = _fixture(tmp_path, monkeypatch, arms=2)
    execute = autoresearch.execute_commands

    def crash_after_prefix(spec, commands, **kwargs):
        callback = kwargs["stage_callback"]

        def checkpoint(partial):
            callback(partial)
            if spec.experiment_id == f.ids[1]:
                raise SystemExit("completed prefix persisted before first eval")

        return execute(spec, commands, **{**kwargs, "stage_callback": checkpoint})

    with monkeypatch.context() as scoped:
        scoped.setattr(autoresearch, "execute_commands", crash_after_prefix)
        with pytest.raises(SystemExit, match="completed prefix persisted"):
            _resume(f)
    before = CycleJournal(f.store, f.value).state
    inputs = _candidate_cursor(f)
    assert not (tmp_path / "partial").exists()
    with _fenced(f):
        pending = _resume(f)
        assert pending["outcome"] == "yielded"
        assert _resume(f) == f.store.campaign_id
    assert CycleJournal(f.store, f.value).state["spent_seconds"] >= before["spent_seconds"]
    assert _candidate_cursor(f) == inputs
    assert all((tmp_path / f"{eid}-trained").read_text() == "x" for eid in f.ids)
    assert (tmp_path / "candidate-tail").exists()
