import time
import sys
import json
from types import SimpleNamespace

import pytest

from scripts import autoresearch_continuation as continuation
from scripts.autoresearch_continuation import (
    ContinuationGrant,
    continue_pending_evaluation,
    execute_with_continuation,
    is_continuation_pending,
)
from scripts.autoresearch_command_cursor import CommandCursor
from slm_training.autoresearch.engine import execute_commands
from slm_training.autoresearch.schemas import ExperimentOutcome
from slm_training.autoresearch.storage import CampaignStore
from tests.test_autoresearch.test_harness import experiment


def test_canonical_run_automatically_continues_pending_eval() -> None:
    command = [*EVAL, "--resume-run", "--max-records-this-run", "4"]
    first = outcome("stopped", pending(command))
    second = outcome("completed", complete(command), metrics={"suites.smoke.n": 4})
    calls = []

    result = continue_pending_evaluation(
        experiment(),
        first,
        deadline=time.monotonic() + 10,
        campaign_manifest_sha256="a" * 64,
        execute_commands=lambda *args, **kwargs: calls.append(args) or second,
        cwd=None,
    )
    assert result.status == "completed"
    assert result.metrics == {"suites.smoke.n": 4}
    assert is_continuation_pending(first) and not is_continuation_pending(result)
    assert len(calls) == 1


EVAL = [sys.executable, "-m", "scripts.evaluate_model", "--partial-scoreboard"]
TRAIN = [sys.executable, "-m", "scripts.train_model", "--steps", "10"]
FINAL = [sys.executable, "-m", "scripts.finalize"]
MANIFEST = "a" * 64


def outcome(status, *stages, **kwargs):
    spec = experiment()
    return ExperimentOutcome(
        experiment_id=spec.experiment_id,
        campaign_id=spec.campaign_id,
        status=status,
        campaign_manifest_sha256=MANIFEST,
        stage_telemetry=stages,
        **kwargs,
    )


def pending(command=EVAL, *, n=4):
    return dict(
        command=list(command),
        exit_code=10,
        resume_pending=True,
        parsed_output={"resume": {"pending_record_n": {"smoke": n}}},
    )


def complete(command):
    return dict(command=list(command), exit_code=0)


def run(commands, executor, cwd, *, spec=None, **options):
    settings = dict(
        wall_seconds=30,
        campaign_manifest_sha256=MANIFEST,
        execute_commands=executor,
        cwd=cwd,
    )
    return execute_with_continuation(
        spec or experiment(), commands, **(settings | options)
    )


@pytest.mark.parametrize(
    "last", [dict(timed_out=True), dict(skipped=True), dict(exit_code=2)]
)
def test_old_pending_telemetry_is_not_a_current_retry(last):
    first = outcome("stopped", pending(), dict(command=FINAL, **last))

    def forbidden(*args, **kwargs):
        pytest.fail("old pending stage was replayed")

    assert (
        continue_pending_evaluation(
            experiment(),
            first,
            deadline=time.monotonic() + 20,
            campaign_manifest_sha256=MANIFEST,
            execute_commands=forbidden,
            cwd=None,
        )
        == first
    )


def test_remaining_suffix_survives_yield_and_original_argv_is_immutable(tmp_path):
    plan = [TRAIN.copy(), EVAL.copy(), FINAL.copy()]
    calls = []

    def execute(spec, commands, **kwargs):
        calls.append([list(c) for c in commands])
        if len(calls) == 1:
            commands[1].extend(["--evaluation-wall-seconds", "1"])
            return outcome("stopped", complete(commands[0]), pending(commands[1]))
        return outcome("completed", *(complete(c) for c in commands))

    result = run(plan, execute, tmp_path)
    assert result.status == "completed"
    assert calls[1] == [EVAL, FINAL]
    assert plan == [TRAIN, EVAL, FINAL]


@pytest.mark.parametrize("new_n", [4, 5])
def test_no_progress_or_regression_stops_retries_despite_new_logs(tmp_path, new_n):
    calls = []

    def execute(spec, commands, **kwargs):
        calls.append(commands)
        stage = pending(commands[0], n=4 if len(calls) == 1 else new_n)
        stage["stdout"] = str(len(calls))
        return outcome("stopped", stage)

    result = run([EVAL], execute, tmp_path)
    assert result.error == "continuation_no_progress:repair_measurement_required"
    assert not is_continuation_pending(result)
    assert len(calls) == 2


def test_missing_successor_does_not_become_success(tmp_path):
    def execute(spec, commands, **kwargs):
        return outcome("completed", complete(commands[0]))

    result = run([EVAL, FINAL], execute, tmp_path)
    assert result.status == "stopped"
    assert result.error == "continuation_required_commands_missing"


def test_real_engine_and_real_children_preserve_trailing_command(tmp_path):
    # Plumbing fixture only: child output is not a model/AgentV evaluation claim.
    prefix = [sys.executable, "-c", "from pathlib import Path; Path('trained').touch()"]
    probe = [
        sys.executable,
        "-c",
        (
            "import json,sys; from pathlib import Path; "
            "p=Path('first'); resumed=p.exists(); p.touch(); "
            "print(json.dumps({'resume':{'pending_record_n':{'smoke':0 if resumed else 1}}})); "
            "sys.exit(0 if resumed else 10)"
        ),
        "scripts.evaluate_model",
        "--partial-scoreboard",
    ]
    final = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('finalized').touch()",
    ]
    result = run([prefix, probe, final], execute_commands, tmp_path, wall_seconds=60)
    assert result.status == "completed"
    assert (tmp_path / "trained").exists() and (tmp_path / "finalized").exists()
    assert len(result.stage_telemetry) == 4


def test_new_store_instance_resumes_without_retraining(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        continuation, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    spec = experiment()
    store = CampaignStore(spec.campaign_id, root=tmp_path)
    calls = []

    def execute(spec, commands, **kwargs):
        calls.append(commands)
        if len(calls) == 1:
            clock[0] += 15
            return outcome("stopped", complete(commands[0]), pending(commands[1]))
        return outcome("completed", *(complete(c) for c in commands))

    arguments = dict(grant=ContinuationGrant("release-environment", 60), spec=spec)
    first = run([TRAIN, EVAL, FINAL], execute, tmp_path, store=store, **arguments)
    assert first.status == "stopped"
    assert is_continuation_pending(first)
    fresh = CampaignStore(spec.campaign_id, root=tmp_path)
    result = run([TRAIN, EVAL, FINAL], execute, tmp_path, store=fresh, **arguments)
    assert result.status == "completed"
    assert calls[1] == [EVAL, FINAL]
    events = fresh.verify_event_chain()
    assert sum(e["event_type"] == "command_cursor_started" for e in events) == 2
    assert (
        run([TRAIN, EVAL, FINAL], execute, tmp_path, store=fresh, **arguments) == result
    )
    assert len(calls) == 2


def test_lost_executor_result_is_reserved_and_not_blindly_replayed(tmp_path):
    spec = experiment()
    store = CampaignStore(spec.campaign_id, root=tmp_path)
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        raise SystemExit("injected controller loss")

    arguments = dict(
        grant=ContinuationGrant("release-environment", 60), spec=spec, store=store
    )
    with pytest.raises(SystemExit):
        run([TRAIN], interrupted, tmp_path, **arguments)
    stopped = run([TRAIN], interrupted, tmp_path, **arguments)
    assert "attempt_result_missing" in stopped.error
    assert not is_continuation_pending(stopped)
    assert len(calls) == 1
    events = store.verify_event_chain()
    assert events[-1]["detail"]["reserved_seconds"] > 0


def test_durable_cursor_rejects_changed_release_or_commands(tmp_path):
    spec = experiment()
    store = CampaignStore(spec.campaign_id, root=tmp_path)

    def execute(spec, commands, **kwargs):
        return outcome("completed", *(complete(c) for c in commands))

    arguments = dict(grant=ContinuationGrant("release", 30), spec=spec, store=store)
    run([FINAL], execute, tmp_path, **arguments)
    with pytest.raises(ValueError, match="immutable inputs"):
        run([TRAIN], execute, tmp_path, **arguments)
    arguments["grant"] = ContinuationGrant("other-release", 30)
    with pytest.raises(ValueError, match="immutable inputs"):
        run([FINAL], execute, tmp_path, **arguments)


@pytest.mark.parametrize("advance", [False, True])
def test_training_yield_uses_validated_override_and_requires_growth(tmp_path, advance):
    calls = []
    resumed = [*TRAIN, "--resume-from", "bundle/last_full_state.pt"]

    def execute(spec, commands, **kwargs):
        calls.append([list(c) for c in commands])
        if len(calls) == 3:
            return outcome("completed", *(complete(c) for c in commands))
        return outcome(
            "stopped",
            dict(
                command=[*commands[0], "--max-wall-minutes", "0.2"],
                exit_code=0,
                resume_pending=True,
                resume_kind="training",
                resume_command=resumed,
                resume_validation="required_by_trainer_before_updates",
                progress_identity="new-bundle"
                if advance and len(calls) == 2
                else "bundle",
                completed_optimizer_updates=4 if advance and len(calls) == 2 else 2,
                requested_optimizer_updates=10,
            ),
        )

    result = run([TRAIN, EVAL, FINAL], execute, tmp_path)
    assert calls[1] == [resumed, EVAL, FINAL]
    assert result.status == ("completed" if advance else "stopped")
    assert len(calls) == (3 if advance else 2)


def test_per_stage_checkpoint_survives_but_does_not_authorize_blind_replay(tmp_path):
    spec = experiment()
    store = CampaignStore(spec.campaign_id, root=tmp_path)

    def execute(spec, commands, *, stage_callback, **kwargs):
        stage_callback(outcome("running", complete(commands[0])))
        raise SystemExit("controller lost after training checkpoint")

    arguments = dict(grant=ContinuationGrant("release", 60), spec=spec, store=store)
    with pytest.raises(SystemExit):
        run([TRAIN, EVAL], execute, tmp_path, **arguments)
    events = store.verify_event_chain()
    directory = store.root / "artifacts/command_cursors"
    artifact = directory / f"{events[-1]['artifact_sha256']}.json"
    assert json.loads(artifact.read_text())["position"] == 1
    resumed = run([TRAIN, EVAL], execute, tmp_path, **arguments)
    assert "attempt_result_missing" in resumed.error
    assert resumed.stage_telemetry[0]["command"] == TRAIN


def test_result_artifact_event_crash_reconciles_without_reexecution(
    tmp_path, monkeypatch
):
    spec = experiment()
    store = CampaignStore(spec.campaign_id, root=tmp_path)
    calls = []

    def execute(spec, commands, **kwargs):
        calls.append(1)
        return outcome("completed", complete(commands[0]))

    append = store.append_event

    def crash(event_type, **kwargs):
        if event_type == "command_cursor_committed":
            raise SystemExit("crash after validated artifact before terminal event")
        return append(event_type, **kwargs)

    monkeypatch.setattr(store, "append_event", crash)
    arguments = dict(grant=ContinuationGrant("release", 60), spec=spec)
    with pytest.raises(SystemExit):
        run([FINAL], execute, tmp_path, store=store, **arguments)
    fresh = CampaignStore(spec.campaign_id, root=tmp_path)
    result = run([FINAL], execute, tmp_path, store=fresh, **arguments)
    assert result.status == "completed"
    assert len(calls) == 1
    last = fresh.verify_event_chain()[-1]["detail"]
    assert last["charge_kind"] == "unobserved_finalization_reservation"


def test_cursor_payload_corruption_is_not_reused(tmp_path):
    spec = experiment()
    store = CampaignStore(spec.campaign_id, root=tmp_path)

    def execute(spec, commands, **kwargs):
        return outcome("completed", complete(commands[0]))

    arguments = dict(grant=ContinuationGrant("release", 60), spec=spec, store=store)
    run([FINAL], execute, tmp_path, **arguments)
    artifact = next((store.root / "artifacts/command_cursors").glob("*.json"))
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="artifact digest"):
        run([FINAL], execute, tmp_path, **arguments)


def test_cursor_writer_exclusion_between_store_instances(tmp_path):
    spec = experiment()
    store = CampaignStore(spec.campaign_id, root=tmp_path)
    with CommandCursor(store, spec, [FINAL], MANIFEST, "release", 60, cwd=tmp_path):
        with pytest.raises(BlockingIOError):
            with CommandCursor(
                store, spec, [FINAL], MANIFEST, "release", 60, cwd=tmp_path
            ):
                pytest.fail("second writer acquired same experiment")


@pytest.mark.parametrize("allowance", [0, -1, float("nan"), float("inf"), 171, 15])
def test_invalid_invocation_allowance_cannot_silently_stall(tmp_path, allowance):
    with pytest.raises(ValueError):
        run(
            [FINAL],
            lambda *a, **kw: pytest.fail("launched"),
            tmp_path,
            wall_seconds=allowance,
        )


def test_controller_overhead_is_charged_and_new_boot_clock_not_compared(
    tmp_path, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(
        continuation, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    spec = experiment()
    store = CampaignStore(spec.campaign_id, root=tmp_path)
    append = store.append_event

    def charge_append(*args, **kwargs):
        clock[0] += 0.25
        return append(*args, **kwargs)

    monkeypatch.setattr(store, "append_event", charge_append)

    def execute(spec, commands, **kwargs):
        clock[0] += 1
        return outcome("completed", complete(commands[0]))

    run(
        [FINAL],
        execute,
        tmp_path,
        spec=spec,
        store=store,
        grant=ContinuationGrant("release", 60),
    )
    clock[0] = -1000  # Simulated new monotonic origin: only durations are persisted.
    with CommandCursor(
        store, spec, [FINAL], MANIFEST, "release", 60, cwd=tmp_path
    ) as cursor:
        assert cursor.spent == pytest.approx(
            1.75
        )  # lock/start/work/commit, before accounting append.
        assert cursor.remaining == pytest.approx(58.25)
