from types import SimpleNamespace

from scripts import autoresearch_continuation as continuation
from scripts.autoresearch_continuation import ContinuationGrant, is_continuation_pending
from slm_training.autoresearch.engine import execute_commands
from slm_training.autoresearch.storage import CampaignStore
from tests.test_autoresearch.test_evaluation_continuation import (
    EVAL, MANIFEST, complete, outcome, pending, run,
)
from tests.test_autoresearch.test_harness import experiment


def test_engine_yields_before_starting_eval_without_finalization_room(tmp_path, monkeypatch):
    from slm_training.autoresearch import engine
    launched = []
    monkeypatch.setattr(engine, "run_bounded_process", lambda *a, **k: launched.append(a))
    result = execute_commands(experiment(), [EVAL], cwd=tmp_path,
                              timeout_seconds=3, campaign_manifest_sha256=MANIFEST)
    assert is_continuation_pending(result)
    assert result.stage_telemetry == () and not launched
    assert result.metrics == {} and result.status == "stopped"


def test_cursor_yields_short_remainder_then_resumes_same_command(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(continuation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    calls = []
    def execute(spec, commands, **kwargs):
        calls.append(commands)
        clock[0] += 12
        return outcome("stopped", pending(commands[0], n=2)) if len(calls) == 1 else outcome("completed", complete(commands[0]))
    spec = experiment()
    grant = ContinuationGrant("release", 100)
    store = CampaignStore(spec.campaign_id, root=tmp_path)
    first = run([EVAL], execute, tmp_path, store=store, grant=grant, spec=spec)
    assert is_continuation_pending(first) and len(calls) == 1
    second = run([EVAL], execute, tmp_path, store=CampaignStore(spec.campaign_id, root=tmp_path), grant=grant, spec=spec)
    assert second.status == "completed" and len(calls) == 2
    assert calls[1] == [EVAL]


def test_resume_command_preserves_locked_run_directory(tmp_path):
    command = [*EVAL, "--run-root", str(tmp_path), "--run-id", "control"]
    resumed = continuation._resume_commands([command], 0, pending(command))
    assert resumed == [command]
    assert command[-1] == "control"


def test_logical_remainder_cannot_yield_forever_without_charging(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(continuation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    spec = experiment()
    store = CampaignStore(spec.campaign_id, root=tmp_path)
    calls = []
    def execute(spec, commands, **kwargs):
        calls.append(commands)
        clock[0] += 12
        return outcome("stopped", pending(commands[0], n=2))
    grant = ContinuationGrant("release", 32)
    first = run([EVAL], execute, tmp_path, store=store, grant=grant, spec=spec)
    assert first.error == "continuation_total_budget_insufficient"
    assert not is_continuation_pending(first)
    for _ in range(2):
        again = run([EVAL], execute, tmp_path, store=CampaignStore(spec.campaign_id, root=tmp_path), grant=grant, spec=spec)
        assert again.error == "continuation_total_budget_insufficient"
        assert not is_continuation_pending(again)
    assert len(calls) == 1


def test_evaluator_wall_deferral_yields_without_false_no_progress(tmp_path):
    calls = []
    spec = experiment()
    grant = ContinuationGrant("release", 100)
    def execute(spec, commands, **kwargs):
        calls.append(commands)
        if len(calls) == 3:
            return outcome("completed", complete(commands[0]))
        stage = pending(commands[0], n=2)
        if len(calls) == 2:
            stage["parsed_output"]["suites"] = {"smoke": {"resume": {
                "schema": "eval_resume/v1", "pending_record_n": 2,
                "decoded_this_run_n": 0, "stop_reason": "evaluation_wall"}}}
        return outcome("stopped", stage)
    first = run([EVAL], execute, tmp_path, store=CampaignStore(spec.campaign_id, root=tmp_path), grant=grant, spec=spec)
    assert is_continuation_pending(first) and len(calls) == 2
    second = run([EVAL], execute, tmp_path, store=CampaignStore(spec.campaign_id, root=tmp_path), grant=grant, spec=spec)
    assert second.status == "completed" and len(calls) == 3


def test_wall_deferral_requires_every_pending_suite_receipt():
    stage = pending(n=1)
    stage["parsed_output"]["resume"]["pending_record_n"]["other"] = 1
    receipt = {"schema": "eval_resume/v1", "pending_record_n": 1,
               "decoded_this_run_n": 0, "stop_reason": "evaluation_wall"}
    stage["parsed_output"]["suites"] = {"smoke": {"resume": receipt}}
    assert not continuation._wall_yield(stage)
    stage["parsed_output"]["suites"]["other"] = {"resume": dict(receipt)}
    assert continuation._wall_yield(stage)
    stage["parsed_output"]["suites"]["other"]["resume"]["pending_record_n"] = 2
    assert not continuation._wall_yield(stage)


def test_interrupted_evaluation_retries_without_repeating_completed_prefix(tmp_path, monkeypatch):
    from slm_training.autoresearch import engine
    from slm_training.harness_core.bounded_process import ProcessOutcome
    from tests.test_autoresearch.test_evaluation_continuation import TRAIN
    spec = experiment()
    calls = []
    def bounded(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            return SimpleNamespace(returncode=-2, timed_out=True, interrupted=True,
                killed=False, outcome=ProcessOutcome.TIMED_OUT, stdout="", stderr="",
                duration_seconds=1)
        if len(calls) == 3:
            import json
            payload = {"resume": {"pending_record_n": {"smoke": 1}, "decoded_this_run_n": {"smoke": 1}}}
            return SimpleNamespace(returncode=10, timed_out=False, interrupted=False,
                killed=False, outcome=ProcessOutcome.COMPLETED, stdout=json.dumps(payload), stderr="", duration_seconds=1)
        return SimpleNamespace(returncode=0, timed_out=False, interrupted=False,
            killed=False, outcome=ProcessOutcome.COMPLETED, stdout="", stderr="", duration_seconds=1)
    monkeypatch.setattr(engine, "run_bounded_process", bounded)
    prefix = ["python", "-c", "pass"]
    eval_command = [*EVAL, "--run-root", str(tmp_path), "--run-id", "control"]
    grant = ContinuationGrant("release", 300)
    settings = dict(spec=spec, grant=grant, wall_seconds=100)
    first = run([prefix, eval_command], execute_commands, tmp_path,
                store=CampaignStore(spec.campaign_id, root=tmp_path), **settings)
    assert is_continuation_pending(first) and first.status == "stopped"
    assert first.stage_telemetry[-1]["timed_out"] is True
    assert first.stage_telemetry[-1]["measurement_complete"] is False
    second = run([prefix, eval_command], execute_commands, tmp_path,
                 store=CampaignStore(spec.campaign_id, root=tmp_path), **settings)
    assert second.status == "completed" and len(calls) == 4
    assert calls[0] == prefix
    assert "--resume-run" not in calls[-1]
    # Non-resumable or unvalidated interrupted stages still cannot authorize a retry.
    raw = dict(first.stage_telemetry[-1])
    raw.pop("resume_validation")
    assert continuation._pending_stage(outcome("stopped", raw)) is None
    raw["resume_validation"] = "required_by_evaluator_before_decode"
    raw["command"] = TRAIN
    assert continuation._pending_stage(outcome("stopped", raw)) is None
