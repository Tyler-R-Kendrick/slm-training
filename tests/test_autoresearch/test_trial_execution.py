"""Real bundle validation and typed trainer continuation, never fake training."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values

from slm_training.autoresearch import engine
from slm_training.autoresearch.trial_execution import incomplete_train_reason, training_resume
from slm_training.harness_core.checkpoint_bundle import stage_checkpoint_bundle
from tests.test_autoresearch.test_harness import experiment


def _trial(tmp_path, *, state=True, updates=2, run_id="trial"):
    files = tmp_path / "raw"
    files.mkdir()
    checkpoint = files / "last.pt"
    checkpoint.write_bytes(b"fixture-not-model-weights")
    checkpoint.with_suffix(".meta.json").write_text(json.dumps({
        "kind": "twotower", "output_contract_version": 2,
    }))
    checkpoint.with_suffix(".tokenizer.json").write_text("{}")
    resume = files / "last_full_state.pt"
    resume.write_bytes(b"fixture-not-exact-state")
    digest = stage_checkpoint_bundle(tmp_path / "trial", checkpoint, {
        "role": "trial_cursor", "run_id": run_id,
        "optimizer_updates": updates, "data_manifest_sha": "a" * 64,
    }, full_state=resume if state else None)
    command = ["python", "-m", "scripts.train_model", "--run-id", "trial", "--steps", "5",
               "--run-root", str(tmp_path), "--initialize-from", "ancestor.pt"]
    summary = {"stopped_on": "wall_time_budget", "steps": updates,
               "checkpoint": str(tmp_path / "trial/bundles" / digest / "last.pt")}
    return command, summary


def test_wall_yield_schedules_only_a_validated_bundle(tmp_path):
    command, summary = _trial(tmp_path)
    event = training_resume(command, summary, cwd=tmp_path)
    assert event["resume_pending"] is True
    assert event["completed_optimizer_updates"] == 2
    assert event["requested_optimizer_updates"] == 5
    assert event["resume_validation"] == "required_by_trainer_before_updates"
    assert "--initialize-from" not in event["resume_command"]
    assert "--initialize-from" in command
    assert Path(event["resume_command"][-1]).name == "last_full_state.pt"
    # Existence/integrity can schedule loading, not certify the fixture payload.
    assert "promotion_eligible" not in event


@pytest.mark.parametrize("state,updates,run_id", case_values(__file__, "test_invalid_wall_cursor_cannot_resume"))
def test_invalid_wall_cursor_cannot_resume(tmp_path, state, updates, run_id):
    command, summary = _trial(tmp_path, state=state, updates=updates, run_id=run_id)
    assert not training_resume(command, summary, cwd=tmp_path).get("resume_pending")


def test_token_endpoint_and_missing_bundle_are_not_wall_yields(tmp_path):
    command, summary = _trial(tmp_path)
    assert training_resume(command, {**summary, "stopped_on": "tokens"}, cwd=tmp_path) == {}
    Path(summary["checkpoint"]).write_bytes(b"corruption")
    assert "resume_refused" in training_resume(command, summary, cwd=tmp_path)


def test_engine_carries_actual_bundle_cursor_and_preserves_declared_commands(tmp_path, monkeypatch):
    command, summary = _trial(tmp_path)
    declared = list(command)
    calls = []

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(summary), stderr="",
                               duration_seconds=1, timed_out=False)

    monkeypatch.setattr(engine, "run_bounded_process", execute)
    result = engine.execute_commands(experiment(), [command], cwd=tmp_path, timeout_seconds=30)
    assert result.status == "stopped"
    assert result.stage_telemetry[-1]["resume_kind"] == "training"
    assert command == declared
    argv, kwargs = calls[0]
    assert float(argv[argv.index("--max-wall-minutes") + 1]) * 60 < kwargs["interrupt_after_seconds"]
    assert result.metrics == {}


def test_boolean_training_count_is_not_completion(tmp_path):
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"fixture")
    reason = incomplete_train_reason(["--steps", "1"], {
        "steps": True, "stopped_on": "steps", "checkpoint": str(checkpoint),
    }, cwd=tmp_path)
    assert "declared steps" in reason


def test_update_yield_requires_bound_cap_and_validated_progress(tmp_path):
    command, summary = _trial(tmp_path, updates=3)
    command[command.index("--steps") + 1] = "6"
    command += ["--max-updates-this-invocation", "3"]
    summary.update(stopped_on="invocation_update_budget", invocation_start_step=0)
    event = training_resume(command, summary, cwd=tmp_path)
    assert event["resume_pending"] and event["requested_optimizer_updates"] == 6
    assert event["resume_command"][event["resume_command"].index("--steps") + 1] == "6"
    assert event["resume_command"][event["resume_command"].index("--max-updates-this-invocation") + 1] == "3"
    for bad in (None, True, -1, 1, 3):
        assert "resume_refused" in training_resume(command, {**summary, "invocation_start_step":bad}, cwd=tmp_path)
    assert "resume_refused" in training_resume(command[:-2], summary, cwd=tmp_path)
    Path(summary["checkpoint"]).write_bytes(b"corrupt")
    assert "resume_refused" in training_resume(command, summary, cwd=tmp_path)


def test_update_cap_is_strict_and_requires_full_state():
    from types import SimpleNamespace
    from pydantic import ValidationError
    from slm_training.autoresearch.schemas import ExperimentKnobs
    from slm_training.harnesses.model_build.resume_contract import invocation_update_limit

    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValidationError):
            ExperimentKnobs(max_updates_this_invocation=bad)
        with pytest.raises(ValueError, match="positive integer"):
            invocation_update_limit(SimpleNamespace(max_updates_this_invocation=bad), 0)
    with pytest.raises(ValueError, match="full_state_checkpoint"):
        invocation_update_limit(SimpleNamespace(max_updates_this_invocation=3, full_state_checkpoint=False), 0)


@pytest.mark.parametrize("attempts,expected_calls,reason", [
    (1, 1, "continuation_total_attempts"),
    (3, 2, "continuation_no_progress"),
])
def test_update_yield_remains_charged_and_cannot_retry_without_progress(
    tmp_path, monkeypatch, attempts, expected_calls, reason,
):
    from scripts import autoresearch_continuation as continuation
    from scripts.autoresearch_command_cursor import ContinuationGrant
    from scripts.autoresearch_continuation import execute_with_continuation
    from slm_training.autoresearch.storage import CampaignStore

    command, summary = _trial(tmp_path, updates=3)
    command[command.index("--steps") + 1] = "6"
    command += ["--max-updates-this-invocation", "3"]
    summary.update(stopped_on="invocation_update_budget", invocation_start_step=0)
    calls = []
    clock = [0.0]
    monkeypatch.setattr(continuation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    def process(argv, **kwargs):
        calls.append(argv)
        clock[0] += 1.0
        return SimpleNamespace(returncode=0, stdout=json.dumps(summary), stderr="",
                               duration_seconds=1, timed_out=False)
    monkeypatch.setattr(engine, "run_bounded_process", process)
    spec = experiment()
    store = CampaignStore(spec.campaign_id, tmp_path / "campaigns")
    result = execute_with_continuation(
        spec, [command], wall_seconds=30,
        grant=ContinuationGrant("fixture-release", 60, attempts),
        campaign_manifest_sha256="a" * 64, execute_commands=engine.execute_commands,
        cwd=tmp_path, store=store,
    )
    assert result.status == "stopped" and reason in result.error
    assert len(calls) == expected_calls
    commits = [json.loads((store.root / "artifacts/command_cursors" /
               (event["artifact_sha256"] + ".json")).read_text())
               for event in store.verify_event_chain()
               if event["event_type"] == "command_cursor_committed"]
    assert commits and sum(row["spent_seconds"] for row in commits) >= expected_calls
