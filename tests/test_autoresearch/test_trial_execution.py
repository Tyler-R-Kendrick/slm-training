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
