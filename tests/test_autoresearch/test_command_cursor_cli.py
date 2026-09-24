"""Real executor/store boundaries and canonical cmd_run yield disposition.

Child fixtures test plumbing, not training quality or AgentV publication.
"""

import sys
from types import SimpleNamespace

import pytest

from scripts import autoresearch, autoresearch_continuation as continuation
from scripts.autoresearch_command_cursor import ContinuationGrant
from slm_training.autoresearch.engine import execute_commands
from slm_training.autoresearch import engine
from slm_training.autoresearch.schemas import ExperimentOutcome
from slm_training.autoresearch.storage import CampaignStore
from tests.test_autoresearch.test_harness import (
    campaign,
    experiment,
    experiment_campaign,
    hypothesis_matrix,
)
from tests.test_autoresearch.test_trial_execution import _trial


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -1, 0])
def test_cli_rejects_invalid_dynamic_allowance(seconds):
    with pytest.raises(ValueError, match="positive and finite"):
        autoresearch._bounded_experiment_seconds(campaign(), seconds)


def test_driver_consumes_current_content_bound_yield_not_exit_code(tmp_path):
    from scripts.autoresearch_command_cursor import (
        record_execution_outcome,
        yielded_outcome_since,
    )

    spec = experiment()
    store = CampaignStore(spec.campaign_id, tmp_path)
    pending = ExperimentOutcome(
        campaign_id=spec.campaign_id,
        experiment_id=spec.experiment_id,
        campaign_manifest_sha256="a" * 64,
        status="stopped",
        error="continuation_budget_pending",
    )
    with pytest.raises(ValueError, match="missing"):
        yielded_outcome_since(store, set(), spec.experiment_id, "a" * 64)
    record_execution_outcome(store, pending, "a" * 64, pending=True)
    got = yielded_outcome_since(store, set(), spec.experiment_id, "a" * 64)
    assert got == pending and continuation.is_continuation_pending(got)
    before = {row["event_id"] for row in store.verify_event_chain()}
    with pytest.raises(ValueError, match="missing"):
        yielded_outcome_since(store, before, spec.experiment_id, "a" * 64)
    with pytest.raises(ValueError, match="mismatch"):
        yielded_outcome_since(store, set(), spec.experiment_id, "b" * 64)
    path = next((store.root / "artifacts/outcomes").glob("*.json"))
    path.write_text(pending.model_copy(update={"error": "forged"}).model_dump_json())
    with pytest.raises(ValueError, match="mismatch"):
        yielded_outcome_since(store, set(), spec.experiment_id, "a" * 64)


def test_legacy_helper_rejects_nonfinite_and_caps_external_deadline(monkeypatch):
    from tests.test_autoresearch.test_evaluation_continuation import (
        EVAL,
        complete,
        outcome,
        pending,
    )

    monkeypatch.setattr(continuation, "time", SimpleNamespace(monotonic=lambda: 10))
    captured = []

    def executor(spec, commands, **kwargs):
        captured.append(kwargs["timeout_seconds"])
        return outcome("completed", complete(commands[0]))

    options = dict(
        campaign_manifest_sha256="a" * 64,
        execute_commands=executor,
        cwd=None,
        commands=[EVAL],
    )
    for invalid in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            continuation.continue_pending_evaluation(
                experiment(), outcome("stopped", pending()), deadline=invalid, **options
            )
    result = continuation.continue_pending_evaluation(
        experiment(), outcome("stopped", pending()), deadline=100000, **options
    )
    assert result.status == "completed" and captured == [155]


def test_refused_training_cursor_cannot_skip_into_evaluation(tmp_path, monkeypatch):
    command, summary = _trial(tmp_path, state=False)
    spec = experiment()
    store = CampaignStore(spec.campaign_id, tmp_path / "campaigns")
    calls = []

    def process(argv, **kwargs):
        import json

        calls.append(argv)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(summary),
            stderr="",
            duration_seconds=1,
            timed_out=False,
        )

    monkeypatch.setattr(engine, "run_bounded_process", process)
    plan = [command, [sys.executable, "-c", "print('must not execute')"]]
    options = dict(
        wall_seconds=30,
        grant=ContinuationGrant("fixture-release", 60),
        campaign_manifest_sha256="a" * 64,
        execute_commands=execute_commands,
        cwd=tmp_path,
        store=store,
    )
    first = continuation.execute_with_continuation(spec, plan, **options)
    second = continuation.execute_with_continuation(spec, plan, **options)
    assert first.status == second.status == "stopped"
    assert "reconciliation_required" in second.error
    assert len(calls) == 1


@pytest.mark.parametrize("update_cap", [False, True])
def test_real_training_resume_producer_consumed_without_recipe_growth(
    tmp_path, monkeypatch, update_cap
):
    import json

    command, summary = _trial(tmp_path, updates=3)
    command[command.index("--steps") + 1] = "6"
    if update_cap:
        command += ["--max-updates-this-invocation", "3"]
        summary.update(stopped_on="invocation_update_budget", invocation_start_step=0)
    calls = []

    def process(argv, **kwargs):
        calls.append(argv)
        payload = (
            summary
            if len(calls) == 1
            else {**summary, "steps": 6, "stopped_on": "steps"}
        )
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
            duration_seconds=1,
            timed_out=False,
        )

    monkeypatch.setattr(engine, "run_bounded_process", process)
    spec = experiment()
    store = CampaignStore(spec.campaign_id, tmp_path / "campaigns")
    result = continuation.execute_with_continuation(
        spec,
        [command],
        wall_seconds=30,
        grant=ContinuationGrant("fixture-release", 60, 2),
        campaign_manifest_sha256="a" * 64,
        execute_commands=execute_commands,
        cwd=tmp_path,
        store=store,
    )
    assert result.status == "completed" and len(calls) == 2
    assert "--resume-from" in calls[1] and "--initialize-from" not in calls[1]
    assert calls[1][calls[1].index("--steps") + 1] == "6"
    commits = [json.loads((store.root / "artifacts/command_cursors" / (event["artifact_sha256"] + ".json")).read_text())
               for event in store.verify_event_chain()
               if event["event_type"] == "command_cursor_committed"]
    assert commits and all(row["spent_seconds"] > 0 for row in commits)
    # The bundle contains fixture bytes; this proves routing, not optimizer parity.


def test_real_engine_callback_and_store_restart_keep_completed_prefix(
    tmp_path, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(
        continuation, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    spec = experiment()
    store = CampaignStore(spec.campaign_id, tmp_path / "campaigns")
    train = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('trained').open('a').write('x')",
    ]
    evaluate = [
        sys.executable,
        "-c",
        (
            "import json,sys; from pathlib import Path; "
            "p=Path('partial'); resumed=p.exists(); p.touch(); "
            "print(json.dumps({'resume':{'pending_record_n':{'smoke':0 if resumed else 1}}})); "
            "sys.exit(0 if resumed else 10)"
        ),
        "scripts.evaluate_model",
        "--partial-scoreboard",
    ]
    finalize = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('finalized').touch()",
    ]

    def bounded(spec, commands, **kwargs):
        result = execute_commands(spec, commands, **kwargs)
        clock[0] += 15
        return result

    options = dict(
        wall_seconds=30,
        grant=ContinuationGrant("fixture-release", 60),
        campaign_manifest_sha256="a" * 64,
        execute_commands=bounded,
        cwd=tmp_path,
    )
    first = continuation.execute_with_continuation(
        spec, [train, evaluate, finalize], store=store, **options
    )
    assert continuation.is_continuation_pending(first)
    fresh = CampaignStore(spec.campaign_id, tmp_path / "campaigns")
    result = continuation.execute_with_continuation(
        spec, [train, evaluate, finalize], store=fresh, **options
    )
    assert result.status == "completed"
    assert (tmp_path / "trained").read_text() == "x" and (
        tmp_path / "finalized"
    ).exists()
    checkpoints = [
        e
        for e in fresh.verify_event_chain()
        if e["event_type"] == "command_cursor_checkpoint"
    ]
    assert len(checkpoints) == 3
    assert len(result.stage_telemetry) == 4


@pytest.mark.parametrize("requested", [None, 30.0])
def test_cmd_run_pending_preserves_total_and_skips_terminal_feedback(
    tmp_path, monkeypatch, requested
):
    configured = campaign()
    store = CampaignStore(configured.campaign_id, tmp_path)
    store.initialize(configured)
    matrix = hypothesis_matrix()
    artifact = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=artifact.stem)
    selected = next(
        h.experiment
        for h in matrix.hypotheses
        if h.experiment.experiment_id == matrix.recommended_experiment_id
    )
    store.lock_experiment_campaign(
        experiment_campaign(experiment_id=selected.experiment_id)
    )
    calls = []

    def grant(root, total, max_attempts=None):
        return ContinuationGrant("fixture-release", total, max_attempts)

    monkeypatch.setattr(autoresearch, "resolved_continuation_grant", grant)

    def execute(spec, commands, **kwargs):
        calls.append(kwargs)
        return ExperimentOutcome(
            experiment_id=spec.experiment_id,
            campaign_id=spec.campaign_id,
            campaign_manifest_sha256=kwargs["campaign_manifest_sha256"],
            status="stopped",
            error="continuation_budget_pending",
        )

    monkeypatch.setattr(autoresearch, "execute_with_continuation", execute)
    monkeypatch.setattr(
        autoresearch, "diagnose_outcome", lambda *_: pytest.fail("yield diagnosed")
    )
    args = SimpleNamespace(
        campaign_id=configured.campaign_id,
        root=tmp_path,
        experiment=None,
        execute=True,
        trackio=False,
        experiment_wall_seconds=requested,
    )
    assert autoresearch.cmd_run(args) == 10
    total = min(configured.budget.max_wall_minutes * 60, 180)
    assert calls[0]["grant"].total_wall_seconds == total
    assert calls[0]["wall_seconds"] == min(requested or total, 170)
    assert calls[0]["store"].root == store.root
    events = store.verify_event_chain()
    assert events[-1]["event_type"] == "experiment_yielded"
    assert not any(
        e["event_type"]
        in ("experiment_finished", "outcome_diagnosed", "hypothesis_feedback_recorded")
        for e in events
    )


def test_cmd_run_real_executor_resumes_yield_before_terminal_feedback(
    tmp_path, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(
        continuation, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    configured = campaign()
    store = CampaignStore(configured.campaign_id, tmp_path / "campaigns")
    store.initialize(configured)
    matrix = hypothesis_matrix()
    artifact = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=artifact.stem)
    store.lock_experiment_campaign(
        experiment_campaign(experiment_id=matrix.recommended_experiment_id)
    )
    monkeypatch.setattr(autoresearch, "ROOT", tmp_path)
    monkeypatch.setattr(
        autoresearch,
        "resolved_continuation_grant",
        lambda root, total, max_attempts=None: ContinuationGrant("fixture-release", total, max_attempts),
    )
    command = [
        sys.executable,
        "-c",
        (
            "import json,sys; from pathlib import Path; "
            "p=Path('partial'); resumed=p.exists(); p.touch(); "
            "print(json.dumps({'resume':{'pending_record_n':{'smoke':0 if resumed else 1}}})); "
            "sys.exit(0 if resumed else 10)"
        ),
        "scripts.evaluate_model",
        "--partial-scoreboard",
    ]
    monkeypatch.setattr(
        autoresearch, "compile_commands", lambda *args, **kwargs: [command]
    )

    def bounded(spec, commands, **kwargs):
        result = execute_commands(spec, commands, **kwargs)
        clock[0] += 15
        return result

    monkeypatch.setattr(autoresearch, "execute_commands", bounded)
    args = SimpleNamespace(
        campaign_id=configured.campaign_id,
        root=tmp_path / "campaigns",
        experiment=None,
        execute=True,
        trackio=False,
        experiment_wall_seconds=30,
    )
    assert autoresearch.cmd_run(args) == 10
    assert not (store.root / "artifacts/diagnoses").exists()
    assert autoresearch.cmd_run(args) == 0
    events = store.verify_event_chain()
    assert sum(e["event_type"] == "experiment_started" for e in events) == 1
    assert sum(e["event_type"] == "experiment_finished" for e in events) == 1
    assert sum(e["event_type"] == "outcome_diagnosed" for e in events) == 1
