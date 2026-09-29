"""Canonical run_cycle -> cmd_run -> real engine children across bounded passes.

Selection/final scientific verdicts are fixtures; this proves orchestration,
not a scientific positive, live agent repair, or service activation.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import (
    autoresearch,
    autoresearch_continuation,
    run_autotrain_continuous as driver,
)
from scripts import (
    autotrain_cycle_context as context,
    autotrain_cycle_prepare as prepare,
)
from scripts.autoresearch_command_cursor import ContinuationGrant
from scripts.autotrain_cycle_execution import resume_cycle
from slm_training.autoresearch.engine import execute_commands
from slm_training.autoresearch.storage import CampaignStore
from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1
from tests.test_autoresearch.test_harness import (
    campaign,
    experiment_campaign,
    hypothesis_matrix,
)


def _store(root):
    configured = campaign()
    store = CampaignStore(configured.campaign_id, root)
    store.initialize(configured)
    return store


def _arm_manifest(store, experiment_id):
    """Fixture manifests carry the configured campaign's continuation grant."""
    manifest = experiment_campaign(experiment_id=experiment_id)
    grant = store.load_campaign().budget.continuation_grant
    if grant is None:
        return manifest
    return manifest.model_copy(
        update={
            "budget": manifest.budget.model_copy(
                update={"continuation_grant": grant}
            )
        }
    )


def _recorded_fixture(tmp_path, root, store, ids, value):
    from slm_training.autoresearch.preflight.compiled_treatment import persist_pair

    paths = {}
    for eid in ids:
        manifest = _arm_manifest(store, eid)
        store.lock_experiment_campaign(manifest)
        paths[eid] = store.write_artifact("fixture_manifests", manifest)
    for pair in value["locked_designs"].values():
        persist_pair(store, pair)
    selection = prepare.RecordedCycleSelection(
        campaign_id=store.campaign_id,
        loop_id="test-loop",
        control_id=ids[0],
        candidate_ids=tuple(ids[1:]),
        manifest_paths=paths,
        arm_wall_seconds=30,
    )
    return prepare.prepare_recorded_cycle(tmp_path, root, driver, selection)


def _matrix_artifacts(store, arms):
    matrix = hypothesis_matrix()
    artifact = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=artifact.stem)
    by_id = {
        h.experiment.experiment_id: store.write_artifact("experiments", h.experiment)
        for h in matrix.hypotheses[:arms]
    }
    return matrix, by_id


def _fixed_grant(cwd, total, max_attempts=None):
    return ContinuationGrant("fixture-release", total, max_attempts)


def _fixture(tmp_path, monkeypatch, *, arms=3, recorded=False, control_failure=False):
    root = tmp_path / "campaigns"
    store = _store(root)
    matrix, by_id = _matrix_artifacts(store, arms)
    ids = list(by_id)

    monkeypatch.setattr(prepare, "resolved_continuation_grant", _fixed_grant)
    monkeypatch.setattr(context, "resolved_continuation_grant", _fixed_grant)
    monkeypatch.setattr(autoresearch, "resolved_continuation_grant", _fixed_grant)
    monkeypatch.setattr(autoresearch, "ROOT", tmp_path)
    monkeypatch.setattr(
        driver,
        "_manifest",
        lambda campaign_id, exp, *args, **kwargs: _arm_manifest(
            store, exp["experiment_id"]
        ),
    )

    def compile_commands(campaign, exp, **kwargs):
        eid = exp.experiment_id
        if control_failure and eid == ids[0]:
            return [[sys.executable, "-c", "raise SystemExit(2)"]]
        prefix = [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path('{eid}-trained').open('a').write('x')",
        ]
        if eid != ids[1]:
            return [prefix]
        evaluation = [
            sys.executable,
            "-c",
            "import json,sys; from pathlib import Path; p=Path('partial'); done=p.exists(); p.touch(); "
            "print(json.dumps({'resume':{'pending_record_n':{'smoke':0 if done else 1}}})); "
            "sys.exit(0 if done else 10)",
            "scripts.evaluate_model",
            "--partial-scoreboard",
        ]
        tail = [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('candidate-tail').touch()",
        ]
        return [prefix, evaluation, tail]

    monkeypatch.setattr(autoresearch, "compile_commands", compile_commands)
    clock = [0.0]
    monkeypatch.setattr(
        autoresearch_continuation, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )

    def execute(spec, commands, **kwargs):
        result = execute_commands(spec, commands, **kwargs)
        clock[0] += 20
        return result

    monkeypatch.setattr(autoresearch, "execute_commands", execute)
    calls, final_calls = [], []

    def launch(cmd, **kwargs):
        assert kwargs["cwd"] == tmp_path
        calls.append(cmd)
        code = autoresearch.main(cmd[3:])
        return SimpleNamespace(returncode=code, timed_out=False, stdout="", stderr="")

    monkeypatch.setattr(driver, "_stage_command", launch)
    monkeypatch.setattr(
        driver,
        "_integrate_origin_main",
        lambda **kwargs: pytest.fail("selected a new cycle"),
    )
    monkeypatch.setattr(
        driver, "_attach_screening_eval_nll", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(driver, "_clear_active_stage", lambda *args: None)
    monkeypatch.setattr(driver, "_run", lambda *args, **kwargs: None)

    def delivery(**kwargs):
        from scripts.autotrain_ledgers import publish_cycle_delivery

        final_calls.append("delivery")
        result = {
            "schema": "autotrain_sdlc_delivery/v1",
            "loop_id": "test-loop",
            "campaign_id": store.campaign_id,
            "positive": False,
            "measurement_complete": True,
            "candidate_id": ids[1],
            "arm_exits": kwargs["arm_exits"],
        }
        return publish_cycle_delivery(root, result)

    def handoff(**kwargs):
        from slm_training.autoresearch.campaign_events import publish_cycle_handoff

        final_calls.append("handoff")
        publish_cycle_handoff(
            store,
            AutotrainCycleHandoffV1(
                campaign_id=store.campaign_id,
                loop_id="test-loop",
                cycle_index=1,
                upstream_commit=kwargs["upstream_commit"],
                integration_commit=kwargs["integration_commit"],
                cycle_role="screening",
                cycle_intent="screening",
                evidence_class="fixture",
                climb_state="inconclusive",
                ship_state="not_evaluated",
                primary_metric="smoke.eval_nll",
                actions=[
                    {
                        "kind": "monitor",
                        "owner": "autotrain",
                        "reason": "fixture boundary",
                        "evidence_ids": ["fixture-only"],
                    }
                ],
            ),
        )

    monkeypatch.setattr(driver, "_phase_a_delivery", delivery)
    monkeypatch.setattr(driver, "_write_cycle_handoff", handoff)
    monkeypatch.setattr(driver, "_screening_enqueue_allowed", lambda **kwargs: False)
    value = {name: None for name in prepare._FIELDS}
    value.update(
        campaign_id=store.campaign_id,
        loop_id="test-loop",
        cycle=1,
        upstream="a" * 40,
        integration="b" * 40,
        role="screening",
        cycle_intent="screening",
        effective_primary="smoke.eval_nll",
        matrix=matrix.model_dump(mode="json"),
        control_eid=ids[0],
        candidate_eid=ids[1],
        order=ids,
        scheduled_order=ids,
        arm_seed=0,
        arm_wall_minutes=0.5,
        by_id={eid: str(path) for eid, path in by_id.items()},
        replay_manifest_paths={},
        replay_manifests={},
        screening_multi=False,
        screening_candidate_ids=ids[1:],
        role_primary={"direction": "decrease"},
        multi_arm_skip=[],
        ar=[sys.executable, "-m", "scripts.autoresearch", "--root", str(root)],
        skip_slugs=[],
        replay=None,
        locked_designs={
            eid: {
                "arm_ids": [ids[0], eid],
                "treatment_ids": ["control-treatment", eid + "-treatment"],
                "replicate_id": "locked-seed",
                "hypothesis_id": eid,
                "design_digest": "a" * 64,
            }
            for eid in ids[1:]
        },
    )
    if recorded:
        locked = _recorded_fixture(tmp_path, root, store, ids, value)
    else:
        locked = prepare.prepare_cycle(tmp_path, root, driver, value, spent_seconds=0)
    return SimpleNamespace(
        root=root,
        store=store,
        value=locked,
        ids=ids,
        calls=calls,
        final_calls=final_calls,
        cwd=tmp_path,
    )


def _resume(fixture):
    return driver.run_cycle(
        cwd=fixture.cwd,
        root=fixture.root,
        loop_id="test-loop",
        train_version="unused",
        steps=1,
        objective="unused",
        primary_metric="unused",
    )


def test_same_campaign_two_passes_preserve_control_tail_and_finalization(
    tmp_path, monkeypatch
):
    f = _fixture(tmp_path, monkeypatch)
    locked = {
        eid: f.store.load_experiment_campaign(eid).manifest_sha256 for eid in f.ids
    }
    first = _resume(f)
    assert first["outcome"] == "yielded" and first["measurement_complete"] is False
    assert first["campaign_id"] == f.store.campaign_id and not f.final_calls
    assert (tmp_path / "hyp-0-trained").read_text() == "x"
    assert (tmp_path / "hyp-1-trained").read_text() == "x"
    assert not (tmp_path / "hyp-2-trained").exists()
    assert not (f.store.root / "cycle_handoff.json").exists()
    second = _resume(f)
    assert second == f.store.campaign_id
    assert f.final_calls == ["delivery", "handoff"]
    assert all((tmp_path / f"{eid}-trained").read_text() == "x" for eid in f.ids)
    assert (tmp_path / "candidate-tail").exists()
    assert [cmd[cmd.index("--experiment") + 1] for cmd in f.calls] == [
        f.value["by_id"][eid] for eid in ("hyp-0", "hyp-1", "hyp-1", "hyp-2")
    ]
    assert locked == {
        eid: f.store.load_experiment_campaign(eid).manifest_sha256 for eid in f.ids
    }
    events = f.store.verify_event_chain()
    attempts = [
        e["detail"] for e in events if e["event_type"] == "experiment_attempt_started"
    ]
    assert [e["treatment_id"] for e in attempts].count("control-treatment") == 2
    assert len({e["detail"]["shared_execution_id"] for e in events
                if e["event_type"] == "experiment_attempt_returned"
                and e["experiment_id"] == f.ids[0]}) == 1
    assert all(e["independent_sample_increment"] == 0 for e in attempts)
    state = context.CycleJournal(f.store, context.load_context(f.store)).state
    assert state["phase"] == "completed" and 0 < state["spent_seconds"] < 180
    assert len(state["outputs"]) == 2
    assert resume_cycle(tmp_path, f.root, "test-loop", driver) is None


@pytest.mark.parametrize("changed", ["manifest", "plan", "source"])
def test_reentry_refuses_changed_locked_inputs(tmp_path, monkeypatch, changed):
    f = _fixture(tmp_path, monkeypatch)
    if changed == "manifest":
        (f.store.root / "manifests/hyp-0.json").write_text("{}")
    elif changed == "plan":
        Path(f.value["by_id"]["hyp-0"]).write_text("{}")
    else:
        monkeypatch.setattr(
            context,
            "resolved_continuation_grant",
            lambda cwd, total: ContinuationGrant("changed-source", total),
        )
    with pytest.raises(ValueError, match="changed"):
        _resume(f)
    assert not f.calls and not f.final_calls


def test_missing_terminal_output_cannot_finish_or_blindly_repeat(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        driver,
        "_stage_command",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, timed_out=False, stdout="fixed", stderr=""
        ),
    )
    with pytest.raises(ValueError, match="terminal outcome"):
        _resume(f)
    result = _resume(f)
    assert result["outcome"] == "capability"
    assert result["reason"] == "driver_attempt_requires_reconciliation"
    assert not f.final_calls


def test_recorded_pair_builder_reuses_real_cmd_run_cursor(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch, arms=2, recorded=True)
    assert _resume(f)["outcome"] == "yielded"
    assert _resume(f) == f.store.campaign_id
    assert len(f.calls) == 3
    assert (tmp_path / "hyp-0-trained").read_text() == "x"
    assert (tmp_path / "hyp-1-trained").read_text() == "x"


def test_terminal_failed_arm_attaches_nll_but_pending_arm_does_not(
    tmp_path, monkeypatch
):
    f = _fixture(tmp_path, monkeypatch, arms=2, control_failure=True)
    attached = []
    monkeypatch.setattr(
        driver,
        "_attach_screening_eval_nll",
        lambda path, *, exit_code: attached.append((path.name, exit_code)),
    )
    first = _resume(f)
    assert first["outcome"] == "yielded"
    assert len(attached) == 1 and attached[0][0] == f.ids[0]
    assert attached[0][1] != 0
    assert _resume(f) == f.store.campaign_id
    assert attached[-1] == (f.ids[1], 0)
