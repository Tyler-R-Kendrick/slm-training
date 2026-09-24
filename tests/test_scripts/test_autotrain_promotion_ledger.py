"""Promotion chunk budget, interruption and paired-completeness contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.casefiles import case_values

from tests.test_scripts.test_run_autotrain_continuous_chunked_promotion import (
    SUITE_N,
    _StubChunkRunner,
    _fake_plan,
    _mod,
    _run_chunks,
)


def test_completed_chunk_proof_is_current_and_cannot_hide_training_failure(tmp_path, monkeypatch):
    from scripts.autotrain_promotion_chunks import completed_chunk_evidence

    camp = tmp_path / "ar" / "camp1"
    runner = _StubChunkRunner(camp, total_n=SUITE_N, per_run=SUITE_N)
    _run_chunks(tmp_path, monkeypatch, plan=_fake_plan(per_run=SUITE_N, run_n=1), runner=runner, arms=["c-promote"])
    outcome = {"status": "stopped", "stage_telemetry": [
        {"command": ["python", "-m", "scripts.train_model"], "exit_code": 0},
        {"command": ["python", "-m", "scripts.evaluate_model"], "timed_out": True},
    ]}
    assert completed_chunk_evidence(camp, "c-promote", outcome)
    outcome["stage_telemetry"][0]["measurement_complete"] = False
    assert not completed_chunk_evidence(camp, "c-promote", outcome)
    outcome["stage_telemetry"][0].pop("measurement_complete")
    board = camp / "runs/c-promote/scoreboard.json"
    board.write_text(board.read_text() + " ")
    assert not completed_chunk_evidence(camp, "c-promote", outcome)
    with pytest.raises(ValueError, match="scoreboard changed"):
        _run_chunks(tmp_path, monkeypatch, plan=_fake_plan(per_run=SUITE_N, run_n=1), runner=runner, arms=["c-promote"])


@pytest.mark.parametrize("real_process", [False, True])
def test_finalizer_uses_original_consumers_and_preserves_retry_identity(tmp_path, monkeypatch, real_process):
    import scripts.autotrain_promotion_finalize as finalize
    from slm_training.autoresearch.storage import CampaignStore

    # Chunk subprocesses are simulated; one variant runs the real bounded
    # supervisor operation with real source/environment probes and lease fencing.
    cwd = Path.cwd() if real_process else tmp_path
    if not real_process:
        monkeypatch.setattr(finalize, "source_identity", lambda cwd: "a" * 64)
        monkeypatch.setattr(finalize, "environment_identity", lambda: {"fixture": True})
    monkeypatch.setattr(_mod, "_git", lambda *args, **kwargs: "")
    store = CampaignStore("camp1", tmp_path / "ar")
    matrix = _mod._matrix(campaign_id="camp1", evidence_snapshot_id="snapshot",
        cites=["fixture://a", "fixture://b", "fixture://c"], role_citations={"research": "fixture://a"},
        train_version="wf_smoke_v2", eval_version="e_test", steps=1, cycle=1,
        role="promotion", promote_levers={"grammar_completion_bounds": True}, promote_control_levers={})
    arms = [item["experiment"]["experiment_id"] for item in matrix["hypotheses"]]
    entry = {"entry_id": "trial", "status": "promoting", "promote_attempts": 1}
    _mod._write_champion_queue(_mod._champion_queue_path(store.root.parent, "loop-p11"), [entry])
    (store.root / "manifests").mkdir(parents=True)
    for arm in arms:
        (store.root / "manifests" / f"{arm}.json").write_text(json.dumps({"seeds": [7]}))
    payload = {"campaign_id": "camp1", "loop_id": "loop-p11", "cycle_index": 1,
        "upstream_commit": "a" * 40, "integration_commit": "a" * 40, "role": "promotion",
        "cycle_intent": "promote", "primary_metric": "held_out.structural_similarity",
        "matrix": matrix, "entry": entry, "control_id": arms[0], "candidate_id": arms[1],
        "arm_order": arms, "arm_seed": 7, "arm_exits": {arm: 10 for arm in arms},
        "arm_skipped": {}, "formal_status": "timed_out", "skip_slugs": []}
    locked = finalize.lock_finalization(store, cwd, payload)
    runner = _StubChunkRunner(store.root, total_n=SUITE_N, per_run=SUITE_N)
    ledger = _run_chunks(tmp_path, monkeypatch, plan=_fake_plan(per_run=SUITE_N, run_n=1), runner=runner, arms=arms)
    assert finalize.finalization_pending(store, ledger)
    if real_process:
        from scripts.autotrain_supervisor_operations import run_operation
        from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime

        with ActivityRuntime(CampaignStore("runtime", store.root.parent / "loops/loop-p11")) as runtime:
            processes = []
            launch = runtime.run

            def capture(*args, **kwargs):
                process = launch(*args, **kwargs)
                processes.append(process)
                return process

            monkeypatch.setattr(runtime, "run", capture)
            result = run_operation(runtime, {"cwd": str(cwd), "root": str(store.root.parent),
                "loop_id": "loop-p11", "operation": "promotion_eval", "campaign_id": "camp1",
                "source_digest": locked["source_sha256"], "environment_digest": locked["environment_sha256"]},
                sequence=1, log_event=lambda _: None)
            assert result is not None, processes[-1].stderr
            assert all(state.status == "succeeded" for state in runtime.snapshot().values())
        queue = _mod._load_champion_queue(_mod._champion_queue_path(store.root.parent, "loop-p11"))
        assert queue[0]["status"] == "promotion_inconclusive"
    else:
        result = finalize.finalize_promotion(store, cwd, _mod, ledger)
        assert result["resolution"]["status"] == "promotion_inconclusive"
    handoff = (store.root / "cycle_handoff.json").read_bytes()
    assert not finalize.finalization_pending(store, ledger)
    assert finalize.finalize_promotion(store, cwd, _mod, ledger)["already_finalized"]
    assert (store.root / "cycle_handoff.json").read_bytes() == handoff
    assert len(runner.launches) == 2  # Finalization did not train or decode again.
    monkeypatch.setattr(finalize, "source_identity", lambda cwd: "b" * 64)
    with pytest.raises(ValueError, match="mismatch"):
        finalize.finalize_promotion(store, cwd, _mod, ledger)


@pytest.mark.parametrize("board", case_values(__file__, "test_missing_quality_counts_cannot_complete_promotion"))
def test_missing_quality_counts_cannot_complete_promotion(tmp_path, board):
    (tmp_path / "scoreboard.json").write_text(json.dumps(board))
    assert _mod._promotion_scoreboard_state(tmp_path)["complete"] is False


def test_chunk_budget_survives_invocation_and_projection_loss(tmp_path, monkeypatch):
    from scripts.autotrain_promotion_chunks import LEDGER_NAME

    plan = _fake_plan(per_run=5, run_n=2)
    runner = _StubChunkRunner(tmp_path / "ar" / "camp1", total_n=SUITE_N, per_run=5)
    first = _run_chunks(tmp_path, monkeypatch, plan=plan, runner=runner, arms=["c-promote"])
    assert first["arms"]["c-promote"]["runs_used"] == 2
    (tmp_path / "ar" / "camp1" / LEDGER_NAME).unlink()
    second = _run_chunks(tmp_path, monkeypatch, plan=plan, runner=runner, arms=["c-promote"])
    assert len(runner.launches) == 2
    assert second["arms"]["c-promote"]["status"] == "chunk_budget_exhausted"
    assert (tmp_path / "ar" / "camp1" / LEDGER_NAME).is_file()


def test_chunk_launch_charged_before_interruption(tmp_path, monkeypatch):
    from scripts.autotrain_promotion_chunks import load_ledger
    from slm_training.autoresearch.storage import CampaignStore

    plan = _fake_plan(per_run=5, run_n=5)
    runner = _StubChunkRunner(tmp_path / "ar" / "camp1", total_n=SUITE_N, per_run=5)

    def interrupted(*args, **kwargs):
        runner(*args, **kwargs)
        raise InterruptedError("after row commit, before terminal event")

    with pytest.raises(InterruptedError):
        _run_chunks(tmp_path, monkeypatch, plan=plan, runner=interrupted, arms=["c-promote"])
    store = CampaignStore("camp1", tmp_path / "ar")
    arm = load_ledger(store)["arms"]["c-promote"]
    assert arm["runs_used"] == 1 and arm["runs"][0]["state"] == "launched"
    assert 0 < arm["runs"][0]["reserved_seconds"] <= 170
    resumed = _run_chunks(tmp_path, monkeypatch, plan=plan, runner=runner, arms=["c-promote"])
    assert resumed["arms"]["c-promote"]["status"] == "complete"
    assert resumed["arms"]["c-promote"]["runs_used"] == 5
    assert len(runner.launches) == 5


def test_chunk_yield_keeps_clock_and_locked_plan(tmp_path, monkeypatch):
    from scripts.autotrain_promotion_chunks import resume_chunks

    plan = _fake_plan(per_run=5, run_n=5)
    runner = _StubChunkRunner(tmp_path / "ar" / "camp1", total_n=SUITE_N, per_run=5)
    original = _mod._run_promotion_eval_chunks
    monkeypatch.setattr(_mod, "_run_promotion_eval_chunks", lambda **kwargs: original(
        **kwargs, deadline=_mod.time.monotonic() + 1))
    ledger = _run_chunks(tmp_path, monkeypatch, plan=plan, runner=runner, arms=["c-promote"])
    assert ledger["arms"]["c-promote"]["status"] == "invocation_yield"
    assert not runner.launches
    assert _mod._attach_promotion_chunks({}, ledger)["measurement_complete"] is False
    resumed = resume_chunks({"root": tmp_path / "ar", "campaign_id": "camp1",
        "cwd": tmp_path, "loop_id": "loop-p11"}, stage_runner=runner,
        scoreboard=_mod._promotion_scoreboard_state)
    assert resumed["arms"]["c-promote"]["status"] == "complete"
    with pytest.raises(ValueError, match="locked plan or arms"):
        _run_chunks(tmp_path, monkeypatch, plan={**plan, "run_n": 10}, runner=runner, arms=["c-promote"])


def test_merged_power_feasibility_uses_final_merged_n(tmp_path: Path) -> None:
    camp_dir = tmp_path / "camp1"
    locked = {"schema": "power_feasibility/v1", "n": 24, "alpha": "1/20", "decisive": True, "required_n": 6, "min_two_sided_p": "1/8388608"}

    def board(run_id: str, completed: int) -> None:
        path = camp_dir / "runs" / run_id / "scoreboard.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "measurement_complete": True,
                    "suites": {"held_out": {
                        "completed_document_n": completed,
                        "selected_record_ids": [f"r{i}" for i in range(completed)],
                        "selection_sha256": "a" * 64,
                    }},
                }
            ),
            encoding="utf-8",
        )

    board("c-control", 24)
    board("c-promote", 24)
    merged = _mod._merged_promotion_power_feasibility(
        camp_dir,
        control_id="c-control",
        candidate_id="c-promote",
        locked=locked,
        primary_metric="held_out.structural_similarity",
    )
    assert merged is not None
    assert merged["source"] == "merged_scoreboard"
    assert merged["merged_n"] == 24 and merged["n"] == 24
    assert merged["decisive"] is True
    # Different selected sets cannot discharge the locked complete-pair contract.
    board("c-promote", 5)
    merged = _mod._merged_promotion_power_feasibility(
        camp_dir,
        control_id="c-control",
        candidate_id="c-promote",
        locked=locked,
        primary_metric="held_out.structural_similarity",
    )
    assert merged is not None
    assert merged["measurement_complete"] is False
    assert merged["reason"] == "selected_pair_identity_mismatch"
    assert merged["decisive"] is False
    assert merged["locked_n"] == 24 and merged["locked_decisive"] is True
    disposition = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=None,
        power_feasibility=merged,
    )
    assert disposition["status"] == "promotion_inconclusive"
    assert any(r.startswith("measurement_incomplete:paired_coverage:") for r in disposition["reasons"])
    # A missing scoreboard cannot reuse the planned decisive verdict as evidence.
    missing = _mod._merged_promotion_power_feasibility(
            camp_dir,
            control_id="c-control",
            candidate_id="missing",
            locked=locked,
            primary_metric="held_out.structural_similarity",
        )
    assert missing["measurement_complete"] is False
    assert missing["decisive"] is False
