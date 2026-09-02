"""P11: promotion tier fits the run cap by chunked, resumable evaluation.

The chunk plan (records per bounded run, locked run budget) is stamped on the
promotion manifest before execution; the driver then finishes each arm's
measurement as a sequence of bounded ``scripts.evaluate_model --resume-run``
subprocesses.  Every subprocess here is a stub that advances a fake partial
scoreboard by ``records_per_run`` records; no model is decoded.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_training.autoresearch.experiment_campaign import (
    PROMOTION_CHUNK_PLAN_SCHEMA,
    ExperimentCampaignV1,
)
from slm_training.levers import MAX_HARNESS_WALL_SECONDS, MAX_RUN_SECONDS

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_autotrain_continuous.py"
_SPEC = importlib.util.spec_from_file_location("run_autotrain_continuous_p11", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)

SUITE_N = 24
DECODE_SECONDS = 30.0


def _policy(**measurement: object) -> SimpleNamespace:
    payload = {"power_gate": {"alpha": "1/20"}}
    return SimpleNamespace(
        measurement={
            "promotion_suite_n": SUITE_N,
            "promotion_decode_timeout_seconds": 24,
            "promotion_suites": ["smoke", "held_out"],
            "thrash_timing": {"eval_overhead_seconds": 8, "p95_margin": 0.0},
            **measurement,
        },
        payload=payload,
        defaults={},
    )


def _eval_json(root: Path, campaign: str, run: str, name: str, p95_ms: float) -> Path:
    path = root / campaign / "runs" / run / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"latency_ms_p95": p95_ms}), encoding="utf-8")
    return path


def test_chunk_plan_fits_records_per_run_to_measured_p95_and_wall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """24-record suites at a 30 s p95: floor((155 - 8) / 30) = 4 records per run."""
    monkeypatch.setattr(_mod, "_promotion_suite_records", lambda suite: SUITE_N)
    source = _eval_json(tmp_path, "c1", "arm", "eval_held_out.json", DECODE_SECONDS * 1000)
    plan = _mod._promotion_chunk_plan(_policy(), root=tmp_path)
    assert plan["schema"] == PROMOTION_CHUNK_PLAN_SCHEMA
    assert plan["suites"] == ["smoke", "held_out"]
    assert plan["suite_n"] == SUITE_N
    assert plan["measured_decode_p95_seconds"] == pytest.approx(DECODE_SECONDS)
    assert plan["measured_decode_p95_source"] == str(source)
    assert plan["per_record_seconds"] == pytest.approx(DECODE_SECONDS)
    assert plan["chunk_wall_seconds"] == pytest.approx(float(MAX_HARNESS_WALL_SECONDS))
    assert plan["records_per_run"] == int((MAX_HARNESS_WALL_SECONDS - 8) // DECODE_SECONDS) == 4
    assert plan["total_record_n"] == 2 * SUITE_N
    assert plan["run_n"] == 12  # ceil(48 / 4)
    assert plan["records_per_run"] * plan["run_n"] >= plan["total_record_n"]
    # A single 24-record suite at p95 = 30 s therefore needs 6 bounded runs,
    # never one 720 s run.
    assert -(-SUITE_N // plan["records_per_run"]) == 6
    assert plan["records_per_run"] * plan["per_record_seconds"] <= MAX_HARNESS_WALL_SECONDS


def test_chunk_plan_never_plans_below_the_locked_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unmeasured or fast p95 falls back to the locked per-record timeout."""
    monkeypatch.setattr(_mod, "_promotion_suite_records", lambda suite: 5)
    unmeasured = _mod._promotion_chunk_plan(_policy(), root=tmp_path / "empty")
    assert unmeasured["measured_decode_p95_seconds"] is None
    assert unmeasured["per_record_seconds"] == pytest.approx(24.0)
    assert unmeasured["records_per_run"] == int((MAX_HARNESS_WALL_SECONDS - 8) // 24)
    assert unmeasured["total_record_n"] == 10  # min(24, 5) per suite
    assert unmeasured["suite_record_plan"]["held_out"] == {"available": 5, "planned": 5}
    _eval_json(tmp_path, "c1", "arm", "eval_smoke.json", 2_000.0)
    fast = _mod._promotion_chunk_plan(_policy(), root=tmp_path)
    assert fast["per_record_seconds"] == pytest.approx(24.0)
    slow_margin = _mod._promotion_chunk_plan(
        _policy(thrash_timing={"eval_overhead_seconds": 8, "p95_margin": 0.15}),
        root=tmp_path,
    )
    assert slow_margin["p95_margin"] == pytest.approx(0.15)
    assert slow_margin["per_record_seconds"] == pytest.approx(24.0)


def test_promotion_manifest_stamps_chunk_plan_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_mod, "_promotion_suite_records", lambda suite: SUITE_N)
    exp = {
        "experiment_id": "c-promote",
        "hypothesis": "Confirmed champion levers hold under promotion primary.",
        "knobs": {"seed": 7, "eval_version": "e_test"},
    }
    man = _mod._manifest(
        "continuous-loop-c1", exp, "a" * 40, role="promotion", cycle_intent="promote"
    )
    plan = man.measurement_chunk_plan
    assert plan is not None
    assert plan["schema"] == PROMOTION_CHUNK_PLAN_SCHEMA
    assert plan["suite_n"] == SUITE_N
    assert plan["run_n"] >= 1 and plan["records_per_run"] >= 1
    assert man.power_feasibility is not None and man.power_feasibility["n"] == SUITE_N
    # An explicit cycle-locked plan is stamped verbatim.
    locked = {**plan, "records_per_run": 3, "run_n": 16}
    man2 = _mod._manifest(
        "continuous-loop-c1",
        exp,
        "a" * 40,
        role="promotion",
        cycle_intent="promote",
        chunk_plan=locked,
    )
    assert man2.measurement_chunk_plan == locked
    restored = ExperimentCampaignV1.model_validate_json(man2.model_dump_json())
    assert restored.measurement_chunk_plan == locked
    screening = _mod._manifest("continuous-loop-c1", exp, "a" * 40, role="screening")
    assert screening.measurement_chunk_plan is None


def test_promotion_matrix_knobs_lock_suite_n_and_per_run_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slm_training.autoresearch.climb_policy import load_climb_policy

    monkeypatch.setattr(_mod, "_promotion_suite_records", lambda suite: SUITE_N)
    policy = load_climb_policy()
    plan = _mod._promotion_chunk_plan(policy)
    matrix = _mod._matrix(
        campaign_id="continuous-loop-c9",
        evidence_snapshot_id="snap",
        cites=["fixture://a", "fixture://b", "fixture://c"],
        role_citations={"research": "fixture://a"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=9,
        role="promotion",
        policy=policy,
        promote_levers={"grammar_completion_bounds": True},
        promote_control_levers={},
        chunk_plan=plan,
    )
    knob_rows = [h["experiment"]["knobs"] for h in matrix["hypotheses"]]
    assert len(knob_rows) >= 2
    for knobs in knob_rows:
        assert knobs["eval_limit"] == SUITE_N
        assert knobs["eval_partial_scoreboard"] is True
        assert knobs["eval_max_records_this_run"] == plan["records_per_run"]
        assert knobs["eval_suites"] == "smoke,held_out"
    control, candidate = knob_rows[0], knob_rows[1]
    lever_keys = {"eval_limit", "eval_partial_scoreboard", "eval_max_records_this_run"}
    assert {k for k in candidate if candidate[k] != control.get(k)} - lever_keys


class _StubChunkRunner:
    """Each launch decodes ``records_per_run`` records into a fake scoreboard."""

    def __init__(self, camp_dir: Path, *, total_n: int, per_run: int) -> None:
        self.camp_dir = camp_dir
        self.total_n = total_n
        self.per_run = per_run
        self.launches: list[dict[str, object]] = []

    def scoreboard(self, run_dir: Path) -> dict[str, object]:
        path = run_dir / "scoreboard.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"decoded": 0}

    def __call__(self, cmd, *, cwd, deadline, root, loop_id, stage):
        run_dir = Path(cmd[cmd.index("--resume-run") + 1])
        assert cmd[cmd.index("--max-records-this-run") + 1] == str(self.per_run)
        assert "--partial-scoreboard" in cmd
        assert float(cmd[cmd.index("--evaluation-wall-seconds") + 1]) == pytest.approx(
            float(MAX_HARNESS_WALL_SECONDS)
        )
        assert deadline - _mod.time.monotonic() <= MAX_RUN_SECONDS + 1e-6
        board = self.scoreboard(run_dir)
        decoded_before = int(board.get("decoded", 0))
        decoded = min(self.total_n, decoded_before + self.per_run)
        pending = self.total_n - decoded
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "scoreboard.json").write_text(
            json.dumps(
                {
                    "decoded": decoded,
                    "measurement_complete": pending == 0,
                    "suites": {
                        "held_out": {
                            "n": self.total_n,
                            "document_n": self.total_n,
                            "completed_document_n": decoded,
                            "incomplete_document_n": pending,
                            "decode_timeout_count": 0,
                        }
                    },
                    "resume": {
                        "pending_record_n": {"held_out": pending},
                        "decoded_this_run_n": {"held_out": decoded - decoded_before},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.launches.append({"run": run_dir.name, "stage": stage, "decoded": decoded})
        return _mod.BoundedProcessResult(
            command=tuple(cmd),
            outcome=_mod.ProcessOutcome.COMPLETED,
            returncode=0 if pending == 0 else 10,
            stdout="{}",
            stderr="",
            duration_seconds=1.0,
        )


def _fake_plan(*, per_run: int, run_n: int, total_n: int = SUITE_N) -> dict[str, object]:
    return {
        "schema": PROMOTION_CHUNK_PLAN_SCHEMA,
        "suites": ["held_out"],
        "suite_n": total_n,
        "total_record_n": total_n,
        "per_record_seconds": DECODE_SECONDS,
        "chunk_wall_seconds": float(MAX_HARNESS_WALL_SECONDS),
        "records_per_run": per_run,
        "run_n": run_n,
    }


def _arm_fixture(camp_dir: Path, eid: str) -> Path:
    run_dir = camp_dir / "runs" / eid
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "last.pt").write_bytes(b"ckpt")
    exp = camp_dir / "artifacts" / "experiments" / f"{eid}.json"
    exp.parent.mkdir(parents=True, exist_ok=True)
    exp.write_text("{}", encoding="utf-8")
    return exp


def _base_eval_cmd(camp_dir: Path, eid: str) -> list[str]:
    return [
        "python",
        "-m",
        "scripts.evaluate_model",
        "--run-root",
        str(camp_dir / "runs"),
        "--run-id",
        eid,
        "--ship-gates",
        "--suites",
        "held_out",
        "--eval-limit",
        str(SUITE_N),
        "--partial-scoreboard",
        "--max-records-this-run",
        "5",
    ]


def _run_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan: dict[str, object],
    runner: _StubChunkRunner,
    arms: list[str],
    missing_checkpoint: frozenset[str] = frozenset(),
) -> dict[str, object]:
    root = tmp_path / "ar"
    camp_dir = root / "camp1"
    paths = {eid: _arm_fixture(camp_dir, eid) for eid in arms}
    for eid in missing_checkpoint:
        (camp_dir / "runs" / eid / "checkpoints" / "last.pt").unlink()
    monkeypatch.setattr(
        _mod,
        "_promotion_chunk_eval_command",
        lambda *, root, campaign_id, experiment_path, run_dir, plan: _mod._set_command_flag(
            _mod._set_command_flag(
                _mod._set_command_flag(
                    _base_eval_cmd(camp_dir, run_dir.name), "--resume-run", str(run_dir)
                ),
                "--max-records-this-run",
                str(int(plan["records_per_run"])),
            ),
            "--evaluation-wall-seconds",
            f"{float(plan['chunk_wall_seconds']):.6f}",
        ),
    )
    monkeypatch.setattr(_mod, "_stage_command", runner)
    return _mod._run_promotion_eval_chunks(
        cwd=tmp_path,
        root=root,
        loop_id="loop-p11",
        campaign_id="camp1",
        camp_dir=camp_dir,
        plan=plan,
        experiment_paths=paths,
        arm_order=arms,
    )


def test_24_record_suite_completes_in_five_bounded_runs_with_one_merged_scoreboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    per_run = int(MAX_HARNESS_WALL_SECONDS // DECODE_SECONDS)
    assert per_run == 5
    plan = _fake_plan(per_run=per_run, run_n=5)
    runner = _StubChunkRunner(tmp_path / "ar" / "camp1", total_n=SUITE_N, per_run=per_run)
    ledger = _run_chunks(tmp_path, monkeypatch, plan=plan, runner=runner, arms=["c-control", "c-promote"])
    assert ledger["schema"] == _mod._PROMOTION_CHUNK_LEDGER_SCHEMA
    for eid in ("c-control", "c-promote"):
        arm = ledger["arms"][eid]
        assert arm["status"] == "complete"
        assert arm["runs_used"] == 5
        assert [row["decoded_this_run_n"] for row in arm["runs"]] == [5, 5, 5, 5, 4]
        assert arm["runs"][-1]["measurement_complete"] is True
        assert arm["runs"][-1]["pending_after"] == 0
        board = json.loads(
            (tmp_path / "ar" / "camp1" / "runs" / eid / "scoreboard.json").read_text()
        )
        assert board["measurement_complete"] is True
        assert board["suites"]["held_out"]["completed_document_n"] == SUITE_N
    assert len(runner.launches) == 10
    # Runs are sequential per arm: control's five before the candidate's.
    assert [row["run"] for row in runner.launches][:5] == ["c-control"] * 5
    ledger_path = tmp_path / "ar" / "camp1" / _mod._PROMOTION_CHUNK_LEDGER_NAME
    assert json.loads(ledger_path.read_text())["arms"]["c-promote"]["status"] == "complete"
    delivery = _mod._attach_promotion_chunks({"reasons": []}, ledger)
    assert delivery["reasons"] == []
    assert "measurement_complete" not in delivery


def test_exhausted_chunk_budget_is_measurement_incomplete_never_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _fake_plan(per_run=5, run_n=3)
    runner = _StubChunkRunner(tmp_path / "ar" / "camp1", total_n=SUITE_N, per_run=5)
    ledger = _run_chunks(tmp_path, monkeypatch, plan=plan, runner=runner, arms=["c-promote"])
    arm = ledger["arms"]["c-promote"]
    assert arm["status"] == "chunk_budget_exhausted"
    assert arm["runs_used"] == 3
    assert arm["runs"][-1]["pending_after"] == SUITE_N - 15
    delivery = _mod._attach_promotion_chunks({"reasons": ["phase_a_positive"]}, ledger)
    assert delivery["measurement_complete"] is False
    assert "measurement_incomplete:c-promote:chunk_budget_exhausted:runs=3/3" in (
        delivery["reasons"]
    )
    camp_dir = tmp_path / "ar" / "camp1"
    reasons = _mod._promotion_measurement_incomplete_reasons(
        camp_dir, control_id="", candidate_id="c-promote", delivery=delivery
    )
    assert reasons == [
        "measurement_incomplete:c-promote:chunk_budget_exhausted:runs=3/3",
        f"measurement_incomplete:c-promote:partial_scoreboard:pending={SUITE_N - 15}",
    ]
    # A partial scoreboard never feeds the power gate: the locked report is
    # returned untouched, the incomplete path decides.
    locked = {"schema": "power_feasibility/v1", "n": 24, "alpha": "1/20", "decisive": True, "required_n": 6, "min_two_sided_p": "1/8388608"}
    merged = _mod._merged_promotion_power_feasibility(
        camp_dir,
        control_id="c-promote",
        candidate_id="c-promote",
        locked=locked,
        primary_metric="held_out.structural_similarity",
    )
    assert merged == locked


def test_resume_skips_stored_records_and_no_checkpoint_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-chunk kill leaves stored records; the next run only decodes the rest."""
    camp_dir = tmp_path / "ar" / "camp1"
    runner = _StubChunkRunner(camp_dir, total_n=SUITE_N, per_run=5)
    # A killed first run persisted 3 records before dying.
    run_dir = camp_dir / "runs" / "c-promote"
    run_dir.mkdir(parents=True)
    (run_dir / "scoreboard.json").write_text(
        json.dumps(
            {
                "decoded": 3,
                "measurement_complete": False,
                "resume": {"pending_record_n": {"held_out": 21}},
            }
        ),
        encoding="utf-8",
    )
    plan = _fake_plan(per_run=5, run_n=5)
    ledger = _run_chunks(tmp_path, monkeypatch, plan=plan, runner=runner, arms=["c-promote"])
    arm = ledger["arms"]["c-promote"]
    assert arm["status"] == "complete"
    assert arm["runs"][0]["pending_before"] == 21
    assert [row["decoded_this_run_n"] for row in arm["runs"]] == [5, 5, 5, 5, 1]
    assert sum(row["decoded_this_run_n"] for row in arm["runs"]) == SUITE_N - 3
    ledger2 = _run_chunks(
        tmp_path,
        monkeypatch,
        plan=plan,
        runner=runner,
        arms=["c-orphan"],
        missing_checkpoint=frozenset({"c-orphan"}),
    )
    assert ledger2["arms"]["c-orphan"]["status"] == "no_checkpoint"
    delivery = _mod._attach_promotion_chunks({}, ledger2)
    assert delivery["measurement_complete"] is False
    assert delivery["reasons"] == ["measurement_incomplete:c-orphan:no_checkpoint_for_chunks"]


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
                    "suites": {"held_out": {"completed_document_n": completed}},
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
    # The smaller completed arm bounds the paired n; five pairs cannot reject
    # at alpha = 1/20, so the merged report is not decisive.
    board("c-promote", 5)
    merged = _mod._merged_promotion_power_feasibility(
        camp_dir,
        control_id="c-control",
        candidate_id="c-promote",
        locked=locked,
        primary_metric="held_out.structural_similarity",
    )
    assert merged is not None
    assert merged["merged_n"] == 5 and merged["n"] == 5
    assert merged["decisive"] is False
    assert merged["locked_n"] == 24 and merged["locked_decisive"] is True
    disposition = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=None,
        power_feasibility=merged,
    )
    assert disposition["status"] == "promotion_failed"
    assert any(r.startswith("promotion_infeasible_by_design:n=5:") for r in disposition["reasons"])
    # A missing scoreboard leaves the locked report untouched.
    assert (
        _mod._merged_promotion_power_feasibility(
            camp_dir,
            control_id="c-control",
            candidate_id="missing",
            locked=locked,
            primary_metric="held_out.structural_similarity",
        )
        == locked
    )
