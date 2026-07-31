"""Phase A positive classification: quality/latency tradeoffs, not naive speed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_autotrain_continuous.py"
)
_SPEC = importlib.util.spec_from_file_location("run_autotrain_continuous", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)

_classify = _mod._classify_metric_tradeoff
_PRIMARY = "smoke.latency_ms_p50"


def _arms(
    *,
    c_lat: float | None,
    t_lat: float | None,
    c_pr: float | None = 1.0,
    t_pr: float | None = 1.0,
    c_mpr: float | None = 0.0,
    t_mpr: float | None = 0.0,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    control = {
        "latency_ms_p50": c_lat,
        "parse_rate": c_pr,
        "meaningful_program_rate": c_mpr,
    }
    candidate = {
        "latency_ms_p50": t_lat,
        "parse_rate": t_pr,
        "meaningful_program_rate": t_mpr,
    }
    return control, candidate


def test_naive_latency_win_with_zero_mpr_is_not_positive() -> None:
    control, candidate = _arms(c_lat=10000.0, t_lat=9000.0, c_mpr=0.0, t_mpr=0.0)
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is False
    assert any(r.startswith("latency_win_rejected_low_mpr") for r in reasons)


def test_latency_win_with_held_quality_is_positive() -> None:
    control, candidate = _arms(
        c_lat=11208.72, t_lat=7676.43, c_mpr=1.0, t_mpr=1.0
    )
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is True
    assert any(r.startswith("primary_metric_win:") for r in reasons)
    assert any(r.startswith("quality_held:") for r in reasons)


def test_quality_win_with_bounded_latency_cost_is_positive() -> None:
    # Control slightly faster but candidate has better meaning — must not fail.
    control, candidate = _arms(
        c_lat=7911.18, t_lat=8197.07, c_mpr=0.0, t_mpr=1.0
    )
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is True
    assert any(r.startswith("quality_metric_win:") for r in reasons)


def test_quality_win_rejected_when_latency_blows_budget() -> None:
    control, candidate = _arms(
        c_lat=5000.0, t_lat=10000.0, c_mpr=0.0, t_mpr=1.0
    )
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is False
    assert any(r.startswith("quality_win_rejected_latency_budget:") for r in reasons)


def test_timeout_band_micro_win_rejected() -> None:
    control, candidate = _arms(
        c_lat=12000.9, t_lat=12000.3, c_mpr=0.33, t_mpr=0.33
    )
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is False
    assert any(r.startswith("latency_win_rejected_timeout_band:") for r in reasons)


def test_efficiency_win_counts_when_faster_with_same_mpr() -> None:
    control, candidate = _arms(
        c_lat=9000.0, t_lat=6000.0, c_mpr=0.6666666667, t_mpr=0.6666666667
    )
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is True
    assert any(
        r.startswith("primary_metric_win:") or r.startswith("efficiency_win:")
        for r in reasons
    )


def test_mpr_regression_blocks_latency_win() -> None:
    control, candidate = _arms(
        c_lat=10000.0, t_lat=5000.0, c_mpr=1.0, t_mpr=0.0
    )
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is False
    assert any("null_or_worse" in r or "low_mpr" in r for r in reasons)


def test_missing_smoke_metrics_are_measurement_incomplete_not_quality_fail() -> None:
    control, candidate = _arms(
        c_lat=None,
        t_lat=None,
        c_pr=None,
        t_pr=None,
        c_mpr=None,
        t_mpr=None,
    )
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is False
    assert any(r.startswith("measurement_incomplete:") for r in reasons)
    assert not any(r.startswith("primary_metric_null_or_worse:") for r in reasons)


def test_climb_policy_measurement_helpers() -> None:
    from slm_training.autoresearch.climb_policy import (
        decode_timeout_seconds_for_role,
        eval_suites_for_role,
        load_climb_policy,
        stage_wall_minutes_for_role,
    )

    policy = load_climb_policy()
    assert stage_wall_minutes_for_role(policy, "screening") >= 8
    assert decode_timeout_seconds_for_role(policy, "screening") >= 20
    assert eval_suites_for_role(policy, "screening") == ("smoke",)


def test_should_enqueue_champion_requires_quality_held() -> None:
    assert _mod._should_enqueue_champion(
        {
            "positive": True,
            "reasons": [
                "primary_metric_win:smoke.latency_ms_p50:10000->8000",
                "quality_held:parse=1.0 mpr=1.0",
            ],
            "candidate_id": "c1-bounds",
            "control_id": "c1-control",
        }
    )
    # Pure latency without quality_held must not enqueue.
    assert not _mod._should_enqueue_champion(
        {
            "positive": True,
            "reasons": ["primary_metric_win:smoke.latency_ms_p50:10000->8000"],
            "candidate_id": "c1-bounds",
            "control_id": "c1-control",
        }
    )
    assert not _mod._should_enqueue_champion(
        {
            "positive": False,
            "reasons": ["quality_metric_win:meaningful_program_rate:0->0.5"],
            "candidate_id": "c1-bounds",
            "control_id": "c1-control",
        }
    )


def test_champion_queue_enqueue_dedup_and_confirm_resolve(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-test"
    camp = root / "continuous-loop-c1"
    exp_dir = camp / "artifacts" / "experiments"
    exp_dir.mkdir(parents=True)
    knobs = {
        "grammar_completion_bounds": True,
        "compact_active_canvas": False,
        "steps": 80,
        "batch_size": 2,
        "train_version": "wf_smoke_v2",
        "seed": 101,
        "decode_timeout_seconds": 24.0,
    }
    (exp_dir / "c1-bounds.json").write_text(
        __import__("json").dumps(
            {"experiment_id": "c1-bounds", "knobs": knobs}
        ),
        encoding="utf-8",
    )
    delivery = {
        "positive": True,
        "campaign_id": "continuous-loop-c1",
        "cycle_index": 1,
        "cycle_role": "screening",
        "candidate_id": "c1-bounds",
        "control_id": "c1-control",
        "control_metrics": {"latency_ms_p50": 10000.0, "meaningful_program_rate": 1.0},
        "candidate_metrics": {"latency_ms_p50": 8000.0, "meaningful_program_rate": 1.0},
        "reasons": [
            "primary_metric_win:smoke.latency_ms_p50:10000->8000",
            "quality_held:parse=1.0 mpr=1.0",
        ],
    }
    entry = _mod._enqueue_champion(
        root=root, loop_id=loop_id, delivery=delivery, camp_dir=camp
    )
    assert entry is not None
    assert entry["status"] == "queued"
    assert entry["knobs"]["grammar_completion_bounds"] is True
    assert "seed" not in entry["knobs"]
    # Dedup: same fingerprint stays single open entry.
    again = _mod._enqueue_champion(
        root=root, loop_id=loop_id, delivery=delivery, camp_dir=camp
    )
    assert again is None
    entries = _mod._load_champion_queue(_mod._champion_queue_path(root, loop_id))
    assert len(entries) == 1
    head = _mod._queue_head_open(entries)
    assert head is not None
    assert head["entry_id"] == entry["entry_id"]

    # Confirm success → confirmed
    confirm_delivery = {
        "positive": True,
        "reasons": [
            "quality_metric_win:meaningful_program_rate:0.5->1.0:lat=9000->8500",
            "quality_held:parse=1.0 mpr=1.0",
        ],
    }
    resolved = _mod._resolve_confirm_result(
        root=root,
        loop_id=loop_id,
        entry=entry,
        delivery=confirm_delivery,
        campaign_id="continuous-loop-c2",
        cycle_index=2,
    )
    assert resolved is not None
    assert resolved["status"] == "confirmed"
    assert _mod._queue_head_open(
        _mod._load_champion_queue(_mod._champion_queue_path(root, loop_id))
    ) is None


def test_champion_confirm_reject_without_quality(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-reject"
    path = _mod._champion_queue_path(root, loop_id)
    entry = {
        "schema": _mod._CHAMPION_QUEUE_SCHEMA,
        "entry_id": "champ-x",
        "status": "confirming",
        "knobs": {"grammar_completion_bounds": True},
        "knobs_fingerprint": "abc",
    }
    _mod._write_champion_queue(path, [entry])
    resolved = _mod._resolve_confirm_result(
        root=root,
        loop_id=loop_id,
        entry=entry,
        delivery={
            "positive": True,
            "reasons": ["primary_metric_win:smoke.latency_ms_p50:10000->9000"],
        },
        campaign_id="c-confirm",
        cycle_index=3,
    )
    assert resolved is not None
    assert resolved["status"] == "rejected"


def test_matrix_confirm_path_same_levers_new_seed() -> None:
    from slm_training.autoresearch.schemas import HypothesisMatrix

    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260731-c9",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=80,
        cycle=9,
        role="screening",
        confirm_levers={
            "grammar_completion_bounds": True,
            "compact_active_canvas": False,
            "steps": 81,
            "batch_size": 2,
            "train_version": "wf_smoke_v2",
        },
    )
    HypothesisMatrix.model_validate(matrix)
    ids = [h["experiment"]["experiment_id"] for h in matrix["hypotheses"]]
    assert ids[0] == "c20260731-c9-control"
    assert ids[1] == "c20260731-c9-confirm"
    assert len(ids) >= 5  # schema floor; only control+confirm execute
    assert matrix["recommended_experiment_id"] == "c20260731-c9-confirm"
    cand = matrix["hypotheses"][1]["experiment"]["knobs"]
    ctrl = matrix["hypotheses"][0]["experiment"]["knobs"]
    assert cand["grammar_completion_bounds"] is True
    assert ctrl["grammar_completion_bounds"] is False
    assert cand["seed"] == ctrl["seed"] == 100_000 + 9
    assert cand["steps"] == 81
