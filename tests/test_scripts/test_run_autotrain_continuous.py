"""Phase A positive classification: quality/latency tradeoffs, not naive speed."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values
from slm_training.autoresearch.schemas import HypothesisMatrix

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_autotrain_continuous.py"
)
_SPEC = importlib.util.spec_from_file_location("run_autotrain_continuous", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)

_classify_metric_tradeoff = _mod._classify_metric_tradeoff
_PRIMARY = "smoke.latency_ms_p50"


def test_package_import_defers_dsl_but_preserves_legacy_exports() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, slm_training; "
                "assert 'slm_training.dsl' not in sys.modules; "
                "from slm_training.data.store import DataStore; "
                "from slm_training import ExampleRecord, parse; "
                "from slm_training.dsl import ProductionCodec; "
                "assert DataStore and ProductionCodec; "
                "assert ExampleRecord.__module__ == 'slm_training.dsl.schema'; "
                "assert callable(parse)"
            ),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(_SCRIPT.parents[1] / "src")},
    )


def test_prepare_control_snapshot_drops_role_unsafe_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "control"
    source.mkdir()
    rows = [
        {
            "id": "valid",
            "prompt": "fixture",
            "openui": 'root = TextContent(":slot_0")',
            "placeholders": [":slot_0"],
            "target_kind": "document",
        },
        {
            "id": "unsafe",
            "prompt": "fixture",
            "openui": 'root = Input(":slot_0")',
            "placeholders": [":slot_0"],
            "target_kind": "document",
        },
    ]
    (source / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        "slm_training.harnesses.model_build.data.load_train_records", lambda _path: []
    )
    monkeypatch.setattr(
        "slm_training.autoresearch.hillclimb.assert_synthesis_feedback_cleared_for_sft",
        lambda _path: None,
    )

    prepared = _mod._prepare_i10_train_dir_for_sft(
        source, output_parent=tmp_path, require_harness_prompt=False
    )

    assert prepared.name == "control_role_safe"
    assert [
        json.loads(line)["id"]
        for line in (prepared / "records.jsonl").read_text().splitlines()
    ] == ["valid"]
    records_mtime = (prepared / "records.jsonl").stat().st_mtime_ns

    assert (
        _mod._prepare_i10_train_dir_for_sft(
            source, output_parent=tmp_path, require_harness_prompt=False
        )
        == prepared
    )
    assert (prepared / "records.jsonl").stat().st_mtime_ns == records_mtime
    assert "I10_SFT_REUSE version=control_role_safe" in capsys.readouterr().out


def test_trim_unexecuted_hypotheses_keeps_required_members() -> None:
    hypotheses = [
        {"experiment": {"experiment_id": f"e{index}"}} for index in range(8)
    ]
    matrix = {
        "recommended_experiment_id": "e6",
        "hypotheses": hypotheses,
        "next_run_priorities": [{"proposed_experiment_id": "e7"}],
    }

    trimmed = _mod._trim_unexecuted_hypotheses(matrix)

    assert [item["experiment"]["experiment_id"] for item in trimmed["hypotheses"]] == [
        "e0",
        "e1",
        "e2",
        "e6",
        "e7",
    ]


@pytest.fixture(autouse=True)
def _clear_dynamic_thrash_bank_cache() -> None:
    """Isolate loop-local self-heal thrash arms across tests."""
    _mod._DYNAMIC_THRASH_ARMS.clear()
    _mod._DYNAMIC_THRASH_LOADED_FOR = None
    yield
    _mod._DYNAMIC_THRASH_ARMS.clear()
    _mod._DYNAMIC_THRASH_LOADED_FOR = None


def _inject_terminal_policy(monkeypatch: pytest.MonkeyPatch, *, park: bool) -> "object":
    """Pin ``terminal.park_on_exhaust`` while keeping every other policy block real."""
    from slm_training.autoresearch import climb_policy as cp

    base = cp.load_climb_policy()
    policy = cp.ClimbPolicy(
        path=base.path,
        schema=base.schema,
        version=base.version,
        sha256=base.sha256,
        payload={**base.payload, "terminal": {"park_on_exhaust": park}},
    )
    monkeypatch.setattr(cp, "load_climb_policy", lambda path=None: policy)
    return policy


def _classify(
    *,
    control: dict[str, float | None],
    candidate: dict[str, float | None],
    primary_metric: str,
) -> tuple[bool, list[str]]:
    return _classify_metric_tradeoff(
        control=control,
        candidate=candidate,
        primary_metric=primary_metric,
        minimum_efficiency_gain_fraction=0.05,
    )


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


def test_parse_rate_below_perfect_is_never_positive() -> None:
    control, candidate = _arms(
        c_lat=5000.0, t_lat=4000.0, c_pr=1.0, t_pr=0.0, c_mpr=0.0, t_mpr=1.0
    )
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is False
    assert any(r.startswith("invalid_grammar:") for r in reasons)


def test_naive_latency_win_with_zero_mpr_is_not_positive() -> None:
    control, candidate = _arms(c_lat=10000.0, t_lat=9000.0, c_mpr=0.0, t_mpr=0.0)
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is False
    assert any(r.startswith("latency_win_rejected_low_mpr") for r in reasons)


def test_latency_win_with_held_quality_is_positive() -> None:
    control, candidate = _arms(c_lat=11208.72, t_lat=7676.43, c_mpr=1.0, t_mpr=1.0)
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is True
    assert any(r.startswith("primary_metric_win:") for r in reasons)
    assert any(r.startswith("quality_held:") for r in reasons)


def test_quality_win_with_bounded_latency_cost_is_positive() -> None:
    # Control slightly faster but candidate has better meaning — must not fail.
    control, candidate = _arms(c_lat=7911.18, t_lat=8197.07, c_mpr=0.0, t_mpr=1.0)
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is True
    assert any(r.startswith("quality_metric_win:") for r in reasons)


def test_quality_win_rejected_when_latency_blows_budget() -> None:
    control, candidate = _arms(c_lat=5000.0, t_lat=10000.0, c_mpr=0.0, t_mpr=1.0)
    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )
    assert positive is False
    assert any(r.startswith("quality_win_rejected_latency_budget:") for r in reasons)


def test_timeout_band_micro_win_rejected() -> None:
    control, candidate = _arms(c_lat=12000.9, t_lat=12000.3, c_mpr=0.33, t_mpr=0.33)
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
    assert any(r.startswith("quality_held:") for r in reasons)


def test_efficiency_win_rejects_mpr_regression_above_floor() -> None:
    control, candidate = _arms(
        c_lat=3772.85,
        t_lat=1074.57,
        c_mpr=0.6666666666666666,
        t_mpr=0.3333333333333333,
    )

    positive, reasons = _classify(
        control=control, candidate=candidate, primary_metric=_PRIMARY
    )

    assert positive is False
    assert any(
        reason.startswith("efficiency_win_rejected_mpr_regression:")
        for reason in reasons
    )


def test_efficiency_micro_win_is_rejected_as_noise() -> None:
    control, candidate = _arms(
        c_lat=3453.06,
        t_lat=3430.55,
        c_mpr=0.3333333333333333,
        t_mpr=0.3333333333333333,
    )
    positive, reasons = _classify(
        control=control,
        candidate=candidate,
        primary_metric="smoke.structural_similarity",
    )
    assert positive is False
    assert any(r.startswith("efficiency_win_rejected_min_effect:") for r in reasons)


def test_mpr_regression_blocks_latency_win() -> None:
    control, candidate = _arms(c_lat=10000.0, t_lat=5000.0, c_mpr=1.0, t_mpr=0.0)
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
        max_consecutive_frozen_replays,
        stage_wall_minutes_for_role,
    )

    policy = load_climb_policy()
    assert stage_wall_minutes_for_role(policy, "screening") == 3
    # Screening decode is thrash-calibrated (fits n×decode under ~70s arm wall).
    assert 1.0 <= decode_timeout_seconds_for_role(policy, "screening") <= 12.0
    assert decode_timeout_seconds_for_role(policy, "promotion") >= 20
    assert eval_suites_for_role(policy, "screening") == ("smoke",)
    assert max_consecutive_frozen_replays(policy) == 1


def test_run_metrics_loads_screening_quality_primary(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "candidate"
    run_dir.mkdir(parents=True)
    (run_dir / "eval_smoke.json").write_text(
        json.dumps({"metrics": {"binder_reference_f1": 0.75}})
    )
    metrics = _mod._run_metrics(tmp_path, "candidate")
    assert metrics["binder_reference_f1"] == 0.75
    assert metrics["smoke.binder_reference_f1"] == 0.75


def test_knobs_fingerprint_excludes_steps_jitter() -> None:
    a = {"grammar_completion_bounds": True, "batch_size": 2, "steps": 80}
    b = {"grammar_completion_bounds": True, "batch_size": 2, "steps": 81}
    assert _mod._knobs_fingerprint(a) == _mod._knobs_fingerprint(b)
    c = {"grammar_completion_bounds": True, "batch_size": 1, "steps": 80}
    assert _mod._knobs_fingerprint(a) != _mod._knobs_fingerprint(c)


def test_confirm_attempts_bound_rejects_queue_head(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-attempts"
    path = _mod._champion_queue_path(root, loop_id)
    entry = {
        "schema": _mod._CHAMPION_QUEUE_SCHEMA,
        "entry_id": "champ-a",
        "status": "queued",
        "confirm_attempts": _mod._MAX_CONFIRM_ATTEMPTS,
        "knobs": {"grammar_completion_bounds": True},
        "knobs_fingerprint": "x",
    }
    _mod._write_champion_queue(path, [entry])
    # Simulate the drop path used when attempts already at max before bump exceeds.
    attempts = _mod._bump_champion_attempt(
        root=root, loop_id=loop_id, entry_id="champ-a", field="confirm_attempts"
    )
    assert attempts == _mod._MAX_CONFIRM_ATTEMPTS + 1
    _mod._update_champion_status(
        root=root,
        loop_id=loop_id,
        entry_id="champ-a",
        status="rejected",
        resolve_reasons=[f"confirm_attempts_exceeded:{attempts}"],
    )
    rows = _mod._load_champion_queue(path)
    assert rows[0]["status"] == "rejected"
    assert rows[0]["confirm_attempts"] == _mod._MAX_CONFIRM_ATTEMPTS + 1


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
    # A declared quality primary win already is quality evidence when the
    # classifier reports no protected-metric regression.
    assert _mod._should_enqueue_champion(
        {
            "positive": True,
            "measurement_complete": True,
            "primary_metric": "smoke.structural_similarity",
            "reasons": ["primary_metric_win:smoke.structural_similarity:0.05->0.14"],
            "candidate_id": "c1-compiler-decision-token",
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
        __import__("json").dumps({"experiment_id": "c1-bounds", "knobs": knobs}),
        encoding="utf-8",
    )
    (exp_dir / "c1-control.json").write_text(
        __import__("json").dumps(
            {
                "experiment_id": "c1-control",
                "knobs": {**knobs, "grammar_completion_bounds": False, "steps": 40},
            }
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
    assert entry["control_knobs"]["grammar_completion_bounds"] is False
    assert entry["control_knobs"]["steps"] == 40
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
        "measurement_complete": True,
        "primary_metric": "smoke.meaningful_program_rate",
        "reasons": [
            "primary_metric_win:smoke.meaningful_program_rate:0.5->1.0:improvement=0.5",
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
    assert (
        _mod._queue_head_open(
            _mod._load_champion_queue(_mod._champion_queue_path(root, loop_id))
        )
        is None
    )


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
            "measurement_complete": True,
            "reasons": ["primary_metric_win:smoke.latency_ms_p50:10000->9000"],
        },
        campaign_id="c-confirm",
        cycle_index=3,
    )
    assert resolved is not None
    assert resolved["status"] == "rejected"


def test_enqueue_champion_skips_rejected_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-reject-dedup"
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
        __import__("json").dumps({"experiment_id": "c1-bounds", "knobs": knobs}),
        encoding="utf-8",
    )
    (exp_dir / "c1-control.json").write_text(
        __import__("json").dumps(
            {
                "experiment_id": "c1-control",
                "knobs": {**knobs, "grammar_completion_bounds": False, "steps": 40},
            }
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
    _mod._update_champion_status(
        root=root,
        loop_id=loop_id,
        entry_id=entry["entry_id"],
        status="rejected",
        resolve_reasons=["confirmation_rejected:primary_quality_not_reheld"],
    )
    again = _mod._enqueue_champion(
        root=root,
        loop_id=loop_id,
        delivery={**delivery, "cycle_index": 4, "campaign_id": "continuous-loop-c4"},
        camp_dir=camp,
    )
    assert again is None
    entries = _mod._load_champion_queue(_mod._champion_queue_path(root, loop_id))
    assert len(entries) == 1
    assert entries[0]["status"] == "rejected"


def test_champion_incomplete_confirmation_stays_retryable(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-incomplete-confirm"
    entry = {
        "schema": _mod._CHAMPION_QUEUE_SCHEMA,
        "entry_id": "champ-incomplete",
        "status": "confirming",
        "knobs": {"ltr_tail_loss_weight": 1.0},
        "knobs_fingerprint": "incomplete",
    }
    _mod._write_champion_queue(_mod._champion_queue_path(root, loop_id), [entry])

    resolved = _mod._resolve_confirm_result(
        root=root,
        loop_id=loop_id,
        entry=entry,
        delivery={
            "positive": False,
            "measurement_complete": False,
            "reasons": ["measurement_incomplete:control:missing_scoreboard"],
        },
        campaign_id="c-incomplete",
        cycle_index=5,
    )

    assert resolved is not None
    assert resolved["status"] == "confirmation_inconclusive"
    assert "resolved_at" not in resolved
    assert _mod._queue_head_open([resolved]) == resolved


def test_champion_confirm_rejects_efficiency_when_primary_regresses(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-quality-regression"
    entry = {
        "schema": _mod._CHAMPION_QUEUE_SCHEMA,
        "entry_id": "champ-quality-regression",
        "status": "confirming",
        "knobs": {"fidelity_loss_weight": 1.5},
        "knobs_fingerprint": "quality-regression",
    }
    _mod._write_champion_queue(_mod._champion_queue_path(root, loop_id), [entry])
    delivery = {
        "positive": True,
        "measurement_complete": True,
        "primary_metric": "smoke.structural_similarity",
        "reasons": [
            "efficiency_win:mpr_per_ms:0.00006->0.00014:gain_fraction=1.1:minimum=0.05",
            "quality_held:parse=1.0 mpr=0.3333333333333333",
            "non_regression_fail:binder_reference_f1:0.95->0.63",
            "primary_metric_null_or_worse:smoke.structural_similarity:control=0.4575 candidate=0.4458 improvement=-0.0117",
        ],
    }

    resolved = _mod._resolve_confirm_result(
        root=root,
        loop_id=loop_id,
        entry=entry,
        delivery=delivery,
        campaign_id="c-confirm-regression",
        cycle_index=4,
    )

    assert resolved is not None
    assert resolved["status"] == "rejected"
    assert (
        "confirmation_rejected:primary_quality_not_reheld"
        in resolved["resolve_reasons"]
    )


def test_revalidate_confirmed_champion_rejects_historical_false_positive(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    campaign_id = "continuous-loop-c-confirm"
    camp = root / campaign_id
    camp.mkdir(parents=True)
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "positive": True,
                "measurement_complete": True,
                "primary_metric": "smoke.structural_similarity",
                "reasons": [
                    "efficiency_win:mpr_per_ms:0.00006->0.00014:gain_fraction=1.1:minimum=0.05",
                    "quality_held:parse=1.0 mpr=0.3333333333333333",
                    "primary_metric_null_or_worse:smoke.structural_similarity:control=0.45 candidate=0.44 improvement=-0.01",
                ],
            }
        )
    )
    entries = [
        {
            "schema": _mod._CHAMPION_QUEUE_SCHEMA,
            "entry_id": "champ-false-confirm",
            "status": "confirmed",
            "confirm_campaign_id": campaign_id,
        }
    ]

    assert _mod._revalidate_confirmed_champion_entries(root, entries) is True
    assert entries[0]["status"] == "rejected"
    assert entries[0]["resolve_reasons"][0] == (
        "confirmation_reclassified_nonpositive_under_current_policy"
    )


def test_refresh_champion_source_recipe_reopens_drifted_phase(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    source = root / "source-campaign" / "artifacts" / "experiments"
    source.mkdir(parents=True)
    candidate_id = "source-capacity-tail"
    control_id = "source-control"
    base = {
        "compiler_alignment_loss_weight": 1.0,
        "compiler_alignment_margin": 1.0,
        "compiler_alignment_stratified": True,
        "compiler_alignment_kind_filter": "all",
        "mixture_sampling_policy": "capacity_aware",
    }
    (source / f"{candidate_id}.json").write_text(
        json.dumps(
            {
                "experiment_id": candidate_id,
                "knobs": {**base, "ltr_tail_loss_weight": 1.0},
            }
        )
    )
    (source / f"{control_id}.json").write_text(
        json.dumps(
            {
                "experiment_id": control_id,
                "knobs": {**base, "ltr_tail_loss_weight": 0.0},
            }
        )
    )
    entries = [
        {
            "entry_id": "legacy-drifted",
            "status": "harness_failure",
            "source_campaign_id": "source-campaign",
            "source_candidate_id": candidate_id,
            "source_control_id": control_id,
            "knobs": {
                "compiler_alignment_loss_weight": 1.0,
                "ltr_tail_loss_weight": 1.0,
            },
            "control_knobs": {
                "compiler_alignment_loss_weight": 1.0,
                "ltr_tail_loss_weight": 0.0,
            },
        }
    ]

    assert _mod._refresh_champion_source_recipes(root, entries) is True
    assert entries[0]["status"] == "queued"
    assert entries[0]["knobs"]["mixture_sampling_policy"] == "capacity_aware"
    assert entries[0]["control_knobs"]["mixture_sampling_policy"] == ("capacity_aware")
    assert entries[0]["resolve_reasons"][0] == ("champion_recipe_repaired_from_source")


def test_champion_projection_covers_every_registered_screening_lever() -> None:
    registered = set().union(
        *(set(extras) for _, _, extras in _mod._SCREENING_ARM_BANK)
    ) - {"_steps_factor"}

    assert registered <= set(_mod._LEVER_KNOB_KEYS)


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
            "component_edge_loss_weight": 1.0,
            "component_edge_decode_weight": 1.0,
            "structural_aux_head_profile": "component-edge",
            "compiler_decode_mode": "tree",
            "mixture_sampling_policy": "capacity_aware",
            "compiler_alignment_semantic_exhaustive": True,
            "grammar_equivalence_cache": True,
            "grammar_draft_window": 16,
            "steps": 81,
            "batch_size": 2,
            "train_version": "wf_smoke_v2",
        },
        confirm_control_levers={
            "steps": 80,
            "batch_size": 2,
            "train_version": "wf_smoke_v2",
            "structural_aux_head_profile": "component-edge",
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
    assert cand["component_edge_loss_weight"] == 1.0
    assert ctrl["component_edge_loss_weight"] == 0.0
    assert cand["component_edge_decode_weight"] == 1.0
    assert ctrl["component_edge_decode_weight"] == 0.0
    assert cand["structural_aux_head_profile"] == "component-edge"
    assert ctrl["structural_aux_head_profile"] == "component-edge"
    assert cand["compiler_decode_mode"] == "tree"
    assert ctrl["compiler_decode_mode"] == "tree"
    assert cand["mixture_sampling_policy"] == "capacity_aware"
    assert cand["compiler_alignment_semantic_exhaustive"] is True
    assert cand["grammar_equivalence_cache"] is True
    assert cand["grammar_draft_window"] == 16
    assert cand["seed"] == ctrl["seed"] == 100_000 + 9
    assert cand["steps"] == 81
    assert ctrl["steps"] == 80


def test_matrix_steps_confirm_preserves_distinct_source_control_recipe() -> None:
    from slm_training.autoresearch.schemas import HypothesisMatrix

    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1758",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1758,
        role="screening",
        confirm_levers={"steps": 44, "batch_size": 2},
        confirm_control_levers={"steps": 22, "batch_size": 2},
    )

    validated = HypothesisMatrix.model_validate(matrix)
    control = validated.hypotheses[0].experiment.knobs
    candidate = validated.hypotheses[1].experiment.knobs
    assert control.steps == 22
    assert candidate.steps == 44
    assert control.seed == candidate.seed == 101758


def test_matrix_steps_promotion_preserves_distinct_source_control_recipe() -> None:
    from slm_training.autoresearch.schemas import HypothesisMatrix

    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1760",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1760,
        role="promotion",
        promote_levers={"steps": 44, "batch_size": 2},
        promote_control_levers={"steps": 22, "batch_size": 2},
    )

    validated = HypothesisMatrix.model_validate(matrix)
    control = validated.hypotheses[0].experiment.knobs
    candidate = validated.hypotheses[1].experiment.knobs
    assert control.steps == 22
    assert candidate.steps == 44
    assert control.seed == candidate.seed == 101760


def test_preexecution_champion_failure_reclaims_attempt(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    confirming = {
        "status": "confirming",
        "confirm_attempts": 1,
        "confirm_campaign_id": "c1758",
    }
    promoting = {
        "status": "promoting",
        "promote_attempts": 1,
        "promotion_campaign_id": "c1759",
    }

    assert _mod._recover_interrupted_champion_entries(root, [confirming, promoting])
    assert confirming["status"] == "queued"
    assert confirming["confirm_attempts"] == 0
    assert promoting["status"] == "confirmed"
    assert promoting["promote_attempts"] == 0


def test_started_champion_measurement_keeps_attempt_spent(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    campaign = root / "c1758"
    campaign.mkdir(parents=True)
    (campaign / "events.jsonl").write_text(
        json.dumps({"event_type": "experiment_started"}) + "\n",
        encoding="utf-8",
    )
    entry = {
        "status": "confirming",
        "confirm_attempts": 1,
        "confirm_campaign_id": "c1758",
    }

    assert _mod._recover_interrupted_champion_entries(root, [entry])
    assert entry["status"] == "confirming"
    assert entry["confirm_attempts"] == 1


def test_historical_incomplete_confirmation_is_reopened(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    campaign = root / "c-incomplete"
    campaign.mkdir(parents=True)
    (campaign / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "measurement_complete": False,
                "reasons": ["measurement_incomplete:control:missing_scoreboard"],
            }
        ),
        encoding="utf-8",
    )
    entry = {
        "status": "rejected",
        "confirm_campaign_id": "c-incomplete",
        "resolved_at": "2026-08-03T00:00:00Z",
    }

    assert _mod._recover_interrupted_champion_entries(root, [entry])
    assert entry["status"] == "confirmation_inconclusive"
    assert "resolved_at" not in entry


def test_confirmation_never_inherits_promotion_role() -> None:
    assert (
        _mod._role_with_confirmation_boundary("promotion", confirming=True)
        == "screening"
    )
    assert (
        _mod._role_with_confirmation_boundary("promotion", confirming=False)
        == "promotion"
    )


def test_parse_skip_slugs_from_cli_value() -> None:
    assert _mod._parse_skip_slugs("") == frozenset()
    assert _mod._parse_skip_slugs("bounds") == frozenset({"bounds"})
    assert _mod._parse_skip_slugs("bounds, component-plan ,,component-edge") == (
        frozenset({"bounds", "component-plan", "component-edge"})
    )


def test_select_recommended_slug_rotates_and_skips() -> None:
    # cycle 1 → first bank arm (bounds)
    assert _mod._select_recommended_slug(1) == "bounds"
    assert _mod._select_recommended_slug(2) == "canvas"
    assert _mod._select_recommended_slug(3) == "both"
    assert _mod._select_recommended_slug(1729) == "canvas"
    # skip bounds → canvas even on cycle 1
    assert _mod._select_recommended_slug(1, skip={"bounds"}) == "canvas"
    # all skipped fails closed rather than recycling a rejected approach
    all_slugs = {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    with pytest.raises(RuntimeError, match="screening arm bank exhausted"):
        _mod._select_recommended_slug(1, skip=all_slugs)
    assert (
        _mod._select_recommended_slug(1817, skip=all_slugs - {"component-edge-token"})
        == "component-edge-token"
    )
    assert (
        _mod._select_recommended_slug(1818, skip=all_slugs - {"component-edge-margin"})
        == "component-edge-margin"
    )
    assert (
        _mod._select_recommended_slug(
            1821, skip=all_slugs - {"compiler-decision-token"}
        )
        == "compiler-decision-token"
    )
    assert (
        _mod._select_recommended_slug(
            1824, skip=all_slugs - {"compiler-decision-margin"}
        )
        == "compiler-decision-margin"
    )
    assert (
        _mod._select_recommended_slug(
            1841,
            skip=all_slugs
            - {"capacity-aware-semantic-exhaustive-compiler-decision-margin"},
        )
        == "capacity-aware-semantic-exhaustive-compiler-decision-margin"
    )
    assert (
        _mod._select_recommended_slug(1856, skip=all_slugs - {"slot-contract-context"})
        == "slot-contract-context"
    )
    assert (
        _mod._select_recommended_slug(1858, skip=all_slugs - {"constraint-graph"})
        == "constraint-graph"
    )


def test_select_recommended_slug_prioritizes_successor_quality_after_legacy_nulls() -> (
    None
):
    skip = {
        "component-plan",
        "component-edge",
        "component-inventory",
        "binder-topology",
        "component-structure",
    }
    assert _mod._select_recommended_slug(1786, skip=skip) == "binder-arity"

    skip.update(
        {
            "binder-arity",
            "binder-component-plan",
            "slot-component-coverage",
            "slot-component-fidelity-coupling",
            "slot-component-inventory-coupling",
            "slot-component-exposure-cap",
            "slot-contract-context",
            "constraint-graph",
            "literal-margin",
            "literal-close",
        }
    )
    assert _mod._select_recommended_slug(1791, skip=skip) == "fidelity"

    skip.add("fidelity")
    assert _mod._select_recommended_slug(1794, skip=skip) == "edge-alignment"

    skip.add("edge-alignment")
    assert _mod._select_recommended_slug(1795, skip=skip) == "semantic-contrast"

    skip.add("semantic-contrast")
    assert (
        _mod._select_recommended_slug(1796, skip=skip)
        == "semantic-contrast-compiler-margin"
    )
    skip.add("semantic-contrast-compiler-margin")
    assert _mod._select_recommended_slug(1797, skip=skip) == "slot-augmentation"

    skip.add("slot-augmentation")
    assert _mod._select_recommended_slug(1797, skip=skip) == "mixed-mask"

    skip.add("mixed-mask")
    assert _mod._select_recommended_slug(1798, skip=skip) == "symbol-boundary"

    skip.add("symbol-boundary")
    assert _mod._select_recommended_slug(1799, skip=skip) == "design-dropout"

    skip.add("design-dropout")
    assert _mod._select_recommended_slug(1800, skip=skip) == "scaffold-prefix"
    skip.add("scaffold-prefix")
    assert _mod._select_recommended_slug(1801, skip=skip) == "scaffold-prefix-structure"
    skip.add("scaffold-prefix-structure")
    assert _mod._select_recommended_slug(1802, skip=skip) == "scaffold-prefix-tail"
    skip.add("scaffold-prefix-tail")
    assert _mod._select_recommended_slug(1803, skip=skip) == "component-token"
    skip.add("component-token")
    assert _mod._select_recommended_slug(1804, skip=skip) == "component-token-prefix"


def test_new_successor_arm_slug_mapping() -> None:
    assert (
        _mod._arm_slug_from_knobs(
            {"ltr_prefix_loss_weight": 1.0, "structure_token_loss_weight": 1.0}
        )
        == "scaffold-prefix-structure"
    )
    assert (
        _mod._arm_slug_from_knobs(
            {"ltr_prefix_loss_weight": 1.0, "ltr_tail_loss_weight": 1.0}
        )
        == "scaffold-prefix-tail"
    )
    assert (
        _mod._arm_slug_from_knobs(
            {"component_token_loss_weight": 1.0, "ltr_prefix_loss_weight": 1.0}
        )
        == "component-token-prefix"
    )


def test_model_lever_arm_slug_mapping() -> None:
    """M-unit ranking-lever arms are visible to the slug classifier."""
    assert (
        _mod._arm_slug_from_knobs(
            {
                "solver_energy_loss_weight": 1.0,
                "solver_energy_decode_weight": 1.0,
                "structural_aux_head_profile": "solver-energy",
                "compiler_decode_mode": "tree",
            }
        )
        == "solver-energy-rerank"
    )
    assert (
        _mod._arm_slug_from_knobs(
            {
                "legal_edit_hazard_loss_weight": 1.0,
                "legal_edit_hazard_decode_weight": 1.0,
                "structural_aux_head_profile": "legal-edit-hazard",
                "compiler_decode_mode": "tree",
            }
        )
        == "legal-edit-hazard"
    )
    # The bank rows themselves round-trip through the classifier.
    by_slug = {slug: extras for slug, _, extras in _mod._SCREENING_ARM_BANK}
    for slug in ("solver-energy-rerank", "legal-edit-hazard"):
        assert _mod._arm_slug_from_knobs(dict(by_slug[slug])) == slug


def test_model_lever_arms_are_successor_selectable() -> None:
    """New arms never perturb the pinned rotation but remain selectable."""
    all_slugs = {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    assert (
        _mod._select_recommended_slug(1, skip=all_slugs - {"solver-energy-rerank"})
        == "solver-energy-rerank"
    )
    assert (
        _mod._select_recommended_slug(1, skip=all_slugs - {"legal-edit-hazard"})
        == "legal-edit-hazard"
    )


def test_confirmation_bypasses_exhausted_screening_selector() -> None:
    all_slugs = {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}

    assert (
        _mod._select_cycle_slug(
            1792,
            predecessor_priority=None,
            skip=all_slugs,
            has_confirm_levers=True,
            has_promote_levers=False,
        )
        is None
    )
    with pytest.raises(RuntimeError, match="screening arm bank exhausted"):
        _mod._select_cycle_slug(
            1792,
            predecessor_priority=None,
            skip=all_slugs,
            has_confirm_levers=False,
            has_promote_levers=False,
        )


def _write_screening_null_camp(
    root: Path,
    *,
    camp_id: str,
    loop_id: str,
    slug: str,
    seed: int,
    predecessor: str | None,
    positive: bool = False,
    knobs_extra: dict | None = None,
) -> None:
    camp = root / camp_id
    camp.mkdir(parents=True, exist_ok=True)
    knobs = {
        "grammar_completion_bounds": slug == "bounds",
        "compact_active_canvas": slug == "canvas",
        "seed": seed,
        **(knobs_extra or {}),
    }
    if slug == "both":
        knobs["grammar_completion_bounds"] = True
        knobs["compact_active_canvas"] = True
    (camp / "campaign.json").write_text(
        json.dumps(
            {
                "campaign_id": camp_id,
                "loop_id": loop_id,
                "predecessor_campaign_id": predecessor,
            }
        ),
        encoding="utf-8",
    )
    (camp / "cycle_handoff.json").write_text(
        json.dumps({"loop_id": loop_id, "cycle_intent": "screening"}),
        encoding="utf-8",
    )
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "candidate_id": f"{camp_id}-{slug}",
                "control_id": f"{camp_id}-control",
                "cycle_intent": "screening",
                "positive": positive,
                "measurement_complete": True,
            }
        ),
        encoding="utf-8",
    )
    (camp / "matrix-proposal.json").write_text(
        json.dumps(
            {
                "hypotheses": [
                    {
                        "experiment": {
                            "experiment_id": f"{camp_id}-{slug}",
                            "knobs": knobs,
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_single_complete_null_does_not_close_arm(tmp_path: Path) -> None:
    """Fixture-noise single-seed null must not permanent-close the thrash arm."""
    root = tmp_path / "autoresearch"
    loop_id = "loop-ms"
    _write_screening_null_camp(
        root,
        camp_id="c1",
        loop_id=loop_id,
        slug="bounds",
        seed=100_001,
        predecessor=None,
        positive=False,
    )
    assert "bounds" not in _mod._recent_completed_nonpositive_slugs(
        root, "c1", min_null_seeds=2
    )
    # Still selectable for a second-seed retest.
    assert _mod._select_recommended_slug(1, skip=set()) == "bounds"


def test_two_distinct_seed_nulls_close_arm(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-ms2"
    _write_screening_null_camp(
        root,
        camp_id="c1",
        loop_id=loop_id,
        slug="bounds",
        seed=100_001,
        predecessor=None,
    )
    _write_screening_null_camp(
        root,
        camp_id="c2",
        loop_id=loop_id,
        slug="bounds",
        seed=100_002,
        predecessor="c1",
    )
    closed = _mod._recent_completed_nonpositive_slugs(root, "c2", min_null_seeds=2)
    assert "bounds" in closed
    # Other arms remain open.
    assert _mod._select_recommended_slug(1, skip=closed) != "bounds"


def test_positive_clears_prior_null_tally(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-ms3"
    _write_screening_null_camp(
        root,
        camp_id="c1",
        loop_id=loop_id,
        slug="bounds",
        seed=1,
        predecessor=None,
        positive=False,
    )
    _write_screening_null_camp(
        root,
        camp_id="c2",
        loop_id=loop_id,
        slug="bounds",
        seed=2,
        predecessor="c1",
        positive=True,
    )
    # After a win, one later null is not enough to re-close.
    _write_screening_null_camp(
        root,
        camp_id="c3",
        loop_id=loop_id,
        slug="bounds",
        seed=3,
        predecessor="c2",
        positive=False,
    )
    assert "bounds" not in _mod._recent_completed_nonpositive_slugs(
        root, "c3", min_null_seeds=2
    )


def test_matrix_thrash_rotation_recommends_non_bounds() -> None:
    from slm_training.autoresearch.schemas import HypothesisMatrix

    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260731-c2",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=80,
        cycle=2,
        role="screening",
        recommended_slug="canvas",
    )
    HypothesisMatrix.model_validate(matrix)
    assert matrix["recommended_experiment_id"] == "c20260731-c2-canvas"
    ids = [h["experiment"]["experiment_id"] for h in matrix["hypotheses"]]
    assert "c20260731-c2-batch1" in ids
    assert "c20260731-c2-bounds" in ids
    assert "c20260731-c2-component-plan" in ids
    assert "c20260731-c2-binder-topology" in ids
    assert "c20260731-c2-binder-arity" in ids
    assert "c20260731-c2-binder-component-plan" in ids
    assert "c20260731-c2-literal-close" in ids


def test_screening_role_generate_batch_size_is_a_registered_knob() -> None:
    """`_matrix(role="screening")` bakes generate_batch_size=1 into every
    hypothesis (fair-share decode timeout fix); ExperimentKnobs must accept
    it or every default continuous cycle fails HypothesisMatrix validation."""
    from slm_training.autoresearch.schemas import HypothesisMatrix

    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260805-c1",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1,
        role="screening",
    )
    HypothesisMatrix.model_validate(matrix)
    for row in matrix["hypotheses"]:
        assert row["experiment"]["knobs"]["generate_batch_size"] == 1


def test_literal_close_arm_is_size_matched_and_changes_only_tail_loss() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1764",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1764,
        role="screening",
        recommended_slug="literal-close",
    )
    by_id = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    control = by_id["c20260802-c1764-control"]
    candidate = by_id["c20260802-c1764-literal-close"]

    assert matrix["recommended_experiment_id"].endswith("-literal-close")
    assert control["ltr_tail_loss_weight"] == 0.0
    assert candidate["ltr_tail_loss_weight"] == 2.0
    assert control["steps"] == candidate["steps"]
    assert control["batch_size"] == candidate["batch_size"]


def test_literal_margin_arm_is_size_matched_and_targets_legal_close_branch() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1765",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1765,
        role="screening",
        recommended_slug="literal-margin",
    )
    by_id = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    control = by_id["c20260802-c1765-control"]
    candidate = by_id["c20260802-c1765-literal-margin"]

    assert matrix["recommended_experiment_id"].endswith("-literal-margin")
    assert control["compiler_alignment_loss_weight"] == 0.0
    assert candidate["compiler_alignment_loss_weight"] == 1.0
    assert candidate["compiler_alignment_margin"] == 1.0
    assert candidate["compiler_alignment_stratified"] is True
    assert candidate["compiler_alignment_kind_filter"] == "literal-close"
    assert control["steps"] == candidate["steps"]
    assert control["batch_size"] == candidate["batch_size"]


def test_completed_literal_close_null_steers_to_literal_margin() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1764",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1764,
        role="screening",
        recommended_slug="literal-close",
    )

    priorities = _mod._completed_candidate_priorities(
        matrix,
        "c20260802-c1764-literal-close",
        resolved_infrastructure=False,
    )

    assert priorities[0].area == "model"
    assert priorities[0].proposed_experiment_id.endswith("-literal-margin")
    assert "literal-margin" in priorities[0].hypothesis


def test_legacy_literal_close_matrix_steers_to_new_literal_margin() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1764",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1764,
        role="screening",
        recommended_slug="literal-close",
    )
    matrix["hypotheses"] = [
        row
        for row in matrix["hypotheses"]
        if not row["experiment"]["experiment_id"].endswith("-literal-margin")
    ]

    priorities = _mod._completed_candidate_priorities(
        matrix,
        "c20260802-c1764-literal-close",
        resolved_infrastructure=False,
    )

    assert priorities[0].proposed_experiment_id == ("c20260802-c1764-literal-margin")


def test_structural_screening_control_matches_recommended_head_capacity() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260731-c1729",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=22,
        cycle=1729,
        role="screening",
        recommended_slug="binder-topology",
    )
    by_id = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }

    control = by_id["c20260731-c1729-control"]
    candidate = by_id["c20260731-c1729-binder-topology"]
    assert control["binder_topology_loss_weight"] == 0.0
    assert candidate["binder_topology_loss_weight"] == 0.25
    assert control["binder_topology_decode_weight"] == 0.0
    assert candidate["binder_topology_decode_weight"] == 1.0
    assert control["structural_aux_head_profile"] == "binder-topology"
    assert candidate["structural_aux_head_profile"] == "binder-topology"
    assert control["compiler_decode_mode"] == "tree"
    assert candidate["compiler_decode_mode"] == "tree"


@pytest.mark.parametrize(
    ("slug", "loss_key", "decode_key", "profile"),
    case_values(__file__, "test_binder_quality_screening_controls_are_size_matched"),
)
def test_binder_quality_screening_controls_are_size_matched(
    slug: str, loss_key: str, decode_key: str, profile: str
) -> None:
    campaign_id = f"continuous-loop-20260802-{slug}"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1786,
        role="screening",
        recommended_slug=slug,
    )
    by_id = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = by_id[f"{prefix}-control"]
    candidate = by_id[f"{prefix}-{slug}"]
    assert control[loss_key] == 0.0
    assert control[decode_key] == 0.0
    assert candidate[loss_key] == 1.0
    assert candidate[decode_key] == 1.0
    assert control["structural_aux_head_profile"] == profile
    assert candidate["structural_aux_head_profile"] == profile
    assert control["compiler_decode_mode"] == "tree"
    assert candidate["compiler_decode_mode"] == "tree"


def test_structural_screening_arms_couple_training_to_decode() -> None:
    by_slug = {slug: extras for slug, _hypothesis, extras in _mod._SCREENING_ARM_BANK}

    for slug, prefix in (
        ("component-plan", "component_plan"),
        ("component-edge", "component_edge"),
        ("component-inventory", "component_inventory"),
        ("binder-topology", "binder_topology"),
        ("binder-arity", "binder_arity"),
        ("binder-component-plan", "binder_component_plan"),
    ):
        assert by_slug[slug][f"{prefix}_loss_weight"] > 0.0
        assert by_slug[slug][f"{prefix}_decode_weight"] > 0.0
        assert by_slug[slug]["compiler_decode_mode"] == "tree"

    joint = by_slug["component-structure"]
    assert joint["component_plan_loss_weight"] > 0.0
    assert joint["component_plan_decode_weight"] > 0.0
    assert joint["component_edge_loss_weight"] > 0.0
    assert joint["component_edge_decode_weight"] > 0.0
    assert joint["compiler_decode_mode"] == "tree"


def test_fidelity_screening_arm_is_size_matched_training_objective() -> None:
    campaign_id = "continuous-loop-20260802-c1791"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=22,
        cycle=1791,
        role="screening",
        recommended_slug="fidelity",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")

    assert knobs[f"{prefix}-control"]["fidelity_loss_weight"] == 0.5
    assert knobs[f"{prefix}-fidelity"]["fidelity_loss_weight"] == 1.5
    assert (
        _mod._arm_slug_from_knobs(
            knobs[f"{prefix}-fidelity"], candidate_id=f"{prefix}-fidelity"
        )
        == "fidelity"
    )


def test_edge_alignment_arm_is_size_matched_training_objective() -> None:
    campaign_id = "continuous-loop-20260802-c1794"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1794,
        role="screening",
        recommended_slug="edge-alignment",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-edge-alignment"]

    assert control["component_edge_alignment_loss_weight"] == 0.0
    assert candidate["component_edge_alignment_loss_weight"] == 1.0
    assert control["structural_aux_head_profile"] == "component-edge"
    assert candidate["structural_aux_head_profile"] == "component-edge"
    assert _mod._arm_slug_from_knobs(candidate) == "edge-alignment"


def test_semantic_contrast_arm_matches_pair_exposure_and_changes_only_weight() -> None:
    campaign_id = "continuous-loop-20260802-c1796"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1796,
        role="screening",
        recommended_slug="semantic-contrast",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-semantic-contrast"]

    assert control["semantic_contrast_loss_weight"] == 0.0
    assert candidate["semantic_contrast_loss_weight"] == 0.25
    for key in (
        "semantic_contrast_dir",
        "semantic_contrast_margin",
        "semantic_contrast_fraction",
        "steps",
        "batch_size",
    ):
        assert control[key] == candidate[key]
    assert _mod._arm_slug_from_knobs(candidate) == "semantic-contrast"


def test_slot_augmentation_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260802-c1799"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1799,
        role="screening",
        recommended_slug="slot-augmentation",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-slot-augmentation"]

    assert control["symbol_slot_augmentation"] is False
    assert candidate["symbol_slot_augmentation"] is True
    assert _mod._arm_slug_from_knobs(candidate) == "slot-augmentation"
    assert "symbol_slot_augmentation" in _mod._LEVER_KNOB_KEYS


def test_mixed_mask_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260802-c1800"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1800,
        role="screening",
        recommended_slug="mixed-mask",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-mixed-mask"]

    assert control["mask_pattern"] == "random"
    assert candidate["mask_pattern"] == "mixed"
    assert _mod._arm_slug_from_knobs(candidate) == "mixed-mask"
    assert "mask_pattern" in _mod._LEVER_KNOB_KEYS


def test_symbol_boundary_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260802-c1801"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1801,
        role="screening",
        recommended_slug="symbol-boundary",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-symbol-boundary"]

    assert control["symbol_boundary_loss_weight"] == 0.0
    assert candidate["symbol_boundary_loss_weight"] == 1.0
    assert _mod._arm_slug_from_knobs(candidate) == "symbol-boundary"
    assert "symbol_boundary_loss_weight" in _mod._LEVER_KNOB_KEYS

    experiment = next(
        row["experiment"]
        for row in matrix["hypotheses"]
        if row["experiment"]["experiment_id"].endswith("-symbol-boundary")
    )
    manifest = _mod._manifest(campaign_id, experiment, "a" * 40)
    arm_shas = {arm.role: arm.config_sha256 for arm in manifest.arms}
    assert arm_shas["control"] != arm_shas["candidate"]


def test_completed_candidate_projects_new_arm_missing_from_frozen_matrix() -> None:
    campaign_id = "continuous-loop-20260802-c1800"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1800,
        role="screening",
        recommended_slug="mixed-mask",
    )
    matrix["hypotheses"] = [
        row
        for row in matrix["hypotheses"]
        if not row["experiment"]["experiment_id"].endswith("-symbol-boundary")
    ]
    candidate_id = "c20260802-c1800-mixed-mask"
    skip = {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    skip.difference_update({"mixed-mask", "symbol-boundary"})

    priorities = _mod._completed_candidate_priorities(
        matrix,
        candidate_id,
        resolved_infrastructure=False,
        skip_slugs=skip,
    )

    assert priorities[0].proposed_experiment_id == "c20260802-c1800-symbol-boundary"
    assert "symbol-boundary" in priorities[0].hypothesis


def test_design_dropout_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260802-c1802"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1802,
        role="screening",
        recommended_slug="design-dropout",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-design-dropout"]
    assert control["design_md_dropout"] == 0.0
    assert candidate["design_md_dropout"] == 0.25
    assert _mod._arm_slug_from_knobs(candidate) == "design-dropout"
    assert "design_md_dropout" in _mod._LEVER_KNOB_KEYS


def test_scaffold_prefix_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260802-c1803"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1803,
        role="screening",
        recommended_slug="scaffold-prefix",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-scaffold-prefix"]
    assert control["ltr_prefix_loss_weight"] == 0.0
    assert candidate["ltr_prefix_loss_weight"] == 1.0
    assert _mod._arm_slug_from_knobs(candidate) == "scaffold-prefix"
    assert "ltr_prefix_loss_weight" in _mod._LEVER_KNOB_KEYS


def test_component_token_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260802-c1804"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1804,
        role="screening",
        recommended_slug="component-token",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-component-token"]
    assert control["component_token_loss_weight"] == 0.0
    assert candidate["component_token_loss_weight"] == 1.0
    assert _mod._arm_slug_from_knobs(candidate) == "component-token"
    assert "component_token_loss_weight" in _mod._LEVER_KNOB_KEYS


def test_structure_token_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260802-c1806"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1806,
        role="screening",
        recommended_slug="structure-token",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-structure-token"]
    assert control["structure_token_loss_weight"] == 0.0
    assert candidate["structure_token_loss_weight"] == 1.0
    assert _mod._arm_slug_from_knobs(candidate) == "structure-token"
    assert "structure_token_loss_weight" in _mod._LEVER_KNOB_KEYS


def test_component_edge_token_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260803-c1817"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1817,
        role="screening",
        recommended_slug="component-edge-token",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-component-edge-token"]
    assert control["component_edge_token_loss_weight"] == 0.0
    assert candidate["component_edge_token_loss_weight"] == 1.0
    assert _mod._arm_slug_from_knobs(candidate) == "component-edge-token"
    assert "component_edge_token_loss_weight" in _mod._LEVER_KNOB_KEYS


def test_component_edge_margin_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260803-c1818"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1818,
        role="screening",
        recommended_slug="component-edge-margin",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-component-edge-margin"]
    assert control["compiler_alignment_loss_weight"] == 0.0
    assert candidate["compiler_alignment_loss_weight"] == 1.0
    assert candidate["compiler_alignment_margin"] == 1.0
    assert candidate["compiler_alignment_kind_filter"] == "component-edge"
    assert _mod._arm_slug_from_knobs(candidate) == "component-edge-margin"


def test_compiler_decision_token_arm_is_dense_and_size_matched() -> None:
    campaign_id = "continuous-loop-20260803-c1821"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1821,
        role="screening",
        recommended_slug="compiler-decision-token",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-compiler-decision-token"]
    assert control["compiler_decision_token_loss_weight"] == 0.0
    assert candidate["compiler_decision_token_loss_weight"] == 1.0
    assert _mod._arm_slug_from_knobs(candidate) == "compiler-decision-token"
    assert "compiler_decision_token_loss_weight" in _mod._LEVER_KNOB_KEYS


def test_compiler_decision_margin_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260803-c1824"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1824,
        role="screening",
        recommended_slug="compiler-decision-margin",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-compiler-decision-margin"]
    assert control["compiler_alignment_loss_weight"] == 0.0
    assert candidate["compiler_alignment_loss_weight"] == 1.0
    assert candidate["compiler_alignment_margin"] == 1.0
    assert candidate["compiler_alignment_stratified"] is True
    assert candidate["compiler_alignment_kind_filter"] == "all"
    assert _mod._arm_slug_from_knobs(candidate) == "compiler-decision-margin"


def test_bounded_compiler_decision_margin_isolates_runtime_treatment() -> None:
    campaign_id = "continuous-loop-20260803-c1825"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1825,
        role="screening",
        recommended_slug="bounded-compiler-decision-margin",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-bounded-compiler-decision-margin"]
    for key in (
        "compiler_alignment_loss_weight",
        "compiler_alignment_margin",
        "compiler_alignment_stratified",
        "compiler_alignment_kind_filter",
    ):
        assert candidate[key] == control[key]
    assert control["grammar_completion_bounds"] is False
    assert candidate["grammar_completion_bounds"] is True
    assert _mod._arm_slug_from_knobs(candidate) == ("bounded-compiler-decision-margin")
    assert f"{prefix}-compiler-decision-margin" not in knobs
    # Private bank keys must never materialize into experiment knobs
    # (extra-forbidden schema). Regression for OFAT control package leak of
    # ``_thrash_slug`` after thrash bank rotate onto treatment_key arms.
    for eid, arm_knobs in knobs.items():
        private = [k for k in arm_knobs if str(k).startswith("_")]
        assert not private, f"{eid} leaked private knobs {private}"
    HypothesisMatrix.model_validate(matrix)


def test_cached_compiler_decision_margin_isolates_runtime_treatment() -> None:
    campaign_id = "continuous-loop-20260803-c1827"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1827,
        role="screening",
        recommended_slug="cached-compiler-decision-margin",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-cached-compiler-decision-margin"]
    for key in (
        "compiler_alignment_loss_weight",
        "compiler_alignment_margin",
        "compiler_alignment_stratified",
        "compiler_alignment_kind_filter",
    ):
        assert candidate[key] == control[key]
    assert control["grammar_equivalence_cache"] is False
    assert candidate["grammar_equivalence_cache"] is True
    assert f"{prefix}-compiler-decision-margin" not in knobs
    assert _mod._arm_slug_from_knobs(candidate) == "cached-compiler-decision-margin"
    HypothesisMatrix.model_validate(matrix)


def test_wide_draft_compiler_decision_margin_isolates_runtime_treatment() -> None:
    campaign_id = "continuous-loop-20260803-c1828"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1828,
        role="screening",
        recommended_slug="wide-draft-compiler-decision-margin",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-wide-draft-compiler-decision-margin"]
    for key in (
        "compiler_alignment_loss_weight",
        "compiler_alignment_margin",
        "compiler_alignment_stratified",
        "compiler_alignment_kind_filter",
    ):
        assert candidate[key] == control[key]
    assert control["grammar_draft_window"] == 8
    assert candidate["grammar_draft_window"] == 16
    assert f"{prefix}-compiler-decision-margin" not in knobs
    assert _mod._arm_slug_from_knobs(candidate) == (
        "wide-draft-compiler-decision-margin"
    )
    HypothesisMatrix.model_validate(matrix)


def test_capacity_aware_compiler_decision_margin_isolates_sampler_treatment() -> None:
    campaign_id = "continuous-loop-20260803-c1829"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1829,
        role="screening",
        recommended_slug="capacity-aware-compiler-decision-margin",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-capacity-aware-compiler-decision-margin"]
    for key in (
        "compiler_alignment_loss_weight",
        "compiler_alignment_margin",
        "compiler_alignment_stratified",
        "compiler_alignment_kind_filter",
    ):
        assert candidate[key] == control[key]
    assert control.get("mixture_sampling_policy") is None
    assert candidate["mixture_sampling_policy"] == "capacity_aware"
    assert f"{prefix}-compiler-decision-margin" not in knobs
    assert _mod._arm_slug_from_knobs(candidate) == (
        "capacity-aware-compiler-decision-margin"
    )
    HypothesisMatrix.model_validate(matrix)


def test_capacity_aware_tail_margin_isolates_closure_treatment() -> None:
    campaign_id = "continuous-loop-20260803-c1830"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1830,
        role="screening",
        recommended_slug="capacity-aware-tail-compiler-decision-margin",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-capacity-aware-tail-compiler-decision-margin"]
    for key in (
        "compiler_alignment_loss_weight",
        "compiler_alignment_margin",
        "compiler_alignment_stratified",
        "compiler_alignment_kind_filter",
        "mixture_sampling_policy",
    ):
        assert candidate[key] == control[key]
    assert control["ltr_tail_loss_weight"] == 0.0
    assert candidate["ltr_tail_loss_weight"] == 1.0
    assert f"{prefix}-capacity-aware-compiler-decision-margin" not in knobs
    assert _mod._arm_slug_from_knobs(candidate) == (
        "capacity-aware-tail-compiler-decision-margin"
    )
    HypothesisMatrix.model_validate(matrix)


def test_capacity_aware_semantic_exhaustive_isolates_decision_coverage() -> None:
    campaign_id = "continuous-loop-20260803-c1841"
    slug = "capacity-aware-semantic-exhaustive-compiler-decision-margin"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1841,
        role="screening",
        recommended_slug=slug,
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-{slug}"]
    for key in (
        "compiler_alignment_loss_weight",
        "compiler_alignment_margin",
        "compiler_alignment_stratified",
        "compiler_alignment_kind_filter",
        "mixture_sampling_policy",
    ):
        assert candidate[key] == control[key]
    assert not control.get("compiler_alignment_semantic_exhaustive", False)
    assert candidate["compiler_alignment_semantic_exhaustive"] is True
    assert f"{prefix}-capacity-aware-compiler-decision-margin" not in knobs
    assert _mod._arm_slug_from_knobs(candidate) == slug
    HypothesisMatrix.model_validate(matrix)


def test_semantic_exhaustive_structure_token_isolates_scaffold_recovery() -> None:
    campaign_id = "continuous-loop-20260803-c1843"
    slug = "capacity-aware-semantic-exhaustive-structure-token-margin"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1843,
        role="screening",
        recommended_slug=slug,
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-{slug}"]
    for key in (
        "compiler_alignment_loss_weight",
        "compiler_alignment_margin",
        "compiler_alignment_stratified",
        "compiler_alignment_semantic_exhaustive",
        "compiler_alignment_kind_filter",
        "mixture_sampling_policy",
    ):
        assert candidate[key] == control[key]
    assert control["structure_token_loss_weight"] == 0.0
    assert candidate["structure_token_loss_weight"] == 1.0
    assert (
        f"{prefix}-capacity-aware-semantic-exhaustive-compiler-decision-margin"
        not in knobs
    )
    assert _mod._arm_slug_from_knobs(candidate) == slug
    HypothesisMatrix.model_validate(matrix)


def test_exposure_targeted_isolates_sampling_policy() -> None:
    campaign_id = "continuous-loop-20260803-c1844"
    slug = "exposure-targeted-compiler-decision-margin"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1844,
        role="screening",
        recommended_slug=slug,
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-{slug}"]
    for key in (
        "compiler_alignment_loss_weight",
        "compiler_alignment_margin",
        "compiler_alignment_stratified",
        "compiler_alignment_kind_filter",
    ):
        assert candidate[key] == control[key]
    assert control["mixture_sampling_policy"] == "capacity_aware"
    assert candidate["mixture_sampling_policy"] == "exposure_targeted"
    assert f"{prefix}-capacity-aware-compiler-decision-margin" not in knobs
    assert _mod._arm_slug_from_knobs(candidate) == slug
    HypothesisMatrix.model_validate(matrix)


def test_exposure_targeted_semantic_exhaustive_isolates_compression() -> None:
    campaign_id = "continuous-loop-20260803-c1847"
    slug = "exposure-targeted-semantic-exhaustive-compiler-decision-margin"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1847,
        role="screening",
        recommended_slug=slug,
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-{slug}"]
    assert control["mixture_sampling_policy"] == "exposure_targeted"
    assert candidate["mixture_sampling_policy"] == "exposure_targeted"
    assert not control.get("compiler_alignment_semantic_exhaustive", False)
    assert candidate["compiler_alignment_semantic_exhaustive"] is True
    assert _mod._arm_slug_from_knobs(candidate) == slug
    HypothesisMatrix.model_validate(matrix)


def test_slot_component_coverage_is_registered_as_distinct_quality_successor() -> None:
    slug = "slot-component-coverage"
    by_slug = {name: extras for name, _hypothesis, extras in _mod._SCREENING_ARM_BANK}
    knobs = by_slug[slug]
    assert knobs["slot_component_loss_weight"] == 1.0
    assert knobs["slot_component_decode_weight"] == 1.0
    assert knobs["compiler_decode_mode"] == "tree"
    assert _mod._arm_slug_from_knobs(knobs) == slug

    skip = set(by_slug) - {slug}
    assert _mod._select_recommended_slug(1900, skip=skip) == slug


def test_slot_component_fidelity_coupling_is_distinct_follow_on() -> None:
    slug = "slot-component-fidelity-coupling"
    by_slug = {name: extras for name, _hypothesis, extras in _mod._SCREENING_ARM_BANK}
    knobs = by_slug[slug]
    assert knobs["slot_component_loss_weight"] == 1.0
    assert knobs["slot_component_decode_weight"] == 1.0
    assert knobs["fidelity_loss_weight"] == 1.5
    assert knobs["compiler_decode_mode"] == "tree"
    assert _mod._arm_slug_from_knobs(knobs) == slug

    skip = set(by_slug) - {slug}
    assert _mod._select_recommended_slug(1901, skip=skip) == slug


def test_slot_component_inventory_coupling_is_distinct_follow_on() -> None:
    slug = "slot-component-inventory-coupling"
    by_slug = {name: extras for name, _hypothesis, extras in _mod._SCREENING_ARM_BANK}
    knobs = by_slug[slug]
    assert knobs["slot_component_loss_weight"] == 1.0
    assert knobs["component_inventory_loss_weight"] == 1.0
    assert knobs["component_inventory_decode_weight"] == 1.0
    assert knobs["compiler_decode_mode"] == "tree"
    assert _mod._arm_slug_from_knobs(knobs) == slug

    skip = set(by_slug) - {slug}
    assert _mod._select_recommended_slug(1902, skip=skip) == slug


def test_slot_component_exposure_cap_is_distinct_data_successor() -> None:
    slug = "slot-component-exposure-cap"
    by_slug = {name: extras for name, _hypothesis, extras in _mod._SCREENING_ARM_BANK}
    knobs = by_slug[slug]
    assert knobs["slot_component_loss_weight"] == 1.0
    assert knobs["mixture_sampling_policy"] == "exposure_targeted"
    assert knobs["mixture_per_template_cap"] == 2
    assert knobs["compiler_decode_mode"] == "tree"
    assert _mod._arm_slug_from_knobs(knobs) == slug

    skip = set(by_slug) - {slug}
    assert _mod._select_recommended_slug(1903, skip=skip) == slug


def test_frozen_screening_retry_preserves_champion_enqueue_semantics() -> None:
    replay = {"handoff": SimpleNamespace(cycle_role="screening")}

    assert _mod._screening_enqueue_allowed(
        cycle_intent="retry_measurement", replay=replay
    )
    assert not _mod._screening_enqueue_allowed(
        cycle_intent="retry_measurement",
        replay={"handoff": SimpleNamespace(cycle_role="promotion")},
    )
    assert not _mod._screening_enqueue_allowed(
        cycle_intent="retry_measurement",
        replay={
            "handoff": SimpleNamespace(cycle_role="screening", cycle_intent="confirm")
        },
    )


def test_completed_confirmation_replay_resolves_original_and_duplicate(
    tmp_path: Path,
) -> None:
    campaign_id = "continuous-loop-20260803-loop-c1838"
    camp_dir = tmp_path / campaign_id
    camp_dir.mkdir()
    (camp_dir / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "cycle_index": 1838,
                "cycle_intent": "retry_measurement",
            }
        )
    )
    (camp_dir / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "positive": True,
                "measurement_complete": True,
                "primary_metric": "smoke.structural_similarity",
                "reasons": ["primary_metric_win:smoke.structural_similarity:0.4->0.44"],
            }
        )
    )
    entries = [
        {
            "entry_id": "original",
            "status": "confirmation_inconclusive",
            "knobs_fingerprint": "same",
        },
        {
            "entry_id": "duplicate",
            "status": "queued",
            "knobs_fingerprint": "same",
            "source_campaign_id": campaign_id,
            "source_candidate_id": "candidate-confirm",
        },
    ]

    assert _mod._reconcile_completed_confirmation_replays(tmp_path, entries)
    assert entries[0]["status"] == "confirmed"
    assert entries[0]["confirm_campaign_id"] == campaign_id
    assert entries[1]["status"] == "skipped_duplicate"


def test_typed_family_balance_arm_is_size_matched_and_replayable() -> None:
    campaign_id = "continuous-loop-20260803-c1807"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1807,
        role="screening",
        recommended_slug="typed-family-balance",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-typed-family-balance"]
    assert control["typed_family_balance_loss_weight"] == 0.0
    assert candidate["typed_family_balance_loss_weight"] == 0.25
    assert _mod._arm_slug_from_knobs(candidate) == "typed-family-balance"
    assert "typed_family_balance_loss_weight" in _mod._LEVER_KNOB_KEYS


def test_container_close_arm_is_size_matched_and_targets_legal_close_branch() -> None:
    campaign_id = "continuous-loop-20260803-c1808"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1808,
        role="screening",
        recommended_slug="container-close",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-container-close"]
    assert control["compiler_alignment_loss_weight"] == 0.0
    assert candidate["compiler_alignment_loss_weight"] == 1.0
    assert candidate["compiler_alignment_margin"] == 1.0
    assert candidate["compiler_alignment_kind_filter"] == "container-close"
    assert _mod._arm_slug_from_knobs(candidate) == "container-close"


def test_balanced_container_close_arm_combines_quality_and_close_objectives() -> None:
    campaign_id = "continuous-loop-20260803-c1809"
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1809,
        role="screening",
        recommended_slug="balanced-container-close",
    )
    knobs = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    prefix = campaign_id.replace("continuous-loop-", "c")
    control = knobs[f"{prefix}-control"]
    candidate = knobs[f"{prefix}-balanced-container-close"]
    assert control["typed_family_balance_loss_weight"] == 0.0
    assert control["compiler_alignment_loss_weight"] == 0.0
    assert candidate["typed_family_balance_loss_weight"] == 0.25
    assert candidate["compiler_alignment_loss_weight"] == 1.0
    assert candidate["compiler_alignment_margin"] == 1.0
    assert candidate["compiler_alignment_kind_filter"] == "container-close"
    assert _mod._arm_slug_from_knobs(candidate) == "balanced-container-close"


def test_confirmed_champion_reconfirms_when_bank_exhausted_before_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Legacy confirm-fallback path: only active while terminal parking is off.
    _inject_terminal_policy(monkeypatch, park=False)
    exhausted = {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    champion = {"status": "confirmed", "entry_id": "champ-1"}

    assert _mod._repeat_confirm_while_waiting_for_promotion(
        cadence_role="screening",
        confirmed_champion=champion,
        cycle=1811,
        skip=exhausted,
    )
    assert not _mod._repeat_confirm_while_waiting_for_promotion(
        cadence_role="promotion",
        confirmed_champion=champion,
        cycle=1812,
        skip=exhausted,
    )
    assert not _mod._repeat_confirm_while_waiting_for_promotion(
        cadence_role="screening",
        confirmed_champion=champion,
        cycle=1811,
        skip=exhausted - {"balanced-container-close"},
    )


def test_completed_frozen_retry_steers_to_distinct_quality_arm() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260731-c10",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=80,
        cycle=10,
        role="screening",
        recommended_slug="batch1",
    )
    candidate_id = "c20260731-c10-batch1"
    matrix["next_run_priorities"][0].update(
        {
            "area": "infrastructure",
            "hypothesis": "The repaired harness completes the frozen replay.",
            "authority": "observed_result",
            "proposed_experiment_id": candidate_id,
        }
    )

    priorities = _mod._completed_retry_priorities(matrix, candidate_id)

    assert priorities[0].area == "model"
    assert priorities[0].proposed_experiment_id == "c20260731-c10-component-plan"
    assert "resolved infrastructure" in priorities[0].expected_information_gain


def test_completed_null_steers_to_distinct_quality_arm() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260731-c1729",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=22,
        cycle=1729,
        role="screening",
        recommended_slug="binder-topology",
    )
    candidate_id = "c20260731-c1729-binder-topology"

    priorities = _mod._completed_candidate_priorities(
        matrix, candidate_id, resolved_infrastructure=False
    )

    assert priorities[0].proposed_experiment_id == "c20260731-c1729-component-plan"
    assert "completed null" in priorities[0].expected_information_gain
    assert "component-plan" in priorities[0].hypothesis


def test_completed_null_with_exhausted_bank_requires_harness_expansion() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1795",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=21,
        cycle=1795,
        role="screening",
        recommended_slug="edge-alignment",
    )
    candidate_id = matrix["recommended_experiment_id"]
    matrix["hypotheses"] = [
        row
        for row in matrix["hypotheses"]
        if row["experiment"]["experiment_id"]
        in {
            candidate_id,
            "c20260802-c1795-control",
        }
    ]

    priorities = _mod._completed_candidate_priorities(
        matrix,
        candidate_id,
        resolved_infrastructure=False,
    )

    assert priorities[0].area == "model_build"
    assert priorities[0].disposition == "monitor"
    assert priorities[0].proposed_experiment_id is None
    assert all(row.disposition != "experiment_next" for row in priorities)
    assert "preregister" in priorities[0].hypothesis


def test_completed_null_prioritizes_new_quality_objective_before_runtime() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1735",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=22,
        cycle=1735,
        role="screening",
        recommended_slug="component-structure",
    )
    candidate_id = matrix["recommended_experiment_id"]

    priorities = _mod._completed_candidate_priorities(
        matrix,
        candidate_id,
        resolved_infrastructure=False,
        skip_slugs={
            "component-plan",
            "component-edge",
            "component-inventory",
            "binder-topology",
        },
    )

    assert priorities[0].area == "model"
    assert priorities[0].proposed_experiment_id.endswith("-binder-arity")
    assert "quality" in priorities[0].hypothesis


def test_completed_confirmation_replaces_stale_preregistered_priorities() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1759",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1759,
        role="screening",
        confirm_levers={"steps": 44, "batch_size": 2},
        confirm_control_levers={"steps": 22, "batch_size": 2},
    )
    candidate_id = matrix["recommended_experiment_id"]

    priorities = _mod._completed_confirmation_priorities(
        matrix,
        candidate_id,
        {
            "positive": False,
            "primary_metric": "smoke.structural_similarity",
            "control_metrics": {
                "smoke.structural_similarity": 0.17416666666666666,
                "meaningful_program_rate": 1 / 3,
            },
            "candidate_metrics": {
                "smoke.structural_similarity": 0.0575,
                "meaningful_program_rate": 0.0,
            },
        },
        {"status": "rejected"},
    )

    assert "rejected the champion fingerprint" in priorities[0].hypothesis
    assert "0.17416666666666666->0.0575" in priorities[0].hypothesis
    assert priorities[0].proposed_experiment_id is None
    assert all(
        priority.proposed_experiment_id != candidate_id for priority in priorities
    )
    executable = [
        priority for priority in priorities if priority.disposition == "experiment_next"
    ]
    assert len(executable) == 1
    assert executable[0].area == "experiments"
    assert executable[0].proposed_experiment_id.endswith("-batch1")
    assert "runtime diagnostic" in executable[0].hypothesis


def test_completed_confirmation_that_reholds_steers_to_promotion() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1760",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1760,
        role="screening",
        confirm_levers={"steps": 44, "batch_size": 2},
        confirm_control_levers={"steps": 22, "batch_size": 2},
    )
    candidate_id = matrix["recommended_experiment_id"]

    priorities = _mod._completed_confirmation_priorities(
        matrix,
        candidate_id,
        {"positive": True},
        {"status": "confirmed"},
    )

    assert priorities[0].area == "promotion"
    assert priorities[0].disposition == "experiment_next"
    assert priorities[0].proposed_experiment_id == candidate_id
    assert "re-held" in priorities[0].hypothesis


def test_completed_confirmation_uses_resolution_not_raw_positive() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260802-c1761",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1761,
        role="screening",
        confirm_levers={"fidelity_loss_weight": 1.5},
        confirm_control_levers={"fidelity_loss_weight": 0.5},
    )
    candidate_id = matrix["recommended_experiment_id"]

    priorities = _mod._completed_confirmation_priorities(
        matrix,
        candidate_id,
        {
            "positive": True,
            "primary_metric": "smoke.structural_similarity",
            "control_metrics": {"smoke.structural_similarity": 0.45},
            "candidate_metrics": {"smoke.structural_similarity": 0.44},
        },
        {"status": "rejected"},
    )

    assert priorities[0].area != "promotion"
    assert all(
        priority.proposed_experiment_id != candidate_id for priority in priorities
    )


def test_queued_candidate_priorities_require_fresh_confirmation_before_lean() -> None:
    priorities = _mod._queued_candidate_priorities("cycle-candidate", "campaign:cycle")

    assert priorities[0].area == "evaluation"
    assert priorities[0].authority == "observed_result"
    assert priorities[0].proposed_experiment_id == (
        "cycle-candidate-fresh-confirmation"
    )
    assert "fresh seed" in priorities[0].hypothesis
    assert priorities[1].area == "lean_model"
    assert priorities[1].authority == "lean_assumption"
    assert "until fresh confirmation" in priorities[1].hypothesis


def test_predecessor_completed_null_drives_next_screening_arm(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "continuous-loop-20260731-c1729"
    camp.mkdir(parents=True)
    matrix = _mod._matrix(
        campaign_id=camp.name,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=22,
        cycle=1729,
        role="screening",
        recommended_slug="binder-topology",
    )
    candidate_id = matrix["recommended_experiment_id"]
    control_id = matrix["hypotheses"][0]["experiment"]["experiment_id"]
    for arm in (control_id, candidate_id):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=0.0,
            structural_similarity=0.1,
            latency_ms_p50=1000.0,
        )
        _write_complete_scoreboard(run, "smoke")
    (camp / "matrix-proposal.json").write_text(json.dumps(matrix))
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "control_id": control_id,
                "candidate_id": candidate_id,
                "positive": False,
                "reasons": ["no_registered_effect"],
            }
        )
    )
    (camp / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "cycle_intent": "screening",
                "priorities": [
                    {
                        "rank": 1,
                        "disposition": "experiment_next",
                        "proposed_experiment_id": candidate_id,
                    }
                ],
            }
        )
    )

    assert (
        _mod._predecessor_priority_slug(root, camp.name, skip=set()) == "component-plan"
    )
    assert (
        _mod._predecessor_priority_slug(root, camp.name, skip={"component-plan"})
        == "component-edge"
    )


def test_predecessor_rejected_confirmation_uses_outcome_conditioned_successor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "continuous-loop-20260802-c1759"
    camp.mkdir(parents=True)
    matrix = _mod._matrix(
        campaign_id=camp.name,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1759,
        role="screening",
        confirm_levers={"steps": 44, "batch_size": 2},
        confirm_control_levers={"steps": 22, "batch_size": 2},
    )
    control_id = matrix["hypotheses"][0]["experiment"]["experiment_id"]
    candidate_id = matrix["recommended_experiment_id"]
    for arm, structure, meaning, latency in (
        (control_id, 0.17416666666666666, 1 / 3, 1362.47),
        (candidate_id, 0.0575, 0.0, 3195.06),
    ):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=meaning,
            structural_similarity=structure,
            latency_ms_p50=latency,
        )
        _write_complete_scoreboard(run, "smoke")
    (camp / "matrix-proposal.json").write_text(json.dumps(matrix))
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "control_id": control_id,
                "candidate_id": candidate_id,
                "positive": True,
                "primary_metric": "smoke.structural_similarity",
                "measurement_complete": True,
                "control_metrics": {
                    "smoke.structural_similarity": 0.17416666666666666,
                    "meaningful_program_rate": 1 / 3,
                },
                "candidate_metrics": {
                    "smoke.structural_similarity": 0.0575,
                    "meaningful_program_rate": 0.0,
                },
            }
        )
    )
    (camp / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "cycle_intent": "confirm",
                "cycle_role": "screening",
                "climb_state": "champion_confirmed",
                "priorities": matrix["next_run_priorities"],
            }
        )
    )

    assert _mod._predecessor_priority_slug(root, camp.name, skip=set()) == "batch1"


def test_recent_completed_nonpositive_slugs_follow_predecessor_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    predecessor: str | None = None
    for cycle, slug in ((1732, "component-plan"), (1733, "component-edge")):
        campaign_id = f"continuous-loop-20260802-c{cycle}"
        camp = root / campaign_id
        matrix = _mod._matrix(
            campaign_id=campaign_id,
            evidence_snapshot_id="snap",
            cites=["docs/a.md", "docs/b.md", "docs/c.md"],
            role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
            train_version="wf_smoke_v2",
            eval_version="e_test",
            steps=22,
            cycle=cycle,
            role="screening",
            recommended_slug=slug,
        )
        candidate_id = matrix["recommended_experiment_id"]
        camp.mkdir(parents=True)
        (camp / "campaign.json").write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "loop_id": "loop-1",
                    "predecessor_campaign_id": predecessor,
                }
            )
        )
        (camp / "matrix-proposal.json").write_text(json.dumps(matrix))
        (camp / "sdlc_delivery.json").write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "cycle_intent": "screening",
                    "positive": False,
                    "measurement_complete": True,
                }
            )
        )
        (camp / "cycle_handoff.json").write_text(
            json.dumps({"loop_id": "loop-1", "cycle_intent": "screening"})
        )
        predecessor = campaign_id

    # Lineage mechanics tests use min_null_seeds=1; production default is 2.
    assert _mod._recent_completed_nonpositive_slugs(
        root, predecessor, min_null_seeds=1
    ) == {
        "component-plan",
        "component-edge",
    }

    assert _mod._recent_completed_nonpositive_slugs(
        root, predecessor, max_cycles=1, min_null_seeds=1
    ) == {"component-edge"}

    newest = root / str(predecessor) / "sdlc_delivery.json"
    newest_delivery = json.loads(newest.read_text())
    newest_delivery["cycle_intent"] = "retry_measurement"
    newest.write_text(json.dumps(newest_delivery))
    (root / str(predecessor) / "cycle_handoff.json").write_text(
        json.dumps({"loop_id": "loop-1", "cycle_intent": "retry_measurement"})
    )
    assert _mod._recent_completed_nonpositive_slugs(
        root, predecessor, min_null_seeds=1
    ) == {
        "component-plan",
        "component-edge",
    }

    newest_delivery = json.loads(newest.read_text())
    newest_delivery["measurement_complete"] = False
    newest.write_text(json.dumps(newest_delivery))
    assert _mod._recent_completed_nonpositive_slugs(
        root, predecessor, min_null_seeds=1
    ) == {"component-plan"}

    newest_delivery = json.loads(newest.read_text())
    candidate_id = newest_delivery["candidate_id"]
    newest_delivery["cycle_intent"] = "retry_measurement"
    newest.write_text(json.dumps(newest_delivery))
    (root / str(predecessor) / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "loop_id": "loop-1",
                "cycle_intent": "screening",
                "climb_state": "rejected",
                "reasons": [
                    f"candidate_runtime_unblock_reproduced:{candidate_id}",
                    "control_runtime_rejected_after_frozen_replay:control",
                ],
            }
        )
    )
    assert _mod._recent_completed_nonpositive_slugs(
        root, predecessor, min_null_seeds=1
    ) == {
        "component-plan",
        "component-edge",
    }


def test_recent_completed_nonpositive_slugs_reclassifies_stale_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "autoresearch"
    campaign_id = "continuous-loop-20260802-c1786"
    camp = root / campaign_id
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=22,
        cycle=1786,
        role="screening",
        recommended_slug="steps",
    )
    candidate_id = matrix["recommended_experiment_id"]
    camp.mkdir(parents=True)
    (camp / "campaign.json").write_text(
        json.dumps({"campaign_id": campaign_id, "loop_id": "loop-1"})
    )
    (camp / "matrix-proposal.json").write_text(json.dumps(matrix))
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "control_id": f"{campaign_id}-control",
                "primary_metric": "smoke.structural_similarity",
                "cycle_intent": "screening",
                "positive": True,
                "measurement_complete": True,
            }
        )
    )
    (camp / "cycle_handoff.json").write_text(
        json.dumps({"loop_id": "loop-1", "cycle_intent": "screening"})
    )
    ledger = root / "loops" / "loop-1" / "hillclimb_iterations.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{}\n")
    stats = ledger.with_name("slug_stats.json")
    stats.write_text(json.dumps({"schema": "thrash_slug_stats/v1", "slugs": {}}))
    calls = 0

    def classify(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "positive": False,
            "control_metrics": {"structural_similarity": 0.1},
            "candidate_metrics": {"structural_similarity": 0.2},
            "reasons": ["primary_quality_win_rejected_latency_budget"],
        }

    monkeypatch.setattr(_mod, "_classify_positive", classify)

    for _ in range(2):
        assert _mod._recent_completed_nonpositive_slugs(
            root, campaign_id, min_null_seeds=1
        ) == {"steps"}
    assert calls == 1
    _mod._RECENT_EXHAUSTION_CACHE.clear()
    assert _mod._recent_completed_nonpositive_slugs(
        root, campaign_id, min_null_seeds=1
    ) == {"steps"}
    assert calls == 1

    ledger.write_text("{}\n{}\n")
    _mod._RECENT_EXHAUSTION_CACHE.clear()
    assert _mod._recent_completed_nonpositive_slugs(
        root, campaign_id, min_null_seeds=1
    ) == {"steps"}
    assert calls == 2


def test_completed_null_does_not_age_out_of_lineage_exhaustion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    old_id = "continuous-loop-20260802-c1"
    old = root / old_id
    matrix = _mod._matrix(
        campaign_id=old_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1,
        role="screening",
        recommended_slug="bounds",
    )
    old.mkdir(parents=True)
    (old / "campaign.json").write_text(
        json.dumps(
            {
                "campaign_id": old_id,
                "loop_id": "loop-1",
                "predecessor_campaign_id": None,
            }
        )
    )
    (old / "matrix-proposal.json").write_text(json.dumps(matrix))
    (old / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "candidate_id": matrix["recommended_experiment_id"],
                "cycle_intent": "screening",
                "positive": False,
                "measurement_complete": True,
            }
        )
    )
    (old / "cycle_handoff.json").write_text(
        json.dumps({"loop_id": "loop-1", "cycle_intent": "screening"})
    )

    predecessor = old_id
    for cycle in range(2, _mod._RECENT_EXHAUSTION_CYCLE_WINDOW + 4):
        campaign_id = f"continuous-loop-20260802-c{cycle}"
        camp = root / campaign_id
        camp.mkdir()
        (camp / "campaign.json").write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "loop_id": "loop-1",
                    "predecessor_campaign_id": predecessor,
                }
            )
        )
        (camp / "cycle_handoff.json").write_text(
            json.dumps({"loop_id": "loop-1", "cycle_intent": "screening"})
        )
        predecessor = campaign_id

    assert "bounds" not in _mod._recent_completed_nonpositive_slugs(
        root,
        predecessor,
        max_cycles=_mod._RECENT_EXHAUSTION_CYCLE_WINDOW,
        min_null_seeds=1,
    )
    assert "bounds" in _mod._recent_completed_nonpositive_slugs(
        root, predecessor, min_null_seeds=1
    )
    # Production default (2 seeds) does not close on a single complete null.
    assert "bounds" not in _mod._recent_completed_nonpositive_slugs(
        root, predecessor, min_null_seeds=2
    )


def test_rejected_confirmation_exhausts_its_source_quality_family(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    campaign_id = "continuous-loop-20260802-c1777"
    camp = root / campaign_id
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1777,
        role="screening",
        confirm_levers={
            "component_inventory_loss_weight": 1.0,
            "component_inventory_decode_weight": 1.0,
        },
        confirm_control_levers={
            "component_inventory_loss_weight": 0.0,
            "component_inventory_decode_weight": 0.0,
        },
    )
    candidate_id = matrix["recommended_experiment_id"]
    camp.mkdir(parents=True)
    (camp / "campaign.json").write_text(
        json.dumps({"campaign_id": campaign_id, "loop_id": "loop-1"})
    )
    (camp / "matrix-proposal.json").write_text(json.dumps(matrix))
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "cycle_intent": "confirm",
                "positive": False,
                "measurement_complete": True,
            }
        )
    )
    (camp / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "loop_id": "loop-1",
                "cycle_intent": "confirm",
                "climb_state": "rejected",
            }
        )
    )

    assert _mod._recent_completed_nonpositive_slugs(
        root, campaign_id, min_null_seeds=1
    ) == {"component-inventory"}
    assert (
        _mod._recent_completed_nonpositive_slugs(root, campaign_id, min_null_seeds=2)
        == set()
    )


def test_predecessor_reclassifies_stale_positive_under_current_policy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "continuous-loop-20260801-c1731"
    matrix = _mod._matrix(
        campaign_id=camp.name,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=22,
        cycle=1731,
        role="screening",
        recommended_slug="component-plan",
    )
    candidate_id = matrix["recommended_experiment_id"]
    control_id = matrix["hypotheses"][0]["experiment"]["experiment_id"]
    for arm, latency in ((control_id, 3453.06), (candidate_id, 3430.55)):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=0.3333333333333333,
            structural_similarity=0.41973333333333335,
            latency_ms_p50=latency,
        )
        _write_complete_scoreboard(run, "smoke")
    (camp / "matrix-proposal.json").write_text(json.dumps(matrix))
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "control_id": control_id,
                "candidate_id": candidate_id,
                "positive": True,
                "reasons": ["historical_noise_scale_efficiency_win"],
            }
        )
    )
    (camp / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "cycle_role": "screening",
                "cycle_intent": "screening",
                "primary_metric": "smoke.structural_similarity",
                "priorities": [
                    {
                        "rank": 1,
                        "disposition": "experiment_next",
                        "proposed_experiment_id": candidate_id,
                    }
                ],
            }
        )
    )

    assert (
        _mod._predecessor_priority_slug(root, camp.name, skip=set()) == "component-edge"
    )


def test_reproduced_timeout_retirement_blocks_reintroduced_exact_arm(
    tmp_path: Path,
) -> None:
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        load_loop_exhausted_ledger,
    )

    root = tmp_path / "autoresearch"
    loop_id = "loop-1"
    predecessor = None
    for cycle in (1, 2):
        campaign_id = f"continuous-loop-test-c{cycle}"
        camp = root / campaign_id
        camp.mkdir(parents=True)
        control_id = f"c-test-c{cycle}-control"
        candidate_id = f"c-test-c{cycle}-binder-arity"
        (camp / "campaign.json").write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "loop_id": loop_id,
                    "predecessor_campaign_id": predecessor,
                }
            )
        )
        (camp / "matrix-proposal.json").write_text(
            json.dumps(
                {
                    "hypotheses": [
                        {
                            "experiment": {
                                "experiment_id": control_id,
                                "knobs": {
                                    "train_version": "wf_smoke_v2",
                                    "eval_version": "e_test",
                                    "binder_arity_loss_weight": 0.0,
                                    "binder_arity_decode_weight": 0.0,
                                },
                            }
                        },
                        {
                            "experiment": {
                                "experiment_id": candidate_id,
                                "knobs": {
                                    "train_version": "wf_smoke_v2",
                                    "eval_version": "e_test",
                                    "binder_arity_loss_weight": 1.0,
                                    "binder_arity_decode_weight": 1.0,
                                },
                            }
                        },
                    ]
                }
            )
        )
        (camp / "sdlc_delivery.json").write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "cycle_index": cycle,
                    "candidate_id": candidate_id,
                    "control_id": control_id,
                    "measurement_complete": cycle == 2,
                    "reasons": [
                        "measurement_incomplete:control:decode_timeout_count=1"
                    ],
                }
            )
        )
        (camp / "cycle_handoff.json").write_text(
            json.dumps(
                {
                    "loop_id": loop_id,
                    "cycle_index": cycle,
                    "cycle_intent": "retry_measurement" if cycle == 1 else "screening",
                    "primary_metric": "smoke.structural_similarity",
                    "reasons": (
                        [f"candidate_runtime_unblock_reproduced:{candidate_id}"]
                        if cycle == 1
                        else []
                    ),
                }
            )
        )
        predecessor = campaign_id

    policy = load_climb_policy()
    retired, signal_sources = _mod._sync_reproduced_timeout_retirements(
        root,
        loop_id,
        predecessor,
        policy=policy,
        train_version="wf_smoke_v2",
        eval_version="e_test",
        primary_metric="smoke.structural_similarity",
        direction="increase",
        claim_class="diagnostic",
    )
    assert retired == {"binder-arity"}
    assert signal_sources == ("continuous-loop-test-c2",)
    ledger = load_loop_exhausted_ledger(root, loop_id, policy)
    assert ledger.entries[0].reason == "reproduced_decode_timeout_retirement"

    current = "continuous-loop-test-c3"
    (root / current).mkdir()
    _mod._persist_selector_harness_signal(root, current, loop_id, signal_sources)
    signal = next((root / current / "artifacts" / "harness_signals").glob("*.json"))
    payload = json.loads(signal.read_text())
    assert payload["code"] == "screening_selector_reintroduced_retired_arm"
    assert payload["reproduced_on_frozen_input"] is True


def test_screening_saturation_parks_with_typed_constraint(tmp_path: Path) -> None:
    from slm_training.autoresearch.climb_policy import load_climb_policy

    root = tmp_path / "autoresearch"
    _write_terminal_feedback(root, "cycle-15")
    handoff = _mod.AutotrainCycleHandoffV1(
        loop_id="loop-1",
        campaign_id="cycle-15",
        cycle_index=15,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        cycle_role="screening",
        cycle_intent="screening",
        evidence_class="fixture",
        climb_state="rejected",
        ship_state="not_evaluated",
        primary_metric="smoke.structural_similarity",
        actions=(
            _mod.AutotrainActionV1(
                kind="document",
                owner="documenting-experiment-results",
                reason="persist the saturated screening result",
                evidence_ids=("campaign:cycle-15",),
            ),
        ),
    )
    (root / "cycle-15" / "cycle_handoff.json").write_text(
        handoff.model_dump_json(), encoding="utf-8"
    )
    status = _mod._park_screening_saturation(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-15",
        cycle_index=15,
        policy=load_climb_policy(),
        ranked_regimes=["component-plan", "component-edge"],
    )
    assert status == _mod._REGIME_PARKED_STATUS
    verdict = json.loads(
        (root / "loops" / "loop-1" / "terminal_verdict.json").read_text()
    )
    assert verdict["binding_constraint"] == "screening_objective_saturated"
    routed = _mod.AutotrainCycleHandoffV1.model_validate_json(
        (root / "cycle-15" / "cycle_handoff.json").read_text(encoding="utf-8")
    )
    assert [action.kind for action in routed.actions] == [
        "rebuild_data",
        "document",
        "next_experiment",
    ]
    # I10: the refresh must target the current rung, never a skipped one.
    assert _mod._current_rung_label() in routed.actions[0].reason
    assert "simplified" not in routed.actions[0].reason
    assert "Researcher once" in routed.actions[-1].reason


def test_saturated_objective_refresh_requires_typed_feedback(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires terminal HypothesisFeedback"):
        _mod._capability_objective_refresh_actions(
            root=tmp_path / "autoresearch", campaign_id="cycle-no-feedback"
        )


def test_promotion_cadence_null_exhausts_completed_arm(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "continuous-loop-20260801-c1732"
    matrix = _mod._matrix(
        campaign_id=camp.name,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=22,
        cycle=1732,
        role="promotion",
        recommended_slug="component-plan",
    )
    candidate_id = matrix["recommended_experiment_id"]
    control_id = matrix["hypotheses"][0]["experiment"]["experiment_id"]
    for arm in (control_id, candidate_id):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_held_out.json",
            suite="held_out",
            parse_rate=1.0,
            meaningful_program_rate=0.0,
            structural_similarity=0.06024,
            latency_ms_p50=2000.0,
        )
        _write_complete_scoreboard(run, "held_out")
    (camp / "matrix-proposal.json").write_text(json.dumps(matrix))
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "control_id": control_id,
                "candidate_id": candidate_id,
                "positive": False,
                "reasons": ["primary_metric_null_or_worse"],
            }
        )
    )
    (camp / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "cycle_role": "promotion",
                "cycle_intent": "promotion",
                "primary_metric": "held_out.structural_similarity",
                "priorities": [
                    {
                        "rank": 1,
                        "disposition": "experiment_next",
                        "proposed_experiment_id": candidate_id,
                    }
                ],
            }
        )
    )

    assert (
        _mod._predecessor_priority_slug(root, camp.name, skip=set()) == "component-edge"
    )


def test_matrix_promote_path_confirmed_knobs() -> None:
    from slm_training.autoresearch.schemas import HypothesisMatrix

    matrix = _mod._matrix(
        campaign_id="continuous-loop-20260731-c8",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=80,
        cycle=8,
        role="promotion",
        promote_levers={
            "grammar_completion_bounds": True,
            "compact_active_canvas": False,
            "steps": 81,
            "batch_size": 2,
            "train_version": "wf_smoke_v2",
            "mixture_sampling_policy": "capacity_aware",
            "compiler_alignment_semantic_exhaustive": True,
            "grammar_equivalence_cache": True,
            "grammar_draft_window": 16,
        },
    )
    HypothesisMatrix.model_validate(matrix)
    assert matrix["recommended_experiment_id"] == "c20260731-c8-promote"
    cand = matrix["hypotheses"][1]["experiment"]["knobs"]
    assert cand["grammar_completion_bounds"] is True
    assert cand["mixture_sampling_policy"] == "capacity_aware"
    assert cand["compiler_alignment_semantic_exhaustive"] is True
    assert cand["grammar_equivalence_cache"] is True
    assert cand["grammar_draft_window"] == 16
    assert cand["steps"] == 81
    # formal_claims must be on the matrix member (not rewritten post-lock).
    promo_exp = matrix["hypotheses"][1]["experiment"]
    assert promo_exp.get("formal_claims")
    assert promo_exp["formal_claims"][0]["template_id"] == (
        _mod._PROMOTE_FORMAL_TEMPLATE_ID
    )


def test_detect_promote_harness_failure_missing_run(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    (camp / "runs" / "c-control").mkdir(parents=True)
    # Control has metrics via suite loader fallback path — write smoke eval.
    smoke = camp / "runs" / "c-control" / "eval_smoke.json"
    smoke.write_text(
        __import__("json").dumps(
            {
                "suite": "smoke",
                "parse_rate": 1.0,
                "structural_similarity": 0.4,
                "meaningful_program_rate": 0.3,
                "latency_ms_p50": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    reasons = _mod.detect_promote_harness_failure(
        camp_dir=camp,
        control_id="c-control",
        candidate_id="c-promote",
        arm_exits={"c-control": 2, "c-promote": 1},
        cert_err="promote_cert_incomplete_metrics:ss=None parse=None",
    )
    assert any(r.startswith("harness_failure:") for r in reasons)
    assert any("missing_promote_run" in r or "promote_arm_exit" in r for r in reasons)


def test_resolve_promotion_harness_failure_not_model_reject(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    loop = "L"
    camp = root / "camp-hf"
    (camp / "runs" / "c-control").mkdir(parents=True)
    (camp / "runs" / "c-control" / "eval_smoke.json").write_text(
        __import__("json").dumps(
            {
                "suite": "smoke",
                "parse_rate": 1.0,
                "structural_similarity": 0.4,
                "latency_ms_p50": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    entry = {
        "entry_id": "champ-hf-1",
        "status": "promoting",
        "knobs_fingerprint": "fp-hf",
        "promote_attempts": 1,
        "knobs": {"grammar_completion_bounds": True},
    }
    path = _mod._champion_queue_path(root, loop)
    _mod._write_champion_queue(path, [entry])
    _mod.record_formal_preflight_status(
        camp,
        status="proved",
        template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID,
    )
    resolved = _mod._resolve_promotion_result(
        root=root,
        loop_id=loop,
        entry=entry,
        delivery={
            "positive": False,
            "control_id": "c-control",
            "candidate_id": "c-promote",
            "reasons": [
                "primary_metric_unavailable",
                "promote_cert_incomplete_metrics:ss=None parse=None",
            ],
        },
        campaign_id="camp-hf",
        cycle_index=3,
        camp_dir=camp,
        formal_preflight_status="proved",
        locked_expectations_sha256="c" * 64,
        arm_exits={"c-control": 2, "c-promote": 1},
        cert_err="promote_cert_incomplete_metrics:ss=None parse=None",
    )
    assert resolved is not None
    assert resolved["status"] == "harness_failure"
    assert resolved["status"] != "promotion_failed"
    assert resolved["status"] != "rejected"
    rows = _mod._load_champion_queue(path)
    assert rows[0]["status"] == "harness_failure"
    # Harness incompletes refund promote_attempts — not a model reject spend.
    assert int(rows[0].get("promote_attempts") or 0) == 0
    assert rows[0].get("last_harness_failure") is True
    head = _mod._queue_head_confirmed(rows)
    assert head is not None and head["entry_id"] == "champ-hf-1"
    ledger = (root / "loops" / loop / "learning_certificate_ledger.jsonl").read_text()
    assert "harness_failure" in ledger


def test_causal_cap_does_not_empty_multi_seed_open_bank() -> None:
    """Confirm CAP skip must not thrash-hard-die when multi-seed-open arms remain."""
    entries = [
        {
            "entry_id": "c1",
            "status": "rejected",
            "source_integration_commit": "tip1",
            "source_candidate_id": "x-literal-close",
            "knobs": {"ltr_tail_loss_weight": 2.0},
            "resolve_reasons": [
                "non_regression_fail:binder_reference_f1:1.0->0.0",
                "primary_metric_null_or_worse:smoke.structural_similarity",
            ],
        },
        {
            "entry_id": "c2",
            "status": "rejected",
            "source_integration_commit": "tip1",
            "source_candidate_id": "y-literal-close",
            "knobs": {"ltr_tail_loss_weight": 2.0},
            "resolve_reasons": [
                "non_regression_fail:binder_reference_f1:1.0->0.0",
            ],
        },
    ]
    hard = _mod._skip_arm_slugs(entries, integration_commit="tip1")
    assert "literal-close" in hard
    soft = _mod._skip_arm_slugs(
        entries, integration_commit="tip1", include_causal_cap=False
    )
    assert "literal-close" not in soft
    closed: set[str] = set()
    open_slugs = _mod._thrash_bank_open_slugs(closed)
    assert "literal-close" in open_slugs
    # Simulated cycle gate: relax CAP when hard skip empties open thrash.
    skip = hard | closed
    if open_slugs and not (open_slugs - skip):
        skip = soft | closed
    assert open_slugs - skip


def test_causal_cap_relaxation_preserves_operator_skip_slugs() -> None:
    """CAP relaxation must not reopen an operator-provided --skip-slugs slug.

    Mirrors run_cycle's soft_skip construction (line ~8777): soft_skip must
    union extra_skip_slugs alongside the queue-derived and recent_exhausted
    skips, or an explicitly skipped arm could be re-selected once the causal
    CAP relaxes.
    """
    entries = [
        {
            "entry_id": "c1",
            "status": "rejected",
            "source_integration_commit": "tip1",
            "source_candidate_id": "x-literal-close",
            "knobs": {"ltr_tail_loss_weight": 2.0},
            "resolve_reasons": [
                "non_regression_fail:binder_reference_f1:1.0->0.0",
                "primary_metric_null_or_worse:smoke.structural_similarity",
            ],
        },
    ]
    extra_skip_slugs = frozenset({"literal-close"})
    hard = _mod._skip_arm_slugs(entries, integration_commit="tip1") | extra_skip_slugs
    soft = (
        _mod._skip_arm_slugs(
            entries, integration_commit="tip1", include_causal_cap=False
        )
        | extra_skip_slugs
    )
    closed: set[str] = set()
    open_slugs = _mod._thrash_bank_open_slugs(closed)
    assert "literal-close" in open_slugs
    skip = hard | closed
    if open_slugs and not (open_slugs - skip):
        skip = soft | closed
    # Relaxation happened (soft_skip lacks the causal CAP entry) but the
    # operator's explicit --skip-slugs slug must still be excluded.
    assert "literal-close" in skip


def test_new_literal_close_successor_slugs() -> None:
    assert (
        _mod._arm_slug_from_knobs(
            {"ltr_tail_loss_weight": 2.0, "structure_token_loss_weight": 1.0}
        )
        == "literal-close-structure"
    )
    assert (
        _mod._arm_slug_from_knobs(
            {"ltr_tail_loss_weight": 2.0, "component_token_loss_weight": 1.0}
        )
        == "literal-close-component-token"
    )
    assert (
        _mod._arm_slug_from_knobs(
            {
                "semantic_contrast_loss_weight": 0.25,
                "structure_token_loss_weight": 1.0,
                "batch_size": 3,
            }
        )
        == "semantic-contrast-structure"
    )


def test_harness_incomplete_reasons_are_not_model_rejects() -> None:
    assert _mod._reason_is_harness_incomplete("harness_failure:missing_promote_run")
    assert _mod._reason_is_harness_incomplete(
        "measurement_incomplete:x:missing_scoreboard"
    )
    assert not _mod._reason_is_harness_incomplete(
        "primary_metric_null_or_worse:smoke.structural_similarity"
    )
    assert _mod._reasons_are_harness_incomplete_only(
        [
            "harness_failure:missing_promote_run",
            "measurement_incomplete:c-control:missing_scoreboard",
            "promote_attempts_exceeded:3>2",
        ]
    )
    assert not _mod._reasons_are_harness_incomplete_only(
        [
            "harness_failure:missing_promote_run",
            "primary_metric_null_or_worse:ss",
        ]
    )


def test_reopen_harness_blocked_champion_after_integration_change(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ar"
    entries = [
        {
            "entry_id": "champ-hf-park",
            "status": "harness_failure",
            "promote_attempts": 0,
            "last_harness_failure": True,
            "harness_failure_integration_commit": "aaa111",
            "resolve_reasons": [
                "harness_failure:missing_promote_run",
                "promote_harness_parked:incomplete_not_model_reject",
            ],
            "knobs": {"ltr_prefix_loss_weight": 1.0},
        },
        {
            "entry_id": "champ-model-fail",
            "status": "promotion_failed",
            "resolve_reasons": [
                "primary_metric_null_or_worse:smoke.structural_similarity"
            ],
            "harness_failure_integration_commit": "aaa111",
            "knobs": {"mask_pattern": "mixed"},
        },
    ]
    assert _mod._reopen_harness_blocked_champions(
        root, entries, integration_commit="bbb222"
    )
    by = {e["entry_id"]: e for e in entries}
    assert by["champ-hf-park"]["status"] == "confirmed"
    assert by["champ-hf-park"]["promote_attempts"] == 0
    assert (
        "harness_retry_after_integration_change"
        in by["champ-hf-park"]["resolve_reasons"]
    )
    # Model reject stays terminal.
    assert by["champ-model-fail"]["status"] == "promotion_failed"


def test_thrash_close_ignores_harness_incomplete_nulls(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    loop = "L"
    # Two complete harness-incomplete "nulls" must not close the arm.
    for i, seed in enumerate((7, 8)):
        camp = root / f"c-hf-{i}"
        camp.mkdir(parents=True)
        (camp / "campaign.json").write_text(
            __import__("json").dumps(
                {
                    "campaign_id": camp.name,
                    "loop_id": loop,
                    "cycle_index": i + 1,
                    "predecessor_campaign_id": None if i == 0 else f"c-hf-{i - 1}",
                }
            ),
            encoding="utf-8",
        )
        (camp / "sdlc_delivery.json").write_text(
            __import__("json").dumps(
                {
                    "candidate_id": f"{camp.name}-scaffold-prefix",
                    "control_id": f"{camp.name}-control",
                    "measurement_complete": True,
                    "positive": False,
                    "cycle_intent": "screening",
                    "harness_failure": True,
                    "reasons": ["harness_failure:missing_promote_run"],
                }
            ),
            encoding="utf-8",
        )
        (camp / "cycle_handoff.json").write_text(
            __import__("json").dumps(
                {
                    "loop_id": loop,
                    "reasons": ["harness_failure:missing_promote_run"],
                    "cycle_intent": "screening",
                }
            ),
            encoding="utf-8",
        )
        (camp / "matrix-proposal.json").write_text(
            __import__("json").dumps(
                {
                    "hypotheses": [
                        {
                            "experiment": {
                                "experiment_id": f"{camp.name}-scaffold-prefix",
                                "knobs": {
                                    "ltr_prefix_loss_weight": 1.0,
                                    "seed": seed,
                                },
                            }
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    closed = _mod._recent_completed_nonpositive_slugs(root, "c-hf-1", min_null_seeds=2)
    assert "scaffold-prefix" not in closed


def test_resolve_promotion_phase_a_alone_cannot_promote(tmp_path: Path) -> None:
    """Proof driver: Phase A quality-held without cert/formal is not promoted."""
    root = tmp_path / "autoresearch"
    loop_id = "loop-promo"
    path = _mod._champion_queue_path(root, loop_id)
    entry = {
        "schema": _mod._CHAMPION_QUEUE_SCHEMA,
        "entry_id": "champ-p",
        "status": "promoting",
        "knobs": {"grammar_completion_bounds": True},
        "knobs_fingerprint": "abc",
    }
    _mod._write_champion_queue(path, [entry])
    camp = root / "c-promo"
    camp.mkdir(parents=True)
    # Explicit missing formal status
    _mod.record_formal_preflight_status(
        camp,
        status="missing",
        template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID,
        reason="test",
    )
    fail = _mod._resolve_promotion_result(
        root=root,
        loop_id=loop_id,
        entry=entry,
        delivery={
            "positive": True,
            "reasons": [
                "quality_metric_win:meaningful_program_rate:0->0.5",
                "primary_metric_win:held_out.structural_similarity:0->0.1",
                "quality_held:parse=1.0 mpr=0.5",
            ],
        },
        campaign_id="c-promo",
        cycle_index=4,
        camp_dir=camp,
    )
    assert fail is not None
    assert fail["status"] == "promotion_failed"
    assert any(
        "formal_preflight" in r or "certificate" in r
        for r in (fail.get("resolve_reasons") or [])
    )


def _v2_cert(
    *, exp_sha: str, authority: str = "assumption_backed", relation: str = "in_band"
) -> dict:
    """Minimal v2 certificate mapping accepted by optimum_feedback."""
    return {
        "schema": "metric_certificate/v2",
        "checker": "leverproof-lean/v2",
        "verified": True,
        "assurance": "calculated_bands_and_observed_raw_samples",
        "run_id": "run-1",
        "evidence_sha256": "a" * 64,
        "feature_flags_sha256": "b" * 64,
        "metric_expectations_sha256": exp_sha,
        "selected_candidate": "candidate",
        "candidates": [{"id": "candidate"}],
        "assessments": [
            {
                "metric_id": "held_out_structural_similarity_pm",
                "authority": authority,
                "relation": relation,
            }
        ],
        "trusted_boundary": ["measurement"],
    }


def test_dispose_champion_promote_requires_locked_expectations_digest() -> None:
    """Missing locked digest must fail closed (never promote)."""
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=_v2_cert(exp_sha="c" * 64),
        locked_expectations_sha256=None,
        phase_a_positive=True,
        phase_a_quality_held=True,
    )
    assert d["status"] == "promotion_failed"
    assert any("locked_expectations" in r for r in d["reasons"])


def test_dispose_champion_promote_smoke_only_not_promoted() -> None:
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=None,
        locked_expectations_sha256="c" * 64,
        phase_a_positive=True,
        phase_a_quality_held=True,
    )
    assert d["status"] == "promotion_failed"
    assert any("certificate" in r for r in d["reasons"])


def test_dispose_champion_promote_missing_formal_not_promoted() -> None:
    exp_sha = "c" * 64
    d = _mod.dispose_champion_promote(
        formal_preflight_status="missing",
        certificate=_v2_cert(exp_sha=exp_sha),
        locked_expectations_sha256=exp_sha,
        phase_a_positive=True,
        phase_a_quality_held=True,
    )
    assert d["status"] == "promotion_failed"
    assert any("formal_preflight_unproved" in r for r in d["reasons"])


def test_dispose_champion_promote_invalid_cert_not_promoted() -> None:
    exp_sha = "c" * 64
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate={"schema": "metric_certificate/v1", "verified": True},
        locked_expectations_sha256=exp_sha,
        phase_a_positive=True,
        phase_a_quality_held=True,
    )
    assert d["status"] == "promotion_failed"
    assert any("v2" in r for r in d["reasons"])


def _held_out_win_metrics(
    *, control_ss: float = 0.10, candidate_ss: float = 0.20
) -> tuple[dict[str, float], dict[str, float]]:
    """Dual-arm metrics with held_out SS win above default min_effect 0.01."""
    control = {
        "structural_similarity": control_ss,
        "held_out.structural_similarity": control_ss,
        "parse_rate": 1.0,
        "held_out.parse_rate": 1.0,
    }
    candidate = {
        "structural_similarity": candidate_ss,
        "held_out.structural_similarity": candidate_ss,
        "parse_rate": 1.0,
        "held_out.parse_rate": 1.0,
    }
    return control, candidate


def test_dispose_champion_promote_in_band_v2_promotes() -> None:
    exp_sha = _mod.locked_promote_expectations_sha256()
    control, candidate = _held_out_win_metrics()
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=_v2_cert(exp_sha=exp_sha, relation="in_band"),
        locked_expectations_sha256=exp_sha,
        phase_a_positive=True,
        phase_a_quality_held=True,
        control_metrics=control,
        candidate_metrics=candidate,
    )
    assert d["status"] == "climb_accepted"
    assert d["cert_policy"] == "continue"
    assert d["promotion_primary_met"] is True
    assert d["primary_improvement"] is not None
    assert float(d["primary_improvement"]) > 0.01
    assert d["emit_five_lane_matrix"] is False
    assert any(r.startswith("promote_primary_win:") for r in d["reasons"])


def test_dispose_champion_promote_null_primary_not_promoted() -> None:
    """Cert continue + formal proved + null held_out delta must not promote."""
    exp_sha = _mod.locked_promote_expectations_sha256()
    control, candidate = _held_out_win_metrics(control_ss=0.25, candidate_ss=0.25)
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=_v2_cert(exp_sha=exp_sha, relation="in_band"),
        locked_expectations_sha256=exp_sha,
        phase_a_positive=False,
        control_metrics=control,
        candidate_metrics=candidate,
    )
    assert d["status"] == "promotion_failed"
    assert d["cert_policy"] == "continue"
    assert d["promotion_primary_met"] is False
    assert any("promote_primary_null_or_insufficient" in r for r in d["reasons"])


def test_dispose_champion_promote_missing_primary_metrics_not_promoted() -> None:
    exp_sha = _mod.locked_promote_expectations_sha256()
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=_v2_cert(exp_sha=exp_sha, relation="in_band"),
        locked_expectations_sha256=exp_sha,
        control_metrics={},
        candidate_metrics={},
    )
    assert d["status"] == "promotion_failed"
    assert d["cert_policy"] == "continue"
    assert any("promote_primary_metrics_missing" in r for r in d["reasons"])


def test_dispose_champion_promote_parse_regression_not_promoted() -> None:
    exp_sha = _mod.locked_promote_expectations_sha256()
    control = {
        "structural_similarity": 0.1,
        "held_out.structural_similarity": 0.1,
        "parse_rate": 1.0,
    }
    candidate = {
        "structural_similarity": 0.3,
        "held_out.structural_similarity": 0.3,
        "parse_rate": 0.5,
    }
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=_v2_cert(exp_sha=exp_sha, relation="in_band"),
        locked_expectations_sha256=exp_sha,
        control_metrics=control,
        candidate_metrics=candidate,
    )
    assert d["status"] == "promotion_failed"
    assert any(
        "invalid_grammar:" in r or "promote_parse_regression" in r for r in d["reasons"]
    )


def test_dispose_champion_promote_assumption_miss_five_lane() -> None:
    exp_sha = _mod.locked_promote_expectations_sha256()
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=_v2_cert(exp_sha=exp_sha, relation="above"),
        locked_expectations_sha256=exp_sha,
    )
    assert d["status"] == "promotion_failed"
    assert d["cert_policy"] == "block_promotion_and_diagnose"
    assert d["emit_five_lane_matrix"] is True
    assert set(d["diagnosis_lanes"]) >= set(_mod._FIVE_LANES)


def test_dispose_champion_promote_theorem_miss_no_promote() -> None:
    exp_sha = _mod.locked_promote_expectations_sha256()
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=_v2_cert(exp_sha=exp_sha, authority="theorem", relation="below"),
        locked_expectations_sha256=exp_sha,
    )
    assert d["status"] == "promotion_failed"
    assert d["cert_policy"] == "stop"
    assert d["emit_five_lane_matrix"] is False
    assert any("theorem" in r for r in d["reasons"])


def test_five_lane_successor_matrix_shape() -> None:
    matrix = _mod.build_five_lane_successor_matrix(
        campaign_id="c1",
        entry={"entry_id": "e1", "knobs_fingerprint": "fp"},
        breaches=[{"metric_id": "x", "authority": "assumption_backed"}],
        cert_policy="block_promotion_and_diagnose",
    )
    assert matrix["schema"] == "autotrain_five_lane_successor/v1"
    assert matrix["lanes"] == list(_mod._FIVE_LANES)
    assert len(matrix["hypotheses"]) == 5
    assert {h["lane"] for h in matrix["hypotheses"]} == set(_mod._FIVE_LANES)


def test_promote_manifest_locks_expectations_and_formal() -> None:
    exp = {
        "experiment_id": "c-promote",
        "hypothesis": "Confirmed champion levers hold under promotion primary.",
        "knobs": {
            "seed": 7,
            "eval_version": "e_test",
            "grammar_completion_bounds": True,
        },
        "formal_claims": [_mod.promote_formal_claim_dict()],
    }
    preflight_sha = "ab" * 32  # 64 hex
    man = _mod._manifest(
        "continuous-loop-c1",
        exp,
        "a" * 40,
        role="promotion",
        cycle_intent="promote",
        formal_preflight_sha256=preflight_sha,
    )
    assert man.metric_expectations_sha256 == _mod.locked_promote_expectations_sha256()
    assert man.metric_expectations_sha256 is not None
    kinds = {a.kind for a in man.artifact_requirements}
    assert "formal_preflight" in kinds
    assert man.formal_obligations
    assert man.formal_obligations[0].policy == "required"
    assert man.formal_obligations[0].template_id == _mod._PROMOTE_FORMAL_TEMPLATE_ID
    assert man.formal_obligations[0].preflight_sha256 == preflight_sha
    assert man.formal_obligations[0].preflight_sha256 != ("0" * 64)


def test_promote_manifest_without_preflight_sha_has_no_placeholder_obligation() -> None:
    """Never bind formal_obligations with zero-digest placeholders."""
    exp = {
        "experiment_id": "c-promote",
        "hypothesis": "Confirmed champion levers hold under promotion primary.",
        "knobs": {"seed": 7, "eval_version": "e_test"},
    }
    man = _mod._manifest(
        "continuous-loop-c1",
        exp,
        "a" * 40,
        role="promotion",
        cycle_intent="promote",
        formal_preflight_sha256=None,
    )
    assert man.metric_expectations_sha256 == _mod.locked_promote_expectations_sha256()
    assert man.formal_obligations == ()


def _write_run_eval(
    camp: Path,
    run_id: str,
    *,
    structural_similarity: float,
    parse_rate: float = 1.0,
) -> None:
    """Seed dual-arm eval artifacts so harness detection sees complete measurement."""
    run_dir = camp / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": 3,
        "document_n": 3,
        "completed_document_n": 3,
        "incomplete_document_n": 0,
        "decode_timeout_count": 0,
        "structural_similarity": structural_similarity,
        "parse_rate": parse_rate,
        "meaningful_program_rate": 0.5,
        "latency_ms_p50": 100.0,
    }
    (run_dir / "eval_held_out.json").write_text(
        __import__("json").dumps(payload), encoding="utf-8"
    )
    (run_dir / "eval_smoke.json").write_text(
        __import__("json").dumps(payload), encoding="utf-8"
    )


def _seed_complete_promotion_pair(
    camp: Path, *, prefix: str, seed: int
) -> tuple[str, str, dict[str, float], dict[str, float]]:
    control_id = f"{prefix}-control"
    candidate_id = f"{prefix}-promote"
    _write_run_eval(camp, control_id, structural_similarity=0.10)
    _write_run_eval(camp, candidate_id, structural_similarity=0.20)
    exp_sha = _mod.locked_promote_expectations_sha256()
    (camp / "metric-certificate.json").write_text(
        json.dumps(_v2_cert(exp_sha=exp_sha, relation="in_band")), encoding="utf-8"
    )
    _mod.record_formal_preflight_status(
        camp,
        status="proved",
        template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID,
    )
    manifests = camp / "manifests"
    manifests.mkdir(parents=True)
    for arm_id in (control_id, candidate_id):
        (manifests / f"{arm_id}.json").write_text(
            json.dumps({"experiment_id": arm_id, "seeds": [seed]}),
            encoding="utf-8",
        )
    control, candidate = _held_out_win_metrics()
    return control_id, candidate_id, control, candidate


def test_resolve_promotion_requires_two_content_bound_seed_pairs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-cert"
    path = _mod._champion_queue_path(root, loop_id)
    entry = {
        "schema": _mod._CHAMPION_QUEUE_SCHEMA,
        "entry_id": "champ-ok",
        "status": "promoting",
        "knobs": {"grammar_completion_bounds": True},
        "knobs_fingerprint": "fpok",
    }
    _mod._write_champion_queue(path, [entry])

    statuses: list[str] = []
    for cycle_index, seed in ((9, 109), (10, 110)):
        campaign_id = f"c-cert-{cycle_index}"
        camp = root / campaign_id
        control_id, candidate_id, control, candidate = _seed_complete_promotion_pair(
            camp, prefix=campaign_id, seed=seed
        )
        delivery = {
            "positive": True,
            "measurement_complete": True,
            "reasons": ["quality_held:parse=1.0 mpr=1.0"],
            "control_id": control_id,
            "candidate_id": candidate_id,
            "control_metrics": control,
            "candidate_metrics": candidate,
            "arm_seed": seed,
            "arm_order": _mod._counterbalanced_arm_order(
                control_id,
                candidate_id,
                cycle_index=cycle_index,
                seed=seed,
                promotion_replicate_index=len(statuses),
            ),
        }
        (camp / "sdlc_delivery.json").write_text(json.dumps(delivery), encoding="utf-8")
        resolved = _mod._resolve_promotion_result(
            root=root,
            loop_id=loop_id,
            entry=_mod._load_champion_queue(path)[0],
            delivery=delivery,
            campaign_id=campaign_id,
            cycle_index=cycle_index,
            camp_dir=camp,
            formal_preflight_status="proved",
            arm_exits={control_id: 0, candidate_id: 0},
        )
        assert resolved is not None
        statuses.append(str(resolved["status"]))

    assert statuses == ["promotion_inconclusive", "climb_accepted"]
    replicate_ledger = _mod._promotion_replicate_ledger_path(root, loop_id)
    rows, error = _mod._record_promotion_replicate(
        root=root,
        loop_id=loop_id,
        entry=_mod._load_champion_queue(path)[0],
        campaign_id=campaign_id,
        cycle_index=cycle_index,
        camp_dir=camp,
        delivery=delivery,
        arm_exits={control_id: 0, candidate_id: 0},
    )
    assert error is None
    assert len(rows) == 2
    replicates = [
        json.loads(line)
        for line in replicate_ledger.read_text(encoding="utf-8").splitlines()
    ]
    assert len(replicates) == 2
    for replicate in replicates:
        seed = replicate["seed"]
        camp = root / replicate["campaign_id"]
        for arm_id in (replicate["control_id"], replicate["candidate_id"]):
            manifest = json.loads((camp / "manifests" / f"{arm_id}.json").read_text())
            assert manifest["seeds"] == [seed]
    ledger = root / "loops" / loop_id / "learning_certificate_ledger.jsonl"
    assert ledger.is_file()
    line = ledger.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert "climb_accepted" in line
    assert "promote_primary_win" in line or "primary_improvement" in line


def test_tampered_promotion_replicate_does_not_satisfy_seed_floor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-tampered"
    entry = {"entry_id": "champ", "knobs_fingerprint": "fingerprint"}
    ledger = _mod._promotion_replicate_ledger_path(root, loop_id)
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "schema": _mod._PROMOTION_REPLICATE_SCHEMA,
                "entry_id": "champ",
                "knobs_fingerprint": "fingerprint",
                "seed": 101,
                "content_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _mod._verified_promotion_replicates(root, loop_id, entry) == []


@pytest.mark.parametrize("mutated", ["manifest", "certificate", "delivery"])
def test_mutated_promotion_evidence_invalidates_replicate(
    tmp_path: Path, mutated: str
) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-mutated"
    campaign_id = "campaign-1"
    camp = root / campaign_id
    entry = {"entry_id": "champ", "knobs_fingerprint": "fingerprint"}
    control_id, candidate_id, control, candidate = _seed_complete_promotion_pair(
        camp, prefix=campaign_id, seed=101
    )
    delivery = {
        "measurement_complete": True,
        "control_id": control_id,
        "candidate_id": candidate_id,
        "control_metrics": control,
        "candidate_metrics": candidate,
        "arm_seed": 101,
        "arm_order": [control_id, candidate_id],
    }
    delivery_path = camp / "sdlc_delivery.json"
    delivery_path.write_text(json.dumps(delivery), encoding="utf-8")
    rows, error = _mod._record_promotion_replicate(
        root=root,
        loop_id=loop_id,
        entry=entry,
        campaign_id=campaign_id,
        cycle_index=1,
        camp_dir=camp,
        delivery=delivery,
        arm_exits={control_id: 0, candidate_id: 0},
    )
    assert error is None and len(rows) == 1

    if mutated == "manifest":
        path = camp / "manifests" / f"{candidate_id}.json"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif mutated == "certificate":
        path = camp / "metric-certificate.json"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    else:
        delivery["arm_seed"] = 102
        delivery_path.write_text(json.dumps(delivery), encoding="utf-8")

    assert _mod._verified_promotion_replicates(root, loop_id, entry) == []


def test_resolve_promotion_null_primary_not_promoted(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-null-primary"
    camp = root / "c-null"
    camp.mkdir(parents=True)
    _write_run_eval(camp, "ctrl", structural_similarity=0.40)
    _write_run_eval(camp, "cand", structural_similarity=0.40)
    exp_sha = _mod.locked_promote_expectations_sha256()
    cert = _v2_cert(exp_sha=exp_sha, relation="in_band")
    (camp / "metric-certificate.json").write_text(
        __import__("json").dumps(cert), encoding="utf-8"
    )
    _mod.record_formal_preflight_status(
        camp, status="proved", template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID
    )
    path = _mod._champion_queue_path(root, loop_id)
    entry = {
        "schema": _mod._CHAMPION_QUEUE_SCHEMA,
        "entry_id": "champ-null",
        "status": "promoting",
        "knobs": {"grammar_completion_bounds": True},
        "knobs_fingerprint": "fpnull",
    }
    _mod._write_champion_queue(path, [entry])
    control, candidate = _held_out_win_metrics(control_ss=0.4, candidate_ss=0.4)
    resolved = _mod._resolve_promotion_result(
        root=root,
        loop_id=loop_id,
        entry=entry,
        delivery={
            "positive": False,
            "reasons": ["primary_metric_null_or_worse:held_out.structural_similarity"],
            "control_metrics": control,
            "candidate_metrics": candidate,
            "control_id": "ctrl",
            "candidate_id": "cand",
        },
        campaign_id="c-null",
        cycle_index=11,
        camp_dir=camp,
        formal_preflight_status="proved",
        arm_exits={"ctrl": 0, "cand": 0},
    )
    assert resolved is not None
    assert resolved["status"] == "promotion_failed"
    reasons = resolved.get("resolve_reasons") or []
    assert any("promote_primary_null_or_insufficient" in str(r) for r in reasons)


def test_resolve_promotion_assumption_miss_writes_five_lane(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-5lane"
    camp = root / "c-5lane"
    camp.mkdir(parents=True)
    exp_sha = _mod.locked_promote_expectations_sha256()
    cert = _v2_cert(exp_sha=exp_sha, relation="above")
    (camp / "metric-certificate.json").write_text(
        __import__("json").dumps(cert), encoding="utf-8"
    )
    _mod.record_formal_preflight_status(
        camp, status="proved", template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID
    )
    path = _mod._champion_queue_path(root, loop_id)
    entry = {
        "schema": _mod._CHAMPION_QUEUE_SCHEMA,
        "entry_id": "champ-miss",
        "status": "promoting",
        "knobs": {"compact_active_canvas": True},
        "knobs_fingerprint": "fpmiss",
    }
    _mod._write_champion_queue(path, [entry])
    resolved = _mod._resolve_promotion_result(
        root=root,
        loop_id=loop_id,
        entry=entry,
        delivery={"positive": False, "reasons": []},
        campaign_id="c-5lane",
        cycle_index=10,
        camp_dir=camp,
        formal_preflight_status="proved",
    )
    assert resolved is not None
    assert resolved["status"] == "promotion_failed"
    five = camp / "five_lane_successor_matrix.json"
    assert five.is_file()
    data = __import__("json").loads(five.read_text(encoding="utf-8"))
    assert data["schema"] == "autotrain_five_lane_successor/v1"
    assert len(data["lanes"]) == 5


def test_resolve_promotion_unreadable_locked_expectations_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-no-lock"
    camp = root / "c-nolock"
    camp.mkdir(parents=True)
    path = _mod._champion_queue_path(root, loop_id)
    entry = {
        "schema": _mod._CHAMPION_QUEUE_SCHEMA,
        "entry_id": "champ-nolock",
        "status": "promoting",
        "knobs": {"grammar_completion_bounds": True},
        "knobs_fingerprint": "fpnl",
    }
    _mod._write_champion_queue(path, [entry])
    _mod.record_formal_preflight_status(
        camp, status="proved", template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID
    )
    exp_sha = "d" * 64
    (camp / "metric-certificate.json").write_text(
        __import__("json").dumps(_v2_cert(exp_sha=exp_sha)), encoding="utf-8"
    )

    def _boom() -> str:
        raise OSError("expectations missing")

    monkeypatch.setattr(_mod, "locked_promote_expectations_sha256", _boom)
    resolved = _mod._resolve_promotion_result(
        root=root,
        loop_id=loop_id,
        entry=entry,
        delivery={"positive": True, "reasons": ["quality_held:parse=1 mpr=1"]},
        campaign_id="c-nolock",
        cycle_index=1,
        camp_dir=camp,
    )
    assert resolved is not None
    assert resolved["status"] == "promotion_failed"
    assert any(
        "locked_expectations" in r for r in (resolved.get("resolve_reasons") or [])
    )


def test_promote_formal_claims_match_obligations_for_execute_binding() -> None:
    """Promote experiment formal_claims must match campaign formal_obligations ids."""
    from slm_training.autoresearch.formal import formal_obligation_id
    from slm_training.autoresearch.schemas import FormalClaimV1

    campaign_id = "continuous-loop-c1"
    experiment_id = "c20260731-c1-promote"
    claim = FormalClaimV1(**_mod.promote_formal_claim_dict())
    exp = {
        "experiment_id": experiment_id,
        "hypothesis": "Confirmed champion levers hold under promotion primary.",
        "knobs": {"seed": 7, "eval_version": "e_test"},
        "formal_claims": [_mod.promote_formal_claim_dict()],
    }
    preflight_sha = "cd" * 32
    man = _mod._manifest(
        campaign_id,
        exp,
        "a" * 40,
        role="promotion",
        cycle_intent="promote",
        formal_preflight_sha256=preflight_sha,
    )
    claim_ids = {formal_obligation_id(campaign_id, experiment_id, claim)}
    obligation_ids = {o.obligation_id for o in man.formal_obligations}
    assert claim_ids == obligation_ids
    assert man.formal_obligations[0].preflight_sha256 == preflight_sha


def test_ensure_promote_formal_preflight_revalidates_recorded_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    from slm_training.lineage.records import canonical_json

    camp = tmp_path / "camp"
    camp.mkdir()
    payload = {
        "schema_version": "FormalPreflightV1",
        "campaign_id": "c1",
        "experiment_id": "e1",
        "obligation_id": "formal-" + ("a" * 16),
        "template_id": _mod._PROMOTE_FORMAL_TEMPLATE_ID,
        "template_version": "v1",
        "claim": _mod.promote_formal_claim_dict()["claim"],
        "policy": "required",
        "status": "proved",
        "evidence_scope": "universal",
        "theorem": "OpenUIProofs.Metrics.structural_similarity_mono",
        "proof_target": "OpenUIProofs.Metrics",
        "assumptions": [],
        "open_assumptions": [],
        "source_digests": {"x": "a" * 64},
        "proof_sha256": "b" * 64,
        "lean_version": "lean",
        "mathlib_version": "mathlib",
        "build_output_sha256": "c" * 64,
        "duration_seconds": 0.1,
        "created_at": "2026-07-31T00:00:00Z",
    }
    content_sha = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    art = camp / "artifacts" / "formal_preflights"
    art.mkdir(parents=True)
    (art / f"{content_sha}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _mod.record_formal_preflight_status(
        camp, status="proved", template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID
    )
    st = json.loads((camp / "formal_preflight_status.json").read_text(encoding="utf-8"))
    st["preflight_sha256"] = content_sha
    (camp / "formal_preflight_status.json").write_text(
        json.dumps(st, indent=2) + "\n", encoding="utf-8"
    )
    validated = SimpleNamespace(status="proved")
    monkeypatch.setattr(
        "slm_training.autoresearch.formal.validate_formal_preflight_artifact",
        lambda *args, **kwargs: validated,
    )
    status, sha = _mod.ensure_promote_formal_preflight(
        camp_dir=camp, campaign_id="c1", experiment_id="e1", run_lean=False
    )
    assert status == "proved"
    assert sha == content_sha
    assert (art / f"{sha}.json").is_file()
    recorded = json.loads(
        (camp / "formal_preflight_status.json").read_text(encoding="utf-8")
    )
    assert recorded["binding_validated_sha256"] == content_sha
    assert _mod._formal_preflight_status(camp) == "proved"


def test_ensure_promote_formal_preflight_rejects_stale_recorded_artifact(
    tmp_path: Path,
) -> None:
    camp = tmp_path / "camp"
    camp.mkdir()
    sha = "a" * 64
    _mod.record_formal_preflight_status(
        camp, status="proved", template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID
    )
    status_path = camp / "formal_preflight_status.json"
    recorded = json.loads(status_path.read_text(encoding="utf-8"))
    recorded["preflight_sha256"] = sha
    status_path.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")

    status, returned_sha = _mod.ensure_promote_formal_preflight(
        camp_dir=camp, campaign_id="c1", experiment_id="e1", run_lean=False
    )

    assert status == "unknown"
    assert returned_sha is None
    assert _mod._formal_preflight_status(camp) == "unknown"
    recorded = json.loads(status_path.read_text(encoding="utf-8"))
    assert recorded["reason"].startswith("cached_formal_preflight_invalid:")


def test_formal_preflight_status_rejects_unvalidated_proved_sidecar(
    tmp_path: Path,
) -> None:
    camp = tmp_path / "camp"
    _mod.record_formal_preflight_status(
        camp, status="proved", template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID
    )
    status_path = camp / "formal_preflight_status.json"
    recorded = json.loads(status_path.read_text(encoding="utf-8"))
    recorded["preflight_sha256"] = "a" * 64
    status_path.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")

    assert _mod._formal_preflight_status(camp) is None


def test_skip_arm_slugs_only_while_open() -> None:
    entries = [
        {
            "status": "promotion_failed",
            "knobs": {"grammar_completion_bounds": True},
            "source_candidate_id": "c-bounds",
        },
        {
            "status": "rejected",
            "knobs": {"grammar_completion_bounds": True, "compact_active_canvas": True},
            "source_candidate_id": "c-both",
        },
        {
            "status": "queued",
            "knobs": {"compact_active_canvas": True},
            "source_candidate_id": "c-canvas",
        },
    ]
    skip = _mod._skip_arm_slugs(entries)
    assert "bounds" not in skip
    assert "both" not in skip
    assert "canvas" in skip


def test_is_champion_lever_includes_training_and_decode_arms() -> None:
    assert _mod._is_champion_lever(
        {"grammar_completion_bounds": True}, candidate_id="x-bounds"
    )
    assert _mod._is_champion_lever({"batch_size": 1}, candidate_id="x-batch1")
    assert _mod._is_champion_lever(
        {"component_edge_loss_weight": 1.0}, candidate_id="x-component-edge"
    )
    assert _mod._is_champion_lever(
        {
            "grammar_completion_bounds": False,
            "compact_active_canvas": False,
            "steps": 160,
        },
        candidate_id="c20260731-c1-steps",
    )
    assert not _mod._is_champion_lever(
        {
            "grammar_completion_bounds": False,
            "compact_active_canvas": False,
            "batch_size": 2,
        },
        candidate_id="c-control",
    )


def _seed_promote_runs(camp: Path) -> None:
    for rid, lat, pr, mpr in (
        ("c-control", 12000.0, 1.0, 0.33),
        ("c-promote", 9000.0, 1.0, 0.5),
    ):
        d = camp / "runs" / rid
        d.mkdir(parents=True)
        (d / "eval_smoke.json").write_text(
            __import__("json").dumps(
                {
                    "latency_ms_p50": lat,
                    "parse_rate": pr,
                    "meaningful_program_rate": mpr,
                    "structural_similarity": mpr,
                    "details": [
                        {
                            "incomplete": False,
                            "parse_ok": pr == 1.0,
                            "structural_similarity": score,
                        }
                        for score in (mpr - 0.1, mpr, mpr + 0.1)
                    ],
                }
            ),
            encoding="utf-8",
        )
        (camp / "artifacts" / "experiments").mkdir(parents=True, exist_ok=True)
        (camp / "artifacts" / "experiments" / f"{rid}.json").write_text(
            __import__("json").dumps(
                {
                    "experiment_id": rid,
                    "knobs": {
                        "grammar_completion_bounds": rid != "c-control",
                        "compact_active_canvas": False,
                        "batch_size": 2,
                        "steps": 80,
                    },
                }
            ),
            encoding="utf-8",
        )


def _raw_resource_candidates() -> list[dict[str, object]]:
    """Explicit fixture samples; production must receive canonical measurements."""

    return [
        {
            "id": arm_id,
            "hardware": "cpu",
            "lever_snapshot_sha256": digest,
            "cold_ns": [cold_ns],
            "warm_ns": [warm_ns, warm_ns + 100],
            "input_units": [3],
            "passes": [8],
            "energy_uj": [energy_uj],
            "cost_micro_usd": [cost],
            "successes": 3,
            "quality_failures": 0,
            "trainable_params": 1_608_962,
        }
        for arm_id, digest, cold_ns, warm_ns, energy_uj, cost in (
            ("control", "a" * 64, 12_000_000_000, 11_000_000_000, 5000, 20),
            ("candidate", "b" * 64, 10_000_000_000, 9_000_000_000, 4500, 18),
        )
    ]


def test_export_promote_metric_certificate_writes_v2(tmp_path: Path) -> None:
    """Real LeverProof path when checker is built; skip on CI without Lean bin."""
    import pytest
    from slm_training.harnesses.experiments.verified_metrics import IN_REPO_CHECKER

    if not IN_REPO_CHECKER.is_file():
        pytest.skip("leverproof-lean binary not built in this environment")

    camp = tmp_path / "camp"
    _seed_promote_runs(camp)
    path, err = _mod.export_promote_metric_certificate(
        camp_dir=camp,
        campaign_id="camp1",
        control_id="c-control",
        candidate_id="c-promote",
        delivery={"reasons": []},
        raw_resource_candidates=_raw_resource_candidates(),
    )
    assert err is None, err
    assert path is not None and path.is_file()
    cert = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert cert["schema"] == "metric_certificate/v2"
    observations = json.loads(
        (camp / "promote" / "metric-observations.json").read_text(encoding="utf-8")
    )
    assert observations["metrics"]["held_out_structural_similarity_pm"] == [
        400,
        500,
        600,
    ]
    assert observations["metrics"]["parse_rate_pm"] == [1000, 1000, 1000]
    from slm_training.harnesses.experiments.verified_metrics import optimum_feedback

    fb = optimum_feedback(cert)
    assert fb["policy"] == "continue"
    control, candidate = _held_out_win_metrics(control_ss=0.25, candidate_ss=0.50)
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=cert,
        locked_expectations_sha256=_mod.locked_promote_expectations_sha256(),
        phase_a_positive=True,
        phase_a_quality_held=True,
        control_metrics=control,
        candidate_metrics=candidate,
    )
    assert d["status"] == "climb_accepted"


def test_export_promote_metric_certificate_fail_closed_without_metrics(
    tmp_path: Path,
) -> None:
    camp = tmp_path / "camp"
    (camp / "runs" / "c-control").mkdir(parents=True)
    (camp / "runs" / "c-promote").mkdir(parents=True)
    path, err = _mod.export_promote_metric_certificate(
        camp_dir=camp,
        campaign_id="camp1",
        control_id="c-control",
        candidate_id="c-promote",
        delivery={},
        raw_resource_candidates=_raw_resource_candidates(),
    )
    assert path is None
    assert err is not None
    assert (
        "incomplete_metrics" in err
        or "checker_missing" in err
        or "raw_metric_observations_missing" in err
        or "raw_resource_evidence_missing" in err
    )


def test_export_promote_metric_certificate_rejects_synthetic_resource_defaults(
    tmp_path: Path,
) -> None:
    camp = tmp_path / "camp"
    _seed_promote_runs(camp)

    path, err = _mod.export_promote_metric_certificate(
        camp_dir=camp,
        campaign_id="camp1",
        control_id="c-control",
        candidate_id="c-promote",
        delivery={},
    )

    assert path is None
    assert err == "promote_evidence_build_failed:raw_resource_evidence_missing"
    assert not (camp / "promote" / "metric-evidence.json").exists()


def test_promote_certificate_checker_uses_bounded_stage_and_120s_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.harnesses.experiments import verified_metrics

    camp = tmp_path / "camp"
    root = tmp_path / "ar"
    checker = tmp_path / "leverproof-lean"
    checker.write_text("checker", encoding="utf-8")
    _seed_promote_runs(camp)
    monkeypatch.setattr(verified_metrics, "IN_REPO_CHECKER", checker)
    monkeypatch.setattr(_mod.time, "monotonic", lambda: 100.0)
    captured: dict[str, object] = {}

    def bounded(cmd, **kwargs):
        captured.update({"cmd": cmd, **kwargs})
        return _mod.BoundedProcessResult(
            command=tuple(cmd),
            outcome=_mod.ProcessOutcome.COMPLETED,
            returncode=0,
            stdout='{"schema":"metric_certificate/v2","selected_candidate":"candidate"}',
            stderr="",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(_mod, "_stage_command", bounded)
    path, err = _mod.export_promote_metric_certificate(
        camp_dir=camp,
        campaign_id="camp1",
        control_id="c-control",
        candidate_id="c-promote",
        delivery={},
        deadline=500.0,
        root=root,
        loop_id="loop-x",
        raw_resource_candidates=_raw_resource_candidates(),
    )

    assert err is None
    assert path is not None
    assert captured["deadline"] == 220.0
    assert captured["root"] == root
    assert captured["loop_id"] == "loop-x"
    assert captured["stage"] == "promotion-certificate"


def test_promote_certificate_checker_timeout_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.harnesses.experiments import verified_metrics

    camp = tmp_path / "camp"
    checker = tmp_path / "leverproof-lean"
    checker.write_text("checker", encoding="utf-8")
    _seed_promote_runs(camp)
    monkeypatch.setattr(verified_metrics, "IN_REPO_CHECKER", checker)

    def timed_out(cmd, **kwargs):
        del kwargs
        return _mod.BoundedProcessResult(
            command=tuple(cmd),
            outcome=_mod.ProcessOutcome.KILLED,
            returncode=-9,
            stdout="",
            stderr="",
            duration_seconds=120.0,
            timed_out=True,
            interrupted=True,
            killed=True,
        )

    monkeypatch.setattr(_mod, "_stage_command", timed_out)
    path, err = _mod.export_promote_metric_certificate(
        camp_dir=camp,
        campaign_id="camp1",
        control_id="c-control",
        candidate_id="c-promote",
        delivery={},
        raw_resource_candidates=_raw_resource_candidates(),
    )

    assert path is None
    assert err == "promote_certify_failed:checker timed out within 120s cap"
    assert not (camp / "metric-certificate.json").exists()


def test_rate_to_pm_helper() -> None:
    assert _mod._rate_to_pm(0.5) == 500
    assert _mod._rate_to_pm(1.0) == 1000
    assert _mod._rate_to_pm(None) is None


def test_enqueue_champion_accepts_steps_arm(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    loop = "L"
    camp = root / "c1"
    exp_dir = camp / "artifacts" / "experiments"
    exp_dir.mkdir(parents=True)
    (exp_dir / "c1-steps.json").write_text(
        __import__("json").dumps(
            {
                "experiment_id": "c1-steps",
                "knobs": {
                    "grammar_completion_bounds": False,
                    "compact_active_canvas": False,
                    "batch_size": 2,
                    "steps": 160,
                    "train_version": "wf_smoke_v2",
                },
            }
        ),
        encoding="utf-8",
    )
    delivery = {
        "positive": True,
        "campaign_id": "c1",
        "cycle_index": 9,
        "cycle_role": "screening",
        "candidate_id": "c1-steps",
        "control_id": "c1-control",
        "control_metrics": {"latency_ms_p50": 10000, "meaningful_program_rate": 0.33},
        "candidate_metrics": {"latency_ms_p50": 8000, "meaningful_program_rate": 0.33},
        "reasons": [
            "primary_metric_win:smoke.latency_ms_p50:10000->8000",
            "quality_held:parse=1.0 mpr=0.33",
        ],
    }
    entry = _mod._enqueue_champion(
        root=root, loop_id=loop, delivery=delivery, camp_dir=camp
    )
    assert entry is not None
    assert entry["status"] == "queued"


def _write_eval(path: Path, *, suite: str, **metrics: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"suite": suite, "n": 5 if suite == "held_out" else 3, **metrics}
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")


def _write_complete_scoreboard(run: Path, *suite_names: str) -> None:
    suites = {}
    for suite in suite_names:
        document_n = 5 if suite == "held_out" else 3
        suites[suite] = {
            "suite": suite,
            "n": document_n,
            "document_n": document_n,
            "completed_document_n": document_n,
            "incomplete_document_n": 0,
            "decode_timeout_document_count": 0,
        }
    (run / "scoreboard.json").write_text(
        json.dumps({"suites": suites}), encoding="utf-8"
    )


def test_run_metrics_loads_held_out_when_preferred(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    run = camp / "runs" / "arm-a"
    _write_eval(
        run / "eval_smoke.json",
        suite="smoke",
        parse_rate=1.0,
        meaningful_program_rate=1.0,
        structural_similarity=0.2,
        latency_ms_p50=1000.0,
    )
    _write_eval(
        run / "eval_held_out.json",
        suite="held_out",
        parse_rate=1.0,
        meaningful_program_rate=0.2,
        structural_similarity=0.42,
        latency_ms_p50=2000.0,
    )
    smoke_only = _mod._run_metrics(camp, "arm-a", prefer_held_out=False)
    assert smoke_only["structural_similarity"] == 0.2
    assert smoke_only["held_out.structural_similarity"] == 0.42
    held = _mod._run_metrics(camp, "arm-a", prefer_held_out=True)
    assert held["structural_similarity"] == 0.42
    assert held["held_out.structural_similarity"] == 0.42
    assert held["smoke.structural_similarity"] == 0.2


def test_classify_positive_rejects_c1731_efficiency_jitter(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    for arm, latency in (("c-control", 3453.06), ("c-component-plan", 3430.55)):
        _write_eval(
            camp / "runs" / arm / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            binder_reference_f1=0.6333333333333333,
            meaningful_program_rate=0.3333333333333333,
            structural_similarity=0.41973333333333335,
            latency_ms_p50=latency,
        )
    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.structural_similarity",
        control_id="c-control",
        candidate_id="c-component-plan",
        role="screening",
    )
    assert result["positive"] is False
    assert any(
        reason.startswith("efficiency_win_rejected_min_effect:")
        for reason in result["reasons"]
    )


def test_classify_positive_rejects_c1819_quality_regression(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    rows = (
        (
            "c-control",
            3772.85,
            0.6666666666666666,
            0.40443333333333337,
            0.9523809523809524,
        ),
        (
            "c-candidate",
            1074.57,
            0.3333333333333333,
            0.17416666666666666,
            0.6333333333333333,
        ),
    )
    for arm, latency, mpr, similarity, binder_f1 in rows:
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            binder_reference_f1=binder_f1,
            meaningful_program_rate=mpr,
            structural_similarity=similarity,
            latency_ms_p50=latency,
        )
        _write_complete_scoreboard(run, "smoke")
        scoreboard = json.loads((run / "scoreboard.json").read_text(encoding="utf-8"))
        # Candidate is the quality regression: higher NLL (worse) + lower SS.
        scoreboard["suites"]["smoke"]["eval_nll"] = 2.0 if arm == "c-control" else 4.0
        (run / "scoreboard.json").write_text(json.dumps(scoreboard), encoding="utf-8")

    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.structural_similarity",
        control_id="c-control",
        candidate_id="c-candidate",
        role="screening",
    )

    assert result["positive"] is False
    assert result["stack_layer"] is False
    assert any(
        reason.startswith("efficiency_win_rejected_mpr_regression:")
        for reason in result["reasons"]
    )
    assert any(
        "primary_metric_null_or_worse:smoke.eval_nll" in reason
        or reason.startswith("non_regression_fail:binder_reference_f1:")
        for reason in result["reasons"]
    )


def test_classify_positive_rejects_quality_win_with_unbounded_latency_cost(
    tmp_path: Path,
) -> None:
    camp = tmp_path / "camp"
    for arm, similarity, latency in (
        ("c-control", 0.13526666666666667, 1105.9),
        ("c-steps", 0.41973333333333335, 2810.05),
    ):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            binder_reference_f1=0.6333333333333333,
            meaningful_program_rate=0.3333333333333333,
            structural_similarity=similarity,
            latency_ms_p50=latency,
        )
        _write_complete_scoreboard(run, "smoke")
        scoreboard = json.loads((run / "scoreboard.json").read_text(encoding="utf-8"))
        scoreboard["suites"]["smoke"]["eval_nll"] = 4.0 if arm == "c-control" else 2.0
        (run / "scoreboard.json").write_text(json.dumps(scoreboard), encoding="utf-8")

    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.structural_similarity",
        control_id="c-control",
        candidate_id="c-steps",
        role="screening",
    )

    assert result["positive"] is False
    assert result["stack_layer"] is False
    assert any(
        reason.startswith("primary_quality_win_rejected_latency_budget:")
        for reason in result["reasons"]
    )


def test_open_champion_is_revalidated_under_current_policy(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-1"
    for arm, similarity, latency in (
        ("c-control", 0.13526666666666667, 1105.9),
        ("c-steps", 0.41973333333333335, 2810.05),
    ):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=0.3333333333333333,
            structural_similarity=similarity,
            latency_ms_p50=latency,
        )
        _write_complete_scoreboard(run, "smoke")
        scoreboard = json.loads((run / "scoreboard.json").read_text(encoding="utf-8"))
        scoreboard["suites"]["smoke"]["eval_nll"] = 4.0 if arm == "c-control" else 2.0
        (run / "scoreboard.json").write_text(json.dumps(scoreboard), encoding="utf-8")
    (camp / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "cycle_role": "screening",
                "primary_metric": "smoke.structural_similarity",
            }
        )
    )
    entries = [
        {
            "entry_id": "champ-1",
            "status": "queued",
            "source_campaign_id": camp.name,
            "source_control_id": "c-control",
            "source_candidate_id": "c-steps",
            "source_role": "screening",
        }
    ]

    assert _mod._revalidate_open_champion_entries(root, entries) is True
    assert entries[0]["status"] == "rejected"
    assert (
        "source_reclassified_nonpositive_under_current_policy"
        in entries[0]["resolve_reasons"]
    )


def test_promote_authority_digest_changes_with_harness_version() -> None:
    from slm_training.autoresearch.climb_policy import promote_authority_sha256

    a = promote_authority_sha256(
        climb_policy_sha256="p" * 64,
        locked_expectations_sha256="e" * 64,
        harness_component_version="v176",
    )
    b = promote_authority_sha256(
        climb_policy_sha256="p" * 64,
        locked_expectations_sha256="e" * 64,
        harness_component_version="v177",
    )
    assert a != b
    assert len(a) == 64


def test_recertify_promoted_requeues_when_evidence_missing(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-recert"
    (root / "loops" / loop_id).mkdir(parents=True)
    entries = [
        {
            "entry_id": "champ-legacy",
            "status": "promoted",
            "knobs_fingerprint": "abc",
            # no promote_authority_sha256, no promotion campaign
            "resolve_reasons": ["cert_policy:continue", "phase_a_positive"],
        }
    ]
    assert _mod._recertify_promoted_champion_entries(root, loop_id, entries) is True
    assert entries[0]["status"] == "confirmed"
    assert entries[0].get("recert_required") is True
    assert any(
        "recert_required" in str(r) for r in entries[0].get("resolve_reasons") or []
    )
    audit = (root / "loops" / loop_id / "historical_reclassification.jsonl").read_text(
        encoding="utf-8"
    )
    assert "promote_authority_recert" in audit
    assert "requeue" in audit


def test_recertify_promoted_fails_null_primary_under_current_rules(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-recert-fail"
    camp = root / "promote-camp"
    camp.mkdir(parents=True)
    exp_sha = _mod.locked_promote_expectations_sha256()
    (camp / "metric-certificate.json").write_text(
        json.dumps(_v2_cert(exp_sha=exp_sha, relation="in_band")), encoding="utf-8"
    )
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "measurement_complete": True,
                "positive": True,
                "control_id": "c-control",
                "candidate_id": "c-promote",
                "control_metrics": {
                    "structural_similarity": 0.25,
                    "held_out.structural_similarity": 0.25,
                    "parse_rate": 1.0,
                },
                "candidate_metrics": {
                    "structural_similarity": 0.25,
                    "held_out.structural_similarity": 0.25,
                    "parse_rate": 1.0,
                },
                "reasons": ["cert_policy:continue"],
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "entry_id": "champ-vacuous",
            "status": "promoted",
            "promotion_campaign_id": camp.name,
            "formal_preflight_status": "proved",
            "resolve_reasons": ["cert_policy:continue", "phase_a_positive"],
        }
    ]
    assert _mod._recertify_promoted_champion_entries(root, loop_id, entries) is True
    assert entries[0]["status"] == "promotion_failed"
    assert any(
        "recert_under_current_policy" in str(r)
        for r in entries[0].get("resolve_reasons") or []
    )


def test_recertify_promoted_keeps_and_restamps_valid_win(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-recert-keep"
    camp = root / "promote-camp-win"
    camp.mkdir(parents=True)
    exp_sha = _mod.locked_promote_expectations_sha256()
    (camp / "metric-certificate.json").write_text(
        json.dumps(_v2_cert(exp_sha=exp_sha, relation="in_band")), encoding="utf-8"
    )
    control, candidate = _held_out_win_metrics()
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "measurement_complete": True,
                "positive": True,
                "control_id": "c-control",
                "candidate_id": "c-promote",
                "control_metrics": control,
                "candidate_metrics": candidate,
                "reasons": ["cert_policy:continue", "promote_primary_win:held_out"],
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "entry_id": "champ-valid",
            "status": "climb_accepted",
            "promotion_campaign_id": camp.name,
            "formal_preflight_status": "proved",
            "promote_authority_sha256": "stale" * 16,
            "resolve_reasons": ["cert_policy:continue"],
        }
    ]
    assert _mod._recertify_promoted_champion_entries(root, loop_id, entries) is True
    assert entries[0]["status"] == "climb_accepted"
    assert entries[0].get("promote_authority_sha256")
    assert entries[0]["promote_authority_sha256"] != "stale" * 16
    assert any(
        "recertified_under_current_promote_authority" in str(r)
        for r in entries[0].get("resolve_reasons") or []
    )
    # Second pass with current stamp is a no-op.
    assert _mod._recertify_promoted_champion_entries(root, loop_id, entries) is False


def test_classify_positive_marks_finalized_decode_timeout_incomplete(
    tmp_path: Path,
) -> None:
    camp = tmp_path / "camp"
    for arm, similarity in (("c-control", 0.2), ("c-candidate", 0.4)):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=1.0,
            structural_similarity=similarity,
            latency_ms_p50=1000.0,
        )
        (run / "scoreboard.json").write_text(
            json.dumps(
                {
                    "suites": [
                        {
                            "suite": "smoke",
                            "n": 3,
                            "completed_document_n": 2,
                            "incomplete_document_n": 1,
                            "decode_timeout_document_count": 1,
                        }
                    ]
                }
            )
        )

    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.structural_similarity",
        control_id="c-control",
        candidate_id="c-candidate",
        role="screening",
    )

    assert result["positive"] is False
    assert _mod._measurement_is_complete(result) is False
    assert any(
        reason.startswith("measurement_incomplete:c-control:smoke:")
        for reason in result["reasons"]
    )


def test_classify_positive_promotion_sees_held_out_primary(tmp_path: Path) -> None:
    """Promotion must ignore a same-leaf smoke override and score held-out."""
    camp = tmp_path / "camp"
    for arm, ss in (("c-control", 0.30), ("c-promote", 0.40)):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=1.0,
            structural_similarity=0.5,
            latency_ms_p50=10000.0,
        )
        _write_eval(
            run / "eval_held_out.json",
            suite="held_out",
            parse_rate=1.0,
            meaningful_program_rate=0.5,
            structural_similarity=ss,
            latency_ms_p50=12000.0,
        )
        _write_complete_scoreboard(run, "smoke", "held_out")
        # No gates.json → avoid fixture_insufficient_n noise for this unit test.
    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.structural_similarity",
        control_id="c-control",
        candidate_id="c-promote",
        role="promotion",
    )
    assert "primary_metric_unavailable" not in (result.get("reasons") or [])
    assert any(
        r.startswith("primary_metric_win:held_out.structural_similarity")
        for r in (result.get("reasons") or [])
    )
    assert result["positive"] is True


def test_classify_positive_rejects_missing_scoreboards(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    for arm, similarity in (("c-control", 0.2), ("c-candidate", 0.4)):
        _write_eval(
            camp / "runs" / arm / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=1.0,
            structural_similarity=similarity,
            latency_ms_p50=1000.0,
        )
    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.structural_similarity",
        control_id="c-control",
        candidate_id="c-candidate",
    )
    assert result["positive"] is False
    assert {
        "measurement_incomplete:c-control:missing_scoreboard",
        "measurement_incomplete:c-candidate:missing_scoreboard",
    }.issubset(result["reasons"])


def test_classify_positive_routes_failed_outcome_to_harness_repair(
    tmp_path: Path,
) -> None:
    camp = tmp_path / "camp"
    for arm, similarity in (("c-control", 0.2), ("c-candidate", 0.4)):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=1.0,
            structural_similarity=similarity,
            latency_ms_p50=1000.0,
        )
        _write_complete_scoreboard(run, "smoke")
    outcomes = camp / "artifacts" / "outcomes"
    outcomes.mkdir(parents=True)
    (outcomes / "candidate.json").write_text(
        json.dumps(
            {
                "experiment_id": "c-candidate",
                "status": "failed",
                "metrics": {},
                "error": "lever_capability_compatibility: unsupported lever",
            }
        ),
        encoding="utf-8",
    )

    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.structural_similarity",
        control_id="c-control",
        candidate_id="c-candidate",
    )

    assert result["positive"] is False
    assert "harness_failure:c-candidate:experiment_failed" in result["reasons"]


@pytest.mark.parametrize(
    ("suites", "reason"),
    case_values(__file__, "test_classify_positive_rejects_invalid_or_empty_suites"),
)
def test_classify_positive_rejects_invalid_or_empty_suites(
    tmp_path: Path, suites: object, reason: str
) -> None:
    camp = tmp_path / "camp"
    for arm, similarity in (("c-control", 0.2), ("c-candidate", 0.4)):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=1.0,
            structural_similarity=similarity,
            latency_ms_p50=1000.0,
        )
        _write_complete_scoreboard(run, "smoke")
    scoreboard = camp / "runs" / "c-candidate" / "scoreboard.json"
    scoreboard.write_text(json.dumps({"suites": suites}), encoding="utf-8")
    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.structural_similarity",
        control_id="c-control",
        candidate_id="c-candidate",
    )
    assert result["positive"] is False
    assert f"measurement_incomplete:c-candidate:{reason}" in result["reasons"]


def test_classify_positive_rejects_inconsistent_suite_counts(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    for arm, similarity in (("c-control", 0.2), ("c-candidate", 0.4)):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=1.0,
            structural_similarity=similarity,
            latency_ms_p50=1000.0,
        )
        _write_complete_scoreboard(run, "smoke")
    scoreboard = camp / "runs" / "c-candidate" / "scoreboard.json"
    payload = json.loads(scoreboard.read_text(encoding="utf-8"))
    payload["suites"]["smoke"]["completed_document_n"] = 2
    scoreboard.write_text(json.dumps(payload), encoding="utf-8")
    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.structural_similarity",
        control_id="c-control",
        candidate_id="c-candidate",
    )
    assert result["positive"] is False
    assert any(
        reason.startswith("measurement_incomplete:c-candidate:smoke:invalid_counts:")
        for reason in result["reasons"]
    )


def test_promotion_cycle_ignores_smoke_cli_primary_override() -> None:
    assert (
        _mod._effective_primary_metric(
            role="promotion",
            policy_metric="held_out.structural_similarity",
            requested_metric="smoke.structural_similarity",
        )
        == "held_out.structural_similarity"
    )


def test_screening_cycle_ignores_held_out_cli_primary_override() -> None:
    assert (
        _mod._effective_primary_metric(
            role="screening",
            policy_metric="smoke.structural_similarity",
            requested_metric="held_out.structural_similarity",
        )
        == "smoke.structural_similarity"
    )


def test_classify_positive_promotion_null_held_out_not_positive(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    for arm in ("c-control", "c-promote"):
        run = camp / "runs" / arm
        _write_eval(
            run / "eval_smoke.json",
            suite="smoke",
            parse_rate=1.0,
            meaningful_program_rate=1.0,
            latency_ms_p50=10000.0,
        )
        _write_eval(
            run / "eval_held_out.json",
            suite="held_out",
            parse_rate=1.0,
            meaningful_program_rate=0.2,
            structural_similarity=0.33582,
            latency_ms_p50=17000.0,
        )
    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="held_out.structural_similarity",
        control_id="c-control",
        candidate_id="c-promote",
        role="promotion",
    )
    assert "primary_metric_unavailable" not in (result.get("reasons") or [])
    assert result["positive"] is False


def test_driver_lock_singleton(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    loop = "loop-x"
    fh1 = _mod.acquire_driver_lock(root, loop, code_sha="abc")
    assert (_mod._driver_lock_path(root, loop)).is_file()
    raised = False
    try:
        _mod.acquire_driver_lock(root, loop, code_sha="def")
    except RuntimeError as exc:
        raised = True
        assert "DRIVER_ALREADY_RUNNING" in str(exc)
    assert raised is True
    import fcntl as _fcntl

    _fcntl.flock(fh1.fileno(), _fcntl.LOCK_UN)
    fh1.close()
    fh2 = _mod.acquire_driver_lock(root, loop, code_sha="ghi")
    fh2.close()


def test_stage_process_updates_child_liveness(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    loop = "loop-x"
    _mod._write_loop_state(
        root,
        _mod.AutotrainLoopStateV1(
            loop_id=loop,
            state="RUNNING",
            phase="running",
            pid=123,
            active_stage="orchestration",
        ),
    )

    _mod._set_active_stage(root, loop, "experiment:arm")
    _mod._set_stage_process(root, loop, "experiment:arm", 456)
    first = _mod.AutotrainLoopStateV1.model_validate_json(
        _mod._loop_state_path(root, loop).read_text()
    )
    _mod._set_stage_process(root, loop, "experiment:arm", 456)
    refreshed = _mod.AutotrainLoopStateV1.model_validate_json(
        _mod._loop_state_path(root, loop).read_text()
    )

    assert first.child_pid == 456
    assert first.stage_started_at is not None
    assert refreshed.stage_started_at == first.stage_started_at
    assert refreshed.heartbeat_at >= first.heartbeat_at

    _mod._clear_active_stage(root, loop)
    completed = _mod.AutotrainLoopStateV1.model_validate_json(
        _mod._loop_state_path(root, loop).read_text()
    )
    assert completed.active_stage is None
    assert completed.child_pid is None
    assert completed.stage_started_at is None


def test_stage_command_publishes_child_pid_and_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ar"
    loop = "loop-x"
    _mod._write_loop_state(
        root,
        _mod.AutotrainLoopStateV1(
            loop_id=loop,
            state="RUNNING",
            phase="running",
            pid=123,
            active_stage="orchestration",
        ),
    )

    def bounded(cmd, **kwargs):
        kwargs["on_start"](456)
        kwargs["on_heartbeat"](456)
        return _mod.BoundedProcessResult(
            command=tuple(cmd),
            outcome=_mod.ProcessOutcome.COMPLETED,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(_mod, "run_bounded_process", bounded)
    _mod._stage_command(
        ["true"],
        cwd=tmp_path,
        root=root,
        loop_id=loop,
        stage="campaign-research",
    )
    state = _mod.AutotrainLoopStateV1.model_validate_json(
        _mod._loop_state_path(root, loop).read_text()
    )

    assert state.active_stage == "campaign-research"
    assert state.child_pid == 456
    assert state.stage_started_at is not None
    assert state.heartbeat_at is not None


def test_promote_formal_preflight_publishes_child_liveness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ar"
    loop = "loop-x"
    camp = root / "camp1"
    _mod._write_loop_state(
        root,
        _mod.AutotrainLoopStateV1(
            loop_id=loop,
            state="RUNNING",
            phase="running",
            pid=123,
            active_stage="orchestration",
        ),
    )

    def successful(
        command,
        *,
        cwd,
        timeout_seconds,
        on_start=None,
        on_heartbeat=None,
    ):
        del cwd
        del timeout_seconds
        assert on_start is not None
        assert on_heartbeat is not None
        on_start(789)
        on_heartbeat(789)
        stdout = "Lean (version 4.30.0)" if "--version" in command else "passed"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("slm_training.autoresearch.formal._run", successful)
    status, sha = _mod.ensure_promote_formal_preflight(
        camp_dir=camp,
        campaign_id="camp1",
        experiment_id="camp1-promote",
        run_lean=True,
        root=root,
        loop_id=loop,
    )
    state = _mod.AutotrainLoopStateV1.model_validate_json(
        _mod._loop_state_path(root, loop).read_text()
    )

    assert status == "proved"
    assert sha is not None
    assert state.active_stage == "promotion-formal-preflight"
    assert state.child_pid == 789
    assert state.heartbeat_at is not None


def test_promote_formal_timeout_obeys_repository_cap() -> None:
    from slm_training.levers import MAX_RUN_SECONDS

    assert _mod._PROMOTE_FORMAL_TIMEOUT_S == float(MAX_RUN_SECONDS)


def test_cycle_deadline_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mod.time, "monotonic", lambda: 10.0)
    with pytest.raises(subprocess.TimeoutExpired, match="autotrain bounded cycle"):
        _mod._remaining_timeout(9.0)


def test_arm_wall_budget_accounts_for_formal_stage_and_reserves_orchestration() -> None:
    from slm_training.levers import (
        HARNESS_FINALIZATION_RESERVE_SECONDS,
        MAX_HARNESS_WALL_SECONDS,
    )

    promotion_minutes = _mod._arm_wall_minutes(3, formal_required=True)
    promotion_expected = min(
        3.0,
        (MAX_HARNESS_WALL_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS) / 3 / 60,
    )
    assert promotion_minutes == pytest.approx(promotion_expected)
    assert _mod._arm_wall_minutes(0.5, formal_required=True) == pytest.approx(
        min(
            0.5,
            (MAX_HARNESS_WALL_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS) / 3 / 60,
        )
    )
    screening_minutes = _mod._arm_wall_minutes(3, formal_required=False)
    screening_expected = min(
        3.0,
        (MAX_HARNESS_WALL_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS) / 2 / 60,
    )
    assert screening_minutes == pytest.approx(screening_expected)
    assert screening_minutes > promotion_minutes
    reserved = 2 * screening_minutes * 60 + HARNESS_FINALIZATION_RESERVE_SECONDS
    assert reserved == pytest.approx(MAX_HARNESS_WALL_SECONDS)


def test_confirmation_during_promotion_cadence_uses_two_arm_budget() -> None:
    formal_required = _mod._formal_lane_required(cycle_intent="confirm", replay=None)

    assert formal_required is False
    assert _mod._arm_wall_minutes(3, formal_required=formal_required) > (
        _mod._arm_wall_minutes(3, formal_required=True)
    )


def test_frozen_formal_replay_retains_formal_lane() -> None:
    replay = {
        "control": {"manifest": SimpleNamespace(formal_obligations=())},
        "candidate": {"manifest": SimpleNamespace(formal_obligations=(object(),))},
    }

    assert _mod._formal_lane_required(cycle_intent="retry_measurement", replay=replay)
    assert _mod._formal_lane_required(cycle_intent="promote", replay=None)


def test_empty_promotion_slot_falls_back_but_frozen_replay_does_not() -> None:
    args = {
        "cadence_role": "promotion",
        "promotion_target_available": False,
        "prior_screening_win_required": True,
    }
    assert _mod._empty_promotion_slot_falls_back(replay=None, **args)
    assert not _mod._empty_promotion_slot_falls_back(replay=object(), **args)
    assert not _mod._empty_promotion_slot_falls_back(
        replay=None,
        **{**args, "promotion_target_available": True},
    )
    assert not _mod._empty_promotion_slot_falls_back(
        replay=None,
        **{**args, "cadence_role": "screening"},
    )


def test_formal_budget_reserves_two_full_arms_and_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slm_training.levers import (
        HARNESS_FINALIZATION_RESERVE_SECONDS,
        MAX_RUN_SECONDS,
    )

    monkeypatch.setattr(_mod.time, "monotonic", lambda: 10.0)
    arm_minutes = 0.75
    deadline = 200.0
    formal = _mod._promotion_formal_budget_seconds(
        deadline=deadline,
        arm_count=2,
        arm_wall_minutes=arm_minutes,
    )
    reserved = 2 * arm_minutes * 60 + HARNESS_FINALIZATION_RESERVE_SECONDS
    capped_remaining = min(float(MAX_RUN_SECONDS), deadline - 10.0)
    assert formal == pytest.approx(capped_remaining - reserved)
    assert formal + reserved == pytest.approx(capped_remaining)


def test_counterbalanced_arm_order_alternates_without_relabeling() -> None:
    first = _mod._counterbalanced_arm_order(
        "control", "candidate", cycle_index=1, seed=101
    )
    second = _mod._counterbalanced_arm_order(
        "control", "candidate", cycle_index=2, seed=102
    )
    assert first == ["control", "candidate"]
    assert second == ["candidate", "control"]
    assert set(first) == set(second) == {"control", "candidate"}


def test_phase_a_uses_explicit_dynamic_arm_ids_and_records_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    camp = tmp_path / "campaign-dynamic"
    camp.mkdir()
    skipped = {
        "successor-dose-3": {
            "reason": "deadline_reserve",
            "remaining_seconds": 1.0,
            "required_seconds": 10.0,
        }
    }
    monkeypatch.setattr(_mod, "_git", lambda *args, **kwargs: "")
    delivery = _mod._phase_a_delivery(
        cwd=tmp_path,
        root=tmp_path,
        loop_id="loop",
        campaign_id="campaign-dynamic",
        primary_metric="smoke.latency_ms_p50",
        control_id="matrix-control",
        candidate_id="successor-dose-3",
        arm_skipped=skipped,
    )
    assert delivery["control_id"] == "matrix-control"
    assert delivery["candidate_id"] == "successor-dose-3"
    assert delivery["arm_skipped"] == skipped


def test_expected_arm_binding_is_append_only_and_content_bound(tmp_path: Path) -> None:
    campaign_id = "campaign-arm-binding"
    campaign = _mod.CampaignSpec(
        campaign_id=campaign_id,
        objective="Bind both decision arms before execution.",
        primary_metric="smoke.parse_rate",
        loop_id="loop-arm-binding",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
    )
    _mod.CampaignStore(campaign_id, tmp_path).initialize(campaign)
    matrix_path = tmp_path / campaign_id / "matrix-proposal.json"
    matrix_path.write_text(json.dumps({"matrix_id": "matrix-1"}), encoding="utf-8")
    event = _mod._bind_expected_arms(
        root=tmp_path,
        campaign_id=campaign_id,
        matrix_path=matrix_path,
        control_id="matrix-control",
        candidate_id="successor-dose-3",
        arm_order=("successor-dose-3", "matrix-control"),
    )
    assert event["event_type"] == "decision_arms_bound"
    assert event["detail"]["expected_arm_ids"] == [
        "matrix-control",
        "successor-dose-3",
    ]
    assert (
        _mod._bind_expected_arms(
            root=tmp_path,
            campaign_id=campaign_id,
            matrix_path=matrix_path,
            control_id="matrix-control",
            candidate_id="successor-dose-3",
            arm_order=("successor-dose-3", "matrix-control"),
        )["event_id"]
        == event["event_id"]
    )


def test_promotion_order_uses_replicate_index_when_cadence_has_same_parity() -> None:
    first = _mod._counterbalanced_arm_order(
        "control",
        "candidate",
        cycle_index=3,
        seed=103,
        promotion_replicate_index=0,
    )
    second = _mod._counterbalanced_arm_order(
        "control",
        "candidate",
        cycle_index=5,
        seed=105,
        promotion_replicate_index=1,
    )
    assert first == ["control", "candidate"]
    assert second == ["candidate", "control"]


def test_driver_requires_room_for_both_arms_before_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_mod.time, "monotonic", lambda: 0.0)
    from slm_training.levers import HARNESS_FINALIZATION_RESERVE_SECONDS

    arm_minutes = 0.75
    required = 2 * arm_minutes * 60 + HARNESS_FINALIZATION_RESERVE_SECONDS
    _mod._require_symmetric_arm_budget(
        deadline=required + 1, arm_count=2, arm_wall_minutes=arm_minutes
    )
    with pytest.raises(subprocess.TimeoutExpired, match="symmetric decision-arm"):
        _mod._require_symmetric_arm_budget(
            deadline=required - 1, arm_count=2, arm_wall_minutes=arm_minutes
        )


def test_post_planning_budget_rejects_shrinking_frozen_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_mod.time, "monotonic", lambda: 10.0)
    with pytest.raises(subprocess.TimeoutExpired, match="symmetric decision-arm"):
        _mod._fit_symmetric_arm_budget(
            deadline=10.0 + 149.0,
            arm_count=2,
            requested_arm_wall_minutes=70 / 60,
        )


def test_fit_arm_budget_leaves_margin_so_deadline_check_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: promote arms skipped when remaining ≈ required by μs."""
    from slm_training.levers import HARNESS_FINALIZATION_RESERVE_SECONDS

    now = 1000.0
    monkeypatch.setattr(_mod.time, "monotonic", lambda: now)
    remaining = 160.0
    deadline = now + remaining
    fitted = _mod._fit_symmetric_arm_budget(
        deadline=deadline,
        arm_count=2,
        requested_arm_wall_minutes=1.2,
    )
    # Simulate a few μs of wall time between fit and execute check.
    monkeypatch.setattr(_mod.time, "monotonic", lambda: now + 1e-4)
    remaining_after = max(0.0, deadline - _mod.time.monotonic())
    required = 2 * fitted * 60.0 + HARNESS_FINALIZATION_RESERVE_SECONDS
    assert remaining_after + 1e-3 >= required


def test_completed_formal_lane_returns_unused_time_to_matched_arms() -> None:
    initial = _mod._arm_wall_minutes(3, formal_required=True)

    assert _mod._post_formal_arm_budget_request(
        policy_minutes=3,
        initial_arm_wall_minutes=initial,
        formal_completed=True,
    ) == pytest.approx(3.0)
    assert _mod._post_formal_arm_budget_request(
        policy_minutes=3,
        initial_arm_wall_minutes=initial,
        formal_completed=False,
    ) == pytest.approx(initial)


def test_formal_campaign_ceiling_allows_dynamic_reclaimed_arm_share() -> None:
    initial = _mod._arm_wall_minutes(3, formal_required=True)
    campaign_ceiling = _mod._post_formal_arm_budget_request(
        policy_minutes=3,
        initial_arm_wall_minutes=initial,
        formal_completed=True,
    )
    reclaimed_share_seconds = 73.305

    assert campaign_ceiling * 60 >= reclaimed_share_seconds
    assert initial * 60 < reclaimed_share_seconds


def test_arm_execution_deadline_preserves_finalization_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slm_training.levers import HARNESS_FINALIZATION_RESERVE_SECONDS

    monkeypatch.setattr(_mod.time, "monotonic", lambda: 10.0)
    cycle_deadline = 100.0

    assert _mod._arm_execution_deadline(
        cycle_deadline=cycle_deadline, arm_wall_minutes=2.0
    ) == pytest.approx(cycle_deadline - HARNESS_FINALIZATION_RESERVE_SECONDS)
    assert _mod._arm_execution_deadline(
        cycle_deadline=cycle_deadline, arm_wall_minutes=0.5
    ) == pytest.approx(40.0)


def test_supervised_cli_runs_exactly_one_agent_owned_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    lock_handle = (tmp_path / "driver.lock").open("w+")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_mod, "_git", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(
        _mod,
        "acquire_driver_lock",
        lambda *args, **kwargs: lock_handle,
    )
    monkeypatch.setattr(
        _mod,
        "run_cycle",
        lambda **kwargs: calls.append(kwargs) or "cycle-1",
    )

    assert _mod.main(["--supervised", "--max-cycles", "1"]) == 0
    assert len(calls) == 1
    assert calls[0]["sync_git"] is False
    assert calls[0]["require_action_receipts"] is True


def test_legacy_unsupervised_cycle_does_not_require_agent_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    lock_handle = (tmp_path / "driver.lock").open("w+")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_mod, "_git", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(
        _mod,
        "acquire_driver_lock",
        lambda *args, **kwargs: lock_handle,
    )
    monkeypatch.setattr(
        _mod,
        "run_cycle",
        lambda **kwargs: calls.append(kwargs) or "cycle-1",
    )

    assert _mod.main(["--max-cycles", "1"]) == 0
    assert calls[0]["sync_git"] is True
    assert calls[0]["require_action_receipts"] is False
    assert calls[0]["startup_commit"] == "a" * 40


def test_cli_reexecs_after_integrating_new_driver_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_handle = (tmp_path / "driver.lock").open("w+")
    reexec: list[tuple[str, list[str]]] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_mod, "_git", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(
        _mod,
        "acquire_driver_lock",
        lambda *args, **kwargs: lock_handle,
    )
    monkeypatch.setattr(
        _mod,
        "run_cycle",
        lambda **kwargs: (_ for _ in ()).throw(_mod._CodeUpdated("new head")),
    )

    def fake_execv(executable: str, argv: list[str]) -> None:
        reexec.append((executable, argv))
        raise SystemExit(0)

    monkeypatch.setattr(_mod.os, "execv", fake_execv)
    with pytest.raises(SystemExit):
        _mod.main(["--max-cycles", "1"])
    assert reexec and reexec[0][0] == _mod.sys.executable


def _priority_matrix() -> dict:
    return {
        "next_run_priorities": [
            {
                "rank": 1,
                "area": "model",
                "hypothesis": "Test the next quality-bearing model lever.",
                "evidence_ids": ["feedback-1"],
                "confidence": 0.5,
                "expected_information_gain": "Resolve one quality residual.",
                "authority": "speculative",
                "disposition": "monitor",
                "proposed_experiment_id": None,
            }
        ]
    }


def _write_terminal_feedback(root: Path, campaign_id: str) -> str:
    from slm_training.autoresearch.schemas import HypothesisFeedback

    feedback_id = "feedback-" + hashlib.sha256(campaign_id.encode()).hexdigest()[:16]
    feedback = HypothesisFeedback(
        feedback_id=feedback_id,
        campaign_id=campaign_id,
        matrix_id=f"matrix-{campaign_id}",
        experiment_id="candidate",
        hypothesis="The current smoke objective has exhausted its legal arms.",
        knob_signature="{}",
        outcome_status="completed",
        diagnosis_target="researcher",
        diagnosis_evidence=("bounded screening objective saturated",),
        recommended_actions=("refresh the capability objective",),
    )
    store = _mod.CampaignStore(campaign_id, root)
    path = store.write_artifact("hypothesizer_feedback", feedback)
    store.append_event(
        "hypothesizer_feedback_recorded",
        experiment_id="candidate",
        status="researcher",
        artifact_sha256=path.stem,
        detail={"feedback_id": feedback_id},
    )
    return feedback_id


def test_cycle_handoff_separates_fixture_climb_from_ship(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-1"
    (camp / "runs" / "cand").mkdir(parents=True)
    (camp / "runs" / "cand" / "gates.json").write_text(
        json.dumps({"authority": "AgentEvals assertions", "pass": False})
    )
    (camp / "runs" / "cand" / "last.pt").write_bytes(b"checkpoint")
    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="promotion",
        cycle_intent="promote",
        primary_metric="held_out.structural_similarity",
        matrix=_priority_matrix(),
        delivery={
            "positive": True,
            "candidate_id": "cand",
            "reasons": ["primary_metric_win:x"],
            "stack_layer": False,
            "arm_order": ["control", "cand"],
            "arm_seed": 101,
        },
        resolution={"status": "climb_accepted", "resolve_reasons": []},
        formal_status="proved",
    )
    assert handoff.climb_state == "climb_accepted"
    assert handoff.ship_state == "blocked"
    assert handoff.evidence_class == "fixture"
    assert handoff.checkpoint_paths == ("runs/cand/last.pt",)
    assert handoff.checkpoint_documentation_required is True
    assert "arm_order:control,cand" in handoff.reasons
    assert "MODEL_CARD" in next(
        action.reason for action in handoff.actions if action.kind == "document"
    )
    assert {action.kind for action in handoff.actions} == {
        "document",
        "next_experiment",
    }
    state = json.loads((root / "loops" / "loop-1" / "state.json").read_text())
    assert state["phase"] == "between_cycles"


def test_predecessor_without_stack_delta_does_not_block_on_deliver_stack(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-1"
    camp.mkdir(parents=True)
    (camp / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "schema_version": "AutotrainCycleHandoffV1",
                "loop_id": "loop-1",
                "campaign_id": "cycle-1",
                "cycle_index": 1,
                "upstream_commit": "a" * 40,
                "integration_commit": "b" * 40,
                "cycle_role": "screening",
                "cycle_intent": "screening",
                "evidence_class": "fixture",
                "climb_state": "candidate_queued",
                "ship_state": "blocked",
                "primary_metric": "smoke.structural_similarity",
                "actions": [
                    {
                        "kind": "deliver_stack",
                        "owner": "sdlc",
                        "reason": "stale",
                        "evidence_ids": ["campaign:cycle-1"],
                    }
                ],
            }
        )
    )
    (camp / "sdlc_delivery.json").write_text(json.dumps({"stack_layer": False}))
    _mod._require_predecessor_actions(root, "loop-1", "cycle-1")


def test_cycle_handoff_routes_exhausted_bank_to_capability_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inject_terminal_policy(monkeypatch, park=False)
    root = tmp_path / "autoresearch"
    (root / "cycle-exhausted").mkdir(parents=True)
    _write_terminal_feedback(root, "cycle-exhausted")
    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-exhausted",
        cycle_index=1795,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="screening",
        primary_metric="held_out.structural_similarity",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "candidate_id": "edge-alignment",
            "control_id": "control",
            "reasons": ["primary_metric_not_improved"],
        },
        resolution=None,
        formal_status="proved",
    )

    assert all(action.kind != "repair_harness" for action in handoff.actions)
    assert handoff.actions[0].kind == "rebuild_data"
    nxt = next(a for a in handoff.actions if a.kind == "next_experiment")
    assert "researcher once" in nxt.reason.lower()
    assert _mod._current_rung_label() in nxt.reason
    assert "simplified-nl-to-ast" not in nxt.reason.lower()
    assert handoff.priorities[0].disposition == "monitor"


def test_cycle_handoff_routes_frozen_harness_repair(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-1"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    predecessor = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"], text=True
    ).strip()
    (camp / "artifacts" / "outcomes").mkdir(parents=True)
    (camp / "artifacts" / "outcomes" / "outcome.json").write_text(
        json.dumps(
            {
                "harness_signals": [
                    {
                        "family": "model_build",
                        "reproduced_on_frozen_input": True,
                    }
                ]
            }
        )
    )
    (camp / "manifests").mkdir()
    experiment = {
        "experiment_id": "cand",
        "campaign_id": "cycle-1",
        "hypothesis": "A frozen harness failure remains replayable after repair.",
        "rationale": "Repair and measurement are independent obligations.",
        "expected_effect": "The same content-bound manifest is replayed.",
        "falsification_criteria": ["The retry disappears after repair."],
        "stop_conditions": ["Stop after the bounded replay."],
        "citations": ["fixture://repair-replay"],
        "knobs": {"steps": 1},
    }
    manifest_path = camp / "manifests" / "cand.json"
    manifest_path.write_text(
        _mod._manifest("cycle-1", experiment, predecessor).model_dump_json(indent=2)
        + "\n"
    )
    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-1",
        cycle_index=1,
        upstream_commit=predecessor,
        integration_commit=predecessor,
        role="promotion",
        cycle_intent="promote",
        primary_metric="held_out.structural_similarity",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "candidate_id": "cand",
            "reasons": ["harness_failure:missing_promote_run"],
            "stack_layer": False,
        },
        resolution={"status": "harness_failure", "resolve_reasons": []},
        formal_status="proved",
    )
    repair = handoff.actions[0]
    assert repair.kind == "repair_harness"
    assert repair.owner == "improve-openui-harnesses"
    assert repair.harness_family == "model_build"
    assert repair.frozen_manifest_sha256 is not None
    retry = handoff.actions[1]
    assert retry.kind == "retry_measurement"
    assert retry.owner == "autotrain"
    assert retry.frozen_manifest_sha256 == repair.frozen_manifest_sha256
    assert "identical frozen arm" in retry.reason
    assert all(action.kind != "next_experiment" for action in handoff.actions)

    repair_index = handoff.actions.index(repair)
    evidence = _mod.bind_autotrain_action_evidence(root, handoff, repair, (head,))
    _mod.append_autotrain_action_receipt(
        root,
        _mod.AutotrainActionReceiptV1(
            loop_id="loop-1",
            campaign_id="cycle-1",
            action_index=repair_index,
            action_sha256=_mod.autotrain_action_sha256(repair),
            action_kind=repair.kind,
            status="completed",
            evidence_uris=(head,),
            evidence=evidence,
        ),
    )
    assert all(
        action.kind != "repair_harness"
        for _, action in _mod.pending_autotrain_actions(root, handoff)
    )
    pending_execution = _mod.pending_autotrain_execution_actions(root, handoff)
    assert pending_execution == ((handoff.actions.index(retry), retry),)


def test_repeated_cycle_failure_blocks_on_third_identical_error(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    for expected in (1, 2, 3):
        count = _mod._record_cycle_failure(
            root=root,
            loop_id="loop-1",
            exc=RuntimeError("same blocker"),
            cycle_index=0,
        )
        assert count == expected
    state = json.loads((root / "loops" / "loop-1" / "state.json").read_text())
    assert state["state"] == "BLOCKED"
    assert state["blocker_count"] == 3


def test_completed_candidate_priorities_accepts_dynamic_successor() -> None:
    """Dynamic compose arms must not raise StopIteration in priority steering."""
    _mod._DYNAMIC_THRASH_ARMS.clear()
    _mod._DYNAMIC_THRASH_ARMS.append(
        (
            "compose-ltr-tail-ltr-prefix",
            "Joint ltr tail and prefix.",
            {
                "ltr_tail_loss_weight": 2.0,
                "ltr_prefix_loss_weight": 1.0,
                "_thrash_slug": "compose-ltr-tail-ltr-prefix",
            },
        )
    )
    matrix = {
        "hypotheses": [
            {
                "experiment": {
                    "experiment_id": "c-control",
                    "knobs": {},
                    "hypothesis": "control",
                }
            },
            {
                "experiment": {
                    "experiment_id": "c-bounds",
                    "knobs": {"grammar_completion_bounds": True},
                    "hypothesis": "bounds",
                }
            },
        ],
        "next_run_priorities": [
            {
                "rank": 1,
                "area": "model_build",
                "hypothesis": "try bounds",
                "evidence_ids": ["a"],
                "confidence": 0.9,
                "expected_information_gain": "x",
                "authority": "observed_result",
                "disposition": "experiment_next",
                "proposed_experiment_id": "c-bounds",
            }
        ],
    }
    # Skip every static quality arm so selection falls through to compose-*.
    skip = {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    rows = _mod._completed_candidate_priorities(
        matrix,
        "c-bounds",
        resolved_infrastructure=True,
        skip_slugs=skip,
    )
    assert rows  # must not raise StopIteration
    assert any(
        (p.proposed_experiment_id or "").endswith("compose-ltr-tail-ltr-prefix")
        or "compose" in str(p.hypothesis or "")
        for p in rows
    ) or any(p.disposition == "monitor" for p in rows)


def test_self_heal_thrash_bank_composes_successors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bank exhaust must self-heal by composing size-matched thrash arms."""
    # Legacy compose synthesis: only active while terminal parking is off.
    _inject_terminal_policy(monkeypatch, park=False)
    _mod._DYNAMIC_THRASH_ARMS.clear()
    _mod._DYNAMIC_THRASH_LOADED_FOR = None
    root = tmp_path / "ar"
    loop = "L"
    closed = {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    assert _mod._self_heal_thrash_bank_exhaust(root, loop, closed=closed, skip=set())
    assert _mod._DYNAMIC_THRASH_ARMS
    path = _mod._dynamic_thrash_arms_path(root, loop)
    assert path.is_file()
    # Newly composed arms are selectable and signature-unique vs static bank.
    static_sigs = {
        _mod._thrash_lever_signature(ex) for _, _, ex in _mod._SCREENING_ARM_BANK
    }
    for slug, _, extras in _mod._DYNAMIC_THRASH_ARMS:
        assert slug.startswith("compose-")
        assert _mod._thrash_lever_signature(extras) not in static_sigs
    slug = _mod._select_recommended_slug(1, skip=closed)
    assert slug.startswith("compose-")
    extras = dict(_mod._DYNAMIC_THRASH_ARMS[0][2])
    assert _mod._arm_slug_from_knobs(extras) == _mod._DYNAMIC_THRASH_ARMS[0][0]


def test_thrash_matrix_dedupes_compose_vs_static_knob_signatures() -> None:
    """Matrix must not fail HypothesisMatrix when compose collides with static."""
    from slm_training.autoresearch.schemas import HypothesisMatrix

    _mod._DYNAMIC_THRASH_ARMS.clear()
    # Deliberate collision: same levers as scaffold-prefix-tail.
    _mod._DYNAMIC_THRASH_ARMS.append(
        (
            "compose-ltr-tail-ltr-prefix",
            "Duplicate of scaffold-prefix-tail levers for collision test.",
            {
                "ltr_tail_loss_weight": 1.0,
                "ltr_prefix_loss_weight": 1.0,
                "_thrash_slug": "compose-ltr-tail-ltr-prefix",
            },
        )
    )
    # Also add a unique compose arm that must survive.
    _mod._DYNAMIC_THRASH_ARMS.append(
        (
            "compose-ltr-prefix-compiler-decision-token",
            "Unique prefix plus compiler-decision token thrash successor.",
            {
                "ltr_prefix_loss_weight": 1.0,
                "compiler_decision_token_loss_weight": 1.0,
                "_thrash_slug": "compose-ltr-prefix-compiler-decision-token",
            },
        )
    )
    matrix = _mod._matrix(
        campaign_id="continuous-loop-dedupe-c1",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=40,
        cycle=1,
        role="screening",
        recommended_slug="compose-ltr-tail-ltr-prefix",
    )
    HypothesisMatrix.model_validate(matrix)
    ids = [h["experiment"]["experiment_id"] for h in matrix["hypotheses"]]
    assert not any(i.endswith("compose-ltr-tail-ltr-prefix") for i in ids)
    assert any(i.endswith("compose-ltr-prefix-compiler-decision-token") for i in ids)
    # Recommended retargets to a real hypothesis id.
    assert matrix["recommended_experiment_id"] in ids


def test_self_heal_cycle_error_recovers_bank_exhaust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Legacy compose recovery: only active while terminal parking is off.
    _inject_terminal_policy(monkeypatch, park=False)
    _mod._DYNAMIC_THRASH_ARMS.clear()
    _mod._DYNAMIC_THRASH_LOADED_FOR = None
    root = tmp_path / "ar"
    loop = "L"
    kind = _mod._self_heal_cycle_error(
        root=root,
        loop_id=loop,
        exc=RuntimeError(_mod._BANK_EXHAUST_MSG),
        integration_commit="abc",
    )
    assert kind == "thrash_bank_compose"
    _mod._clear_loop_blocker(root, loop, reason=kind)
    state = json.loads((root / "loops" / loop / "state.json").read_text())
    assert state["state"] == "IDLE"
    assert state["blocker_count"] == 0


def test_screening_bank_fingerprint_tracks_bank_identity() -> None:
    fingerprint = _mod._screening_bank_fingerprint(policy_sha256="s")
    assert fingerprint == _mod._screening_bank_fingerprint(policy_sha256="s")
    assert fingerprint != _mod._screening_bank_fingerprint(policy_sha256="t")
    _mod._DYNAMIC_THRASH_ARMS.append(
        ("compose-fp-probe", "Fingerprint probe.", {"ltr_tail_loss_weight": 2.0})
    )
    assert fingerprint != _mod._screening_bank_fingerprint(policy_sha256="s")


def test_self_heal_unblock_does_not_tag_compose_when_bank_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Open bank is not a thrash_bank_compose recovery (root-cause of 3377 heals)."""
    _inject_terminal_policy(monkeypatch, park=False)
    _mod._DYNAMIC_THRASH_ARMS.clear()
    _mod._DYNAMIC_THRASH_LOADED_FOR = None
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    root.mkdir(parents=True)
    report = _mod.self_heal_unblock_loop(cwd=repo, root=root, loop_id="L")
    assert "thrash_bank_compose" not in (report.get("soft_healed") or [])
    assert not _mod._DYNAMIC_THRASH_ARMS


def test_self_heal_thrash_bank_compose_only_when_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inject_terminal_policy(monkeypatch, park=False)
    _mod._DYNAMIC_THRASH_ARMS.clear()
    _mod._DYNAMIC_THRASH_LOADED_FOR = None
    root = tmp_path / "ar"
    loop = "L"
    closed = {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    result = _mod._self_heal_thrash_bank_exhaust(root, loop, closed=closed, skip=set())
    assert result.composed
    assert result.available
    already = _mod._self_heal_thrash_bank_exhaust(root, loop, closed=set(), skip=set())
    assert already.available
    assert not already.composed


def test_park_policy_retires_confirm_fallback_and_compose_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _inject_terminal_policy(monkeypatch, park=True)
    exhausted = {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    champion = {"status": "confirmed", "entry_id": "champ-1"}
    assert not _mod._repeat_confirm_while_waiting_for_promotion(
        cadence_role="screening",
        confirmed_champion=champion,
        cycle=1811,
        skip=exhausted,
    )
    root = tmp_path / "ar"
    assert not _mod._self_heal_thrash_bank_exhaust(
        root, "L", closed=exhausted, skip=set()
    )
    assert not _mod._DYNAMIC_THRASH_ARMS
    assert not _mod._dynamic_thrash_arms_path(root, "L").is_file()
    # Open arms remaining never park: heal still reports the bank as usable.
    assert _mod._self_heal_thrash_bank_exhaust(root, "L", closed=set(), skip=set())


def test_should_enqueue_rejects_fixture_volume_win() -> None:
    # Latency-primary fixture ticks without quality_held stay out of the queue.
    assert not _mod._should_enqueue_champion(
        {
            "positive": False,
            "primary_metric": "smoke.latency_ms_p50",
            "measurement_complete": True,
            "reasons": [
                "fixture_insufficient_n:c159-control",
                "primary_metric_win:smoke.latency_ms_p50:10000->8000",
                "fixture_insufficient_n_alone",
            ],
            "control_metrics": {"parse_rate": 1.0, "meaningful_program_rate": 0.3},
            "candidate_metrics": {"parse_rate": 1.0, "meaningful_program_rate": 0.3},
        }
    )


def test_should_enqueue_fixture_volume_structural_win() -> None:
    """Smoke below Lean floor must not enqueue, even with an SS tick."""
    delivery = {
        "positive": False,
        "primary_metric": "smoke.structural_similarity",
        "measurement_complete": True,
        "reasons": [
            "fixture_insufficient_n:c159-control",
            "fixture_insufficient_n:c159-typed-family-balance",
            "primary_metric_win:smoke.structural_similarity:"
            "0.174->0.214:improvement=0.04",
            "fixture_insufficient_n_alone",
        ],
        "control_metrics": {
            "parse_rate": 1.0,
            "meaningful_program_rate": 0.333,
            "structural_similarity": 0.174,
            "binder_reference_f1": 0.6,
        },
        "candidate_metrics": {
            "parse_rate": 1.0,
            "meaningful_program_rate": 0.333,
            "structural_similarity": 0.214,
            "binder_reference_f1": 0.7,
        },
        "candidate_id": "c159-typed-family-balance",
        "control_id": "c159-control",
    }
    assert not _mod._is_confirm_candidate_win(delivery)
    assert not _mod._should_enqueue_champion(delivery)


def test_should_enqueue_lean_floor_ss_win_with_held_mpr() -> None:
    delivery = {
        "positive": False,
        "primary_metric": "smoke.structural_similarity",
        "measurement_complete": True,
        "reasons": [
            "fixture_insufficient_n:c-control",
            "primary_metric_win:smoke.structural_similarity:0.13->0.20:improvement=0.07",
            "fixture_volume_gate_ship_only",
        ],
        "control_metrics": {
            "parse_rate": 1.0,
            "meaningful_program_rate": 0.167,
            "structural_similarity": 0.13,
            "binder_reference_f1": 0.53,
            "n": 6,
        },
        "candidate_metrics": {
            "parse_rate": 1.0,
            "meaningful_program_rate": 0.167,
            "structural_similarity": 0.20,
            "binder_reference_f1": 0.53,
            "n": 6,
        },
    }
    assert _mod._is_confirm_candidate_win(delivery)
    assert _mod._should_enqueue_champion(delivery)


def test_should_not_enqueue_mpr_zero_or_quality_identity() -> None:
    mpr0 = {
        "positive": False,
        "primary_metric": "smoke.structural_similarity",
        "measurement_complete": True,
        "reasons": [
            "primary_metric_win:smoke.structural_similarity:0.11->0.20:improvement=0.09",
            "fixture_volume_gate_ship_only",
        ],
        "control_metrics": {
            "parse_rate": 1.0,
            "meaningful_program_rate": 0.167,
            "structural_similarity": 0.11,
            "binder_reference_f1": 0.53,
            "n": 6,
        },
        "candidate_metrics": {
            "parse_rate": 1.0,
            "meaningful_program_rate": 0.0,
            "structural_similarity": 0.20,
            "binder_reference_f1": 0.0,
            "n": 6,
        },
    }
    assert not _mod._should_enqueue_champion(mpr0)
    ident = {
        "positive": False,
        "primary_metric": "smoke.structural_similarity",
        "measurement_complete": True,
        "reasons": ["mechanism_no_effect:quality_metrics_identical"],
        "control_metrics": {
            "parse_rate": 1.0,
            "meaningful_program_rate": 0.167,
            "structural_similarity": 0.13,
            "binder_reference_f1": 0.53,
        },
        "candidate_metrics": {
            "parse_rate": 1.0,
            "meaningful_program_rate": 0.167,
            "structural_similarity": 0.13,
            "binder_reference_f1": 0.53,
        },
    }
    assert _mod._quality_metrics_identical(ident)
    assert not _mod._should_enqueue_champion(ident)


def test_handoff_parks_exhausted_bank_despite_experiment_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _inject_terminal_policy(monkeypatch, park=True)
    monkeypatch.setattr(
        _mod,
        "_recent_completed_nonpositive_slugs",
        lambda *args, **kwargs: {slug for slug, _, _ in _mod._SCREENING_ARM_BANK},
    )
    monkeypatch.setattr(_mod, "_thrash_bank_open_slugs", lambda closed: set())
    root = tmp_path / "autoresearch"
    (root / "cycle-exhausted").mkdir(parents=True)
    _write_terminal_feedback(root, "cycle-exhausted")
    matrix = _priority_matrix()
    matrix["next_run_priorities"] = [
        {
            "rank": 1,
            "area": "model",
            "hypothesis": "Rematch a just-lost decoder slug.",
            "evidence_ids": ["feedback-1"],
            "confidence": 0.9,
            "expected_information_gain": "More smoke.",
            "authority": "observed_result",
            "disposition": "experiment_next",
            "proposed_experiment_id": "c-next-literal-close-structure",
        }
    ]
    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-exhausted",
        cycle_index=1795,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="screening",
        primary_metric="smoke.structural_similarity",
        matrix=matrix,
        delivery={
            "positive": False,
            "candidate_id": "literal-close-structure",
            "control_id": "control",
            "reasons": ["primary_metric_null_or_worse"],
        },
        resolution=None,
        formal_status="proved",
    )
    assert handoff.terminal_verdict is not None
    assert handoff.terminal_verdict.binding_constraint == "quality_arm_bank_exhausted"
    assert [action.kind for action in handoff.actions] == [
        "rebuild_data",
        "document",
        "next_experiment",
    ]
    assert policy.payload["terminal"]["park_on_exhaust"] is True
    state = json.loads((root / "loops" / "loop-1" / "state.json").read_text())
    assert state["state"] == "BLOCKED"
    assert state["next_action"] == "rebuild_data"


def test_arm_slug_recovers_snapshot_suffix_when_thrash_slug_stripped() -> None:
    _mod._DYNAMIC_THRASH_ARMS.append(
        (
            "simplified-nl-frontier-c52",
            "snapshot leftover",
            {"train_version": "frontier_simplified_nl_c52_v1"},
        )
    )
    knobs = {
        "train_version": "frontier_simplified_nl_c52_v1",
        "fidelity_loss_weight": 0.5,
    }
    assert (
        _mod._arm_slug_from_knobs(
            knobs,
            candidate_id="c193-simplified-nl-frontier-c52",
        )
        == "simplified-nl-frontier-c52"
    )


def test_arm_slug_does_not_map_steps40_variant_to_steps() -> None:
    _mod._DYNAMIC_THRASH_ARMS.append(
        (
            "simplified-nl-c52-steps40",
            "snapshot leftover",
            {"train_version": "frontier_simplified_nl_c52_v1"},
        )
    )
    knobs = {"train_version": "frontier_simplified_nl_c52_v1", "steps": 40}
    assert (
        _mod._arm_slug_from_knobs(
            knobs,
            candidate_id="c58-simplified-nl-c52-steps40",
        )
        == "simplified-nl-c52-steps40"
    )


def test_recent_completed_closes_snapshot_clones_by_train_version(
    tmp_path: Path,
) -> None:
    _mod._DYNAMIC_THRASH_ARMS.extend(
        [
            (
                "simplified-nl-frontier-c52",
                "front",
                {"train_version": "frontier_simplified_nl_c52_v1"},
            ),
            (
                "simplified-nl-c52-steps40",
                "clone",
                {"train_version": "frontier_simplified_nl_c52_v1"},
            ),
        ]
    )
    root = tmp_path / "autoresearch"
    predecessor: str | None = None
    for cycle, slug in (
        (180, "simplified-nl-frontier-c52"),
        (181, "simplified-nl-c52-steps40"),
    ):
        campaign_id = f"continuous-loop-20260814-c{cycle}"
        candidate_id = f"{campaign_id}-{slug}"
        camp = root / campaign_id
        camp.mkdir(parents=True)
        (camp / "campaign.json").write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "loop_id": "loop-1",
                    "predecessor_campaign_id": predecessor,
                }
            )
        )
        (camp / "matrix-proposal.json").write_text(
            json.dumps(
                {
                    "hypotheses": [
                        {
                            "experiment": {
                                "experiment_id": candidate_id,
                                "knobs": {
                                    "train_version": "frontier_simplified_nl_c52_v1",
                                    "seed": 100000 + cycle,
                                },
                            }
                        }
                    ]
                }
            )
        )
        (camp / "sdlc_delivery.json").write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "cycle_intent": "screening",
                    "positive": False,
                    "measurement_complete": True,
                }
            )
        )
        (camp / "cycle_handoff.json").write_text(
            json.dumps({"loop_id": "loop-1", "cycle_intent": "screening"})
        )
        predecessor = campaign_id

    closed = _mod._recent_completed_nonpositive_slugs(root, predecessor)
    assert "simplified-nl-frontier-c52" in closed
    assert "simplified-nl-c52-steps40" in closed


def test_heal_slug_not_closed_by_nulls_on_distinct_snapshots(
    tmp_path: Path,
) -> None:
    """Two nulls on the shared heal slug but distinct train_versions must not
    close the slug — snapshot close is by train_version identity, not slug
    spelling (each fresh heal snapshot is a new approach)."""
    _mod._DYNAMIC_THRASH_ARMS.append(
        (
            _mod._HEAL_RESUME_SLUG,
            "heal",
            {"train_version": "continuous_i10_loop_1_c9_harness", "heal_resume": True},
        )
    )
    root = tmp_path / "autoresearch"
    predecessor: str | None = None
    for cycle, version in (
        (169, "continuous_i10_loop_1_c7_harness"),
        (170, "continuous_i10_loop_1_c8_harness"),
    ):
        campaign_id = f"continuous-loop-20260819-c{cycle}"
        candidate_id = f"{campaign_id}-{_mod._HEAL_RESUME_SLUG}"
        camp = root / campaign_id
        camp.mkdir(parents=True)
        (camp / "campaign.json").write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "loop_id": "loop-1",
                    "predecessor_campaign_id": predecessor,
                }
            )
        )
        (camp / "matrix-proposal.json").write_text(
            json.dumps(
                {
                    "hypotheses": [
                        {
                            "experiment": {
                                "experiment_id": candidate_id,
                                "knobs": {
                                    "train_version": version,
                                    "seed": 100000 + cycle,
                                },
                            }
                        }
                    ]
                }
            )
        )
        (camp / "sdlc_delivery.json").write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "cycle_intent": "screening",
                    "positive": False,
                    "measurement_complete": True,
                }
            )
        )
        (camp / "cycle_handoff.json").write_text(
            json.dumps({"loop_id": "loop-1", "cycle_intent": "screening"})
        )
        predecessor = campaign_id

    closed = _mod._recent_completed_nonpositive_slugs(root, predecessor)
    assert _mod._HEAL_RESUME_SLUG not in closed


def test_handoff_parks_when_only_snapshot_leftovers_remain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _inject_terminal_policy(monkeypatch, park=True)
    root = tmp_path / "autoresearch"
    _mod._append_dynamic_thrash_arms(
        root,
        "loop-1",
        [
            (
                "simplified-nl-frontier-c48",
                "c48 leftover",
                {"train_version": "frontier_simplified_nl_c48_v1"},
            ),
            (
                "simplified-nl-frontier-c52",
                "c52 leftover",
                {"train_version": "frontier_simplified_nl_c52_v1"},
            ),
        ],
    )
    monkeypatch.setattr(
        _mod,
        "_recent_completed_nonpositive_slugs",
        lambda *args, **kwargs: {slug for slug, _, _ in _mod._SCREENING_ARM_BANK},
    )
    (root / "cycle-snapshots").mkdir(parents=True)
    _write_terminal_feedback(root, "cycle-snapshots")
    matrix = _priority_matrix()
    matrix["next_run_priorities"] = [
        {
            "rank": 1,
            "area": "experiments",
            "hypothesis": "The recent quality families are exhausted; run c48.",
            "evidence_ids": ["feedback-1"],
            "confidence": 0.9,
            "expected_information_gain": "Rematch a snapshot clone.",
            "authority": "observed_result",
            "disposition": "experiment_next",
            "proposed_experiment_id": "c-next-simplified-nl-frontier-c48",
        }
    ]
    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-snapshots",
        cycle_index=193,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="screening",
        primary_metric="smoke.structural_similarity",
        matrix=matrix,
        delivery={
            "positive": False,
            "candidate_id": "simplified-nl-frontier-c52",
            "control_id": "control",
            "reasons": ["fixture_insufficient_n_alone"],
        },
        resolution=None,
        formal_status="proved",
    )
    assert handoff.terminal_verdict is not None
    assert handoff.terminal_verdict.binding_constraint == "quality_arm_bank_exhausted"
    assert [action.kind for action in handoff.actions] == [
        "rebuild_data",
        "document",
        "next_experiment",
    ]
    assert policy.payload["terminal"]["park_on_exhaust"] is True
    assert _mod._open_slugs_are_snapshot_leftovers(
        {"simplified-nl-frontier-c48", "simplified-nl-frontier-c52"}
    )
    leftover = _mod._thrash_bank_open_slugs(
        {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    )
    # Isolate open set excludes snapshot train_version arms, so leftover
    # is empty and park-before-select fires instead of smoking c96.
    assert leftover == set()


def test_park_screening_saturation_executes_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _inject_terminal_policy(monkeypatch, park=True)
    root = tmp_path / "autoresearch"
    camp = root / "cycle-park"
    camp.mkdir(parents=True)
    _write_terminal_feedback(root, "cycle-park")
    handoff = _mod.AutotrainCycleHandoffV1(
        loop_id="loop-1",
        campaign_id="cycle-park",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        cycle_role="screening",
        cycle_intent="screening",
        evidence_class="fixture",
        climb_state="rejected",
        ship_state="not_evaluated",
        primary_metric="smoke.structural_similarity",
        actions=(
            _mod.AutotrainActionV1(
                kind="document",
                owner="documenting-experiment-results",
                reason="closeout",
                evidence_ids=("campaign:cycle-park",),
            ),
        ),
    )
    (camp / "cycle_handoff.json").write_text(
        handoff.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    called: dict[str, object] = {}

    def _rebuild(**kwargs: object) -> str:
        called.update(kwargs)
        return "rebuild_data"

    monkeypatch.setattr(_mod, "_self_heal_rebuild_data", _rebuild)
    cwd = tmp_path / "repo"
    cwd.mkdir()
    status = _mod._park_screening_saturation(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-park",
        cycle_index=1,
        policy=policy,
        ranked_regimes=["bounds"],
        cwd=cwd,
    )
    assert status == _mod._REGIME_PARKED_STATUS
    assert called.get("campaign_id") == "cycle-park"
    assert called.get("cwd") == cwd


def test_handoff_does_not_park_leftover_isolate_ofat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inject_terminal_policy(monkeypatch, park=True)
    monkeypatch.setattr(
        _mod,
        "_recent_completed_nonpositive_slugs",
        lambda *args, **kwargs: (
            {slug for slug, _, _ in _mod._SCREENING_ARM_BANK} - {"legal-edit-hazard"}
        ),
    )
    root = tmp_path / "autoresearch"
    (root / "cycle-leftover").mkdir(parents=True)
    _write_terminal_feedback(root, "cycle-leftover")
    matrix = _priority_matrix()
    matrix["next_run_priorities"] = [
        {
            "rank": 1,
            "area": "model_build",
            "hypothesis": "Bank exhausted; wire a new objective.",
            "evidence_ids": ["feedback-1"],
            "confidence": 0.95,
            "expected_information_gain": "Avoid rematch.",
            "authority": "observed_result",
            "disposition": "monitor",
            "proposed_experiment_id": None,
        }
    ]
    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-leftover",
        cycle_index=167,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="screening",
        primary_metric="smoke.structural_similarity",
        matrix=matrix,
        delivery={
            "positive": False,
            "candidate_id": "legal-edit-hazard",
            "control_id": "control",
            "reasons": ["primary_metric_null_or_worse"],
        },
        resolution=None,
        formal_status="proved",
    )
    assert handoff.terminal_verdict is None
    assert "rebuild_data" not in {action.kind for action in handoff.actions}
    assert "next_experiment" in {action.kind for action in handoff.actions}


def test_local_rebuild_argv_keeps_policy_plan_surface() -> None:
    argv = _mod._local_rebuild_data_argv(train_version="continuous_i10_test")
    assert "--synthesis-plan" in argv
    assert "simplified_nl" not in argv
    assert "--unique-root-target" in argv
    # Heal snapshots stay under outputs/; publishing into tracked resources/
    # mid-cycle parks the loop as foreign_dirty_tree.
    assert "--no-publish" in argv
    # Rung-honest heal identity: the resume arm must not claim a skipped rung.
    assert "simplified" not in _mod._HEAL_RESUME_SLUG


def test_local_rebuild_argv_shaped_by_adequacy_stays_wall_capped() -> None:
    from slm_training.autoresearch.sample_adequacy import (
        SampleAdequacyObservation,
        compute_sample_adequacy,
    )

    report = compute_sample_adequacy(
        SampleAdequacyObservation(
            observed_records=101,
            component_witnesses={"Button": 60, "SwitchGroup": 2},
        )
    ).model_dump(mode="json")
    argv = _mod._local_rebuild_data_argv(
        train_version="continuous_i10_test", adequacy=report
    )
    # Targeted mode reaches the local build; the promotion-scale component
    # minimum and floor-sized root target stay off the wall-capped CPU heal.
    assert "--generation-mode" in argv
    assert argv[argv.index("--generation-mode") + 1] == "until_coverage"
    assert "--component-coverage-minimum" not in argv
    target = int(argv[argv.index("--unique-root-target") + 1])
    assert target <= _mod._LOCAL_I10_ROOT_CAP


def test_sample_adequacy_report_reads_fixture_stats(tmp_path: Path) -> None:
    stats_dir = tmp_path / "src/slm_training/resources/data/train/wf_smoke_v2"
    stats_dir.mkdir(parents=True)
    (stats_dir / "stats.json").write_text(
        json.dumps(
            {
                "record_count": 101,
                "component_histogram": {"Button": 60, "SwitchGroup": 2},
            }
        )
    )
    report = _mod._sample_adequacy_report(tmp_path)
    assert report is not None
    assert report["verdict"] == "generate_more"
    assert report["coverage_deficits"] == {"SwitchGroup": 2}
    # No stats anywhere: no report, heal proceeds unshaped.
    assert _mod._sample_adequacy_report(tmp_path / "missing") is None


def test_heal_resume_arm_stays_open_when_snapshots_are_excluded() -> None:
    _mod._DYNAMIC_THRASH_ARMS.append(
        (
            _mod._HEAL_RESUME_SLUG,
            "I10 heal",
            {"train_version": "continuous_i10_loop_c196", "heal_resume": True},
        )
    )
    leftover = _mod._thrash_bank_open_slugs(
        {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    )
    assert leftover == {_mod._HEAL_RESUME_SLUG}
    assert not _mod._open_slugs_are_snapshot_leftovers(leftover)
    assert _mod._select_recommended_slug(197, skip=set()) == _mod._HEAL_RESUME_SLUG


def test_thrash_bank_exhaust_does_not_park_selectable_heal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed heal slug + unused train_version is not snapshot leftover park."""
    _inject_terminal_policy(monkeypatch, park=True)
    root = tmp_path / "autoresearch"
    loop = "loop-1"
    version = "continuous_i10_loop_1_c509_harness"
    _mod._register_i10_heal_arm(root, loop, train_version=version)
    closed = {_mod._HEAL_RESUME_SLUG} | {
        slug for slug, _, _ in _mod._SCREENING_ARM_BANK
    }
    monkeypatch.setattr(
        _mod,
        "_train_version_has_complete_nonpositive",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        _mod, "_latest_cycle", lambda *args, **kwargs: (509, "cycle-509")
    )
    assert _mod._selectable_process_arm(root, loop, predecessor_campaign_id="cycle-509")
    assert _mod._self_heal_thrash_bank_exhaust(
        root,
        loop,
        closed=closed,
        skip=closed,
        predecessor_campaign_id="cycle-509",
    )
    assert (
        _mod._select_recommended_slug(512, skip=closed, root=root, loop_id=loop)
        == _mod._HEAL_RESUME_SLUG
    )


def _write_heal_snapshot(cwd: Path, version: str) -> Path:
    train_dir = cwd / "outputs" / "data" / "train" / version
    train_dir.mkdir(parents=True)
    for name in (
        "quality_report.json",
        "synthesis_feedback.json",
        "data_manifest.json",
    ):
        (train_dir / name).write_text("{}\n", encoding="utf-8")
    return train_dir


def test_regime_parked_recovers_lost_heal_arm(tmp_path: Path) -> None:
    """An acked rebuild_data whose arm registration was lost must not park forever."""
    root = tmp_path / "autoresearch"
    loop = "loop-1"
    version = "continuous_i10_loop_1_c168"
    _write_heal_snapshot(tmp_path, version)
    verdict = root / "loops" / loop / "terminal_verdict.json"
    verdict.parent.mkdir(parents=True)
    verdict.write_text(
        json.dumps(
            {
                "schema_version": "regime_exhausted_verdict/v1",
                "campaign_id": "cycle-parked",
                "loop_id": loop,
                "cycle_index": 168,
                "binding_constraint": "screening_objective_saturated",
                "bank_fingerprint": _mod._screening_bank_fingerprint(),
            }
        )
    )
    assert _mod._check_regime_parked(root=root, loop_id=loop, cwd=tmp_path) is None
    slugs = {slug for slug, _, _ in _mod._DYNAMIC_THRASH_ARMS}
    assert _mod._HEAL_RESUME_SLUG in slugs
    assert not verdict.is_file()


def test_regime_parked_skips_tombstoned_heal_version(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop = "loop-1"
    version = "continuous_i10_loop_1_c168"
    _write_heal_snapshot(tmp_path, version)
    tombstones = _mod._heal_retired_versions_path(root, loop)
    tombstones.parent.mkdir(parents=True)
    tombstones.write_text(
        json.dumps({"train_version": version}) + "\n", encoding="utf-8"
    )
    verdict = root / "loops" / loop / "terminal_verdict.json"
    verdict.write_text(
        json.dumps(
            {
                "schema_version": "regime_exhausted_verdict/v1",
                "campaign_id": "cycle-parked",
                "loop_id": loop,
                "cycle_index": 168,
                "binding_constraint": "screening_objective_saturated",
                "bank_fingerprint": _mod._screening_bank_fingerprint(),
            }
        )
    )
    assert (
        _mod._check_regime_parked(root=root, loop_id=loop, cwd=tmp_path)
        == _mod._REGIME_PARKED_STATUS
    )
    assert verdict.is_file()


def _write_heal_null_lineage(
    root: Path, *, version: str, loop: str, n_nulls: int = 2
) -> str:
    predecessor: str | None = None
    campaign_id = ""
    for i in range(n_nulls):
        campaign_id = f"continuous-loop-spent-c{i}"
        candidate_id = f"{campaign_id}-{_mod._HEAL_RESUME_SLUG}"
        camp = root / campaign_id
        camp.mkdir(parents=True)
        (camp / "campaign.json").write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "loop_id": loop,
                    "predecessor_campaign_id": predecessor,
                }
            )
        )
        (camp / "matrix-proposal.json").write_text(
            json.dumps(
                {
                    "hypotheses": [
                        {
                            "experiment": {
                                "experiment_id": candidate_id,
                                "knobs": {
                                    "train_version": version,
                                    "seed": 100000 + i,
                                },
                            }
                        }
                    ]
                }
            )
        )
        (camp / "sdlc_delivery.json").write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "cycle_intent": "screening",
                    "positive": False,
                    "measurement_complete": True,
                }
            )
        )
        (camp / "cycle_handoff.json").write_text(
            json.dumps({"loop_id": loop, "cycle_intent": "screening"})
        )
        predecessor = campaign_id
    return campaign_id


def test_regime_parked_does_not_unpark_spent_heal_snapshot(tmp_path: Path) -> None:
    """A registered heal arm whose train_version is already null-closed must
    stay parked — unparking on mere process-arm presence is the spin."""
    root = tmp_path / "autoresearch"
    loop = "loop-1"
    version = "continuous_i10_loop_1_c168"
    _write_heal_snapshot(tmp_path, version)
    last = _write_heal_null_lineage(root, version=version, loop=loop)
    _mod._register_i10_heal_arm(root, loop, train_version=version)
    verdict = root / "loops" / loop / "terminal_verdict.json"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    verdict.write_text(
        json.dumps(
            {
                "schema_version": "regime_exhausted_verdict/v1",
                "campaign_id": last,
                "loop_id": loop,
                "cycle_index": 168,
                "binding_constraint": "screening_objective_saturated",
                "bank_fingerprint": _mod._screening_bank_fingerprint(),
            }
        )
    )
    assert (
        _mod._check_regime_parked(root=root, loop_id=loop, cwd=tmp_path)
        == _mod._REGIME_PARKED_STATUS
    )
    assert verdict.is_file()
    assert version in _mod._retired_heal_versions(root, loop)


def test_retire_i10_heal_arm_writes_tombstone(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop = "loop-1"
    _mod._append_dynamic_thrash_arms(
        root,
        loop,
        [
            (
                _mod._HEAL_RESUME_SLUG,
                "I10 heal",
                {"train_version": "continuous_i10_loop_1_c9", "heal_resume": True},
            )
        ],
    )
    assert _mod._retire_i10_heal_arm(root, loop, reason="complete_measurement:c9")
    assert _mod._retired_heal_versions(root, loop) == {"continuous_i10_loop_1_c9"}


def test_regime_parked_resumes_when_heal_arm_is_open(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop = "loop-1"
    verdict = root / "loops" / loop / "terminal_verdict.json"
    verdict.parent.mkdir(parents=True)
    verdict.write_text(
        json.dumps(
            {
                "schema_version": "regime_exhausted_verdict/v1",
                "campaign_id": "cycle-parked",
                "loop_id": loop,
                "cycle_index": 196,
                "binding_constraint": "screening_objective_saturated",
                "bank_fingerprint": _mod._screening_bank_fingerprint(),
            }
        )
    )
    _mod._append_dynamic_thrash_arms(
        root,
        loop,
        [
            (
                _mod._HEAL_RESUME_SLUG,
                "I10 heal",
                {"train_version": "continuous_i10_loop_c196", "heal_resume": True},
            )
        ],
    )
    assert _mod._check_regime_parked(root=root, loop_id=loop, cwd=tmp_path) is None
    assert not verdict.is_file()


def test_process_arm_outranks_predecessor_leftover_rematch() -> None:
    _mod._DYNAMIC_THRASH_ARMS.append(
        (
            _mod._HEAL_RESUME_SLUG,
            "I10 heal",
            {"train_version": "continuous_i10_loop_c196", "heal_resume": True},
        )
    )
    try:
        chosen = _mod._select_cycle_slug(
            199,
            predecessor_priority="simplified-nl-c78-all-maxchildren3",
            skip=set(),
            has_confirm_levers=False,
            has_promote_levers=False,
        )
    finally:
        _mod._DYNAMIC_THRASH_ARMS.pop()
    assert chosen == _mod._HEAL_RESUME_SLUG


def test_self_heal_rebuild_data_acks_local_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "autoresearch"
    campaign_id = "cycle-parked"
    camp = root / campaign_id
    camp.mkdir(parents=True)
    handoff = {
        "schema_version": "AutotrainCycleHandoffV1",
        "loop_id": "loop-1",
        "campaign_id": campaign_id,
        "cycle_index": 196,
        "upstream_commit": "a" * 40,
        "integration_commit": "b" * 40,
        "cycle_role": "screening",
        "cycle_intent": "screening",
        "evidence_class": "fixture",
        "climb_state": "rejected",
        "ship_state": "blocked",
        "primary_metric": "smoke.structural_similarity",
        "reasons": ["fixture_insufficient_n_alone"],
        "priorities": [],
        "actions": [
            {
                "schema_version": "AutotrainActionV1",
                "kind": "rebuild_data",
                "owner": "synthesis-feedback",
                "reason": "expand simplified-NL inventory",
                "evidence_ids": [f"campaign:{campaign_id}"],
            }
        ],
        "created_at": "2026-08-14T00:00:00Z",
    }
    (camp / "cycle_handoff.json").write_text(json.dumps(handoff) + "\n")
    version = _mod._local_i10_train_version("loop-1", 196)
    train_dir = tmp_path / "outputs" / "data" / "train" / version
    train_dir.mkdir(parents=True)
    for name in ("manifest.json", "quality_report.json", "synthesis_feedback.json"):
        (train_dir / name).write_text(json.dumps({"ok": True, "name": name}) + "\n")

    def _forbid_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("existing artifacts must not rebuild")

    monkeypatch.setattr(_mod, "run_bounded_process", _forbid_build)
    monkeypatch.setattr(
        _mod,
        "_sample_adequacy_report",
        lambda cwd: {"verdict": "generate_more"},
    )
    monkeypatch.setattr(_mod, "_thrash_bank_open_slugs", lambda closed: set())
    kind = _mod._self_heal_rebuild_data(
        cwd=tmp_path, root=root, loop_id="loop-1", campaign_id=campaign_id
    )
    assert kind == "rebuild_data"
    assert (camp / "quality_report.json").is_file()
    assert (camp / "sample_adequacy.json").is_file()
    receipts = (root / "loops" / "loop-1" / "action_receipts.jsonl").read_text()
    assert "rebuild_data" in receipts
    assert "sample_adequacy.json" not in receipts
    assert any(
        slug == _mod._HEAL_RESUME_SLUG
        for slug, _, extras in _mod._all_screening_arm_bank()
    )


def test_self_heal_rebuild_data_skips_i10_arm_when_leftover_ofat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "autoresearch"
    campaign_id = "cycle-leftover-heal"
    camp = root / campaign_id
    camp.mkdir(parents=True)
    handoff = {
        "schema_version": "AutotrainCycleHandoffV1",
        "loop_id": "loop-1",
        "campaign_id": campaign_id,
        "cycle_index": 167,
        "upstream_commit": "a" * 40,
        "integration_commit": "b" * 40,
        "cycle_role": "screening",
        "cycle_intent": "screening",
        "evidence_class": "fixture",
        "climb_state": "rejected",
        "ship_state": "blocked",
        "primary_metric": "smoke.structural_similarity",
        "reasons": ["fixture_insufficient_n_alone"],
        "priorities": [],
        "actions": [
            {
                "schema_version": "AutotrainActionV1",
                "kind": "rebuild_data",
                "owner": "synthesis-feedback",
                "reason": "expand simplified-NL inventory",
                "evidence_ids": [f"campaign:{campaign_id}"],
            }
        ],
        "created_at": "2026-08-14T00:00:00Z",
    }
    (camp / "cycle_handoff.json").write_text(json.dumps(handoff) + "\n")
    version = _mod._local_i10_train_version("loop-1", 167)
    train_dir = tmp_path / "outputs" / "data" / "train" / version
    train_dir.mkdir(parents=True)
    for name in ("manifest.json", "quality_report.json", "synthesis_feedback.json"):
        (train_dir / name).write_text(json.dumps({"ok": True, "name": name}) + "\n")

    monkeypatch.setattr(
        _mod,
        "run_bounded_process",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(
        _mod, "_thrash_bank_open_slugs", lambda closed: {"legal-edit-hazard"}
    )
    monkeypatch.setattr(_mod, "_open_slugs_are_snapshot_leftovers", lambda slugs: False)
    kind = _mod._self_heal_rebuild_data(
        cwd=tmp_path, root=root, loop_id="loop-1", campaign_id=campaign_id
    )
    assert kind == "rebuild_data"
    assert not any(
        slug == _mod._HEAL_RESUME_SLUG
        for slug, _, extras in _mod._all_screening_arm_bank()
    )


def test_supervisor_noops_when_regime_parked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_autotrain_supervisor",
        Path(__file__).resolve().parents[2] / "scripts" / "run_autotrain_supervisor.py",
    )
    assert spec is not None and spec.loader is not None
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    root = tmp_path / "autoresearch"
    loop = "loop-parked"
    verdict = root / "loops" / loop / "terminal_verdict.json"
    verdict.parent.mkdir(parents=True)
    verdict.write_text(
        json.dumps(
            {
                "schema_version": "regime_exhausted_verdict/v1",
                "campaign_id": "cycle-exhausted",
                "loop_id": loop,
                "cycle_index": 170,
                "binding_constraint": "quality_arm_bank_exhausted",
                "closed_slugs": [],
                "policy_sha256": None,
                "resume_predicate": "I10 objective preregistered",
                "bank_fingerprint": _mod._screening_bank_fingerprint(),
            }
        )
    )
    launched: list[list[str]] = []

    def _forbid_launch(*args: object, **kwargs: object) -> object:
        launched.append(list(args[0]) if args else [])
        raise AssertionError("parked supervisor must not start a smoke cycle")

    monkeypatch.setattr(supervisor.subprocess, "run", _forbid_launch)
    fake_continuous = SimpleNamespace(
        _check_regime_parked=lambda **kwargs: _mod._REGIME_PARKED_STATUS,
        self_heal_unblock_loop=lambda **kwargs: {
            "soft_healed": [],
            "hard_pending": [{"kind": "rebuild_data", "reason": "still pending"}],
            "blocker_cleared": False,
            "predecessor_campaign_id": "cycle-exhausted",
        },
    )
    monkeypatch.setattr(supervisor, "_load_continuous", lambda: fake_continuous)
    rc = supervisor.main(
        [
            "--loop-id",
            loop,
            "--root",
            str(root),
            "--max-cycles",
            "3",
            "--train-version",
            "wf_smoke_v2",
            "--steps",
            "20",
        ]
    )
    assert rc == 0
    assert launched == []


def test_bank_exhaust_parks_loop_under_typed_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.autoresearch.schemas import RegimeExhaustedVerdictV1

    policy = _inject_terminal_policy(monkeypatch, park=True)
    monkeypatch.setattr(
        _mod,
        "_recent_completed_nonpositive_slugs",
        lambda *args, **kwargs: {slug for slug, _, _ in _mod._SCREENING_ARM_BANK},
    )
    root = tmp_path / "autoresearch"
    (root / "cycle-exhausted").mkdir(parents=True)
    feedback_id = _write_terminal_feedback(root, "cycle-exhausted")
    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-exhausted",
        cycle_index=1795,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="screening",
        primary_metric="held_out.structural_similarity",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "candidate_id": "edge-alignment",
            "control_id": "control",
            "reasons": ["primary_metric_not_improved"],
        },
        resolution=None,
        formal_status="proved",
    )

    assert handoff.terminal_verdict is not None
    assert handoff.terminal_verdict.binding_constraint == "quality_arm_bank_exhausted"
    assert handoff.terminal_verdict.bank_fingerprint == (
        _mod._screening_bank_fingerprint(policy_sha256=policy.sha256)
    )
    assert [action.kind for action in handoff.actions] == [
        "rebuild_data",
        "document",
        "next_experiment",
    ]
    assert all(feedback_id in action.evidence_ids for action in handoff.actions[::2])
    assert handoff.actions[0].owner == "synthesis-feedback"
    assert handoff.actions[-1].owner == "autotrain"
    assert _mod._current_rung_label() in handoff.actions[-1].reason
    assert "simplified-NL-to-AST" not in handoff.actions[-1].reason
    verdict_path = _mod._terminal_verdict_path(root, "loop-1")
    assert verdict_path.is_file()
    persisted = RegimeExhaustedVerdictV1.model_validate_json(
        verdict_path.read_text(encoding="utf-8")
    )
    assert persisted.bank_fingerprint == handoff.terminal_verdict.bank_fingerprint
    state = json.loads((root / "loops" / "loop-1" / "state.json").read_text())
    assert state["state"] == "BLOCKED"
    assert state["phase"] == "blocked"
    assert state["blocker_fingerprint"] == "quality_arm_bank_exhausted"


def test_regime_parked_early_return_and_fingerprint_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ar"
    loop = "loop-parked"
    verdict_path = _mod._terminal_verdict_path(root, loop)
    verdict_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_verdict(fingerprint: str) -> None:
        verdict_path.write_text(
            json.dumps(
                {
                    "schema_version": "regime_exhausted_verdict/v1",
                    "campaign_id": "cycle-exhausted",
                    "loop_id": loop,
                    "cycle_index": 1795,
                    "binding_constraint": "quality_arm_bank_exhausted",
                    "closed_slugs": [],
                    "policy_sha256": None,
                    "resume_predicate": "bank identity changes",
                    "bank_fingerprint": fingerprint,
                }
            )
        )

    # Unchanged fingerprint: the loop stays parked without running anything.
    _write_verdict(_mod._screening_bank_fingerprint())
    assert (
        _mod._check_regime_parked(root=root, loop_id=loop, cwd=tmp_path)
        == _mod._REGIME_PARKED_STATUS
    )
    assert verdict_path.is_file()
    out = capsys.readouterr().out
    assert f"REGIME_PARKED loop={loop}" in out
    assert "constraint=quality_arm_bank_exhausted" in out

    # Changed fingerprint: archive the verdict deterministically and resume.
    _write_verdict("0" * 64)
    assert _mod._check_regime_parked(root=root, loop_id=loop, cwd=tmp_path) is None
    assert not verdict_path.is_file()
    resolved = verdict_path.with_name("terminal_verdict.resolved.c1795.json")
    assert resolved.is_file()
    out = capsys.readouterr().out
    assert "REGIME_RESUMED reason=bank_identity_changed" in out
    state = json.loads((root / "loops" / loop / "state.json").read_text())
    assert state["state"] == "IDLE"
    assert state["blocker_count"] == 0
    # No verdict file means no park check applies at all.
    assert _mod._check_regime_parked(root=root, loop_id=loop, cwd=tmp_path) is None


def test_run_cycle_short_circuits_on_parked_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ar"
    loop = "loop-parked"
    verdict_path = _mod._terminal_verdict_path(root, loop)
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text(
        json.dumps(
            {
                "schema_version": "regime_exhausted_verdict/v1",
                "campaign_id": "cycle-exhausted",
                "loop_id": loop,
                "cycle_index": 3,
                "binding_constraint": "quality_arm_bank_exhausted",
                "closed_slugs": [],
                "policy_sha256": None,
                "resume_predicate": "bank identity changes",
                "bank_fingerprint": _mod._screening_bank_fingerprint(),
            }
        )
    )
    monkeypatch.setattr(
        _mod,
        "_git",
        lambda *args, **kwargs: "" if args[0] == "status" else "a" * 40,
    )
    status = _mod.run_cycle(
        cwd=tmp_path,
        root=root,
        loop_id=loop,
        train_version="wf_smoke_v2",
        steps=1,
        objective="objective",
        primary_metric="smoke.structural_similarity",
        sync_git=False,
        require_action_receipts=False,
    )
    assert status == _mod._REGIME_PARKED_STATUS
    # Parked cycles run no experiment and write no campaign bundle.
    assert not list(root.glob("*/campaign.json"))


def test_promotion_manifest_embeds_locked_power_feasibility() -> None:
    experiment = {
        "experiment_id": "cand-promote",
        "hypothesis": "A confirmed champion holds its held-out primary win.",
        "knobs": {"seed": 7, "eval_version": "e_test"},
    }
    manifest = _mod._manifest("cycle-1", experiment, "a" * 40, role="promotion")
    report = manifest.power_feasibility
    assert report is not None
    assert report["schema"] == "power_feasibility/v1"
    # Policy v8 sets measurement.promotion_suite_n to the exact sign-test
    # floor, so promote campaigns lock a decisive feasibility report.
    assert report["n"] == 24
    assert report["required_n"] == 6
    assert report["decisive"] is True
    screening = _mod._manifest("cycle-1", experiment, "a" * 40)
    assert screening.power_feasibility is None


def test_dispose_champion_promote_refuses_infeasible_power_report() -> None:
    from fractions import Fraction

    from slm_training.autoresearch import evidence_ledger as ev

    infeasible = ev.power_feasibility_report(3, Fraction(1, 20))
    assert infeasible["decisive"] is False
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=None,
        power_feasibility=infeasible,
    )
    assert d["status"] == "promotion_failed"
    assert any(r.startswith("promotion_infeasible_by_design:") for r in d["reasons"])

    # A decisive report passes through to the unchanged downstream gates.
    feasible = ev.power_feasibility_report(6, Fraction(1, 20))
    assert feasible["decisive"] is True
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=None,
        power_feasibility=feasible,
    )
    assert not any(
        r.startswith("promotion_infeasible_by_design:") for r in d["reasons"]
    )
    assert d["status"] == "promotion_failed"
    assert any(r.startswith("promote_requires_certificate") for r in d["reasons"])

    # Absent report (legacy / screening campaigns): dispose is untouched.
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=None,
    )
    assert not any(
        r.startswith("promotion_infeasible_by_design:") for r in d["reasons"]
    )


def test_campaign_power_feasibility_reads_candidate_manifest(
    tmp_path: Path,
) -> None:
    camp = tmp_path / "camp"
    (camp / "manifests").mkdir(parents=True)
    report = {
        "schema": "power_feasibility/v1",
        "n": 3,
        "alpha": "1/20",
        "min_two_sided_p": "1/4",
        "decisive": False,
        "required_n": 6,
    }
    (camp / "manifests" / "cand.json").write_text(
        json.dumps({"power_feasibility": report})
    )
    assert _mod._campaign_power_feasibility(camp, "cand") == report
    assert _mod._campaign_power_feasibility(camp, "") is None
    assert _mod._campaign_power_feasibility(camp, "missing") is None


def test_repeated_timeouts_remain_soft_and_never_block(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    for _ in range(4):
        count = _mod._record_cycle_failure(
            root=root,
            loop_id="loop-1",
            exc=subprocess.TimeoutExpired("bounded-stage", 3),
            cycle_index=2,
        )
        assert count == 0
    state = json.loads((root / "loops" / "loop-1" / "state.json").read_text())
    assert state["state"] == "IDLE"
    assert state["blocker_count"] == 0
    assert state["next_action"] == "retry incomplete cycle"


def test_campaign_identity_is_loop_scoped() -> None:
    first = _mod._campaign_id("continuous-a", 7, date="20260801")
    second = _mod._campaign_id("continuous-b", 7, date="20260801")
    assert first != second
    assert first.endswith("-c7")
    assert second.endswith("-c7")


def test_latest_cycle_uses_highest_index_but_last_completed_predecessor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    for index in (10, 11):
        campaign_id = f"cycle-{index}"
        camp = root / campaign_id
        camp.mkdir(parents=True)
        (camp / "campaign.json").write_text(
            json.dumps(
                {
                    "loop_id": "loop-1",
                    "campaign_id": campaign_id,
                    "cycle_index": index,
                }
            )
        )
    (root / "cycle-10" / "cycle_handoff.json").write_text("{}\n")

    assert _mod._latest_cycle(root, "loop-1") == (11, "cycle-10")
    assert _mod._campaign_at_cycle(root, "loop-1", 11) == "cycle-11"


def test_terminal_interrupted_replay_finalizes_without_rerunning_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "autoresearch"
    campaign_id = "cycle-1"
    camp = root / campaign_id
    (camp / "manifests").mkdir(parents=True)
    (camp / "campaign.json").write_text(
        json.dumps(
            {
                "loop_id": "loop-1",
                "campaign_id": campaign_id,
                "cycle_index": 1,
                "primary_metric": "smoke.binder_reference_f1",
                "upstream_commit": "a" * 40,
                "integration_commit": "b" * 40,
            }
        )
    )
    matrix = {
        **_priority_matrix(),
        "recommended_experiment_id": "candidate",
        "hypotheses": [{"experiment": {"experiment_id": "control"}}],
    }
    (camp / "matrix-proposal.json").write_text(json.dumps(matrix))
    for experiment_id in ("control", "candidate"):
        (camp / "manifests" / f"{experiment_id}.json").write_text(
            json.dumps(
                {
                    "replay_of_manifest_sha256": "c" * 64,
                    "claim_class": "diagnostic",
                }
            )
        )
    store = _mod.CampaignStore(campaign_id, root)
    store.append_event("experiment_finished", experiment_id="control")

    subprocesses: list[list[str]] = []
    handoffs: list[dict] = []
    monkeypatch.setattr(
        _mod, "_run", lambda command, **_kwargs: subprocesses.append(command)
    )
    monkeypatch.setattr(
        _mod,
        "_phase_a_delivery",
        lambda **_kwargs: {
            "positive": False,
            "candidate_id": "candidate",
            "reasons": ["measurement_incomplete:no_smoke_metrics"],
        },
    )
    monkeypatch.setattr(
        _mod,
        "_write_cycle_handoff",
        lambda **kwargs: handoffs.append(kwargs),
    )

    assert (
        _mod._finalize_terminal_interrupted_replay(
            cwd=tmp_path,
            root=root,
            loop_id="loop-1",
            deadline=float("inf"),
        )
        is None
    )
    assert subprocesses == []

    store.append_event("experiment_finished", experiment_id="candidate")
    candidate_manifest = camp / "manifests" / "candidate.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "replay_of_manifest_sha256": "c" * 64,
                "claim_class": "promotion_candidate",
            }
        )
    )
    assert (
        _mod._finalize_terminal_interrupted_replay(
            cwd=tmp_path,
            root=root,
            loop_id="loop-1",
            deadline=float("inf"),
        )
        is None
    )
    assert subprocesses == []
    candidate_manifest.write_text(
        json.dumps(
            {
                "replay_of_manifest_sha256": "c" * 64,
                "claim_class": "diagnostic",
            }
        )
    )
    assert (
        _mod._finalize_terminal_interrupted_replay(
            cwd=tmp_path,
            root=root,
            loop_id="loop-1",
            deadline=float("inf"),
        )
        == campaign_id
    )
    assert len(subprocesses) == 1
    assert "run" not in subprocesses[0]
    assert handoffs[0]["cycle_intent"] == "retry_measurement"
    assert handoffs[0]["campaign_id"] == campaign_id


@pytest.mark.parametrize(
    "candidate_slug", ("batch1", "component-plan", "literal-close")
)
def test_frozen_replay_preserves_recipe_and_links_current_main_successor(
    candidate_slug: str,
) -> None:
    old_campaign = "continuous-loop-20260801-loop-12345678-c1710"
    new_campaign = "continuous-loop-20260801-loop-12345678-c1712"
    matrix = _mod._matrix(
        campaign_id=new_campaign,
        evidence_snapshot_id="snapshot-1",
        cites=["docs/design/autoresearch-autotraining.md"],
        role_citations={
            "research": "docs/design/autoresearch-autotraining.md",
            "prior_result": "docs/design/autoresearch-autotraining.md",
        },
        train_version="wf_smoke_v2",
        eval_version="e938_role_safe_all_targets_v2",
        steps=22,
        cycle=1712,
        recommended_slug=candidate_slug,
    )
    old_control = json.loads(json.dumps(matrix["hypotheses"][0]["experiment"]))
    old_candidate = json.loads(
        json.dumps(
            next(
                row["experiment"]
                for row in matrix["hypotheses"]
                if row["experiment"]["experiment_id"].endswith(f"-{candidate_slug}")
            )
        )
    )
    old_control.update(
        experiment_id="c20260801-loop-12345678-c1710-control",
        campaign_id=old_campaign,
    )
    old_candidate.update(
        experiment_id=f"c20260801-loop-12345678-c1710-{candidate_slug}",
        campaign_id=old_campaign,
    )
    old_control["knobs"].update(steps=80, seed=101710, batch_size=2)
    old_candidate["knobs"].update(steps=80, seed=101710, batch_size=1)
    old_commit = "a" * 40
    control_manifest = _mod._manifest(old_campaign, old_control, old_commit)
    candidate_manifest = _mod._manifest(old_campaign, old_candidate, old_commit)
    replay = {
        "control": {
            "experiment": old_control,
            "manifest": control_manifest,
            "manifest_sha256": "b" * 64,
        },
        "candidate": {
            "experiment": old_candidate,
            "manifest": candidate_manifest,
            "manifest_sha256": "c" * 64,
        },
    }

    replay_manifests = _mod._apply_frozen_replay(matrix, replay, new_campaign)
    assert len(matrix["hypotheses"]) == 5
    recommended = next(
        row["experiment"]
        for row in matrix["hypotheses"]
        if row["experiment"]["experiment_id"] == matrix["recommended_experiment_id"]
    )
    assert recommended["knobs"] == old_candidate["knobs"]
    current_commit = "d" * 40
    successor = _mod._replay_successor_manifest(
        replay_manifests[matrix["recommended_experiment_id"]]["manifest"],
        frozen_manifest_sha256="c" * 64,
        campaign_id=new_campaign,
        experiment_id=matrix["recommended_experiment_id"],
        integration_commit=current_commit,
    )
    assert successor.source_commit == current_commit
    assert successor.replay_of_manifest_sha256 == "c" * 64
    assert successor.endpoints == candidate_manifest.endpoints
    assert successor.arms == candidate_manifest.arms
    assert successor.seeds == candidate_manifest.seeds
    assert successor.stopping_rules == candidate_manifest.stopping_rules
    assert successor.promotion_gates == candidate_manifest.promotion_gates
    formal_manifest = _mod._manifest(
        old_campaign,
        old_candidate,
        old_commit,
        role="promotion",
        cycle_intent="promote",
        formal_preflight_sha256="e" * 64,
    )
    assert formal_manifest.formal_obligations
    formal_successor = _mod._replay_successor_manifest(
        formal_manifest,
        frozen_manifest_sha256="f" * 64,
        campaign_id=new_campaign,
        experiment_id="new-promote",
        integration_commit=current_commit,
    )
    assert formal_successor.formal_obligations == ()
    rebound = _mod._bind_fresh_replay_formal_preflight(
        formal_successor,
        formal_manifest,
        preflight_sha256="1" * 64,
        formal_claims=[_mod.promote_formal_claim_dict()],
    )
    assert len(rebound.formal_obligations) == 1
    assert rebound.formal_obligations[0].preflight_sha256 == "1" * 64
    assert (
        rebound.formal_obligations[0].template_id
        == formal_manifest.formal_obligations[0].template_id
    )
    assert rebound.formal_obligations[0].obligation_id == _mod.formal_obligation_id(
        new_campaign,
        "new-promote",
        _mod.FormalClaimV1(**_mod.promote_formal_claim_dict()),
    )
    assert (
        rebound.formal_obligations[0].obligation_id
        != formal_manifest.formal_obligations[0].obligation_id
    )
    promote_experiment = json.loads(json.dumps(old_control))
    promote_experiment["experiment_id"] = "c20260801-loop-12345678-c1710-promote"
    promote_experiment["hypothesis"] = (
        "Promotion retest of confirmed champion levers under held-out suites."
    )
    promote_experiment["formal_claims"] = [_mod.promote_formal_claim_dict()]
    promote_experiment["knobs"].update(
        typed_family_balance_loss_weight=0.25,
        compiler_alignment_loss_weight=1.0,
        compiler_alignment_margin=1.0,
        compiler_alignment_kind_filter="container-close",
        compiler_alignment_stratified=True,
    )
    promote_manifest = _mod._manifest(
        old_campaign,
        promote_experiment,
        old_commit,
        role="promotion",
        cycle_intent="promote",
        formal_preflight_sha256="e" * 64,
    )
    promotion_replay = {
        "control": replay["control"],
        "candidate": {
            "experiment": promote_experiment,
            "manifest": promote_manifest,
            "manifest_sha256": "f" * 64,
        },
    }
    promotion_matrix = _mod._matrix(
        campaign_id=new_campaign,
        evidence_snapshot_id="snapshot-2",
        cites=["docs/design/autoresearch-autotraining.md"],
        role_citations={
            "research": "docs/design/autoresearch-autotraining.md",
            "prior_result": "docs/design/autoresearch-autotraining.md",
        },
        train_version="wf_smoke_v2",
        eval_version="e938_role_safe_all_targets_v2",
        steps=22,
        cycle=1712,
        role="promotion",
        recommended_slug="batch1",
    )
    applied = _mod._apply_frozen_replay(
        promotion_matrix, promotion_replay, new_campaign
    )
    assert promotion_matrix["recommended_experiment_id"].endswith("-promote")
    assert promotion_matrix["recommended_experiment_id"] in applied
    promoted = next(
        item["experiment"]
        for item in promotion_matrix["hypotheses"]
        if item["experiment"]["experiment_id"]
        == promotion_matrix["recommended_experiment_id"]
    )
    assert promoted["knobs"] == promote_experiment["knobs"]
    assert promoted["formal_claims"] == promote_experiment["formal_claims"]

    confirm_experiment = json.loads(json.dumps(old_candidate))
    confirm_experiment["experiment_id"] = "c20260801-loop-12345678-c1710-confirm"
    confirm_manifest = _mod._manifest(
        old_campaign,
        confirm_experiment,
        old_commit,
        role="screening",
        cycle_intent="confirm",
    )
    confirmation_replay = {
        "control": replay["control"],
        "candidate": {
            "experiment": confirm_experiment,
            "manifest": confirm_manifest,
            "manifest_sha256": "1" * 64,
        },
    }
    confirmation_matrix = _mod._matrix(
        campaign_id=new_campaign,
        evidence_snapshot_id="snapshot-3",
        cites=["docs/design/autoresearch-autotraining.md"],
        role_citations={
            "research": "docs/design/autoresearch-autotraining.md",
            "prior_result": "docs/design/autoresearch-autotraining.md",
        },
        train_version="wf_smoke_v2",
        eval_version="e938_role_safe_all_targets_v2",
        steps=22,
        cycle=1712,
        role="screening",
        recommended_slug="batch1",
    )
    applied = _mod._apply_frozen_replay(
        confirmation_matrix, confirmation_replay, new_campaign
    )
    assert confirmation_matrix["recommended_experiment_id"].endswith("-confirm")
    confirmed = next(
        item["experiment"]
        for item in confirmation_matrix["hypotheses"]
        if item["experiment"]["experiment_id"]
        == confirmation_matrix["recommended_experiment_id"]
    )
    assert confirmed["knobs"] == confirm_experiment["knobs"]
    assert confirmation_matrix["recommended_experiment_id"] in applied


def test_frozen_replay_restores_omitted_formal_claim_from_proved_artifact(
    tmp_path: Path,
) -> None:
    campaign_id = "continuous-loop-20260801-loop-12345678-c1714"
    experiment_id = "c20260801-loop-12345678-c1714-promote"
    experiment = {
        "experiment_id": experiment_id,
        "campaign_id": campaign_id,
        "hypothesis": "Replay a promotion candidate with a governed proof.",
        "rationale": "The historic replay omitted the experiment claim.",
        "expected_effect": "The proof-bound frozen candidate reaches evaluation.",
        "falsification_criteria": ["Formal claim recovery fails closed."],
        "stop_conditions": ["Stop after the bounded evaluation."],
        "citations": ["fixture://formal-replay"],
        "knobs": {"steps": 1, "batch_size": 1, "seed": 7},
        "formal_claims": [],
    }
    preflight_sha = "e" * 64
    manifest = _mod._manifest(
        campaign_id,
        experiment,
        "a" * 40,
        role="promotion",
        cycle_intent="promote",
        formal_preflight_sha256=preflight_sha,
    )
    obligation = manifest.formal_obligations[0]
    manifest = manifest.model_copy(
        update={
            "formal_obligations": (
                obligation.model_copy(update={"obligation_id": "formal-" + "0" * 16}),
            )
        }
    )
    claim = _mod.FormalClaimV1(**_mod.promote_formal_claim_dict())
    current_obligation_id = _mod.formal_obligation_id(campaign_id, experiment_id, claim)
    preflight = _mod.FormalPreflightV1(
        campaign_id=campaign_id,
        experiment_id=experiment_id,
        obligation_id=current_obligation_id,
        template_id=claim.template_id,
        template_version="v1",
        claim=claim.claim,
        policy=claim.policy,
        status="proved",
        evidence_scope="universal",
        theorem="structuralSimilarity_monotone",
        proof_target="Structural similarity monotonicity",
        source_digests={"Main.lean": "1" * 64},
        proof_sha256="2" * 64,
        lean_version="v4.20.0",
        mathlib_version="fixture",
        build_output_sha256="3" * 64,
        duration_seconds=0.1,
    )
    camp_dir = tmp_path / campaign_id
    artifact = camp_dir / "artifacts" / "formal_preflights" / f"{preflight_sha}.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(preflight.model_dump_json(indent=2) + "\n")

    _mod._restore_frozen_formal_claims(camp_dir, experiment, manifest)

    assert experiment["formal_claims"] == [claim.model_dump()]


def test_frozen_replay_finds_completed_train_across_retry_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    source_campaign = "cycle-1"
    initialized_only_campaign = "cycle-2"
    retry_campaign = "cycle-3"
    source_dir = root / source_campaign
    retry_dir = root / retry_campaign
    source_experiment = {
        "experiment_id": "source-batch1",
        "campaign_id": source_campaign,
        "hypothesis": "A completed source train can be reused.",
        "rationale": "Frozen replay preserves the exact training configuration.",
        "expected_effect": "The successor starts at evaluation.",
        "falsification_criteria": ["The manifest lineage does not verify."],
        "stop_conditions": ["Stop after the bounded evaluation."],
        "citations": ["fixture://source"],
        "knobs": {"steps": 1, "batch_size": 1, "seed": 7},
    }
    source_manifest = _mod._manifest(source_campaign, source_experiment, "a" * 40)
    source_path = source_dir / "manifests" / "source-batch1.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source_manifest.model_dump_json(indent=2) + "\n")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    retry_manifest = _mod._replay_successor_manifest(
        source_manifest,
        frozen_manifest_sha256=source_sha,
        campaign_id=retry_campaign,
        experiment_id="retry-batch1",
        integration_commit="b" * 40,
    )
    retry_path = retry_dir / "manifests" / "retry-batch1.json"
    retry_path.parent.mkdir(parents=True)
    retry_path.write_text(retry_manifest.model_dump_json(indent=2) + "\n")
    initialized_only_dir = root / initialized_only_campaign
    initialized_only_dir.mkdir(parents=True)
    (initialized_only_dir / "campaign.json").write_text(
        _mod.CampaignSpec(
            campaign_id=initialized_only_campaign,
            objective="initialized recovery gap",
            primary_metric="smoke.parse_rate",
            loop_id="loop-1",
            cycle_index=2,
            predecessor_campaign_id=source_campaign,
            upstream_commit="b" * 40,
            integration_commit="b" * 40,
        ).model_dump_json(indent=2)
        + "\n"
    )
    (retry_dir / "campaign.json").write_text(
        _mod.CampaignSpec(
            campaign_id=retry_campaign,
            objective="fixture replay",
            primary_metric="smoke.parse_rate",
            loop_id="loop-1",
            cycle_index=3,
            predecessor_campaign_id=initialized_only_campaign,
            upstream_commit="b" * 40,
            integration_commit="b" * 40,
        ).model_dump_json(indent=2)
        + "\n"
    )
    checkpoint = source_dir / "runs" / "source-batch1" / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint.parents[1] / "train_summary.json").write_text(
        json.dumps(
            {
                "run_id": "source-batch1",
                "stopped_on": "steps",
                "steps": 1,
                "checkpoint": str(checkpoint),
            }
        )
    )

    reuse = _mod._completed_frozen_train_source(
        root=root,
        campaign_dir=retry_dir,
        manifest=retry_manifest,
        manifest_path=retry_path,
    )

    assert reuse is not None
    assert reuse["run_dir"] == source_dir / "runs" / "source-batch1"
    assert reuse["manifest_paths"] == (retry_path, source_path)


def test_frozen_train_reuse_keeps_completed_checkpoint_when_eval_timed_out(
    tmp_path: Path,
) -> None:
    """Training reuse skips only training; the successor still reruns evaluation."""

    root = tmp_path / "autoresearch"
    source_campaign = "cycle-timeout"
    source_dir = root / source_campaign
    source_experiment = {
        "experiment_id": "source-control",
        "campaign_id": source_campaign,
        "hypothesis": "Timeout incomplete control must force a fresh eval.",
        "rationale": "Reuse would re-run the same timed-out measurement.",
        "expected_effect": "Successor trains and evaluates without frozen reuse.",
        "falsification_criteria": ["Train reuse is offered despite decode timeouts."],
        "stop_conditions": ["Stop after the bounded evaluation."],
        "citations": ["fixture://timeout-source"],
        "knobs": {"steps": 1, "batch_size": 1, "seed": 7},
    }
    source_manifest = _mod._manifest(source_campaign, source_experiment, "a" * 40)
    source_path = source_dir / "manifests" / "source-control.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source_manifest.model_dump_json(indent=2) + "\n")
    checkpoint = source_dir / "runs" / "source-control" / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint.parents[1] / "train_summary.json").write_text(
        json.dumps(
            {
                "run_id": "source-control",
                "stopped_on": "steps",
                "steps": 1,
                "checkpoint": str(checkpoint),
            }
        )
    )
    (checkpoint.parents[1] / "eval_smoke.json").write_text(
        json.dumps(
            {
                "decode_timeout_count": 3,
                "document_n": 3,
                "completed_document_n": 0,
                "incomplete_document_n": 3,
            }
        )
    )

    reuse = _mod._completed_frozen_train_source(
        root=root,
        campaign_dir=source_dir,
        manifest=source_manifest,
        manifest_path=source_path,
    )

    assert reuse == {
        "run_dir": checkpoint.parents[1],
        "manifest_paths": (source_path,),
    }


def test_digestless_frozen_retry_does_not_stall_cycle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "autoresearch"
    campaign_id = "cycle-1"
    handoff = _mod.AutotrainCycleHandoffV1(
        loop_id="loop-1",
        campaign_id=campaign_id,
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        cycle_role="screening",
        cycle_intent="screening",
        evidence_class="fixture",
        climb_state="inconclusive",
        ship_state="blocked",
        primary_metric="smoke.parse_rate",
        actions=(
            _mod.AutotrainActionV1(
                kind="retry_measurement",
                owner="autotrain",
                reason="measurement incomplete but manifest was not written",
                evidence_ids=(f"campaign:{campaign_id}",),
            ),
        ),
    )
    path = root / campaign_id / "cycle_handoff.json"
    path.parent.mkdir(parents=True)
    path.write_text(handoff.model_dump_json(indent=2) + "\n")

    assert _mod._load_frozen_replay(root, "loop-1", campaign_id) is None
    assert (
        "FROZEN_REPLAY_SKIP reason=missing_frozen_manifest_sha256"
        in capsys.readouterr().out
    )


def test_invalid_frozen_configuration_is_not_replayed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "autoresearch"
    campaign_id = "cycle-invalid"
    camp = root / campaign_id
    experiment = {
        "experiment_id": "candidate-invalid",
        "campaign_id": campaign_id,
        "hypothesis": "A compiler-path lever requires its decode companion.",
        "rationale": "The exact frozen recipe is intentionally invalid.",
        "expected_effect": "Proposal validation prevents execution.",
        "falsification_criteria": ["The invalid arm reaches training."],
        "stop_conditions": ["Stop at capability validation."],
        "citations": ["fixture://invalid"],
        "knobs": {
            "steps": 1,
            "binder_topology_decode_weight": 1.0,
            "compiler_decode_mode": "off",
        },
    }
    manifest = _mod._manifest(campaign_id, experiment, "a" * 40)
    manifest_path = camp / "manifests" / "candidate-invalid.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    handoff = _mod.AutotrainCycleHandoffV1(
        loop_id="loop-1",
        campaign_id=campaign_id,
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        cycle_role="screening",
        cycle_intent="screening",
        evidence_class="fixture",
        climb_state="harness_failure",
        ship_state="blocked",
        primary_metric="smoke.parse_rate",
        actions=(
            _mod.AutotrainActionV1(
                kind="retry_measurement",
                owner="autotrain",
                reason="retry the frozen incomplete arm",
                evidence_ids=(f"campaign:{campaign_id}",),
                frozen_manifest_sha256=digest,
            ),
        ),
    )
    (camp / "cycle_handoff.json").write_text(handoff.model_dump_json(indent=2) + "\n")
    outcomes = camp / "artifacts" / "outcomes"
    outcomes.mkdir(parents=True)
    (outcomes / "candidate.json").write_text(
        json.dumps(
            {
                "experiment_id": "candidate-invalid",
                "status": "failed",
                "metrics": {},
                "error": "lever_capability_compatibility: unsupported enabled levers",
            }
        ),
        encoding="utf-8",
    )

    assert _mod._load_frozen_replay(root, "loop-1", campaign_id) is None
    output = capsys.readouterr().out
    assert "FROZEN_REPLAY_SKIP reason=nonreplayable_configuration" in output
    assert "detail=lever_capability_compatibility" in output


def test_reserved_runtime_owner_failure_is_not_replayed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "autoresearch"
    campaign_id = "cycle-runtime-owner"
    camp = root / campaign_id
    experiment = {
        "experiment_id": "candidate-runtime-owner",
        "campaign_id": campaign_id,
        "hypothesis": "A reserved owner must not be replayed before implementation.",
        "rationale": "The exact frozen recipe reached the model constructor.",
        "expected_effect": "The owner is either implemented or skipped fail-closed.",
        "falsification_criteria": ["The reserved owner silently runs."],
        "stop_conditions": ["Stop at runtime-owner validation."],
        "citations": ["fixture://runtime-owner"],
        "knobs": {"steps": 1, "binder_slot_ownership_decode_weight": 1.0},
    }
    manifest = _mod._manifest(campaign_id, experiment, "a" * 40)
    manifest_path = camp / "manifests" / "candidate-runtime-owner.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    handoff = _mod.AutotrainCycleHandoffV1(
        loop_id="loop-1",
        campaign_id=campaign_id,
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        cycle_role="screening",
        cycle_intent="screening",
        evidence_class="fixture",
        climb_state="harness_failure",
        ship_state="blocked",
        primary_metric="smoke.parse_rate",
        actions=(
            _mod.AutotrainActionV1(
                kind="retry_measurement",
                owner="autotrain",
                reason="retry",
                evidence_ids=(f"campaign:{campaign_id}",),
                frozen_manifest_sha256=digest,
            ),
        ),
    )
    (camp / "cycle_handoff.json").write_text(handoff.model_dump_json(indent=2) + "\n")
    outcomes = camp / "artifacts" / "outcomes"
    outcomes.mkdir(parents=True)
    (outcomes / "candidate.json").write_text(
        json.dumps(
            {
                "experiment_id": "candidate-runtime-owner",
                "status": "failed",
                "metrics": {},
                "error": "unsupported compiler auxiliary lever(s); no runtime owner is implemented",
            }
        ),
        encoding="utf-8",
    )

    assert _mod._load_frozen_replay(root, "loop-1", campaign_id) is None
    assert "nonreplayable_configuration" in capsys.readouterr().out


def test_measurement_completion_requires_both_arm_metrics_and_no_soft_failure() -> None:
    complete = {
        "control_metrics": {"parse_rate": 1.0},
        "candidate_metrics": {"parse_rate": 1.0},
        "reasons": ["primary_metric_null_or_worse:smoke.parse_rate"],
    }
    assert _mod._measurement_is_complete(complete)
    assert not _mod._measurement_is_complete(
        {**complete, "reasons": ["measurement_incomplete:no_smoke_metrics"]}
    )
    assert not _mod._measurement_is_complete(
        {**complete, "candidate_metrics": {"parse_rate": None}}
    )


def test_continuous_evidence_is_bounded_to_predecessor_and_loop(tmp_path: Path) -> None:
    roots = _mod._continuous_evidence_roots(
        tmp_path / "autoresearch", "loop-1", "campaign-6"
    )
    assert roots == (
        tmp_path / "autoresearch" / "campaign-6" / "cycle_handoff.json",
        tmp_path / "autoresearch" / "campaign-6" / "measured-results-continuous.md",
        tmp_path / "autoresearch" / "campaign-6" / "run_insights.json",
        tmp_path
        / "autoresearch"
        / "campaign-6"
        / "artifacts"
        / "hypothesizer_feedback",
        tmp_path / "autoresearch" / "loops" / "loop-1" / "hillclimb_iterations.jsonl",
        tmp_path / "autoresearch" / "loops" / "loop-1" / "thrash_timing.jsonl",
        tmp_path / "autoresearch" / "loops" / "loop-1" / "exhausted_knob_ledger.json",
        tmp_path / "autoresearch" / "loops" / "loop-1" / "champion_queue.jsonl",
        tmp_path / "autoresearch" / "sdlc_delivery_ledger.jsonl",
    )


def test_predecessor_prerequisite_must_be_acknowledged(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    campaign_id = "cycle-1"
    handoff = _mod.AutotrainCycleHandoffV1(
        loop_id="loop-1",
        campaign_id=campaign_id,
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        cycle_role="screening",
        cycle_intent="screening",
        evidence_class="fixture",
        climb_state="rejected",
        ship_state="blocked",
        primary_metric="smoke.parse_rate",
        actions=(
            _mod.AutotrainActionV1(
                kind="document",
                owner="documenting-experiment-results",
                reason="Persist the result before advancing.",
                evidence_ids=(f"campaign:{campaign_id}",),
            ),
        ),
    )
    path = root / campaign_id / "cycle_handoff.json"
    path.parent.mkdir(parents=True)
    path.write_text(handoff.model_dump_json(indent=2) + "\n")

    with pytest.raises(RuntimeError, match="0:document"):
        _mod._require_predecessor_actions(root, "loop-1", campaign_id)
    _mod._require_predecessor_actions(root, "loop-1", "historical-without-handoff")


def test_cycle_handoff_marks_incomplete_measurement_for_frozen_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-1"
    (camp / "manifests").mkdir(parents=True)
    (camp / "manifests" / "cand.json").write_text("{}\n")
    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="screening",
        primary_metric="smoke.binder_reference_f1",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "candidate_id": "cand",
            "reasons": ["measurement_incomplete:no_smoke_metrics"],
            "stack_layer": False,
        },
        resolution=None,
        formal_status=None,
    )
    assert handoff.climb_state == "inconclusive"
    retry = next(
        action for action in handoff.actions if action.kind == "retry_measurement"
    )
    assert retry.frozen_manifest_sha256 == hashlib.sha256(b"{}\n").hexdigest()
    assert all(action.kind != "next_experiment" for action in handoff.actions)


def test_cycle_handoff_uses_delivery_completeness_for_wall_timeout_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-1"
    (camp / "manifests").mkdir(parents=True)
    (camp / "manifests" / "cand.json").write_text("{}\n")

    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="screening",
        primary_metric="smoke.structural_similarity",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "candidate_id": "cand",
            "measurement_complete": False,
            "reasons": ["wall_timeout:control", "primary_metric_unavailable"],
            "stack_layer": False,
        },
        resolution=None,
        formal_status=None,
    )

    assert handoff.climb_state == "inconclusive"
    assert handoff.priorities[0].area == "infrastructure"
    assert handoff.priorities[0].proposed_experiment_id == "cand"
    assert "exact frozen" in handoff.priorities[0].hypothesis
    assert any(action.kind == "retry_measurement" for action in handoff.actions)
    assert all(action.kind != "next_experiment" for action in handoff.actions)


def test_finalized_decode_timeout_routes_directly_to_runtime_repair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-1"
    run = camp / "runs" / "cand"
    run.mkdir(parents=True)
    (camp / "manifests").mkdir()
    (camp / "manifests" / "cand.json").write_text("{}\n")
    (run / "scoreboard.json").write_text(
        json.dumps(
            {
                "evals": {"runner": {"name": "AgentV", "execution_errors": 0}},
                "gates": {"authority": "AgentEvals assertions", "pass": False},
                "suites": {
                    "smoke": {
                        "n": 3,
                        "completed_document_n": 2,
                        "incomplete_document_n": 1,
                        "decode_timeout_document_count": 1,
                    }
                },
            }
        )
    )

    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="screening",
        primary_metric="smoke.binder_reference_f1",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "candidate_id": "cand",
            "reasons": [
                "measurement_incomplete:decode_timeout",
                "harness_failure:cand:experiment_failed",
            ],
            "stack_layer": False,
        },
        resolution=None,
        formal_status=None,
    )

    # Screening thrash decode/wall timeout residual is soft: emit
    # next_experiment (not hard repair_harness) so continuous thrash continues.
    assert all(action.kind != "repair_harness" for action in handoff.actions)
    nxt = next(a for a in handoff.actions if a.kind == "next_experiment")
    assert "timeout" in (nxt.reason or "").lower()
    assert any(action.kind == "document" for action in handoff.actions)


def test_replayed_finalized_decode_timeout_rejects_runtime_arm(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-2"
    run = camp / "runs" / "cand"
    run.mkdir(parents=True)
    (camp / "manifests").mkdir()
    (camp / "campaign.json").write_text(
        json.dumps({"predecessor_campaign_id": "cycle-1"})
    )
    (camp / "manifests" / "cand.json").write_text("{}\n")
    (run / "scoreboard.json").write_text(
        json.dumps(
            {
                "evals": {"runner": {"name": "AgentV", "execution_errors": 0}},
                "gates": {"authority": "AgentEvals assertions", "pass": False},
                "suites": {
                    "smoke": {
                        "n": 3,
                        "completed_document_n": 0,
                        "incomplete_document_n": 3,
                        "decode_timeout_document_count": 3,
                    }
                },
            }
        )
    )

    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-2",
        cycle_index=2,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="retry_measurement",
        primary_metric="smoke.binder_reference_f1",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "candidate_id": "cand",
            "reasons": [
                "measurement_incomplete:decode_timeout",
                "harness_failure:cand:experiment_failed",
            ],
            "stack_layer": False,
        },
        resolution=None,
        formal_status=None,
    )

    assert handoff.climb_state == "rejected"
    assert any("candidate_runtime_rejected" in reason for reason in handoff.reasons)
    assert all(action.kind != "repair_harness" for action in handoff.actions)
    assert all(action.kind != "retry_measurement" for action in handoff.actions)
    assert any(action.kind == "next_experiment" for action in handoff.actions)


def test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-2"
    (camp / "manifests").mkdir(parents=True)
    (camp / "campaign.json").write_text(
        json.dumps({"predecessor_campaign_id": "cycle-1"})
    )
    (camp / "manifests" / "cand.json").write_text("{}\n")
    for experiment_id in ("control", "cand"):
        run = camp / "runs" / experiment_id
        run.mkdir(parents=True)
        (run / "scoreboard.json").write_text(
            json.dumps(
                {
                    "evals": {"runner": {"name": "AgentV", "execution_errors": 0}},
                    "gates": {"authority": "AgentEvals assertions", "pass": False},
                    "suites": {
                        "smoke": {
                            "n": 3,
                            "completed_document_n": 2,
                            "incomplete_document_n": 1,
                            "decode_timeout_document_count": 1,
                        }
                    },
                }
            )
        )

    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-2",
        cycle_index=2,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="promotion",
        cycle_intent="retry_measurement",
        primary_metric="held_out.structural_similarity",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "control_id": "control",
            "candidate_id": "cand",
            "measurement_complete": False,
            "reasons": [
                "measurement_incomplete:control:decode_timeout",
                "measurement_incomplete:cand:decode_timeout",
            ],
            "stack_layer": False,
        },
        resolution=None,
        formal_status=None,
    )

    assert handoff.climb_state == "inconclusive"
    assert all("candidate_runtime_rejected" not in reason for reason in handoff.reasons)
    # Dual-arm thrash decode timeouts stay soft (next_experiment), not hard repair.
    assert all(action.kind != "repair_harness" for action in handoff.actions)
    assert any(action.kind == "next_experiment" for action in handoff.actions)


def test_numeric_literal_close_starvation_steers_new_training_arm(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-1"
    run = camp / "runs" / "cand"
    run.mkdir(parents=True)
    (camp / "manifests").mkdir()
    (camp / "manifests" / "cand.json").write_text("{}\n")
    traces = [
        {
            "record_id": "smoke-1",
            "prefix_text": f'root = Slider("$6", "discrete", {"1" * n}',
            "chosen_token": "B:31",
            "legal_candidates": 12,
        }
        for n in range(6, 10)
    ]
    (run / "eval_smoke.json").write_text(
        json.dumps(
            {
                "decode_timeout_document_count": 1,
                "decode_stats": {"constrained_selection_traces": traces},
            }
        )
    )

    assert _mod._has_numeric_literal_close_starvation(camp, "cand")
    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="promotion",
        cycle_intent="retry_measurement",
        primary_metric="held_out.structural_similarity",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "candidate_id": "cand",
            "measurement_complete": False,
            "reasons": ["measurement_incomplete:decode_timeout"],
            "stack_layer": False,
        },
        resolution=None,
        formal_status=None,
    )

    assert handoff.priorities[0].area == "model_build"
    assert handoff.priorities[0].proposed_experiment_id == "cand-literal-close"
    successor = next(
        action for action in handoff.actions if action.kind == "next_experiment"
    )
    assert successor.owner == "autotrain"
    assert "registered typed tail-weighted LTR signal" in successor.reason
    assert "do not replay" in successor.reason
    assert all(action.kind != "repair_harness" for action in handoff.actions)
    assert _mod._predecessor_priority_slug(root, "cycle-1", skip=set()) == (
        "literal-close"
    )
    assert (
        _mod._predecessor_priority_slug(root, "cycle-1", skip={"literal-close"})
        == "literal-close"
    )

    predecessor = "cycle-1"
    for index, slug in ((2, "bounds"), (3, "confirm")):
        successor = root / f"cycle-{index}"
        successor.mkdir()
        (successor / "campaign.json").write_text(
            json.dumps({"predecessor_campaign_id": predecessor})
        )
        (successor / "sdlc_delivery.json").write_text(
            json.dumps({"candidate_id": f"cand-{slug}"})
        )
        (successor / "cycle_handoff.json").write_text(
            json.dumps(
                {
                    "priorities": [
                        {
                            "rank": 1,
                            "area": "experiments",
                            "authority": "speculative",
                            "confidence": 0.6,
                            "disposition": "experiment_next",
                            "proposed_experiment_id": f"cand-{slug}",
                        }
                    ]
                }
            )
        )
        predecessor = f"cycle-{index}"

    assert (
        _mod._predecessor_priority_slug(root, predecessor, skip={"literal-close"})
        == "literal-close"
    )
    assert (
        _mod._predecessor_priority_slug(
            root,
            predecessor,
            skip={"literal-close"},
            closed={"literal-close"},
        )
        is None
    )


def test_control_only_model_timeout_replays_without_fake_harness_repair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-1"
    control = camp / "runs" / "control"
    candidate = camp / "runs" / "literal-close"
    control.mkdir(parents=True)
    candidate.mkdir(parents=True)
    (camp / "manifests").mkdir()
    (camp / "manifests" / "literal-close.json").write_text("{}\n")
    traces = [
        {
            "record_id": "smoke-1",
            "prefix_text": f'root = Slider("$6", "discrete", {"1" * n}',
            "chosen_token": "B:31",
            "legal_candidates": 12,
        }
        for n in range(6, 10)
    ]
    (control / "eval_smoke.json").write_text(
        json.dumps(
            {
                "n": 1,
                "completed_document_n": 0,
                "incomplete_document_n": 1,
                "decode_timeout_document_count": 1,
                "decode_stats": {"constrained_selection_traces": traces},
            }
        )
    )
    (control / "scoreboard.json").write_text(
        json.dumps(
            {
                "evals": {
                    "runner": {
                        "name": "AgentV",
                        "execution_errors": 0,
                    }
                },
                "gates": {"authority": "AgentEvals assertions", "pass": False},
                "suites": {
                    "smoke": {
                        "n": 1,
                        "completed_document_n": 0,
                        "incomplete_document_n": 1,
                        "decode_timeout_document_count": 1,
                    }
                },
            }
        )
    )
    (candidate / "eval_smoke.json").write_text(
        json.dumps(
            {
                "n": 1,
                "completed_document_n": 1,
                "incomplete_document_n": 0,
                "decode_timeout_document_count": 0,
            }
        )
    )

    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="screening",
        primary_metric="smoke.structural_similarity",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "control_id": "control",
            "candidate_id": "literal-close",
            "measurement_complete": False,
            "reasons": ["harness_failure:control:experiment_failed"],
        },
        resolution=None,
        formal_status=None,
    )

    assert any(action.kind == "retry_measurement" for action in handoff.actions)
    assert all(action.kind != "repair_harness" for action in handoff.actions)
    assert "tail-supervised candidate completed" in handoff.priorities[0].hypothesis

    replay_handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="promotion",
        cycle_intent="retry_measurement",
        primary_metric="held_out.structural_similarity",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "control_id": "control",
            "candidate_id": "literal-close",
            "measurement_complete": False,
            "reasons": ["measurement_incomplete:literal-close:missing_scoreboard"],
        },
        resolution=None,
        formal_status="proved",
    )
    assert replay_handoff.climb_state == "inconclusive"
    assert not any(
        reason.startswith("candidate_runtime_unblock_reproduced:")
        for reason in replay_handoff.reasons
    )
    assert any(action.kind == "retry_measurement" for action in replay_handoff.actions)


def test_cycle_handoff_exhausts_identical_replays_into_harness_repair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    prior = root / "cycle-1"
    current = root / "cycle-2"
    (current / "manifests").mkdir(parents=True)
    prior.mkdir(parents=True)
    (current / "manifests" / "cand.json").write_text("{}\n")
    (prior / "campaign.json").write_text(json.dumps({"predecessor_campaign_id": None}))
    (prior / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "loop_id": "loop-1",
                "campaign_id": "cycle-1",
                "cycle_intent": "promotion",
            }
        )
    )
    (current / "campaign.json").write_text(
        json.dumps({"predecessor_campaign_id": "cycle-1"})
    )

    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-1",
        campaign_id="cycle-2",
        cycle_index=2,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        role="screening",
        cycle_intent="retry_measurement",
        primary_metric="smoke.binder_reference_f1",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "candidate_id": "cand",
            "reasons": ["measurement_incomplete:no_smoke_metrics"],
            "stack_layer": False,
        },
        resolution=None,
        formal_status=None,
    )

    repair = next(
        action for action in handoff.actions if action.kind == "repair_harness"
    )
    assert repair.owner == "improve-openui-harnesses"
    assert repair.frozen_manifest_sha256 == hashlib.sha256(b"{}\n").hexdigest()
    assert "(1/1)" in repair.reason
    retry = next(
        action for action in handoff.actions if action.kind == "retry_measurement"
    )
    assert retry.frozen_manifest_sha256 == repair.frozen_manifest_sha256
    assert handoff.actions.index(repair) < handoff.actions.index(retry)

    prior_handoff = json.loads((prior / "cycle_handoff.json").read_text())
    prior_handoff["actions"] = [repair.model_dump(mode="json")]
    (prior / "cycle_handoff.json").write_text(json.dumps(prior_handoff))
    assert (
        _mod._consecutive_frozen_replays(root, "loop-1", "cycle-2", "retry_measurement")
        == 1
    )


def test_causal_family_saturates_per_integrated_code() -> None:
    entries = [
        {
            "status": "rejected",
            "source_integration_commit": "a" * 40,
            "knobs": {"grammar_completion_bounds": True},
        },
        {
            "status": "promotion_failed",
            "source_integration_commit": "a" * 40,
            "knobs": {"grammar_completion_bounds": True},
        },
    ]
    assert "bounds" in _mod._skip_arm_slugs(entries, integration_commit="a" * 40)
    assert "bounds" not in _mod._skip_arm_slugs(entries, integration_commit="b" * 40)


def test_dispose_champion_promote_formal_timeout_is_inconclusive_not_failed() -> None:
    d = _mod.dispose_champion_promote(
        formal_preflight_status="timed_out",
        certificate=None,
        locked_expectations_sha256="a" * 64,
        phase_a_positive=True,
        phase_a_quality_held=True,
    )
    assert d["status"] == "promotion_inconclusive"
    assert d.get("timeout") is True
    assert d.get("inconclusive") is True
    assert any("formal_preflight_timed_out" in r for r in d["reasons"])
    assert any("measurement_incomplete" in r for r in d["reasons"])
    assert d["status"] != "promotion_failed"
    assert d["status"] != "rejected"


def test_resolve_promotion_formal_timeout_refunds_attempt_and_stays_retriable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ar"
    loop = "L"
    camp = root / "camp-to"
    camp.mkdir(parents=True)
    entry = {
        "entry_id": "champ-to-1",
        "status": "promoting",
        "knobs_fingerprint": "fp1",
        "promote_attempts": 1,
        "knobs": {"grammar_completion_bounds": True},
    }
    path = _mod._champion_queue_path(root, loop)
    _mod._write_champion_queue(path, [entry])
    _mod.record_formal_preflight_status(
        camp,
        status="timed_out",
        template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID,
        reason="formal_preflight_timed_out:wall_s=600",
        timeout_seconds=600.0,
        duration_seconds=600.1,
        timed_out=True,
    )
    resolved = _mod._resolve_promotion_result(
        root=root,
        loop_id=loop,
        entry=entry,
        delivery={
            "positive": False,
            "reasons": ["fixture_insufficient_n:arm"],
        },
        campaign_id="camp-to",
        cycle_index=9,
        camp_dir=camp,
        formal_preflight_status="timed_out",
        locked_expectations_sha256="b" * 64,
    )
    assert resolved is not None
    assert resolved["status"] == "promotion_inconclusive"
    rows = _mod._load_champion_queue(path)
    assert rows[0]["status"] == "promotion_inconclusive"
    # Attempt refunded so timeout is not a permanent rejection path.
    assert int(rows[0].get("promote_attempts") or 0) == 0
    assert rows[0].get("last_formal_timeout") is True
    # Retriable head for next promotion cadence.
    head = _mod._queue_head_confirmed(rows)
    assert head is not None
    assert head["entry_id"] == "champ-to-1"
    # Ledger captures inconclusive outcome (not promotion_failed).
    ledger = root / "loops" / loop / "learning_certificate_ledger.jsonl"
    line = ledger.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert "promotion_inconclusive" in line
    assert "timed_out" in line


def test_fit_screening_decode_fits_arm_wall() -> None:
    """n×decode + train floor must not exceed the symmetric arm wall."""
    from slm_training.autoresearch.climb_policy import load_climb_policy

    policy = load_climb_policy()
    fitted, meta = _mod._fit_screening_decode_timeout_seconds(policy)
    arm = float(meta["arm_wall_seconds"])
    n = int(meta["smoke_n"])
    train = float(meta["min_train_floor_seconds"])
    overhead = float(meta["eval_overhead_seconds"])
    assert fitted * n + train + overhead <= arm + 1e-6
    assert fitted <= 12.0  # thrash-calibrated, not ship 24s


def test_fit_screening_decode_carries_certified_sample_size_report() -> None:
    """Auto policy mode embeds the screening_sample_size/v1 verdict in meta."""
    from slm_training.autoresearch.climb_policy import load_climb_policy

    policy = load_climb_policy()
    fitted, meta = _mod._fit_screening_decode_timeout_seconds(policy)
    report = meta["screening_sample_size"]
    assert report is not None
    assert report["schema_version"] == "screening_sample_size/v1"
    assert report["decidability_floor_n"] == 6  # exact sign-test floor, alpha=1/20
    assert report["promotion_authority"] is False
    if report["verdict"] == "feasible":
        assert int(meta["smoke_n"]) >= 6
        assert report["must_generate"] is False
    else:
        assert report["verdict"] == "infeasible_range_empty"
        assert "suite_volume" in report["binding_constraints"]
        assert report["must_generate"] is True
        assert int(meta["smoke_n"]) == 0


def test_screening_matrix_uses_fitted_decode_and_thrash_steps() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-timing-c1",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=80,
        cycle=4,
        role="screening",
        recommended_slug="bounds",
    )
    cand = next(
        h["experiment"]
        for h in matrix["hypotheses"]
        if str(h["experiment"]["experiment_id"]).endswith("-bounds")
    )
    knobs = cand["knobs"]
    assert float(knobs["decode_timeout_seconds"]) <= 12.0  # policy.v1.json v5
    # Floor-fit steps (cold-start 5 sps × 0.9 × grown floor) + cycle%3.
    assert int(knobs["steps"]) >= 50
    assert int(knobs["steps"]) <= 403


def test_write_thrash_timing_records_completeness(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "c-time"
    camp.mkdir(parents=True)
    path = _mod._write_thrash_timing(
        camp,
        loop_id="loop-t",
        campaign_id="c-time",
        cycle_index=9,
        role="screening",
        measurement_complete=False,
        arm_wall_seconds=70.0,
        decode_fit={"fitted_decode_timeout_seconds": 8.0},
        reasons=["measurement_incomplete:x:missing_scoreboard", "empty_metrics:y"],
        control_metrics={"structural_similarity": None},
        candidate_metrics={},
    )
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "thrash_timing/v1"
    assert data["complete"] is False
    assert any("measurement_incomplete" in r for r in data["incomplete_reasons"])
    ledger = root / "loops" / "loop-t" / "thrash_timing.jsonl"
    assert ledger.is_file()


def test_screening_steps_fitter_telemetry_clamp_and_cold_start() -> None:
    fitted, ev = _mod._fit_screening_steps(
        floor_seconds=20.0,
        measured_steps_per_sec=6.5,
        steps_max=400,
    )
    assert fitted == min(400, int(20.0 * 6.5 * 0.9))  # 117
    assert ev["cold_start"] is False
    clamped, clamp_ev = _mod._fit_screening_steps(
        floor_seconds=20.0,
        measured_steps_per_sec=100.0,
        steps_max=40,
    )
    assert clamped == 40
    assert clamp_ev["steps_max"] == 40
    cold, cold_ev = _mod._fit_screening_steps(
        floor_seconds=20.0,
        measured_steps_per_sec=None,
        steps_max=400,
    )
    assert cold_ev["cold_start"] is True
    assert cold == min(400, int(20.0 * _mod._COLD_START_STEPS_PER_SEC * 0.9))
    policy = SimpleNamespace(
        measurement={"thrash_timing": {"min_train_floor_seconds": 20}}
    )
    assert _mod._screening_thrash_steps(policy, 22) == cold


def test_screening_steps_from_train_summary_telemetry(tmp_path: Path) -> None:
    camp = tmp_path / "continuous-loop-x" / "runs" / "arm"
    camp.mkdir(parents=True)
    (camp / "train_summary.json").write_text(
        json.dumps({"steps": 22, "elapsed_wall_seconds": 3.36, "device": "cpu"}),
        encoding="utf-8",
    )
    policy = SimpleNamespace(
        measurement={"thrash_timing": {"min_train_floor_seconds": 20}}
    )
    steps = _mod._screening_thrash_steps(
        policy, 22, telemetry_root=tmp_path, floor_seconds=20.0
    )
    assert steps == min(400, int(20.0 * (22 / 3.36) * 0.9))


def test_screening_steps_use_slower_arm_from_latest_complete_pair(
    tmp_path: Path,
) -> None:
    older = tmp_path / "continuous-loop-paired" / "runs"
    newer = tmp_path / "continuous-loop-incomplete" / "runs"
    for arm, wall in (("control", 20.0), ("candidate", 10.0)):
        run = older / arm
        run.mkdir(parents=True)
        (run / "train_summary.json").write_text(
            json.dumps({"steps": 100, "elapsed_wall_seconds": wall}),
            encoding="utf-8",
        )
    run = newer / "candidate"
    run.mkdir(parents=True)
    newest = run / "train_summary.json"
    newest.write_text(
        json.dumps({"steps": 100, "elapsed_wall_seconds": 5.0}),
        encoding="utf-8",
    )
    newest.touch()

    payload = _mod._latest_train_telemetry_payload(tmp_path)

    assert payload is not None
    assert str(payload["_telemetry_path"]).endswith("paired/runs/control/train_summary.json")
    assert len(payload["_telemetry_paths"]) == 2
    assert _mod._steps_per_sec_from_train_payload(payload) == 5.0


def test_write_thrash_timing_includes_steps_fit(tmp_path: Path) -> None:
    camp = tmp_path / "c-fit"
    camp.mkdir(parents=True)
    path = _mod._write_thrash_timing(
        camp,
        loop_id="loop-t",
        campaign_id="c-fit",
        cycle_index=1,
        role="screening",
        measurement_complete=True,
        arm_wall_seconds=70.0,
        decode_fit={
            "fitted_decode_timeout_seconds": 7.0,
            "fitted_steps": 117,
            "steps_fit": {"cold_start": False, "fitted_steps": 117},
            "train_device": "cpu",
        },
        reasons=[],
        control_metrics={"structural_similarity": 0.1},
        candidate_metrics={"structural_similarity": 0.2},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["fitted_steps"] == 117
    assert data["train_device"] == "cpu"
    assert data["steps_fit"]["fitted_steps"] == 117


def test_grown_train_floor_never_exceeds_arm_wall() -> None:
    from slm_training.autoresearch.climb_policy import load_climb_policy

    policy = load_climb_policy()
    fitted, meta = _mod._fit_screening_decode_timeout_seconds(
        policy, arm_wall_seconds=70.0
    )
    wall = float(meta["arm_wall_seconds"])
    floor = float(meta["grown_train_floor_seconds"])
    eval_s = float(meta["eval_budget_seconds"])
    overhead = float(meta["eval_overhead_seconds"])
    assert floor + eval_s + overhead <= wall + 1e-9
    assert eval_s == pytest.approx(fitted * float(meta["smoke_n"]))
    assert floor >= float(meta["min_train_floor_seconds"]) - 1e-9


def test_screening_cuda_device_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    assert _mod._screening_max_gpu_hours(role="screening", device="cpu") == 0.0
    assert _mod._screening_max_gpu_hours(role="screening", device="cuda") > 0.0
    assert _mod._screening_max_gpu_hours(role="confirm", device="cuda") == 0.0

    class _Boom:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("driver missing")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_Boom))
    assert _mod._screening_train_device() == "cpu"

    class _NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_NoCuda))
    assert _mod._screening_train_device() == "cpu"


def test_semantic_contrast_thrash_arm_uses_batch_size_at_least_3() -> None:
    matrix = _mod._matrix(
        campaign_id="continuous-loop-sc-c1",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=80,
        cycle=3,
        role="screening",
        recommended_slug="semantic-contrast",
    )
    cand = next(
        h["experiment"]
        for h in matrix["hypotheses"]
        if str(h["experiment"]["experiment_id"]).endswith("-semantic-contrast")
    )
    assert int(cand["knobs"]["batch_size"]) >= 3
    assert float(cand["knobs"]["semantic_contrast_loss_weight"]) > 0


def test_thrash_matrix_strips_stale_feedback_when_no_live_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent hypothesize fails if matrix.feedback_ids disagree with live feedback."""
    from slm_training.autoresearch.schemas import HypothesisMatrix

    matrix = _mod._matrix(
        campaign_id="continuous-loop-fb-c1",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=40,
        cycle=1,
        role="screening",
        recommended_slug="bounds",
        feedback=[{"feedback_id": "feedback-stale", "matrix_id": "old-m1"}],
        previous_matrix_id="old-m1",
    )
    assert matrix.get("feedback_ids") == ["feedback-stale"]
    # Driver thrash path always strips feedback binds (handoff pred ≠ lineage).
    promote_levers = None
    confirm_levers = None
    replay = None
    if (
        promote_levers is None
        and confirm_levers is None
        and replay is None
        and (matrix.get("feedback_ids") or matrix.get("predecessor_matrix_id"))
    ):
        matrix = dict(matrix)
        matrix.pop("feedback_ids", None)
        matrix.pop("predecessor_matrix_id", None)
    HypothesisMatrix.model_validate(matrix)
    assert "feedback_ids" not in matrix
    assert "predecessor_matrix_id" not in matrix


def test_thrash_does_not_bind_handoff_pred_feedback_ids(tmp_path: Path) -> None:
    """Thrash must not pin handoff-pred feedback; hypothesize rebinds lineage.

    Continuous pred is the last *handoff* campaign, while hypothesize walks
    full loop lineage and may select a different formed matrix (e.g. incomplete
    successor with partial feedback). Pinning handoff ids aborts thrash with
    'agent hypothesis matrix conflicts with supplied feedback ids'.
    """
    from slm_training.autoresearch.providers import AgentHypothesisProvider
    from slm_training.autoresearch.schemas import (
        CampaignSpec,
        EvidenceSnapshot,
        HypothesisFeedback,
        HypothesisMatrix,
    )

    # Thrash build path: no feedback passed (bind_pred_feedback=False).
    matrix = _mod._matrix(
        campaign_id="continuous-loop-fb-c2",
        evidence_snapshot_id="snap-c2",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=40,
        cycle=2,
        role="screening",
        recommended_slug="semantic-contrast",
        feedback=None,
        previous_matrix_id=None,
    )
    assert not matrix.get("feedback_ids")
    assert not matrix.get("predecessor_matrix_id")
    validated = HypothesisMatrix.model_validate(matrix)

    # Live lineage feedback differs from handoff-pred ids thrash used to bind.
    # Empty matrix feedback_ids lets AgentHypothesisProvider rebind safely.
    live = HypothesisFeedback(
        feedback_id="feedback-abcdef0123456789",
        campaign_id="lineage-camp",
        matrix_id="lineage-m1",
        experiment_id="exp-control",
        hypothesis="A sufficiently detailed lineage diagnosis hypothesis.",
        knob_signature='{"steps": 40}',
        outcome_status="stopped",
        diagnosis_target="infrastructure",
        diagnosis_evidence=("Lineage arm finished without dual-arm scoreboard.",),
        recommended_actions=("Rotate thrash under fitted decode budget.",),
    )
    path = tmp_path / "thrash-matrix.json"
    path.write_text(validated.model_dump_json(), encoding="utf-8")
    camp = CampaignSpec(
        campaign_id="continuous-loop-fb-c2",
        objective="Size-matched thrash under wall cap.",
        primary_metric="smoke.structural_similarity",
    )
    evidence = EvidenceSnapshot(snapshot_id="snap-c2", roots=("outputs",), items=())
    result = AgentHypothesisProvider(path).propose(camp, evidence, [], (live,))
    assert result.matrix.feedback_ids == (live.feedback_id,)
    assert result.matrix.predecessor_matrix_id == live.matrix_id


def test_matrix_climb_control_uses_champion_baseline() -> None:
    from slm_training.autoresearch.schemas import HypothesisMatrix
    from slm_training.autoresearch.thrash_regime import (
        REGIME_CLIMB,
        decide_screening_regime,
    )

    climb_knobs = {
        "binder_arity_loss_weight": 1.0,
        "binder_arity_decode_weight": 1.0,
        "compiler_decode_mode": "tree",
        "structural_aux_head_profile": "binder-arity",
    }
    regime = decide_screening_regime(
        climb_baseline_knobs=climb_knobs,
        compiler_ms_timeout=False,
    )
    assert regime.regime == REGIME_CLIMB
    matrix = _mod._matrix(
        campaign_id="continuous-loop-regime-climb-c1",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=5,
        role="screening",
        recommended_slug="bounds",
        thrash_regime=regime,
    )
    HypothesisMatrix.model_validate(matrix)
    by_id = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    control = by_id["cregime-climb-c1-control"]
    candidate = by_id["cregime-climb-c1-bounds"]
    assert matrix["thrash_regime"]["regime"] == "climb"
    assert "[climb]" in matrix["selection_rationale"]
    # Climb control carries sticky champion levers.
    assert control["binder_arity_loss_weight"] == 1.0
    assert control["compiler_decode_mode"] == "tree"
    assert control.get("grammar_completion_bounds") in (False, None)
    # Treatment = champion + residual decode cost lever.
    assert candidate["binder_arity_loss_weight"] == 1.0
    assert candidate["grammar_completion_bounds"] is True
    assert candidate["compiler_decode_mode"] == "tree"


def test_matrix_climb_skips_noop_residual_when_champion_already_has_bounds() -> None:
    """Climb baseline that already includes a residual bank arm must still validate.

    Control is signed from full knobs(); treatment must use the same materialization
    so no-op residuals (e.g. bounds on a bounds champion) are skipped rather than
    duplicating control signatures.
    """
    from slm_training.autoresearch.schemas import HypothesisMatrix
    from slm_training.autoresearch.thrash_regime import decide_screening_regime

    regime = decide_screening_regime(
        climb_baseline_knobs={
            "grammar_completion_bounds": True,
            "compact_active_canvas": True,
            "compiler_decode_mode": "tree",
            "binder_arity_loss_weight": 1.0,
            "binder_arity_decode_weight": 1.0,
        },
        compiler_ms_timeout=False,
    )
    matrix = _mod._matrix(
        campaign_id="continuous-loop-regime-climb-overlap-c1",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=11,
        role="screening",
        recommended_slug="bounds",
        thrash_regime=regime,
    )
    HypothesisMatrix.model_validate(matrix)
    ids = [h["experiment"]["experiment_id"] for h in matrix["hypotheses"]]
    assert not any(i.endswith("-bounds") for i in ids)
    assert not any(i.endswith("-canvas") for i in ids)
    # both = bounds+canvas is also a no-op on this baseline
    assert not any(i.endswith("-both") for i in ids)
    rec = matrix["recommended_experiment_id"]
    assert rec in ids
    assert not rec.endswith("-control")
    assert not rec.endswith("-bounds")
    by_id = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    control = by_id["cregime-climb-overlap-c1-control"]
    assert control["grammar_completion_bounds"] is True
    assert control["binder_arity_loss_weight"] == 1.0
    # Every treatment differs from control on at least one non-measurement lever.
    measurement = {
        "seed",
        "steps",
        "decode_timeout_seconds",
        "generate_batch_size",
        "eval_suites",
    }
    ctrl_view = {k: v for k, v in control.items() if k not in measurement}
    for eid, knobs in by_id.items():
        if eid.endswith("-control"):
            continue
        view = {k: v for k, v in knobs.items() if k not in measurement}
        assert view != ctrl_view, f"no-op treatment leaked: {eid}"


def test_matrix_isolate_control_zeroes_quality_levers() -> None:
    from slm_training.autoresearch.schemas import HypothesisMatrix

    matrix = _mod._matrix(
        campaign_id="continuous-loop-regime-isolate-c1",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=3,
        role="screening",
        recommended_slug="binder-arity",
    )
    HypothesisMatrix.model_validate(matrix)
    by_id = {
        row["experiment"]["experiment_id"]: row["experiment"]["knobs"]
        for row in matrix["hypotheses"]
    }
    control = by_id["cregime-isolate-c1-control"]
    candidate = by_id["cregime-isolate-c1-binder-arity"]
    assert matrix["thrash_regime"]["regime"] == "isolate"
    assert control["binder_arity_loss_weight"] == 0.0
    assert candidate["binder_arity_loss_weight"] == 1.0
    # Constrained decode preserved (no unconstrained production residual).
    assert candidate.get("allow_unconstrained_fallback") in (None, False)


def test_predecessor_compiler_ms_timeout_from_eval_detail(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    camp = root / "pred-c1"
    run = camp / "runs" / "pred-c1-control"
    run.mkdir(parents=True)
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "measurement_complete": False,
                "reasons": [
                    "measurement_incomplete:pred-c1-control:smoke:"
                    "incomplete_document_n=1:decode_timeout_count=1"
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "eval_smoke.json").write_text(
        json.dumps(
            {
                "decode_timeout_count": 1,
                "details": [
                    {
                        "id": "smoke_hero_01",
                        "incomplete": True,
                        "decode_outcome_detail": (
                            "timeout_dominant_phase=compiler_ms(14697ms/17334ms)"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _mod._predecessor_compiler_ms_timeout(root, "pred-c1") is True
    assert _mod._predecessor_compiler_ms_timeout(root, "missing") is False


def test_select_cycle_slug_timeout_outranks_quality_predecessor() -> None:
    from slm_training.autoresearch.thrash_regime import decide_screening_regime

    regime = decide_screening_regime(
        climb_baseline_knobs=None,
        compiler_ms_timeout=True,
    )
    slug = _mod._select_cycle_slug(
        1,
        predecessor_priority="binder-arity",
        skip=set(),
        has_confirm_levers=False,
        has_promote_levers=False,
        thrash_regime=regime,
    )
    assert slug == "bounds"
    assert slug != "binder-arity"


def test_write_cycle_handoff_records_thrash_regime(tmp_path: Path) -> None:
    """Handoff must carry thrash_regime when the screening matrix labeled one."""
    from slm_training.autoresearch.schemas import HypothesisMatrix

    root = tmp_path / "autoresearch"
    campaign_id = "continuous-loop-handoff-regime-c1"
    camp = root / campaign_id
    camp.mkdir(parents=True)
    commit = "a" * 40
    (camp / "campaign.json").write_text(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "loop_id": "loop-h",
                "cycle_index": 1,
                "upstream_commit": commit,
                "integration_commit": commit,
            }
        ),
        encoding="utf-8",
    )
    matrix = _mod._matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=20,
        cycle=1,
        role="screening",
        recommended_slug="canvas",
    )
    HypothesisMatrix.model_validate(matrix)
    assert matrix.get("thrash_regime", {}).get("regime") == "isolate"
    delivery = {
        "measurement_complete": True,
        "positive": False,
        "reasons": [],
        "candidate_id": matrix["recommended_experiment_id"],
        "control_id": f"{campaign_id.replace('continuous-loop-', 'c')}-control".replace(
            "continuous-loop-", "c"
        ),
    }
    # control id from matrix
    control_id = next(
        h["experiment"]["experiment_id"]
        for h in matrix["hypotheses"]
        if str(h["experiment"]["experiment_id"]).endswith("-control")
    )
    delivery["control_id"] = control_id
    (camp / "sdlc_delivery.json").write_text(json.dumps(delivery), encoding="utf-8")
    (camp / "matrix-proposal.json").write_text(json.dumps(matrix), encoding="utf-8")
    handoff = _mod._write_cycle_handoff(
        root=root,
        loop_id="loop-h",
        campaign_id=campaign_id,
        cycle_index=1,
        upstream_commit=commit,
        integration_commit=commit,
        role="screening",
        cycle_intent="screening",
        primary_metric="smoke.structural_similarity",
        matrix=matrix,
        delivery=delivery,
        resolution=None,
        formal_status=None,
    )
    assert handoff.thrash_regime is not None
    assert handoff.thrash_regime.get("regime") == "isolate"
    dumped = json.loads((camp / "cycle_handoff.json").read_text(encoding="utf-8"))
    assert dumped.get("thrash_regime", {}).get("regime") == "isolate"


def _init_git_repo(path: Path) -> None:
    subprocess.check_call(["git", "init"], cwd=path, stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=path)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=path)
    (path / "README.md").write_text("# test\n", encoding="utf-8")
    (path / "docs" / "design").mkdir(parents=True)
    (path / "docs" / "design" / ".gitkeep").write_text("", encoding="utf-8")
    subprocess.check_call(["git", "add", "README.md", "docs"], cwd=path)
    subprocess.check_call(
        ["git", "commit", "-m", "init"], cwd=path, stdout=subprocess.DEVNULL
    )


def _document_handoff_campaign(
    root: Path,
    *,
    loop_id: str,
    campaign_id: str,
    actions: list[dict],
) -> None:
    camp = root / campaign_id
    camp.mkdir(parents=True, exist_ok=True)
    commit = "a" * 40
    handoff = {
        "schema_version": "AutotrainCycleHandoffV1",
        "loop_id": loop_id,
        "campaign_id": campaign_id,
        "cycle_index": 1,
        "upstream_commit": commit,
        "integration_commit": commit,
        "cycle_role": "screening",
        "cycle_intent": "screening",
        "evidence_class": "fixture",
        "climb_state": "candidate_queued",
        "ship_state": "blocked",
        "primary_metric": "smoke.structural_similarity",
        "actions": actions,
        "checkpoint_paths": [],
        "checkpoint_documentation_required": False,
    }
    (camp / "cycle_handoff.json").write_text(
        json.dumps(handoff, indent=2) + "\n", encoding="utf-8"
    )
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "schema": "autotrain_sdlc_delivery/v1",
                "loop_id": loop_id,
                "campaign_id": campaign_id,
                "positive": False,
                "stack_layer": False,
                "measurement_complete": True,
                "control_metrics": {"structural_similarity": 0.1},
                "candidate_metrics": {"structural_similarity": 0.1},
                "reasons": ["primary_metric_null_or_worse"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_self_heal_document_actions_writes_commits_acks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-test-continuous-openui-local-c1"
    _document_handoff_campaign(
        root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        actions=[
            {
                "kind": "document",
                "owner": "documenting-experiment-results",
                "reason": "persist this cycle's JSON and markdown under docs/design",
                "evidence_ids": [f"campaign:{campaign_id}"],
            },
            {
                "kind": "next_experiment",
                "owner": "autotrain",
                "reason": "consume the ranked successor priorities",
                "evidence_ids": [f"campaign:{campaign_id}"],
            },
        ],
    )
    import slm_training.autoresearch.storage as storage

    monkeypatch.setattr(storage, "_REPO_ROOT", repo)
    with pytest.raises(RuntimeError, match="unacknowledged actions"):
        _mod._require_predecessor_actions(root, loop_id, campaign_id)
    kind = _mod._self_heal_document_actions(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert kind == "document_closeout"
    md = repo / "docs" / "design" / f"{campaign_id}-results.md"
    js = repo / "docs" / "design" / f"{campaign_id}-results.json"
    assert md.is_file() and js.is_file()
    tracked = subprocess.check_output(
        ["git", "ls-files", "--error-unmatch", str(md.relative_to(repo))],
        cwd=repo,
        text=True,
    )
    assert tracked.strip()
    _mod._require_predecessor_actions(root, loop_id, campaign_id)


def test_self_heal_document_actions_reuses_clean_connector_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-test-continuous-openui-local-c2"
    _document_handoff_campaign(
        root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        actions=[
            {
                "kind": "document",
                "owner": "documenting-experiment-results",
                "reason": "persist docs",
                "evidence_ids": [f"campaign:{campaign_id}"],
            }
        ],
    )
    md, js = _mod._continuous_docs_paths(repo, campaign_id)
    md.write_text(f"# {campaign_id}\n\nconnector evidence\n", encoding="utf-8")
    js.write_text(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "loop_id": loop_id,
                "version_stamp": {"code_dirty": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.check_call(["git", "add", "docs"], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "connector docs"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    import slm_training.autoresearch.storage as storage

    monkeypatch.setattr(storage, "_REPO_ROOT", repo)
    monkeypatch.setattr(
        _mod,
        "_git_commit_paths",
        lambda *_args, **_kwargs: pytest.fail("clean connector docs were regenerated"),
    )

    assert (
        _mod._self_heal_document_actions(
            cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
        )
        == "document_closeout"
    )
    assert "connector evidence" in md.read_text(encoding="utf-8")
    _mod._require_predecessor_actions(root, loop_id, campaign_id)


def test_checkpoint_recipes_and_existing_notes_are_reusable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "README.md").write_text("# test\n", encoding="utf-8")
    (repo / "docs" / "MODEL_CARD.md").write_text("# card\n", encoding="utf-8")
    root = repo / "outputs" / "autoresearch"
    campaign_id = "continuous-loop-test-c3"
    checkpoint = root / campaign_id / "runs" / "cand" / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint.parent.parent / "train_summary.json").write_text(
        json.dumps(
            {
                "model": "twotower",
                "device": "cpu",
                "steps": 12,
                "record_count": 34,
                "elapsed_wall_seconds": 5.0,
                "last_loss": 1.25,
                "recipe": {"seed": 7},
            }
        ),
        encoding="utf-8",
    )
    checkpoint.with_suffix(".meta.json").write_text(
        json.dumps({"parameter_count": 1234}), encoding="utf-8"
    )
    handoff = _mod.AutotrainCycleHandoffV1(
        loop_id="loop-1",
        campaign_id=campaign_id,
        cycle_index=3,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        cycle_role="screening",
        cycle_intent="screening",
        evidence_class="fixture",
        climb_state="inconclusive",
        ship_state="blocked",
        primary_metric="smoke.structural_similarity",
        actions=(
            {
                "kind": "document",
                "owner": "documenting-experiment-results",
                "reason": "persist checkpoint docs",
                "evidence_ids": (f"campaign:{campaign_id}",),
            },
        ),
        checkpoint_paths=("runs/cand/checkpoints/last.pt",),
        checkpoint_documentation_required=True,
    )

    recipes = _mod._checkpoint_recipes(cwd=repo, root=root, handoff=handoff)
    first = _mod._append_checkpoint_doc_notes(
        repo,
        campaign_id=campaign_id,
        checkpoint_paths=handoff.checkpoint_paths,
        checkpoint_recipes=recipes,
    )
    second = _mod._append_checkpoint_doc_notes(
        repo,
        campaign_id=campaign_id,
        checkpoint_paths=handoff.checkpoint_paths,
        checkpoint_recipes=recipes,
    )

    assert recipes[0]["trainable_params"] == 1234
    assert recipes[0]["local_path"].endswith("runs/cand/checkpoints/last.pt")
    assert {path.name for path in first} == {"README.md", "MODEL_CARD.md"}
    assert {path.name for path in second} == {"README.md", "MODEL_CARD.md"}
    assert "1,234 trainable parameters" in (repo / "README.md").read_text()


def test_self_heal_does_not_ack_repair_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    loop_id = "loop-1"
    campaign_id = "continuous-loop-test-c1"
    _document_handoff_campaign(
        root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        actions=[
            {
                "kind": "repair_harness",
                "owner": "improve-openui-harnesses",
                "reason": "repair the canonical owner",
                "evidence_ids": [f"campaign:{campaign_id}"],
                "harness_family": "model_build",
            },
            {
                "kind": "document",
                "owner": "documenting-experiment-results",
                "reason": "persist docs",
                "evidence_ids": [f"campaign:{campaign_id}"],
            },
        ],
    )
    import slm_training.autoresearch.storage as storage

    monkeypatch.setattr(storage, "_REPO_ROOT", repo)
    kind = _mod._self_heal_document_actions(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert kind == "document_closeout"
    with pytest.raises(RuntimeError, match="repair_harness"):
        _mod._require_predecessor_actions(root, loop_id, campaign_id)


def test_self_heal_dirty_tree_continuous_docs_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    allowed = repo / "docs" / "design" / "continuous-openui-local-c9-results.md"
    allowed.write_text("# auto\n", encoding="utf-8")
    kind = _mod._self_heal_continuous_dirty_tree(cwd=repo, loop_id="loop-1")
    assert kind == "dirty_tree_closeout"
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    assert status.strip() == ""

    # Foreign WIP must not be auto-committed.
    (repo / "src_extra.py").write_text("x=1\n", encoding="utf-8")
    kind2 = _mod._self_heal_continuous_dirty_tree(cwd=repo, loop_id="loop-1")
    assert kind2 is None
    status2 = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    assert "src_extra.py" in status2


def test_self_heal_cycle_error_document_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-20260805-continuous-openui-local-c74"
    # Lineage discovery uses campaign.json cycle_index + loop_id.
    camp = root / campaign_id
    camp.mkdir(parents=True)
    (camp / "campaign.json").write_text(
        json.dumps(
            {
                "schema_version": "CampaignSpec",
                "campaign_id": campaign_id,
                "loop_id": loop_id,
                "cycle_index": 74,
                "objective": "t",
                "primary_metric": "smoke.structural_similarity",
                "upstream_commit": "a" * 40,
                "integration_commit": "b" * 40,
            }
        ),
        encoding="utf-8",
    )
    _document_handoff_campaign(
        root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        actions=[
            {
                "kind": "document",
                "owner": "documenting-experiment-results",
                "reason": "persist docs",
                "evidence_ids": [f"campaign:{campaign_id}"],
            }
        ],
    )
    import slm_training.autoresearch.storage as storage

    monkeypatch.setattr(storage, "_REPO_ROOT", repo)
    kind = _mod._self_heal_cycle_error(
        root=root,
        loop_id=loop_id,
        cwd=repo,
        exc=RuntimeError(
            f"predecessor {campaign_id} has unacknowledged actions: 0:document"
        ),
    )
    assert kind == "document_closeout"
    _mod._require_predecessor_actions(root, loop_id, campaign_id)


def test_is_continuous_closeout_path_allowlist() -> None:
    assert _mod._is_continuous_closeout_path(
        "docs/design/continuous-openui-local-c1-results.md"
    )
    assert _mod._is_continuous_closeout_path(
        "docs/design/continuous-loop-20260805-x-c1-results.json"
    )
    assert _mod._is_continuous_closeout_path("docs/MODEL_CARD.md")
    assert _mod._is_continuous_closeout_path(
        "src/slm_training/resources/test_seeds.jsonl"
    )
    assert _mod._is_continuous_closeout_path(
        "src/slm_training/resources/data/eval/"
        "e938_role_safe_all_targets_smoke6_v1/screening_sample_size.json"
    )
    assert not _mod._is_continuous_closeout_path("scripts/run_autotrain_continuous.py")
    assert not _mod._is_continuous_closeout_path("docs/design/other-topic.md")


def test_serena_comment_strip_restores_and_continues(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    yml = repo / ".serena" / "project.yml"
    yml.parent.mkdir()
    yml.write_text("# tracked comment\nlanguage: python\n", encoding="utf-8")
    subprocess.check_call(["git", "add", ".serena/project.yml"], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "serena"], cwd=repo, stdout=subprocess.DEVNULL
    )
    yml.write_text("language: python\n", encoding="utf-8")
    root = repo / "outputs" / "autoresearch"
    root.mkdir(parents=True)
    report = _mod.self_heal_unblock_loop(cwd=repo, root=root, loop_id="L")
    assert "serena_project_yml_comment_strip" in (report.get("soft_healed") or [])
    assert not any(
        item.get("kind") == "foreign_dirty_tree"
        for item in report.get("hard_pending") or []
    )
    assert yml.read_text(encoding="utf-8") == "# tracked comment\nlanguage: python\n"


def test_serena_semantic_edit_still_parks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    yml = repo / ".serena" / "project.yml"
    yml.parent.mkdir()
    yml.write_text("language: python\n", encoding="utf-8")
    subprocess.check_call(["git", "add", ".serena/project.yml"], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "serena"], cwd=repo, stdout=subprocess.DEVNULL
    )
    yml.write_text("language: typescript\n", encoding="utf-8")
    root = repo / "outputs" / "autoresearch"
    root.mkdir(parents=True)
    report = _mod.self_heal_unblock_loop(cwd=repo, root=root, loop_id="L")
    assert "serena_project_yml_comment_strip" not in (report.get("soft_healed") or [])
    assert any(
        item.get("kind") == "foreign_dirty_tree"
        for item in report.get("hard_pending") or []
    )


def test_serena_local_dirt_is_not_foreign() -> None:
    assert (
        _mod._normalize_repo_relpath(".serena/memories/note.md")
        == ".serena/memories/note.md"
    )
    assert _mod._normalize_repo_relpath("./docs/design/x.json") == "docs/design/x.json"
    assert not _mod._is_foreign_dirty_path(".serena/memories/note.md")
    assert not _mod._is_foreign_dirty_path("./.serena/cache/index")
    assert not _mod._is_foreign_dirty_path(".serena")
    assert not _mod._is_foreign_dirty_path(".pytest_cache/v/cache")
    assert _mod._is_foreign_dirty_path(".serena/project.yml")
    assert _mod._is_foreign_dirty_path(".serena/.gitignore")


def test_loop_owned_generated_path_is_not_foreign() -> None:
    path = "src/slm_training/resources/evidence_store/local_index.jsonl"
    assert _mod._is_loop_owned_generated_path(path)
    assert not _mod._is_foreign_dirty_path(path)
    sidecar = (
        "src/slm_training/resources/data/eval/"
        "e938_role_safe_all_targets_smoke6_v1/screening_sample_size.json"
    )
    assert _mod._is_loop_owned_generated_path(sidecar)
    assert not _mod._is_foreign_dirty_path(sidecar)
    assert _mod._is_process_arm({"heal_resume": True})
    assert _mod._is_process_arm({"process_arm": True, "process_role": "first_snapshot"})
    assert not _mod._is_process_arm({"train_version": "wf_smoke_v2"})


def test_self_heal_restores_loop_owned_generated_dirt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    mirror = (
        repo
        / "src"
        / "slm_training"
        / "resources"
        / "evidence_store"
        / "local_index.jsonl"
    )
    mirror.parent.mkdir(parents=True)
    mirror.write_text('{"ok": true}\n', encoding="utf-8")
    subprocess.check_call(["git", "add", str(mirror.relative_to(repo))], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "seed mirror"], cwd=repo, stdout=subprocess.DEVNULL
    )
    mirror.write_text('{"ok": false, "dirty": true}\n', encoding="utf-8")
    root = repo / "outputs" / "autoresearch"
    root.mkdir(parents=True)
    report = _mod.self_heal_unblock_loop(
        cwd=repo, root=root, loop_id="continuous-openui-local"
    )
    assert "loop_owned_generated_dirt" in report.get("soft_healed", [])
    assert not any(
        item.get("kind") == "foreign_dirty_tree"
        for item in report.get("hard_pending") or []
    )
    assert mirror.read_text(encoding="utf-8") == '{"ok": true}\n'
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    assert "local_index.jsonl" not in status


def test_load_frozen_replay_skips_missing_control_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ar"
    loop = "loop-1"
    campaign_id = "camp-1"
    camp = root / campaign_id
    manifests = camp / "manifests"
    manifests.mkdir(parents=True)
    sha = "f" * 64
    handoff = {
        "schema_version": "AutotrainCycleHandoffV1",
        "loop_id": loop,
        "campaign_id": campaign_id,
        "cycle_index": 1,
        "upstream_commit": "a" * 40,
        "integration_commit": "b" * 40,
        "cycle_role": "screening",
        "cycle_intent": "retry_measurement",
        "evidence_class": "fixture",
        "climb_state": "harness_failure",
        "ship_state": "blocked",
        "primary_metric": "smoke.structural_similarity",
        "actions": [
            {
                "kind": "retry_measurement",
                "owner": "autotrain",
                "reason": "replay",
                "evidence_ids": [f"campaign:{campaign_id}"],
                "frozen_manifest_sha256": sha,
            }
        ],
    }
    (camp / "cycle_handoff.json").write_text(json.dumps(handoff) + "\n")
    (camp / "matrix-proposal.json").write_text(
        json.dumps(
            {
                "hypotheses": [
                    {"experiment": {"experiment_id": "camp-1-control"}},
                    {"experiment": {"experiment_id": "camp-1-cand"}},
                ]
            }
        )
    )
    cand_path = manifests / "camp-1-cand.json"

    class _M:
        experiment_id = "camp-1-cand"

    monkeypatch.setattr(
        _mod, "_manifest_with_sha", lambda camp_dir, digest: (cand_path, _M())
    )
    monkeypatch.setattr(
        _mod, "_nonreplayable_configuration_failure", lambda *a, **k: None
    )
    result = _mod._load_frozen_replay(root, loop, campaign_id)
    assert result is None


def test_self_heal_thrash_timeout_repair_bypasses_blocking_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-test-c80"
    camp = root / campaign_id
    camp.mkdir(parents=True)
    (camp / "campaign.json").write_text(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "loop_id": loop_id,
                "cycle_index": 80,
            }
        ),
        encoding="utf-8",
    )
    sha = "c" * 64
    handoff = {
        "schema_version": "AutotrainCycleHandoffV1",
        "loop_id": loop_id,
        "campaign_id": campaign_id,
        "cycle_index": 80,
        "upstream_commit": "a" * 40,
        "integration_commit": "b" * 40,
        "cycle_role": "screening",
        "cycle_intent": "screening",
        "evidence_class": "fixture",
        "climb_state": "inconclusive",
        "ship_state": "blocked",
        "primary_metric": "smoke.structural_similarity",
        "reasons": [
            "measurement_incomplete:cand:smoke:incomplete_document_n=1:decode_timeout_count=1",
            "fixture_insufficient_n_alone",
        ],
        "actions": [
            {
                "kind": "repair_harness",
                "owner": "improve-openui-harnesses",
                "reason": (
                    "AgentV finalized every record disposition and reported an "
                    "internal decode timeout; repair canonical model-build runtime"
                ),
                "evidence_ids": [f"campaign:{campaign_id}"],
                "harness_family": "model_build",
                "frozen_manifest_sha256": sha,
            },
            {
                "kind": "retry_measurement",
                "owner": "autotrain",
                "reason": "replay after repair",
                "evidence_ids": [f"campaign:{campaign_id}"],
                "frozen_manifest_sha256": sha,
            },
            {
                "kind": "document",
                "owner": "documenting-experiment-results",
                "reason": "persist docs",
                "evidence_ids": [f"campaign:{campaign_id}"],
            },
        ],
        "checkpoint_paths": [],
        "checkpoint_documentation_required": False,
    }
    (camp / "cycle_handoff.json").write_text(json.dumps(handoff) + "\n")
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "schema": "autotrain_sdlc_delivery/v1",
                "positive": False,
                "measurement_complete": False,
                "arm_exits": {"cand": 124, "control": 124},
                "reasons": [
                    "measurement_incomplete:cand:decode_timeout_count=1",
                ],
            }
        )
        + "\n"
    )
    import slm_training.autoresearch.storage as storage

    monkeypatch.setattr(storage, "_REPO_ROOT", repo)
    with pytest.raises(RuntimeError, match="repair_harness"):
        _mod._require_predecessor_actions(root, loop_id, campaign_id)
    kind = _mod._self_heal_thrash_timeout_repair(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert kind in {"thrash_timeout_repair_bypass", "document_closeout"}
    rebuilt = json.loads((camp / "cycle_handoff.json").read_text())
    kinds = [a["kind"] for a in rebuilt["actions"]]
    assert "repair_harness" not in kinds
    assert "next_experiment" in kinds
    _mod._require_predecessor_actions(root, loop_id, campaign_id)


def test_self_heal_cycle_error_repairs_thrash_timeout_from_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-test-c81"
    camp = root / campaign_id
    camp.mkdir(parents=True)
    (camp / "campaign.json").write_text(
        json.dumps({"campaign_id": campaign_id, "loop_id": loop_id, "cycle_index": 81})
    )
    sha = "d" * 64
    handoff = {
        "schema_version": "AutotrainCycleHandoffV1",
        "loop_id": loop_id,
        "campaign_id": campaign_id,
        "cycle_index": 81,
        "upstream_commit": "a" * 40,
        "integration_commit": "b" * 40,
        "cycle_role": "screening",
        "cycle_intent": "screening",
        "evidence_class": "fixture",
        "climb_state": "inconclusive",
        "ship_state": "blocked",
        "primary_metric": "smoke.structural_similarity",
        "reasons": ["measurement_incomplete:x:decode_timeout_count=1"],
        "actions": [
            {
                "kind": "repair_harness",
                "owner": "improve-openui-harnesses",
                "reason": "internal decode timeout; repair runtime",
                "evidence_ids": [f"campaign:{campaign_id}"],
                "harness_family": "model_build",
                "frozen_manifest_sha256": sha,
            },
            {
                "kind": "document",
                "owner": "documenting-experiment-results",
                "reason": "persist docs",
                "evidence_ids": [f"campaign:{campaign_id}"],
            },
        ],
    }
    (camp / "cycle_handoff.json").write_text(json.dumps(handoff) + "\n")
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "arm_exits": {"a": 124, "b": 124},
                "reasons": ["measurement_incomplete:decode_timeout_count=1"],
            }
        )
    )
    import slm_training.autoresearch.storage as storage

    monkeypatch.setattr(storage, "_REPO_ROOT", repo)
    kind = _mod._self_heal_cycle_error(
        root=root,
        loop_id=loop_id,
        cwd=repo,
        exc=RuntimeError(
            f"predecessor {campaign_id} has unacknowledged actions: 0:repair_harness"
        ),
    )
    assert kind is not None
    _mod._require_predecessor_actions(root, loop_id, campaign_id)


def test_delivery_is_thrash_timeout_residual_detects_wall_exits() -> None:
    assert _mod._delivery_is_thrash_timeout_residual(
        {"arm_exits": {"a": 124, "b": 124}, "reasons": []}
    )
    assert _mod._delivery_is_thrash_timeout_residual(
        {"reasons": ["measurement_incomplete:x:decode_timeout_count=1"]}
    )
    # Bare measurement_incomplete is not enough (also appears on real harness bugs).
    assert not _mod._delivery_is_thrash_timeout_residual(
        {"reasons": ["measurement_incomplete:cand:incomplete_document_n=1"]}
    )
    assert not _mod._delivery_is_thrash_timeout_residual(
        {"reasons": ["harness_failure:AgentV SDK is unavailable; run npm ci"]}
    )


def test_self_heal_unblock_loop_soft_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-unblock-doc-c1"
    _document_handoff_campaign(
        root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        actions=[
            {
                "kind": "document",
                "owner": "documenting-experiment-results",
                "reason": "persist docs",
                "evidence_ids": [f"campaign:{campaign_id}"],
            },
            {
                "kind": "next_experiment",
                "owner": "autotrain",
                "reason": "continue",
                "evidence_ids": [f"campaign:{campaign_id}"],
            },
        ],
    )
    (root / campaign_id / "campaign.json").write_text(
        json.dumps({"campaign_id": campaign_id, "loop_id": loop_id, "cycle_index": 1})
    )
    import slm_training.autoresearch.storage as storage

    monkeypatch.setattr(storage, "_REPO_ROOT", repo)
    report = _mod.self_heal_unblock_loop(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert report["blocker_cleared"] is True
    assert not report["hard_pending"]
    _mod._require_predecessor_actions(root, loop_id, campaign_id)


def test_self_heal_unblock_loop_hard_agentv_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-unblock-hard-c1"
    camp = root / campaign_id
    camp.mkdir(parents=True)
    (camp / "campaign.json").write_text(
        json.dumps({"campaign_id": campaign_id, "loop_id": loop_id, "cycle_index": 1})
    )
    sha = "e" * 64
    handoff = {
        "schema_version": "AutotrainCycleHandoffV1",
        "loop_id": loop_id,
        "campaign_id": campaign_id,
        "cycle_index": 1,
        "upstream_commit": "a" * 40,
        "integration_commit": "b" * 40,
        "cycle_role": "screening",
        "cycle_intent": "screening",
        "evidence_class": "fixture",
        "climb_state": "harness_failure",
        "ship_state": "blocked",
        "primary_metric": "smoke.structural_similarity",
        "reasons": ["harness_failure:AgentV SDK is unavailable; run npm ci"],
        "actions": [
            {
                "kind": "repair_harness",
                "owner": "improve-openui-harnesses",
                "reason": "AgentV SDK is unavailable; run npm ci",
                "evidence_ids": [f"campaign:{campaign_id}"],
                "harness_family": "model_build",
                "frozen_manifest_sha256": sha,
            }
        ],
    }
    (camp / "cycle_handoff.json").write_text(json.dumps(handoff) + "\n")
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "reasons": ["harness_failure:AgentV SDK is unavailable; run npm ci"],
                "arm_exits": {"a": 1},
            }
        )
    )
    import slm_training.autoresearch.storage as storage

    monkeypatch.setattr(storage, "_REPO_ROOT", repo)
    report = _mod.self_heal_unblock_loop(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert report["blocker_cleared"] is False
    assert any(h.get("kind") == "repair_harness" for h in report["hard_pending"])
    with pytest.raises(RuntimeError, match="repair_harness"):
        _mod._require_predecessor_actions(root, loop_id, campaign_id)


def test_soft_document_failures_never_block(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    for i in range(5):
        count = _mod._record_cycle_failure(
            root=root,
            loop_id="loop-1",
            exc=RuntimeError("predecessor c1 has unacknowledged actions: 0:document"),
            cycle_index=i,
        )
        assert count == 0
    state = json.loads((root / "loops" / "loop-1" / "state.json").read_text())
    assert state["state"] == "IDLE"
    assert state["blocker_count"] == 0


def test_last_cycle_failure_message_reads_tail(tmp_path: Path) -> None:
    root = tmp_path / "ar"
    loop = "L"
    path = root / "loops" / loop / "cycle_failures.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "loop_id": loop,
                "blocking": True,
                "message": "predecessor x has unacknowledged actions: 0:repair_harness",
            }
        )
        + "\n"
    )
    assert "repair_harness" in (_mod._last_cycle_failure_message(root, loop) or "")


def test_self_heal_bank_exhaust_repair_rewrites_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bank-exhaust repair_harness must compose arms and clear predecessor gate."""
    # Legacy compose repair path: only active while terminal parking is off.
    _inject_terminal_policy(monkeypatch, park=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root = repo / "outputs" / "autoresearch"
    loop_id = "continuous-openui-local"
    campaign_id = "continuous-loop-bank-exhaust-c1"
    camp = root / campaign_id
    camp.mkdir(parents=True)
    (camp / "campaign.json").write_text(
        json.dumps({"campaign_id": campaign_id, "loop_id": loop_id, "cycle_index": 1})
    )
    handoff = {
        "schema_version": "AutotrainCycleHandoffV1",
        "loop_id": loop_id,
        "campaign_id": campaign_id,
        "cycle_index": 1,
        "upstream_commit": "a" * 40,
        "integration_commit": "b" * 40,
        "cycle_role": "screening",
        "cycle_intent": "screening",
        "evidence_class": "fixture",
        "climb_state": "rejected",
        "ship_state": "blocked",
        "primary_metric": "smoke.structural_similarity",
        "reasons": ["primary_metric_null_or_worse"],
        "actions": [
            {
                "kind": "repair_harness",
                "owner": "improve-openui-harnesses",
                "reason": (
                    "registered quality-arm bank exhausted; preregister and wire "
                    "a distinct size-matched model-build objective before the next run"
                ),
                "evidence_ids": [f"campaign:{campaign_id}"],
                "harness_family": "model_build",
            },
            {
                "kind": "document",
                "owner": "documenting-experiment-results",
                "reason": "persist docs",
                "evidence_ids": [f"campaign:{campaign_id}"],
            },
        ],
    }
    (camp / "cycle_handoff.json").write_text(json.dumps(handoff) + "\n")
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "positive": False,
                "measurement_complete": True,
                "reasons": ["primary_metric_null_or_worse"],
            }
        )
    )
    import slm_training.autoresearch.storage as storage

    monkeypatch.setattr(storage, "_REPO_ROOT", repo)
    # Pretend static bank is fully closed so compose is forced.
    monkeypatch.setattr(
        _mod,
        "_recent_completed_nonpositive_slugs",
        lambda root, pred: {slug for slug, _, _ in _mod._SCREENING_ARM_BANK},
    )
    monkeypatch.setattr(_mod, "_load_champion_queue", lambda path: [])
    monkeypatch.setattr(_mod, "_skip_arm_slugs", lambda *a, **k: set())
    with pytest.raises(RuntimeError, match="repair_harness"):
        _mod._require_predecessor_actions(root, loop_id, campaign_id)
    kind = _mod._self_heal_bank_exhaust_repair(
        cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    assert kind in {"bank_exhaust_compose", "document_closeout"}
    rebuilt = json.loads((camp / "cycle_handoff.json").read_text())
    kinds = [a["kind"] for a in rebuilt["actions"]]
    assert "repair_harness" not in kinds
    assert "next_experiment" in kinds
    _mod._require_predecessor_actions(root, loop_id, campaign_id)


def test_is_bank_exhaust_repair_action_matches_handoff_reason() -> None:
    from slm_training.autoresearch.schemas import AutotrainActionV1

    a = AutotrainActionV1(
        kind="repair_harness",
        owner="improve-openui-harnesses",
        reason=(
            "registered quality-arm bank exhausted; preregister and wire a distinct "
            "size-matched model-build objective before the next run"
        ),
        evidence_ids=("campaign:x",),
        harness_family="model_build",
    )
    assert _mod._is_bank_exhaust_repair_action(a)
    b = AutotrainActionV1(
        kind="repair_harness",
        owner="improve-openui-harnesses",
        reason="AgentV SDK is unavailable; run npm ci",
        evidence_ids=("campaign:x",),
        harness_family="model_build",
    )
    assert not _mod._is_bank_exhaust_repair_action(b)


def test_self_heal_incomplete_merge_prefers_main_for_harness(
    tmp_path: Path,
) -> None:
    """Interrupted origin/main merge (UU) must finish without human re-prompt."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    # shared base with harness
    (repo / "harness.py").write_text("v1\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "harness.py"], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "harness v1"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    # trunk branch (incoming "main" side of the merge)
    subprocess.check_call(
        ["git", "branch", "-f", "trunk"], cwd=repo, stdout=subprocess.DEVNULL
    )
    # diverge loop branch (current HEAD)
    (repo / "harness.py").write_text("loop-local\n", encoding="utf-8")
    (repo / "docs" / "design" / "continuous-loop-x-results.md").write_text(
        "closeout\n", encoding="utf-8"
    )
    subprocess.check_call(
        ["git", "add", "harness.py", "docs/design/continuous-loop-x-results.md"],
        cwd=repo,
    )
    subprocess.check_call(
        ["git", "commit", "-m", "loop diverge"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    loop_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    # trunk advances with conflicting harness
    subprocess.check_call(
        ["git", "checkout", "trunk"], cwd=repo, stdout=subprocess.DEVNULL
    )
    (repo / "harness.py").write_text("main-fixed\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "harness.py"], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "harness fix on main"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["git", "checkout", loop_branch],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    # merge trunk → conflict (same as interrupted origin/main merge)
    merge = subprocess.run(
        ["git", "merge", "--no-edit", "trunk"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert merge.returncode != 0, "expected conflict"
    assert _mod._merge_head_path(repo) is not None
    root = repo / "outputs" / "autoresearch"
    root.mkdir(parents=True)
    kind = _mod._self_heal_incomplete_merge(
        cwd=repo, root=root, loop_id="continuous-openui-local"
    )
    assert kind == "git_merge_complete"
    assert _mod._merge_head_path(repo) is None
    assert (repo / "harness.py").read_text(encoding="utf-8") == "main-fixed\n"
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    )
    assert porcelain.strip() == ""
    # unblock loop must not report foreign_dirty after heal
    report = _mod.self_heal_unblock_loop(
        cwd=repo, root=root, loop_id="continuous-openui-local"
    )
    assert not any(
        h.get("kind") == "foreign_dirty_tree" for h in report.get("hard_pending") or []
    )


def test_self_heal_incomplete_merge_aborts_when_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed merge commit must abort MERGE_HEAD, not leave foreign_dirty."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "harness.py").write_text("v1\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "harness.py"], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "harness v1"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["git", "branch", "-f", "trunk"], cwd=repo, stdout=subprocess.DEVNULL
    )
    (repo / "harness.py").write_text("loop-local\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "harness.py"], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "loop diverge"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    loop_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    subprocess.check_call(
        ["git", "checkout", "trunk"], cwd=repo, stdout=subprocess.DEVNULL
    )
    (repo / "harness.py").write_text("main-fixed\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "harness.py"], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "harness fix on main"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["git", "checkout", loop_branch],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    merge = subprocess.run(
        ["git", "merge", "--no-edit", "trunk"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert merge.returncode != 0, "expected conflict"
    assert _mod._merge_head_path(repo) is not None
    real_run = _mod._run

    def _run_fail_commit(cmd: list[str], **kwargs: object) -> None:
        if kwargs.get("stage") == "self-heal-merge-commit":
            raise RuntimeError("hook: version-stamps history order")
        real_run(cmd, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_mod, "_run", _run_fail_commit)
    root = repo / "outputs" / "autoresearch"
    root.mkdir(parents=True)
    kind = _mod._self_heal_incomplete_merge(
        cwd=repo, root=root, loop_id="continuous-openui-local"
    )
    assert kind is None
    assert _mod._merge_head_path(repo) is None
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    )
    assert porcelain.strip() == ""
    assert (repo / "harness.py").read_text(encoding="utf-8") == "loop-local\n"


def test_integrate_origin_main_skips_diverged_unmergeable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diverged origin/main must abort MERGE_HEAD and not raise CYCLE_ERROR."""
    origin = tmp_path / "origin"
    repo = tmp_path / "work"
    origin.mkdir()
    _init_git_repo(origin)
    subprocess.check_call(
        ["git", "branch", "-M", "main"], cwd=origin, stdout=subprocess.DEVNULL
    )
    subprocess.check_call(
        ["git", "clone", str(origin), str(repo)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=repo)
    (repo / "conflict.txt").write_text("loop\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "conflict.txt"], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "loop diverge"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    (origin / "conflict.txt").write_text("main\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "conflict.txt"], cwd=origin)
    subprocess.check_call(
        ["git", "commit", "-m", "main diverge"],
        cwd=origin,
        stdout=subprocess.DEVNULL,
    )
    real_run = _mod._run

    def _run_fail_commit(cmd: list[str], **kwargs: object) -> None:
        if kwargs.get("stage") == "self-heal-merge-commit":
            raise RuntimeError("hook: version-stamps history order")
        real_run(cmd, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_mod, "_run", _run_fail_commit)
    root = repo / "outputs" / "autoresearch"
    root.mkdir(parents=True)
    kind = _mod._integrate_origin_main(
        cwd=repo, root=root, loop_id="continuous-openui-local"
    )
    assert kind == "git_ancestry_skip"
    assert _mod._merge_head_path(repo) is None
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    )
    assert porcelain.strip() == ""
    exc = subprocess.CalledProcessError(
        1, ["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"]
    )
    wrapped = _mod._self_heal_git_ancestry(
        cwd=repo, root=root, loop_id="continuous-openui-local", exc=exc
    )
    assert wrapped == "git_ancestry_skip"
    assert _mod._merge_head_path(repo) is None


def test_upstream_commit_for_init_uses_head_when_diverged(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    repo = tmp_path / "work"
    origin.mkdir()
    _init_git_repo(origin)
    subprocess.check_call(
        ["git", "branch", "-M", "main"], cwd=origin, stdout=subprocess.DEVNULL
    )
    subprocess.check_call(
        ["git", "clone", str(origin), str(repo)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=repo)
    root = repo / "outputs" / "autoresearch"
    root.mkdir(parents=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    main = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=repo, text=True
    ).strip()
    assert (
        _mod._upstream_commit_for_init(
            cwd=repo,
            root=root,
            loop_id="loop-1",
            upstream=main,
            integration=head,
        )
        == main
    )
    (repo / "conflict.txt").write_text("loop\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "conflict.txt"], cwd=repo)
    subprocess.check_call(
        ["git", "commit", "-m", "loop diverge"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    (origin / "conflict.txt").write_text("main\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "conflict.txt"], cwd=origin)
    subprocess.check_call(
        ["git", "commit", "-m", "main diverge"],
        cwd=origin,
        stdout=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["git", "fetch", "origin", "main"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    main = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=repo, text=True
    ).strip()
    assert head != main
    assert (
        _mod._upstream_commit_for_init(
            cwd=repo,
            root=root,
            loop_id="loop-1",
            upstream=main,
            integration=head,
        )
        == head
    )


def _write_registry(path: Path, *, paths: list[str], version: str = "v3") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "version_registry/v1",
                "components": {
                    "harness.experiments.slm228_spectral_disposition": {
                        "version": version,
                        "kind": "harness",
                        "paths": paths,
                        "history": [
                            {
                                "version": version,
                                "date": "2026-08-01",
                                "note": "no-bump: prior entry unrelated to this test.",
                            }
                        ],
                    },
                    "harness.unrelated_component": {
                        "version": "v1",
                        "kind": "harness",
                        "paths": ["docs/design/unrelated.md"],
                        "history": [
                            {
                                "version": "v1",
                                "date": "2026-08-01",
                                "note": "no-bump: unrelated component seed entry.",
                            }
                        ],
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_auto_no_bump_version_registry_records_checkpoint_note_only_change(
    tmp_path: Path,
) -> None:
    """Checkpoint-note-only README/MODEL_CARD edits must not repeatedly hard-block
    the continuous driver's self-heal document commit on the version-stamp gate."""
    registry_path = tmp_path / "src" / "slm_training" / "resources" / "versions.json"
    _write_registry(registry_path, paths=["README.md", "docs/MODEL_CARD.md"])

    result = _mod._auto_no_bump_version_registry(
        tmp_path,
        touched_rel_paths=["README.md", "docs/MODEL_CARD.md"],
        loop_id="continuous-openui-scheduled-w1tlbr",
        campaign_id="continuous-loop-20260808-c1",
    )
    assert result == registry_path

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    owner = registry["components"]["harness.experiments.slm228_spectral_disposition"]
    assert len(owner["history"]) == 2
    assert owner["version"] == "v3"  # never bumps for a doc-prose-only change
    top_note = owner["history"][0]["note"]
    assert top_note.startswith("no-bump:")
    assert "continuous-loop-20260808-c1" in top_note

    # Unrelated components (no touched path in their `paths`) stay untouched.
    unrelated = registry["components"]["harness.unrelated_component"]
    assert len(unrelated["history"]) == 1


def test_auto_no_bump_version_registry_is_idempotent_per_campaign(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "src" / "slm_training" / "resources" / "versions.json"
    _write_registry(registry_path, paths=["README.md"])

    kwargs = dict(
        touched_rel_paths=["README.md"],
        loop_id="continuous-openui-scheduled-w1tlbr",
        campaign_id="continuous-loop-20260808-c1",
    )
    first = _mod._auto_no_bump_version_registry(tmp_path, **kwargs)
    assert first is not None
    second = _mod._auto_no_bump_version_registry(tmp_path, **kwargs)
    assert second is None  # same campaign already recorded; no duplicate entry

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    owner = registry["components"]["harness.experiments.slm228_spectral_disposition"]
    assert len(owner["history"]) == 2


def test_auto_no_bump_version_registry_noop_without_owning_component(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "src" / "slm_training" / "resources" / "versions.json"
    _write_registry(registry_path, paths=["docs/design/unrelated.md"])

    result = _mod._auto_no_bump_version_registry(
        tmp_path,
        touched_rel_paths=["README.md", "docs/MODEL_CARD.md"],
        loop_id="continuous-openui-scheduled-w1tlbr",
        campaign_id="continuous-loop-20260808-c1",
    )
    assert result is None


def test_screening_matrix_sets_latency_probe_knobs() -> None:
    """Policy latency_probe block compiles per-arm probe knobs (screening)."""
    matrix = _mod._matrix(
        campaign_id="continuous-loop-latprobe-c1",
        evidence_snapshot_id="snap",
        cites=["docs/a.md", "docs/b.md", "docs/c.md"],
        role_citations={"research": "docs/a.md", "prior_result": "docs/b.md"},
        train_version="wf_smoke_v2",
        eval_version="e_test",
        steps=80,
        cycle=4,
        role="screening",
        recommended_slug="bounds",
    )
    knobs = matrix["hypotheses"][0]["experiment"]["knobs"]
    assert knobs["latency_probe_records"] == 1
    assert knobs["latency_probe_planned_n"] >= 1


def test_classify_positive_types_latency_preflight_missing_scoreboard(
    tmp_path: Path,
) -> None:
    """A probe-skipped eval reads as its typed cause, not a bare missing file."""
    camp = tmp_path / "camp"
    (camp / "artifacts" / "outcomes").mkdir(parents=True)
    (camp / "runs").mkdir()
    outcome = {
        "experiment_id": "c-candidate",
        "status": "completed",
        "metrics": {},
        "stage_telemetry": [
            {
                "command": ["python", "-m", "scripts.evaluate_model"],
                "skipped": True,
                "latency_preflight": {
                    "schema": "latency_preflight/v1",
                    "verdict": "latency_preflight_infeasible",
                },
            }
        ],
    }
    (camp / "artifacts" / "outcomes" / "o1.json").write_text(json.dumps(outcome))
    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="smoke.structural_similarity",
        control_id="c-control",
        candidate_id="c-candidate",
    )
    assert result["positive"] is False
    assert (
        "measurement_incomplete:c-candidate:latency_preflight_infeasible"
        in result["reasons"]
    )
    # Still a harness-incomplete reason: never a model reject.
    assert _mod._reason_is_harness_incomplete(
        "measurement_incomplete:c-candidate:latency_preflight_infeasible"
    )


def test_run_arm_eval_nll_writes_smoke_eval_nll(tmp_path: Path) -> None:
    from slm_training.autoresearch.climb_policy import screening_nll_definition_hash

    run_dir = tmp_path / "runs" / "arm"
    run_dir.mkdir(parents=True)
    (run_dir / "scoreboard.json").write_text(
        json.dumps({"suites": {"smoke": {"n": 6, "structural_similarity": 0.1}}}),
        encoding="utf-8",
    )
    out = _mod._run_arm_eval_nll(run_dir, eval_nll=3.25)
    assert out["eval_nll"] == 3.25
    scoreboard = json.loads((run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert scoreboard["suites"]["smoke"]["eval_nll"] == 3.25
    assert scoreboard["suites"]["smoke"]["eval_nll_claim_class"] == "diagnostic"
    assert scoreboard["suites"]["smoke"]["eval_nll_definition_hash"] == (
        screening_nll_definition_hash(
            arm_loss_weights={"binder_arity_loss_weight": 7.0}
        )
    )
    metrics = _mod._run_metrics(tmp_path, "arm")
    assert metrics["smoke.eval_nll"] == 3.25
    assert metrics["eval_nll"] == 3.25


def test_attach_screening_eval_nll_skips_without_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "arm"
    run_dir.mkdir(parents=True)
    (run_dir / "scoreboard.json").write_text(
        json.dumps({"suites": {"smoke": {"n": 6}}}), encoding="utf-8"
    )
    assert _mod._attach_screening_eval_nll(run_dir) is None
    scoreboard = json.loads((run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert "eval_nll" not in scoreboard["suites"]["smoke"]


def test_attach_screening_eval_nll_uses_reused_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "runs" / "arm"
    run_dir.mkdir(parents=True)
    (run_dir / "scoreboard.json").write_text(
        json.dumps({"suites": {"smoke": {"n": 6}}}), encoding="utf-8"
    )
    checkpoint = tmp_path / "source" / "last.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "slm_training.data.store.DataStore.resolve_path",
        lambda *_args: tmp_path / "eval",
    )
    monkeypatch.setattr(_mod, "default_eval_version", lambda: "test-eval")

    def fake_eval(
        target: Path, *, test_dir: Path, checkpoint: Path
    ) -> dict[str, Any]:
        seen.update(target=target, test_dir=test_dir, checkpoint=checkpoint)
        return {"eval_nll": 1.25}

    monkeypatch.setattr(_mod, "_run_arm_eval_nll", fake_eval)
    assert _mod._attach_screening_eval_nll(
        run_dir, checkpoint=checkpoint
    ) == {"eval_nll": 1.25}
    assert seen["checkpoint"] == checkpoint


def test_fit_screening_candidate_count_never_kills_for_k() -> None:
    n, reason = _mod._fit_screening_candidate_count(
        max_candidates=6,
        arm_wall_seconds=70.0,
        stage_remaining_seconds=180.0,
        finalization_reserve=15.0,
    )
    assert n == 1
    assert reason in {
        "stage_wall_fits_one_candidate",
        "stage_wall_fitted_candidate_count",
    }
    n6, reason6 = _mod._fit_screening_candidate_count(
        max_candidates=6,
        arm_wall_seconds=20.0,
        stage_remaining_seconds=180.0,
        finalization_reserve=15.0,
    )
    assert n6 == 6
    assert reason6 is None
    n3, reason3 = _mod._fit_screening_candidate_count(
        max_candidates=6,
        arm_wall_seconds=40.0,
        stage_remaining_seconds=180.0,
        finalization_reserve=15.0,
    )
    assert n3 == 3
    assert reason3 == "stage_wall_fitted_candidate_count"


def test_multi_arm_bind_locks_rule_before_experiment_started(tmp_path: Path) -> None:
    from slm_training.autoresearch.experiment_campaign import (
        CampaignArmV1,
        CampaignControlV1,
        CampaignEndpointV1,
        CampaignGateV1,
        ExperimentCampaignV1,
        MultiplicityFamilyV1,
    )
    from slm_training.autoresearch.experiment_campaign import (
        ArtifactRequirementV1,
    )
    from slm_training.autoresearch.schemas import CampaignBudget

    campaign_id = "campaign-multiarm"
    campaign = _mod.CampaignSpec(
        campaign_id=campaign_id,
        objective="Lock k screening arms before execution.",
        primary_metric="smoke.eval_nll",
        loop_id="loop-multiarm",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
    )
    store = _mod.CampaignStore(campaign_id, tmp_path)
    store.initialize(campaign)
    matrix_path = tmp_path / campaign_id / "matrix-proposal.json"
    matrix_path.write_text(json.dumps({"matrix_id": "matrix-k"}), encoding="utf-8")
    order = ("ctrl", "a1", "a2", "a3")
    bound = _mod._bind_expected_arms(
        root=tmp_path,
        campaign_id=campaign_id,
        matrix_path=matrix_path,
        control_id="ctrl",
        candidate_id="a1",
        candidate_ids=("a2", "a3"),
        arm_order=order,
        selection_rule="best_by_primary_then_smallest",
    )
    assert bound["event_type"] == "decision_arms_bound"
    assert bound["detail"]["selection_rule"] == "best_by_primary_then_smallest"
    assert bound["detail"]["expected_arm_ids"] == ["ctrl", "a1", "a2", "a3"]
    manifest = ExperimentCampaignV1(
        campaign_id=campaign_id,
        experiment_id="a1",
        hypothesis="Screen k size-matched candidates against one control.",
        decision="Pick winner by locked selection_rule only.",
        endpoints=(
            CampaignEndpointV1(
                endpoint_id="primary",
                metric="smoke.eval_nll",
                role="primary",
                direction="decrease",
                minimum_effect=0.05,
            ),
        ),
        arms=(
            CampaignArmV1(arm_id="ctrl", role="control", config_sha256="c" * 64),
            CampaignArmV1(arm_id="a1", role="candidate", config_sha256="d" * 64),
            CampaignArmV1(arm_id="a2", role="candidate", config_sha256="e" * 64),
            CampaignArmV1(arm_id="a3", role="candidate", config_sha256="f" * 64),
        ),
        seeds=(7,),
        budget=CampaignBudget(max_experiments=4, max_wall_minutes=3),
        stopping_rules=("Stop after locked arms finish.",),
        controls=(
            CampaignControlV1(
                control_id="matched-control",
                description="Shared size-matched screening control.",
                kind="negative",
            ),
        ),
        negative_controls=("matched-control",),
        multiplicity_families=(
            MultiplicityFamilyV1(
                family_id="primary-family", hypothesis_ids=("primary",), alpha=0.05
            ),
        ),
        promotion_gates=(
            CampaignGateV1(
                gate_id="promote-primary",
                endpoint_id="primary",
                operator="le",
                threshold=-0.05,
            ),
        ),
        rollback_gates=(
            CampaignGateV1(
                gate_id="rollback-primary",
                endpoint_id="primary",
                operator="gt",
                threshold=1e9,
            ),
        ),
        artifact_requirements=(ArtifactRequirementV1(kind="version_stamp"),),
        claim_class="diagnostic",
        source_commit="b" * 40,
        source_dirty=False,
        author="test",
        selection_rule="best_by_primary_then_smallest",
    )
    store.lock_experiment_campaign(manifest)
    store.append_event("experiment_started", experiment_id="ctrl", status="running")
    types = [row["event_type"] for row in store.verify_event_chain()]
    assert types.index("decision_arms_bound") < types.index(
        "experiment_campaign_locked"
    )
    assert types.index("experiment_campaign_locked") < types.index("experiment_started")


def test_size_match_skip_reason_typed() -> None:
    control = {"d_model": 128, "denoiser_layers": 4}
    wide = {"d_model": 256, "denoiser_layers": 4}
    reason = _mod._size_match_skip_reason(control, wide)
    assert reason is not None
    assert reason.startswith("capacity_unmatched:")
    assert _mod._size_match_skip_reason(control, control) is None
