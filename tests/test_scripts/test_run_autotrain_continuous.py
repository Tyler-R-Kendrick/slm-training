"""Phase A positive classification: quality/latency tradeoffs, not naive speed."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

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
        stage_wall_minutes_for_role,
    )

    policy = load_climb_policy()
    assert stage_wall_minutes_for_role(policy, "screening") == 3
    assert decode_timeout_seconds_for_role(policy, "screening") >= 20
    assert eval_suites_for_role(policy, "screening") == ("smoke",)


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


def test_select_recommended_slug_rotates_and_skips() -> None:
    # cycle 1 → first bank arm (bounds)
    assert _mod._select_recommended_slug(1) == "bounds"
    assert _mod._select_recommended_slug(2) == "canvas"
    assert _mod._select_recommended_slug(3) == "both"
    # skip bounds → canvas even on cycle 1
    assert _mod._select_recommended_slug(1, skip={"bounds"}) == "canvas"
    # all skipped → still returns rotated head
    all_slugs = {slug for slug, _, _ in _mod._SCREENING_ARM_BANK}
    assert _mod._select_recommended_slug(1, skip=all_slugs) == "bounds"


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
        },
    )
    HypothesisMatrix.model_validate(matrix)
    assert matrix["recommended_experiment_id"] == "c20260731-c8-promote"
    cand = matrix["hypotheses"][1]["experiment"]["knobs"]
    assert cand["grammar_completion_bounds"] is True
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
    assert int(rows[0].get("promote_attempts") or 0) == 0
    head = _mod._queue_head_confirmed(rows)
    assert head is not None and head["entry_id"] == "champ-hf-1"
    ledger = (root / "loops" / loop / "learning_certificate_ledger.jsonl").read_text()
    assert "harness_failure" in ledger


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
    assert any("promote_parse_regression" in r for r in d["reasons"])


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


def test_resolve_promotion_with_in_band_cert_promotes(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-cert"
    camp = root / "c-cert"
    camp.mkdir(parents=True)
    _write_run_eval(camp, "c-control", structural_similarity=0.10)
    _write_run_eval(camp, "c-promote", structural_similarity=0.20)
    exp_sha = _mod.locked_promote_expectations_sha256()
    cert = _v2_cert(exp_sha=exp_sha, relation="in_band")
    (camp / "metric-certificate.json").write_text(
        __import__("json").dumps(cert), encoding="utf-8"
    )
    _mod.record_formal_preflight_status(
        camp,
        status="proved",
        template_id=_mod._PROMOTE_FORMAL_TEMPLATE_ID,
    )
    path = _mod._champion_queue_path(root, loop_id)
    entry = {
        "schema": _mod._CHAMPION_QUEUE_SCHEMA,
        "entry_id": "champ-ok",
        "status": "promoting",
        "knobs": {"grammar_completion_bounds": True},
        "knobs_fingerprint": "fpok",
    }
    _mod._write_champion_queue(path, [entry])
    control, candidate = _held_out_win_metrics()
    resolved = _mod._resolve_promotion_result(
        root=root,
        loop_id=loop_id,
        entry=entry,
        delivery={
            "positive": True,
            "reasons": ["quality_held:parse=1.0 mpr=1.0"],
            "control_id": "c-control",
            "candidate_id": "c-promote",
            "control_metrics": control,
            "candidate_metrics": candidate,
        },
        campaign_id="c-cert",
        cycle_index=9,
        camp_dir=camp,
        arm_exits={"c-control": 0, "c-promote": 0},
    )
    assert resolved is not None
    assert resolved["status"] == "climb_accepted"
    ledger = root / "loops" / loop_id / "learning_certificate_ledger.jsonl"
    assert ledger.is_file()
    line = ledger.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert "climb_accepted" in line
    assert "promote_primary_win" in line or "primary_improvement" in line


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


def test_ensure_promote_formal_preflight_content_addressed_when_recorded(
    tmp_path: Path,
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
    status, sha = _mod.ensure_promote_formal_preflight(
        camp_dir=camp, campaign_id="c1", experiment_id="e1", run_lean=False
    )
    assert status == "proved"
    assert sha == content_sha
    assert (art / f"{sha}.json").is_file()


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


def test_is_champion_lever_includes_steps_and_batch1() -> None:
    assert _mod._is_champion_lever(
        {"grammar_completion_bounds": True}, candidate_id="x-bounds"
    )
    assert _mod._is_champion_lever({"batch_size": 1}, candidate_id="x-batch1")
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
    )
    assert err is None, err
    assert path is not None and path.is_file()
    cert = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert cert["schema"] == "metric_certificate/v2"
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
    )
    assert path is None
    assert err is not None
    assert "incomplete_metrics" in err or "checker_missing" in err


def test_rate_to_pm_and_latency_helpers() -> None:
    assert _mod._rate_to_pm(0.5) == 500
    assert _mod._rate_to_pm(1.0) == 1000
    assert _mod._rate_to_pm(None) is None
    assert _mod._latency_ms_to_ns(1.0) == 1_000_000


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


def test_classify_positive_promotion_sees_held_out_primary(tmp_path: Path) -> None:
    """Regression: promote primary must not be unavailable when held_out eval exists."""
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
        # No gates.json → avoid fixture_insufficient_n noise for this unit test.
    result = _mod._classify_positive(
        camp_dir=camp,
        primary_metric="held_out.structural_similarity",
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


def test_promote_formal_timeout_obeys_repository_cap() -> None:
    from slm_training.levers import MAX_RUN_SECONDS

    assert _mod._PROMOTE_FORMAL_TIMEOUT_S == float(MAX_RUN_SECONDS)


def test_cycle_deadline_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mod.time, "monotonic", lambda: 10.0)
    with pytest.raises(subprocess.TimeoutExpired, match="autotrain bounded cycle"):
        _mod._remaining_timeout(9.0)


def test_arm_wall_budget_is_symmetric_and_reserves_orchestration() -> None:
    from slm_training.levers import MAX_HARNESS_WALL_SECONDS

    arm_minutes = _mod._arm_wall_minutes(3)
    assert arm_minutes * 60 * 3 == pytest.approx(MAX_HARNESS_WALL_SECONDS)
    assert _mod._arm_wall_minutes(0.5) == 0.5


def test_driver_requires_room_for_both_arms_before_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_mod.time, "monotonic", lambda: 0.0)
    _mod._require_symmetric_arm_budget(
        deadline=120.0, arm_count=2, arm_wall_minutes=0.75
    )
    with pytest.raises(subprocess.TimeoutExpired, match="symmetric decision-arm"):
        _mod._require_symmetric_arm_budget(
            deadline=100.0, arm_count=2, arm_wall_minutes=0.75
        )


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
        },
        resolution={"status": "climb_accepted", "resolve_reasons": []},
        formal_status="proved",
    )
    assert handoff.climb_state == "climb_accepted"
    assert handoff.ship_state == "blocked"
    assert handoff.evidence_class == "fixture"
    assert handoff.checkpoint_paths == ("runs/cand/last.pt",)
    assert handoff.checkpoint_documentation_required is True
    assert "MODEL_CARD" in next(
        action.reason for action in handoff.actions if action.kind == "document"
    )
    assert {action.kind for action in handoff.actions} == {
        "document",
        "deliver_stack",
        "next_experiment",
    }
    state = json.loads((root / "loops" / "loop-1" / "state.json").read_text())
    assert state["phase"] == "between_cycles"


def test_cycle_handoff_routes_frozen_harness_repair(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    camp = root / "cycle-1"
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
    (camp / "manifests" / "cand.json").write_text("{}\n")
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
    assert all(action.kind != "next_experiment" for action in handoff.actions)


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


def test_continuous_evidence_is_bounded_to_predecessor_and_loop(tmp_path: Path) -> None:
    roots = _mod._continuous_evidence_roots(
        tmp_path / "autoresearch", "loop-1", "campaign-6"
    )
    assert roots == (
        tmp_path / "autoresearch" / "campaign-6",
        tmp_path / "autoresearch" / "loops" / "loop-1",
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
