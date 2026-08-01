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
    assert any("formal_preflight" in r or "certificate" in r for r in (fail.get("resolve_reasons") or []))


def _v2_cert(*, exp_sha: str, authority: str = "assumption_backed", relation: str = "in_band") -> dict:
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


def test_dispose_champion_promote_in_band_v2_promotes() -> None:
    exp_sha = _mod.locked_promote_expectations_sha256()
    d = _mod.dispose_champion_promote(
        formal_preflight_status="proved",
        certificate=_v2_cert(exp_sha=exp_sha, relation="in_band"),
        locked_expectations_sha256=exp_sha,
        phase_a_positive=True,
        phase_a_quality_held=True,
    )
    assert d["status"] == "promoted"
    assert d["cert_policy"] == "continue"
    assert d["emit_five_lane_matrix"] is False


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
        certificate=_v2_cert(
            exp_sha=exp_sha, authority="theorem", relation="below"
        ),
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
        "knobs": {"seed": 7, "eval_version": "e_test", "grammar_completion_bounds": True},
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


def test_resolve_promotion_with_in_band_cert_promotes(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-cert"
    camp = root / "c-cert"
    camp.mkdir(parents=True)
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
    resolved = _mod._resolve_promotion_result(
        root=root,
        loop_id=loop_id,
        entry=entry,
        delivery={
            "positive": True,
            "reasons": ["quality_held:parse=1.0 mpr=1.0"],
        },
        campaign_id="c-cert",
        cycle_index=9,
        camp_dir=camp,
    )
    assert resolved is not None
    assert resolved["status"] == "promoted"
    ledger = root / "loops" / loop_id / "learning_certificate_ledger.jsonl"
    assert ledger.is_file()
    line = ledger.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert "promoted" in line


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
