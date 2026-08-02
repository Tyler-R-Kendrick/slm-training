"""Phase A positive classification: quality/latency tradeoffs, not naive speed."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_autotrain_continuous.py"
)
_SPEC = importlib.util.spec_from_file_location("run_autotrain_continuous", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)

_classify_metric_tradeoff = _mod._classify_metric_tradeoff
_PRIMARY = "smoke.latency_ms_p50"


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
    assert decode_timeout_seconds_for_role(policy, "screening") >= 20
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
            "component_edge_loss_weight": 1.0,
            "structural_aux_head_profile": "component-edge",
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
    assert cand["structural_aux_head_profile"] == "component-edge"
    assert ctrl["structural_aux_head_profile"] == "component-edge"
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


def test_select_recommended_slug_rotates_and_skips() -> None:
    # cycle 1 → first bank arm (bounds)
    assert _mod._select_recommended_slug(1) == "bounds"
    assert _mod._select_recommended_slug(2) == "canvas"
    assert _mod._select_recommended_slug(3) == "both"
    assert _mod._select_recommended_slug(1729) == "binder-topology"
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
    assert "c20260731-c2-component-plan" in ids
    assert "c20260731-c2-binder-topology" in ids
    assert "c20260731-c2-literal-close" in ids


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
    assert control["structural_aux_head_profile"] == "binder-topology"
    assert candidate["structural_aux_head_profile"] == "binder-topology"


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


def test_completed_null_labels_runtime_successor_as_diagnostic() -> None:
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

    assert priorities[0].area == "experiments"
    assert priorities[0].proposed_experiment_id.endswith("-bounds")
    assert "diagnostic" in priorities[0].hypothesis
    assert "latency_ms_p50" in priorities[0].hypothesis
    assert "quality hypothesis" not in priorities[0].hypothesis


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
                "positive": False,
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
                "climb_state": "rejected",
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

    assert _mod._recent_completed_nonpositive_slugs(root, predecessor) == {
        "component-plan",
        "component-edge",
    }
    assert _mod._recent_completed_nonpositive_slugs(
        root, predecessor, max_cycles=1
    ) == {"component-edge"}

    newest = root / str(predecessor) / "sdlc_delivery.json"
    newest_delivery = json.loads(newest.read_text())
    newest_delivery["measurement_complete"] = False
    newest.write_text(json.dumps(newest_delivery))
    assert _mod._recent_completed_nonpositive_slugs(root, predecessor) == {
        "component-plan"
    }


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
    )

    assert path is None
    assert err == "promote_certify_failed:checker timed out within 120s cap"
    assert not (camp / "metric-certificate.json").exists()


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


def test_arm_wall_budget_is_symmetric_and_reserves_orchestration() -> None:
    from slm_training.levers import (
        HARNESS_FINALIZATION_RESERVE_SECONDS,
        MAX_HARNESS_WALL_SECONDS,
    )

    arm_minutes = _mod._arm_wall_minutes(3)
    expected = min(
        3.0,
        (MAX_HARNESS_WALL_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS) / 3 / 60,
    )
    assert arm_minutes == pytest.approx(expected)
    assert _mod._arm_wall_minutes(0.5) == pytest.approx(
        min(
            0.5,
            (MAX_HARNESS_WALL_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS) / 3 / 60,
        )
    )
    reserved = 2 * arm_minutes * 60 + HARNESS_FINALIZATION_RESERVE_SECONDS
    assert reserved < MAX_HARNESS_WALL_SECONDS


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


def test_frozen_replay_preserves_recipe_and_links_current_main_successor() -> None:
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
        recommended_slug="batch1",
    )
    old_control = json.loads(json.dumps(matrix["hypotheses"][0]["experiment"]))
    old_candidate = json.loads(
        json.dumps(
            next(
                row["experiment"]
                for row in matrix["hypotheses"]
                if row["experiment"]["experiment_id"].endswith("-batch1")
            )
        )
    )
    old_control.update(
        experiment_id="c20260801-loop-12345678-c1710-control",
        campaign_id=old_campaign,
    )
    old_candidate.update(
        experiment_id="c20260801-loop-12345678-c1710-batch1",
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
    with pytest.raises(RuntimeError, match="fresh Lean preflight"):
        _mod._require_automatic_replayable(formal_manifest)


def test_frozen_replay_finds_completed_train_across_retry_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "autoresearch"
    source_campaign = "cycle-1"
    retry_campaign = "cycle-2"
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
    (retry_dir / "campaign.json").write_text(
        _mod.CampaignSpec(
            campaign_id=retry_campaign,
            objective="fixture replay",
            primary_metric="smoke.parse_rate",
            loop_id="loop-1",
            cycle_index=2,
            predecessor_campaign_id=source_campaign,
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
        cycle_intent="retry_measurement",
        primary_metric="smoke.binder_reference_f1",
        matrix=_priority_matrix(),
        delivery={
            "positive": False,
            "candidate_id": "cand",
            "reasons": ["measurement_incomplete:decode_timeout"],
            "stack_layer": False,
        },
        resolution=None,
        formal_status=None,
    )

    repair = next(
        action for action in handoff.actions if action.kind == "repair_harness"
    )
    retry = next(
        action for action in handoff.actions if action.kind == "retry_measurement"
    )
    assert repair.owner == "improve-openui-harnesses"
    assert repair.harness_family == "model_build"
    assert "finalized every record disposition" in repair.reason
    assert repair.frozen_manifest_sha256 == hashlib.sha256(b"{}\n").hexdigest()
    assert retry.frozen_manifest_sha256 == repair.frozen_manifest_sha256
    assert handoff.actions.index(repair) < handoff.actions.index(retry)


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
    repair = next(
        action for action in handoff.actions if action.kind == "repair_harness"
    )
    assert "do not replay" in repair.reason
    assert _mod._predecessor_priority_slug(root, "cycle-1", skip=set()) == (
        "literal-close"
    )


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
