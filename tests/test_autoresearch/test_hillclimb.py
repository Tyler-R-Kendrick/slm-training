"""Hill-climb governance gates (claim class, exhausted knobs, synthesis, EG_params)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.experiment_campaign import (
    ExperimentCampaignV1,
    campaign_manifest_sha256,
    validate_result_claim,
)
from slm_training.autoresearch.hillclimb import (
    ExhaustedKnobLedger,
    HillClimbError,
    assert_capacity_growth_allowed,
    assert_climb_label_allowed,
    assert_matrix_knobs_not_exhausted,
    assert_synthesis_feedback_cleared_for_sft,
    climb_step_eligibility,
    data_eval_identity,
    knob_signature_sha256,
    open_synthesis_recommendations,
    record_null_outcome,
    validate_causal_campaign_shape,
)
from tests.test_autoresearch.test_experiment_campaign import (
    _complete_result,
    _manifest,
    _manifest_payload,
)


def test_fixture_cannot_be_labeled_climb_step() -> None:
    result = climb_step_eligibility(
        claim_class="fixture",
        locked_eval_manifest_sha256=None,
        seeds=(0, 1, 2),
        label_as_climb=True,
    )
    assert not result.eligible
    assert any("claim_class_not_climb" in f for f in result.failures)
    with pytest.raises(HillClimbError, match="claim_class_not_climb"):
        assert_climb_label_allowed(
            claim_class="fixture",
            locked_eval_manifest_sha256=None,
            seeds=(0, 1),
            label_as_climb=True,
        )


def test_promotion_class_with_locked_eval_and_seeds_is_climb_eligible() -> None:
    result = climb_step_eligibility(
        claim_class="promotion_candidate",
        locked_eval_manifest_sha256="e" * 64,
        seeds=(0, 1),
        primary_endpoint_seed_values=(0.12, 0.14),
        baseline_primary=0.0,
        minimum_effect=0.01,
        label_as_climb=True,
    )
    assert result.eligible, result.failures


def test_promotion_class_without_primary_seed_values_is_not_eligible() -> None:
    result = climb_step_eligibility(
        claim_class="promotion_candidate",
        locked_eval_manifest_sha256="e" * 64,
        seeds=(0, 1),
        primary_endpoint_seed_values=None,
        label_as_climb=True,
    )
    assert not result.eligible
    assert "primary_seed_values_missing" in result.failures


def test_validate_result_claim_requires_seed_values_for_promotion(tmp_path: Path) -> None:
    """Promotion-class climb eligibility is default-on (label_as_climb not needed)."""
    manifest = _manifest()
    result = _complete_result(
        manifest, tmp_path, primary_endpoint_seed_values=()
    )
    failures = validate_result_claim(manifest, result, artifact_root=tmp_path)
    assert any("primary_seed_values_missing" in f for f in failures)


def test_validate_result_claim_rejects_fixture_when_label_as_climb() -> None:
    fixture = _manifest(claim_class="fixture", locked_eval_manifest_sha256=None)
    # Fixture campaigns need fewer promotion artifacts — rebuild minimal fixture.
    fixture = ExperimentCampaignV1.model_validate(
        _manifest_payload(
            claim_class="fixture",
            locked_eval_manifest_sha256=None,
            artifact_requirements=[{"kind": "version_stamp", "minimum_count": 1}],
            mechanism_off_arm_ids=(),
            executable_kill_criteria=(),
            controls=[
                {
                    "control_id": "unchanged-baseline",
                    "description": "Unchanged baseline must reproduce.",
                    "kind": "negative",
                }
            ],
            negative_controls=["unchanged-baseline"],
            arms=[
                {
                    "arm_id": "control",
                    "role": "control",
                    "config_sha256": "c" * 64,
                },
                {
                    "arm_id": "candidate",
                    "role": "candidate",
                    "config_sha256": "d" * 64,
                },
            ],
        )
    )
    # Build a minimal result without full promotion artifacts.
    result_payload = {
        "campaign_id": fixture.campaign_id,
        "experiment_id": fixture.experiment_id,
        "manifest_sha256": campaign_manifest_sha256(fixture),
        "claim_class": "fixture",
        "exploratory": True,
    }
    from slm_training.autoresearch.experiment_campaign import CampaignResultV1

    result = CampaignResultV1.model_validate(result_payload)
    failures = validate_result_claim(fixture, result, label_as_climb=True)
    assert any(f.startswith("climb:") for f in failures)


def test_promotion_campaign_requires_causal_shape() -> None:
    with pytest.raises(ValidationError, match="causal shape|mechanism_off|kill"):
        _manifest(
            mechanism_off_arm_ids=(),
            executable_kill_criteria=(),
        )


def test_promotion_campaign_accepts_causal_shape() -> None:
    manifest = _manifest()
    assert manifest.mechanism_off_arm_ids == ("mechanism_off",)
    assert validate_causal_campaign_shape(manifest) == ()


def test_exhausted_knob_blocks_replay_same_identity() -> None:
    ledger = ExhaustedKnobLedger()
    knobs = {"lr": 3e-4, "steps": 200, "batch_size": 4}
    sig = knob_signature_sha256(knobs)
    identity = data_eval_identity(
        data_manifest_sha256="1" * 64,
        locked_eval_manifest_sha256="2" * 64,
        claim_class="fixture",
    )
    record_null_outcome(
        ledger,
        knobs=knobs,
        data_eval_identity=identity,
        claim_class="fixture",
    )
    with pytest.raises(HillClimbError, match="exhausted"):
        assert_matrix_knobs_not_exhausted(
            knob_signatures=[sig],
            ledger=ledger,
            data_eval_identity=identity,
            claim_class="fixture",
        )
    # Upgraded claim class is allowed.
    assert_matrix_knobs_not_exhausted(
        knob_signatures=[sig],
        ledger=ledger,
        data_eval_identity=identity,
        claim_class="promotion_candidate",
    )
    # Changed data/eval identity is allowed.
    other = data_eval_identity(
        data_manifest_sha256="3" * 64,
        locked_eval_manifest_sha256="2" * 64,
        claim_class="fixture",
    )
    assert_matrix_knobs_not_exhausted(
        knob_signatures=[sig],
        ledger=ledger,
        data_eval_identity=other,
        claim_class="fixture",
    )


def test_exhausted_ledger_roundtrip(tmp_path: Path) -> None:
    ledger = ExhaustedKnobLedger()
    record_null_outcome(
        ledger,
        knobs={"lr": 1e-3},
        data_eval_identity="abc",
        claim_class="fixture",
    )
    path = tmp_path / "exhausted_knob_ledger.json"
    ledger.save(path)
    restored = ExhaustedKnobLedger.load(path)
    assert restored.is_exhausted(
        knob_signature_sha256=knob_signature_sha256({"lr": 1e-3}),
        data_eval_identity="abc",
        claim_class="fixture",
    )


def test_synthesis_gate_blocks_open_recommendations(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "synthesis_feedback.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recommendations": [
                    {
                        "code": "low_yield",
                        "target_kind": "synthesizer",
                        "target": "foo",
                        "suggestion": "fix the producer",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(HillClimbError, match="open synthesis_feedback"):
        assert_synthesis_feedback_cleared_for_sft(train_dir)
    # Action record clears the gate.
    (train_dir / "synthesis_feedback_actions.json").write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "code": "low_yield",
                        "target": "foo",
                        "note": "reduced expansion volume",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = assert_synthesis_feedback_cleared_for_sft(train_dir)
    assert result.allowed
    assert result.reason == "actions_recorded"


def test_synthesis_gate_accepts_waiver(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "synthesis_feedback.json").write_text(
        json.dumps(
            {
                "recommendations": [
                    {"code": "redundant_expansion", "target": "bar"}
                ]
            }
        ),
        encoding="utf-8",
    )
    (train_dir / "synthesis_feedback_actions.json").write_text(
        json.dumps(
            {
                "waivers": [
                    {
                        "code": "redundant_expansion",
                        "reason": "diagnostic re-run only",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = assert_synthesis_feedback_cleared_for_sft(train_dir)
    assert result.allowed


def test_synthesis_gate_allows_missing_feedback(tmp_path: Path) -> None:
    result = assert_synthesis_feedback_cleared_for_sft(tmp_path)
    assert result.allowed
    assert result.reason == "no_feedback_artifact"


def test_synthesis_gate_fails_closed_on_corrupt_feedback(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "synthesis_feedback.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(HillClimbError, match="corrupt JSON"):
        assert_synthesis_feedback_cleared_for_sft(train_dir)


def test_synthesis_gate_rejects_star_in_actions(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "synthesis_feedback.json").write_text(
        json.dumps({"recommendations": [{"code": "low_yield"}]}),
        encoding="utf-8",
    )
    (train_dir / "synthesis_feedback_actions.json").write_text(
        json.dumps({"actions": [{"code": "*"}]}),
        encoding="utf-8",
    )
    with pytest.raises(HillClimbError, match="waivers only"):
        assert_synthesis_feedback_cleared_for_sft(train_dir)


def test_open_synthesis_recommendations_skips_closed() -> None:
    open_recs = open_synthesis_recommendations(
        {
            "recommendations": [
                {"code": "low_yield", "status": "done"},
                {"code": "eval_leakage_source"},
            ]
        }
    )
    assert len(open_recs) == 1
    assert open_recs[0]["code"] == "eval_leakage_source"


def test_capacity_growth_fails_without_eg_params() -> None:
    with pytest.raises(HillClimbError, match="capacity growth"):
        assert_capacity_growth_allowed(
            baseline_params=1_000_000,
            candidate_params=2_000_000,
            eg_params_by_seed=None,
        )


def test_capacity_growth_passes_with_eg_params_lcb() -> None:
    check = assert_capacity_growth_allowed(
        baseline_params=1_000_000,
        candidate_params=2_000_000,
        eg_params_by_seed=(1.2, 1.3, 1.1),
    )
    assert check["pass"]


def test_capacity_not_increased_always_passes() -> None:
    check = assert_capacity_growth_allowed(
        baseline_params=2_000_000,
        candidate_params=1_000_000,
        eg_params_by_seed=None,
    )
    assert check["pass"]


def test_validate_result_claim_eg_params_on_promotion(tmp_path: Path) -> None:
    manifest = _manifest()
    result = _complete_result(manifest, tmp_path)
    failures = validate_result_claim(
        manifest,
        result,
        artifact_root=tmp_path,
        baseline_trainable_params=100,
        candidate_trainable_params=200,
        eg_params_by_seed=None,
    )
    assert any("eg_params" in f for f in failures)
    ok = validate_result_claim(
        manifest,
        result,
        artifact_root=tmp_path,
        baseline_trainable_params=100,
        candidate_trainable_params=200,
        eg_params_by_seed=(1.5, 1.4),
    )
    assert not any("eg_params" in f for f in ok)


def test_validate_result_claim_eg_params_from_result_fields(tmp_path: Path) -> None:
    """Production paths can embed capacity evidence on the result object."""
    manifest = _manifest()
    result = _complete_result(
        manifest,
        tmp_path,
        baseline_trainable_params=100,
        candidate_trainable_params=200,
        eg_params_by_seed=(),
    )
    failures = validate_result_claim(manifest, result, artifact_root=tmp_path)
    assert any("eg_params" in f for f in failures)


def test_validate_hypothesis_matrix_rejects_exhausted_knob_signature() -> None:
    """Engine production hook: exhausted ledger blocks matrix validation."""
    from slm_training.autoresearch.engine import validate_hypothesis_matrix
    from tests.test_autoresearch.test_harness import (
        campaign,
        hypothesis_matrix,
        matrix_evidence,
        source,
    )

    matrix = hypothesis_matrix()
    first_knobs = matrix.hypotheses[0].experiment.knobs.model_dump(
        exclude_none=True, mode="json"
    )
    sig = knob_signature_sha256(first_knobs)
    identity = data_eval_identity(
        data_manifest_sha256="snap",
        locked_eval_manifest_sha256=None,
        claim_class="fixture",
    )
    ledger = ExhaustedKnobLedger()
    record_null_outcome(
        ledger,
        knobs=first_knobs,
        data_eval_identity=identity,
        claim_class="fixture",
    )
    with pytest.raises(ValueError, match="exhausted"):
        validate_hypothesis_matrix(
            campaign(),
            matrix,
            matrix_evidence(),
            [source()],
            exhausted_knob_ledger=ledger,
            exhausted_data_eval_identity=identity,
            exhausted_claim_class="fixture",
        )
    # Different identity still allowed.
    validate_hypothesis_matrix(
        campaign(),
        matrix,
        matrix_evidence(),
        [source()],
        exhausted_knob_ledger=ledger,
        exhausted_data_eval_identity=data_eval_identity(
            data_manifest_sha256="other",
            locked_eval_manifest_sha256=None,
            claim_class="fixture",
        ),
        exhausted_claim_class="fixture",
    )
    assert sig  # signature computed for ledger exhaust above


_POLARITY_CASES = [
    # --- increase (structural_similarity) ---
    (
        "held_out.structural_similarity",
        "increase",
        {"held_out.structural_similarity": 0.75},
        None,
        None,
        0.01,
        False,
        "increase_absolute_alone_not_null",
    ),
    (
        "held_out.structural_similarity",
        "increase",
        {"held_out.structural_similarity": 0.80},
        0.75,
        None,
        0.01,
        False,
        "increase_absolute_win",
    ),
    (
        "held_out.structural_similarity",
        "increase",
        {"held_out.structural_similarity": 0.75},
        0.75,
        None,
        0.01,
        True,
        "increase_absolute_tie_null",
    ),
    (
        "held_out.structural_similarity",
        "increase",
        {"held_out.structural_similarity": 0.70},
        0.75,
        None,
        0.01,
        True,
        "increase_absolute_regression_null",
    ),
    (
        "held_out.structural_similarity",
        "increase",
        {"held_out.structural_similarity": 0.80, "primary_delta": 0.05},
        None,
        None,
        0.01,
        False,
        "increase_explicit_delta_win",
    ),
    (
        "held_out.structural_similarity",
        "increase",
        {"held_out.structural_similarity": 0.75, "primary_delta": 0.0},
        None,
        None,
        0.01,
        True,
        "increase_explicit_delta_null",
    ),
    (
        "held_out.structural_similarity",
        "increase",
        {"held_out.structural_similarity": 0.76},
        0.75,
        (0.74, 0.76),
        0.05,
        True,
        "increase_seed_bound_within_noise",
    ),
    # --- decrease (continuous default smoke.latency_ms_p50) ---
    (
        "smoke.latency_ms_p50",
        "decrease",
        {"smoke.latency_ms_p50": 10.0},
        None,
        None,
        0.0,
        False,
        "decrease_absolute_alone_not_null",
    ),
    (
        "smoke.latency_ms_p50",
        "decrease",
        {"smoke.latency_ms_p50": 10.0},
        11.0,
        None,
        0.0,
        False,
        "decrease_latency_win_not_null",
    ),
    (
        "smoke.latency_ms_p50",
        "decrease",
        {"smoke.latency_ms_p50": 11.0},
        11.0,
        None,
        0.0,
        True,
        "decrease_latency_tie_null",
    ),
    (
        "smoke.latency_ms_p50",
        "decrease",
        {"smoke.latency_ms_p50": 12.5},
        11.0,
        None,
        0.0,
        True,
        "decrease_latency_regression_null",
    ),
    (
        "smoke.latency_ms_p50",
        "decrease",
        {"smoke.latency_ms_p50": 10.0, "primary_delta": -1.0},
        None,
        None,
        0.0,
        False,
        "decrease_explicit_delta_win",
    ),
    (
        "smoke.latency_ms_p50",
        "decrease",
        {"smoke.latency_ms_p50": 12.5, "primary_delta": 1.5},
        None,
        None,
        0.0,
        True,
        "decrease_explicit_delta_regression",
    ),
    (
        "smoke.latency_ms_p50",
        "decrease",
        {"smoke.latency_ms_p50": 9.2},
        11.0,
        # Tight low seeds so UCB stays below control 11.
        (9.0, 9.2),
        0.0,
        False,
        "decrease_seed_bound_win",
    ),
    (
        "smoke.latency_ms_p50",
        "decrease",
        {"smoke.latency_ms_p50": 12.0},
        11.0,
        (11.5, 12.5),
        0.0,
        True,
        "decrease_seed_bound_regression_null",
    ),
]


@pytest.mark.parametrize(
    "primary_metric,direction,metrics,baseline,seeds,minimum_effect,expect_null,label",
    _POLARITY_CASES,
    ids=[c[-1] for c in _POLARITY_CASES],
)
def test_measured_null_polarity_table(
    primary_metric: str,
    direction: str,
    metrics: dict,
    baseline: float | None,
    seeds: tuple[float, ...] | None,
    minimum_effect: float,
    expect_null: bool,
    label: str,
) -> None:
    """Table: (increase|decrease) × (win|null|regression) × (abs|delta|seeds)."""
    from slm_training.autoresearch.hillclimb import (
        infer_metric_direction,
        is_measured_null_feedback,
    )

    assert infer_metric_direction(primary_metric) == direction
    got = is_measured_null_feedback(
        outcome_status="completed",
        metrics=metrics,
        primary_metric=primary_metric,
        direction=direction,  # type: ignore[arg-type]
        baseline_primary=baseline,
        primary_seed_values=seeds,
        minimum_effect=minimum_effect,
    )
    assert got is expect_null, f"{label}: expected null={expect_null}, got {got}"


def test_absolute_lookup_ignores_baseline_prefixed_keys() -> None:
    """baseline.smoke.latency_ms_p50 must not be read as the candidate absolute."""
    from slm_training.autoresearch.hillclimb import resolve_primary_effect

    improvement, source = resolve_primary_effect(
        {
            "smoke.latency_ms_p50": 10.0,
            "baseline.smoke.latency_ms_p50": 11.0,
            "baseline_primary": 11.0,
        },
        primary_metric="smoke.latency_ms_p50",
        direction="decrease",
    )
    assert source == "absolute_vs_baseline"
    # Win: 10 vs 11 → improvement +1
    assert improvement == pytest.approx(1.0)


def test_feedback_path_latency_null_and_win(tmp_path: Path) -> None:
    """Shipped autoresearch path: continuous default latency polarity."""
    import scripts.autoresearch as autoresearch_cli
    from slm_training.autoresearch.hillclimb import (
        load_exhausted_ledger,
        matrix_data_eval_identity,
    )
    from slm_training.autoresearch.schemas import (
        CampaignSpec,
        Diagnosis,
        ExperimentOutcome,
    )
    from slm_training.autoresearch.storage import CampaignStore
    from tests.test_autoresearch.test_harness import hypothesis_matrix

    root = tmp_path / "autoresearch"
    campaign = CampaignSpec(
        campaign_id="latency-loop",
        objective="Reduce smoke latency versus the matched control.",
        primary_metric="smoke.latency_ms_p50",
        researcher_mode="fixture",
    )
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(campaign)
    matrix = hypothesis_matrix(campaign_id=campaign.campaign_id)
    control = ExperimentOutcome(
        experiment_id="control-matched",
        campaign_id=campaign.campaign_id,
        status="completed",
        metrics={"smoke.latency_ms_p50": 11.0, "held_out.structural_similarity": 0.75},
    )
    store.write_artifact("outcomes", control)

    # Regression 12.5 vs control 11 → null → exhaust.
    regression = ExperimentOutcome(
        experiment_id=matrix.hypotheses[0].experiment.experiment_id,
        campaign_id=campaign.campaign_id,
        status="completed",
        metrics={"smoke.latency_ms_p50": 12.5, "held_out.structural_similarity": 0.75},
    )
    diag = Diagnosis(
        experiment_id=regression.experiment_id,
        target="researcher",
        confidence=0.7,
        evidence=("latency regressed versus matched control",),
        recommended_actions=("do not retry this knob signature",),
    )
    autoresearch_cli._record_hypothesis_feedback(store, matrix, regression, diag)
    identity = matrix_data_eval_identity(
        evidence_snapshot_id=matrix.evidence_snapshot_id,
        primary_metric=campaign.primary_metric,
        claim_class="fixture",
    )
    knobs0 = matrix.hypotheses[0].experiment.knobs.model_dump(
        exclude_none=True, mode="json"
    )
    assert load_exhausted_ledger(store.root).is_exhausted(
        knob_signature_sha256=knob_signature_sha256(knobs0),
        data_eval_identity=identity,
        claim_class="fixture",
    )

    # Win 10 vs control 11 → not null → do not exhaust.
    win = ExperimentOutcome(
        experiment_id=matrix.hypotheses[1].experiment.experiment_id,
        campaign_id=campaign.campaign_id,
        status="completed",
        metrics={"smoke.latency_ms_p50": 10.0, "held_out.structural_similarity": 0.75},
    )
    win_diag = Diagnosis(
        experiment_id=win.experiment_id,
        target="none",
        confidence=0.6,
        evidence=("latency improved versus matched control",),
        recommended_actions=("retain as evidence",),
    )
    autoresearch_cli._record_hypothesis_feedback(store, matrix, win, win_diag)
    knobs1 = matrix.hypotheses[1].experiment.knobs.model_dump(
        exclude_none=True, mode="json"
    )
    assert not load_exhausted_ledger(store.root).is_exhausted(
        knob_signature_sha256=knob_signature_sha256(knobs1),
        data_eval_identity=identity,
        claim_class="fixture",
    )


def test_evaluate_promotion_passes_eg_params_to_governance(tmp_path: Path) -> None:
    """Real evaluate_promotion path surfaces capacity into validate_result_claim."""
    from slm_training.harnesses.experiments.promotion import evaluate_promotion

    manifest = _manifest()
    result = _complete_result(
        manifest,
        tmp_path,
        baseline_trainable_params=None,
        candidate_trainable_params=None,
    )
    # Growing capacity without EG_params fails campaign governance.
    out = evaluate_promotion(
        campaign_manifest=manifest,
        campaign_result=result,
        artifact_root=tmp_path,
        baseline_trainable_params=100,
        candidate_trainable_params=200,
        eg_params_by_seed=None,
        ship_suites={
            "smoke": {
                "n": 32,
                "parse_rate": 1.0,
                "structural_similarity": 1.0,
                "component_type_recall": 1.0,
                "component_type_precision": 1.0,
                "slot_exact": 1.0,
            }
        },
    )
    gov = out["checks"]["campaign_governance"]
    assert gov["pass"] is False
    assert any("eg_params" in f for f in gov["failures"])
