"""Tests for versioned climb policy + continuous/loop climb gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.autoresearch.climb_policy import (
    CLIMB_RESOURCE_DIR,
    assert_cycle_cadence,
    assert_recipe_null_regime_pressure,
    assert_rung_allows_promotion,
    classify_positive_metrics,
    climb_policy_content_digest,
    cycle_role_for_index,
    eval_suites_for_role,
    is_recipe_tweak_knobs,
    load_climb_policy,
    load_loop_exhausted_ledger,
    loop_data_eval_identity,
    primary_for_role,
    promotion_primary_effect_met,
    save_loop_exhausted_ledger,
    screening_nll_definition_hash,
    screening_smoke_n_for_policy,
)
from slm_training.autoresearch.hillclimb import (
    ExhaustedKnobLedger,
    HillClimbError,
    record_null_outcome,
)


def test_load_climb_policy_has_volatile_fields_and_digest() -> None:
    policy = load_climb_policy()
    assert policy.schema == "autotrain_climb_policy/v1"
    assert policy.version
    assert len(policy.sha256) == 64
    assert policy.screening_primary["metric"] == "smoke.eval_nll"
    assert policy.screening_primary["direction"] == "decrease"
    assert policy.promotion_primary["metric"]
    assert policy.promotion_dispose["require_primary_win"] is True
    assert policy.positive_classification["minimum_efficiency_gain_fraction"] == 0.05
    assert policy.cadence["screening_cycles_per_promotion"]
    assert policy.exhausted_identity_fields
    assert policy.recipe_tweak_knobs
    terminal = policy.payload.get("terminal")
    assert isinstance(terminal, dict)
    assert terminal["park_on_exhaust"] is True
    digest1 = climb_policy_content_digest(policy.payload)
    mutated = dict(policy.payload)
    screening = dict(mutated["screening_primary"])
    screening["minimum_effect"] = float(screening.get("minimum_effect") or 0) + 1.0
    mutated["screening_primary"] = screening
    digest2 = climb_policy_content_digest(mutated)
    assert digest1 != digest2


def test_promotion_primary_effect_met_null_and_win() -> None:
    policy = load_climb_policy()
    primary = dict(policy.promotion_primary)
    ok, reasons, delta = promotion_primary_effect_met(
        control_metrics={"held_out.structural_similarity": 0.2, "parse_rate": 1.0},
        candidate_metrics={"held_out.structural_similarity": 0.2, "parse_rate": 1.0},
        promotion_primary=primary,
    )
    assert ok is False
    assert delta == 0.0
    assert any("promote_primary_null_or_insufficient" in r for r in reasons)

    ok2, reasons2, delta2 = promotion_primary_effect_met(
        control_metrics={"structural_similarity": 0.1, "parse_rate": 1.0},
        candidate_metrics={"structural_similarity": 0.25, "parse_rate": 1.0},
        promotion_primary=primary,
    )
    assert ok2 is True
    assert delta2 is not None and delta2 > 0.01
    assert any(r.startswith("promote_primary_win:") for r in reasons2)

    ok3, reasons3, _ = promotion_primary_effect_met(
        control_metrics={},
        candidate_metrics={},
        promotion_primary=primary,
    )
    assert ok3 is False
    assert any("promote_primary_metrics_missing" in r for r in reasons3)


def test_cycle_cadence_screening_then_promotion() -> None:
    policy = load_climb_policy()
    n = int(policy.cadence["screening_cycles_per_promotion"])
    # First n cycles screening, then promotion
    for i in range(1, n + 1):
        assert cycle_role_for_index(policy, i) == "screening"
        assert_cycle_cadence(policy, cycle_index=i, claimed_role="screening")
    assert cycle_role_for_index(policy, n + 1) == "promotion"
    assert_cycle_cadence(policy, cycle_index=n + 1, claimed_role="promotion")
    fallback_role = assert_cycle_cadence(
        policy,
        cycle_index=n + 1,
        claimed_role="screening",
        claim_class="diagnostic",
        promotion_target_available=False,
    )
    assert fallback_role == "screening"
    assert eval_suites_for_role(policy, fallback_role) == ("smoke",)
    with pytest.raises(HillClimbError, match="cadence violation"):
        assert_cycle_cadence(
            policy,
            cycle_index=n + 1,
            claimed_role="screening",
            promotion_target_available=True,
        )
    confirmation_role = assert_cycle_cadence(
        policy,
        cycle_index=n + 1,
        claimed_role="screening",
        claim_class="diagnostic",
        promotion_target_available=True,
        confirmation_pending=True,
    )
    assert confirmation_role == "screening"
    assert eval_suites_for_role(policy, confirmation_role) == ("smoke",)
    with pytest.raises(HillClimbError, match="cadence violation"):
        assert_cycle_cadence(policy, cycle_index=1, claimed_role="promotion")
    with pytest.raises(HillClimbError, match="promotion-role"):
        assert_cycle_cadence(
            policy,
            cycle_index=n + 1,
            claimed_role="promotion",
            claim_class="fixture",
        )


def test_classify_positive_screening_quality_win_and_regression() -> None:
    policy = load_climb_policy(str(CLIMB_RESOURCE_DIR / "policy.v1.json"))
    # Structure is the primary; parse and binder correctness must not regress.
    win = classify_positive_metrics(
        policy,
        role="screening",
        control_metrics={
            "structural_similarity": 0.4,
            "binder_reference_f1": 1.0,
            "parse_rate": 1.0,
        },
        candidate_metrics={
            "structural_similarity": 0.6,
            "binder_reference_f1": 1.0,
            "parse_rate": 1.0,
        },
    )
    assert win["positive"] is True
    assert any(r.startswith("primary_metric_win") for r in win["reasons"])

    fixture_tick = classify_positive_metrics(
        policy,
        role="screening",
        control_metrics={
            "structural_similarity": 0.174,
            "binder_reference_f1": 0.63,
            "parse_rate": 1.0,
        },
        candidate_metrics={
            "structural_similarity": 0.214,
            "binder_reference_f1": 0.63,
            "parse_rate": 1.0,
        },
        fixture_insufficient_n=True,
    )
    assert fixture_tick["positive"] is False
    assert any(r.startswith("primary_metric_win") for r in fixture_tick["reasons"])
    assert "fixture_insufficient_n_alone" in fixture_tick["reasons"]

    # Lower structure is not positive.
    bad = classify_positive_metrics(
        policy,
        role="screening",
        control_metrics={
            "structural_similarity": 0.6,
            "binder_reference_f1": 1.0,
            "parse_rate": 1.0,
        },
        candidate_metrics={
            "structural_similarity": 0.4,
            "binder_reference_f1": 1.0,
            "parse_rate": 1.0,
        },
    )
    assert bad["positive"] is False
    assert any("null_or_worse" in r for r in bad["reasons"])

    illegal = classify_positive_metrics(
        policy,
        role="screening",
        control_metrics={
            "structural_similarity": 0.4,
            "binder_reference_f1": 1.0,
            "parse_rate": 1.0,
        },
        candidate_metrics={
            "structural_similarity": 0.9,
            "binder_reference_f1": 1.0,
            "parse_rate": 0.0,
        },
    )
    assert illegal["positive"] is False
    assert any(r.startswith("invalid_grammar:") for r in illegal["reasons"])


def test_classify_positive_capacity_growth_without_eg_params() -> None:
    policy = load_climb_policy(str(CLIMB_RESOURCE_DIR / "policy.v1.json"))
    result = classify_positive_metrics(
        policy,
        role="screening",
        control_metrics={
            "structural_similarity": 0.4,
            "binder_reference_f1": 1.0,
            "parse_rate": 1.0,
        },
        candidate_metrics={
            "structural_similarity": 0.6,
            "binder_reference_f1": 1.0,
            "parse_rate": 1.0,
        },
        baseline_trainable_params=1_000_000,
        candidate_trainable_params=2_000_000,
        eg_params_by_seed=None,
    )
    assert result["positive"] is False
    assert any("eg_params" in r for r in result["reasons"])


def test_classify_positive_capacity_growth_with_eg_params() -> None:
    policy = load_climb_policy(str(CLIMB_RESOURCE_DIR / "policy.v1.json"))
    result = classify_positive_metrics(
        policy,
        role="screening",
        control_metrics={
            "structural_similarity": 0.4,
            "binder_reference_f1": 1.0,
            "parse_rate": 1.0,
        },
        candidate_metrics={
            "structural_similarity": 0.6,
            "binder_reference_f1": 1.0,
            "parse_rate": 1.0,
        },
        baseline_trainable_params=1_000_000,
        candidate_trainable_params=2_000_000,
        eg_params_by_seed=(1.2, 1.2),
    )
    assert result["positive"] is True


def test_loop_identity_changes_with_eval_or_claim() -> None:
    policy = load_climb_policy()
    a = loop_data_eval_identity(
        policy,
        claim_class="fixture",
        train_version="wf_smoke_v2",
        eval_version="e938",
        primary_metric="smoke.latency_ms_p50",
        direction="decrease",
    )
    b = loop_data_eval_identity(
        policy,
        claim_class="fixture",
        train_version="wf_smoke_v2",
        eval_version="other",
        primary_metric="smoke.latency_ms_p50",
        direction="decrease",
    )
    c = loop_data_eval_identity(
        policy,
        claim_class="promotion_candidate",
        train_version="wf_smoke_v2",
        eval_version="e938",
        primary_metric="smoke.latency_ms_p50",
        direction="decrease",
    )
    assert a != b
    assert a != c


def test_loop_exhausted_rejects_same_identity(tmp_path: Path) -> None:
    policy = load_climb_policy()
    root = tmp_path / "autoresearch"
    loop_id = "loop-1"
    identity = loop_data_eval_identity(
        policy,
        claim_class="fixture",
        train_version="wf_smoke_v2",
        eval_version="e938",
        primary_metric="smoke.latency_ms_p50",
        direction="decrease",
    )
    ledger = ExhaustedKnobLedger()
    knobs = {"steps": 20, "lr": 3e-4}
    record_null_outcome(
        ledger,
        knobs=knobs,
        data_eval_identity=identity,
        claim_class="fixture",
        reason="recipe_null",
        note="recipe_family",
    )
    save_loop_exhausted_ledger(ledger, root, loop_id, policy)
    restored = load_loop_exhausted_ledger(root, loop_id, policy)
    from slm_training.autoresearch.hillclimb import (
        assert_matrix_knobs_not_exhausted,
        knob_signature_sha256,
    )

    with pytest.raises(HillClimbError, match="exhausted"):
        assert_matrix_knobs_not_exhausted(
            knob_signatures=[knob_signature_sha256(knobs)],
            ledger=restored,
            data_eval_identity=identity,
            claim_class="fixture",
        )
    # Claim-class upgrade allowed
    assert_matrix_knobs_not_exhausted(
        knob_signatures=[knob_signature_sha256(knobs)],
        ledger=restored,
        data_eval_identity=identity,
        claim_class="promotion_candidate",
    )


def test_recipe_null_cap_requires_regime_transition() -> None:
    policy = load_climb_policy()
    cap = int(policy.recipe_null_cap["max_nulls_per_family"])
    identity = "abc"
    ledger = ExhaustedKnobLedger()
    for i in range(cap):
        record_null_outcome(
            ledger,
            knobs={"steps": 10 + i, "lr": 1e-3},
            data_eval_identity=identity,
            claim_class="fixture",
            reason="recipe_null",
            note="recipe_family",
        )
    with pytest.raises(HillClimbError, match="regime_transition"):
        assert_recipe_null_regime_pressure(
            policy,
            ledger=ledger,
            data_eval_identity=identity,
            claim_class="fixture",
            matrix_has_regime_transition=False,
            proposed_knobs=[{"steps": 99, "lr": 1e-4}],
        )
    # With regime transition present, pure recipe still rejected after cap
    with pytest.raises(HillClimbError, match="recipe-tweak"):
        assert_recipe_null_regime_pressure(
            policy,
            ledger=ledger,
            data_eval_identity=identity,
            claim_class="fixture",
            matrix_has_regime_transition=True,
            proposed_knobs=[{"steps": 99, "lr": 1e-4}],
        )
    # Architecture knob + regime transition allowed
    assert_recipe_null_regime_pressure(
        policy,
        ledger=ledger,
        data_eval_identity=identity,
        claim_class="fixture",
        matrix_has_regime_transition=True,
        proposed_knobs=[{"grammar_completion_bounds": True, "steps": 20}],
    )


def test_is_recipe_tweak_knobs() -> None:
    policy = load_climb_policy()
    assert is_recipe_tweak_knobs(policy, {"steps": 20, "lr": 1e-3})
    assert not is_recipe_tweak_knobs(policy, {"grammar_completion_bounds": True})


def test_rung_gate_optional() -> None:
    policy = load_climb_policy()
    # Default policy carries durable evidence for the prior rung.
    assert_rung_allows_promotion(policy, claimed_role="promotion")


def test_continuous_classify_positive_entry(tmp_path: Path) -> None:
    """Drive real continuous _classify_positive with constructed run dirs."""
    from scripts.run_autotrain_continuous import _classify_positive

    camp = tmp_path / "camp"
    control = camp / "runs" / "exp-control"
    cand = camp / "runs" / "exp-candidate"
    control.mkdir(parents=True)
    cand.mkdir(parents=True)
    # Latency win requires held quality and non-empty meaning (PR #1234).
    (control / "eval_smoke.json").write_text(
        json.dumps(
            {
                "n": 3,
                "document_n": 3,
                "completed_document_n": 3,
                "incomplete_document_n": 0,
                "decode_timeout_count": 0,
                "latency_ms_p50": 11.0,
                "parse_rate": 1.0,
                "meaningful_program_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (cand / "eval_smoke.json").write_text(
        json.dumps(
            {
                "n": 3,
                "document_n": 3,
                "completed_document_n": 3,
                "incomplete_document_n": 0,
                "decode_timeout_count": 0,
                "latency_ms_p50": 10.0,
                "parse_rate": 1.0,
                "meaningful_program_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    complete_scoreboard = {
        "suites": {
            "smoke": {
                "n": 3,
                "document_n": 3,
                "completed_document_n": 3,
                "incomplete_document_n": 0,
                "decode_timeout_count": 0,
            }
        }
    }
    for run_dir in (control, cand):
        (run_dir / "scoreboard.json").write_text(
            json.dumps(complete_scoreboard), encoding="utf-8"
        )
    win = _classify_positive(
        camp_dir=camp,
        primary_metric="smoke.latency_ms_p50",
        control_id="exp-control",
        candidate_id="exp-candidate",
        role="screening",
    )
    assert win["positive"] is True

    (cand / "eval_smoke.json").write_text(
        json.dumps(
            {
                "n": 3,
                "document_n": 3,
                "completed_document_n": 3,
                "incomplete_document_n": 0,
                "decode_timeout_count": 0,
                "latency_ms_p50": 12.5,
                "parse_rate": 1.0,
                "meaningful_program_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    bad = _classify_positive(
        camp_dir=camp,
        primary_metric="smoke.latency_ms_p50",
        control_id="exp-control",
        candidate_id="exp-candidate",
        role="screening",
    )
    assert bad["positive"] is False

    (cand / "eval_smoke.json").write_text(
        json.dumps(
            {
                "n": 3,
                "document_n": 3,
                "completed_document_n": 3,
                "incomplete_document_n": 0,
                "decode_timeout_count": 0,
                "latency_ms_p50": 10.0,
                "parse_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    # Win on metric but uncharged capacity growth
    outcomes = camp / "artifacts" / "outcomes"
    outcomes.mkdir(parents=True)
    (outcomes / "control.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-control",
                "status": "completed",
                "metrics": {"trainable_params": 100},
            }
        ),
        encoding="utf-8",
    )
    (outcomes / "cand.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-candidate",
                "status": "completed",
                "metrics": {"trainable_params": 200},
            }
        ),
        encoding="utf-8",
    )
    grow = _classify_positive(
        camp_dir=camp,
        primary_metric="smoke.latency_ms_p50",
        control_id="exp-control",
        candidate_id="exp-candidate",
        role="screening",
        baseline_trainable_params=100,
        candidate_trainable_params=200,
        eg_params_by_seed=None,
    )
    assert grow["positive"] is False
    assert any("eg_params" in r for r in grow["reasons"])


def test_synthesis_still_blocks_without_action(tmp_path: Path) -> None:
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        synthesis_policy_allows_sft,
    )
    from slm_training.autoresearch.hillclimb import HillClimbError

    policy = load_climb_policy()
    train = tmp_path / "train"
    train.mkdir()
    (train / "synthesis_feedback.json").write_text(
        json.dumps({"recommendations": [{"code": "low_yield", "target": "foo"}]}),
        encoding="utf-8",
    )
    with pytest.raises(HillClimbError, match="open synthesis"):
        synthesis_policy_allows_sft(policy, train_dir=train)
    (train / "synthesis_feedback_actions.json").write_text(
        json.dumps({"actions": [{"code": "low_yield", "note": "fixed"}]}),
        encoding="utf-8",
    )
    assert synthesis_policy_allows_sft(policy, train_dir=train) is True


def test_primary_for_role_differs() -> None:
    policy = load_climb_policy()
    s = primary_for_role(policy, "screening")
    p = primary_for_role(policy, "promotion")
    assert s["metric"] != p["metric"] or s["direction"] != p["direction"]
    assert p.get("require_locked_eval") is True


def test_policy_v14_screening_primary_is_smoke_eval_nll() -> None:
    policy = load_climb_policy(str(CLIMB_RESOURCE_DIR / "policy.v2.json"))
    assert policy.version == "v14"
    screening = primary_for_role(policy, "screening")
    assert screening["metric"] == "smoke.eval_nll"
    assert screening["direction"] == "decrease"
    assert policy.defaults.get("claim_class_screening") == "diagnostic"
    assert policy.promotion_primary["metric"] == "held_out.structural_similarity"
    secondary = policy.screening_quality_secondary
    assert secondary is not None
    assert secondary["metric"] == "smoke.structural_similarity"
    meas = policy.measurement
    assert meas["thrash_timing"]["screening_thrash_steps"] == "fit_to_train_floor"
    assert meas["thrash_timing"]["screening_thrash_steps_max"] == 400
    assert meas["warm_start"]["enabled"] is True
    assert meas["paired_test"]["kind"] == "wilcoxon_signed_rank"
    assert meas["multi_arm"]["max_arms_per_cycle"] == 6


@pytest.mark.parametrize(
    "control_nll,candidate_nll,expect_positive",
    [
        (2.0, 1.9, True),
        (2.0, 2.1, False),
        (2.0, 1.96, False),
    ],
)
def test_classify_positive_screening_nll_direction_decrease(
    control_nll: float, candidate_nll: float, expect_positive: bool
) -> None:
    policy = load_climb_policy(str(CLIMB_RESOURCE_DIR / "policy.v2.json"))
    result = classify_positive_metrics(
        policy,
        role="screening",
        control_metrics={
            "smoke.eval_nll": control_nll,
            "smoke.structural_similarity": 0.1,
            "parse_rate": 1.0,
        },
        candidate_metrics={
            "smoke.eval_nll": candidate_nll,
            "smoke.structural_similarity": 0.1,
            "parse_rate": 1.0,
        },
    )
    assert result["positive"] is expect_positive
    if expect_positive:
        assert any(r.startswith("primary_metric_win:smoke.eval_nll") for r in result["reasons"])
    else:
        assert any("primary_metric_null_or_worse:smoke.eval_nll" in r for r in result["reasons"])
    assert any("screening_quality_secondary:smoke.structural_similarity" in r for r in result["reasons"])
    assert any("recorded_not_verdict" in r for r in result["reasons"])


def test_screening_nll_definition_hash_ignores_arm_loss_weights() -> None:
    control = screening_nll_definition_hash()
    candidate = screening_nll_definition_hash(
        arm_loss_weights={"binder_arity_loss_weight": 1.0}
    )
    assert control == candidate
    assert len(control) == 64


def test_train_eval_identity_differs_when_versions_change() -> None:
    from slm_training.autoresearch.climb_policy import train_eval_from_knobs

    policy = load_climb_policy()
    a = loop_data_eval_identity(
        policy,
        claim_class="fixture",
        train_version="wf_smoke_v2",
        eval_version="e938",
        primary_metric="smoke.latency_ms_p50",
        direction="decrease",
    )
    b = loop_data_eval_identity(
        policy,
        claim_class="fixture",
        train_version="other_train",
        eval_version="e938",
        primary_metric="smoke.latency_ms_p50",
        direction="decrease",
    )
    assert a != b
    tv, ev = train_eval_from_knobs(
        [{"train_version": "wf_smoke_v2", "eval_version": "e938", "steps": 20}]
    )
    assert tv == "wf_smoke_v2"
    assert ev == "e938"


def test_synthesis_policy_uses_action_filenames(tmp_path: Path) -> None:
    """Policy action_filenames drive the SFT gate lookup, not only hard-coded names."""
    from slm_training.autoresearch.hillclimb import (
        assert_synthesis_feedback_cleared_for_sft,
    )

    policy = load_climb_policy()
    names = tuple(policy.synthesis_loop["action_filenames"])
    assert names  # externalized
    train = tmp_path / "train"
    train.mkdir()
    (train / "synthesis_feedback.json").write_text(
        json.dumps({"recommendations": [{"code": "low_yield"}]}),
        encoding="utf-8",
    )
    # Policy-named action file (first name) clears the gate
    (train / names[0]).write_text(
        json.dumps({"actions": [{"code": "low_yield", "note": "fixed"}]}),
        encoding="utf-8",
    )
    assert_synthesis_feedback_cleared_for_sft(
        train,
        action_filenames=names,
        allow_global_waiver_code=str(
            policy.synthesis_loop.get("allow_global_waiver_code") or "*"
        ),
    )


def test_rung_gate_wired_via_cadence_when_enabled() -> None:
    """assert_cycle_cadence invokes rung gate (enabled path fails without prior rung)."""
    policy = load_climb_policy()
    # Mutate a copy of payload into a temporary policy object by reconstructing
    raw = dict(policy.payload)
    gates = dict(raw["rung_gates"])
    gates["enabled"] = True
    gates["current_rung"] = "grammar_2_ast"
    gates["order"] = ["ast_2_ast", "grammar_2_ast"]
    gates["block_promotion_if_prior_rung_unmet"] = True
    raw["rung_gates"] = gates
    from slm_training.autoresearch.climb_policy import ClimbPolicy

    enabled = ClimbPolicy(
        path=policy.path,
        schema=policy.schema,
        version=policy.version,
        sha256=policy.sha256,
        payload=raw,
    )
    n = int(enabled.cadence["screening_cycles_per_promotion"])
    promo_idx = n + 1
    with pytest.raises(HillClimbError, match="rung gate|prior rung"):
        assert_cycle_cadence(
            enabled,
            cycle_index=promo_idx,
            claimed_role="promotion",
            claim_class="promotion_candidate",
            certified_rungs=(),
        )
    # Prior rung certified → promotion allowed
    assert_cycle_cadence(
        enabled,
        cycle_index=promo_idx,
        claimed_role="promotion",
        claim_class="promotion_candidate",
        certified_rungs=("ast_2_ast",),
    )


def test_policy_require_multi_seed_false_allows_single_seed_promotion() -> None:
    """validate_causal_campaign_shape honors policy min_seeds / require_multi_seed."""
    from slm_training.autoresearch.climb_policy import ClimbPolicy, promotion_seed_floor
    from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
    from slm_training.autoresearch.hillclimb import validate_causal_campaign_shape
    from tests.test_autoresearch.test_experiment_campaign import _manifest_payload

    # Build a valid promotion campaign (default policy needs 2 seeds), then
    # model_copy to one seed and prove causal shape accepts policy override.
    base = ExperimentCampaignV1.model_validate(_manifest_payload())
    one_seed = base.model_copy(update={"seeds": (7,)})
    assert (
        validate_causal_campaign_shape(one_seed, min_seeds=1, require_multi_seed=False)
        == ()
    )
    # No kwargs → loads real committed policy (min_seeds=2) → fails single seed
    default_failures = validate_causal_campaign_shape(one_seed)
    assert any(f.startswith("insufficient_seeds:1<2") for f in default_failures)

    policy = load_climb_policy()
    raw = dict(policy.payload)
    promo = dict(raw["promotion_primary"])
    promo["require_multi_seed"] = False
    promo["min_seeds"] = 1
    raw["promotion_primary"] = promo
    relaxed = ClimbPolicy(
        path=policy.path,
        schema=policy.schema,
        version=policy.version,
        sha256=policy.sha256,
        payload=raw,
    )
    floor, require = promotion_seed_floor(relaxed)
    assert floor == 1 and require is False
    assert (
        validate_causal_campaign_shape(
            one_seed, min_seeds=floor, require_multi_seed=require
        )
        == ()
    )
    # Committed policy still defaults to multi-seed >= 2 for production.
    assert promotion_seed_floor(policy)[0] >= 2


def test_continuous_manifest_promotion_uses_held_out_primary() -> None:
    """Each bounded promotion manifest honestly declares its one executed seed."""
    from scripts.run_autotrain_continuous import _manifest
    from slm_training.autoresearch.climb_policy import promotion_seed_floor

    policy = load_climb_policy()
    promo = primary_for_role(policy, "promotion")
    exp = {
        "experiment_id": "hyp-1",
        "hypothesis": "Held-out quality improves under the candidate lever.",
        "knobs": {
            "seed": 7,
            "steps": 20,
            "train_version": "wf_smoke_v2",
            "eval_version": "e938_role_safe_all_targets_v2",
        },
    }
    man = _manifest(
        "camp-promo",
        exp,
        "a" * 40,
        role="promotion",
        policy=policy,
    )
    assert man.claim_class == "promotion_candidate"
    assert man.endpoints[0].metric == promo["metric"]
    assert man.endpoints[0].direction == promo["direction"]
    assert man.seeds == (7,)
    assert promotion_seed_floor(policy)[0] >= 2
    assert man.replicate_ledger_schema == "autotrain_promotion_replicate/v1"
    assert man.replicate_seed_floor == promotion_seed_floor(policy)[0]
    assert man.locked_eval_manifest_sha256
    assert man.mechanism_off_arm_ids
    assert man.executable_kill_criteria

    screen = _manifest(
        "camp-screen",
        exp,
        "a" * 40,
        role="screening",
        policy=policy,
    )
    assert screen.claim_class == "diagnostic"
    assert screen.endpoints[0].metric == primary_for_role(policy, "screening")["metric"]
    # Role policy owns the stage wall within the canonical repository cap.
    from slm_training.autoresearch.climb_policy import stage_wall_minutes_for_role

    assert screen.budget.max_wall_minutes == float(
        stage_wall_minutes_for_role(policy, "screening")
    )
    assert man.budget.max_wall_minutes == float(
        stage_wall_minutes_for_role(policy, "promotion")
    )
    assert len(screen.seeds) == 1


def test_classify_positive_rejects_partial_suite_completion() -> None:
    """A control arm that only finished part of the suite must not win or lose
    against a candidate that finished the full suite -- the averaged metric
    reflects which documents happened to complete under the wall cap, not a
    real effect of the candidate's levers."""
    policy = load_climb_policy()
    result = classify_positive_metrics(
        policy,
        role="promotion",
        control_metrics={
            "held_out.structural_similarity": 0.3417,
            "held_out.completed_document_n": 1,
            "held_out.n": 5,
        },
        candidate_metrics={
            "held_out.structural_similarity": 0.37006,
            "held_out.completed_document_n": 5,
            "held_out.n": 5,
        },
    )
    assert result["positive"] is False
    assert any(
        r.startswith("primary_metric_incomparable_partial_suite")
        for r in result["reasons"]
    )
    assert not any(r.startswith("primary_metric_win") for r in result["reasons"])
    assert not any(
        r.startswith("primary_metric_null_or_worse") for r in result["reasons"]
    )

    # Both arms completing the full suite is unaffected by the gate.
    full = classify_positive_metrics(
        policy,
        role="promotion",
        control_metrics={
            "held_out.structural_similarity": 0.30,
            "held_out.completed_document_n": 5,
            "held_out.n": 5,
        },
        candidate_metrics={
            "held_out.structural_similarity": 0.40,
            "held_out.completed_document_n": 5,
            "held_out.n": 5,
        },
    )
    assert full["positive"] is True
    assert any(r.startswith("primary_metric_win") for r in full["reasons"])


class _StubPolicy:
    def __init__(self, measurement: dict, payload: dict | None = None) -> None:
        self._measurement = measurement
        self.payload = payload or {}

    @property
    def measurement(self) -> dict:
        return dict(self._measurement)


def test_screening_smoke_n_fixed_mode_returns_configured_without_report() -> None:
    policy = _StubPolicy({"screening_smoke_n": 5})
    n, report = screening_smoke_n_for_policy(policy)
    assert n == 5
    assert report is None


def test_screening_smoke_n_auto_mode_uses_max_of_floors() -> None:
    policy = _StubPolicy(
        {
            "screening_smoke_n": 3,
            "screening_smoke_n_mode": "auto",
            "screening_sample_size": {
                "max_candidate_n": 64,
                "default_decode_floor_seconds": 2,
                "minimum_effect": "1/100",
                "observed_sd": "1/10",
            },
        },
        payload={"power_gate": {"enabled": True, "alpha": "1/20"}},
    )
    n, report = screening_smoke_n_for_policy(
        policy, arm_wall_seconds=70.0, suite_records=64
    )
    assert report is not None
    assert report["power_floor_n"] is not None
    assert report["decidability_floor_n"] == 6
    if report["verdict"] == "feasible":
        assert n == max(report["decidability_floor_n"], report["power_floor_n"])
        assert n == report["chosen_n"]


def test_screening_smoke_n_auto_mode_feasible_climbs_at_floor() -> None:
    policy = _StubPolicy(
        {
            "screening_smoke_n": 3,
            "screening_smoke_n_mode": "auto",
            "screening_sample_size": {
                "max_candidate_n": 64,
                "default_decode_floor_seconds": 2,
            },
        },
        payload={"power_gate": {"enabled": True, "alpha": "1/20"}},
    )
    n, report = screening_smoke_n_for_policy(
        policy, arm_wall_seconds=70.0, suite_records=24
    )
    assert n == 6  # exact sign-test floor at alpha=1/20, smallest sufficient
    assert report is not None
    assert report["verdict"] == "feasible"
    assert report["promotion_authority"] is False


def test_screening_smoke_n_auto_mode_infeasible_does_not_return_fallback_n() -> None:
    policy = _StubPolicy(
        {
            "screening_smoke_n": 3,
            "screening_smoke_n_mode": "auto",
            "screening_sample_size": {"default_decode_floor_seconds": 2},
        }
    )
    n, report = screening_smoke_n_for_policy(
        policy, arm_wall_seconds=70.0, suite_records=3
    )
    assert n == 0
    assert report is not None
    assert report["verdict"] == "infeasible_range_empty"
    assert "suite_volume" in report["binding_constraints"]
    assert report["must_generate"] is True


def test_screening_smoke_n_live_policy_is_auto_with_fallback() -> None:
    policy = load_climb_policy()
    assert policy.measurement.get("screening_smoke_n_mode") == "auto"
    n, report = screening_smoke_n_for_policy(policy)
    assert n == int(policy.measurement["screening_smoke_n"])
    assert report is not None
    assert report["verdict"] == "insufficient_evidence"
    assert report["chosen_n"] is None  # no arm wall: budget undecidable


# ---------------------------------------------------------------------------
# RC1 replay: smoke.eval_nll primary must never park on a borrowed SD
# ---------------------------------------------------------------------------


def test_default_policy_is_v3_with_nll_unit_minimum_effect() -> None:
    policy = load_climb_policy()
    assert policy.path.name == "policy.v3.json"
    assert policy.version == "v15"
    screening = policy.screening_primary
    assert screening["metric"] == "smoke.eval_nll"
    assert screening["minimum_effect"] == pytest.approx(0.02)
    assert screening["minimum_effect_units"] == "nats/token"
    assert screening["minimum_effect_rationale"]
    block = policy.measurement["screening_sample_size"]
    assert block["observed_sd_by_metric"] == {}
    assert "observed_sd" not in block  # no untagged scalar to borrow
    # v3 is v2 plus the RC1 fields only.
    v2 = json.loads((CLIMB_RESOURCE_DIR / "policy.v2.json").read_text(encoding="utf-8"))
    v3 = dict(policy.payload)
    for key, value in v2.items():
        if key in {"version", "description", "screening_primary", "measurement"}:
            continue
        assert v3[key] == value, f"policy.v3.json changed v2 field {key!r}"
    for key, value in v2["measurement"].items():
        if key == "screening_sample_size":
            continue
        assert v3["measurement"][key] == value


@pytest.mark.parametrize("arm_wall_seconds,expected_n", [(180.0, 24), (70.0, 21)])
def test_screening_smoke_n_rc1_replay_eval_nll_primary_is_runnable(
    arm_wall_seconds: float, expected_n: int
) -> None:
    # RC1 inputs verbatim: policy primary smoke.eval_nll, suite 24 records.
    # Before: MEASURED_PAIRED_SD (smoke.structural_similarity) was borrowed,
    # power floor 96 > 24 -> infeasible_range_empty -> n=0 -> park every cycle.
    policy = load_climb_policy()
    n, report = screening_smoke_n_for_policy(
        policy, arm_wall_seconds=arm_wall_seconds, suite_records=24
    )
    assert report is not None
    assert n >= 6
    assert n == expected_n  # min(suite 24, budget ceiling, max_candidate_n 64)
    assert report["verdict"] in {"insufficient_evidence", "feasible"}
    assert report["must_generate"] is False
    assert report["chosen_n"] == n
    assert report["decidability_floor_n"] == 6
    assert report["binding_constraints"] == []
    if report["verdict"] == "insufficient_evidence":
        assert report["power_floor_status"] == "unmeasured"
        assert report["power_floor_n"] is None
        assert report["observed_sd_source"] == "unmeasured"
        assert report["observed_sd_metric"] == "smoke.eval_nll"
    else:  # a same-metric SD was recorded (P3): power floor is measured
        assert report["power_floor_status"] == "measured"
        assert report["observed_sd_source"] != "unmeasured"


def test_screening_smoke_n_unmeasured_sd_generates_only_below_exact_floor() -> None:
    policy = load_climb_policy()
    n, report = screening_smoke_n_for_policy(
        policy, arm_wall_seconds=180.0, suite_records=5
    )
    assert n == 0
    assert report is not None
    assert report["verdict"] == "infeasible_range_empty"
    assert report["must_generate"] is True
    assert report["n_min"] == 6
    n, report = screening_smoke_n_for_policy(
        policy, arm_wall_seconds=180.0, suite_records=6
    )
    assert n == 6
    assert report is not None
    assert report["must_generate"] is False


def test_screening_smoke_n_measured_same_metric_sd_yields_power_floor() -> None:
    policy = _StubPolicy(
        {
            "screening_smoke_n": 3,
            "screening_smoke_n_mode": "auto",
            "screening_sample_size": {
                "max_candidate_n": 64,
                "default_decode_floor_seconds": 2,
                "observed_sd_by_metric": {"eval_nll": "1/50"},
            },
        },
        payload={
            "power_gate": {"enabled": True, "alpha": "1/20"},
            "screening_primary": {
                "metric": "smoke.eval_nll",
                "direction": "decrease",
                "minimum_effect": 0.02,
            },
        },
    )
    n, report = screening_smoke_n_for_policy(
        policy, arm_wall_seconds=180.0, suite_records=24
    )
    assert report is not None
    assert report["power_floor_status"] == "measured"
    assert report["observed_sd_source"] == "policy"
    assert report["power_floor_n"] == 8  # ((1.96+0.84)*0.02/0.02)^2 -> 8
    assert report["verdict"] == "feasible"
    assert n == 8 == report["chosen_n"]


def test_screening_smoke_n_never_borrows_another_metrics_sd() -> None:
    # A scalar observed_sd tagged for structural_similarity must not power a
    # smoke.eval_nll primary; the floor stays unmeasured.
    policy = _StubPolicy(
        {
            "screening_smoke_n": 3,
            "screening_smoke_n_mode": "auto",
            "screening_sample_size": {
                "max_candidate_n": 64,
                "default_decode_floor_seconds": 2,
                "observed_sd": "1741/10000",
                "observed_sd_metric": "smoke.structural_similarity",
                "observed_sd_by_metric": {"structural_similarity": "1741/10000"},
            },
        },
        payload={
            "power_gate": {"enabled": True, "alpha": "1/20"},
            "screening_primary": {
                "metric": "smoke.eval_nll",
                "direction": "decrease",
                "minimum_effect": 0.05,
            },
        },
    )
    n, report = screening_smoke_n_for_policy(
        policy, arm_wall_seconds=180.0, suite_records=24
    )
    assert report is not None
    assert report["power_floor_status"] == "unmeasured"
    assert report["power_floor_n"] is None
    assert n == 24
    # The same declaration powers its own metric.
    policy.payload["screening_primary"]["metric"] = "smoke.structural_similarity"
    n, report = screening_smoke_n_for_policy(
        policy, arm_wall_seconds=180.0, suite_records=24
    )
    assert report is not None
    assert report["power_floor_status"] == "measured"
    assert report["power_floor_n"] == 96
    assert n == 0 and report["verdict"] == "infeasible_range_empty"
