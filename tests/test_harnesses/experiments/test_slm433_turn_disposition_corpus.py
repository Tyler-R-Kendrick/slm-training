"""Tests for SLM-433 (VAR3-03): TurnDispositionHead trained on a real corpus."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm433_turn_disposition_corpus import (
    ARMS,
    run_experiment,
    train_disposition_head,
)
from slm_training.data.flow.turn_disposition_corpus import build_turn_disposition_corpus


def _small_payload(**overrides):
    kwargs = dict(n_per_tier=6, train_steps=10, corpus_seed=101, train_seed=101)
    kwargs.update(overrides)
    return run_experiment(**kwargs)


def test_run_experiment_reports_all_three_arms_on_identical_held_out_split() -> None:
    payload = _small_payload()
    assert set(payload["arm_scores"]) == set(ARMS)
    counts = {arm: payload["arm_scores"][arm]["case_count"] for arm in ARMS}
    assert len(set(counts.values())) == 1  # matched budget: identical case_count


def test_wrong_op_and_abstention_are_never_blended_away() -> None:
    payload = _small_payload()
    for arm in ARMS:
        score = payload["arm_scores"][arm]
        assert "wrong_op_rate" in score
        assert "abstention_rate" in score
        assert "composite_penalized_error_rate" in score


def test_disposition_is_one_of_the_two_named_outcomes() -> None:
    payload = _small_payload()
    assert payload["disposition"] in {
        "trained_head_improves_on_derived_only",
        "no_held_out_improvement",
    }


def test_no_promotion_or_ship_claim_anywhere() -> None:
    payload = _small_payload()
    honesty = payload["honesty"].lower()
    assert "not a ship claim" in honesty
    assert "no promotion" in honesty


def test_leakage_safe_split_is_reported_true() -> None:
    payload = _small_payload()
    assert payload["leakage_safe_split"] is True


def test_derived_only_never_predicts_clarify() -> None:
    """derived_only's own contract: clarify never fires."""
    payload = _small_payload()
    assert payload["arm_scores"]["derived_only"]["abstention_rate"] == 0.0
    assert payload["arm_scores"]["disposition_off"]["abstention_rate"] == 0.0


def test_derived_only_strictly_beats_disposition_off_on_out_of_scope_and_answer() -> None:
    """derived_only correctly resolves out_of_scope/answer/forced-singleton
    for free; disposition_off forces a (definitionally wrong) emit there."""
    payload = _small_payload()
    off = payload["arm_scores"]["disposition_off"]
    derived = payload["arm_scores"]["derived_only"]
    assert derived["wrong_op_count"] <= off["wrong_op_count"]
    assert derived["composite_penalized_error_rate"] < off["composite_penalized_error_rate"]


def test_run_experiment_is_deterministic() -> None:
    payload_a = _small_payload()
    payload_b = _small_payload()
    assert payload_a["arm_scores"] == payload_b["arm_scores"]
    assert payload_a["disposition"] == payload_b["disposition"]


def test_train_disposition_head_masks_forced_examples_out_of_scored_training() -> None:
    """Only scored rows are ever passed to training -- forced/answer/oos
    rows never reach the classifier (I1 precedence unchanged)."""
    corpus = build_turn_disposition_corpus(n_per_tier=6, seed=202)
    scored_train = corpus.scored("train")
    assert scored_train
    assert all(row.turn_kind == "scored" for row in scored_train)
    model, final_loss = train_disposition_head(scored_train, steps=5, seed=202)
    assert model is not None
    assert final_loss >= 0.0
