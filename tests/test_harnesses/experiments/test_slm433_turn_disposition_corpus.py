"""Tests for SLM-433 (VAR3-03): TurnDispositionHead trained on a real corpus."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm433_turn_disposition_corpus import (
    ARMS,
    run_experiment,
    run_seed_sweep_experiment,
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


def _small_sweep(**overrides):
    # steps/seed count are deliberately tiny: these tests validate the
    # sign-test bookkeeping, not statistical power -- see the NOTE below on
    # where the real (slow, 10-seed, 200-step) evidence surface is generated.
    kwargs = dict(n_per_tier=6, train_steps=3, seeds=(101, 102, 103))
    kwargs.update(overrides)
    return run_seed_sweep_experiment(**kwargs)


def test_seed_sweep_reports_every_seed_with_no_exclusion() -> None:
    seeds = (201, 202, 203)
    payload = _small_sweep(seeds=seeds)
    assert payload["seeds"] == list(seeds)
    assert [row["seed"] for row in payload["per_seed"]] == list(seeds)
    assert payload["wins"] + payload["ties"] + payload["losses"] == len(seeds)


def test_seed_sweep_disposition_follows_sign_test_not_a_single_seed() -> None:
    payload = _small_sweep()
    if payload["wins"] > payload["losses"]:
        assert payload["disposition"] == "trained_head_improves_on_derived_only"
    else:
        assert payload["disposition"] == "no_held_out_improvement"


def test_seed_sweep_ties_do_not_count_as_wins() -> None:
    """A wins == losses sweep (including 0-0) must be the negative
    disposition, not the positive one -- ties never break in favor of the
    hypothesis."""
    payload = _small_sweep(seeds=(301,))
    trained = payload["per_seed"][0]["trained_composite"]
    derived = payload["per_seed"][0]["derived_composite"]
    if trained == derived:
        assert payload["wins"] == 0 and payload["losses"] == 0
        assert payload["disposition"] == "no_held_out_improvement"


def test_seed_sweep_is_deterministic() -> None:
    a = _small_sweep()
    b = _small_sweep()
    assert a["per_seed"] == b["per_seed"]
    assert a["disposition"] == b["disposition"]


# NOTE: the full preregistered 10-seed x n_per_tier=40 x train_steps=200
# sweep (the actual evidence surface behind the committed docs/design
# artifact) is deliberately NOT run at that scale in this fast unit-test
# suite -- it takes minutes, well past this repo's per-job CI time budget.
# It is generated by `python -m scripts.run_slm433_turn_disposition_corpus`
# and committed as docs/design/var3-03-turn-disposition-corpus-training-*;
# regenerate and re-commit that artifact directly if the corpus/training
# code changes, rather than gating it behind a slow pytest case.
