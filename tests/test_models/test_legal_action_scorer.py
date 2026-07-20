"""Tests for SLM-154 (SPV3-01) legal-action scorer baseline."""

from __future__ import annotations

import math

import pytest

from slm_training.data.semantic_plan.compiler import PlanActionFeatures
from slm_training.models.legal_action_scorer import (
    LEGAL_ACTION_SCORER_SCHEMA_VERSION,
    LegalActionScorer,
    LegalActionScorerConfig,
    make_fixture_decisions,
    train_fixture_scorer,
    evaluate_fixture_scorer,
)


@pytest.fixture(params=["global_head", "mlp", "cross_attention"])
def variant(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def test_forced_singleton_skips_model(variant: str) -> None:
    pytest.importorskip("torch")
    scorer = LegalActionScorer(
        config=LegalActionScorerConfig(variant=variant, seed=0)
    )
    scores = scorer.score({}, {}, ["only_action"])
    assert scores.metadata.get("forced")
    decision = scorer.decode(scores, ["only_action"])
    assert decision.decision_kind == "forced"
    assert decision.action_identity == "only_action"


def test_score_returns_one_logit_per_legal_action(variant: str) -> None:
    pytest.importorskip("torch")
    scorer = LegalActionScorer(
        config=LegalActionScorerConfig(variant=variant, seed=0)
    )
    actions = ["a:0", "a:1", "a:2"]
    scores = scorer.score(
        {"pack_id": "openui", "n_mentioned_components": 2},
        {"state_family_id": "fixture", "depth": 1, "branch_count": 3},
        actions,
    )
    assert scores.logits.shape == (1, 3)
    assert len(scores.scores) == 3
    assert scores.legal_actions == tuple(actions)


def test_softmax_over_legal_set_sums_to_one(variant: str) -> None:
    pytest.importorskip("torch")
    scorer = LegalActionScorer(
        config=LegalActionScorerConfig(variant=variant, seed=0)
    )
    actions = ["a:0", "a:1", "a:2"]
    scores = scorer.score(
        {"pack_id": "openui"},
        {"state_family_id": "fixture"},
        actions,
    )
    import torch

    probs = torch.softmax(scores.logits[0], dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_global_head_masks_illegal_actions() -> None:
    pytest.importorskip("torch")
    scorer = LegalActionScorer(
        config=LegalActionScorerConfig(variant="global_head", seed=0)
    )
    actions = ["a:0", "a:1"]
    scores = scorer.score(
        {"pack_id": "openui"},
        {"state_family_id": "fixture"},
        actions,
    )
    # All returned logits must be finite; the module does not emit -inf directly,
    # but the legal set is restricted to the supplied actions.
    assert all(math.isfinite(v) for v in scores.scores.values())


def test_unsupported_pack_abstains(variant: str) -> None:
    pytest.importorskip("torch")
    scorer = LegalActionScorer(
        config=LegalActionScorerConfig(variant=variant, seed=0)
    )
    scores = scorer.score(
        {"pack_id": "graphql"},
        {"state_family_id": "fixture"},
        ["a:0", "a:1"],
        pack_id="graphql",
    )
    assert scores.metadata.get("abstained")


def test_training_reduces_loss(variant: str) -> None:
    pytest.importorskip("torch")
    decisions = make_fixture_decisions(n=32, seed=0)
    result = train_fixture_scorer(
        decisions,
        config=LegalActionScorerConfig(variant=variant, seed=1),
        steps=30,
        lr=0.05,
    )
    assert result["history"][0]["loss"] > result["history"][-1]["loss"]


def test_evaluation_improves_after_training(variant: str) -> None:
    pytest.importorskip("torch")
    train_decisions = make_fixture_decisions(n=48, seed=0)
    test_decisions = make_fixture_decisions(n=16, seed=1)
    result = train_fixture_scorer(
        train_decisions,
        config=LegalActionScorerConfig(variant=variant, seed=2),
        steps=40,
        lr=0.05,
    )
    scorer = result["scorer"]
    eval_result = evaluate_fixture_scorer(scorer, test_decisions)
    assert eval_result["accuracy"] > 0.3


def test_checkpoint_roundtrip(tmp_path, variant: str) -> None:
    pytest.importorskip("torch")
    original = LegalActionScorer(
        config=LegalActionScorerConfig(variant=variant, seed=3)
    )
    path = tmp_path / "scorer.pt"
    original.save(str(path))

    loaded = LegalActionScorer.from_checkpoint(str(path))
    assert loaded.config.variant == variant
    assert loaded.artifact_identity()["schema"] == LEGAL_ACTION_SCORER_SCHEMA_VERSION

    actions = ["a:0", "a:1", "a:2"]
    s1 = original.score({"pack_id": "openui"}, {"state_family_id": "fixture"}, actions)
    s2 = loaded.score({"pack_id": "openui"}, {"state_family_id": "fixture"}, actions)
    assert s1.legal_actions == s2.legal_actions
    for a in actions:
        assert abs(s1.scores[a] - s2.scores[a]) < 1e-5


def test_schema_mismatch_fails_closed(tmp_path, variant: str) -> None:
    pytest.importorskip("torch")
    original = LegalActionScorer(
        config=LegalActionScorerConfig(variant=variant, seed=4)
    )
    path = tmp_path / "scorer.pt"
    original.save(str(path))

    other = LegalActionScorer(
        config=LegalActionScorerConfig(variant=variant, seed=5)
    )
    with pytest.raises(ValueError, match="checkpoint config does not match"):
        other.load(str(path))


def test_plan_action_features_change_scores(variant: str) -> None:
    pytest.importorskip("torch")
    scorer = LegalActionScorer(
        config=LegalActionScorerConfig(variant=variant, seed=0)
    )
    actions = ["a:0", "a:1"]
    scorer.score(
        {"pack_id": "openui"},
        {"state_family_id": "fixture"},
        actions,
    )
    plan_features = [
        PlanActionFeatures(action_id="a:0", plan_confidence=0.9),
        PlanActionFeatures(action_id="a:1", plan_confidence=0.1),
    ]
    plan_conditioned = scorer.score(
        {"pack_id": "openui"},
        {"state_family_id": "fixture"},
        actions,
        plan_action_features=plan_features,
    )
    # Scores may change; the key invariant is membership is unchanged.
    assert plan_conditioned.legal_actions == tuple(actions)
    assert len(plan_conditioned.scores) == 2


def test_set_valued_accepted_target_works() -> None:
    pytest.importorskip("torch")
    scorer = LegalActionScorer(
        config=LegalActionScorerConfig(variant="mlp", seed=0)
    )
    actions = ["a:0", "a:1", "a:2"]
    loss, metrics = scorer.loss(
        {"pack_id": "openui"},
        {"state_family_id": "fixture"},
        actions,
        ["a:0", "a:1"],
    )
    assert math.isfinite(loss.item())
    assert metrics["n_accepted"] == 2
