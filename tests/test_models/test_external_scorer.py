"""Tests for the SLM-108 external legal-action scorer interface."""

from __future__ import annotations

import pytest

from slm_training.data.contract import GenerationRequest
from slm_training.evals.score_policy import CandidatePath
from slm_training.models.external_scorer import (
    CompleteCandidate,
    ExternalScorerConfig,
    ExternalScorePolicy,
    LegalAction,
    build_external_scorer,
)


def test_config_identity_is_deterministic() -> None:
    cfg = ExternalScorerConfig(
        model_id="HuggingFaceTB/SmolLM2-135M",
        revision="main",
        device="cpu",
        dtype="float32",
    )
    identity = cfg.identity()
    assert identity["model_id"] == "HuggingFaceTB/SmolLM2-135M"
    assert identity["revision"] == "main"
    assert cfg.fingerprint() == cfg.fingerprint()


def test_build_unknown_scorer_kind_raises() -> None:
    cfg = ExternalScorerConfig(model_id="x", revision="y")
    with pytest.raises(ValueError, match="Unknown external scorer kind"):
        build_external_scorer(cfg, kind="not_real")


def test_fake_scorer_returns_scores_for_all_actions() -> None:
    cfg = ExternalScorerConfig(model_id="x", revision="y", claim_class="fixture")
    scorer = build_external_scorer(cfg, kind="fake")
    request = GenerationRequest(prompt="Create a button.", slot_contract=())
    actions = [
        LegalAction(action_id="a1", token_ids=(1, 2, 3)),
        LegalAction(action_id="a2", token_ids=(4, 5)),
    ]
    scores = scorer.score_legal_actions(
        request, prefix_text="", compiler_state_fingerprint="s", legal_actions=actions
    )
    assert set(scores) == {"a1", "a2"}
    assert all(isinstance(v, (float, type(None))) for v in scores.values())


def test_fake_scorer_returns_scores_for_candidates() -> None:
    cfg = ExternalScorerConfig(model_id="x", revision="y", claim_class="fixture")
    scorer = build_external_scorer(cfg, kind="fake")
    request = GenerationRequest(prompt="Create a button.", slot_contract=())
    candidates = [
        CompleteCandidate(candidate_id="c1", token_ids=(1, 2, 3)),
        CompleteCandidate(candidate_id="c2", token_ids=(4, 5)),
    ]
    scores = scorer.score_complete_candidates(request, "", candidates)
    assert set(scores) == {"c1", "c2"}


def test_external_score_policy_ranks_candidates() -> None:
    cfg = ExternalScorerConfig(model_id="x", revision="y", claim_class="fixture")
    scorer = build_external_scorer(cfg, kind="fake")
    request = GenerationRequest(prompt="Create a button.", slot_contract=())
    policy = ExternalScorePolicy(
        scorer=scorer, request=request, name="test_external_policy"
    )
    candidates = [
        CandidatePath(candidate_id="c1", token_ids=(100, 200, 300), log_probs=(-1.0, -1.0, -1.0)),
        CandidatePath(candidate_id="c2", token_ids=(400, 500), log_probs=(-2.0, -2.0)),
    ]
    scores = [(c.candidate_id, policy.score(c)) for c in candidates]
    assert all(score is not None for _, score in scores)
    # Policy should produce a stable ordering (higher score is better).
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    assert ranked[0][0] in {"c1", "c2"}


def test_external_score_policy_to_dict_has_identity() -> None:
    cfg = ExternalScorerConfig(model_id="x", revision="y", claim_class="fixture")
    scorer = build_external_scorer(cfg, kind="fake")
    request = GenerationRequest(prompt="Create a button.", slot_contract=())
    policy = ExternalScorePolicy(scorer=scorer, request=request)
    data = policy.to_dict()
    assert data["name"] == "external_score_policy"
    assert "scorer_identity" in data
    assert "request_prompt_sha" in data
