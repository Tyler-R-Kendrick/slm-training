"""Tests for the dynamic pointer scorer module (SLM-168)."""

from __future__ import annotations

import pytest
import torch

from slm_training.models.dynamic_pointer_scorer import (
    DynamicPointerScorer,
    DynamicPointerScorerConfig,
    PointerCandidate,
    PointerCandidateSet,
    count_pointer_scorer_parameters,
    estimate_pointer_scorer_flops,
)


def _sample_candidate_set() -> PointerCandidateSet:
    return PointerCandidateSet(
        candidates=(
            PointerCandidate(
                stable_id=":hero.title",
                display_text="Hero title",
                kind="slot",
                type_name="string",
                provenance="request_contract",
            ),
            PointerCandidate(
                stable_id=":hero.body",
                display_text="Hero body",
                kind="slot",
                type_name="string",
                provenance="request_contract",
            ),
            PointerCandidate(
                stable_id="schema:Hero",
                display_text="Hero",
                kind="schema_entity",
                type_name="component",
                provenance="compiler_scope",
            ),
        ),
        permitted_sources=("request_contract", "compiler_scope"),
        manifest_hash="abc123",
    )


def test_legacy_tokens_mode_has_no_parameters() -> None:
    config = DynamicPointerScorerConfig(pointer_mode="legacy_tokens")
    scorer = DynamicPointerScorer(config)
    assert scorer.scorer is None
    assert count_pointer_scorer_parameters(scorer) == 0


def test_dynamic_head_mode_has_parameters() -> None:
    config = DynamicPointerScorerConfig(
        pointer_mode="dynamic_head",
        d_model=64,
        pointer_hidden_dim=128,
        pointer_heads=2,
    )
    scorer = DynamicPointerScorer(config)
    assert scorer.scorer is not None
    params = count_pointer_scorer_parameters(scorer)
    assert params > 0


def test_forward_shape() -> None:
    config = DynamicPointerScorerConfig(
        pointer_mode="dynamic_head",
        d_model=64,
    )
    scorer = DynamicPointerScorer(config)
    candidates = _sample_candidate_set()
    state_vec = torch.randn(64)
    logits = scorer.forward(state_vec, candidates)
    assert logits.shape == (len(candidates),)


def test_legacy_tokens_forward_returns_zeros() -> None:
    config = DynamicPointerScorerConfig(pointer_mode="legacy_tokens")
    scorer = DynamicPointerScorer(config)
    candidates = _sample_candidate_set()
    state_vec = torch.randn(64)
    logits = scorer.forward(state_vec, candidates)
    assert torch.allclose(logits, torch.zeros(len(candidates)), atol=1e-6)


def test_score_returns_log_probabilities() -> None:
    config = DynamicPointerScorerConfig(pointer_mode="dynamic_head", d_model=64)
    scorer = DynamicPointerScorer(config)
    candidates = _sample_candidate_set()
    state_vec = torch.randn(64)
    log_probs = scorer.score(state_vec, candidates)
    assert log_probs.shape == (len(candidates),)
    assert torch.allclose(log_probs.exp().sum(), torch.tensor(1.0), atol=1e-4)


def test_score_respects_mask() -> None:
    config = DynamicPointerScorerConfig(pointer_mode="dynamic_head", d_model=64)
    scorer = DynamicPointerScorer(config)
    candidates = _sample_candidate_set()
    state_vec = torch.randn(64)
    mask = torch.tensor([True, False, True])
    log_probs = scorer.score(state_vec, candidates, mask=mask)
    assert log_probs[1].item() == float("-inf")
    assert torch.allclose(log_probs.exp().sum(), torch.tensor(1.0), atol=1e-4)


def test_invalid_pointer_mode_raises() -> None:
    config = DynamicPointerScorerConfig(pointer_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DynamicPointerScorer(config)


def test_save_and_load_roundtrip(tmp_path) -> None:
    config = DynamicPointerScorerConfig(
        pointer_mode="dynamic_head",
        d_model=64,
        pointer_hidden_dim=128,
    )
    scorer = DynamicPointerScorer(config)
    path = tmp_path / "pointer_scorer.json"
    scorer.save(path)
    loaded = DynamicPointerScorer.from_checkpoint(path)
    assert loaded.config.pointer_mode == "dynamic_head"
    assert loaded.config.d_model == 64


def test_estimate_flops() -> None:
    config = DynamicPointerScorerConfig(
        pointer_mode="dynamic_head",
        d_model=64,
        pointer_hidden_dim=128,
    )
    scorer = DynamicPointerScorer(config)
    flops = estimate_pointer_scorer_flops(scorer, n_candidates=5)
    assert flops > 0

    legacy = DynamicPointerScorer(DynamicPointerScorerConfig(pointer_mode="legacy_tokens"))
    assert estimate_pointer_scorer_flops(legacy, n_candidates=5) == 0
