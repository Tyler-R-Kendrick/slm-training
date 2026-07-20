"""Tests for slm_training.models.global_semantic_critic_selector (SLM-150)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from slm_training.harnesses.experiments.candidate_selector import SelectionCandidate
from slm_training.models.global_semantic_critic_selector import (
    GlobalSemanticCriticSelector,
)
from slm_training.models.global_semantic_critic import (
    GlobalSemanticCritic,
    GlobalSemanticCriticConfig,
)


def _candidate(
    candidate_id: str,
    *,
    value_score: float | None = 0.5,
    generator_score: float | None = None,
    features: dict | None = None,
) -> SelectionCandidate:
    return SelectionCandidate(
        candidate_id=candidate_id,
        canonical_program=f"(program {candidate_id})",
        ast_fingerprint=f"fp_{candidate_id}",
        generator_id="genA",
        generator_score=generator_score,
        value_score=value_score,
        energy_score=None,
        semantic_success=None,
        acceptable_set=False,
        available_features=features or {"component_count": 2, "depth": 1},
    )


def _selector(threshold: float = 0.0, lambda_global: float = 1.0) -> GlobalSemanticCriticSelector:
    critic = GlobalSemanticCritic(
        config=GlobalSemanticCriticConfig(seed=0, confidence_threshold=threshold),
        device="cpu",
    )
    return GlobalSemanticCriticSelector(
        critic, lambda_global=lambda_global, confidence_threshold=threshold
    )


def test_selector_candidate_set_immutable() -> None:
    selector = _selector(threshold=0.0)
    candidates = [
        _candidate("a", value_score=0.1),
        _candidate("b", value_score=0.9),
        _candidate("c", value_score=0.5),
    ]
    decision = selector.select(
        prompt_context={"pack_id": "openui"},
        structured_contract={"required_component_count": 3},
        candidates=candidates,
    )
    input_ids = {c.candidate_id for c in candidates}
    assert decision.selected_candidate_id in input_ids
    assert not decision.abstained


def test_selector_deterministic_same_group() -> None:
    selector = _selector(threshold=0.0)
    candidates = [
        _candidate("a", value_score=0.2),
        _candidate("b", value_score=0.8),
        _candidate("c", value_score=0.5),
    ]
    decision1 = selector.select(
        prompt_context={"pack_id": "openui"},
        structured_contract={"required_component_count": 3},
        candidates=candidates,
    )
    decision2 = selector.select(
        prompt_context={"pack_id": "openui"},
        structured_contract={"required_component_count": 3},
        candidates=candidates,
    )
    assert decision1.selected_candidate_id == decision2.selected_candidate_id


def test_selector_lambda_zero_uses_local_scores_only() -> None:
    # With lambda=0 the critic energy is ignored; the highest value_score wins.
    selector = _selector(threshold=0.0, lambda_global=0.0)
    candidates = [
        _candidate("low_value", value_score=0.1, features={"component_count": 5}),
        _candidate("high_value", value_score=0.9, features={"component_count": 1}),
    ]
    decision = selector.select(
        prompt_context={"pack_id": "openui"},
        structured_contract={"required_component_count": 4},
        candidates=candidates,
    )
    assert decision.selected_candidate_id == "high_value"


def test_selector_abstains_when_all_candidates_unsupported() -> None:
    selector = _selector(threshold=0.0)
    candidates = [
        _candidate("a", value_score=0.5),
        _candidate("b", value_score=0.6),
    ]
    decision = selector.select(
        prompt_context={"pack_id": "unsupported"},
        structured_contract={"required_component_count": 3},
        candidates=candidates,
    )
    assert decision.abstained
    assert decision.selected_candidate_id is None
    assert decision.reason_code == "all_abstained"


def test_selector_abstains_on_low_confidence() -> None:
    selector = _selector(threshold=1.0)
    candidates = [
        _candidate("a", value_score=0.5),
        _candidate("b", value_score=0.6),
    ]
    decision = selector.select(
        prompt_context={"pack_id": "openui"},
        structured_contract={"required_component_count": 3},
        candidates=candidates,
    )
    assert decision.abstained
    assert decision.selected_candidate_id is None
    assert decision.reason_code == "all_abstained"


def test_selector_empty_candidates() -> None:
    selector = _selector(threshold=0.0)
    decision = selector.select(
        prompt_context={"pack_id": "openui"},
        structured_contract={},
        candidates=[],
    )
    assert decision.abstained
    assert decision.reason_code == "empty_candidate_set"
