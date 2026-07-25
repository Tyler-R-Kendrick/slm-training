from __future__ import annotations

import math

import pytest

from slm_training.flow.proposals import (
    CandidateFeatureObject,
    CandidateProposalPolicy,
    ProposalTrainingRowV1,
)


def _candidate(candidate_id: str, value: float) -> CandidateFeatureObject:
    return CandidateFeatureObject(
        candidate_id=candidate_id,
        family="ReplaceProduction",
        feature_digest=f"digest-{candidate_id}",
        values=(value,),
    )


def test_policy_is_permutation_stable_and_falls_back_exactly() -> None:
    policy = CandidateProposalPolicy(name="test", k=1)

    def score(_state: str, item: CandidateFeatureObject) -> float:
        return item.values[0]

    left = policy.propose(
        state_fingerprint="state",
        candidates=(_candidate("b", 1.0), _candidate("a", 1.0)),
        score=score,
        calibrated_coverage={1: 0.5},
    )
    right = policy.propose(
        state_fingerprint="state",
        candidates=(_candidate("a", 1.0), _candidate("b", 1.0)),
        score=score,
        calibrated_coverage={1: 0.5},
    )
    assert left.proposed_candidate_ids == right.proposed_candidate_ids == ("a",)
    assert left.fallback_required
    assert left.exact_membership_preserved
    assert left.scheduled_candidate_ids == ("a", "b")


def test_policy_corruption_fails_closed() -> None:
    decision = CandidateProposalPolicy(name="test", k=1).propose(
        state_fingerprint="state",
        candidates=(_candidate("a", 1.0), _candidate("b", 2.0)),
        score=lambda _state, item: math.nan if item.candidate_id == "a" else 1.0,
    )
    assert decision.fallback_reason == "non_finite_score"
    assert decision.exact_membership_preserved


def test_training_row_preserves_multi_positive_and_unknown() -> None:
    row = ProposalTrainingRowV1(
        row_id="row",
        state_fingerprint="state",
        hole_id="hole",
        complete_candidate_ids=("a", "b", "c"),
        target_candidate_ids=("a",),
        acceptable_candidate_ids=("a", "b"),
        supported_candidate_ids=("a", "b"),
        unsupported_candidate_ids=(),
        unknown_candidate_ids=("c",),
        candidate_feature_digests=(("a", "da"), ("b", "db"), ("c", "dc")),
        split="train",
        lineage_digest="lineage",
        checkpoint_digest=None,
        config_digest="config",
        bridge_version="bridge/v1",
    )
    payload = row.to_dict()
    assert payload["acceptable_candidate_ids"] == ["a", "b"]
    assert payload["unknown_candidate_ids"] == ["c"]
    assert not payload["contains_final_source"]
    assert not payload["contains_future_witness_text"]


def test_training_row_rejects_unknown_negative() -> None:
    with pytest.raises(ValueError, match="UNKNOWN"):
        ProposalTrainingRowV1(
            row_id="row",
            state_fingerprint="state",
            hole_id="hole",
            complete_candidate_ids=("a",),
            target_candidate_ids=(),
            acceptable_candidate_ids=(),
            supported_candidate_ids=(),
            unsupported_candidate_ids=("a",),
            unknown_candidate_ids=("a",),
            candidate_feature_digests=(("a", "da"),),
            split="dev",
            lineage_digest="lineage",
            checkpoint_digest=None,
            config_digest="config",
            bridge_version="bridge/v1",
        )
