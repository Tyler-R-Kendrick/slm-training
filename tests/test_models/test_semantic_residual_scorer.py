"""Residual scorer authority and anti-E237 guards."""

from __future__ import annotations

from slm_training.data.progspec.semantic_evidence import (
    FactorRepresentation,
    project_program_factors,
)
from slm_training.models.semantic_residual_scorer import (
    SemanticResidualScorerConfig,
    SemanticResidualScorerV1,
    assert_score_keys_legal,
)


def test_singleton_zero_work() -> None:
    snap = project_program_factors(
        program_id="s",
        components=(("root", "Card"),),
        legal_candidate_ids=("=",),
    )
    scorer = SemanticResidualScorerV1(
        SemanticResidualScorerConfig(
            representation=FactorRepresentation.ROLE_PORTED_FACTOR,
            decode_apply=True,
            alpha=1.0,
        )
    )
    result = scorer.score(
        legal_candidate_ids=("=",),
        coverage="complete",
        snapshot=snap,
        expected_source_digest=snap.source_digest,
        baseline_scores={"=": 1.0},
    )
    assert result.singleton_skip is True
    assert result.applications == 0
    assert result.deltas == {}
    assert result.telemetry["neural_forwards"] == 0
    assert result.telemetry["semantic_factor_propagation_calls"] == 0


def test_unknown_preserved_on_incomplete() -> None:
    snap = project_program_factors(
        program_id="u",
        components=(("Card", "Card"),),
        legal_candidate_ids=("a", "b"),
        coverage="incomplete",
    )
    scorer = SemanticResidualScorerV1(
        SemanticResidualScorerConfig(decode_apply=True, alpha=1.0)
    )
    result = scorer.score(
        legal_candidate_ids=("a", "b"),
        coverage="incomplete",
        snapshot=snap,
    )
    assert result.unknown_preserved is True
    assert result.applications == 0


def test_decode_off_zero_alpha_no_application() -> None:
    snap = project_program_factors(
        program_id="d",
        components=(("Card", "Card"), ("b1", "TextContent")),
        binder_edges=(("Card", "b1", "USE"),),
        legal_candidate_ids=("b1", "b2"),
    )
    scorer = SemanticResidualScorerV1(
        SemanticResidualScorerConfig(
            representation=FactorRepresentation.ROLE_PORTED_FACTOR,
            decode_apply=False,
            alpha=0.0,
        )
    )
    result = scorer.score(
        legal_candidate_ids=("b1", "b2"),
        coverage="complete",
        snapshot=snap,
        expected_source_digest=snap.source_digest,
        baseline_scores={"b1": 0.1, "b2": 0.2},
    )
    assert result.applications == 0
    assert result.choice_changed is False
    assert all(v == 0.0 for v in result.deltas.values())


def test_score_keys_must_be_legal() -> None:
    assert_score_keys_legal({"a": 1.0}, ("a", "b"))
    try:
        assert_score_keys_legal({"z": 1.0}, ("a", "b"))
        raise AssertionError("expected illegal key rejection")
    except ValueError as exc:
        assert "outside legal set" in str(exc)


def test_stale_digest_fail_closed() -> None:
    snap = project_program_factors(
        program_id="stale",
        components=(("Card", "Card"),),
        legal_candidate_ids=("a", "b"),
    )
    scorer = SemanticResidualScorerV1(
        SemanticResidualScorerConfig(decode_apply=True, alpha=1.0)
    )
    result = scorer.score(
        legal_candidate_ids=("a", "b"),
        coverage="complete",
        snapshot=snap,
        expected_source_digest="0" * 64,
    )
    assert result.stale_digest_rejection is True
    assert result.applications == 0
