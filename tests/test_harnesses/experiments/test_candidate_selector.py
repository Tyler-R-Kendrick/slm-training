"""Tests for slm_training.harnesses.experiments.candidate_selector (SLM-127)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.harnesses.experiments.candidate_selector import (
    CandidateSelectionGroupV1,
    EnergyScoreSelector,
    HardThenSimpleSelector,
    LearnedCandidateSelector,
    ModelScoreSelector,
    SelectionCandidate,
    ValueScoreSelector,
    evaluate_selector,
    load_selection_groups,
    make_fixture_groups,
    select_threshold_on_validation,
    selection_group_from_dict,
    selection_group_to_dict,
    train_selector_fixture,
    write_selection_groups,
)
from slm_training.lineage.records import content_sha


def _candidate(
    candidate_id: str,
    *,
    generator_score: float | None = 0.5,
    value_score: float | None = 0.5,
    energy_score: float | None = 0.5,
    semantic_success: bool | None = None,
    acceptable: bool = False,
    generator_id: str = "genA",
) -> SelectionCandidate:
    return SelectionCandidate(
        candidate_id=candidate_id,
        canonical_program=f"(program {candidate_id})",
        ast_fingerprint=f"fp_{candidate_id}",
        generator_id=generator_id,
        generator_score=generator_score,
        value_score=value_score,
        energy_score=energy_score,
        semantic_success=semantic_success,
        acceptable_set=acceptable,
        available_features={},
    )


def _group(
    candidates: list[SelectionCandidate],
    *,
    acceptable: list[str] | None = None,
    split: str = "train",
    group_id: str = "g",
) -> CandidateSelectionGroupV1:
    acceptable = acceptable or []
    return CandidateSelectionGroupV1(
        group_id=group_id,
        prompt_hash=content_sha(group_id),
        contract_hash=content_sha(group_id + "_contract"),
        generator_id="genA",
        checkpoint_sha=content_sha({"group": group_id}),
        seed=0,
        k=len(candidates),
        candidates=tuple(candidates),
        acceptable_set=tuple(acceptable),
        oracle_best_id=acceptable[0] if acceptable else None,
        split=split,
    )


def test_selection_candidate_round_trip() -> None:
    original = _candidate("c1", generator_score=0.9, acceptable=True)
    restored = SelectionCandidate.from_dict(original.to_dict())
    assert restored == original


def test_group_round_trip() -> None:
    group = _group(
        [_candidate("c0", acceptable=True), _candidate("c1")],
        acceptable=["c0"],
        split="validation",
        group_id="roundtrip",
    )
    data = selection_group_to_dict(group)
    restored = selection_group_from_dict(data)
    assert restored == group


def test_jsonl_round_trip(tmp_path: Path) -> None:
    groups = make_fixture_groups()
    path = tmp_path / "groups.jsonl"
    write_selection_groups(str(path), groups)
    loaded = load_selection_groups(str(path))
    assert loaded == groups


def test_baseline_selectors_avoid_hard_failed_candidates() -> None:
    groups = make_fixture_groups()
    selectors = [
        ModelScoreSelector(),
        ValueScoreSelector(),
        EnergyScoreSelector(),
        HardThenSimpleSelector(),
    ]
    for selector in selectors:
        for group in groups:
            decision = selector.select(
                prompt_context={},
                structured_contract={},
                candidates=group.candidates,
            )
            assert decision.selected_candidate_id is not None
            selected = next(
                c for c in group.candidates if c.candidate_id == decision.selected_candidate_id
            )
            assert selected.semantic_success is not False, (
                f"{selector.selector_id} selected a hard-failed candidate in {group.group_id}"
            )


def test_missing_score_fallback() -> None:
    group = _group(
        [
            _candidate("c0", generator_score=None),
            _candidate("c1", generator_score=None),
        ],
        group_id="missing",
    )
    selector = ModelScoreSelector()
    decision = selector.select(
        prompt_context={},
        structured_contract={},
        candidates=group.candidates,
    )
    assert decision.selected_candidate_id in {"c0", "c1"}
    assert decision.reason_code == "missing_score_fallback"


def test_nan_score_fallback() -> None:
    group = _group(
        [
            _candidate("c0", generator_score=float("nan")),
            _candidate("c1", generator_score=float("nan")),
        ],
        group_id="nan",
    )
    selector = ModelScoreSelector()
    decision = selector.select(
        prompt_context={},
        structured_contract={},
        candidates=group.candidates,
    )
    assert decision.selected_candidate_id in {"c0", "c1"}
    assert decision.reason_code == "missing_score_fallback"


def test_metrics_distinguish_pass_at_k_and_selected_pass_at_k() -> None:
    # All groups have an acceptable candidate, but the model-score selector picks a
    # non-acceptable one because the acceptable candidate has the lower score.
    group = _group(
        [
            _candidate("bad", generator_score=0.9, acceptable=False),
            _candidate("good", generator_score=0.1, acceptable=True),
        ],
        acceptable=["good"],
        split="test",
        group_id="distinguish",
    )
    metrics = evaluate_selector([group], ModelScoreSelector())
    assert metrics["pass_at_k"] == 1.0
    assert metrics["selected_pass_at_k"] == 0.0
    assert metrics["invalid_over_valid_count"] == 1


def test_calibration_uses_validation_split_only() -> None:
    pytest.importorskip("torch")
    groups = make_fixture_groups()
    model, _recipe = train_selector_fixture(groups, epochs=10)
    selector = LearnedCandidateSelector(model, threshold_manifest=None)
    manifest = select_threshold_on_validation(groups, selector, target_risk=0.05)
    val_groups = [g for g in groups if g.split == "validation"]
    expected_hash = content_sha([selection_group_to_dict(g) for g in val_groups])
    assert manifest.calibration_set_hash == expected_hash


def test_learned_selector_abstains_on_no_positive_groups() -> None:
    pytest.importorskip("torch")
    groups = make_fixture_groups()
    model, _recipe = train_selector_fixture(groups, epochs=20)
    selector_no_threshold = LearnedCandidateSelector(model, threshold_manifest=None)
    manifest = select_threshold_on_validation(groups, selector_no_threshold, target_risk=0.05)
    selector = LearnedCandidateSelector(model, threshold_manifest=manifest)
    test_groups = [g for g in groups if g.split == "test"]
    metrics = evaluate_selector(test_groups, selector, threshold_manifest=manifest)
    no_positive = [g for g in test_groups if not g.acceptable_set]
    if no_positive:
        for group in no_positive:
            decision = selector.select(
                prompt_context={},
                structured_contract={},
                candidates=group.candidates,
            )
            assert decision.abstained, (
                f"learned selector should abstain on no-positive group {group.group_id}"
            )
    # The fixture is separable, so the learned selector should not pick invalid over valid.
    assert metrics["invalid_over_valid_count"] == 0
