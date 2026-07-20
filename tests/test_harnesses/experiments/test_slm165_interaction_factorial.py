"""Tests for SLM-165 (SDE1-03) 2×2×2 interaction factorial fixture harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.slm165_interaction_factorial import (
    ACTION_INIT_LEVELS,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    SIBLING_MARGIN_LEVELS,
    TYPE_BALANCE_LEVELS,
    InteractionCell,
    build_cells,
    resolve_action_init_winner,
    resolve_sibling_margin_winner,
    run_fixture_campaign,
    validate_manifest,
)

def test_build_cells_produces_eight_per_seed() -> None:
    cells = build_cells(seeds=(0, 1, 2))
    assert len(cells) == 24
    per_seed = {}
    for cell in cells:
        per_seed.setdefault(cell.seed, set()).add(cell.cell_id)
    assert len(per_seed) == 3
    for seed, ids in per_seed.items():
        assert len(ids) == 8, f"seed {seed} has {len(ids)} cells"


def test_cells_contain_all_factor_values() -> None:
    cells = build_cells(seeds=(0,))
    seen_action = {c.action_init for c in cells}
    seen_balance = {c.type_balance for c in cells}
    seen_margin = {c.sibling_margin for c in cells}
    assert seen_action == set(ACTION_INIT_LEVELS)
    assert seen_balance == set(TYPE_BALANCE_LEVELS)
    assert seen_margin == set(SIBLING_MARGIN_LEVELS)


def test_dependency_resolution_fail_closed_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ValueError, match="not found"):
        resolve_action_init_winner(missing, strict=True)


def test_dependency_resolution_fail_closed_missing_winner_field(tmp_path: Path) -> None:
    path = tmp_path / "no_winner.json"
    path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no winner field"):
        resolve_action_init_winner(path, strict=True)


def test_sibling_margin_resolution_fail_closed_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ValueError, match="not found"):
        resolve_sibling_margin_winner(missing, strict=True)


def test_target_decisions_equal_across_cells() -> None:
    cells = build_cells(seeds=(0, 1, 2))
    decisions = {c.target_decisions for c in cells}
    assert len(decisions) == 1


def test_synthetic_fixture_recovers_interaction() -> None:
    report = run_fixture_campaign(
        run_id="test_slm165_interaction",
        seeds=(0, 1, 2),
        strict_winners=False,
    )
    fa = report.factorial_analysis
    assert fa["full_minus_best_two_way"] >= 0.05
    assert fa["three_way_interaction"] > 0
    assert fa["verdict"] == "synergistic"


def test_factorial_analysis_has_main_effects_and_interactions() -> None:
    report = run_fixture_campaign(
        run_id="test_slm165_analysis",
        seeds=(0, 1, 2),
        strict_winners=False,
    )
    fa = report.factorial_analysis
    assert "main_effects" in fa
    assert "action_init" in fa["main_effects"]
    assert "type_balance" in fa["main_effects"]
    assert "sibling_margin" in fa["main_effects"]
    assert "two_way_interactions" in fa
    assert len(fa["two_way_interactions"]) == 3
    assert "three_way_interaction" in fa


def test_report_version_stamp() -> None:
    report = run_fixture_campaign(
        run_id="test_slm165_stamp",
        seeds=(0,),
        strict_winners=False,
    )
    assert report.version_stamp["stamp_schema"] == "version_stamp/v1"
    components = report.version_stamp.get("components", {})
    assert "harness.experiments" in components
    assert "harness.experiments.slm165_interaction_factorial" in components


def test_validate_manifest() -> None:
    cells = build_cells(seeds=(0,))
    assert validate_manifest(cells) == []

    duplicated = cells + (cells[0],)
    errors = validate_manifest(duplicated)
    assert any("duplicate cell_id" in e for e in errors)

    bad = InteractionCell(
        cell_id="A_bad",
        action_init="invalid_action",
        type_balance="neutral",
        sibling_margin="none",
        seed=0,
        action_embedding_init="invalid_action",
        slot_component_loss_weight=1.0,
        slot_component_class_balance_power=0.0,
        slot_component_owner_rare_threshold=0,
        slot_component_owner_rare_multiplier=1,
        legal_margin_mode="none",
        targeted_margin_value=0.0,
        target_decisions=1000,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("invalid action_init" in e for e in errors)


def test_run_fixture_campaign_status_and_claim_class() -> None:
    report = run_fixture_campaign(
        run_id="test_slm165_status",
        seeds=(0,),
        strict_winners=False,
    )
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.experiment_id == EXPERIMENT_ID
