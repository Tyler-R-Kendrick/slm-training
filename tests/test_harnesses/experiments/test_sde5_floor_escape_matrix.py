"""Tests for SLM-210 (SDE5-03) floor-escape matrix fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm209_debt_targeted_curriculum import (
    SDE5_EXPERIMENT_ID,
    SDE5_MATRIX_SET,
    SDE5_MATRIX_VERSION,
    SDE5FloorEscapeCellV1,
    SDE5FloorEscapeMatrixV1,
    build_sde5_floor_escape_matrix,
    build_synthetic_debt_and_events,
    render_sde5_floor_escape_markdown,
    run_sde5_floor_escape_fixture,
    validate_sde5_floor_escape_matrix,
)


def _build_matrix(n_states: int = 80, seeds: tuple[int, ...] = (0,)) -> SDE5FloorEscapeMatrixV1:
    debts, events = build_synthetic_debt_and_events(n_states=n_states, seed=0)
    return build_sde5_floor_escape_matrix(
        debts,
        events,
        seeds=seeds,
        total_decision_budget=40,
        per_group_cap=4,
    )


def test_matrix_has_preregistered_cells() -> None:
    matrix = _build_matrix()
    cell_ids = {cell.cell_id for cell in matrix.cells}
    assert cell_ids == {f"C{i}" for i in range(9)}
    for cell in matrix.cells:
        assert isinstance(cell, SDE5FloorEscapeCellV1)


def test_cells_vary_axes_correctly() -> None:
    matrix = _build_matrix()
    by_id = {cell.cell_id: cell for cell in matrix.cells}
    assert by_id["C0"].prompt_plan_soft_features is False
    assert by_id["C0"].grammar_aligned_mass is False
    assert by_id["C0"].exposure_policy == "uniform"

    assert by_id["C1"].prompt_plan_soft_features is True
    assert by_id["C1"].grammar_aligned_mass is False
    assert by_id["C1"].exposure_policy == "uniform"

    assert by_id["C2"].prompt_plan_soft_features is False
    assert by_id["C2"].grammar_aligned_mass is True
    assert by_id["C2"].exposure_policy == "uniform"

    assert by_id["C3"].prompt_plan_soft_features is False
    assert by_id["C3"].grammar_aligned_mass is False
    assert by_id["C3"].exposure_policy == "preregistered_composite"

    assert by_id["C7"].prompt_plan_soft_features is True
    assert by_id["C7"].grammar_aligned_mass is True
    assert by_id["C7"].exposure_policy == "preregistered_composite"

    assert by_id["C8"].prompt_plan_soft_features is True
    assert by_id["C8"].grammar_aligned_mass is True
    assert by_id["C8"].exposure_policy == "debt_weight_permuted"


def test_no_gold_plan_features_in_production_arms() -> None:
    matrix = _build_matrix()
    for cell in matrix.cells:
        for feature in cell.plan_features:
            assert feature.get("provenance") != "gold"
            assert "gold" not in feature.get("role_id", "")


def test_exposure_budget_equality_across_matched_cells() -> None:
    matrix = _build_matrix()
    for cell in matrix.cells:
        assert cell.selection_cell.decision_budget == matrix.total_decision_budget
        assert cell.selection_cell.per_group_cap == matrix.per_group_cap
        assert len(cell.selection_cell.selections) == matrix.total_decision_budget


def test_anti_gaming_suite_for_non_control_cells() -> None:
    matrix = _build_matrix()
    for cell in matrix.cells:
        if cell.cell_id == "C0":
            assert cell.runs_anti_gaming_suite is False
        else:
            assert cell.runs_anti_gaming_suite is True


def test_checkpoint_selection_ignores_final_suite() -> None:
    matrix = _build_matrix()
    for cell in matrix.cells:
        assert "final_suite" not in cell.checkpoint_selection_rule


def test_default_off_path_matches_control() -> None:
    matrix = _build_matrix()
    c0 = next(cell for cell in matrix.cells if cell.cell_id == "C0")
    assert c0.prompt_plan_soft_features is False
    assert c0.grammar_aligned_mass is False
    assert c0.exposure_policy == "uniform"
    assert c0.mass_audit["policy"] == "raw_cumulative"
    assert c0.plan_features == ()


def test_mass_audit_uses_grammar_aligned_mass_when_enabled() -> None:
    matrix = _build_matrix()
    c2 = next(cell for cell in matrix.cells if cell.cell_id == "C2")
    assert c2.grammar_aligned_mass is True
    assert c2.mass_audit["policy"] == "grammar_aligned_mass"
    assert "score" in c2.mass_audit


def test_asap_decode_follows_mass_lever() -> None:
    matrix = _build_matrix()
    for cell in matrix.cells:
        assert cell.asap_decode == cell.grammar_aligned_mass


def test_matrix_round_trip() -> None:
    matrix = _build_matrix()
    reconstructed = SDE5FloorEscapeMatrixV1.from_dict(matrix.to_dict())
    assert reconstructed.matrix_set == SDE5_MATRIX_SET
    assert reconstructed.matrix_version == SDE5_MATRIX_VERSION
    assert reconstructed.experiment_id == SDE5_EXPERIMENT_ID
    assert len(reconstructed.cells) == len(matrix.cells)


def test_validate_sde5_floor_escape_matrix_accepts_valid() -> None:
    matrix = _build_matrix()
    assert validate_sde5_floor_escape_matrix(matrix) == []


def test_validate_rejects_missing_cells() -> None:
    matrix = _build_matrix()
    matrix = SDE5FloorEscapeMatrixV1.from_dict(
        {
            **matrix.to_dict(),
            "cells": [matrix.cells[0].to_dict()],
            "lineage": {**matrix.lineage, "seeds": [0]},
        }
    )
    errors = validate_sde5_floor_escape_matrix(matrix)
    assert any("missing cells" in e for e in errors)


def test_validate_rejects_gold_plan_feature() -> None:
    matrix = _build_matrix()
    bad_cell = SDE5FloorEscapeCellV1.from_dict(
        {
            **matrix.cells[0].to_dict(),
            "plan_features": [
                {"role_id": "gold_plan", "provenance": "gold"},
            ],
        }
    )
    matrix = SDE5FloorEscapeMatrixV1.from_dict(
        {**matrix.to_dict(), "cells": [bad_cell.to_dict()]}
    )
    errors = validate_sde5_floor_escape_matrix(matrix)
    assert any("gold plan feature" in e for e in errors)


def test_validate_rejects_control_with_anti_gaming() -> None:
    matrix = _build_matrix()
    c0 = next(cell for cell in matrix.cells if cell.cell_id == "C0")
    bad_cell = SDE5FloorEscapeCellV1.from_dict(
        {**c0.to_dict(), "runs_anti_gaming_suite": True}
    )
    cells = [cell.to_dict() for cell in matrix.cells if cell.cell_id != "C0"]
    cells.append(bad_cell.to_dict())
    matrix = SDE5FloorEscapeMatrixV1.from_dict(
        {**matrix.to_dict(), "cells": cells}
    )
    errors = validate_sde5_floor_escape_matrix(matrix)
    assert any("C0 must not run anti-gaming" in e for e in errors)


def test_run_fixture_campaign_produces_manifest() -> None:
    matrix = run_sde5_floor_escape_fixture(
        output_dir=None,
        seeds=(0,),
        n_states=80,
        total_decision_budget=40,
        per_group_cap=4,
        seed=0,
        write_design_docs=False,
    )
    assert matrix.matrix_set == SDE5_MATRIX_SET
    assert matrix.matrix_version == SDE5_MATRIX_VERSION
    assert matrix.experiment_id == SDE5_EXPERIMENT_ID
    assert len(matrix.cells) == 9
    assert matrix.lineage["synthetic_state_count"] == 80


def test_version_stamp_includes_sde5_component() -> None:
    matrix = run_sde5_floor_escape_fixture(
        output_dir=None,
        seeds=(0,),
        n_states=40,
        total_decision_budget=20,
        per_group_cap=4,
        seed=0,
        write_design_docs=False,
    )
    components = matrix.version_stamp.get("components", {})
    assert "harness.experiments" in components
    assert "harness.experiments.slm209_debt_targeted_curriculum" in components
    assert "harness.experiments.sde5_floor_escape_matrix" in components
    assert "harness.preference.constraint_debt" in components


def test_render_markdown_contains_all_cells() -> None:
    matrix = _build_matrix()
    md = render_sde5_floor_escape_markdown(matrix)
    for cell in matrix.cells:
        assert cell.cell_id in md
    assert "wiring / fixture only" in md
    assert "SDE5-03" in md


def test_multiple_seeds_expand_cells() -> None:
    matrix = _build_matrix(seeds=(0, 1))
    assert len(matrix.cells) == 18
    by_seed: dict[int, set[str]] = {}
    for cell in matrix.cells:
        by_seed.setdefault(cell.selection_cell.seed, set()).add(cell.cell_id)
    assert by_seed[0] == by_seed[1] == {f"C{i}" for i in range(9)}
