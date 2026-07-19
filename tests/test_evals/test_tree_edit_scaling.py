"""Regression tests for X22 beam-width × edit-depth scaling wiring (EFS2-01)."""

from __future__ import annotations

import pytest

from slm_training.evals.tree_edit_scaling import (
    InferenceMode,
    TreeEditScalingConfig,
    run_scaling_grid,
    run_tree_edit_scaling_cell,
)


SEED_PROGRAM = (
    'root = Stack([n0], "column")\n'
    'n0 = TextContent(":content.body")'
)
INVENTORY = [":content.body", ":content.title", ":action.save"]


def test_all_beam_states_are_valid() -> None:
    config = TreeEditScalingConfig(beam_width=4, max_edit_depth=2, seed=0)
    result = run_tree_edit_scaling_cell(SEED_PROGRAM, INVENTORY, config)
    assert all(state.valid for state in result.final_beam)


def test_edit_depth_never_exceeds_max() -> None:
    config = TreeEditScalingConfig(beam_width=4, max_edit_depth=2, seed=0)
    result = run_tree_edit_scaling_cell(SEED_PROGRAM, INVENTORY, config)
    assert all(state.edit_depth <= 2 for state in result.final_beam)


def test_beam_width_respected() -> None:
    config = TreeEditScalingConfig(beam_width=2, max_edit_depth=4, seed=0)
    result = run_tree_edit_scaling_cell(SEED_PROGRAM, INVENTORY, config)
    assert len(result.final_beam) <= 2


def test_duplicate_states_not_repeated() -> None:
    config = TreeEditScalingConfig(beam_width=16, max_edit_depth=2, seed=0)
    result = run_tree_edit_scaling_cell(SEED_PROGRAM, INVENTORY, config)
    fps = [state.fingerprint for state in result.final_beam]
    assert len(fps) == len(set(fps))


def test_invalid_seed_raises() -> None:
    with pytest.raises(ValueError):
        run_tree_edit_scaling_cell("not a program", INVENTORY, TreeEditScalingConfig(1, 1))


def test_grid_has_all_nine_cells() -> None:
    result = run_scaling_grid(
        [SEED_PROGRAM],
        INVENTORY,
        seeds=(0, 1),
        expand_per_state=2,
        max_search_steps=4,
        mode=InferenceMode.DETERMINISTIC,
    )
    assert len(result.cells) == 9
    widths = {c.beam_width for c in result.cells}
    depths = {c.max_edit_depth for c in result.cells}
    assert widths == {1, 4, 16}
    assert depths == {1, 2, 4}


def test_telemetry_counts_are_non_negative() -> None:
    config = TreeEditScalingConfig(beam_width=4, max_edit_depth=2, seed=0)
    result = run_tree_edit_scaling_cell(SEED_PROGRAM, INVENTORY, config)
    assert result.telemetry.steps >= 0
    assert result.telemetry.invalid_attempts >= 0
    assert result.telemetry.duplicate_prunes >= 0
    assert result.telemetry.visited_states >= 0


def test_round_trip_dict() -> None:
    result = run_tree_edit_scaling_cell(
        SEED_PROGRAM, INVENTORY, TreeEditScalingConfig(beam_width=1, max_edit_depth=1, seed=7)
    )
    data = result.to_dict()
    assert data["config"]["beam_width"] == 1
    assert data["config"]["max_edit_depth"] == 1
    assert data["seed_program"] == SEED_PROGRAM
