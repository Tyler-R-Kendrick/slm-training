"""B3 capacity ladder (SLM-23): matched surface vs choice representation arms.

Pins the matched-recipe contract the capacity-vs-d_model study depends on: the
two output-representation arms (``lexer`` surface-token control vs ``choice``
codec) register the *same* rung set and differ only in ``output_tokenizer``.
Registration/wiring evidence only — no trained-quality claim here.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.ladder import (
    CAPACITY_ARMS,
    CAPACITY_WIDTHS,
    capacity_ladder,
    capacity_ladder_arms,
    model_build_config_for_point,
)
from scripts.run_scaling_ladder import _wall_minutes
from slm_training.levers import MAX_RUN_MINUTES

# Non-tokenizer ModelBuildConfig fields that must be identical across the two
# arms for the comparison to isolate the representation. output_tokenizer is the
# one field allowed to differ; run_id/run_root differ only by the arm's
# ladder_id, so they are excluded from the matched-field check.
_ALLOWED_TO_DIFFER = {"output_tokenizer", "run_id", "run_root"}


def test_capacity_arms_match_on_everything_but_tokenizer(tmp_path: Path) -> None:
    arms = capacity_ladder_arms(base_token_budget=1_000)

    # Both arms registered.
    assert set(arms) == set(CAPACITY_ARMS) == {"lexer", "choice"}

    lexer, choice = arms["lexer"], arms["choice"]

    # >= 3 d_model rungs, matched rung set across arms.
    assert len(CAPACITY_WIDTHS) >= 3
    lexer_ids = [p.point_id for p in lexer.points]
    choice_ids = [p.point_id for p in choice.points]
    assert lexer_ids == choice_ids
    assert len(lexer_ids) == len(CAPACITY_WIDTHS)
    assert {p.d_model for p in lexer.points} == set(CAPACITY_WIDTHS)

    # Only output_tokenizer differs on the ladder itself.
    assert lexer.output_tokenizer == "lexer"
    assert choice.output_tokenizer == "choice"
    assert lexer.points == choice.points  # identical widths/depths/budgets

    # The 3 rungs x 2 arms row set = 6 registered rows, all matched-recipe.
    rows = 0
    for point in lexer.points:
        lex_cfg = model_build_config_for_point(
            point, lexer, train_dir=tmp_path, test_dir=None,
            run_root=tmp_path / "runs", seed=0, steps=10, batch_size=2,
        )
        cho_cfg = model_build_config_for_point(
            point, choice, train_dir=tmp_path, test_dir=None,
            run_root=tmp_path / "runs", seed=0, steps=10, batch_size=2,
        )
        assert lex_cfg.output_tokenizer == "lexer"
        assert cho_cfg.output_tokenizer == "choice"
        # Matched recipe: same mask pattern, non-LTR, same width/budget/steps.
        assert lex_cfg.mask_pattern == cho_cfg.mask_pattern == "diffusion"
        assert lex_cfg.grammar_ltr_primary is cho_cfg.grammar_ltr_primary is False
        for f in fields(lex_cfg):
            if f.name in _ALLOWED_TO_DIFFER:
                continue
            assert getattr(lex_cfg, f.name) == getattr(cho_cfg, f.name), f.name
        rows += 2
    assert rows == 2 * len(CAPACITY_WIDTHS)


def test_capacity_ladder_single_arm_is_scratch_track() -> None:
    lad = capacity_ladder("choice", base_token_budget=1_000)
    assert lad.track == "scratch"
    assert lad.ladder_id == "capacity_choice_v1"
    assert (lad.decode_frozen or {}).get("mask_pattern") == "diffusion"
    assert (lad.decode_frozen or {}).get("grammar_ltr_primary") is False


def test_ladder_wall_budget_is_configurable_but_capped() -> None:
    assert _wall_minutes("0.25") == 0.25
    assert _wall_minutes(str(MAX_RUN_MINUTES)) == float(MAX_RUN_MINUTES)
    with pytest.raises(
        argparse.ArgumentTypeError, match=f"at most {MAX_RUN_MINUTES}"
    ):
        _wall_minutes(str(MAX_RUN_MINUTES + 0.1))
