"""Tests for SLM-418 (DSH5-10) seventh slice: held-out pairwise-preference
measurement against a real ``TwoTowerModel`` checkpoint.

Covers the new functions in ``slm_training.harnesses.preference.train``
(``pairwise_preference_margin``, ``held_out_pairwise_accuracy``,
``evaluate_replay_preference_held_out_benefit``). See
``docs/design/dsh5-10-replay-preference-rows.md`` "Seventh slice" for the
full disposition -- this is fixture-scale wiring evidence, never a certified
benefit claim.
"""

from __future__ import annotations

import pytest
import torch

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.preference import PreferencePair
from slm_training.harnesses.preference.train import (
    evaluate_replay_preference_held_out_benefit,
    held_out_pairwise_accuracy,
    pairwise_preference_margin,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def _tiny_model() -> TwoTowerModel:
    record = ExampleRecord(
        id="a",
        prompt="Card",
        openui='root = TextContent(":slot_0")',
        split="train",
        placeholders=[":slot_0"],
    )
    return TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            d_model=16,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            max_target_len=8,
            seed=0,
        ),
        device="cpu",
    )


def _pair(prompt: str = "root = TextContent(\":hero.title\")") -> PreferencePair:
    return PreferencePair(
        prompt=prompt,
        chosen="undo",
        rejected="checkout:some-other-state",
        design_md=None,
        chosen_score=0.0,
        rejected_score=0.0,
        meta={"schema": "operator_replay_preference_pair/v1"},
    )


# --------------------------------------------------------------------------- #
# pairwise_preference_margin / held_out_pairwise_accuracy
# --------------------------------------------------------------------------- #
def test_pairwise_preference_margin_is_a_finite_float() -> None:
    model = _tiny_model()
    margin = pairwise_preference_margin(model, _pair())
    assert isinstance(margin, float)
    assert margin == margin  # not NaN


def test_held_out_pairwise_accuracy_is_bounded_and_reports_n_pairs() -> None:
    model = _tiny_model()
    pairs = [_pair(), _pair(prompt="root = TextContent(\":hero.other\")")]
    result = held_out_pairwise_accuracy(model, pairs)
    assert 0.0 <= result["pairwise_preference_accuracy"] <= 1.0
    assert result["n_pairs"] == 2
    assert isinstance(result["mean_margin"], float)


def test_held_out_pairwise_accuracy_refuses_an_empty_pair_list() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError):
        held_out_pairwise_accuracy(model, [])


def test_held_out_pairwise_accuracy_is_reproducible_under_a_fixed_seed() -> None:
    """Not unconditionally deterministic: ``_logprob_of_target`` applies fresh
    random masking noise each call (the same noise ``dpo_loss`` trains
    under), so this only reproduces when the caller fixes the RNG themselves
    immediately before each call -- exactly as ``held_out_pairwise_accuracy``'s
    own docstring says.
    """
    model = _tiny_model()
    pairs = [_pair()]
    torch.manual_seed(0)
    first = held_out_pairwise_accuracy(model, pairs)
    torch.manual_seed(0)
    second = held_out_pairwise_accuracy(model, pairs)
    assert first == second


# --------------------------------------------------------------------------- #
# evaluate_replay_preference_held_out_benefit
# --------------------------------------------------------------------------- #
def test_evaluate_held_out_benefit_with_only_a_baseline_checkpoint(tmp_path) -> None:
    model = _tiny_model()
    ckpt = tmp_path / "baseline.pt"
    model.save(ckpt)

    report = evaluate_replay_preference_held_out_benefit(
        baseline_checkpoint=ckpt,
        held_out_pairs=[_pair()],
        trained_checkpoint=None,
        device="cpu",
    )
    assert report["verdict"] == "no_trained_checkpoint"
    assert report["trained"] is None
    assert 0.0 <= report["baseline"]["pairwise_preference_accuracy"] <= 1.0


def test_evaluate_held_out_benefit_compares_baseline_and_trained_checkpoints(
    tmp_path,
) -> None:
    baseline_model = _tiny_model()
    baseline_ckpt = tmp_path / "baseline.pt"
    baseline_model.save(baseline_ckpt)

    # A second checkpoint sharing the same architecture (a real, distinct
    # TwoTowerModel instance -- different init seed, so its weights differ
    # from the baseline's) stands in for a "trained" checkpoint here; the
    # real end-to-end pipeline instead runs `slm preference train` between
    # the two saves (see the script/doc reproduction commands).
    trained_model = TwoTowerModel.from_records(
        [
            ExampleRecord(
                id="a",
                prompt="Card",
                openui='root = TextContent(":slot_0")',
                split="train",
                placeholders=[":slot_0"],
            )
        ],
        config=TwoTowerConfig(
            d_model=16, n_heads=4, context_layers=1, denoiser_layers=1,
            max_target_len=8, seed=1,
        ),
        device="cpu",
    )
    trained_ckpt = tmp_path / "trained.pt"
    trained_model.save(trained_ckpt)

    report = evaluate_replay_preference_held_out_benefit(
        baseline_checkpoint=baseline_ckpt,
        held_out_pairs=[_pair(), _pair(prompt="root = TextContent(\":hero.other\")")],
        trained_checkpoint=trained_ckpt,
        device="cpu",
    )
    assert report["verdict"] in {
        "benefit_observed_fixture_scale",
        "no_benefit_fixture_scale",
    }
    assert report["trained"] is not None
    assert report["baseline"]["n_pairs"] == 2
    assert report["trained"]["n_pairs"] == 2
