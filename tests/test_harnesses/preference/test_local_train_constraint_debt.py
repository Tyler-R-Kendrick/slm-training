from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.preference.constraint_debt import ConstraintDebtV1
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    split_for_group,
)
from slm_training.harnesses.preference.local_train import (
    _guard_objective_tensors,
    diagnose_metric_complete_gradient_feasibility,
    local_decision_loss,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def _event(
    *,
    good=(0,),
    bad=(1,),
    legal=(0, 1),
    group="group",
    split=None,
    decision_kind="component",
) -> DecisionEventV1:
    target_split = split or "train"
    while split_for_group(group) != target_split:
        group += "x"
    return DecisionEventV1(
        event_id=f"event-{group}-{good}-{bad}",
        group_id=group,
        context_text="Generate a card",
        canvas_ids=(1, 0, 0, 0),
        position=1,
        good_token_ids=good,
        bad_token_ids=bad,
        legal_token_ids=legal,
        evidence_kind="counterfactual",
        evidence_confidence=1.0,
        decision_kind=decision_kind,
        split=target_split,
        policy_checkpoint_sha="policy",
        tokenizer_sha="tokenizer",
        decode_config_hash="decode",
        seed=0,
        trajectory_id="trace",
    )


def _model() -> TwoTowerModel:
    record = ExampleRecord(
        id="a",
        prompt="Card",
        openui='root = TextContent(":card.title")',
        split="train",
        placeholders=[":card.title"],
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


def _approx_dict(left, right, rel=None, abs=None):
    """Recursively assert dict equality with pytest.approx for floats."""
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _approx_dict(left[key], right[key], rel=rel, abs=abs)
    elif isinstance(left, float):
        assert left == pytest.approx(right, rel=rel, abs=abs)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _approx_dict(a, b, rel=rel, abs=abs)
    else:
        assert left == right


def test_diagnostic_with_debt_writer_emits_both_spaces() -> None:
    model = _model()
    train = _event(good=(0,), bad=(1,), legal=(0, 1), group="train")
    held = _event(
        good=(0,),
        bad=(1,),
        legal=(0, 1),
        group="held",
        split="held_out",
    )
    emitted: list[ConstraintDebtV1] = []

    diagnose_metric_complete_gradient_feasibility(
        model,
        [train, held],
        objective="ftpo_set",
        probability_space="full_vocab",
        debt_writer=emitted.append,
    )

    assert len(emitted) == 4  # 2 events x 2 probability spaces
    spaces = [row.probability_space for row in emitted]
    assert spaces == ["full_vocab", "legal_tokens", "full_vocab", "legal_tokens"]
    splits = [row.split for row in emitted]
    assert splits == ["train", "train", "held_out", "held_out"]
    for row in emitted:
        assert row.state_id
        assert row.decision_kind == "component"
        assert row.good_support_count == 1
        assert row.bad_support_count == 1
        assert row.legal_support_count == 2


def test_diagnostic_output_unchanged_with_debt_writer_none() -> None:
    model = _model()
    train = _event(good=(0,), bad=(1,), legal=(0, 1), group="train")
    held = _event(
        good=(0,),
        bad=(1,),
        legal=(0, 1),
        group="held",
        split="held_out",
    )

    without_writer = diagnose_metric_complete_gradient_feasibility(
        model, [train, held], objective="ftpo_set", probability_space="full_vocab"
    )
    with_writer = diagnose_metric_complete_gradient_feasibility(
        model,
        [train, held],
        objective="ftpo_set",
        probability_space="full_vocab",
        debt_writer=lambda _: None,
    )

    _approx_dict(without_writer, with_writer, rel=1e-9, abs=1e-12)


def test_guard_objective_tensors_matches_direct_masses() -> None:
    """Regression guard: the refactor must not change returned tensors."""
    logits = torch.tensor([1.0, 0.0, -1.0, 2.0], requires_grad=True)
    event = _event(good=(0,), bad=(1,), legal=(0, 1, 2))
    loss, _ = local_decision_loss(logits, event, objective="ftpo_set")
    values = _guard_objective_tensors(
        logits, event, objective="ftpo_set", probability_space="full_vocab"
    )

    probs = torch.softmax(logits.detach(), dim=-1)
    expected_good_mass = float(probs[event.good_token_ids].sum())
    expected_bad_mass = float(probs[event.bad_token_ids].sum())

    assert float(values["loss"]) == pytest.approx(float(loss))
    assert float(values["good_probability_mass"]) == pytest.approx(
        -expected_good_mass
    )
    assert float(values["bad_probability_mass"]) == pytest.approx(expected_bad_mass)
    assert float(values["mean_margin"]) == pytest.approx(-1.0)


def test_legal_space_guard_objective_tensors_matches_direct_masses() -> None:
    logits = torch.tensor([1.0, 0.0, -1.0, 2.0], requires_grad=True)
    event = _event(good=(0,), bad=(1,), legal=(0, 1, 2))
    values = _guard_objective_tensors(
        logits, event, objective="ftpo_set", probability_space="legal_tokens"
    )

    legal_ids = torch.tensor(event.legal_token_ids)
    legal_logits = logits.detach().index_select(0, legal_ids)
    legal_probs = torch.softmax(legal_logits, dim=-1)
    legal_index = {tid: i for i, tid in enumerate(event.legal_token_ids)}
    expected_good = float(legal_probs[legal_index[event.good_token_ids[0]]])
    expected_bad = float(legal_probs[legal_index[event.bad_token_ids[0]]])

    assert float(values["good_probability_mass"]) == pytest.approx(-expected_good)
    assert float(values["bad_probability_mass"]) == pytest.approx(expected_bad)


def test_debt_writer_does_not_alter_gradients() -> None:
    model = _model()
    train = _event(good=(0,), bad=(1,), legal=(0, 1), group="train")
    held = _event(
        good=(0,),
        bad=(1,),
        legal=(0, 1),
        group="held",
        split="held_out",
    )

    before = copy.deepcopy(model.state_dict())
    diagnose_metric_complete_gradient_feasibility(
        model,
        [train, held],
        objective="ftpo_set",
        debt_writer=lambda _: None,
    )
    after = model.state_dict()

    for key in before:
        assert torch.equal(before[key], after[key])
