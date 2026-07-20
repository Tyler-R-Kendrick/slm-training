from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.preference.constraint_debt import (
    ConstraintDebtV1,
    compute_constraint_debt_v1,
)
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    split_for_group,
)


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


def _fake_event(**kwargs):
    """Build a minimal event-like object for edge cases V1 cannot represent."""
    defaults = {
        "event_id": "fake",
        "group_id": "group",
        "trajectory_id": "trace",
        "policy_checkpoint_sha": "policy",
        "tokenizer_sha": "tokenizer",
        "decode_config_hash": "decode",
        "decision_kind": "component",
        "split": "train",
        "good_token_ids": (0,),
        "bad_token_ids": (1,),
        "legal_token_ids": (0, 1),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_uniform_full_vocab_masses() -> None:
    logits = torch.zeros(3)
    event = _event(good=(0,), bad=(1,), legal=(0, 1))
    debt = compute_constraint_debt_v1(logits, event, probability_space="full_vocab")

    assert debt.legal_mass == pytest.approx(2 / 3)
    assert debt.good_mass == pytest.approx(1 / 3)
    assert debt.bad_mass == pytest.approx(1 / 3)
    assert debt.ambiguous_mass == pytest.approx(0.0)
    assert debt.unobserved_mass == pytest.approx(1 / 3)
    assert debt.legal_mass_deficit == pytest.approx(1 / 3)
    assert debt.pre_post_mask_kl == pytest.approx(0.0, abs=1e-6)
    assert debt.legal_debt == pytest.approx(-math.log(2 / 3 + 1e-12))
    assert debt.good_debt == pytest.approx(-math.log(1 / 3 + 1e-12))
    assert debt.bad_debt == pytest.approx(-math.log(1 / 3 + 1e-12))
    assert debt.single_legal_action is False
    assert debt.empty_good_partition is False
    assert debt.empty_bad_partition is False


def test_uniform_legal_renormalized_masses() -> None:
    logits = torch.zeros(3)
    event = _event(good=(0,), bad=(1,), legal=(0, 1))
    debt = compute_constraint_debt_v1(logits, event, probability_space="legal_tokens")

    assert debt.legal_mass == pytest.approx(1.0)
    assert debt.good_mass == pytest.approx(0.5)
    assert debt.bad_mass == pytest.approx(0.5)
    assert debt.ambiguous_mass == pytest.approx(0.0)
    assert debt.unobserved_mass == pytest.approx(0.0)
    assert debt.legal_mass_deficit == pytest.approx(0.0)
    # legal_probs = 1/2 over legal; full_probs[legal] = 1/3 -> KL = log(3/2)
    assert debt.pre_post_mask_kl == pytest.approx(math.log(1.5), abs=1e-6)
    assert debt.legal_debt == pytest.approx(-math.log(1.0 + 1e-12))
    assert debt.good_debt == pytest.approx(-math.log(0.5 + 1e-12))
    assert debt.bad_debt == pytest.approx(-math.log(0.5 + 1e-12))


def test_numerical_stability_at_extreme_masses() -> None:
    # Good token gets almost all mass; bad token gets almost none.
    logits = torch.tensor([0.0, -30.0, -30.0])
    event = _event(good=(0,), bad=(1,), legal=(0, 1))
    debt = compute_constraint_debt_v1(logits, event, probability_space="full_vocab")

    assert debt.good_mass > 0.999
    assert debt.bad_mass < 1e-12
    assert debt.good_debt == pytest.approx(0.0, abs=1e-9)
    assert debt.bad_debt > 0.0
    assert math.isfinite(debt.bad_debt)


def test_empty_good_partition() -> None:
    logits = torch.zeros(2)
    event = _fake_event(
        good_token_ids=(),
        bad_token_ids=(1,),
        legal_token_ids=(0, 1),
    )
    debt = compute_constraint_debt_v1(logits, event, probability_space="legal_tokens")

    assert debt.good_mass == pytest.approx(0.0)
    assert debt.good_debt is None
    assert debt.empty_good_partition is True
    assert debt.empty_bad_partition is False


def test_empty_bad_partition() -> None:
    logits = torch.zeros(2)
    event = _fake_event(
        good_token_ids=(0,),
        bad_token_ids=(),
        legal_token_ids=(0, 1),
    )
    debt = compute_constraint_debt_v1(logits, event, probability_space="legal_tokens")

    assert debt.bad_mass == pytest.approx(0.0)
    assert debt.bad_debt is None
    assert debt.empty_good_partition is False
    assert debt.empty_bad_partition is True


def test_single_legal_action_flag() -> None:
    logits = torch.zeros(1)
    event = _fake_event(
        good_token_ids=(0,),
        bad_token_ids=(),
        legal_token_ids=(0,),
    )
    debt = compute_constraint_debt_v1(logits, event, probability_space="legal_tokens")

    assert debt.single_legal_action is True
    assert debt.legal_mass == pytest.approx(1.0)
    assert debt.good_mass == pytest.approx(1.0)
    assert debt.good_debt == pytest.approx(-math.log(1.0 + 1e-12))


def test_legal_space_kl_zero_when_full_distribution_uniform_over_legal() -> None:
    # All probability mass is uniform over the two legal tokens.
    logits = torch.tensor([0.0, 0.0, -1e9])
    event = _event(good=(0,), bad=(1,), legal=(0, 1))
    debt = compute_constraint_debt_v1(logits, event, probability_space="legal_tokens")

    assert debt.pre_post_mask_kl == pytest.approx(0.0, abs=1e-6)
    assert debt.unobserved_mass == pytest.approx(0.0, abs=1e-6)


def test_full_vocab_vs_legal_invariants() -> None:
    logits = torch.tensor([2.0, -1.0, 0.0, 3.0])
    event = _event(good=(0,), bad=(1,), legal=(0, 1, 2))
    full = compute_constraint_debt_v1(logits, event, probability_space="full_vocab")
    legal = compute_constraint_debt_v1(logits, event, probability_space="legal_tokens")

    assert full.legal_mass < 1.0
    assert full.legal_mass_deficit == pytest.approx(1.0 - full.legal_mass)
    assert full.unobserved_mass == full.legal_mass_deficit

    assert legal.legal_mass == pytest.approx(1.0)
    assert legal.legal_mass_deficit == pytest.approx(0.0)
    assert legal.unobserved_mass == pytest.approx(0.0)


def test_to_dict_from_dict_round_trip() -> None:
    logits = torch.zeros(3)
    event = _event(good=(0,), bad=(1,), legal=(0, 1))
    debt = compute_constraint_debt_v1(logits, event, probability_space="full_vocab")
    data = debt.to_dict()
    restored = ConstraintDebtV1.from_dict(data)

    assert restored == debt


def test_from_dict_rejects_unknown_fields() -> None:
    logits = torch.zeros(3)
    event = _event(good=(0,), bad=(1,), legal=(0, 1))
    data = compute_constraint_debt_v1(
        logits, event, probability_space="full_vocab"
    ).to_dict()
    data["unknown_field"] = "x"

    with pytest.raises(ValueError, match="unknown constraint debt fields"):
        ConstraintDebtV1.from_dict(data)


def test_probability_space_validation() -> None:
    event = _event()
    with pytest.raises(ValueError, match="unknown probability space"):
        compute_constraint_debt_v1(torch.zeros(3), event, probability_space="invalid")  # type: ignore[arg-type]
