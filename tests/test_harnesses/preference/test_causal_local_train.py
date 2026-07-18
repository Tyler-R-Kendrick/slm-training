"""Toy-logit unit tests for the LDI1-02 causal exact-state objectives."""

from __future__ import annotations

import pytest
import torch

from slm_training.harnesses.preference.causal_local_train import causal_decision_loss
from slm_training.harnesses.preference.decision_events_v2 import ObjectiveView


def _view(
    good: tuple[int, ...],
    bad: tuple[int, ...] = (),
    *,
    ambiguous: tuple[int, ...] = (),
    unobserved: tuple[int, ...] = (),
    weights: tuple[tuple[int, float], ...] | None = None,
    trainable: bool = True,
) -> ObjectiveView:
    return ObjectiveView(
        good_action_ids=good,
        bad_action_ids=bad,
        ambiguous_action_ids=ambiguous,
        unobserved_action_ids=unobserved,
        weights=weights if weights is not None else tuple((a, 1.0) for a in good),
        materializer_id="toy",
        materializer_config_hash="toy",
        trainable=trainable,
    )


def test_ftpo_single_gradient_moves_good_above_bad() -> None:
    logits = torch.tensor([0.0, -1.0, 1.0, 0.0], requires_grad=True)
    loss, metrics = causal_decision_loss(
        logits, _view((1,), (2,)), legal_action_ids=(0, 1, 2, 3), objective="ftpo_single"
    )
    loss.backward()
    # Descent must raise the good logit (neg grad) and lower the bad logit (pos grad).
    assert logits.grad[1] < 0 < logits.grad[2]
    assert metrics["mean_margin"] == pytest.approx(-2.0)


def test_ftpo_set_supports_multiple_good_bad_with_weights() -> None:
    logits = torch.tensor([0.5, 0.2, -0.5, -1.0], requires_grad=True)
    view = _view((0, 1), (2, 3), weights=((0, 2.0), (1, 0.5)))
    loss, metrics = causal_decision_loss(
        logits, view, legal_action_ids=(0, 1, 2, 3), objective="ftpo_set"
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert 0.0 <= metrics["active_weight"] <= 2.0


def test_legal_set_mass_normalizes_over_legal_actions_only() -> None:
    # An illegal token's logit must not change a legal-space objective.
    base = [0.0, 1.0, -1.0]
    view = _view((1,), (2,))
    loss_low, _ = causal_decision_loss(
        torch.tensor(base + [0.0]),
        view,
        legal_action_ids=(0, 1, 2),
        objective="legal_set_mass",
    )
    loss_high, _ = causal_decision_loss(
        torch.tensor(base + [100.0]),
        view,
        legal_action_ids=(0, 1, 2),
        objective="legal_set_mass",
    )
    assert loss_low.item() == pytest.approx(loss_high.item())


def test_legal_set_mass_rewards_good_and_penalizes_bad() -> None:
    view = _view((1,), (2,))
    weak, _ = causal_decision_loss(
        torch.tensor([0.0, 0.0, 0.0]), view, legal_action_ids=(0, 1, 2), objective="legal_set_mass"
    )
    strong, m = causal_decision_loss(
        torch.tensor([0.0, 3.0, -3.0]), view, legal_action_ids=(0, 1, 2), objective="legal_set_mass"
    )
    assert strong.item() < weak.item()
    assert m["good_legal_mass"] > m["bad_legal_mass"]


def test_unlikelihood_gradient_reduces_bad_mass() -> None:
    logits = torch.tensor([0.0, 0.0, 2.0, 0.0], requires_grad=True)
    loss, _ = causal_decision_loss(
        logits, _view((1,), (2,)), legal_action_ids=(0, 1, 2, 3), objective="unlikelihood"
    )
    loss.backward()
    # Reducing the loss must push the bad logit down.
    assert logits.grad[2] > 0


def test_reference_tether_excludes_target_tokens_within_grace() -> None:
    logits = torch.tensor([0.0, 3.0, -3.0, 0.5])
    reference = torch.zeros(4)
    _, metrics = causal_decision_loss(
        logits,
        _view((1,), (2,)),
        legal_action_ids=(0, 1, 2, 3),
        objective="ftpo_single",
        reference_logits=reference,
        non_target_tether=1.0,
        target_tether=1.0,
        target_grace=5.0,
    )
    # Targets {1,2} drift 3.0 < grace 5.0 => no target penalty; non-targets {0,3} penalized.
    assert metrics["target_excess_logit_mse"] == pytest.approx(0.0)
    assert metrics["non_target_logit_mse"] > 0.0


def test_non_target_tether_penalizes_drift_off_the_decision() -> None:
    logits = torch.tensor([0.0, 1.0, -1.0, 4.0], requires_grad=True)
    reference = torch.zeros(4)
    loss, _ = causal_decision_loss(
        logits,
        _view((1,), (2,)),
        legal_action_ids=(0, 1, 2, 3),
        objective="ftpo_single",
        reference_logits=reference,
        non_target_tether=1.0,
    )
    loss.backward()
    # The drifted non-target logit (index 3) is pulled back toward the reference.
    assert logits.grad[3] > 0


def test_constraint_shadow_view_is_refused() -> None:
    view = _view((), (), ambiguous=(1, 2), trainable=False)
    with pytest.raises(ValueError, match="non-trainable objective view"):
        causal_decision_loss(
            torch.zeros(3), view, legal_action_ids=(0, 1, 2), objective="legal_set_mass"
        )


def test_actions_outside_legal_set_are_rejected() -> None:
    with pytest.raises(ValueError, match="inside the legal set"):
        causal_decision_loss(
            torch.zeros(4), _view((1,), (5,)), legal_action_ids=(0, 1, 2), objective="ftpo_single"
        )


def test_ftpo_single_requires_exactly_one_good_and_one_bad() -> None:
    with pytest.raises(ValueError, match="exactly one good and one bad"):
        causal_decision_loss(
            torch.zeros(4),
            _view((0, 1), (2,)),
            legal_action_ids=(0, 1, 2, 3),
            objective="ftpo_single",
        )


def test_non_one_dimensional_logits_are_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        causal_decision_loss(
            torch.zeros(2, 3),
            _view((1,), (2,)),
            legal_action_ids=(0, 1, 2),
            objective="legal_set_mass",
        )
