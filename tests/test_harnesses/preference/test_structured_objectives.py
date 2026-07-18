"""Tests for the LDI3-01 structured local-preference objectives (SLM-128).

Pin the objective math against hand-computed toy examples, gradient correctness
(finite-difference gradcheck), the barrier confidence gate + erosion metric, the
state-normalized ratio control, and fail-closed config validation. No model
update is exercised.
"""

from __future__ import annotations

import math

import pytest
import torch

from slm_training.harnesses.preference.decision_events_v2 import ObjectiveView
from slm_training.harnesses.preference.structured_objectives import (
    StructuredObjectiveConfig,
    StructuredObjectiveError,
    structured_decision_loss,
)


def _view(good, bad, ambiguous=(), unobserved=(), weights=None) -> ObjectiveView:
    partitioned = set(good) | set(bad) | set(ambiguous) | set(unobserved)
    weights = weights or tuple((int(a), 1.0) for a in partitioned)
    return ObjectiveView(
        tuple(good), tuple(bad), tuple(ambiguous), tuple(unobserved), weights, "toy", "hash"
    )


def test_legal_set_ftpo_pairwise_matches_hand_value() -> None:
    logits = torch.tensor([1.0, 0.0, -5.0])  # action 0 good, 1 bad, 2 filler
    view = _view(good=(0,), bad=(1,))
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="pairwise", epsilon=2.0, tau=1.0)
    loss, m = structured_decision_loss(logits, view, legal_action_ids=(0, 1, 2), config=cfg)
    # delta=1; hinge=clamp((2-1)/2)=0.5; softplus(1)=0.31326*... -> 0.5*softplus(1)
    expected = 0.5 * math.log1p(math.e)
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    assert m["active_pair_fraction"] == pytest.approx(1.0)


def test_legal_mass_margin_sums_over_legal_only() -> None:
    logits = torch.tensor([1.0, 0.0, 10.0])  # action 2 huge but NOT legal
    view = _view(good=(0,), bad=(1,))
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass", temperature=1.0)
    loss, m = structured_decision_loss(logits, view, legal_action_ids=(0, 1), config=cfg)
    pg = math.e / (math.e + 1)
    pb = 1 / (math.e + 1)
    expected = math.log1p(math.exp(math.log(pb) - math.log(pg)))
    assert loss.item() == pytest.approx(expected, rel=1e-4)
    # legal mass ignores the huge illegal action 2.
    assert m["good_legal_mass"] == pytest.approx(pg, rel=1e-4)
    assert m["good_legal_mass"] + m["bad_legal_mass"] == pytest.approx(1.0, rel=1e-5)


def test_set_size_normalization_bounds_large_sets() -> None:
    # Two matched states, one with many more pairs; per-state mean keeps them comparable.
    logits = torch.tensor([0.5, 0.5, 0.5, -0.5, -0.5, -0.5])
    small = _view(good=(0,), bad=(3,))
    big = _view(good=(0, 1, 2), bad=(3, 4, 5))
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="pairwise")
    legal = (0, 1, 2, 3, 4, 5)
    ls, _ = structured_decision_loss(logits, small, legal_action_ids=legal, config=cfg)
    lb, _ = structured_decision_loss(logits, big, legal_action_ids=legal, config=cfg)
    assert ls.item() == pytest.approx(lb.item(), rel=1e-6)  # mean, not sum


@pytest.mark.parametrize(
    "cfg",
    [
        StructuredObjectiveConfig(name="legal_set_ftpo", variant="pairwise"),
        StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass"),
        StructuredObjectiveConfig(name="tab_barrier", barrier_p=0.5),
        StructuredObjectiveConfig(name="tbpo_inspired"),
    ],
)
def test_gradcheck_each_objective(cfg: StructuredObjectiveConfig) -> None:
    torch.manual_seed(0)
    view = _view(good=(0, 1), bad=(2, 3))
    legal = (0, 1, 2, 3, 4)
    ref = torch.randn(5, dtype=torch.double)

    def fn(logits: torch.Tensor) -> torch.Tensor:
        loss, _ = structured_decision_loss(
            logits, view, legal_action_ids=legal, config=cfg, reference_logits=ref
        )
        return loss

    logits = torch.randn(5, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (logits,), eps=1e-6, atol=1e-4)


def test_numeric_stability_tiny_masses() -> None:
    logits = torch.tensor([-40.0, -40.0, 40.0, -40.0], requires_grad=True)  # tiny good mass
    view = _view(good=(0,), bad=(1,))
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass")
    loss, _ = structured_decision_loss(logits, view, legal_action_ids=(0, 1, 2, 3), config=cfg)
    assert torch.isfinite(loss)
    loss.backward()  # gradients defined despite tiny masses
    assert torch.isfinite(logits.grad).all()


def test_barrier_activates_only_for_underconfident_critical_good() -> None:
    # good action 0 is under-confident (low logit); barrier should lift it.
    logits = torch.tensor([-2.0, 3.0, 0.0])
    view = _view(good=(0,), bad=(1,))
    cfg = StructuredObjectiveConfig(name="tab_barrier", barrier_p=0.5, barrier_strength=1.0)
    mask = torch.tensor([1.0])  # action 0 is critical
    loss, m = structured_decision_loss(
        logits, view, legal_action_ids=(0, 1, 2), config=cfg, critical_good_mask=mask
    )
    assert m["barrier_active_fraction"] == pytest.approx(1.0)
    assert m["barrier_loss"] > 0.0
    # A confident good action gets no barrier.
    conf = torch.tensor([5.0, -3.0, 0.0])
    _, m2 = structured_decision_loss(
        conf, view, legal_action_ids=(0, 1, 2), config=cfg, critical_good_mask=mask
    )
    assert m2["barrier_active_fraction"] == pytest.approx(0.0)
    assert m2["barrier_loss"] == pytest.approx(0.0)


def test_barrier_reports_erosion_vs_reference() -> None:
    logits = torch.tensor([-1.0, 2.0, 0.0])
    ref = torch.tensor([2.0, -1.0, 0.0])  # good action 0 was far more likely in the reference
    view = _view(good=(0,), bad=(1,))
    cfg = StructuredObjectiveConfig(name="tab_barrier", barrier_p=0.5)
    _, m = structured_decision_loss(
        logits, view, legal_action_ids=(0, 1, 2), config=cfg, reference_logits=ref
    )
    assert m["erosion_rate"] == pytest.approx(1.0)  # good prob dropped vs reference


def test_tbpo_requires_reference_and_enough_support() -> None:
    view = _view(good=(0,), bad=(1,))
    cfg = StructuredObjectiveConfig(name="tbpo_inspired")
    logits = torch.tensor([1.0, 0.0])
    with pytest.raises(StructuredObjectiveError):
        structured_decision_loss(logits, view, legal_action_ids=(0, 1), config=cfg)  # no ref
    # Single-action legal set -> disabled.
    with pytest.raises(StructuredObjectiveError):
        structured_decision_loss(
            torch.tensor([1.0]), _view(good=(0,), bad=()), legal_action_ids=(0,),
            config=cfg, reference_logits=torch.tensor([0.0]),
        )


def test_config_fails_closed_on_unknown_field_and_roundtrips() -> None:
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass", epsilon=1.5)
    again = StructuredObjectiveConfig.from_mapping({"name": "legal_set_ftpo", "variant": "mass", "epsilon": 1.5})
    assert cfg.fingerprint() == again.fingerprint()
    with pytest.raises(StructuredObjectiveError):
        StructuredObjectiveConfig.from_mapping({"name": "legal_set_ftpo", "bogus": 1})
    with pytest.raises(StructuredObjectiveError):
        StructuredObjectiveConfig(name="legal_set_ftpo", variant="nope")
    with pytest.raises(StructuredObjectiveError):
        StructuredObjectiveConfig(name="tab_barrier", barrier_p=1.5)


def test_architecture_neutral_same_call_two_logit_sources() -> None:
    # The same objective serves "causal" and "twotower" logits — both are 1-D.
    view = _view(good=(0,), bad=(1,))
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="pairwise")
    causal_logits = torch.tensor([1.0, 0.0, 0.0])
    twotower_logits = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    lc, _ = structured_decision_loss(causal_logits, view, legal_action_ids=(0, 1, 2), config=cfg)
    lt, _ = structured_decision_loss(twotower_logits, view, legal_action_ids=(0, 1, 2, 3, 4), config=cfg)
    assert lc.item() == pytest.approx(lt.item(), rel=1e-6)


def test_refuses_nontrainable_view_and_out_of_legal_actions() -> None:
    shadow = ObjectiveView((0,), (1,), (), (), ((0, 1.0),), "m", "h", trainable=False)
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="pairwise")
    with pytest.raises(StructuredObjectiveError):
        structured_decision_loss(torch.tensor([1.0, 0.0]), shadow, legal_action_ids=(0, 1), config=cfg)
    good_view = _view(good=(0,), bad=(9,))  # 9 not legal
    with pytest.raises(StructuredObjectiveError):
        structured_decision_loss(torch.tensor([1.0, 0.0]), good_view, legal_action_ids=(0, 1), config=cfg)


def test_ambiguous_unobserved_mass_reported_separately_and_validated() -> None:
    logits = torch.tensor([1.0, 0.0, 0.5, -0.5])
    view = _view(good=(0,), bad=(1,), ambiguous=(2,), unobserved=(3,))
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass")
    _, m = structured_decision_loss(logits, view, legal_action_ids=(0, 1, 2, 3), config=cfg)
    # The four legal partitions cover the legal set -> masses sum to 1, each reported.
    total = (
        m["good_legal_mass"] + m["bad_legal_mass"]
        + m["ambiguous_legal_mass"] + m["unobserved_legal_mass"]
    )
    assert total == pytest.approx(1.0, rel=1e-5)
    assert m["ambiguous_legal_mass"] > 0.0 and m["unobserved_legal_mass"] > 0.0
    assert m["num_ambiguous"] == 1.0 and m["num_unobserved"] == 1.0
    # Ambiguous/unobserved ids must be legal (validated here, never silently mislabeled).
    with pytest.raises(StructuredObjectiveError, match="inside the legal set"):
        structured_decision_loss(
            logits, _view(good=(0,), bad=(1,), ambiguous=(9,)),
            legal_action_ids=(0, 1, 2, 3), config=cfg,
        )


def test_locality_tether_composes_and_is_separately_metered() -> None:
    logits = torch.tensor([1.0, 0.0, 1.0, 0.0])  # off-target token 2 drifts from reference
    ref = torch.zeros(4)
    view = _view(good=(0,), bad=(1,))
    plain = StructuredObjectiveConfig(name="legal_set_ftpo", variant="pairwise")
    tethered = StructuredObjectiveConfig(
        name="legal_set_ftpo", variant="pairwise", non_target_tether=1.0
    )
    base, _ = structured_decision_loss(logits, view, legal_action_ids=(0, 1, 2, 3), config=plain)
    loss, m = structured_decision_loss(
        logits, view, legal_action_ids=(0, 1, 2, 3), config=tethered, reference_logits=ref
    )
    assert m["non_target_logit_mse"] == pytest.approx(0.5)  # (1^2 + 0^2) / 2 off-target
    assert loss.item() == pytest.approx(base.item() + 0.5, rel=1e-5)  # separately metered
    with pytest.raises(StructuredObjectiveError, match="reference"):
        structured_decision_loss(logits, view, legal_action_ids=(0, 1, 2, 3), config=tethered)


def test_barrier_role_weight_down_weights_structural_tokens() -> None:
    logits = torch.tensor([-2.0, 3.0, 0.0])  # good action 0 under-confident + critical
    view = _view(good=(0,), bad=(1,))
    cfg = StructuredObjectiveConfig(name="tab_barrier", barrier_p=0.5)
    mask = torch.tensor([1.0])
    _, full = structured_decision_loss(
        logits, view, legal_action_ids=(0, 1, 2), config=cfg, critical_good_mask=mask
    )
    _, low = structured_decision_loss(
        logits, view, legal_action_ids=(0, 1, 2), config=cfg,
        critical_good_mask=mask, good_role_weights=torch.tensor([0.1]),
    )
    assert low["barrier_loss"] < full["barrier_loss"]  # structural down-weight shrinks anchor
    assert low["mean_role_weight"] == pytest.approx(0.1)
    # default_role_weight is now honored (previously a dead config field).
    zero_cfg = StructuredObjectiveConfig(name="tab_barrier", barrier_p=0.5, default_role_weight=0.0)
    _, zeroed = structured_decision_loss(
        logits, view, legal_action_ids=(0, 1, 2), config=zero_cfg, critical_good_mask=mask
    )
    assert zeroed["barrier_loss"] == pytest.approx(0.0)
