"""Tests for OpenUI-native structured local-preference objectives (LDI3-01 / SLM-128)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.preference.structured_objectives import (  # noqa: E402
    StateBaseline,
    StructuredObjectiveConfig,
    StructuredObjectiveError,
    StructuredObjectiveInput,
    legal_probability_masses,
    structured_objective_batch_loss,
    structured_objective_loss,
    structured_objective_report,
    token_erosion_rate,
)


def _inp(logits, **overrides) -> StructuredObjectiveInput:
    base = dict(
        logits=torch.tensor(logits, dtype=torch.float64),
        legal_ids=(0, 1),
        good_ids=(0,),
        bad_ids=(1,),
    )
    base.update(overrides)
    return StructuredObjectiveInput(**base)  # type: ignore[arg-type]


# --- Objective A: Legal-Set FTPO ------------------------------------------------------


def test_mass_margin_matches_hand_computed_value() -> None:
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass_margin")
    inp = _inp([0.0, 0.0, 0.0], legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,))
    loss, report = structured_objective_loss(inp, cfg)
    # p_legal uniform -> P_G == P_B -> softplus(log 1) == ln 2.
    assert abs(float(loss) - math.log(2)) < 1e-6
    assert abs(report["good_mass"] - 1 / 3) < 1e-9


def test_pairwise_margin_matches_hand_computed_value() -> None:
    cfg = StructuredObjectiveConfig(
        name="legal_set_ftpo", variant="pairwise_margin", epsilon=2.0, tau=1.0
    )
    inp = _inp([1.0, 1.0])  # delta == 0
    loss, _ = structured_objective_loss(inp, cfg)
    # single pair: weighted_mean == softplus((eps - 0)/tau) == softplus(2).
    assert abs(float(loss) - math.log1p(math.e**2)) < 1e-6


def test_state_normalization_prevents_large_set_dominance() -> None:
    cfg_norm = StructuredObjectiveConfig(
        name="legal_set_ftpo", variant="pairwise_margin", state_normalized=True
    )
    cfg_raw = StructuredObjectiveConfig(
        name="legal_set_ftpo", variant="pairwise_margin", state_normalized=False
    )
    small = _inp([2.0, 0.0])  # 1x1, satisfied margin -> small loss
    big_logits = [0.0] * 12
    big = _inp(
        big_logits, legal_ids=tuple(range(12)), good_ids=(0, 1, 2), bad_ids=(3, 4, 5)
    )  # 3x3, delta 0 -> large loss
    la = float(structured_objective_loss(small, cfg_norm)[0])
    lb = float(structured_objective_loss(big, cfg_norm)[0])
    norm_batch = float(structured_objective_batch_loss([small, big], cfg_norm)[0])
    raw_batch = float(structured_objective_batch_loss([small, big], cfg_raw)[0])
    # Normalized batch is the plain mean of state losses; raw batch is pulled toward the
    # 9-pair state, so it sits strictly above the mean.
    assert abs(norm_batch - (la + lb) / 2) < 1e-9
    assert raw_batch > norm_batch


def test_legal_mass_sums_over_only_the_legal_set() -> None:
    logits = [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]  # index 5 is out of legal
    inp = _inp(
        logits, legal_ids=(0, 1, 2, 3), good_ids=(0,), bad_ids=(1,),
        ambiguous_ids=(2,), unobserved_ids=(3,),
    )
    masses = legal_probability_masses(inp)
    total = sum(float(masses[k]) for k in ("good", "bad", "ambiguous", "unobserved"))
    assert abs(total - 1.0) < 1e-9  # the huge out-of-legal logit contributes nothing


# --- Objective B: TAB-PO-inspired barrier ---------------------------------------------


def test_barrier_activates_only_for_critical_underconfident_good() -> None:
    cfg = StructuredObjectiveConfig(name="tab_po_inspired_barrier", barrier_p=0.1)
    logits = [3.0, -1.0, -1.0, 0.0, 0.0]
    inp = _inp(
        logits, legal_ids=(0, 1, 2, 3, 4), good_ids=(0, 1, 2), bad_ids=(3,),
        good_critical=(True, True, False),  # 0 confident, 1 critical+underconfident, 2 not critical
    )
    _loss, report = structured_objective_loss(inp, cfg)
    assert report["barrier_active_count"] == 1
    assert 1 in report["anchored_probabilities"]  # only token 1 anchored
    assert 0 not in report["anchored_probabilities"]


def test_structural_role_gets_default_low_weight() -> None:
    cfg = StructuredObjectiveConfig(
        name="tab_po_inspired_barrier", barrier_p=0.5, structural_role_weight=0.0
    )
    inp = _inp(
        [-3.0, 0.0, 0.0], legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,),
        good_roles=("punctuation",), good_critical=(True,),
    )
    _loss, report = structured_objective_loss(inp, cfg)
    assert report["barrier_active_count"] == 0  # structural role -> zero weight -> skipped


def test_token_erosion_rate_detects_dropped_good_probabilities() -> None:
    result = token_erosion_rate({0: 0.5, 1: 0.3}, {0: 0.4, 1: 0.35}, preference_improved=True)
    assert result["eroded_fraction"] == 0.5
    assert result["eroded_tokens"] == [0]
    assert result["preference_improved"] is True


def test_ambiguous_and_unobserved_are_normalized_but_never_targets() -> None:
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass_margin")
    inp = _inp(
        [0.0, 0.0, 5.0, 0.0], legal_ids=(0, 1, 2, 3), good_ids=(0,), bad_ids=(1,),
        ambiguous_ids=(2,), unobserved_ids=(3,),
    )
    _loss, report = structured_objective_loss(inp, cfg)
    # Ambiguous mass is part of legal normalization (nonzero) but not a good/bad target.
    assert report["ambiguous_mass"] > 0.0
    assert report["good_mass"] > 0.0 and report["bad_mass"] > 0.0


# --- Objective C: TBPO-inspired ratio control -----------------------------------------


def test_ratio_control_compares_at_state_and_disables_without_reference() -> None:
    cfg = StructuredObjectiveConfig(name="tbpo_inspired_ratio", baseline_type="advantage")
    ref = torch.zeros(3, dtype=torch.float64)
    active = _inp(
        [1.0, 0.0, 0.0], legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,), reference_logits=ref
    )
    _loss, report = structured_objective_loss(active, cfg)
    assert report["control_active"] is True

    no_ref = _inp([1.0, 0.0, 0.0], legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,))
    _loss2, report2 = structured_objective_loss(no_ref, cfg)
    assert report2["control_active"] is False  # inadequate support -> disabled


def test_state_baseline_fits_from_train_and_round_trips() -> None:
    ref = torch.zeros(2, dtype=torch.float64)
    train = [_inp([2.0, 0.0], reference_logits=ref), _inp([0.0, 2.0], reference_logits=ref)]
    baseline = StateBaseline.fit(train)
    assert baseline.fitted
    restored = StateBaseline.from_dict(baseline.to_dict())
    assert abs(restored.value().item() - baseline.value().item()) < 1e-12
    # A no-reference corpus yields an unfitted default baseline.
    assert not StateBaseline.fit([_inp([1.0, 0.0])]).fitted


# --- Gradients / numerics -------------------------------------------------------------


def test_mass_margin_gradient_matches_finite_difference() -> None:
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass_margin")
    logits = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    inp = StructuredObjectiveInput(
        logits=logits, legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,)
    )
    loss, _ = structured_objective_loss(inp, cfg)
    (grad,) = torch.autograd.grad(loss, logits)
    eps = 1e-6

    def _loss_at(vec: torch.Tensor) -> float:
        alt = StructuredObjectiveInput(
            logits=vec, legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,)
        )
        return float(structured_objective_loss(alt, cfg)[0])

    with torch.no_grad():
        plus = logits.clone()
        plus[0] += eps
        minus = logits.clone()
        minus[0] -= eps
        finite = (_loss_at(plus) - _loss_at(minus)) / (2 * eps)
    assert abs(finite - float(grad[0])) < 1e-6


def test_tiny_probability_is_numerically_stable() -> None:
    cfg = StructuredObjectiveConfig(name="tab_po_inspired_barrier", barrier_p=0.5)
    logits = torch.tensor([-1e4, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    inp = StructuredObjectiveInput(
        logits=logits, legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,),
        good_critical=(True,),
    )
    loss, report = structured_objective_loss(inp, cfg)
    assert math.isfinite(float(loss))
    (grad,) = torch.autograd.grad(loss, logits)
    assert torch.isfinite(grad).all()


# --- Config + architecture neutrality -------------------------------------------------


def test_config_round_trips_and_fingerprints_deterministically() -> None:
    cfg = StructuredObjectiveConfig(
        name="tab_po_inspired_barrier", role_weights=(("title", 2.0),), barrier_p=0.2
    )
    assert StructuredObjectiveConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.fingerprint() == StructuredObjectiveConfig.from_dict(cfg.to_dict()).fingerprint()
    other = StructuredObjectiveConfig(name="tab_po_inspired_barrier", barrier_p=0.3)
    assert cfg.fingerprint() != other.fingerprint()
    with pytest.raises(StructuredObjectiveError, match="unknown structured objective config"):
        StructuredObjectiveConfig.from_dict({**cfg.to_dict(), "mystery": 1})
    with pytest.raises(StructuredObjectiveError, match="barrier_p"):
        StructuredObjectiveConfig(name="tab_po_inspired_barrier", barrier_p=1.5)


def test_same_implementation_is_architecture_neutral() -> None:
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass_margin")
    values = [0.4, -0.3, 0.2]
    # "TwoTower" and "causal" trainers both extract this exact logit row -> same call.
    twotower = _inp(values, legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,))
    causal = _inp(values, legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,))
    assert abs(
        float(structured_objective_loss(twotower, cfg)[0])
        - float(structured_objective_loss(causal, cfg)[0])
    ) < 1e-12


def test_report_is_no_update_and_carries_fingerprint() -> None:
    cfg = StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass_margin")
    corpus = [
        _inp([0.4, -0.3, 0.1], legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,)),
        _inp([0.0, 0.5, -0.2], legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(1,)),
    ]
    report = structured_objective_report(corpus, cfg)
    assert report["kind"] == "structured_objective_report"
    assert report["config_fingerprint"] == cfg.fingerprint()
    assert "Adapted" in report["adaptation"]
    assert len(report["per_state"]) == 2


def test_input_rejects_illegal_and_empty_action_sets() -> None:
    with pytest.raises(StructuredObjectiveError, match="non-empty"):
        _inp([0.0, 0.0], good_ids=())
    with pytest.raises(StructuredObjectiveError, match="subset of the legal set"):
        _inp([0.0, 0.0, 0.0], legal_ids=(0, 1), good_ids=(0,), bad_ids=(2,))
    with pytest.raises(StructuredObjectiveError, match="disjoint"):
        _inp([0.0, 0.0, 0.0], legal_ids=(0, 1, 2), good_ids=(0,), bad_ids=(0,))
