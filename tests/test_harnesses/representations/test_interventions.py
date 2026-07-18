"""LDI4-02 causal-intervention primitives + the matched S0-S7 fixture matrix."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.representations.interventions import (  # noqa: E402
    ArmEffect,
    ablate_feature,
    apply_direction,
    classify_arm,
    diffmean_direction,
    run_fixture_matrix,
)
from slm_training.harnesses.representations.sae import SparseAutoencoder  # noqa: E402
from slm_training.harnesses.representations.spec import SAEArm, SAEConfig  # noqa: E402


def test_apply_direction_moves_along_unit_vector():
    h = torch.zeros(3, 4)
    d = torch.tensor([2.0, 0.0, 0.0, 0.0])
    out = apply_direction(h, d, 1.0)
    assert torch.allclose(out[:, 0], torch.ones(3))  # normalized then dosed


def test_ablate_feature_removes_that_feature_contribution():
    sae = SparseAutoencoder(SAEConfig(d_in=8, expansion_factor=4, seed=0))
    h = torch.randn(5, 8)
    z = sae.encode(h)
    active = int(torch.argmax(z.sum(dim=0)))
    ablated = ablate_feature(sae, h, active)
    full = sae(h)[0]
    # ablating the most-active feature changes the reconstruction.
    assert not torch.allclose(ablated, full)


def test_diffmean_direction_points_at_the_separating_axis():
    pos = torch.randn(50, 4) + torch.tensor([3.0, 0.0, 0.0, 0.0])
    neg = torch.randn(50, 4)
    d = diffmean_direction(pos, neg)
    assert int(torch.argmax(d.abs())) == 0


def test_fixture_matrix_has_all_arms_localized_baselines_and_is_wiring_only():
    report = run_fixture_matrix(seed=0)
    assert report["status"] == "wiring_only"
    arms = {a["arm_id"]: a for a in report["arms"]}
    assert set(arms) == {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"}
    # the parent control does nothing; the DiffMean baseline moves the target.
    assert arms["S0"]["target_effect"] == 0.0
    assert arms["S2"]["target_effect"] > 0.0
    # every arm carries an honest classification, never an unconditional SAE "win".
    for a in arms.values():
        assert a["classification"] in ("diagnostic_only", "causal_but_inferior", "competitive", "rejected")


def test_classify_arm_rejects_non_localized_or_damaging():
    arm = SAEArm("S6", "top_sae_feature", "sparse", "train_only", (1.0,))
    not_localized = ArmEffect(arm, target_effect=1.0, preservation_damage=0.0, wrong_site_effect=1.0, trainable_params=1)
    assert classify_arm(not_localized, best_baseline_effect=1.0, preservation_budget=0.5) == "rejected"
    damaging = ArmEffect(arm, target_effect=1.0, preservation_damage=5.0, wrong_site_effect=0.0, trainable_params=1)
    assert classify_arm(damaging, best_baseline_effect=1.0, preservation_budget=0.5) == "rejected"
    diagnostic = ArmEffect(arm, target_effect=0.01, preservation_damage=0.0, wrong_site_effect=0.0, trainable_params=1)
    assert classify_arm(diagnostic, best_baseline_effect=1.0, preservation_budget=0.5) == "diagnostic_only"
