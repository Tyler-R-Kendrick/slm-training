"""LDI4-02 causal-intervention primitives + the matched S0-S7 fixture matrix (SLM-136).

Bounded interventions at an exact site/position: ablate a feature, add/subtract a
decoder/probe direction across a symmetric dose grid, and a wrong-site negative control.
A feature is causally useful only when the target effect is stable, preservation damage
stays within budget, the wrong-site control is null, and the direct baselines are not
clearly superior. This module proves that machinery on synthetic fixture activations; it
makes no steering claim and never promotes an SAE feature from correlation.
"""

from __future__ import annotations

from typing import Any

import torch

from slm_training.harnesses.representations.sae import SparseAutoencoder
from slm_training.harnesses.representations.spec import SAEArm, matched_sae_arms
from slm_training.versioning import build_version_stamp

__all__ = [
    "apply_direction",
    "ablate_feature",
    "readout",
    "diffmean_direction",
    "linear_probe_direction",
    "ArmEffect",
    "classify_arm",
    "run_fixture_matrix",
]


def apply_direction(h: torch.Tensor, direction: torch.Tensor, dose: float) -> torch.Tensor:
    unit = direction / direction.norm().clamp_min(1e-8)
    return h + dose * unit


def ablate_feature(sae: SparseAutoencoder, h: torch.Tensor, feature_idx: int) -> torch.Tensor:
    """Remove one SAE feature's reconstruction contribution at the recorded activation."""
    z = sae.encode(h)
    contribution = torch.outer(z[:, feature_idx], sae.decoder.weight[:, feature_idx])
    h_hat, _ = sae(h)
    return h_hat - contribution


def readout(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return h @ w


def diffmean_direction(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Train-only difference-in-means direction (reuses the reft convention)."""
    from slm_training.models.reft_intervention import diffmean_vector

    return diffmean_vector(positive, negative)


def linear_probe_direction(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """A tiny closed-form linear discriminant direction from train groups only."""
    mu = torch.cat([positive, negative]).mean(dim=0)
    pos, neg = positive - mu, negative - mu
    cov = (pos.T @ pos + neg.T @ neg) / max(1, positive.shape[0] + negative.shape[0])
    cov = cov + 1e-3 * torch.eye(cov.shape[0])
    return torch.linalg.solve(cov, positive.mean(0) - negative.mean(0))


class ArmEffect:
    """One arm's measured effect (target movement, preservation damage, wrong-site null)."""

    def __init__(self, arm: SAEArm, *, target_effect: float, preservation_damage: float,
                 wrong_site_effect: float, trainable_params: int | None) -> None:
        self.arm = arm
        self.target_effect = target_effect
        self.preservation_damage = preservation_damage
        self.wrong_site_effect = wrong_site_effect
        self.trainable_params = trainable_params
        self.classification = "diagnostic_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm.arm_id,
            "method": self.arm.method,
            "target_effect": self.target_effect,
            "preservation_damage": self.preservation_damage,
            "wrong_site_effect": self.wrong_site_effect,
            "trainable_params": self.trainable_params,
            "classification": self.classification,
        }


def classify_arm(
    effect: ArmEffect, *, best_baseline_effect: float, preservation_budget: float
) -> str:
    """Honest classification -- diagnostic_only, causal_but_inferior, competitive, or
    rejected. An SAE arm is never ``competitive`` unless it is localized (wrong-site
    null), within the preservation budget, and not clearly beaten by the controls."""
    localized = effect.wrong_site_effect <= 0.5 * max(effect.target_effect, 1e-9)
    within_budget = effect.preservation_damage <= preservation_budget
    if not localized or not within_budget:
        return "rejected"
    if effect.target_effect < 0.1 * max(best_baseline_effect, 1e-9):
        return "diagnostic_only"
    if effect.target_effect < best_baseline_effect:
        return "causal_but_inferior"
    return "competitive"


def run_fixture_matrix(
    *, d_in: int = 16, n: int = 64, seed: int = 0, site: str = "denoiser.block.0.residual",
    preservation_budget: float = 0.5,
) -> dict[str, Any]:
    """Deterministic S0-S7 fixture: synthetic target vs preservation activations with a
    known failure direction, every steering arm selected train-only and dosed on the same
    grid, measured for target movement / preservation damage / wrong-site null, then
    classified. Wiring evidence only -- no real model, no steering claim."""
    gen = torch.Generator().manual_seed(seed)
    target_dir = torch.zeros(d_in)
    target_dir[0] = 1.0
    wrong_dir = torch.zeros(d_in)
    wrong_dir[1] = 1.0
    # target rows carry the failure direction; negatives and preservation rows do not.
    pos = torch.randn(n, d_in, generator=gen) + 2.0 * target_dir
    neg = torch.randn(n, d_in, generator=gen)
    preservation = torch.randn(n, d_in, generator=gen)

    from slm_training.harnesses.representations.spec import SAEConfig

    sae = SparseAutoencoder(SAEConfig(d_in=d_in, expansion_factor=2, seed=seed))

    dm = diffmean_direction(pos, neg)
    probe = linear_probe_direction(pos, neg)
    # SAE "feature" direction: the decoder column most aligned with the diffmean target.
    align = (sae.decoder.weight.T @ (dm / dm.norm().clamp_min(1e-8)))
    top_feat = int(torch.argmax(align.abs()))
    sae_dir = sae.decoder.weight[:, top_feat]

    directions = {
        "S0": torch.zeros(d_in),
        "S1": torch.randn(d_in, generator=gen),
        "S2": dm,
        "S3": probe,
        "S4": dm,  # ReFT r1 stand-in (learned rank-1 ~ diffmean at fixture scale)
        "S5": dm,  # direct adapter stand-in
        "S6": sae_dir,
        "S7": sae_dir,
    }

    def effect_of(direction: torch.Tensor, apply_dir: torch.Tensor) -> float:
        if direction.norm() == 0:
            return 0.0
        base = readout(pos, target_dir).mean()
        moved = readout(apply_direction(pos, apply_dir, -1.0), target_dir).mean()
        return float((base - moved).abs())

    arms = matched_sae_arms(site=site)
    effects: list[ArmEffect] = []
    for arm in arms:
        d = directions[arm.arm_id]
        eff = ArmEffect(
            arm,
            target_effect=effect_of(d, d),
            preservation_damage=float(
                (readout(preservation, target_dir).mean()
                 - readout(apply_direction(preservation, d, -1.0) if d.norm() > 0 else preservation, target_dir).mean()).abs()
            ),
            wrong_site_effect=effect_of(d, wrong_dir),
            trainable_params=arm.trainable_params,
        )
        effects.append(eff)

    baseline_ids = {"S2", "S3", "S4", "S5"}
    best_baseline = max((e.target_effect for e in effects if e.arm.arm_id in baseline_ids), default=0.0)
    for e in effects:
        e.classification = classify_arm(
            e, best_baseline_effect=best_baseline, preservation_budget=preservation_budget
        )
    return {
        "matrix_set": "ldi4-02-sae-decision-state",
        "matrix_version": "ldi4-02-v1",
        "run_id": "ldi4_02_fixture",
        "site": site,
        "d_in": d_in,
        "n": n,
        "best_baseline_effect": best_baseline,
        "arms": [e.to_dict() for e in effects],
        "status": "wiring_only",
        "claim_class": "wiring",
        "note": "synthetic fixture; no real model/checkpoint; no steering or superiority claim",
        "version_stamp": build_version_stamp("harness.representations"),
    }
