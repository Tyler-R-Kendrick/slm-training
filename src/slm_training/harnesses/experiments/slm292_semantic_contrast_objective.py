"""Tiny local wiring control for SLM-292; it is not a quality evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from slm_training.models.semantic_contrast_loss import load_semantic_contrast_pairs
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.versioning import build_version_stamp


def run_local_control(corpus: Path, *, seed: int = 0) -> dict:
    """Prove disabled parity and enabled pair telemetry on two local pairs."""
    pairs = load_semantic_contrast_pairs(corpus)[:2]
    records = [side for pair in pairs for side in (pair.positive, pair.negative)]

    def build(weight: float) -> TwoTowerModel:
        return TwoTowerModel.from_records(
            records,
            config=TwoTowerConfig(
                d_model=32,
                n_heads=4,
                context_layers=1,
                denoiser_layers=1,
                output_tokenizer="lexer",
                seed=seed,
                semantic_contrast_loss_weight=weight,
                semantic_contrast_margin=1.0,
            ),
        )

    baseline = build(0.0)
    disabled = build(0.0)
    state = baseline._rng.getstate()
    torch.manual_seed(seed + 1)
    baseline_loss = baseline.training_loss(records)
    baseline._rng.setstate(state)
    disabled._rng.setstate(state)
    torch.manual_seed(seed + 1)
    disabled_loss = disabled.training_loss(records)

    enabled = build(0.25)
    enabled._rng.setstate(state)
    torch.manual_seed(seed + 1)
    enabled_loss = enabled.training_loss(records)
    metrics = enabled.last_training_metrics
    return {
        "schema": "slm292_semantic_contrast_objective/v1",
        "claim_class": "wiring_only",
        "recipe": {
            "device": "cpu",
            "backend": "scratch",
            "pairs": len(pairs),
            "seeds": 1,
            "checkpoint_written": False,
        },
        "disabled": {
            "objective": 0.0,
            "legacy_loss_bit_exact": bool(torch.equal(baseline_loss, disabled_loss)),
        },
        "enabled": {
            "loss": float(enabled_loss.detach()),
            "semantic_contrast_loss": metrics["semantic_contrast_loss"],
            "pairs": metrics["semantic_contrast_pairs"],
            "family_sides": metrics["semantic_contrast_family_sides"],
            "positive_nll": metrics["semantic_contrast_positive_nll"],
            "negative_nll": metrics["semantic_contrast_negative_nll"],
            "score_distance": metrics["semantic_contrast_score_distance"],
            "violation_rate": metrics["semantic_contrast_violation_rate"],
        },
        "promotion": {"eligible": False, "reason": "local wiring control only"},
        "version_stamp": build_version_stamp(
            "model.twotower", "harness.model_build.train", "harness.experiments.slm292_semantic_contrast_objective"
        ),
    }
