"""Tests for the SLM-243 recursive-update verdict contract."""

from slm_training.harnesses.experiments.slm243_recursive_update_gate import (
    RecursiveUpdateVerdict,
    classify_recursive_update_gate,
)


def _rows(*, layerscale_ratio: float = 0.2) -> list[dict]:
    rows = []
    for variant in (
        "current_v1",
        "delta_only",
        "layerscale",
        "gated_private",
        "current_true_empty",
        "layerscale_private",
    ):
        for depth in (1, 2, 4, 6, 8):
            for seed in (24301, 24302, 24303):
                ratio = 1.0
                if variant == "layerscale":
                    ratio = layerscale_ratio
                rows.append(
                    {
                        "variant": variant,
                        "depth": depth,
                        "seed": seed,
                        "all_finite": True,
                        "cross_entropy": 2.0,
                        "maximum_update_ratio": ratio,
                        "gradient_norm": 1.0,
                    }
                )
    return rows


def test_classifier_prefers_layerscale_on_three_paired_seeds() -> None:
    gate = classify_recursive_update_gate(
        _rows(), depths=(1, 2, 4, 6, 8), seeds=(24301, 24302, 24303)
    )
    assert gate.verdict == RecursiveUpdateVerdict.LAYERSCALE_PREFERRED.value
    assert gate.maximum_authorized_depth == 8
    assert gate.allowed_slm233_modes == ("layerscale_diagnostic",)
    assert "ship_readiness" in gate.blocked_claims


def test_classifier_fails_inconclusive_on_missing_cell() -> None:
    gate = classify_recursive_update_gate(
        _rows()[:-1], depths=(1, 2, 4, 6, 8), seeds=(24301, 24302, 24303)
    )
    assert gate.verdict == RecursiveUpdateVerdict.INCONCLUSIVE.value
    assert gate.maximum_authorized_depth == 0
