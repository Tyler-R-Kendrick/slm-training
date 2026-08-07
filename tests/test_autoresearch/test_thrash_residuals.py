"""Tests for cheap thrash residual classification and soft ranking."""

from __future__ import annotations

from slm_training.autoresearch.thrash_residuals import (
    RESIDUAL_CONTROL_SPIKE_SHARED,
    RESIDUAL_EFFICIENCY_WIN_QUALITY_HELD,
    RESIDUAL_HIGH_BAND_ABSOLUTE,
    RESIDUAL_PRIMARY_UP_BINDER_DOWN,
    aggregate_slug_stats,
    classify_delivery_residual,
    pick_soft_ranked_slug,
    residual_boosts_from_observations,
    slug_from_candidate_id,
    soft_rank_slugs,
)


def _delivery(
    *,
    cand_id: str,
    ss_c: float,
    ss_k: float,
    mpr_c: float = 0.33,
    mpr_k: float = 0.33,
    b_c: float = 0.95,
    b_k: float = 0.63,
    reasons: list[str] | None = None,
    positive: bool = False,
    complete: bool = True,
) -> dict:
    return {
        "measurement_complete": complete,
        "positive": positive,
        "candidate_id": cand_id,
        "control_id": "control",
        "control_metrics": {
            "structural_similarity": ss_c,
            "meaningful_program_rate": mpr_c,
            "binder_reference_f1": b_c,
        },
        "candidate_metrics": {
            "structural_similarity": ss_k,
            "meaningful_program_rate": mpr_k,
            "binder_reference_f1": b_k,
        },
        "reasons": reasons or [],
    }


def test_slug_from_candidate_id() -> None:
    assert (
        slug_from_candidate_id(
            "c20260806-continuous-openui-local-8c0b60dd-c251-"
            "compose-ltr-tail-compiler-decision-token"
        )
        == "compose-ltr-tail-compiler-decision-token"
    )
    assert slug_from_candidate_id("...-c10-control") is None


def test_classify_primary_up_binder_down_c251_shape() -> None:
    # Live c251-shaped delivery: efficiency + quality_held + binder drop + small ΔSS.
    d = _delivery(
        cand_id=(
            "c20260806-continuous-openui-local-8c0b60dd-c251-"
            "compose-ltr-tail-compiler-decision-token"
        ),
        ss_c=0.396,
        ss_k=0.420,
        reasons=[
            "fixture_insufficient_n:control",
            "efficiency_win:mpr_per_ms:a->b:gain_fraction=0.68:minimum=0.05",
            "quality_held:parse=1.0 mpr=0.33",
            "non_regression_fail:binder_reference_f1:0.95->0.63",
            "primary_metric_null_or_worse:smoke.structural_similarity:"
            "control=0.396 candidate=0.420 improvement=0.024",
        ],
    )
    obs = classify_delivery_residual(d, campaign_id="camp-c251", cycle_index=251)
    assert obs is not None
    assert obs.residual_class == RESIDUAL_PRIMARY_UP_BINDER_DOWN
    assert obs.slug == "compose-ltr-tail-compiler-decision-token"
    boosts = residual_boosts_from_observations([obs])
    assert boosts.get("compose-ltr-tail-compiler-decision-token", 0) > 0


def test_classify_control_spike_shared_no_densify() -> None:
    d = _delivery(
        cand_id="loop-c10-bounds",
        ss_c=0.40,
        ss_k=0.401,
        b_c=0.63,
        b_k=0.63,
        reasons=["fixture_insufficient_n_alone", "primary_metric_null_or_worse:x"],
    )
    obs = classify_delivery_residual(d, campaign_id="c", cycle_index=10)
    assert obs is not None
    assert obs.residual_class == RESIDUAL_CONTROL_SPIKE_SHARED
    boosts = residual_boosts_from_observations([obs])
    assert "bounds" not in boosts  # explicit non-densify


def test_classify_efficiency_win_quality_held() -> None:
    d = _delivery(
        cand_id="loop-c5-literal-close",
        ss_c=0.17,
        ss_k=0.17,
        reasons=[
            "efficiency_win:mpr_per_ms:1->2:gain_fraction=0.5:minimum=0.05",
            "quality_held:parse=1.0 mpr=0.33",
            "fixture_insufficient_n:x",
        ],
    )
    obs = classify_delivery_residual(d, campaign_id="c", cycle_index=5)
    assert obs is not None
    assert obs.residual_class == RESIDUAL_EFFICIENCY_WIN_QUALITY_HELD


def test_classify_high_band_with_primary_up() -> None:
    d = _delivery(
        cand_id="loop-c9-compose-ltr-tail-ltr-prefix",
        ss_c=0.20,
        ss_k=0.40,
        b_c=0.63,
        b_k=0.63,
        reasons=["primary_metric_win:ss:0.2->0.4", "fixture_insufficient_n:x"],
    )
    obs = classify_delivery_residual(d, campaign_id="c", cycle_index=9)
    assert obs is not None
    assert obs.residual_class in {
        RESIDUAL_HIGH_BAND_ABSOLUTE,
        RESIDUAL_PRIMARY_UP_BINDER_DOWN,  # only if binder fail present
    }
    assert obs.residual_class == RESIDUAL_HIGH_BAND_ABSOLUTE


def test_incomplete_not_interesting() -> None:
    d = _delivery(
        cand_id="loop-c1-bounds",
        ss_c=0.5,
        ss_k=0.6,
        complete=False,
        reasons=["measurement_incomplete"],
    )
    assert classify_delivery_residual(d, campaign_id="c") is None


def test_soft_rank_prefers_residual_boost() -> None:
    ordered = soft_rank_slugs(
        ["bounds", "canvas", "compose-x"],
        residual_boosts={"compose-x": 5.0},
        rotation_order=["bounds", "canvas", "compose-x"],
    )
    assert ordered[0] == "compose-x"
    assert pick_soft_ranked_slug(
        ["bounds", "canvas"],
        residual_boosts={},
        rotation_order=["canvas", "bounds"],
    ) == "canvas"


def test_aggregate_slug_stats() -> None:
    deliveries = [
        _delivery(
            cand_id="loop-c1-bounds",
            ss_c=0.1,
            ss_k=0.2,
            positive=True,
            reasons=["primary_metric_win:x"],
        ),
        _delivery(
            cand_id="loop-c2-bounds",
            ss_c=0.2,
            ss_k=0.15,
            reasons=["non_regression_fail:binder_reference_f1:a->b"],
        ),
        _delivery(
            cand_id="loop-c3-canvas",
            ss_c=0.1,
            ss_k=0.1,
            complete=False,
        ),
    ]
    stats = aggregate_slug_stats(deliveries)
    assert stats["bounds"].n_complete == 2
    assert stats["bounds"].n_positive == 1
    assert stats["bounds"].binder_fail_rate == 0.5
    assert stats["canvas"].incomplete_n == 1
