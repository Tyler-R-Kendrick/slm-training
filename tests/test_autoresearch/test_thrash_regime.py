"""Dual-regime thrash pure helpers (isolate / climb / timeout residual)."""

from __future__ import annotations

import pytest

from slm_training.autoresearch.thrash_regime import (
    DECODE_RESIDUAL_SLUGS,
    REGIME_CLIMB,
    REGIME_ISOLATE,
    REGIME_TIMEOUT_DECODE_RESIDUAL,
    compose_climb_control_levers,
    compose_isolate_control_levers,
    compose_treatment_levers,
    decide_screening_regime,
    is_compiler_ms_timeout_signal,
    select_climb_baseline_entry,
    select_decode_residual_slug,
    select_recommended_slug_for_regime,
)


def test_isolate_decision_without_champion() -> None:
    decision = decide_screening_regime(
        climb_baseline_knobs=None,
        compiler_ms_timeout=False,
    )
    assert decision.regime == REGIME_ISOLATE
    assert decision.base_regime == REGIME_ISOLATE
    assert decision.timeout_residual is False
    assert decision.climb_baseline is None


def test_climb_decision_with_sticky_baseline() -> None:
    baseline = {
        "binder_arity_loss_weight": 1.0,
        "binder_arity_decode_weight": 1.0,
        "compiler_decode_mode": "tree",
        "seed": 999,  # measurement — stripped
        "steps": 40,  # thrash budget — stripped
    }
    decision = decide_screening_regime(
        climb_baseline_knobs=baseline,
        compiler_ms_timeout=False,
    )
    assert decision.regime == REGIME_CLIMB
    assert decision.base_regime == REGIME_CLIMB
    assert decision.timeout_residual is False
    assert decision.climb_baseline is not None
    control = compose_climb_control_levers(decision.climb_baseline)
    assert control["binder_arity_loss_weight"] == 1.0
    assert control["compiler_decode_mode"] == "tree"
    assert "seed" not in control
    assert "steps" not in control


def test_timeout_residual_prefers_decode_arm_on_climb_baseline() -> None:
    baseline = {
        "component_plan_loss_weight": 1.0,
        "compiler_decode_mode": "tree",
    }
    decision = decide_screening_regime(
        climb_baseline_knobs=baseline,
        compiler_ms_timeout=True,
    )
    assert decision.regime == REGIME_TIMEOUT_DECODE_RESIDUAL
    assert decision.base_regime == REGIME_CLIMB
    assert decision.timeout_residual is True

    bank = list(DECODE_RESIDUAL_SLUGS) + ["binder-arity", "fidelity"]
    skip: set[str] = set()
    slug = select_recommended_slug_for_regime(
        decision=decision,
        cycle=1,
        skip=skip,
        bank_slugs=bank,
        isolate_selector=lambda c, s: "binder-arity",
    )
    assert slug == "bounds"
    assert slug in DECODE_RESIDUAL_SLUGS
    assert slug != "binder-arity"


def test_timeout_residual_falls_back_when_decode_bank_closed() -> None:
    decision = decide_screening_regime(
        climb_baseline_knobs=None,
        compiler_ms_timeout=True,
    )
    skip = set(DECODE_RESIDUAL_SLUGS)
    slug = select_recommended_slug_for_regime(
        decision=decision,
        cycle=7,
        skip=skip,
        bank_slugs=list(DECODE_RESIDUAL_SLUGS) + ["fidelity"],
        isolate_selector=lambda c, s: "fidelity",
    )
    assert slug == "fidelity"


def test_compose_treatment_is_residual_only_over_control() -> None:
    control = compose_climb_control_levers(
        {
            "binder_arity_loss_weight": 1.0,
            "compiler_decode_mode": "tree",
        }
    )
    residual = compose_isolate_control_levers(
        precursor_extras={"grammar_completion_bounds": True}
    )
    treatment = compose_treatment_levers(
        control_levers=control,
        residual_extras=residual,
    )
    assert treatment["binder_arity_loss_weight"] == 1.0
    assert treatment["grammar_completion_bounds"] is True
    assert treatment["compiler_decode_mode"] == "tree"
    # Residual must not clear champion quality levers.
    assert control["binder_arity_loss_weight"] == 1.0


def test_isolate_control_is_empty_or_precursor() -> None:
    assert compose_isolate_control_levers() == {}
    assert compose_isolate_control_levers(
        precursor_extras={"grammar_completion_bounds": True}
    ) == {"grammar_completion_bounds": True}


def test_compiler_ms_timeout_signal_from_detail() -> None:
    assert is_compiler_ms_timeout_signal(
        decode_outcome_detail="timeout_dominant_phase=compiler_ms(14697ms/17334ms)",
        incomplete=True,
        decode_timeout_count=1,
    )
    assert not is_compiler_ms_timeout_signal(
        decode_outcome_detail=None,
        incomplete=False,
        decode_timeout_count=0,
    )


def test_select_climb_baseline_prefers_accepted_then_confirmed() -> None:
    queue = [
        {
            "status": "rejected",
            "knobs": {"binder_arity_loss_weight": 0.1},
        },
        {
            "status": "confirmed",
            "knobs": {"fidelity_loss_weight": 1.5},
        },
        {
            "status": "climb_accepted",
            "knobs": {"component_plan_loss_weight": 1.0},
        },
    ]
    entry = select_climb_baseline_entry(queue)
    assert entry is not None
    assert entry["knobs"]["component_plan_loss_weight"] == 1.0


def test_decode_residual_slug_skips_closed() -> None:
    assert (
        select_decode_residual_slug(skip={"bounds"}, bank_slugs=DECODE_RESIDUAL_SLUGS)
        == "canvas"
    )
    assert select_decode_residual_slug(skip=set(DECODE_RESIDUAL_SLUGS)) is None


def test_wrong_control_baseline_fails_assertions() -> None:
    """Guard against tests that would pass if climb accidentally used zero control."""
    baseline = {"grammar_completion_bounds": True, "compiler_decode_mode": "tree"}
    control = compose_climb_control_levers(baseline)
    with pytest.raises(AssertionError):
        assert control == {}
    with pytest.raises(AssertionError):
        assert "grammar_completion_bounds" not in control


# --- P4: timeout cause gates decode residual routing ---------------------------


def test_classify_timeout_cause_budget_vs_slow_tail() -> None:
    from slm_training.autoresearch.thrash_regime import (
        TIMEOUT_CAUSE_BUDGET,
        TIMEOUT_CAUSE_NONE,
        TIMEOUT_CAUSE_SLOW_DECODE,
        classify_timeout_cause,
    )

    # Measured p95 (30 s) exceeds the applied 12 s timeout: the budget was
    # infeasible, so the timeout is a budget fact.
    assert (
        classify_timeout_cause(p95_seconds=30.0, timeout_seconds=12.0, timeout_count=3)
        == TIMEOUT_CAUSE_BUDGET
    )
    # p95 fits the timeout yet a tail record timed out: genuinely slow decode.
    assert (
        classify_timeout_cause(p95_seconds=9.0, timeout_seconds=12.0, timeout_count=1)
        == TIMEOUT_CAUSE_SLOW_DECODE
    )
    assert (
        classify_timeout_cause(p95_seconds=9.0, timeout_seconds=12.0, timeout_count=0)
        == TIMEOUT_CAUSE_NONE
    )
    # Undecidable without a measured cost or an applied timeout.
    assert (
        classify_timeout_cause(p95_seconds=None, timeout_seconds=12.0, timeout_count=2)
        == TIMEOUT_CAUSE_NONE
    )
    assert (
        classify_timeout_cause(p95_seconds=30.0, timeout_seconds=None, timeout_count=2)
        == TIMEOUT_CAUSE_NONE
    )


def test_budget_timeout_never_routes_to_decode_residual_slugs() -> None:
    from slm_training.autoresearch.thrash_regime import TIMEOUT_CAUSE_BUDGET

    baseline = {"component_plan_loss_weight": 1.0, "compiler_decode_mode": "tree"}
    decision = decide_screening_regime(
        climb_baseline_knobs=baseline,
        compiler_ms_timeout=True,
        timeout_cause=TIMEOUT_CAUSE_BUDGET,
    )
    assert decision.timeout_residual is False
    assert decision.regime == REGIME_CLIMB
    assert decision.base_regime == REGIME_CLIMB
    assert "budget_timeout" in decision.reason
    assert "recalibrate_budget_not_residual" in decision.reason
    bank = list(DECODE_RESIDUAL_SLUGS) + ["binder-arity", "fidelity"]
    slug = select_recommended_slug_for_regime(
        decision=decision,
        cycle=1,
        skip=set(),
        bank_slugs=bank,
        isolate_selector=lambda c, s: "binder-arity",
    )
    assert slug == "binder-arity"
    assert slug not in DECODE_RESIDUAL_SLUGS
    # A feasible-budget slow tail keeps the residual route.
    slow = decide_screening_regime(
        climb_baseline_knobs=baseline,
        compiler_ms_timeout=True,
        timeout_cause="slow_decode_timeout",
    )
    assert slow.timeout_residual is True
