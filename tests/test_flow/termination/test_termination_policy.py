"""Tests for the TerminationPolicy protocol and reference arms."""

from __future__ import annotations



from slm_training.flow.termination import (
    HOLD,
    MAX_STEPS,
    STOP,
    STOP_EDIT,
    AbsorbingHazardPolicy,
    ExplicitStopPolicy,
    FixedKPlusSelectorPolicy,
    FixedKPolicy,
    HybridMinProgressPolicy,
    OracleLengthPolicy,
    TerminationContext,
    TerminationPolicy,
    build_termination_policy,
)


def test_build_termination_policy_returns_protocol() -> None:
    policy = build_termination_policy("fixed_k", k=3)
    assert isinstance(policy, TerminationPolicy)
    assert policy.name == "fixed_k"


def test_explicit_stop_triggers_on_high_score() -> None:
    policy = ExplicitStopPolicy(stop_threshold=0.5, max_steps=10)
    ctx = TerminationContext(state_fingerprint="s", step_index=0, stop_score=0.7)
    decision = policy.decide(ctx)
    assert decision.action == STOP
    assert decision.reason == STOP_EDIT


def test_explicit_stop_holds_on_low_score() -> None:
    policy = ExplicitStopPolicy(stop_threshold=0.5, max_steps=10)
    ctx = TerminationContext(state_fingerprint="s", step_index=0, stop_score=0.3)
    decision = policy.decide(ctx)
    assert decision.action == HOLD


def test_explicit_stop_respects_max_steps() -> None:
    policy = ExplicitStopPolicy(stop_threshold=0.9, max_steps=2)
    ctx = TerminationContext(state_fingerprint="s", step_index=2, stop_score=0.0)
    decision = policy.decide(ctx)
    assert decision.action == STOP
    assert decision.reason == MAX_STEPS


def test_absorbing_hazard_stops_on_low_hazard() -> None:
    policy = AbsorbingHazardPolicy(hazard_threshold=1e-6, absorb_threshold=0.9)
    ctx = TerminationContext(state_fingerprint="s", step_index=0, total_hazard=1e-9)
    decision = policy.decide(ctx)
    assert decision.action == STOP


def test_fixed_k_stops_at_k() -> None:
    policy = FixedKPolicy(k=3, max_steps=10)
    assert policy.decide(TerminationContext(state_fingerprint="s", edit_count=2)).action == HOLD
    assert policy.decide(TerminationContext(state_fingerprint="s", edit_count=3)).action == STOP


def test_fixed_k_plus_selector_requires_selector() -> None:
    policy = FixedKPlusSelectorPolicy(k=2, selector_threshold=0.5)
    assert policy.decide(TerminationContext(state_fingerprint="s", edit_count=2, selector_prob=0.6)).action == STOP
    assert policy.decide(TerminationContext(state_fingerprint="s", edit_count=2, selector_prob=0.1)).action == HOLD


def test_hybrid_min_progress_stops_on_stop_score() -> None:
    policy = HybridMinProgressPolicy(min_k=2, stop_threshold=0.5, selector_threshold=0.5)
    assert policy.decide(TerminationContext(state_fingerprint="s", stop_score=0.6)).action == STOP


def test_oracle_length_stops_at_oracle() -> None:
    policy = OracleLengthPolicy(oracle_edit_count=3)
    assert policy.decide(TerminationContext(state_fingerprint="s", edit_count=2)).action == HOLD
    assert policy.decide(TerminationContext(state_fingerprint="s", edit_count=3)).action == STOP


def test_all_registered_arm_names() -> None:
    from slm_training.flow.termination import POLICY_REGISTRY

    expected = {
        "explicit_stop",
        "absorbing_hazard",
        "fixed_k",
        "fixed_k_plus_selector",
        "hybrid_min_progress",
        "oracle_length",
    }
    assert set(POLICY_REGISTRY) == expected


def test_policy_round_trip_dict() -> None:
    policy = ExplicitStopPolicy(stop_threshold=0.7, max_steps=12)
    data = policy.to_dict()
    assert data["name"] == "explicit_stop"
    assert data["stop_threshold"] == 0.7
    assert data["max_steps"] == 12
