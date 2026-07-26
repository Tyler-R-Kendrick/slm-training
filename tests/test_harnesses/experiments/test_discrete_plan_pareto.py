from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.discrete_plan_pareto import (
    AP022_DISPOSITION,
    ARMS,
    KNOWN_COLLAPSED_ARMS,
    REFINEMENT_ROUNDS,
    SUCCESSOR_CANDIDATE_ARM,
    ParetoLatencyRow,
    build_campaign,
    canonical_arm,
    pareto_frontier,
    summarize_pareto,
)


def _row(**overrides: object) -> ParetoLatencyRow:
    defaults = dict(
        record_id="r0",
        arm="no_plan",
        refinement_round=1,
        path="constrained",
        total_latency_ms=10.0,
        plan_latency_ms=0.0,
        generation_latency_ms=8.0,
        verification_latency_ms=2.0,
        peak_memory_proxy=100.0,
        output_tokens=12,
        metrics={"binding_aware_meaningful_v2": 0.5, "binder_reference_f1": 0.5},
    )
    defaults.update(overrides)
    return ParetoLatencyRow(**defaults)  # type: ignore[arg-type]


def test_known_collapsed_arm_can_never_be_marked_promotion_eligible() -> None:
    """AP-022's closed ignored_or_collapsed verdict is a permanent, structural
    fact this harness enforces in code -- not something a later measurement
    can silently override by claiming promotion_eligible=True (I14)."""
    row = _row(arm="learned_abstract_plan", promotion_eligible=True)
    with pytest.raises(ValueError, match="closed by AP-022"):
        row.validate()
    # The same arm without a promotion claim is fine: it stays measurable as
    # a latency/diagnostic reference point.
    _row(arm="learned_abstract_plan", promotion_eligible=False).validate()


def test_known_collapsed_arms_are_a_subset_of_arms() -> None:
    assert KNOWN_COLLAPSED_ARMS <= set(ARMS)
    assert SUCCESSOR_CANDIDATE_ARM not in KNOWN_COLLAPSED_ARMS
    assert SUCCESSOR_CANDIDATE_ARM in ARMS


def test_oracle_plan_alias_resolves_to_ap022s_gold_semantic_plan() -> None:
    assert canonical_arm("oracle_plan") == "gold_semantic_plan"
    _row(arm="oracle_plan").validate()


def test_unknown_arm_or_path_or_round_rejected() -> None:
    with pytest.raises(ValueError, match="unknown discrete-plan-pareto arm"):
        _row(arm="not_a_real_arm").validate()
    with pytest.raises(ValueError, match="unknown decode path"):
        _row(path="not_a_real_path").validate()
    with pytest.raises(ValueError, match="refinement_round must be one of"):
        _row(refinement_round=3).validate()


def test_phase_latencies_cannot_exceed_reported_total() -> None:
    with pytest.raises(ValueError, match="must not exceed the reported total"):
        _row(total_latency_ms=1.0, plan_latency_ms=5.0).validate()


def test_raw_path_is_never_promotion_eligible() -> None:
    with pytest.raises(ValueError, match="diagnostic-only"):
        _row(path="raw", promotion_eligible=True).validate()


def test_campaign_candidate_is_the_untested_successor_not_the_closed_arm() -> None:
    campaign = build_campaign()
    assert {arm.arm_id for arm in campaign.arms} == set(ARMS)
    candidates = [arm for arm in campaign.arms if arm.role == "candidate"]
    assert [arm.arm_id for arm in candidates] == [SUCCESSOR_CANDIDATE_ARM]
    assert set(campaign.negative_controls) == KNOWN_COLLAPSED_ARMS
    assert campaign.claim_class == "wiring"


def test_pareto_frontier_excludes_dominated_points() -> None:
    points = [
        ("cheap_low_quality", 5.0, 0.1),
        ("expensive_high_quality", 20.0, 0.9),
        ("dominated", 20.0, 0.5),  # same latency as above, worse quality
        ("dominated_by_cheap", 6.0, 0.05),  # worse on both axes than cheap_low_quality
    ]
    frontier = pareto_frontier(points)
    assert set(frontier) == {"cheap_low_quality", "expensive_high_quality"}


def test_pareto_frontier_keeps_exact_ties() -> None:
    points = [("a", 10.0, 0.5), ("b", 10.0, 0.5)]
    assert set(pareto_frontier(points)) == {"a", "b"}


def test_summarize_pareto_reports_pending_and_closed_arms_honestly() -> None:
    rows = [_row(arm="no_plan", refinement_round=r) for r in REFINEMENT_ROUNDS]
    report = summarize_pareto(rows)
    assert report["measured_arms"] == ["no_plan"]
    assert SUCCESSOR_CANDIDATE_ARM in report["arms_pending"]
    assert "retrieval_baseline" in report["arms_pending"]
    assert report["arms_closed"] == tuple(sorted(KNOWN_COLLAPSED_ARMS))
    assert report["ap022_disposition"] == dict(AP022_DISPOSITION)
    assert report["ship_eligible"] is False
    assert report["claim_class"] == "wiring"
    # 4 constrained-path points, all the same arm at different rounds; none
    # dominate each other unless latency/quality actually vary.
    assert len(report["non_dominated"]) >= 1
