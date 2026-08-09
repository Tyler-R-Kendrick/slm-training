"""SIE-007 (SLM-473): the controller substrate must stay safe and advisory.

Each test reproduces one way a value-of-compute controller could stop being a
diagnostic and start being an authority: executing arbitrary code, guessing on
UNKNOWN input, running at all on a decided singleton, changing legality, or
scoring itself against something other than the observed outcome.
"""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.compute_state_controller import (
    COMPUTE_STATE_FEATURES_SCHEMA,
    CONTROLLER_ACTIONS,
    CONTROLLER_MAX_NODES,
    FEATURE_ORDER,
    ComputeStateFeaturesV1,
    ControllerRuleV1,
    controller_can_change_legality,
    features_from_decode_record,
    oracle_action_for_record,
    replay_rule,
)
from slm_training.models.decode_stats import DecodeStats, build_decode_stats_record


def _record(
    *,
    legal_domain_size: int | None = 4,
    legal_domain_status: str = "complete",
    **stat_kwargs: float,
):
    return build_decode_stats_record(
        DecodeStats(**stat_kwargs),
        measurement_stage="steady_state",
        legal_domain_size=legal_domain_size,
        legal_domain_status=legal_domain_status,
    )


def _rule(**kwargs) -> ControllerRuleV1:
    defaults = {
        "rule_id": "r0",
        "action_expressions": {
            # v3 = forwards_count, v2 = ambiguous_rows_forwarded
            "neural_forward": "v2",
            "deterministic_ranker": "v5",
        },
    }
    defaults.update(kwargs)
    return ControllerRuleV1(**defaults)


def test_feature_vector_is_total_and_pinned_to_the_frozen_vocabulary() -> None:
    features = features_from_decode_record(_record())
    assert features.schema_version == COMPUTE_STATE_FEATURES_SCHEMA
    assert len(features.feature_vector()) == len(FEATURE_ORDER)
    with pytest.raises(ValueError, match="not total"):
        ComputeStateFeaturesV1(
            schema_version=COMPUTE_STATE_FEATURES_SCHEMA,
            values={"legal_domain_size": 1.0},
            legal_domain_status="complete",
            measurement_stage="steady_state",
            completeness="complete",
        )


def test_absent_measurement_is_unknown_not_zero() -> None:
    features = features_from_decode_record(
        _record(legal_domain_size=None, legal_domain_status="unknown")
    )
    assert features.values["legal_domain_size"] is None
    assert features.values["legal_domain_complete"] is None
    assert "legal_domain_size" in features.unknown_features()
    # A measured zero stays a zero, so UNKNOWN and "measured empty" differ.
    assert features.values["forwards_count"] == 0.0


def test_exact_singleton_bypasses_the_controller_with_zero_work() -> None:
    features = features_from_decode_record(_record(legal_domain_size=1))
    assert features.is_exact_singleton
    recommendation = _rule().recommend(features)
    assert recommendation.expressions_evaluated == 0
    assert recommendation.action == "deterministic_ranker"
    assert recommendation.score is None


def test_incomplete_domain_of_size_one_is_not_a_singleton() -> None:
    features = features_from_decode_record(
        _record(legal_domain_size=1, legal_domain_status="incomplete")
    )
    assert not features.is_exact_singleton


def test_rule_abstains_on_unknown_features_rather_than_guessing() -> None:
    features = features_from_decode_record(
        _record(legal_domain_size=None, legal_domain_status="unknown")
    )
    # v0 = legal_domain_size, which is UNKNOWN on this record.
    recommendation = ControllerRuleV1(
        rule_id="r-unknown", action_expressions={"neural_forward": "v0"}
    ).recommend(features)
    assert recommendation.action == "abstain"
    assert recommendation.expressions_evaluated == 0
    assert "legal_domain_size" in recommendation.reason


@pytest.mark.parametrize(
    "malicious",
    [
        "__import__('os').system('rm -rf /')",
        "(+ v0 __import__)",
        "eval",
        "(exec v0)",
        "(+ v0 1.5)",
        "(+ v0 v999)",
        "",
        "(+ v0",
        "(+ v0) trailing",
    ],
)
def test_malicious_or_malformed_rules_are_rejected_at_construction(malicious: str) -> None:
    with pytest.raises(ValueError):
        ControllerRuleV1(rule_id="r-bad", action_expressions={"neural_forward": malicious})


def test_rule_complexity_is_bounded() -> None:
    over_budget = "(+ v0 " * (CONTROLLER_MAX_NODES + 2)
    over_budget += "v0" + ")" * (CONTROLLER_MAX_NODES + 2)
    with pytest.raises(ValueError, match="max_depth|max_nodes"):
        ControllerRuleV1(rule_id="r-big", action_expressions={"neural_forward": over_budget})


def test_unbound_coefficient_and_unscorable_action_are_rejected() -> None:
    with pytest.raises(ValueError, match="unbound coefficient"):
        ControllerRuleV1(rule_id="r-c", action_expressions={"neural_forward": "(+ v0 c0)"})
    with pytest.raises(ValueError, match="unscorable action"):
        ControllerRuleV1(rule_id="r-a", action_expressions={"abstain": "v0"})
    with pytest.raises(ValueError, match="unscorable action"):
        ControllerRuleV1(rule_id="r-a", action_expressions={"launch_missiles": "v0"})


def test_abstain_margin_suppresses_a_coin_flip() -> None:
    features = features_from_decode_record(_record(forwards_count=3, probes_count=3))
    tie = {"neural_forward": "v3", "bounded_search": "v8"}
    assert ControllerRuleV1(rule_id="r-tie", action_expressions=tie).recommend(features).action in {
        "bounded_search",
        "neural_forward",
    }
    margined = ControllerRuleV1(
        rule_id="r-margin", action_expressions=tie, abstain_margin=0.5
    ).recommend(features)
    assert margined.action == "abstain"


def test_recommendation_is_deterministic_over_a_frozen_record() -> None:
    record = _record(forwards_count=7, ambiguous_rows_forwarded=2)
    rule = _rule()
    first = rule.recommend(features_from_decode_record(record))
    second = rule.recommend(features_from_decode_record(record))
    assert first.to_dict() == second.to_dict()


def test_controller_never_changes_legality() -> None:
    assert controller_can_change_legality() is False


def test_oracle_action_reads_observed_outcomes_only() -> None:
    assert oracle_action_for_record(_record(legal_domain_size=1)) == "deterministic_ranker"
    assert oracle_action_for_record(_record(unconstrained_retries=1)) == "repair"
    assert oracle_action_for_record(_record(attempts=3)) == "bounded_search"
    assert oracle_action_for_record(_record(forwards_count=2)) == "neural_forward"
    assert oracle_action_for_record(_record(forwards_count=0)) == "deterministic_ranker"


def test_replay_scores_regret_and_does_not_reward_blanket_abstention() -> None:
    records = (
        _record(forwards_count=5, ambiguous_rows_forwarded=5),
        _record(legal_domain_size=1),
        _record(unconstrained_retries=1, ambiguous_rows_forwarded=1),
    )
    report = replay_rule(_rule(), records)
    assert len(report.recommendations) == len(records) == len(report.oracle_actions)
    assert 0.0 <= report.regret <= 1.0
    assert report.to_dict()["rule_id"] == "r0"

    # A rule that only ever abstains gets no credit: zero agreements, and its
    # regret is not a free 0.0 win over a rule that actually decided.
    always_unknown = ControllerRuleV1(
        rule_id="r-abstainer", action_expressions={"neural_forward": "v0"}
    )
    unknown_records = (
        _record(legal_domain_size=None, legal_domain_status="unknown"),
        _record(legal_domain_size=None, legal_domain_status="unknown"),
    )
    abstained = replay_rule(always_unknown, unknown_records)
    assert abstained.abstentions == len(unknown_records)
    assert abstained.agreements == 0
    assert abstained.expressions_evaluated == 0
    # Never the vacuous 1 - 0/0 = 0 that would outrank every rule that decided.
    assert abstained.regret == 1.0
    assert abstained.regret >= report.regret


def test_replay_is_deterministic() -> None:
    records = (_record(forwards_count=1), _record(attempts=2))
    assert replay_rule(_rule(), records).to_dict() == replay_rule(_rule(), records).to_dict()


def test_actions_are_a_closed_enum_including_abstain() -> None:
    assert "abstain" in CONTROLLER_ACTIONS
    assert set(CONTROLLER_ACTIONS) == {
        "deterministic_ranker",
        "neural_forward",
        "bounded_search",
        "repair",
        "abstain",
    }


def test_features_round_trip_and_reject_a_foreign_version() -> None:
    payload = features_from_decode_record(_record()).to_dict()
    assert ComputeStateFeaturesV1.from_dict(payload).to_dict() == payload
    with pytest.raises(ValueError, match="unsupported compute state features schema"):
        ComputeStateFeaturesV1.from_dict({**payload, "schema_version": "compute_state_features/v0"})
