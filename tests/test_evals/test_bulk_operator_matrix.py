"""SLM-411 crossover preflight invariants."""

from __future__ import annotations

from slm_training.evals.bulk_operator_matrix import (
    REQUIRED_ARMS,
    build_bulk_operator_crossover_report,
    run_local_preflight,
)


def test_local_probe_uses_live_legal_bulk_actions_without_a_crossover_claim() -> None:
    report = run_local_preflight({"verdict": "reject", "reason": "no complete rows"})

    assert report["verdict"] == "unavailable"
    assert report["systems"]["missing_arms"] == list(REQUIRED_ARMS)
    assert report["fixture_contract_satisfied"] is True
    assert [probe["fanout"] for probe in report["legal_set_fixture_probes"]] == [1, 2, 4, 8]
    assert all(probe["legal_actions"] > 0 for probe in report["legal_set_fixture_probes"])
    assert all(probe["replay_exact"] for probe in report["legal_set_fixture_probes"])
    assert all(probe["primitive_equivalent"] for probe in report["legal_set_fixture_probes"])


def test_no_fixture_or_policy_decision_can_invent_missing_serving_arms() -> None:
    report = build_bulk_operator_crossover_report(
        typed_policy_decision={"verdict": "measured"},
        probes=(),
    )

    assert report["fixture_contract_satisfied"] is False
    assert report["verdict"] == "unavailable"
    assert report["systems"]["missing_arms"] == list(REQUIRED_ARMS)
    assert report["ship_claim"] is False
