"""Tests for the LDI2-03 TwoTower adapter/full-update campaign matrix (SLM-126).

These pin the orchestration invariants — the authorization gate fails closed,
only an ``authorized`` diagnostic permits training, the exact-signature guard
rejects any protected regression, and no arm fabricates a result without an
executable policy + admitted corpus. No training or quality claim is exercised.
"""

from __future__ import annotations

import pytest

from slm_training.harnesses.preference.twotower_adapter_matrix import (
    ArmResult,
    GuardMetrics,
    build_arms,
    describe_campaign,
    exact_signature_guard,
    read_authorization,
    run_arm,
)


def test_read_authorization_maps_all_states() -> None:
    assert read_authorization({"status": "expired"})[0] == "expired"
    assert read_authorization({"status": "not_authorized", "reason": "x"})[0] == "no_safe_direction"
    assert read_authorization({"status": "completed", "result": {"decision": "authorized"}})[0] == "authorized"
    assert read_authorization({"status": "completed", "result": {"decision": "repair_evidence"}})[0] == "repair_evidence"


def test_authorization_fails_closed_on_missing_or_unknown_decision() -> None:
    # Completed but no result -> not authorized.
    assert read_authorization({"status": "completed", "result": None})[0] == "no_safe_direction"
    # Completed with an unrecognized decision -> fail closed, never authorized.
    assert read_authorization({"status": "completed", "result": {"decision": "great"}})[0] == "no_safe_direction"
    # Empty report -> fail closed.
    assert read_authorization({})[0] == "no_safe_direction"


def test_build_arms_has_matched_t0_t5() -> None:
    arms = build_arms(authorized_rank=8, lower_rank=4, higher_rank=16)
    labels = [a.label for a in arms]
    assert labels == ["T0-parent", "T1-full", "T2-adapter", "T3-lower", "T4-higher", "T5-tether"]
    assert arms[0].update_space == "parent" and arms[1].update_space == "full"
    assert arms[2].rank == 8 and arms[-1].reference_tether is True
    # Capacity controls are optional.
    assert [a.label for a in build_arms(authorized_rank=8, include_tether_ablation=False)] == [
        "T0-parent",
        "T1-full",
        "T2-adapter",
    ]


def test_build_arms_rejects_nonpositive_rank() -> None:
    with pytest.raises(ValueError):
        build_arms(authorized_rank=0)


def test_unauthorized_blocks_trainable_but_not_parent() -> None:
    arms = {a.label: a for a in build_arms(authorized_rank=8)}
    # no_safe_direction blocks the adapter arm...
    res = run_arm(arms["T2-adapter"], decision="no_safe_direction", corpus_admitted=True)
    assert res.status == "blocked" and "not authorized" in res.reason
    # ...but the parent control stays admissible and simply expires without a policy.
    parent = run_arm(arms["T0-parent"], decision="no_safe_direction", corpus_admitted=False)
    assert parent.status == "expired"
    assert parent.metrics is None


def test_authorized_but_unadmitted_corpus_blocks() -> None:
    arm = {a.label: a for a in build_arms(authorized_rank=8)}["T2-adapter"]
    res = run_arm(arm, decision="authorized", corpus_admitted=False)
    assert res.status == "blocked" and "blocked_by_corpus" in res.reason


def test_run_arm_expires_without_policy_never_fabricates() -> None:
    arm = {a.label: a for a in build_arms(authorized_rank=8)}["T2-adapter"]
    res = run_arm(arm, decision="authorized", corpus_admitted=True, policy_factory=None)
    assert res.status == "expired"
    assert res.metrics is None
    assert "GPU" in res.reason


def test_exact_signature_guard_rejects_each_regression() -> None:
    base = {"held_out_loss": 1.0, "good_mass": 0.5, "bad_mass": 0.2, "margin": 0.3, "locality": 0.1}
    ok = exact_signature_guard(GuardMetrics(pre=base, post=dict(base, held_out_loss=0.9)))
    assert ok[0] is True
    # loss up -> reject
    assert exact_signature_guard(GuardMetrics(pre=base, post=dict(base, held_out_loss=1.1)))[0] is False
    # good_mass down -> reject
    assert exact_signature_guard(GuardMetrics(pre=base, post=dict(base, good_mass=0.4)))[0] is False
    # bad_mass up -> reject
    assert exact_signature_guard(GuardMetrics(pre=base, post=dict(base, bad_mass=0.3)))[0] is False
    # locality (drift) up -> reject
    assert exact_signature_guard(GuardMetrics(pre=base, post=dict(base, locality=0.2)))[0] is False


def test_describe_campaign_json_safe_and_gated() -> None:
    import json

    arms = build_arms(authorized_rank=8)
    desc = describe_campaign(arms, decision="no_safe_direction")
    json.dumps(desc)
    assert desc["trainable_permitted"] is False
    assert desc["authorization"] == "no_safe_direction"
    assert describe_campaign(arms, decision="authorized")["trainable_permitted"] is True


def test_invalid_status_rejected() -> None:
    with pytest.raises(ValueError):
        ArmResult("z", "promoted", "nope")  # type: ignore[arg-type]
