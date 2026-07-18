"""Tests for the LDI1-03 causal adapter/objective campaign matrix (SLM-122).

These pin the orchestration invariants only — the matrix is well-formed, arms
vary only their declared levers, the eligibility gates fail closed, and no arm
fabricates a result without an executable policy + admitted corpus. No training
or quality claim is exercised.
"""

from __future__ import annotations

import pytest

from slm_training.harnesses.preference.causal_adapter_matrix import (
    ArmResult,
    CampaignConfig,
    CorpusSupport,
    build_stage0,
    build_stage1,
    build_stage2,
    classify_arm,
    describe_campaign,
    needs_replication,
    only_declared_levers_differ,
    run_arm,
)

CFG = CampaignConfig(base_model_id="pinned/base", base_model_revision="rev0")
FULL = CorpusSupport(admitted=True, has_pairs=True, has_set_valued=True, trainable_events=64)


def test_stage0_has_parent_plus_six_controls() -> None:
    arms = build_stage0(CFG)
    labels = [a.label for a in arms]
    assert labels == ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    # C0 is the untrained parent baseline.
    assert arms[0].levers.objective is None
    assert all(a.stage == 0 for a in arms)


def test_manifest_is_deterministic() -> None:
    a = describe_campaign(build_stage0(CFG))
    b = describe_campaign(build_stage0(CFG))
    assert a == b


def test_arms_differ_only_in_declared_levers() -> None:
    arms = {a.label: a for a in build_stage0(CFG)}
    c0 = arms["C0"]
    # C1 varies only its objective vs the C0 baseline.
    assert only_declared_levers_differ(c0, arms["C1"])
    # A forged arm that also flips an undeclared lever (rank) is rejected.
    forged = arms["C1"]
    forged = type(forged)(
        arm_id=forged.arm_id,
        stage=forged.stage,
        label=forged.label,
        levers=forged.levers.__class__(objective="unlikelihood", rank=999),
        declared_levers=("objective",),  # rank NOT declared
    )
    assert not only_declared_levers_differ(c0, forged)


def test_set_valued_objective_fails_closed_without_support() -> None:
    # ftpo_set (C3) must block, not silently narrow to single pairs.
    c3 = {a.label: a for a in build_stage0(CFG)}["C3"]
    pairs_only = CorpusSupport(admitted=True, has_pairs=True, has_set_valued=False)
    res = classify_arm(c3, corpus=pairs_only)
    assert res.status == "blocked"
    assert "blocked_by_corpus" in res.reason


def test_unsupported_method_is_not_supported_no_fallback() -> None:
    c1 = {a.label: a for a in build_stage0(CFG)}["C1"]
    adalora = type(c1)(
        arm_id="x/adalora",
        stage=2,
        label="method-adalora",
        levers=c1.levers.__class__(objective="unlikelihood", method="adalora"),
        declared_levers=("method",),
    )
    # Experimental method without explicit opt-in -> not_supported (never lora).
    res = classify_arm(adalora, corpus=FULL, allow_experimental_methods=False)
    assert res.status == "not_supported"
    # And a genuinely unknown method is also rejected.
    unknown = type(adalora)(
        arm_id="x/bogus",
        stage=2,
        label="method-bogus",
        levers=adalora.levers.__class__(objective="unlikelihood", method="bogus"),
        declared_levers=("method",),
    )
    assert classify_arm(unknown, corpus=FULL).status == "not_supported"


def test_unadmitted_corpus_blocks_trainable_but_not_parent() -> None:
    arms = {a.label: a for a in build_stage0(CFG)}
    empty = CorpusSupport(admitted=False)
    assert classify_arm(arms["C1"], corpus=empty).status == "blocked"
    # C0 parent needs no corpus.
    assert classify_arm(arms["C0"], corpus=empty).status == "admitted"


def test_run_arm_expires_without_policy_never_fabricates() -> None:
    c1 = {a.label: a for a in build_stage0(CFG)}["C1"]
    res = run_arm(c1, corpus=FULL, seed=0, policy_factory=None)
    assert res.status == "expired"
    assert res.metrics is None
    assert "GPU" in res.reason


def test_run_arm_propagates_classification_block() -> None:
    c3 = {a.label: a for a in build_stage0(CFG)}["C3"]
    res = run_arm(c3, corpus=CorpusSupport(admitted=True, has_pairs=True), seed=1)
    assert res.status == "blocked"


def test_stage1_matches_alpha_to_rank_across_placements() -> None:
    arms = build_stage1(CFG, best_objective="ftpo_single")
    assert len(arms) == len(set(CFG.ranks)) * 2  # ranks x {all, last_k}
    for a in arms:
        assert a.levers.alpha == a.levers.rank  # matched scaling
        assert "rank" in a.declared_levers and "alpha" in a.declared_levers
    assert build_stage1(CFG, best_objective="ftpo_single") == arms  # deterministic


def test_stage1_rejects_unsupported_best_objective() -> None:
    with pytest.raises(ValueError):
        build_stage1(CFG, best_objective="not_an_objective")


def test_stage2_sweeps_methods_only() -> None:
    arms = build_stage2(CFG, best_objective="ftpo_single", best_rank=32)
    assert [a.levers.method for a in arms] == list(CFG.methods)
    for a in arms:
        assert a.declared_levers == ("method",)


def test_needs_replication_only_for_improving_completed() -> None:
    improved = ArmResult("a", "completed", "ok", metrics={"pre_loss": 1.0, "post_loss": 0.5})
    flat = ArmResult("b", "completed", "ok", metrics={"pre_loss": 1.0, "post_loss": 1.0})
    expired = ArmResult("c", "expired", "no gpu")
    assert needs_replication(improved)
    assert not needs_replication(flat)
    assert not needs_replication(expired)


def test_invalid_status_rejected() -> None:
    with pytest.raises(ValueError):
        ArmResult("z", "promoted", "nope")  # type: ignore[arg-type]


def test_describe_campaign_is_json_safe_and_counts() -> None:
    import json

    desc = describe_campaign(build_stage0(CFG))
    json.dumps(desc)  # must not raise
    assert desc["arm_count"] == 7
    assert desc["arms_by_stage"]["0"] == 7
    assert "no quality claim" in desc["claim"]
