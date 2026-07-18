"""DecisionEventV2 (LDI0-02 / SLM-116) tests.

Exercises the V2 contract directly: state identity independent of action
evidence/order, append-only content-dedup merge, fail-closed validation,
materializer identity + the non-semantic constraint-shadow guard, one-way V1
migration, causal/TwoTower replay sufficiency, and separately-fingerprinted
manifests.
"""

from __future__ import annotations

import pytest

from slm_training.harnesses.preference.decision_events_v2 import (
    PARETO_METRICS,
    ActionOutcomeV2,
    DecisionStateV2,
    assert_semantic_trainable,
    check_split_homogeneity,
    decision_v2_manifest,
    load_action_outcomes,
    load_decision_states,
    materialize,
    merge_action_evidence,
    migrate_v1_event,
    validate_state_action_table,
    write_action_outcomes,
    write_decision_states,
)
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    split_for_group,
)


def _held_out_group() -> str:
    g = "held"
    while split_for_group(g) != "held_out":
        g += "x"
    return g


def make_state(group="grpA", arch="twotower", legal=(3, 4, 5), **kw) -> DecisionStateV2:
    base = dict(
        group_id=group,
        architecture=arch,
        context_text="ctx",
        context_ids=(1, 2) if arch == "causal" else None,
        canvas_ids=(1, 2, 3) if arch == "twotower" else None,
        decision_position=1,
        generation_step=0,
        legal_action_ids=legal,
        decision_kind="slot",
        abstract_state_role="counterfactual",
        grammar_state_hash="g",
        policy_checkpoint_sha="ckpt",
        tokenizer_sha="tok",
        decode_config_hash="dch",
        verifier_bundle_hash="vb",
        split=split_for_group(group),
    )
    base.update(kw)
    return DecisionStateV2(**base)


def make_outcome(state, action, *, legal=True, verified=True, value=1.0):
    m = 1.0 if verified else 0.0
    return ActionOutcomeV2(
        state_id=state.state_id,
        action_id=action,
        legal=legal,
        rollout_policy_sha="ckpt",
        continuation_seeds=(0,),
        outcome_hashes=("h0",),
        verifier_vectors=({"gate": "G0", "ok": verified}, {"gate": "G1", "ok": verified}),
        reward_vectors=({name: m for name in PARETO_METRICS},),
        mean_value=value,
        evidence_ids=("e",),
        evidence_confidence=1.0,
    )


# 1. canonical hashing + order/evidence independence
def test_state_id_independent_of_order_and_evidence():
    a = make_state(legal=(5, 3, 4))
    b = make_state(legal=(3, 4, 5))  # different declared order
    assert a.state_id == b.state_id
    # Recomputing from the stored fields is stable.
    assert a.state_id == a.canonical_state_id()
    # state_id has no dependency on action evidence — it is a state-only hash.
    assert "good" not in a.to_dict() and "bad" not in a.to_dict()


# 2. append-only evidence merge, order-independent dedup
def test_append_only_merge_dedups_by_content():
    s = make_state()
    a = make_outcome(s, 3)
    b = make_outcome(s, 4)
    merged1 = merge_action_evidence([a, b, a])
    merged2 = merge_action_evidence([b, a, a])
    assert len(merged1) == 2
    assert {o.content_id() for o in merged1} == {o.content_id() for o in merged2}


# 3. conflicting state metadata: bad state_id and split straddle
def test_wrong_state_id_rejected():
    with pytest.raises(ValueError, match="canonical hash"):
        make_state(state_id="deadbeef")


def test_same_state_two_splits_rejected():
    # Identical model-state fields, different group_id → same state_id, but the
    # two groups fall on different splits: a leak.
    train = make_state(group="grpTrain")
    assert train.split == "train"
    held = make_state(group=_held_out_group())
    assert held.state_id == train.state_id and held.split == "held_out"
    with pytest.raises(ValueError, match="both"):
        check_split_homogeneity([train, held])


# 4. missing runtime identity fails closed
def test_missing_verifier_bundle_rejected():
    with pytest.raises(ValueError, match="verifier_bundle_hash"):
        make_state(verifier_bundle_hash="")


# 5. action outside legal set (semantic outcome) fails closed
def test_legal_outcome_outside_legal_set_rejected():
    s = make_state(legal=(3, 4, 5))
    bad = make_outcome(s, 9, legal=True)
    with pytest.raises(ValueError, match="outside the declared legal set"):
        validate_state_action_table(s, [bad])
    with pytest.raises(ValueError, match="outside the declared legal set"):
        materialize(s, [bad], "set_valued_v1")


# 6. materializer version/config identity
def test_materializer_identity_reflects_config():
    s = make_state()
    outs = [make_outcome(s, 3), make_outcome(s, 4)]
    v_default = materialize(s, outs, "pareto_pass_fail_v1")
    v_default2 = materialize(s, outs, "pareto_pass_fail_v1")
    v_configured = materialize(s, outs, "pareto_pass_fail_v1", {"tie_break": "first"})
    assert v_default.materializer_id == "pareto_pass_fail_v1"
    assert v_default.materializer_config_hash == v_default2.materializer_config_hash
    assert v_default.materializer_config_hash != v_configured.materializer_config_hash


# 7. multi-good/multi-bad + unresolved (unobserved) actions
def test_set_valued_multi_good_bad_and_unobserved():
    s = make_state(legal=(3, 4, 5, 6))
    outs = [
        make_outcome(s, 3, verified=True),
        make_outcome(s, 4, verified=True),
        make_outcome(s, 5, verified=False),
    ]
    view = materialize(s, outs, "set_valued_v1")
    assert view.good_action_ids == (3, 4)
    assert view.bad_action_ids == (5,)
    assert view.unobserved_action_ids == (6,)  # legal but never rolled out


# 8. V1 semantic + constraint-shadow migration (one-way, non-mutating, incomplete)
def test_v1_migration_marks_partial_evidence():
    g = "grpMig"
    cf = DecisionEventV1(
        event_id="cf1",
        group_id=g,
        context_text="c",
        canvas_ids=(1, 2, 3),
        position=1,
        good_token_ids=(3,),
        bad_token_ids=(4,),
        legal_token_ids=(3, 4, 5),
        evidence_kind="counterfactual",
        evidence_confidence=0.9,
        decision_kind="slot",
        split=split_for_group(g),
        policy_checkpoint_sha="ckpt",
        tokenizer_sha="tok",
        decode_config_hash="dch",
        seed=0,
        trajectory_id="t",
    )
    before = cf.to_dict()
    state, outcomes = migrate_v1_event(cf)
    assert cf.to_dict() == before  # migration does not mutate V1
    assert state.architecture == "twotower"
    assert all(o.migrated_incomplete for o in outcomes)  # no fabricated rollouts
    # Constraint shadow migrates to a non-semantic legality diagnostic.
    shadow = DecisionEventV1(
        event_id="sh1",
        group_id=g,
        context_text="c",
        canvas_ids=(1, 2, 3),
        position=1,
        good_token_ids=(3,),
        bad_token_ids=(9,),
        legal_token_ids=(3, 4, 5),
        evidence_kind="constraint_shadow",
        evidence_confidence=1.0,
        decision_kind="constraint_shadow",
        split=split_for_group(g),
        policy_checkpoint_sha="ckpt",
        tokenizer_sha="tok",
        decode_config_hash="dch",
        seed=0,
        trajectory_id="t",
    )
    s2, outs2 = migrate_v1_event(shadow)
    view = materialize(s2, outs2, "constraint_shadow_v1")
    assert view.semantic is False
    with pytest.raises(ValueError, match="non-semantic"):
        assert_semantic_trainable(view)


# 9. causal + TwoTower replay sufficiency
def test_causal_requires_ids_and_replays():
    with pytest.raises(ValueError, match="causal state requires context_ids"):
        make_state(arch="causal", context_ids=None)
    causal = make_state(arch="causal")
    twotower = make_state(arch="twotower")
    for state in (causal, twotower):
        # Round-trip through dict then recompute a deterministic decision over the
        # stored replay inputs: the state alone reproduces it (no retokenization).
        restored = type(state).from_dict(state.to_dict())
        assert restored.state_id == state.state_id
        assert restored.replay_inputs() == state.replay_inputs()


def test_twotower_requires_canvas():
    with pytest.raises(ValueError, match="twotower state requires canvas_ids"):
        make_state(arch="twotower", canvas_ids=None)


# 10. manifest fingerprints: order-independent, evidence-sensitive, separate
def test_manifest_fingerprints_separate_and_stable(tmp_path):
    s = make_state()
    outs = [make_outcome(s, 3), make_outcome(s, 4)]
    m1 = decision_v2_manifest([s], outs, dataset_id="d")
    m2 = decision_v2_manifest([s], list(reversed(outs)), dataset_id="d")
    assert m1 == m2  # row order does not move any fingerprint
    assert {"states_fingerprint", "action_evidence_fingerprint", "objective_views_fingerprint"} <= m1.keys()
    m3 = decision_v2_manifest([s], outs + [make_outcome(s, 5)], dataset_id="d")
    assert m3["action_evidence_fingerprint"] != m1["action_evidence_fingerprint"]
    assert m3["states_fingerprint"] == m1["states_fingerprint"]  # states unchanged


# 11. semantic trainer guard passes a real semantic view
def test_semantic_view_passes_guard():
    s = make_state()
    view = materialize(s, [make_outcome(s, 3), make_outcome(s, 4)], "set_valued_v1")
    assert assert_semantic_trainable(view) is view


# 12. write/read round-trip preserves G0–G12 + duplicate-safe atomic writes
def test_round_trip_preserves_evidence_and_dedups(tmp_path):
    s = make_state()
    gates = tuple({"gate": f"G{i}", "ok": True} for i in range(13))  # G0–G12
    rich = ActionOutcomeV2(
        state_id=s.state_id,
        action_id=3,
        legal=True,
        rollout_policy_sha="ckpt",
        continuation_seeds=(0, 1),
        outcome_hashes=("h0", "h1"),
        verifier_vectors=gates,
        reward_vectors=({name: 1.0 for name in PARETO_METRICS},),
        mean_value=1.0,
        evidence_ids=("e",),
        evidence_confidence=1.0,
    )
    states_path = tmp_path / "states.jsonl"
    outs_path = tmp_path / "outcomes.jsonl"
    assert write_decision_states(states_path, [s, s]) == 1  # dedup by state_id
    assert write_action_outcomes(outs_path, [rich, rich]) == 1  # dedup by content
    (loaded_state,) = load_decision_states(states_path)
    (loaded_outcome,) = load_action_outcomes(outs_path)
    assert loaded_state == s
    assert loaded_outcome == rich
    assert len(loaded_outcome.verifier_vectors) == 13  # full ordered G0–G12 survives
