"""Tests for DecisionEventV2 (LDI0-02).

Contract: ``docs/design/local-decision-interventions.md``. These exercise the
E284-blocker fix — a decision state's identity is separated from its per-action
verifier evidence and from any materialized objective view — plus the fail-closed
validation, append-only evidence merge, honest V1 migration, and the
non-trainable constraint-shadow materializer.
"""

from __future__ import annotations

import pytest

from slm_training.harnesses.preference.counterfactuals import _METRICS
from slm_training.harnesses.preference.decision_events_v2 import (
    GATE_ORDER,
    ActionOutcomeV2,
    DecisionStateV2,
    ObjectiveView,
    admit_semantic_corpus,
    append_action_outcomes,
    decision_state_manifest,
    materialize_constraint_shadow,
    materialize_pareto,
    materialize_set_valued,
    materialize_single_best_worst,
    materialize_thresholded,
    migrate_v1_event,
    objective_view_support,
    write_action_outcomes,
)
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    split_for_group,
)

GROUP = "ldi0-02-group"
GATE_VECTOR = tuple((gate, "skip") for gate in GATE_ORDER)


def _state(**overrides: object) -> DecisionStateV2:
    kwargs: dict[str, object] = {
        "group_id": GROUP,
        "architecture": "twotower",
        "context_text": "root=Stack([",
        "canvas_ids": (1, 2, 3),
        "decision_position": 1,
        "legal_action_ids": (4, 9, 10),
        "decision_kind": "component",
        "abstract_state_role": "component_slot",
        "grammar_state_hash": "gsh",
        "policy_checkpoint_sha": "pcs",
        "tokenizer_sha": "tsha",
        "decode_config_hash": "dch",
        "verifier_bundle_hash": "vbh",
        "split": split_for_group(GROUP),
    }
    kwargs.update(overrides)
    return DecisionStateV2(**kwargs)  # type: ignore[arg-type]


def _outcome(action_id: int, *, reward: float = 0.5, **overrides: object) -> ActionOutcomeV2:
    kwargs: dict[str, object] = {
        "state_id": _state().state_id,
        "action_id": action_id,
        "legal": True,
        "rollout_policy_sha": "pcs",
        "continuation_seeds": (0,),
        "verifier_vectors": (GATE_VECTOR,),
        "reward_vectors": (tuple((metric, reward) for metric in _METRICS),),
        "evidence_ids": (f"probe:{action_id}",),
        "evidence_confidence": 0.9,
    }
    kwargs.update(overrides)
    return ActionOutcomeV2(**kwargs)  # type: ignore[arg-type]


def _v1_event(**overrides: object) -> DecisionEventV1:
    kwargs: dict[str, object] = {
        "event_id": "e1",
        "group_id": GROUP,
        "context_text": "root=Stack([",
        "canvas_ids": (1, 2, 3),
        "position": 1,
        "good_token_ids": (4,),
        "bad_token_ids": (9,),
        "legal_token_ids": (4, 9, 10),
        "evidence_kind": "counterfactual",
        "evidence_confidence": 0.9,
        "decision_kind": "component",
        "split": split_for_group(GROUP),
        "policy_checkpoint_sha": "pcs",
        "tokenizer_sha": "tsha",
        "decode_config_hash": "dch",
        "seed": 0,
        "trajectory_id": "traj-1",
    }
    kwargs.update(overrides)
    return DecisionEventV1(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# State identity is separate from action evidence (the E284 fix).
# --------------------------------------------------------------------------- #
def test_state_id_is_independent_of_action_evidence_order_and_augmentation() -> None:
    state = _state()
    a = [_outcome(4), _outcome(9)]
    b = [_outcome(9), _outcome(4), _outcome(4)]  # reordered + duplicated
    # The state id depends only on the state; evidence never touches it.
    assert _state().state_id == state.state_id
    merged_a = append_action_outcomes((), a)
    merged_b = append_action_outcomes((), b)
    assert {o.action_id for o in merged_a} == {o.action_id for o in merged_b}
    assert state.state_id == _state().state_id  # unchanged by any evidence handling


def test_two_label_samples_of_one_state_share_one_state_id() -> None:
    # Same exact state, different sampled good/bad labels -> one state row.
    assert _state().state_id == _state().state_id
    # A different legal set is a different state.
    assert _state(legal_action_ids=(4, 9)).state_id != _state().state_id


def test_state_id_changes_for_every_hard_state_field() -> None:
    base = _state().state_id
    assert _state(context_text="other").state_id != base
    assert _state(decision_position=2).state_id != base
    assert _state(decision_kind="bind").state_id != base
    assert _state(policy_checkpoint_sha="other").state_id != base
    assert _state(verifier_bundle_hash="other").state_id != base
    assert _state(grammar_state_hash="other").state_id != base


def test_state_round_trips_and_rejects_tampered_id() -> None:
    state = _state()
    assert DecisionStateV2.from_dict(state.to_dict()) == state
    data = state.to_dict()
    data["state_id"] = "deadbeef"
    with pytest.raises(ValueError, match="tampered"):
        DecisionStateV2.from_dict(data)


# --------------------------------------------------------------------------- #
# Fail-closed validation.
# --------------------------------------------------------------------------- #
def test_state_rejects_split_not_derived_from_group() -> None:
    other = "train" if split_for_group(GROUP) == "held_out" else "held_out"
    with pytest.raises(ValueError, match="split must be derived"):
        _state(split=other)


def test_state_rejects_unknown_fields() -> None:
    data = _state().to_dict()
    data["surprise"] = 1
    with pytest.raises(ValueError, match="unknown decision state fields"):
        DecisionStateV2.from_dict(data)


def test_causal_state_requires_prefix_and_twotower_requires_canvas() -> None:
    with pytest.raises(ValueError, match="causal state requires context_ids"):
        _state(architecture="causal", canvas_ids=None, context_ids=None, decision_position=3)
    with pytest.raises(ValueError, match="twotower state requires canvas_ids"):
        _state(canvas_ids=None)
    # A valid causal state keys off the prefix.
    causal = _state(architecture="causal", canvas_ids=None, context_ids=(1, 2, 3), decision_position=3)
    assert causal.state_id != _state().state_id


def test_action_outcome_requires_complete_ordered_gate_vector() -> None:
    partial = GATE_VECTOR[:5]
    with pytest.raises(ValueError, match="complete ordered G0-G12"):
        _outcome(4, verifier_vectors=(partial,))
    with pytest.raises(ValueError, match="gate status"):
        _outcome(4, verifier_vectors=(tuple((gate, "maybe") for gate in GATE_ORDER),))


def test_action_outcome_round_trips() -> None:
    outcome = _outcome(4)
    assert ActionOutcomeV2.from_dict(outcome.to_dict()) == outcome


# --------------------------------------------------------------------------- #
# Append-only evidence merge.
# --------------------------------------------------------------------------- #
def test_append_only_merge_dedups_by_content_not_order(tmp_path) -> None:
    first = [_outcome(4)]
    # Re-observing the identical evidence must collapse; new evidence appends.
    second = [_outcome(4), _outcome(9, reward=0.2)]
    merged = append_action_outcomes(first, second)
    assert [o.action_id for o in merged] == [4, 9]
    path = tmp_path / "outcomes.jsonl"
    assert write_action_outcomes(path, merged) == 2


# --------------------------------------------------------------------------- #
# Objective materialization.
# --------------------------------------------------------------------------- #
def test_pareto_view_is_trainable_and_partitions_actions() -> None:
    state = _state()
    good = _outcome(4, reward=0.9)
    bad = _outcome(9, reward=0.1)
    view = materialize_pareto(state, [good, bad])
    assert view.trainable is True
    assert view.good_action_ids == (4,)
    assert view.bad_action_ids == (9,)
    # Legal action 10 was never observed.
    assert 10 in view.unobserved_action_ids


def test_constraint_shadow_view_is_not_trainable() -> None:
    state = _state()
    view = materialize_constraint_shadow(state, [_outcome(4), _outcome(9)])
    assert view.trainable is False
    assert view.good_action_ids == ()
    assert view.bad_action_ids == ()
    assert set(view.ambiguous_action_ids) == {4, 9}


def test_materializer_rejects_foreign_state_outcome() -> None:
    other_state = _state(context_text="different")
    foreign = _outcome(4, state_id=other_state.state_id)
    with pytest.raises(ValueError, match="does not belong to this state"):
        materialize_pareto(_state(), [foreign])


def test_semantic_materializer_rejects_illegal_or_out_of_domain_outcome() -> None:
    state = _state()
    # An illegal outcome must never become a semantic good/bad label.
    with pytest.raises(ValueError, match="requires a legal action outcome"):
        materialize_pareto(state, [_outcome(4, legal=False)])
    # An action outside the state's legal set is rejected before materialization.
    with pytest.raises(ValueError, match="outside the state's legal set"):
        materialize_thresholded(state, [_outcome(99)])


# --------------------------------------------------------------------------- #
# V1 migration is honest about incompleteness.
# --------------------------------------------------------------------------- #
def test_v1_counterfactual_migrates_with_partial_incomplete_evidence() -> None:
    migrated = migrate_v1_event(_v1_event())
    assert migrated.complete is False
    assert migrated.state.architecture == "twotower"
    assert migrated.state.split == split_for_group(GROUP)
    actions = {o.action_id for o in migrated.outcomes}
    assert actions == {4, 9}
    # No replayable verifier/reward evidence may be fabricated from V1.
    assert all(o.verifier_vectors == () for o in migrated.outcomes)
    assert all(o.reward_vectors == () for o in migrated.outcomes)


def test_v1_constraint_shadow_migrates_as_legality_diagnostic() -> None:
    migrated = migrate_v1_event(_v1_event(evidence_kind="constraint_shadow"))
    assert migrated.complete is False
    assert "legality diagnostic" in migrated.note


def test_v1_migration_is_idempotent() -> None:
    event = _v1_event()
    assert migrate_v1_event(event) == migrate_v1_event(event)


# --------------------------------------------------------------------------- #
# Remaining materializer views.
# --------------------------------------------------------------------------- #
def test_thresholded_view_respects_threshold_and_confidence() -> None:
    state = _state()
    high = _outcome(4, reward=0.8)
    low = _outcome(9, reward=0.2)
    unsure = _outcome(10, reward=0.9, evidence_confidence=0.1)
    view = materialize_thresholded(
        state, [high, low, unsure], metric="reward", threshold=0.5, min_confidence=0.5
    )
    assert view.good_action_ids == (4,)
    assert view.bad_action_ids == (9,)
    assert view.ambiguous_action_ids == (10,)  # below the confidence floor
    assert view.trainable is True


def test_single_best_worst_view() -> None:
    state = _state()
    view = materialize_single_best_worst(
        state, [_outcome(4, reward=0.9), _outcome(9, reward=0.1)], metric="reward"
    )
    assert view.good_action_ids == (4,)
    assert view.bad_action_ids == (9,)


def test_set_valued_view_uses_verifier_verdicts() -> None:
    state = _state()
    passing = _outcome(4)  # all-skip gate vector -> no failing gate -> verified
    failing = _outcome(
        9, verifier_vectors=(tuple((gate, "fail" if gate == "G2" else "skip") for gate in GATE_ORDER),)
    )
    view = materialize_set_valued(state, [passing, failing])
    assert view.good_action_ids == (4,)
    assert view.bad_action_ids == (9,)
    assert view.trainable is True


# --------------------------------------------------------------------------- #
# Manifest separately fingerprints states / evidence / views.
# --------------------------------------------------------------------------- #
def test_manifest_fingerprints_are_separate_and_order_independent() -> None:
    state = _state()
    outcomes = [_outcome(4), _outcome(9)]
    view = materialize_pareto(state, outcomes)
    manifest = decision_state_manifest([state], outcomes, [view])
    reordered = decision_state_manifest([state], list(reversed(outcomes)), [view])
    # Row order does not change the fingerprint (evidence is content-sorted).
    assert manifest["outcome_fingerprint"] == reordered["outcome_fingerprint"]
    # The three concerns fingerprint independently.
    assert manifest["state_fingerprint"] != manifest["outcome_fingerprint"]
    assert manifest["view_fingerprint"] != manifest["state_fingerprint"]


# --------------------------------------------------------------------------- #
# Objective-support admission (the E284 fix).
# --------------------------------------------------------------------------- #
def _split_group(split: str) -> str:
    index = 0
    while split_for_group(f"obj{index}") != split:
        index += 1
    return f"obj{index}"


def _state_for(group: str) -> DecisionStateV2:
    return _state(group_id=group, split=split_for_group(group))


def _view(
    good: tuple[int, ...],
    bad: tuple[int, ...],
    *,
    materializer_id: str = "pareto_v2",
    trainable: bool = True,
) -> ObjectiveView:
    return ObjectiveView(
        good_action_ids=good,
        bad_action_ids=bad,
        ambiguous_action_ids=(),
        unobserved_action_ids=(),
        weights=(),
        materializer_id=materializer_id,
        materializer_config_hash="cfg",
        trainable=trainable,
    )


def test_objective_support_gap_matches_e284_pattern() -> None:
    train_state = _state_for(_split_group("train"))
    held_state = _state_for(_split_group("held_out"))
    # State support (good-only) matches, but objective support (good+bad) differs,
    # so a held-out objective signature is uncovered — exactly the E284 mechanism.
    train_view = _view((4,), (9,))
    held_view = _view((4,), (10,))
    report = objective_view_support([(train_state, train_view), (held_state, held_view)])
    assert report["held_out_coverage"]["passed"] is False
    assert len(report["held_out_coverage"]["uncovered"]) == 1
    with pytest.raises(ValueError, match="lacks train support"):
        admit_semantic_corpus(
            [(train_state, train_view), (held_state, held_view)], materializer_id="pareto_v2"
        )


def test_objective_support_passes_when_held_out_signature_is_covered() -> None:
    view = _view((4,), (9,))
    report = admit_semantic_corpus(
        [(_state_for(_split_group("train")), view), (_state_for(_split_group("held_out")), view)],
        materializer_id="pareto_v2",
    )
    assert report["held_out_coverage"]["passed"] is True


def test_admission_refuses_non_trainable_constraint_shadow() -> None:
    shadow = _view((), (), materializer_id="constraint_shadow_diagnostic_v2", trainable=False)
    with pytest.raises(ValueError, match="non-trainable"):
        admit_semantic_corpus(
            [(_state_for(_split_group("train")), shadow)], materializer_id="pareto_v2"
        )


def test_admission_refuses_materializer_mismatch() -> None:
    view = _view((4,), (9,), materializer_id="thresholded_v2")
    with pytest.raises(ValueError, match="do not match the requested"):
        admit_semantic_corpus(
            [(_state_for(_split_group("train")), view)], materializer_id="pareto_v2"
        )
