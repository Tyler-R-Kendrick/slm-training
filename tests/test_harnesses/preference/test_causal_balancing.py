"""Tests for LDI1-02 deterministic strata balancing."""

from __future__ import annotations

import pytest

from slm_training.harnesses.preference.causal_balancing import (
    CausalTrainingItem,
    balance_items,
)
from slm_training.harnesses.preference.decision_events_v2 import (
    DecisionStateV2,
    ObjectiveView,
)
from slm_training.harnesses.preference.local_decisions import split_for_group

GROUP = "ldi1-02-group"


def _state(tag: str, *, decision_kind: str = "component") -> DecisionStateV2:
    return DecisionStateV2(
        group_id=GROUP,
        architecture="causal",
        context_text="root=Stack([",
        context_ids=(1, 2, 3),
        decision_position=1,
        legal_action_ids=(4, 9, 10),
        decision_kind=decision_kind,
        abstract_state_role="component_slot",
        grammar_state_hash=f"gsh-{tag}",
        policy_checkpoint_sha="pcs",
        tokenizer_sha="tsha",
        decode_config_hash="dch",
        verifier_bundle_hash="vbh",
        split=split_for_group(GROUP),
    )


def _view(*, trainable: bool = True) -> ObjectiveView:
    return ObjectiveView(
        good_action_ids=(4,),
        bad_action_ids=(9,),
        ambiguous_action_ids=(),
        unobserved_action_ids=(10,),
        weights=((4, 1.0),),
        materializer_id="set_valued_v2",
        materializer_config_hash="h",
        trainable=trainable,
    )


def _item(tag: str, *, decision_kind: str = "component", trainable: bool = True, suite: str = "s") -> CausalTrainingItem:
    return CausalTrainingItem(_state(tag, decision_kind=decision_kind), _view(trainable=trainable), suite)


def test_balancing_equalizes_strata_without_duplicating_evidence() -> None:
    items = [
        _item("a", decision_kind="component"),
        _item("b", decision_kind="component"),
        _item("c", decision_kind="component"),
        _item("d", decision_kind="prop"),
    ]
    balanced, report = balance_items(items, strata=["decision_kind"], seed=0)
    assert report["cap"] == 1  # smallest stratum size
    assert report["after"] == {"component": 1, "prop": 1}
    assert report["excluded_by_cap"] == 2
    # No duplication: every kept state is distinct.
    assert len({it.state.state_id for it in balanced}) == len(balanced) == 2


def test_balancing_is_deterministic_in_seed() -> None:
    items = [_item(t, decision_kind="component") for t in "abcdef"] + [_item("z", decision_kind="prop")]
    first, _ = balance_items(items, strata=["decision_kind"], seed=7)
    second, _ = balance_items(items, strata=["decision_kind"], seed=7)
    assert [it.state.state_id for it in first] == [it.state.state_id for it in second]


def test_balancing_excludes_nontrainable_constraint_shadows() -> None:
    items = [_item("a"), _item("b", trainable=False)]
    balanced, report = balance_items(items, strata=["decision_kind"], seed=0)
    assert report["excluded_nontrainable"] == 1
    assert all(it.view.trainable for it in balanced)


def test_explicit_per_stratum_cap_reports_excess() -> None:
    items = [_item(t) for t in "abcd"]
    balanced, report = balance_items(items, strata=["decision_kind"], seed=1, per_stratum=2)
    assert report["cap"] == 2
    assert report["effective_count"] == 2
    assert report["excluded_by_cap"] == 2


def test_unknown_stratum_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown balance stratum"):
        balance_items([_item("a")], strata=["not_a_stratum"], seed=0)


def test_empty_strata_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one stratum"):
        balance_items([_item("a")], strata=[], seed=0)
