"""Tests for the LDI3-03 counterfactual action-value probe (SLM-131).

Cover admission (heuristic quarantine), legal-action selection, cache identity,
resumable rollout orchestration with a mock backend, the pure value materializers,
and the strict same-state good/bad partition (incl. a fixture with multiple good,
one bad, one ambiguous, and one unobserved action). No model or training runs.
"""

from __future__ import annotations

import pytest

from slm_training.harnesses.preference.decision_events_v2 import DecisionStateV2
from slm_training.harnesses.preference.counterfactual_probe import (
    CandidateState,
    ProbeConfig,
    RawOutcome,
    admit_states,
    binary_verdict,
    lexicographic_key,
    outcome_cache_key,
    pareto_front,
    run_probe,
    scalar_value,
    select_actions,
    semantic_partition,
)


def _state(sid: str, legal=(1, 2, 3), **kw) -> DecisionStateV2:
    base = dict(
        group_id="grp",
        architecture="twotower",
        context_text="ctx",
        decision_position=0,
        legal_action_ids=tuple(legal),
        decision_kind="component",
        abstract_state_role="root",
        grammar_state_hash="gh",
        policy_checkpoint_sha="pol",
        tokenizer_sha="tok",
        decode_config_hash="dec",
        verifier_bundle_hash="ver",
        split="train",
        canvas_ids=(1, 2, 3),
        state_id=sid,
    )
    base.update(kw)
    return DecisionStateV2(**base)


def _gate_vec(g1: str) -> tuple[tuple[str, str], ...]:
    # complete ordered G0-G12 vector, with G1 varied
    gates = {f"G{i}": "pass" for i in range(13)}
    gates["G1"] = g1
    return tuple((f"G{i}", gates[f"G{i}"]) for i in range(13))


class _Backend:
    """Deterministic mock: action 2/3 pass G1, action 4 fails, others baseline."""

    def __init__(self) -> None:
        self.calls = 0

    def rollout(self, state, action_id, seed):
        self.calls += 1
        g1 = "pass" if action_id in (2, 3) else ("fail" if action_id == 4 else "pass" if seed == 0 else "fail")
        return RawOutcome(canonical_output=f"o{action_id}", finish_reason="stop", verifier_vector=_gate_vec(g1))


def test_admission_quarantines_heuristic_and_orders_deterministically() -> None:
    cands = [
        CandidateState(_state("s1"), "heuristic_only", 5.0),
        CandidateState(_state("s2"), "immediate_verifier_failure", 1.0),
        CandidateState(_state("s3"), "detector_localized", 9.0),
    ]
    admitted, rejected = admit_states(cands)
    # ordered by -priority (state_id is a recomputed canonical hash, used only as tiebreak)
    assert [c.reason for c in admitted] == ["detector_localized", "immediate_verifier_failure"]
    assert [c.reason for c in rejected] == ["heuristic_only"]
    # allow_heuristic lets it through but callers still treat it as diagnostic.
    admitted2, rejected2 = admit_states(cands, allow_heuristic=True)
    assert len(admitted2) == 3 and not rejected2


def test_select_actions_all_within_cap_and_policy_always_included() -> None:
    st = _state("s", legal=(1, 2, 3))
    sel, excl = select_actions(st, policy_action=2, cap=8)
    assert sel == (1, 2, 3) and excl == ()
    big = _state("b", legal=tuple(range(10)))
    sel2, excl2 = select_actions(big, policy_action=7, cap=3, policy_probs={i: i / 10 for i in range(10)})
    assert 7 in sel2 and len(sel2) == 3
    assert set(sel2).isdisjoint(excl2) and set(sel2) | set(excl2) == set(range(10))
    with pytest.raises(ValueError):
        select_actions(st, policy_action=99)  # not legal


def test_cache_key_changes_with_identity() -> None:
    k = outcome_cache_key("s", 1, 0, policy_sha="p", decoder_hash="d", verifier_hash="v")
    assert k == outcome_cache_key("s", 1, 0, policy_sha="p", decoder_hash="d", verifier_hash="v")
    assert k != outcome_cache_key("s", 1, 0, policy_sha="p2", decoder_hash="d", verifier_hash="v")
    assert k != outcome_cache_key("s", 1, 0, policy_sha="p", decoder_hash="d", verifier_hash="v2")


def test_run_probe_is_resumable_via_cache() -> None:
    st = _state("s", legal=(1, 2))
    cfg = ProbeConfig(seeds=(0, 1), min_rollouts=2)
    backend = _Backend()
    cache: dict = {}
    first = run_probe([CandidateState(st, "detector_localized")], backend, config=cfg, cache=cache)
    calls_after_first = backend.calls
    assert len(first) == 2  # one ActionOutcomeV2 per legal action
    # Re-run with the populated cache: no new backend calls, identical manifest.
    second = run_probe([CandidateState(st, "detector_localized")], backend, config=cfg, cache=cache)
    assert backend.calls == calls_after_first
    assert [o.outcome_hashes for o in first] == [o.outcome_hashes for o in second]


def test_value_materializers() -> None:
    st = _state("s", legal=(1, 2, 3, 4))
    cfg = ProbeConfig(seeds=(0,), min_rollouts=1)
    outcomes = run_probe([CandidateState(st, "detector_localized")], _Backend(), config=cfg)
    by = {o.action_id: o for o in outcomes}
    # actions 2,3 pass G1 -> on the Pareto front; 4 fails.
    front = pareto_front(outcomes, ("G0", "G1"))
    assert 2 in front and 3 in front and 4 not in front
    assert scalar_value(by[2], {"G1": 1.0}) == 1.0
    assert scalar_value(by[4], {"G1": 1.0}) == 0.0
    assert lexicographic_key(by[2], ("G0", "G1")) == (1.0, 1.0)
    assert binary_verdict(by[2], ("G0", "G1")) == "pass"
    assert binary_verdict(by[4], ("G0", "G1")) == "fail"
    # a required gate absent from evidence -> unresolved, never a silent pass/fail.
    assert binary_verdict(by[2], ("G0", "G99")) == "unresolved"


def test_semantic_partition_multi_good_bad_ambiguous_unobserved() -> None:
    # legal 1..5; probe 1(policy baseline),2,3(good),4(bad); 5 never probed (unobserved).
    st = _state("s", legal=(1, 2, 3, 4, 5))
    cfg = ProbeConfig(seeds=(0, 1, 2), min_rollouts=3, min_effect=0.3, required_gates=("G0",))
    outcomes = run_probe(
        [CandidateState(st, "detector_localized")], _Backend(), config=cfg,
        selection={st.state_id: (1, 2, 3, 4)},
    )
    part = semantic_partition(st, outcomes, config=cfg, policy_action=1)
    # policy action 1 has G1 pass only on seed 0 (baseline 1/3); 2 and 3 pass all (good).
    assert set(part.good_action_ids) >= {2, 3}
    assert 4 in part.bad_action_ids
    assert 5 in part.unobserved_action_ids  # legal but never probed
    # every legal action is accounted for exactly once
    allp = set(part.good_action_ids) | set(part.bad_action_ids) | set(part.ambiguous_action_ids) | set(part.unobserved_action_ids)
    assert allp == {1, 2, 3, 4, 5}


def test_insufficient_rollouts_are_ambiguous_not_good() -> None:
    st = _state("s", legal=(1, 2))
    cfg = ProbeConfig(seeds=(0,), min_rollouts=3, min_effect=0.1)  # only 1 rollout < min
    outcomes = run_probe([CandidateState(st, "detector_localized")], _Backend(), config=cfg)
    part = semantic_partition(st, outcomes, config=cfg, policy_action=1)
    assert not part.good_action_ids  # nothing promoted on thin evidence
    assert set(part.ambiguous_action_ids) == {1, 2}
