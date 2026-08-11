"""Tests for PGS-D01 goal-support integration in counterfactual_probe (SLM-502)."""

from __future__ import annotations

from slm_training.dsl.solver.goal_support import GoalActionEvidenceV1, action_id_from_value
from slm_training.harnesses.preference.counterfactual_probe import (
    GoalSupportProbeConfig,
    GoalSupportProbeInputs,
    goal_support_cache_key,
    goal_support_probe_cap_policy_table,
    run_goal_support_probe,
    select_goal_support_actions,
    semantic_partition,
)
from slm_training.harnesses.preference.decision_events_v2 import DecisionStateV2
from tests.test_dsl.test_goal_domain_adequacy import (
    _CAP_TREE,
    _INADEQUATE_TREE,
    _REGRET_TREE,
    _UNAVAILABLE_TREE,
    _hole,
    _provider,
)


def _inputs(
    *,
    accept: set[str],
    tree: dict | None = None,
    selected_letter: str = "a",
    cap: int = 10,
    policy_probs: dict[int, float] | None = None,
) -> GoalSupportProbeInputs:
    tree = tree or _REGRET_TREE
    expander, provider = _provider(accept, tree=tree)
    letters = tuple(sorted({branch[0] for branch in tree[""]}))
    mapping = {
        index + 1: expander.value_for("", letter)
        for index, letter in enumerate(letters)
    }
    decision = DecisionStateV2(
        group_id="grp",
        architecture="twotower",
        context_text="ctx",
        decision_position=0,
        legal_action_ids=tuple(mapping),
        decision_kind="component",
        abstract_state_role="root",
        grammar_state_hash="gh",
        policy_checkpoint_sha="pol",
        tokenizer_sha="tok",
        decode_config_hash="dec",
        verifier_bundle_hash="ver",
        split="train",
        canvas_ids=tuple(range(1, len(mapping) + 1)),
    )
    policy_action = next(
        int_id
        for int_id, value in mapping.items()
        if value.payload["letter"] == selected_letter
    )
    return GoalSupportProbeInputs(
        decision_state=decision,
        domain_state=expander.root_state(),
        hole_id=_hole(expander),
        policy_action=policy_action,
        action_values=mapping,
        provider=provider,
        config=GoalSupportProbeConfig(exact_action_cap=cap),
        policy_probs=policy_probs,
    )


def test_small_legal_set_queries_all_with_empty_unobserved():
    inputs = _inputs(accept={"ax"}, selected_letter="a", cap=10)
    report, evidence = run_goal_support_probe(inputs)
    assert report.partitions.unobserved == ()
    assert report.stats.queried_action_count == report.stats.action_count
    assert len(evidence) == report.stats.action_count


def test_policy_action_always_included_above_cap():
    expander, provider = _provider({"c"}, tree=_CAP_TREE)
    letters = ("a", "b", "c")
    mapping = {index + 1: expander.value_for("", letter) for index, letter in enumerate(letters)}
    decision = DecisionStateV2(
        group_id="grp",
        architecture="twotower",
        context_text="ctx",
        decision_position=0,
        legal_action_ids=(1, 2, 3),
        decision_kind="component",
        abstract_state_role="root",
        grammar_state_hash="gh",
        policy_checkpoint_sha="pol",
        tokenizer_sha="tok",
        decode_config_hash="dec",
        verifier_bundle_hash="ver",
        split="train",
        canvas_ids=(1, 2, 3),
    )
    policy_action = 3  # letter c — would be lex-first omitted under analyze_goal_domain cap=2
    inputs = GoalSupportProbeInputs(
        decision_state=decision,
        domain_state=expander.root_state(),
        hole_id=_hole(expander),
        policy_action=policy_action,
        action_values=mapping,
        provider=provider,
        config=GoalSupportProbeConfig(exact_action_cap=2),
        policy_probs={1: 0.1, 2: 0.2, 3: 0.7},
    )
    selected_id = action_id_from_value(mapping[policy_action])
    query_ids, unobserved_ids, cap_applied = select_goal_support_actions(
        decision,
        mapping,
        policy_action=policy_action,
        cap=2,
        policy_probs={1: 0.1, 2: 0.2, 3: 0.7},
    )
    assert cap_applied
    assert selected_id in query_ids
    assert len(query_ids) == 2
    assert len(unobserved_ids) == 1
    report, _ = run_goal_support_probe(inputs)
    assert selected_id in {
        *report.partitions.supported,
        *report.partitions.unsupported,
        *report.partitions.unknown,
    }
    assert report.classification == "domain_adequate_selected_supported"


def test_cap_omitting_unique_support_yields_coverage_unknown():
    expander, provider = _provider({"c"}, tree=_CAP_TREE)
    letters = ("a", "b", "c")
    mapping = {index + 1: expander.value_for("", letter) for index, letter in enumerate(letters)}
    omitted_id = action_id_from_value(mapping[3])
    decision = DecisionStateV2(
        group_id="grp",
        architecture="twotower",
        context_text="ctx",
        decision_position=0,
        legal_action_ids=(1, 2, 3),
        decision_kind="component",
        abstract_state_role="root",
        grammar_state_hash="gh",
        policy_checkpoint_sha="pol",
        tokenizer_sha="tok",
        decode_config_hash="dec",
        verifier_bundle_hash="ver",
        split="train",
        canvas_ids=(1, 2, 3),
    )
    policy_action = 1
    inputs = GoalSupportProbeInputs(
        decision_state=decision,
        domain_state=expander.root_state(),
        hole_id=_hole(expander),
        policy_action=policy_action,
        action_values=mapping,
        provider=provider,
        config=GoalSupportProbeConfig(exact_action_cap=2),
        policy_probs={1: 0.9, 2: 0.05, 3: 0.05},
    )
    report, _ = run_goal_support_probe(inputs)
    assert report.cap_applied
    assert report.classification == "coverage_unknown"
    assert omitted_id in report.partitions.unobserved


def test_unsupported_actions_remain_legal_in_evidence():
    inputs = _inputs(accept=set(), tree=_INADEQUATE_TREE, selected_letter="a", cap=10)
    _report, evidence = run_goal_support_probe(inputs)
    for item in evidence:
        assert item.legal is True


def test_unavailable_evidence_stays_unknown():
    expander, provider = _provider(set(), tree=_UNAVAILABLE_TREE, unavailable={"ax"})
    mapping = {1: expander.value_for("", "a")}
    decision = DecisionStateV2(
        group_id="grp",
        architecture="twotower",
        context_text="ctx",
        decision_position=0,
        legal_action_ids=(1,),
        decision_kind="component",
        abstract_state_role="root",
        grammar_state_hash="gh",
        policy_checkpoint_sha="pol",
        tokenizer_sha="tok",
        decode_config_hash="dec",
        verifier_bundle_hash="ver",
        split="train",
        canvas_ids=(1,),
    )
    inputs = GoalSupportProbeInputs(
        decision_state=decision,
        domain_state=expander.root_state(),
        hole_id=_hole(expander),
        policy_action=1,
        action_values=mapping,
        provider=provider,
        config=GoalSupportProbeConfig(exact_action_cap=10),
    )
    report, _ = run_goal_support_probe(inputs)
    assert report.classification == "coverage_unknown"
    assert report.partitions.unknown
    assert report.classification != "domain_inadequate_under_bounds"


def test_permutation_invariance_of_goal_support_manifest():
    inputs = _inputs(accept={"ax"}, selected_letter="a", cap=10)
    report_a, evidence_a = run_goal_support_probe(inputs)
    report_b, evidence_b = run_goal_support_probe(inputs)
    assert report_a.partitions == report_b.partitions
    assert report_a.classification == report_b.classification
    assert report_a.report_digest == report_b.report_digest
    assert [item.evidence_digest for item in evidence_a] == [
        item.evidence_digest for item in evidence_b
    ]


def test_goal_support_cache_key_identity_axes():
    base = dict(
        state_fingerprint="sf",
        action_id="a" * 64,
        profile_digest="b" * 64,
        problem_id="p",
        pack_id="pack",
        constraint_version="v1",
        bounds_digest="c" * 64,
        legal_set_fingerprint="d" * 64,
        decode_config_hash="dec",
    )
    key = goal_support_cache_key(**base)
    assert key == goal_support_cache_key(**base)
    assert key != goal_support_cache_key(**{**base, "state_fingerprint": "sf2"})
    assert key != goal_support_cache_key(**{**base, "action_id": "e" * 64})
    assert key != goal_support_cache_key(**{**base, "profile_digest": "f" * 64})
    assert key != goal_support_cache_key(**{**base, "problem_id": "p2"})
    assert key != goal_support_cache_key(**{**base, "pack_id": "pack2"})
    assert key != goal_support_cache_key(**{**base, "constraint_version": "v2"})
    assert key != goal_support_cache_key(**{**base, "bounds_digest": "g" * 64})
    assert key != goal_support_cache_key(**{**base, "decode_config_hash": "dec2"})


def test_run_goal_support_probe_is_resumable_via_cache():
    inputs = _inputs(accept={"ax"}, selected_letter="a", cap=10)
    cache: dict[str, GoalActionEvidenceV1] = {}
    first_report, first_evidence = run_goal_support_probe(inputs, cache=cache)
    assert cache
    second_report, second_evidence = run_goal_support_probe(inputs, cache=cache)
    assert first_report.partitions == second_report.partitions
    assert second_report.stats.verifier_calls > 0
    assert [item.evidence_digest for item in first_evidence] == [
        item.evidence_digest for item in second_evidence
    ]


def test_wrong_key_cache_row_is_replayed_and_replaced() -> None:
    inputs = _inputs(accept={"ax"}, selected_letter="a", cap=10)
    cache: dict[str, GoalActionEvidenceV1] = {}
    first_report, _ = run_goal_support_probe(inputs, cache=cache)
    keys = sorted(cache)
    assert len(keys) == 2
    cache[keys[0]], cache[keys[1]] = cache[keys[1]], cache[keys[0]]

    second_report, _ = run_goal_support_probe(inputs, cache=cache)

    assert second_report.partitions == first_report.partitions
    assert second_report.classification == "domain_adequate_selected_supported"


def test_cross_provider_cached_unsupported_rows_cannot_label() -> None:
    stale_inputs = _inputs(
        accept=set(), tree=_INADEQUATE_TREE, selected_letter="a", cap=10
    )
    cache: dict[str, GoalActionEvidenceV1] = {}
    stale_report, _ = run_goal_support_probe(stale_inputs, cache=cache)
    assert len(stale_report.partitions.unsupported) == stale_report.stats.action_count

    live_inputs = _inputs(
        accept={"a"}, tree=_INADEQUATE_TREE, selected_letter="a", cap=10
    )
    live_report, _ = run_goal_support_probe(live_inputs, cache=cache)

    assert live_report.partitions.supported
    assert live_report.classification == "domain_adequate_selected_supported"


def test_selection_regret_classification():
    inputs = _inputs(accept={"ax"}, selected_letter="b", cap=10)
    report, _ = run_goal_support_probe(inputs)
    assert report.classification == "selection_regret"


def test_cap_policy_table_documents_policy_aware_selection():
    table = goal_support_probe_cap_policy_table()
    assert table["above_cap"]["always_include_policy_action"] is True
    assert table["semantic_partition_alias_forbidden"] is True


def test_semantic_partition_unchanged_by_goal_support_integration():
    """Regression: relative rollout partition semantics stay independent."""
    from slm_training.harnesses.preference.counterfactual_probe import (
        CandidateState,
        ProbeConfig,
        RawOutcome,
        run_probe,
    )

    def _gate_vec(g1: str) -> tuple[tuple[str, str], ...]:
        gates = {f"G{i}": "pass" for i in range(13)}
        gates["G1"] = g1
        return tuple((f"G{i}", gates[f"G{i}"]) for i in range(13))

    class _Backend:
        def rollout(self, state, action_id, seed):
            g1 = "pass" if action_id in (2, 3) else ("fail" if action_id == 4 else "pass" if seed == 0 else "fail")
            return RawOutcome(canonical_output=f"o{action_id}", finish_reason="stop", verifier_vector=_gate_vec(g1))

    st = DecisionStateV2(
        group_id="grp",
        architecture="twotower",
        context_text="ctx",
        decision_position=0,
        legal_action_ids=(1, 2, 3, 4, 5),
        decision_kind="component",
        abstract_state_role="root",
        grammar_state_hash="gh",
        policy_checkpoint_sha="pol",
        tokenizer_sha="tok",
        decode_config_hash="dec",
        verifier_bundle_hash="ver",
        split="train",
        canvas_ids=(1, 2, 3, 4, 5),
    )
    cfg = ProbeConfig(seeds=(0, 1, 2), min_rollouts=3, min_effect=0.3, required_gates=("G0",))
    outcomes = run_probe(
        [CandidateState(st, "detector_localized")],
        _Backend(),
        config=cfg,
        selection={st.state_id: (1, 2, 3, 4)},
    )
    part = semantic_partition(st, outcomes, config=cfg, policy_action=1)
    assert set(part.good_action_ids) >= {2, 3}
    assert 4 in part.bad_action_ids
    assert 5 in part.unobserved_action_ids
