"""Tests for PGS-D02 goal-support ObjectiveView materialization (SLM-503)."""

from __future__ import annotations

import pytest

from slm_training.dsl.solver.goal_support import (
    GoalActionEvidenceV1,
    GoalDomainAdequacyReportV1,
    action_id_from_value,
)
from slm_training.harnesses.preference.counterfactual_probe import (
    GoalSupportProbeConfig,
    GoalSupportProbeInputs,
    run_goal_support_probe,
)
from slm_training.harnesses.preference.decision_events_v2 import (
    GOAL_SUPPORT_MATERIALIZER_ID,
    DecisionStateV2,
    GoalSupportStateBinding,
    admit_semantic_corpus,
    goal_support_materializer_config_hash,
    goal_support_trainability_table,
    guard_goal_support_view,
    materialize_goal_support,
)
from slm_training.harnesses.preference.local_decisions import split_for_group
from tests.test_dsl.test_goal_domain_adequacy import (
    _REGRET_TREE,
    _hole,
    _provider,
)


def _probe_bundle(*, accept: set[str], selected_letter: str = "a"):
    expander, provider = _provider(accept, tree=_REGRET_TREE)
    letters = tuple(sorted({branch[0] for branch in _REGRET_TREE[""]}))
    mapping = {
        index + 1: expander.value_for("", letter)
        for index, letter in enumerate(letters)
    }
    decision = DecisionStateV2(
        group_id="pgs-d02",
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
        split=split_for_group("pgs-d02"),
        canvas_ids=tuple(range(1, len(mapping) + 1)),
    )
    policy_action = next(
        int_id
        for int_id, value in mapping.items()
        if value.payload["letter"] == selected_letter
    )
    inputs = GoalSupportProbeInputs(
        decision_state=decision,
        domain_state=expander.root_state(),
        hole_id=_hole(expander),
        policy_action=policy_action,
        action_values=mapping,
        provider=provider,
        config=GoalSupportProbeConfig(exact_action_cap=10),
    )
    report, evidence = run_goal_support_probe(inputs)
    digest_map = {action_id_from_value(value): int_id for int_id, value in mapping.items()}
    binding = GoalSupportStateBinding(
        domain_state_fingerprint=expander.root_state().fingerprint,
        action_id_map=digest_map,
    )
    return decision, report, evidence, binding


def _materialize(state, report, evidence, binding):
    return materialize_goal_support(
        state, report, evidence, binding, profile_mode="production_exact"
    )


def test_partitions_map_exactly_and_cover_legal_set():
    state, report, evidence, binding = _probe_bundle(accept={"ax", "by"})
    view = _materialize(state, report, evidence, binding)
    supported = {binding.action_id_map[d] for d in report.partitions.supported}
    unsupported = {binding.action_id_map[d] for d in report.partitions.unsupported}
    unknown = {binding.action_id_map[d] for d in report.partitions.unknown}
    unobserved = {binding.action_id_map[d] for d in report.partitions.unobserved}
    assert set(view.good_action_ids) == supported
    assert set(view.bad_action_ids) == unsupported
    assert set(view.ambiguous_action_ids) == unknown
    assert set(view.unobserved_action_ids) == unobserved
    assert (
        set(view.good_action_ids)
        | set(view.bad_action_ids)
        | set(view.ambiguous_action_ids)
        | set(view.unobserved_action_ids)
    ) == set(state.legal_action_ids)


def test_multiple_supported_remain_set_valued_positives():
    state, report, evidence, binding = _probe_bundle(accept={"ax", "by"})
    view = _materialize(state, report, evidence, binding)
    assert len(view.good_action_ids) >= 2
    assert view.weights == tuple((action, 1.0) for action in view.good_action_ids)


def test_unknown_and_unobserved_never_hard_good_or_bad():
    state, report, evidence, binding = _probe_bundle(accept={"ax"})
    view = _materialize(state, report, evidence, binding)
    hard = set(view.good_action_ids) | set(view.bad_action_ids)
    assert not hard & set(view.ambiguous_action_ids)
    assert not hard & set(view.unobserved_action_ids)


def test_unsupported_evidence_retains_legal_true():
    state, report, evidence, binding = _probe_bundle(accept={"ax"})
    by_id = {row.action_id: row for row in evidence}
    for digest in report.partitions.unsupported:
        assert by_id[digest].legal
        assert binding.action_id_map[digest] in _materialize(
            state, report, evidence, binding
        ).bad_action_ids


def test_reordered_evidence_materializes_identically():
    state, report, evidence, binding = _probe_bundle(accept={"ax", "by"})
    forward = _materialize(state, report, evidence, binding)
    backward = _materialize(state, report, tuple(reversed(evidence)), binding)
    assert forward == backward


def test_duplicate_action_evidence_fails():
    state, report, evidence, binding = _probe_bundle(accept={"ax", "by"})
    dup = (evidence[0], evidence[0], *evidence[1:])
    with pytest.raises(ValueError, match="duplicate"):
        _materialize(state, report, dup, binding)


def test_stale_report_digest_fails():
    state, report, evidence, binding = _probe_bundle(accept={"ax", "by"})
    payload = report.to_dict(include_digest=True)
    payload["report_digest"] = "0" * 64
    with pytest.raises(ValueError, match="report_digest"):
        _materialize(
            state,
            GoalDomainAdequacyReportV1.from_dict(payload),
            evidence,
            binding,
        )


def test_evidence_partition_mismatch_fails():
    _state, _report, evidence, _binding = _probe_bundle(accept={"ax", "by"})
    row = evidence[0]
    with pytest.raises(ValueError, match="evidence_digest"):
        GoalActionEvidenceV1.from_dict(
            {**row.to_dict(include_digest=True), "partition": "unknown"}
        )


def test_evaluation_oracle_requires_diagnostic_permission():
    state, report, evidence, binding = _probe_bundle(accept={"ax", "by"})
    with pytest.raises(ValueError, match="allow_evaluation_diagnostic"):
        materialize_goal_support(
            state,
            report,
            evidence,
            binding,
            profile_mode="evaluation_oracle",
        )


def test_evaluation_oracle_diagnostic_is_non_trainable():
    state, report, evidence, binding = _probe_bundle(accept={"ax", "by"})
    view = materialize_goal_support(
        state,
        report,
        evidence,
        binding,
        profile_mode="evaluation_oracle",
        allow_evaluation_diagnostic=True,
    )
    assert view.trainable is False
    with pytest.raises(ValueError, match="non-trainable"):
        guard_goal_support_view(view)
    with pytest.raises(ValueError, match="non-trainable"):
        admit_semantic_corpus(
            [(state, view)],
            materializer_id=GOAL_SUPPORT_MATERIALIZER_ID,
            materializer_config_hash=view.materializer_config_hash,
        )


def test_advisory_diagnostic_is_non_trainable():
    state, report, evidence, binding = _probe_bundle(accept={"ax", "by"})
    view = materialize_goal_support(
        state,
        report,
        evidence,
        binding,
        profile_mode="advisory_diagnostic",
    )
    assert view.trainable is False


def test_production_exact_hard_profile_is_trainable():
    state, report, evidence, binding = _probe_bundle(accept={"ax", "by"})
    view = _materialize(state, report, evidence, binding)
    assert view.trainable is True
    assert view.materializer_id == GOAL_SUPPORT_MATERIALIZER_ID
    guard_goal_support_view(view)


def test_config_hash_changes_with_report_digest():
    state, report, evidence, binding = _probe_bundle(accept={"ax"})
    base = goal_support_materializer_config_hash(
        profile_mode="production_exact", report=report
    )
    _state2, report2, evidence2, binding2 = _probe_bundle(accept={"by"})
    other = goal_support_materializer_config_hash(
        profile_mode="production_exact", report=report2
    )
    assert base != other
    view1 = _materialize(state, report, evidence, binding)
    view2 = materialize_goal_support(
        _state2, report2, evidence2, binding2, profile_mode="production_exact"
    )
    assert view1.materializer_config_hash != view2.materializer_config_hash


def test_trainability_table_documents_profile_modes():
    table = goal_support_trainability_table()
    assert table["production_exact"]["may_supervise_semantic_corpus"] is True
    assert table["evaluation_oracle"]["promotion_allowed"] is False
    assert table["advisory_diagnostic"]["hard_good_bad_labels_forbidden"] is True
