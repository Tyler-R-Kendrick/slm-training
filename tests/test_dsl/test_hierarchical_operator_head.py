"""Wiring/invariant tests for the DSH3-15 hierarchical operator action head.

These tests prove wiring safety only (feature-off parity, legal-set-
membership closure, permutation invariance) -- not a causal-improvement
claim. DSH3-15's own "Decision" text gates a positive causal result behind
its rejected DSH3-14/SLM-382 prerequisite (see
``docs/design/e803-reserved-operator-baseline-20260723``), and
``slm_training.evals.cap2_disposition`` already records ``HIERARCHICAL_HEAD``
as ``UNRUN_CONDITIONAL`` for that reason. Nothing here changes that verdict.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    HierarchicalDecisionKind,
    HierarchicalOperatorActionHead,
    HierarchicalOperatorHeadConfigV1,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    build_operator_action_features,
    build_reference_table,
    enumerate_operator_legal_set,
    group_operator_features,
)
from slm_training.dsl.pack import get_pack
from slm_training.models.dynamic_pointer_scorer import count_pointer_scorer_parameters

SOURCE = 'root = TextContent(":before")'
REQUEST_ID = "hier-head-test"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _build_fixture(*, seed: int = 7):
    base = get_pack("openui")
    state = OperatorStateV1.from_source(base, SOURCE)

    value_descriptors = tuple(
        ReferenceDescriptorV1(
            ref_kind=RefKind.VALUE,
            semantic_fingerprint=_sha(f"value-{i}"),
            value_type="openui.string",
        )
        for i in range(3)
    )
    role_descriptors = tuple(
        ReferenceDescriptorV1(
            ref_kind=RefKind.ROLE,
            semantic_fingerprint=_sha(f"role-{i}"),
            value_type="openui.role",
        )
        for i in range(2)
    )
    table = build_reference_table(
        request_id=REQUEST_ID,
        state_digest=state.state_digest,
        branch_digest=_sha("branch"),
        descriptors=value_descriptors + role_descriptors,
        seed=seed,
    )
    # Executors accept any argument of the declared ref kind rather than
    # closing over specific ref identities from ``table`` -- opaque ids are
    # request/permutation-local and must never gate operator legality here.
    def execute_set_value(_state, arguments):
        (argument,) = arguments
        return OperatorMutationV1(
            source='root = TextContent(":afterset")',
            effect=ActionEffectV1(
                property_deltas=(
                    EffectDeltaV1(
                        EffectDeltaKind.PROPERTY, argument.value, "before", "after"
                    ),
                ),
                compiler_coverage=CompilerCoverage.EXACT,
            ),
        )

    def execute_bind_role(_state, arguments):
        (argument,) = arguments
        return OperatorMutationV1(
            source='root = TextContent(":bound")',
            effect=ActionEffectV1(
                scope_deltas=(
                    EffectDeltaV1(
                        EffectDeltaKind.SCOPE, argument.value, "unbound", "bound"
                    ),
                ),
                compiler_coverage=CompilerCoverage.EXACT,
            ),
        )

    set_value_decl = AstOperatorV1(
        operator_id="openui.fixture_set_value",
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(EffectDeltaKind.PROPERTY,),
        locality="node",
        cost=1.0,
    )
    bind_role_decl = AstOperatorV1(
        operator_id="openui.fixture_bind_role",
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("role", RefKind.ROLE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(EffectDeltaKind.SCOPE,),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1(
        (
            RegisteredOperatorV1(set_value_decl, execute_set_value),
            RegisteredOperatorV1(bind_role_decl, execute_bind_role),
        )
    )
    pack = replace(base, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="hier-head-test",
        compiler_version="v1",
        source_artifact_digest=_sha(SOURCE),
        request_id=REQUEST_ID,
    )
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=table,
        provenance=provenance,
    )
    return pack, library, state, table, provenance, legal_set


def test_config_defaults_to_off_with_zero_parameters() -> None:
    config = HierarchicalOperatorHeadConfigV1()
    assert config.mode == "off"
    head = HierarchicalOperatorActionHead(config)
    assert head.enabled is False
    assert count_pointer_scorer_parameters(head._scorer) == 0


def test_features_are_constructed_solely_from_the_live_legal_set() -> None:
    pack, library, state, _table, provenance, legal_set = _build_fixture()
    features = build_operator_action_features(
        legal_set=legal_set, library=library, pack=pack, state=state, provenance=provenance
    )
    legal_application_ids = {a.application_id for a in legal_set.operator_actions}
    legal_semantic_ids = {a.semantic_id for a in legal_set.operator_actions}
    assert {f.application_id for f in features} == legal_application_ids
    assert {f.semantic_id for f in features} == legal_semantic_ids
    assert len(features) == len(legal_set.operator_actions) == 5

    groups = group_operator_features(features)
    assert {g.operator_id for g in groups} == {
        "openui.fixture_set_value",
        "openui.fixture_bind_role",
    }
    set_value_group = next(g for g in groups if g.operator_id == "openui.fixture_set_value")
    bind_role_group = next(g for g in groups if g.operator_id == "openui.fixture_bind_role")
    assert set_value_group.candidate_count == 3
    assert set_value_group.dominant_effect_kind == EffectDeltaKind.PROPERTY.value
    assert bind_role_group.candidate_count == 2
    assert bind_role_group.dominant_effect_kind == EffectDeltaKind.SCOPE.value


def test_off_mode_always_defers_and_never_touches_legal_membership() -> None:
    pack, library, state, table, provenance, legal_set = _build_fixture()
    features = build_operator_action_features(
        legal_set=legal_set, library=library, pack=pack, state=state, provenance=provenance
    )
    head = HierarchicalOperatorActionHead(HierarchicalOperatorHeadConfigV1(mode="off"))
    state_vector = torch.zeros(head.config.d_model)

    decision = head.decide(state_vector, legal_set, features)

    assert decision.decision is HierarchicalDecisionKind.DEFER
    assert decision.operator_id is None
    assert decision.application_id is None
    # Off mode still shadow-scores the full legal set (uniform/no-op scores).
    assert len(decision.operator_scores) == 2  # two distinct operators
    assert len(decision.argument_scores) == len(legal_set.operator_actions)
    assert all(score == 0.0 for score in decision.operator_scores)
    assert all(score == 0.0 for score in decision.argument_scores)
    # The legal set itself is byte-identical regardless of the head's mode.
    replay_legal_set = enumerate_operator_legal_set(
        pack=pack, library=library, state=state, reference_table=table, provenance=provenance
    )
    assert replay_legal_set.fingerprint == legal_set.fingerprint


def test_enabled_mode_never_selects_outside_the_legal_set() -> None:
    pack, library, state, _table, provenance, legal_set = _build_fixture()
    features = build_operator_action_features(
        legal_set=legal_set, library=library, pack=pack, state=state, provenance=provenance
    )
    head = HierarchicalOperatorActionHead(
        HierarchicalOperatorHeadConfigV1(mode="dynamic_head", d_model=16, hidden_dim=16)
    )
    assert head.enabled is True
    assert count_pointer_scorer_parameters(head._scorer) > 0
    state_vector = torch.zeros(head.config.d_model)

    decision = head.decide(state_vector, legal_set, features)

    assert decision.decision is HierarchicalDecisionKind.APPLY
    legal_application_ids = {a.application_id for a in legal_set.operator_actions}
    assert decision.application_id in legal_application_ids
    legal_operator_ids = {a.operator_id for a in legal_set.operator_actions}
    assert decision.operator_id in legal_operator_ids
    # Zero false hard prunes/admissions: the chosen action's operator_id must
    # agree with the legal action it names.
    chosen = next(
        a for a in legal_set.operator_actions if a.application_id == decision.application_id
    )
    assert chosen.operator_id == decision.operator_id


def test_candidates_outside_the_legal_set_are_rejected() -> None:
    pack, library, state, _table, provenance, legal_set = _build_fixture()
    features = build_operator_action_features(
        legal_set=legal_set, library=library, pack=pack, state=state, provenance=provenance
    )
    forged = replace(features[0], application_id=_sha("not-a-live-application"))
    head = HierarchicalOperatorActionHead(HierarchicalOperatorHeadConfigV1(mode="off"))
    state_vector = torch.zeros(head.config.d_model)
    with pytest.raises(ValueError, match="not members of the live legal set"):
        head.decide(state_vector, legal_set, (forged, *features[1:]))


def test_stale_replay_fails_closed_instead_of_silently_admitting() -> None:
    pack, library, state, _table, provenance, legal_set = _build_fixture()
    stale_action = replace(legal_set.operator_actions[0], application_id=_sha("stale"))
    stale_entry = replace(
        legal_set.entries[0],
        legal_actions=(stale_action, *legal_set.entries[0].legal_actions[1:]),
    )
    stale_legal_set = replace(
        legal_set, entries=(stale_entry, *legal_set.entries[1:])
    )
    with pytest.raises(ValueError, match="did not replay identically"):
        build_operator_action_features(
            legal_set=stale_legal_set,
            library=library,
            pack=pack,
            state=state,
            provenance=provenance,
        )


def test_permutation_of_reference_table_preserves_the_hierarchical_choice() -> None:
    """Opaque IDs/order carry no semantics: a permuted reference table must
    reproduce the identical operator/semantic choice under a fixed head."""
    pack, library, state, table, provenance, legal_set = _build_fixture(seed=7)
    features = build_operator_action_features(
        legal_set=legal_set, library=library, pack=pack, state=state, provenance=provenance
    )

    permuted_table = table.permuted(seed=99)
    assert permuted_table.fingerprint != table.fingerprint
    permuted_legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=permuted_table,
        provenance=provenance,
    )
    permuted_features = build_operator_action_features(
        legal_set=permuted_legal_set,
        library=library,
        pack=pack,
        state=state,
        provenance=provenance,
    )

    # application_id/opaque identity legitimately differs after permutation...
    assert {f.application_id for f in features} != {
        f.application_id for f in permuted_features
    }
    # ...but semantic_id (the permutation-invariant identity) does not.
    assert {f.semantic_id for f in features} == {f.semantic_id for f in permuted_features}

    head = HierarchicalOperatorActionHead(
        HierarchicalOperatorHeadConfigV1(mode="dynamic_head", d_model=16, hidden_dim=16)
    )
    state_vector = torch.zeros(head.config.d_model)
    decision = head.decide(state_vector, legal_set, features)
    permuted_decision = head.decide(state_vector, permuted_legal_set, permuted_features)

    assert decision.decision is permuted_decision.decision is HierarchicalDecisionKind.APPLY
    assert decision.operator_id == permuted_decision.operator_id
    chosen = next(
        f for f in features if f.application_id == decision.application_id
    )
    permuted_chosen = next(
        f for f in permuted_features if f.application_id == permuted_decision.application_id
    )
    assert chosen.semantic_id == permuted_chosen.semantic_id
