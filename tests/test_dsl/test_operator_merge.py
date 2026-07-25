from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BranchEditV1,
    CompilerCoverage,
    ConversationStateNodeV1,
    EffectDeltaKind,
    EffectDeltaV1,
    MergeConflictKind,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    branch_fingerprint,
    build_reference_table,
    clone_reference_table_for_branch,
    merge_conversation_branches,
    replay_branch_merge,
)
from slm_training.dsl.operators.contracts import _fingerprint
from slm_training.dsl.pack import get_pack

SOURCE = (
    'root = Card([TextContent(":hero.title"), '
    'TextContent(":hero.body")], "clear")'
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.merge_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id="merge-request",
    )


def _effect(
    *,
    target,
    category: str,
    coverage: CompilerCoverage,
) -> ActionEffectV1:
    delta = EffectDeltaV1(
        kind=EffectDeltaKind(category),
        target=target,
        before="before",
        after="after",
    )
    values = {f"{category}_deltas": (delta,)}
    return ActionEffectV1(
        **values,
        compiler_coverage=coverage,
    )


class _Fixture:
    def __init__(self) -> None:
        self.pack = get_pack("openui")
        self.state = OperatorStateV1.from_source(self.pack, SOURCE)
        self.root_branch = branch_fingerprint(
            self.state.state_digest, _sha("merge-root")
        )
        descriptors = tuple(
            ReferenceDescriptorV1(
                ref_kind=RefKind.VALUE,
                semantic_fingerprint=_sha(name),
                value_type=f"openui.{name}",
            )
            for name in ("title", "body")
        )
        self.root_table = build_reference_table(
            request_id="merge-request",
            state_digest=self.state.state_digest,
            branch_digest=self.root_branch,
            descriptors=descriptors,
            seed=1,
        )
        self.base = ConversationStateNodeV1(
            parent_state_id=None,
            branch_digest=self.root_branch,
            state=self.state,
            reference_table=self.root_table,
        )
        self.authorities = {}

    def branch(
        self,
        *,
        name: str,
        target_name: str,
        replacement: str,
        category: str = "property",
        coverage: CompilerCoverage = CompilerCoverage.EXACT,
        operator_id: str | None = None,
        commutes_with: tuple[str, ...] = (),
        stale_effect_ref: bool = False,
    ) -> BranchEditV1:
        branch = branch_fingerprint(
            self.state.state_digest, _sha(f"merge-{name}")
        )
        table = clone_reference_table_for_branch(
            self.root_table, branch_digest=branch, seed=len(self.authorities) + 3
        )
        input_node = ConversationStateNodeV1(
            parent_state_id=self.base.state_id,
            branch_digest=branch,
            state=self.state,
            reference_table=table,
        )
        target = next(
            entry.ref
            for entry in table.entries
            if entry.descriptor.value_type == f"openui.{target_name}"
        )
        if stale_effect_ref:
            target = next(
                entry.ref
                for entry in self.root_table.entries
                if entry.descriptor.value_type == f"openui.{target_name}"
            )
        operator_id = operator_id or f"openui.merge_{name}"
        declaration = AstOperatorV1(
            operator_id=operator_id,
            version="v1",
            domain="openui.ast",
            codomain="openui.ast",
            argument_slots=(),
            preconditions=(),
            effect_signature=(EffectDeltaKind(category),),
            locality="node",
            cost=1.0,
            commutes_with=commutes_with,
        )
        before = f":hero.{target_name}"

        def execute(state, _arguments):
            if before not in state.source:
                raise ValueError("fixture target is unavailable")
            return OperatorMutationV1(
                source=state.source.replace(before, replacement),
                effect=_effect(
                    target=target,
                    category=category,
                    coverage=coverage,
                ),
            )

        library = OperatorLibraryV1(
            (RegisteredOperatorV1(declaration, execute),)
        )
        branch_pack = replace(self.pack, operator_library=library)
        applied = library.apply(
            branch_pack,
            input_node.state,
            operator_id,
            (),
            _provenance(input_node.state),
        )
        assert applied.succeeded and applied.state is not None
        output_table = build_reference_table(
            request_id=table.request_id,
            state_digest=applied.state.state_digest,
            branch_digest=branch,
            descriptors=tuple(entry.descriptor for entry in table.entries),
            seed=len(self.authorities) + 13,
        )
        output_node = ConversationStateNodeV1(
            parent_state_id=input_node.state_id,
            branch_digest=branch,
            state=applied.state,
            reference_table=output_table,
        )
        self.authorities[input_node.state_id] = (branch_pack, library)
        return BranchEditV1(input_node, output_node, applied.application)

    def resolve(self, node):
        return self.authorities[node.state_id]


def test_disjoint_merge_is_valid_replayable_and_order_invariant() -> None:
    fixture = _Fixture()
    left = fixture.branch(
        name="left", target_name="title", replacement=":hero.heading"
    )
    right = fixture.branch(
        name="right", target_name="body", replacement=":hero.copy"
    )
    decision = merge_conversation_branches(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    reversed_decision = merge_conversation_branches(
        pack=fixture.pack,
        base=fixture.base,
        left=right,
        right=left,
        authority_resolver=fixture.resolve,
    )

    assert decision.succeeded
    assert decision.decision_id == reversed_decision.decision_id
    assert decision.merge is not None
    assert ":hero.heading" in decision.merge.merged_state.source
    assert ":hero.copy" in decision.merge.merged_state.source
    assert (
        OperatorStateV1.from_source(
            fixture.pack, decision.merge.merged_state.source
        )
        == decision.merge.merged_state
    )
    assert replay_branch_merge(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
        recorded=decision,
    ).decision_id == decision.decision_id


def test_mutually_declared_commuting_equal_overlap_can_merge() -> None:
    fixture = _Fixture()
    left = fixture.branch(
        name="commute_left",
        target_name="title",
        replacement=":hero.heading",
        operator_id="openui.commute_left",
        commutes_with=("openui.commute_right",),
    )
    right = fixture.branch(
        name="commute_right",
        target_name="title",
        replacement=":hero.heading",
        operator_id="openui.commute_right",
        commutes_with=("openui.commute_left",),
    )
    decision = merge_conversation_branches(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    assert decision.succeeded


@pytest.mark.parametrize(
    ("category", "left_operator", "expected"),
    (
        ("property", "openui.change", MergeConflictKind.SAME_NODE_INCOMPATIBLE_EDIT),
        ("cardinality", "openui.change", MergeConflictKind.ROLE_CARDINALITY),
        ("topology", "openui.change", MergeConflictKind.CHILD_ORDER),
        ("scope", "openui.change", MergeConflictKind.SCOPE_BINDER),
        ("topology", "openui.remove_node", MergeConflictKind.DELETE_MODIFY),
    ),
)
def test_overlapping_effects_return_specific_typed_conflicts(
    category: str,
    left_operator: str,
    expected: MergeConflictKind,
) -> None:
    fixture = _Fixture()
    left = fixture.branch(
        name="conflict_left",
        target_name="title",
        replacement=":hero.left",
        category=category,
        operator_id=left_operator,
    )
    right = fixture.branch(
        name="conflict_right",
        target_name="title",
        replacement=":hero.right",
        category=category,
        operator_id="openui.modify_node",
    )
    decision = merge_conversation_branches(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    assert not decision.succeeded
    assert decision.conflict is not None
    assert decision.conflict.kind is expected
    assert len(decision.conflict.target_fingerprints) == 1


def test_stale_refs_and_inexact_effects_refuse_without_mutation() -> None:
    stale_fixture = _Fixture()
    stale = stale_fixture.branch(
        name="stale",
        target_name="title",
        replacement=":hero.left",
        stale_effect_ref=True,
    )
    fresh = stale_fixture.branch(
        name="fresh", target_name="body", replacement=":hero.right"
    )
    stale_decision = merge_conversation_branches(
        pack=stale_fixture.pack,
        base=stale_fixture.base,
        left=stale,
        right=fresh,
        authority_resolver=stale_fixture.resolve,
    )
    assert stale_decision.conflict is not None
    assert stale_decision.conflict.kind is MergeConflictKind.STALE_REF

    forged_fixture = _Fixture()
    forged_source = forged_fixture.branch(
        name="forged",
        target_name="title",
        replacement=":hero.left",
    )
    old_authority = forged_fixture.authorities[
        forged_source.input_node.state_id
    ]
    target_ref = forged_source.application.effect.property_deltas[0].target
    forged_table = replace(
        forged_source.input_node.reference_table,
        entries=tuple(
            replace(
                entry,
                descriptor=replace(
                    entry.descriptor, value_type="openui.forged"
                ),
            )
            if entry.ref == target_ref
            else entry
            for entry in forged_source.input_node.reference_table.entries
        ),
    )
    forged_input = replace(
        forged_source.input_node, reference_table=forged_table
    )
    forged = BranchEditV1(
        forged_input,
        replace(
            forged_source.output_node,
            parent_state_id=forged_input.state_id,
        ),
        forged_source.application,
    )
    forged_fixture.authorities[forged_input.state_id] = old_authority
    forged_peer = forged_fixture.branch(
        name="forged_peer",
        target_name="body",
        replacement=":hero.right",
    )
    forged_decision = merge_conversation_branches(
        pack=forged_fixture.pack,
        base=forged_fixture.base,
        left=forged,
        right=forged_peer,
        authority_resolver=forged_fixture.resolve,
    )
    assert forged_decision.conflict is not None
    assert forged_decision.conflict.kind is MergeConflictKind.STALE_REF

    partial_fixture = _Fixture()
    partial = partial_fixture.branch(
        name="partial",
        target_name="title",
        replacement=":hero.left",
        coverage=CompilerCoverage.APPROXIMATE,
    )
    exact = partial_fixture.branch(
        name="exact", target_name="body", replacement=":hero.right"
    )
    partial_decision = merge_conversation_branches(
        pack=partial_fixture.pack,
        base=partial_fixture.base,
        left=partial,
        right=exact,
        authority_resolver=partial_fixture.resolve,
    )
    assert partial_decision.conflict is not None
    assert partial_decision.conflict.kind is MergeConflictKind.UNSUPPORTED_EFFECT


def test_conflict_identity_is_deterministic_and_provenance_complete() -> None:
    fixture = _Fixture()
    left = fixture.branch(
        name="identity_left",
        target_name="title",
        replacement=":hero.left",
    )
    right = fixture.branch(
        name="identity_right",
        target_name="title",
        replacement=":hero.right",
    )
    decision = merge_conversation_branches(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    assert decision.conflict is not None
    assert decision.conflict.base_state_id == fixture.base.state_id
    assert decision.conflict.branch_state_ids == tuple(
        sorted((left.output_node.state_id, right.output_node.state_id))
    )
    assert decision.conflict.application_ids == tuple(
        sorted(
            (
                left.application.application_id,
                right.application.application_id,
            )
        )
    )
    assert decision.conflict.conflict_id == _fingerprint(
        decision.conflict.to_dict(include_conflict_id=False)
    )
    assert replay_branch_merge(
        pack=fixture.pack,
        base=fixture.base,
        left=right,
        right=left,
        authority_resolver=fixture.resolve,
        recorded=decision,
    ).decision_id == decision.decision_id
