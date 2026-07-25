"""Tests for the sanitized operator-policy model boundary (DSH3-22)."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    CompilerCoverage,
    CompilerFact,
    LegalSetCoverage,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    OperatorSupportVerdict,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    build_reference_table,
    enumerate_operator_legal_set,
)
from slm_training.dsl.operators.legal_set import OperatorLegalSetV1
from slm_training.dsl.operators.references import ReferenceTableV1
from slm_training.dsl.pack import get_pack
from slm_training.models.operator_policy_view import (
    FORBIDDEN_FIELD_NAMES,
    ForbiddenFieldError,
    OperatorPolicyViewError,
    build_operator_policy_input,
    validate_no_forbidden_fields,
)

SOURCE = 'root = TextContent(":hero.title")'
OPERATOR_ID = "openui.fixture_policy_view"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _value_descriptor(index: int) -> ReferenceDescriptorV1:
    return ReferenceDescriptorV1(
        ref_kind=RefKind.VALUE,
        semantic_fingerprint=_sha(f"semantic-value-{index}"),
        value_type="openui.string",
        compiler_facts=(CompilerFact.VALUE_VISIBLE,),
    )


class _Harness:
    """One reusable pack/library/state/provenance rig; tables vary per test."""

    def __init__(self, *, allowed_indices: frozenset[int], repeated: bool = False):
        self.base_pack = get_pack("openui")
        self.state = OperatorStateV1.from_source(self.base_pack, SOURCE)
        self.allowed = {
            _value_descriptor(index).semantic_fingerprint for index in allowed_indices
        }
        self._semantic_by_ref: dict[object, str] = {}

        def execute(operator_state, arguments):
            semantic = self._semantic_by_ref[arguments[0].value]
            if semantic not in self.allowed:
                raise OperatorRejectedError("fixture.not_legal")
            return OperatorMutationV1(
                source=operator_state.source.replace(":hero.title", ":hero.body"),
                effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
            )

        declaration = AstOperatorV1(
            operator_id=OPERATOR_ID,
            version="v1",
            domain="openui.ast",
            codomain="openui.ast",
            argument_slots=(
                OperatorArgumentSlotV1(
                    "value", RefKind.VALUE, BindingPhase.APPLICATION, repeated=repeated
                ),
            ),
            preconditions=(),
            effect_signature=(),
            locality="node",
            cost=1.0,
        )
        self.library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
        self.pack = replace(self.base_pack, operator_library=self.library)
        self.provenance = ApplicationProvenanceV1(
            pack_id="openui",
            compiler_id="openui.fixture",
            compiler_version="v1",
            source_artifact_digest=_sha(SOURCE),
            request_id="request-1",
        )

    def table(self, *, count: int, seed: int, request_id: str = "request-1"):
        descriptors = tuple(_value_descriptor(index) for index in range(count))
        return build_reference_table(
            request_id=request_id,
            state_digest=self.state.state_digest,
            branch_digest=_sha("branch"),
            descriptors=descriptors,
            seed=seed,
        )

    def enumerate(self, table: ReferenceTableV1) -> OperatorLegalSetV1:
        self._semantic_by_ref.clear()
        self._semantic_by_ref.update(
            {entry.ref: entry.descriptor.semantic_fingerprint for entry in table.entries}
        )
        return enumerate_operator_legal_set(
            pack=self.pack,
            library=self.library,
            state=self.state,
            reference_table=table,
            provenance=self.provenance,
        )

    def view(self, table: ReferenceTableV1):
        legal_set = self.enumerate(table)
        return legal_set, build_operator_policy_input(table, legal_set, self.library)


def test_supported_operator_view_carries_only_allowlisted_facts() -> None:
    harness = _Harness(allowed_indices=frozenset({1, 3}))
    table = harness.table(count=4, seed=7)
    legal_set, view = harness.view(table)

    assert legal_set.legal_operator_ids == (OPERATOR_ID,)
    assert len(view.reference_rows) == 4
    assert len(view.action_rows) == 1
    action = view.action_rows[0]
    assert action.operator_id == OPERATOR_ID
    assert action.operator_version == "v1"
    assert action.locality == "node"
    assert action.cost == 1.0
    assert action.verdict is OperatorSupportVerdict.SUPPORTED
    assert action.coverage is LegalSetCoverage.COMPLETE
    slot = action.argument_slots[0]
    assert slot.domain_complete is True
    assert set(slot.candidate_rows) == {view.reference_rows[i].row for i in range(4)}
    for ref in view.reference_rows:
        assert ref.ref_kind is RefKind.VALUE
        assert ref.value_type == "openui.string"
        assert not ref.has_parent
        assert ref.parent_row is None
        assert ref.relative_position is None


def test_partial_legal_set_from_unbounded_repeated_slot_is_explicit() -> None:
    harness = _Harness(allowed_indices=frozenset({0}), repeated=True)
    table = harness.table(count=3, seed=11)
    legal_set, view = harness.view(table)

    assert legal_set.coverage is LegalSetCoverage.PARTIAL
    action = view.action_rows[0]
    assert action.verdict is OperatorSupportVerdict.UNKNOWN
    assert action.coverage is LegalSetCoverage.PARTIAL
    assert action.argument_slots[0].domain_complete is False
    assert view.coverage is LegalSetCoverage.PARTIAL


def test_index_reference_exposes_parent_row_and_relative_position() -> None:
    harness = _Harness(allowed_indices=frozenset())
    parent_semantic = _sha("parent-node")
    parent = ReferenceDescriptorV1(
        ref_kind=RefKind.NODE,
        semantic_fingerprint=parent_semantic,
        value_type="openui.node",
    )
    child_index = ReferenceDescriptorV1(
        ref_kind=RefKind.INDEX,
        semantic_fingerprint=_sha("child-index"),
        value_type="openui.index",
        parent_fingerprint=parent_semantic,
        parent_order_digest=_sha("order"),
        position=2,
    )
    table = build_reference_table(
        request_id="request-1",
        state_digest=harness.state.state_digest,
        branch_digest=_sha("branch"),
        descriptors=(parent, child_index),
        seed=3,
    )
    legal_set, view = harness.view(table)

    by_kind = {ref.ref_kind: ref for ref in view.reference_rows}
    index_view = by_kind[RefKind.INDEX]
    parent_view = by_kind[RefKind.NODE]
    assert index_view.has_parent is True
    assert index_view.parent_row == parent_view.row
    assert index_view.relative_position == 2
    assert parent_view.relative_position is None


def test_opaque_id_and_candidate_order_permutation_preserve_the_view() -> None:
    harness = _Harness(allowed_indices=frozenset({1, 3}))
    table = harness.table(count=4, seed=7)
    _, view = harness.view(table)

    permuted = table.permuted(seed=99)
    _, permuted_view = harness.view(permuted)

    assert view.to_dict() == permuted_view.to_dict()


def test_changing_only_allocation_seed_leaves_the_view_unchanged() -> None:
    """A different opaque-ID allocation seed changes every application/proof
    hash downstream (semantic_id, application_id, proof_fingerprint all
    depend on ref identity), but the sanitized view is byte-identical."""
    harness = _Harness(allowed_indices=frozenset({1, 3}))
    table_a = harness.table(count=4, seed=7)
    table_b = harness.table(count=4, seed=1234)

    legal_set_a, view_a = harness.view(table_a)
    legal_set_b, view_b = harness.view(table_b)

    actions_a = {action.application_id for action in legal_set_a.operator_actions}
    actions_b = {action.application_id for action in legal_set_b.operator_actions}
    assert actions_a.isdisjoint(actions_b), "fixture did not actually vary the hash chain"
    assert view_a.to_dict() == view_b.to_dict()


def test_stale_reference_table_cannot_produce_a_view() -> None:
    harness = _Harness(allowed_indices=frozenset({1, 3}))
    table = harness.table(count=4, seed=7)
    legal_set = harness.enumerate(table)
    other_table = harness.table(count=4, seed=8)

    with pytest.raises(OperatorPolicyViewError) as excinfo:
        build_operator_policy_input(other_table, legal_set, harness.library)
    assert excinfo.value.code == "policy_view.stale_reference_table"


def test_view_never_carries_a_forbidden_field(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(allowed_indices=frozenset({1, 3}))
    table = harness.table(count=4, seed=7)
    _, view = harness.view(table)

    # Direct-construction regression: __post_init__ runs the same recursive
    # validator, so a payload that slipped a forbidden key past a future
    # to_dict() edit would fail closed at construction time, not silently.
    validate_no_forbidden_fields(view.to_dict())


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_FIELD_NAMES))
def test_forbidden_field_is_rejected_at_every_nesting_depth(forbidden: str) -> None:
    shallow = {forbidden: "leak"}
    nested = {"a": {"b": [{"c": (1, {forbidden: "leak"})}]}}
    for payload in (shallow, nested):
        with pytest.raises(ForbiddenFieldError):
            validate_no_forbidden_fields(payload)


def test_allowlisted_payload_passes_forbidden_field_validation() -> None:
    validate_no_forbidden_fields(
        {"ref_kind": "value", "compiler_facts": ["value.visible"], "row": 0}
    )
