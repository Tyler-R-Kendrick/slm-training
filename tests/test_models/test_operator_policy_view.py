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
    SelectorFact,
    SelectorKind,
    branch_fingerprint,
    build_bulk_selector,
    build_openui_bulk_operator_library,
    build_openui_local_operator_context,
    build_reference_table,
    enumerate_operator_legal_set,
)
from slm_training.dsl.operators.legal_set import OperatorLegalSetV1
from slm_training.dsl.operators.references import ReferenceTableV1
from slm_training.dsl.operators.local import NodeLocationV1
from slm_training.dsl.pack import get_pack
from slm_training.models.operator_policy_view import (
    FORBIDDEN_FIELD_NAMES,
    ForbiddenFieldError,
    OperatorActionViewV1,
    OperatorArgumentSlotViewV1,
    OperatorPolicyInputV1,
    OperatorPolicyViewError,
    ReferenceModelViewV1,
    build_operator_policy_input,
    operator_policy_input_from_dict,
    tag_recently_touched_rows,
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


def test_bulk_selector_row_is_sanitized_and_joined_to_the_live_legal_set() -> None:
    base_pack = get_pack("openui")
    source = 'root = Card([Stack([TextContent(":item")])], "clear")'
    state = OperatorStateV1.from_source(base_pack, source)
    context = build_openui_local_operator_context(
        base_pack,
        state,
        request_id="request-1",
        branch_digest=branch_fingerprint(state.state_digest, _sha("branch")),
        seed=7,
        values=("row",),
    )
    target = next(
        ref
        for ref in context.references(RefKind.NODE)
        if isinstance(context.payload(ref), NodeLocationV1)
        and context.payload(ref).path == ("root", "props", "children", 0)
    )
    context, selector = build_bulk_selector(
        context,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_sha("root-scope"),
        matching_refs=(target,),
        max_fanout=1,
        seed=11,
        compiler_facts=(
            SelectorFact.EXACT_FINITE,
            SelectorFact.FANOUT_BOUNDED,
            SelectorFact.PACK_AUTHORIZED,
            SelectorFact.SCOPE_ROOTED,
        ),
    )
    library = build_openui_bulk_operator_library(context)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(source),
        request_id="request-1",
    )
    legal_set = enumerate_operator_legal_set(
        pack=replace(base_pack, operator_library=library),
        library=library,
        state=state,
        reference_table=context.reference_table,
        provenance=provenance,
    )
    view = build_operator_policy_input(context.reference_table, legal_set, library)

    selector_row = next(row for row in view.reference_rows if row.ref_kind is RefKind.SELECTOR)
    assert selector_row.compiler_facts == (
        SelectorFact.EXACT_FINITE,
        SelectorFact.FANOUT_BOUNDED,
        SelectorFact.PACK_AUTHORIZED,
        SelectorFact.SCOPE_ROOTED,
    )
    assert selector_row.selector_kind is SelectorKind.COMPONENT_TYPE_IN_SCOPE
    assert selector_row.selector_cardinality == selector_row.selector_max_fanout == 1
    bulk_action = next(action for action in view.action_rows if action.operator_id == "openui.map_set_property")
    selector_slot = next(slot for slot in bulk_action.argument_slots if slot.slot_id == "selector")
    assert selector_slot.candidate_rows == (selector_row.row,)
    payload = view.to_dict()
    assert "target_fingerprints" not in str(payload)
    assert "opaque_id" not in str(payload)
    assert operator_policy_input_from_dict(payload).to_dict() == payload


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


def test_canonical_row_maps_keep_external_labels_joined_to_persisted_rows() -> None:
    """Supervision stored beside a canonical view must remap both join axes."""
    view = OperatorPolicyInputV1(
        reference_rows=(
            ReferenceModelViewV1(
                row=0,
                ref_kind=RefKind.VALUE,
                value_type="openui.string",
                compiler_facts=(CompilerFact.VALUE_VISIBLE,),
                has_parent=False,
                parent_row=None,
                relative_position=None,
            ),
            ReferenceModelViewV1(
                row=1,
                ref_kind=RefKind.NODE,
                value_type="openui.element",
                compiler_facts=(CompilerFact.NODE_VISIBLE,),
                has_parent=False,
                parent_row=None,
                relative_position=None,
            ),
        ),
        action_rows=(
            OperatorActionViewV1(
                row=0,
                operator_id="openui.z_action",
                operator_version="v1",
                locality="node",
                cost=1.0,
                effect_signature=(),
                argument_slots=(
                    OperatorArgumentSlotViewV1(
                        slot_id="value",
                        ref_kind=RefKind.VALUE,
                        binding_phase=BindingPhase.APPLICATION,
                        required=True,
                        repeated=False,
                        candidate_rows=(0,),
                        domain_complete=True,
                    ),
                ),
                verdict=OperatorSupportVerdict.SUPPORTED,
                coverage=LegalSetCoverage.COMPLETE,
            ),
            OperatorActionViewV1(
                row=1,
                operator_id="openui.a_action",
                operator_version="v1",
                locality="node",
                cost=1.0,
                effect_signature=(),
                argument_slots=(),
                verdict=OperatorSupportVerdict.SUPPORTED,
                coverage=LegalSetCoverage.COMPLETE,
            ),
        ),
        ordinary_action_count=0,
        coverage=LegalSetCoverage.COMPLETE,
    )

    reference_rows, action_rows = view.canonical_row_maps()
    persisted = view.to_dict()
    assert reference_rows == {0: 1, 1: 0}
    assert action_rows == {0: 1, 1: 0}
    assert persisted["action_rows"][action_rows[0]]["operator_id"] == "openui.z_action"
    assert (
        persisted["action_rows"][action_rows[0]]["argument_slots"][0][
            "candidate_rows"
        ]
        == [reference_rows[0]]
    )
    assert operator_policy_input_from_dict(persisted).to_dict() == persisted


def test_stale_reference_table_cannot_produce_a_view() -> None:
    harness = _Harness(allowed_indices=frozenset({1, 3}))
    table = harness.table(count=4, seed=7)
    legal_set = harness.enumerate(table)
    other_table = harness.table(count=4, seed=8)

    with pytest.raises(OperatorPolicyViewError) as excinfo:
        build_operator_policy_input(other_table, legal_set, harness.library)
    assert excinfo.value.code == "policy_view.stale_reference_table"


# --------------------------------------------------------------------------- #
# recently_touched: the SLM-418 ninth slice's bounded history bit.
# --------------------------------------------------------------------------- #
_NODE_FOCUS_SEMANTIC = _sha("node-focus")


def _mixed_descriptors() -> tuple[ReferenceDescriptorV1, ...]:
    """Two content-identical VALUE rows plus one distinguishable NODE row."""
    return (
        _value_descriptor(0),
        _value_descriptor(1),
        ReferenceDescriptorV1(
            ref_kind=RefKind.NODE,
            semantic_fingerprint=_NODE_FOCUS_SEMANTIC,
            value_type="openui.node",
        ),
    )


def _mixed_table(harness: _Harness, *, seed: int) -> ReferenceTableV1:
    return build_reference_table(
        request_id="request-1",
        state_digest=harness.state.state_digest,
        branch_digest=_sha("branch"),
        descriptors=_mixed_descriptors(),
        seed=seed,
    )


def test_recently_touched_defaults_off_and_round_trips() -> None:
    harness = _Harness(allowed_indices=frozenset({1, 3}))
    table = harness.table(count=4, seed=7)
    _, view = harness.view(table)

    assert all(not row.recently_touched for row in view.reference_rows)
    payload = view.to_dict()
    assert all(row["recently_touched"] is False for row in payload["reference_rows"])
    assert operator_policy_input_from_dict(payload).to_dict() == payload

    tagged = tag_recently_touched_rows(view, (2,))
    assert [row.recently_touched for row in tagged.reference_rows] == [
        False,
        False,
        True,
        False,
    ]
    tagged_payload = tagged.to_dict()
    assert sum(row["recently_touched"] for row in tagged_payload["reference_rows"]) == 1
    assert tagged_payload != payload
    assert operator_policy_input_from_dict(tagged_payload).to_dict() == tagged_payload
    # It is not, and must never become, a forbidden identity field.
    validate_no_forbidden_fields(tagged_payload)


def test_recently_touched_defaults_off_for_payloads_persisted_before_it_existed() -> None:
    harness = _Harness(allowed_indices=frozenset({1, 3}))
    table = harness.table(count=4, seed=7)
    _, view = harness.view(table)

    legacy = view.to_dict()
    for row in legacy["reference_rows"]:
        del row["recently_touched"]
    rehydrated = operator_policy_input_from_dict(legacy)
    assert all(not row.recently_touched for row in rehydrated.reference_rows)


def test_recently_touched_travels_with_content_not_row_order() -> None:
    """The permutation-equivariance regression for the new field.

    Two views built from opaque-ID/row-order permutations of the same table,
    with the *same content* tagged, must serialize identically; tagging a
    *different* reference must not. Without ``recently_touched`` in
    ``canonical_row_maps``'s sort key the first assertion fails, because two
    otherwise identical rows would keep build-time order.
    """
    harness = _Harness(allowed_indices=frozenset({0, 1}))
    table = _mixed_table(harness, seed=7)
    permuted = table.permuted(seed=99)
    assert tuple(entry.ref.opaque_id for entry in permuted.entries) != tuple(
        entry.ref.opaque_id for entry in table.entries
    )

    def tagged(source: ReferenceTableV1, semantic: str) -> dict:
        _, view = harness.view(source)
        row = next(
            index
            for index, entry in enumerate(source.entries)
            if entry.descriptor.semantic_fingerprint == semantic
        )
        return tag_recently_touched_rows(view, (row,)).to_dict()

    value_semantic = _value_descriptor(0).semantic_fingerprint
    assert tagged(table, _NODE_FOCUS_SEMANTIC) == tagged(permuted, _NODE_FOCUS_SEMANTIC)
    assert tagged(table, value_semantic) == tagged(permuted, value_semantic)
    assert tagged(table, _NODE_FOCUS_SEMANTIC) != tagged(permuted, value_semantic)

    _, untagged_view = harness.view(table)
    assert untagged_view.to_dict() != tagged(table, _NODE_FOCUS_SEMANTIC)


def test_two_content_identical_rows_stay_canonically_ordered_when_one_is_tagged() -> None:
    """The exact case the field exists for: rows differing only in this bit.

    The two VALUE rows are indistinguishable in every other allowed field, so
    only ``recently_touched`` can separate them -- and the canonical payload
    must still be independent of which build-time row happened to carry it.
    """
    harness = _Harness(allowed_indices=frozenset({0, 1}))
    table = _mixed_table(harness, seed=13)
    permuted = table.permuted(seed=5)

    def tagged(source: ReferenceTableV1, semantic: str) -> dict:
        _, view = harness.view(source)
        row = next(
            index
            for index, entry in enumerate(source.entries)
            if entry.descriptor.semantic_fingerprint == semantic
        )
        return tag_recently_touched_rows(view, (row,)).to_dict()

    first = _value_descriptor(0).semantic_fingerprint
    second = _value_descriptor(1).semantic_fingerprint
    # Content-identical rows: tagging either one is the same sanitized fact.
    assert tagged(table, first) == tagged(table, second)
    assert tagged(table, first) == tagged(permuted, second)


def test_tag_recently_touched_rows_fails_closed() -> None:
    harness = _Harness(allowed_indices=frozenset({1, 3}))
    table = harness.table(count=4, seed=7)
    _, view = harness.view(table)

    with pytest.raises(OperatorPolicyViewError) as unknown:
        tag_recently_touched_rows(view, (4,))
    assert unknown.value.code == "policy_view.unknown_reference_row"

    with pytest.raises(OperatorPolicyViewError) as negative:
        tag_recently_touched_rows(view, (-1,))
    assert negative.value.code == "policy_view.unknown_reference_row"

    with pytest.raises(OperatorPolicyViewError) as duplicate:
        tag_recently_touched_rows(view, (1, 1))
    assert duplicate.value.code == "policy_view.duplicate_reference_row"

    # An empty tag set is a legitimate answer (no preceding edit), not an error.
    assert tag_recently_touched_rows(view, ()) is view


def test_recently_touched_must_be_a_bool() -> None:
    with pytest.raises(ValueError):
        ReferenceModelViewV1(
            row=0,
            ref_kind=RefKind.VALUE,
            value_type="openui.string",
            compiler_facts=(CompilerFact.VALUE_VISIBLE,),
            has_parent=False,
            parent_row=None,
            relative_position=None,
            recently_touched=1,  # type: ignore[arg-type]
        )


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
