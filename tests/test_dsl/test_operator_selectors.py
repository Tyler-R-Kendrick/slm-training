from __future__ import annotations

import hashlib
import json

import pytest

from slm_training.dsl.operators import (
    CompilerFact,
    RefKind,
    ReferenceDescriptorV1,
    ReferenceResolutionError,
    ReferenceTableV1,
    SelectorBuildVerdict,
    SelectorContextV1,
    SelectorDescriptorV1,
    SelectorEntryV1,
    SelectorFact,
    SelectorKind,
    SelectorRef,
    attach_selector,
    branch_fingerprint,
    build_reference_table,
    build_selector,
    build_selector_descriptor,
    build_selector_from_scope,
)

_STATE = "a" * 64
_ROOT = "b" * 64
_NONCE = "c" * 64
_SCOPE = "1" * 64


def _branch(nonce: str = _NONCE) -> str:
    return branch_fingerprint(_ROOT, nonce)


def _fingerprint_descriptor(seed: str) -> ReferenceDescriptorV1:
    return ReferenceDescriptorV1(
        ref_kind=RefKind.NODE,
        semantic_fingerprint=hashlib.sha256(seed.encode()).hexdigest(),
        value_type="openui.stack" if "stack" in seed else "openui.card",
        compiler_facts=(CompilerFact.NODE_VISIBLE,),
    )


def _table(count: int = 3, *, seed: int = 1) -> ReferenceTableV1:
    descriptors = tuple(
        _fingerprint_descriptor(f"node-{i}-stack" if i % 2 == 0 else f"node-{i}-card")
        for i in range(count)
    )
    return build_reference_table(
        request_id="req-1",
        state_digest=_STATE,
        branch_digest=_branch(),
        descriptors=descriptors,
        seed=seed,
    )


def _stack_refs(table: ReferenceTableV1) -> tuple:
    return tuple(
        entry.ref for entry in table.entries if entry.descriptor.value_type == "openui.stack"
    )


def test_selector_kind_is_closed_and_rejects_arbitrary_strings() -> None:
    with pytest.raises(ValueError):
        SelectorKind("free_form_query")
    with pytest.raises(ValueError):
        SelectorFact("user.alice said so")


def test_zero_one_and_many_match_selectors_build_exact_membership() -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    assert len(stack_refs) >= 2

    zero_descriptor = build_selector_descriptor(
        table=table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=(),
        max_fanout=8,
    )
    assert zero_descriptor.cardinality == 0

    one_descriptor = build_selector_descriptor(
        table=table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs[:1],
        max_fanout=8,
    )
    assert one_descriptor.cardinality == 1

    many_descriptor = build_selector_descriptor(
        table=table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
    )
    assert many_descriptor.cardinality == len(stack_refs)
    assert many_descriptor.target_fingerprints == tuple(
        sorted(many_descriptor.target_fingerprints)
    )


def test_target_order_carries_no_semantics() -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    forward = build_selector_descriptor(
        table=table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
    )
    backward = build_selector_descriptor(
        table=table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=tuple(reversed(stack_refs)),
        max_fanout=8,
    )
    assert forward == backward
    assert forward.fingerprint == backward.fingerprint


def test_duplicate_targets_are_rejected() -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    with pytest.raises(ReferenceResolutionError) as raised:
        build_selector_descriptor(
            table=table,
            selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            scope_fingerprint=_SCOPE,
            matching_refs=(stack_refs[0], stack_refs[0]),
            max_fanout=8,
        )
    assert raised.value.code == "selector.duplicate_target"


def test_gold_derived_selectors_are_rejected_as_member_missing() -> None:
    table = _table(4)
    foreign_ref = SelectorRef("req-1", "not-a-real-member")
    with pytest.raises(ReferenceResolutionError) as raised:
        build_selector_descriptor(
            table=table,
            selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            scope_fingerprint=_SCOPE,
            matching_refs=(foreign_ref,),
            max_fanout=8,
        )
    assert raised.value.code == "selector.member_missing"


def test_fanout_boundary_at_and_over_the_bound() -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    assert len(stack_refs) == 2
    exact = build_selector_descriptor(
        table=table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=len(stack_refs),
    )
    assert exact.cardinality == exact.max_fanout

    with pytest.raises(ReferenceResolutionError) as raised:
        build_selector_descriptor(
            table=table,
            selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            scope_fingerprint=_SCOPE,
            matching_refs=stack_refs,
            max_fanout=len(stack_refs) - 1,
        )
    assert raised.value.code == "selector.fanout_overflow"


def test_allocation_uses_the_permutation_safe_reference_table_path() -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    new_table, ref = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=42,
    )
    assert isinstance(ref, SelectorRef)
    assert ref.request_id == "req-1"
    assert len(new_table.selectors) == 1
    # Opaque wiring: id/name/order carry no semantics of their own.
    assert ref.opaque_id != "openui.stack"


def test_opaque_id_and_candidate_order_permutation_preserve_resolution() -> None:
    table = _table(5, seed=7)
    stack_refs = _stack_refs(table)
    built_table, ref = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=11,
    )
    permuted = built_table.permuted(seed=99)
    descriptor = next(entry.descriptor for entry in built_table.selectors)
    permuted_ref = next(
        entry.ref for entry in permuted.selectors if entry.descriptor == descriptor
    )
    assert permuted_ref.opaque_id != ref.opaque_id

    current_targets = descriptor.target_fingerprints
    first_resolved = built_table.resolve_selector(
        ref,
        state_digest=_STATE,
        branch_digest=_branch(),
        expected_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        current_scope_fingerprint=_SCOPE,
        current_target_fingerprints=current_targets,
    )
    second_resolved = permuted.resolve_selector(
        permuted_ref,
        state_digest=_STATE,
        branch_digest=_branch(),
        expected_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        current_scope_fingerprint=_SCOPE,
        current_target_fingerprints=current_targets,
    )
    assert first_resolved == second_resolved
    result = lambda value: json.dumps(  # noqa: E731 - compact fixture compiler
        {"members": value.target_fingerprints}, sort_keys=True
    )
    assert result(first_resolved) == result(second_resolved)


@pytest.mark.parametrize(
    ("change", "code"),
    (
        ({"state_digest": "f" * 64}, "selector.stale_state"),
        ({"branch_digest": branch_fingerprint(_ROOT, "9" * 64)}, "selector.cross_branch"),
    ),
)
def test_state_and_branch_changes_fail_closed(change: dict, code: str) -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    built_table, ref = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=3,
    )
    descriptor = next(entry.descriptor for entry in built_table.selectors)
    kwargs = {
        "state_digest": _STATE,
        "branch_digest": _branch(),
        "expected_kind": SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        "current_scope_fingerprint": _SCOPE,
        "current_target_fingerprints": descriptor.target_fingerprints,
        **change,
    }
    with pytest.raises(ReferenceResolutionError) as raised:
        built_table.resolve_selector(ref, **kwargs)
    assert raised.value.code == code


def test_cross_request_reuse_fails_closed() -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    built_table, _ = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=3,
    )
    descriptor = next(entry.descriptor for entry in built_table.selectors)
    foreign_ref = SelectorRef("other-request", "whatever")
    with pytest.raises(ReferenceResolutionError) as raised:
        built_table.resolve_selector(
            foreign_ref,
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            current_scope_fingerprint=_SCOPE,
            current_target_fingerprints=descriptor.target_fingerprints,
        )
    assert raised.value.code == "selector.cross_request"


def test_wrong_kind_fails_closed() -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    built_table, ref = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=3,
    )
    descriptor = next(entry.descriptor for entry in built_table.selectors)
    with pytest.raises(ReferenceResolutionError) as raised:
        built_table.resolve_selector(
            ref,
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=SelectorKind.SCHEMA_ROLE_IN_SCOPE,
            current_scope_fingerprint=_SCOPE,
            current_target_fingerprints=descriptor.target_fingerprints,
        )
    assert raised.value.code == "selector.wrong_kind"


def test_changed_scope_and_changed_target_set_fail_closed() -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    built_table, ref = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=3,
    )
    descriptor = next(entry.descriptor for entry in built_table.selectors)

    with pytest.raises(ReferenceResolutionError) as raised:
        built_table.resolve_selector(
            ref,
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            current_scope_fingerprint="2" * 64,
            current_target_fingerprints=descriptor.target_fingerprints,
        )
    assert raised.value.code == "selector.scope_changed"

    with pytest.raises(ReferenceResolutionError) as raised:
        built_table.resolve_selector(
            ref,
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            current_scope_fingerprint=_SCOPE,
            current_target_fingerprints=descriptor.target_fingerprints[:-1],
        )
    assert raised.value.code == "selector.target_set_changed"


def test_current_fanout_overflow_fails_closed_on_resolve() -> None:
    table = _table(6, seed=5)
    stack_refs = _stack_refs(table)
    built_table, ref = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=len(stack_refs),
        seed=3,
    )
    descriptor = next(entry.descriptor for entry in built_table.selectors)
    inflated = (*descriptor.target_fingerprints, "0" * 64)
    with pytest.raises(ReferenceResolutionError) as raised:
        built_table.resolve_selector(
            ref,
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            current_scope_fingerprint=_SCOPE,
            current_target_fingerprints=inflated,
        )
    assert raised.value.code == "selector.fanout_overflow"


def test_duplicate_current_targets_fail_closed_on_resolve() -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    built_table, ref = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=3,
    )
    descriptor = next(entry.descriptor for entry in built_table.selectors)
    duplicated = (*descriptor.target_fingerprints, descriptor.target_fingerprints[0])
    with pytest.raises(ReferenceResolutionError) as raised:
        built_table.resolve_selector(
            ref,
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            current_scope_fingerprint=_SCOPE,
            current_target_fingerprints=duplicated,
        )
    assert raised.value.code == "selector.duplicate_target"


def test_missing_and_duplicate_selector_entries_fail_closed() -> None:
    table = _table(4)
    stack_refs = _stack_refs(table)
    built_table, ref = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=3,
    )
    with pytest.raises(ReferenceResolutionError) as raised:
        built_table.resolve_selector(
            SelectorRef("req-1", "missing-id"),
            state_digest=_STATE,
            branch_digest=_branch(),
            expected_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            current_scope_fingerprint=_SCOPE,
            current_target_fingerprints=(),
        )
    assert raised.value.code == "selector.missing"

    duplicated_entry = built_table.selectors[0]
    with pytest.raises(ReferenceResolutionError) as raised:
        ReferenceTableV1(
            request_id=built_table.request_id,
            state_digest=built_table.state_digest,
            branch_digest=built_table.branch_digest,
            entries=built_table.entries,
            selectors=(duplicated_entry, duplicated_entry),
        )
    assert raised.value.code == "ref.duplicate"


def test_nested_scopes_produce_independent_selectors() -> None:
    table = _table(6, seed=2)
    stack_refs = _stack_refs(table)
    outer, outer_ref = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=3,
    )
    inner_scope = "3" * 64
    inner, inner_ref = build_selector(
        outer,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=inner_scope,
        matching_refs=stack_refs[:1],
        max_fanout=8,
        seed=4,
    )
    assert len(inner.selectors) == 2
    # Attaching one more selector reallocates every selector's opaque ID (the
    # same permutation-safe rule entries already follow), so ``outer_ref`` is
    # stale on ``inner`` — look descriptors up by content, not by stale ref.
    outer_descriptor = next(
        entry.descriptor
        for entry in inner.selectors
        if entry.descriptor.scope_fingerprint == _SCOPE
    )
    inner_descriptor = next(
        entry.descriptor
        for entry in inner.selectors
        if entry.descriptor.scope_fingerprint == inner_scope
    )
    assert outer_descriptor.scope_fingerprint == _SCOPE
    assert inner_descriptor.scope_fingerprint == inner_scope
    assert outer_descriptor.target_fingerprints != inner_descriptor.target_fingerprints
    assert not any(entry.ref == outer_ref for entry in inner.selectors)


def test_mixed_selector_kinds_coexist_in_one_table() -> None:
    table = _table(6, seed=9)
    stack_refs = _stack_refs(table)
    all_refs = tuple(entry.ref for entry in table.entries)
    with_component = build_selector_descriptor(
        table=table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
    )
    with_descendants = build_selector_descriptor(
        table=table,
        selector_kind=SelectorKind.DESCENDANTS_WITH_ROLE,
        scope_fingerprint=_SCOPE,
        matching_refs=all_refs,
        max_fanout=8,
    )
    new_table = attach_selector(table, with_component, seed=1)
    new_table = attach_selector(new_table, with_descendants, seed=2)
    assert {entry.descriptor.selector_kind for entry in new_table.selectors} == {
        SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        SelectorKind.DESCENDANTS_WITH_ROLE,
    }


def test_truncated_scan_is_unknown_and_never_forceable() -> None:
    table = _table(6, seed=10)
    truncated = build_selector_from_scope(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        predicate=lambda entry: entry.descriptor.value_type == "openui.stack",
        max_fanout=8,
        max_candidates_scanned=1,
    )
    assert truncated.verdict is SelectorBuildVerdict.UNKNOWN
    assert truncated.descriptor is None
    assert truncated.evaluated_candidates < truncated.total_candidates
    # There is no API that turns an UNKNOWN result into a committed selector.
    assert not hasattr(truncated, "force")

    complete = build_selector_from_scope(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        predicate=lambda entry: entry.descriptor.value_type == "openui.stack",
        max_fanout=8,
        max_candidates_scanned=len(table.entries),
    )
    assert complete.verdict is SelectorBuildVerdict.RESOLVED
    assert complete.descriptor is not None
    assert complete.evaluated_candidates == complete.total_candidates


def test_context_resolver_returns_defensive_copies_of_members() -> None:
    table = _table(4, seed=6)
    stack_refs = _stack_refs(table)
    built_table, _ = build_selector(
        table,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=3,
    )
    descriptor = next(entry.descriptor for entry in built_table.selectors)
    context = SelectorContextV1(built_table)
    members_first = context.resolve_members(descriptor)
    members_second = context.resolve_members(descriptor)
    assert members_first == members_second
    assert members_first is not members_second
    assert set(members_first) == set(stack_refs)
    # Mutating the returned tuple's storage cannot reach table internals: refs
    # are frozen dataclasses, so there is no live handle to compiler state.
    with pytest.raises(TypeError):
        members_first[0] = members_first[0]  # type: ignore[index]


def test_selector_ref_table_migrates_legacy_payload_and_rejects_unknown_schema() -> None:
    table = _table(2, seed=8)
    legacy_payload = table.to_dict()
    legacy_payload["schema"] = "operator_reference_table/v1"
    del legacy_payload["selectors"]
    migrated = ReferenceTableV1.from_dict(legacy_payload)
    assert migrated.selectors == ()
    assert migrated.entries == table.entries

    unknown_payload = table.to_dict()
    unknown_payload["schema"] = "operator_reference_table/v3"
    with pytest.raises(ReferenceResolutionError) as raised:
        ReferenceTableV1.from_dict(unknown_payload)
    assert raised.value.code == "ref.table_schema_unsupported"


def test_selector_table_round_trip_contains_no_raw_query_or_surface() -> None:
    table = _table(4, seed=12)
    stack_refs = _stack_refs(table)
    built_table, _ = build_selector(
        table,
        selector_kind=SelectorKind.SYMBOL_REFERENCES,
        scope_fingerprint=_SCOPE,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=13,
        compiler_facts=(SelectorFact.EXACT_FINITE, SelectorFact.FANOUT_BOUNDED),
    )
    payload = built_table.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    assert "surface" not in encoded
    assert "display" not in encoded
    assert "query" not in encoded
    assert ReferenceTableV1.from_dict(payload) == built_table
    assert ReferenceTableV1.from_dict(payload).fingerprint == built_table.fingerprint


def test_selector_descriptor_rejects_unsorted_targets_constructed_directly() -> None:
    table = _table(4, seed=15)
    stack_refs = _stack_refs(table)
    by_ref = {entry.ref: entry.descriptor.fingerprint for entry in table.entries}
    fingerprints = tuple(sorted(by_ref[ref] for ref in stack_refs))
    assert len(fingerprints) >= 2
    with pytest.raises(ReferenceResolutionError) as raised:
        SelectorDescriptorV1(
            selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
            semantic_fingerprint="4" * 64,
            scope_fingerprint=_SCOPE,
            target_fingerprints=tuple(reversed(fingerprints)),
            cardinality=len(fingerprints),
            max_fanout=8,
        )
    assert raised.value.code == "selector.unsorted_targets"


def test_selector_entry_rejects_ref_kind_mismatch() -> None:
    from slm_training.dsl.operators import NodeRef

    descriptor = SelectorDescriptorV1(
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        semantic_fingerprint="5" * 64,
        scope_fingerprint=_SCOPE,
        target_fingerprints=(),
        cardinality=0,
        max_fanout=8,
    )
    with pytest.raises(ValueError, match="SelectorRef"):
        SelectorEntryV1(ref=NodeRef("req-1", "n1"), descriptor=descriptor)  # type: ignore[arg-type]
