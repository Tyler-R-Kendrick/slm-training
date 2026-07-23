from __future__ import annotations

import hashlib
import itertools
from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
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
    deserialize_operator_action,
    enumerate_operator_legal_set,
    serialize_operator_action,
)
from slm_training.dsl.pack import get_pack

SOURCE = 'root = TextContent(":hero.title")'
UPDATED = 'root = TextContent(":hero.body")'
OPERATOR_ID = "openui.fixture_legal_set"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _descriptor(index: int) -> ReferenceDescriptorV1:
    return ReferenceDescriptorV1(
        ref_kind=RefKind.VALUE,
        semantic_fingerprint=_sha(f"semantic-value-{index}"),
        value_type="openui.string",
    )


def _fixture(
    *,
    count: int,
    allowed_indices: frozenset[int] = frozenset(),
    seed: int = 7,
    repeated: bool = False,
):
    base_pack = get_pack("openui")
    state = OperatorStateV1.from_source(base_pack, SOURCE)
    descriptors = tuple(_descriptor(index) for index in range(count))
    table = build_reference_table(
        request_id="request-1",
        state_digest=state.state_digest,
        branch_digest=_sha("branch"),
        descriptors=descriptors,
        seed=seed,
    )
    semantic_by_ref = {
        entry.ref: entry.descriptor.semantic_fingerprint for entry in table.entries
    }
    allowed = {
        _descriptor(index).semantic_fingerprint for index in allowed_indices
    }
    calls: list[str] = []

    def execute(operator_state, arguments):
        semantic = semantic_by_ref[arguments[0].value]
        calls.append(semantic)
        if semantic not in allowed:
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
                "value",
                RefKind.VALUE,
                BindingPhase.APPLICATION,
                repeated=repeated,
            ),
        ),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(base_pack, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(SOURCE),
        request_id="request-1",
    )
    return pack, library, state, table, provenance, calls


def _enumerate(fixture, *, maximum: int = 10_000, ordinary=()):
    pack, library, state, table, provenance, _ = fixture
    return enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=table,
        provenance=provenance,
        ordinary_nonoperator_actions=ordinary,
        max_combinations_per_operator=maximum,
    )


def test_exact_small_legal_set_matches_independent_brute_force() -> None:
    fixture = _fixture(count=4, allowed_indices=frozenset({1, 3}))
    result = _enumerate(fixture)
    pack, library, state, table, provenance, _ = fixture
    declaration = library.lookup(OPERATOR_ID)
    brute_force = set()
    for entry in table.entries:
        arguments = declaration.validate_arguments(
            (
                deserialize_operator_action(
                    serialize_operator_action(
                        OPERATOR_ID,
                        (BoundArgumentV1("value", entry.ref),),
                    )
                )[1][0],
            )
        )
        application = library.dry_run(
            pack, state, OPERATOR_ID, arguments, provenance
        )
        if application.succeeded:
            brute_force.add(application.application_id)

    assert result.coverage is LegalSetCoverage.COMPLETE
    assert result.legal_operator_ids == (OPERATOR_ID,)
    assert {action.application_id for action in result.operator_actions} == brute_force
    assert all(action.proof_fingerprint for action in result.operator_actions)
    assert all("pack.parse" in action.proof_checks for action in result.operator_actions)
    assert all(
        library.dry_run(pack, state, action.operator_id, action.arguments, provenance)
        .succeeded
        for action in result.operator_actions
    )


def test_complete_empty_domain_is_exactly_hard_prunable() -> None:
    result = _enumerate(_fixture(count=0))
    entry = result.entries[0]
    assert entry.verdict is OperatorSupportVerdict.UNSUPPORTED
    assert entry.coverage is LegalSetCoverage.COMPLETE
    assert entry.total_combinations == 0
    assert result.hard_prunable_operator_ids == (OPERATOR_ID,)
    assert result.operator_actions == ()


def test_budget_truncation_is_lazy_unknown_and_never_hard_prunes() -> None:
    late_index = max(range(100), key=lambda index: _descriptor(index).fingerprint)
    fixture = _fixture(count=100, allowed_indices=frozenset({late_index}))
    result = _enumerate(fixture, maximum=3)
    entry = result.entries[0]
    assert len(fixture[-1]) == 3
    assert entry.evaluated_combinations == 3
    assert entry.total_combinations == 100
    assert entry.coverage is LegalSetCoverage.PARTIAL
    assert entry.verdict is OperatorSupportVerdict.UNKNOWN
    assert result.hard_prunable_operator_ids == ()
    assert result.retained_operator_ids == (OPERATOR_ID,)
    assert result.forced_action is None

    pack, library, state, table, provenance, _ = fixture
    late_ref = next(
        candidate.ref
        for candidate in table.entries
        if candidate.descriptor.semantic_fingerprint
        == _descriptor(late_index).semantic_fingerprint
    )
    assert library.dry_run(
        pack,
        state,
        OPERATOR_ID,
        (BoundArgumentV1("value", late_ref),),
        provenance,
    ).succeeded


def test_partial_witness_stays_supported_but_never_forces_singleton() -> None:
    fixture = _fixture(count=8, allowed_indices=frozenset(range(8)))
    result = _enumerate(fixture, maximum=1)
    assert result.entries[0].verdict is OperatorSupportVerdict.SUPPORTED
    assert result.entries[0].coverage is LegalSetCoverage.PARTIAL
    assert len(result.operator_actions) == 1
    assert result.hard_prunable_operator_ids == ()
    assert result.forced_action is None


def test_repeated_unbounded_slot_is_unknown_without_execution() -> None:
    fixture = _fixture(count=3, repeated=True)
    result = _enumerate(fixture)
    entry = result.entries[0]
    assert fixture[-1] == []
    assert entry.coverage is LegalSetCoverage.PARTIAL
    assert entry.verdict is OperatorSupportVerdict.UNKNOWN
    assert dict(entry.rejection_counts) == {
        "operator.repeated_slot_unbounded": 1
    }
    assert result.hard_prunable_operator_ids == ()


def test_ref_and_candidate_permutation_preserve_semantic_legal_membership() -> None:
    first = _enumerate(
        _fixture(count=6, allowed_indices=frozenset({0, 2, 5}), seed=3)
    )
    second = _enumerate(
        _fixture(count=6, allowed_indices=frozenset({0, 2, 5}), seed=91)
    )
    assert first.state_fingerprint == second.state_fingerprint
    assert first.registry_fingerprint == second.registry_fingerprint
    assert first.legal_operator_ids == second.legal_operator_ids
    assert {action.semantic_id for action in first.operator_actions} == {
        action.semantic_id for action in second.operator_actions
    }

    first_partial = _enumerate(
        _fixture(count=6, allowed_indices=frozenset(range(6)), seed=3),
        maximum=2,
    )
    second_partial = _enumerate(
        _fixture(count=6, allowed_indices=frozenset(range(6)), seed=91),
        maximum=2,
    )
    assert {action.semantic_id for action in first_partial.operator_actions} == {
        action.semantic_id for action in second_partial.operator_actions
    }


def test_reserved_serialization_is_canonical_typed_and_strict() -> None:
    result = _enumerate(
        _fixture(count=1, allowed_indices=frozenset({0}))
    )
    action = result.operator_actions[0]
    operator_id, arguments = deserialize_operator_action(action.serialized)
    assert operator_id == OPERATOR_ID
    assert arguments == action.arguments
    assert serialize_operator_action(operator_id, arguments) == action.serialized
    assert action.serialized.startswith(f"OPERATOR {OPERATOR_ID} value=value:")

    for malformed in (
        "",
        "operator openui.fixture",
        "OPERATOR",
        "OPERATOR OpenUI.bad",
        f"OPERATOR {OPERATOR_ID} value=bogus:req:ref",
        f"OPERATOR {OPERATOR_ID} value=value:req:ref value=value:req:other",
    ):
        with pytest.raises(ValueError):
            deserialize_operator_action(malformed)


def test_complete_singleton_force_emit_preserves_all_ordinary_actions() -> None:
    operator_only = _enumerate(
        _fixture(count=1, allowed_indices=frozenset({0}))
    )
    assert operator_only.forced_action == operator_only.operator_actions[0].serialized

    ordinary = ("GRAMMAR alpha", "TOKEN beta", "GRAMMAR alpha")
    mixed = _enumerate(
        _fixture(count=1, allowed_indices=frozenset({0})),
        ordinary=ordinary,
    )
    assert mixed.ordinary_nonoperator_actions == ordinary
    assert mixed.all_serialized_actions[:3] == ordinary
    assert mixed.forced_action is None


def test_declaration_order_defines_hierarchical_product_not_input_order() -> None:
    fixture = _fixture(count=3, allowed_indices=frozenset({0, 1, 2}))
    result = _enumerate(fixture)
    domains = result.entries[0].argument_domains
    semantic_by_ref = {
        entry.ref: entry.descriptor.fingerprint for entry in fixture[3].entries
    }
    assert [semantic_by_ref[ref] for ref in domains[0].candidates] == sorted(
        semantic_by_ref.values()
    )
    assert len(tuple(itertools.product(domains[0].candidates))) == 3
