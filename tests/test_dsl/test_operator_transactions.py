from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    OPERATOR_TRANSACTION_MAX_ACTIONS,
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    OperatorConflictEdgeV1,
    OperatorDependencyEdgeV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorReadWriteSetV1,
    OperatorStateV1,
    OperatorTransactionProofV1,
    OperatorTransactionRejectionKind,
    PreconditionV1,
    PreparedOperatorActionV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    SemanticArgumentV1,
    TargetFootprintV1,
    build_operator_transaction,
    build_reference_table,
    derive_read_write_set,
    prepare_operator_action,
    prepare_operator_transaction,
    replay_operator_transaction,
)
from slm_training.dsl.operators.contracts import OperatorRef, _fingerprint
from slm_training.dsl.pack import get_pack

SOURCE = (
    'root = Card([TextContent(":hero.alpha"), TextContent(":hero.beta")], "clear")'
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Fixture:
    """Real ``openui`` pack authority; synthetic single-purpose operators.

    Mirrors ``tests/test_dsl/test_operator_merge.py``'s ``_Fixture`` style:
    one real, pack-verified base state and table, plus ad hoc
    ``AstOperatorV1`` declarations whose executors return exactly the effect
    a given test wants (never a parallel execution path — each executor
    still goes through the real ``OperatorLibraryV1``/pack-authority round
    trip via ``prepare_operator_action``'s ``library.dry_run`` call).
    """

    def __init__(self) -> None:
        self.pack = get_pack("openui")
        self.state = OperatorStateV1.from_source(self.pack, SOURCE)
        self.request_id = "tx-request"
        self.branch_digest = _sha("tx-branch")
        descriptors = tuple(
            ReferenceDescriptorV1(
                ref_kind=RefKind.VALUE,
                semantic_fingerprint=_sha(name),
                value_type=f"openui.{name}",
            )
            for name in ("alpha", "beta", "gamma", "delta", "epsilon")
        )
        self.table = build_reference_table(
            request_id=self.request_id,
            state_digest=self.state.state_digest,
            branch_digest=self.branch_digest,
            descriptors=descriptors,
            seed=1,
        )
        self._registered: dict[str, RegisteredOperatorV1] = {}
        self._library: OperatorLibraryV1 | None = None
        self._pack_with_library = None

    def target(self, name: str) -> OperatorRef:
        return next(
            entry.ref
            for entry in self.table.entries
            if entry.descriptor.value_type == f"openui.{name}"
        )

    def provenance(self) -> ApplicationProvenanceV1:
        return ApplicationProvenanceV1(
            pack_id="openui",
            compiler_id="openui.transaction_fixture",
            compiler_version="v1",
            source_artifact_digest=_sha(self.state.source),
            request_id=self.request_id,
        )

    def declare(
        self,
        *,
        operator_id: str,
        write_target: OperatorRef | None = None,
        write_category: str = "property",
        read_target: OperatorRef | None = None,
        commutes_with: tuple[str, ...] = (),
        coverage: CompilerCoverage = CompilerCoverage.EXACT,
    ) -> tuple[str, tuple[BoundArgumentV1, ...]]:
        argument_slots: tuple[OperatorArgumentSlotV1, ...] = ()
        preconditions: tuple[PreconditionV1, ...] = ()
        arguments: tuple[BoundArgumentV1, ...] = ()
        if read_target is not None:
            argument_slots = (
                OperatorArgumentSlotV1("in", RefKind.VALUE, BindingPhase.STATE),
            )
            preconditions = (PreconditionV1("fixture.reads_in", ("in",)),)
            arguments = (BoundArgumentV1("in", read_target),)
        effect_signature = (
            (EffectDeltaKind(write_category),) if write_target is not None else ()
        )
        declaration = AstOperatorV1(
            operator_id=operator_id,
            version="v1",
            domain="openui.ast",
            codomain="openui.ast",
            argument_slots=argument_slots,
            preconditions=preconditions,
            effect_signature=effect_signature,
            locality="node",
            cost=1.0,
            commutes_with=commutes_with,
        )

        def execute(
            state: OperatorStateV1, _arguments: tuple[BoundArgumentV1, ...]
        ) -> OperatorMutationV1:
            if write_target is None:
                effect = ActionEffectV1(compiler_coverage=coverage)
            else:
                delta = EffectDeltaV1(
                    EffectDeltaKind(write_category), write_target, "before", "after"
                )
                effect = ActionEffectV1(
                    **{f"{write_category}_deltas": (delta,)},
                    compiler_coverage=coverage,
                )
            return OperatorMutationV1(source=state.source, effect=effect)

        self._registered[operator_id] = RegisteredOperatorV1(declaration, execute)
        self._library = None
        self._pack_with_library = None
        return operator_id, arguments

    @property
    def library(self) -> OperatorLibraryV1:
        if self._library is None:
            self._library = OperatorLibraryV1(tuple(self._registered.values()))
        return self._library

    @property
    def pack_with_library(self):
        if self._pack_with_library is None:
            self._pack_with_library = replace(
                self.pack, operator_library=self.library
            )
        return self._pack_with_library

    def prepare(
        self, operator_id: str, arguments: tuple[BoundArgumentV1, ...]
    ) -> PreparedOperatorActionV1:
        return prepare_operator_action(
            pack=self.pack_with_library,
            library=self.library,
            state=self.state,
            reference_table=self.table,
            operator_id=operator_id,
            arguments=arguments,
            provenance=self.provenance(),
        )

    def requests_transaction(
        self, requests: tuple[tuple[str, tuple[BoundArgumentV1, ...]], ...]
    ):
        return prepare_operator_transaction(
            pack=self.pack_with_library,
            library=self.library,
            state=self.state,
            reference_table=self.table,
            requests=requests,
            provenance=self.provenance(),
        )

    def build(self, prepared: tuple[PreparedOperatorActionV1, ...]):
        return build_operator_transaction(
            library=self.library,
            state=self.state,
            reference_table=self.table,
            prepared_actions=prepared,
            provenance=self.provenance(),
        )


def test_disjoint_property_edits_build_one_transaction_no_conflicts() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.tx_write_alpha", write_target=fixture.target("alpha")
    )
    write_b = fixture.declare(
        operator_id="openui.tx_write_beta", write_target=fixture.target("beta")
    )
    decision = fixture.requests_transaction((write_a, write_b))
    assert decision.succeeded
    transaction = decision.transaction
    assert transaction is not None
    assert len(transaction.prepared_actions) == 2
    assert transaction.dependency_edges == ()
    assert transaction.conflict_graph == ()
    assert transaction.coverage is CompilerCoverage.EXACT
    ids = tuple(action.semantic_action_id for action in transaction.prepared_actions)
    assert ids == tuple(sorted(ids))
    assert transaction.proof.composite_effect_fingerprint == (
        transaction.composite_effect.fingerprint
    )


def test_declared_commuting_write_write_overlap_is_resolved_with_a_dependency_edge() -> (
    None
):
    fixture = _Fixture()
    left = fixture.declare(
        operator_id="openui.tx_commute_left",
        write_target=fixture.target("alpha"),
        commutes_with=("openui.tx_commute_right",),
    )
    right = fixture.declare(
        operator_id="openui.tx_commute_right",
        write_target=fixture.target("alpha"),
        commutes_with=("openui.tx_commute_left",),
    )
    decision = fixture.requests_transaction((left, right))
    assert decision.succeeded
    transaction = decision.transaction
    assert transaction is not None
    assert len(transaction.conflict_graph) == 1
    assert transaction.conflict_graph[0].resolved_by_commutativity
    assert len(transaction.dependency_edges) == 1
    assert transaction.dependency_edges[0].reason == "write_write_order"


def test_write_write_conflict_without_commuting_rejects() -> None:
    fixture = _Fixture()
    left = fixture.declare(
        operator_id="openui.tx_conflict_left", write_target=fixture.target("alpha")
    )
    right = fixture.declare(
        operator_id="openui.tx_conflict_right", write_target=fixture.target("alpha")
    )
    decision = fixture.requests_transaction((left, right))
    assert not decision.succeeded
    assert decision.rejection is not None
    assert decision.rejection.kind is OperatorTransactionRejectionKind.CONFLICT
    assert len(decision.rejection.target_fingerprints) == 1


def test_write_read_conflict_without_commuting_rejects() -> None:
    fixture = _Fixture()
    writer = fixture.declare(
        operator_id="openui.tx_writer", write_target=fixture.target("alpha")
    )
    reader = fixture.declare(
        operator_id="openui.tx_reader",
        write_target=fixture.target("beta"),
        read_target=fixture.target("alpha"),
    )
    decision = fixture.requests_transaction((writer, reader))
    assert not decision.succeeded
    assert decision.rejection is not None
    assert decision.rejection.kind is OperatorTransactionRejectionKind.CONFLICT


def test_dependency_chain_orders_actions_by_write_before_read() -> None:
    fixture = _Fixture()
    first = fixture.declare(
        operator_id="openui.tx_chain_a",
        write_target=fixture.target("alpha"),
        commutes_with=("openui.tx_chain_b",),
    )
    second = fixture.declare(
        operator_id="openui.tx_chain_b",
        write_target=fixture.target("beta"),
        read_target=fixture.target("alpha"),
        commutes_with=("openui.tx_chain_a", "openui.tx_chain_c"),
    )
    third = fixture.declare(
        operator_id="openui.tx_chain_c",
        write_target=fixture.target("gamma"),
        read_target=fixture.target("beta"),
        commutes_with=("openui.tx_chain_b",),
    )
    decision = fixture.requests_transaction((first, second, third))
    assert decision.succeeded
    transaction = decision.transaction
    assert transaction is not None
    assert len(transaction.dependency_edges) == 2
    assert all(
        edge.reason == "write_before_read" for edge in transaction.dependency_edges
    )
    edge_pairs = {
        (edge.predecessor_semantic_id, edge.successor_semantic_id)
        for edge in transaction.dependency_edges
    }
    id_by_operator = {
        action.operator_fingerprint: action.semantic_action_id
        for action in transaction.prepared_actions
    }
    a_id = id_by_operator[fixture.library.lookup("openui.tx_chain_a").fingerprint]
    b_id = id_by_operator[fixture.library.lookup("openui.tx_chain_b").fingerprint]
    c_id = id_by_operator[fixture.library.lookup("openui.tx_chain_c").fingerprint]
    assert (a_id, b_id) in edge_pairs
    assert (b_id, c_id) in edge_pairs


def test_mutual_write_read_overlap_forms_a_dependency_cycle_and_rejects() -> None:
    fixture = _Fixture()
    left = fixture.declare(
        operator_id="openui.tx_cycle_left",
        write_target=fixture.target("alpha"),
        read_target=fixture.target("beta"),
        commutes_with=("openui.tx_cycle_right",),
    )
    right = fixture.declare(
        operator_id="openui.tx_cycle_right",
        write_target=fixture.target("beta"),
        read_target=fixture.target("alpha"),
        commutes_with=("openui.tx_cycle_left",),
    )
    decision = fixture.requests_transaction((left, right))
    assert not decision.succeeded
    assert decision.rejection is not None
    assert decision.rejection.kind is OperatorTransactionRejectionKind.DEPENDENCY_CYCLE


def test_unknown_effect_and_approximate_coverage_reject_as_unknown_footprint() -> None:
    fixture = _Fixture()
    exact = fixture.declare(
        operator_id="openui.tx_exact", write_target=fixture.target("alpha")
    )
    approximate = fixture.declare(
        operator_id="openui.tx_approximate",
        write_target=fixture.target("beta"),
        coverage=CompilerCoverage.APPROXIMATE,
    )
    decision = fixture.requests_transaction((exact, approximate))
    assert not decision.succeeded
    assert decision.rejection is not None
    assert (
        decision.rejection.kind is OperatorTransactionRejectionKind.UNKNOWN_FOOTPRINT
    )


def test_action_set_permutation_shares_canonical_transaction_identity() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.tx_perm_a", write_target=fixture.target("alpha")
    )
    write_b = fixture.declare(
        operator_id="openui.tx_perm_b", write_target=fixture.target("beta")
    )
    write_c = fixture.declare(
        operator_id="openui.tx_perm_c", write_target=fixture.target("gamma")
    )
    forward = fixture.requests_transaction((write_a, write_b, write_c))
    backward = fixture.requests_transaction((write_c, write_b, write_a))
    assert forward.succeeded and backward.succeeded
    assert forward.decision_id == backward.decision_id
    assert forward.transaction is not None and backward.transaction is not None
    assert (
        forward.transaction.transaction_id == backward.transaction.transaction_id
    )


def test_conflict_derives_from_target_fingerprint_not_operator_name() -> None:
    fixture = _Fixture()
    # Two entirely differently-named operators, no accidental name overlap,
    # writing the exact same semantic target: must still conflict.
    left = fixture.declare(
        operator_id="openui.totally_unrelated_name_one",
        write_target=fixture.target("alpha"),
    )
    right = fixture.declare(
        operator_id="openui.totally_unrelated_name_two",
        write_target=fixture.target("alpha"),
    )
    decision = fixture.requests_transaction((left, right))
    assert not decision.succeeded
    assert decision.rejection is not None
    assert decision.rejection.kind is OperatorTransactionRejectionKind.CONFLICT


def test_mixed_bases_reject_as_stale_base() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.tx_base_a", write_target=fixture.target("alpha")
    )
    prepared_a = fixture.prepare(*write_a)

    other_source = (
        'root = Card([TextContent(":hero.other")], "clear")'
    )
    other_state = OperatorStateV1.from_source(fixture.pack, other_source)
    other_table = build_reference_table(
        request_id=fixture.request_id,
        state_digest=other_state.state_digest,
        branch_digest=fixture.branch_digest,
        descriptors=(
            ReferenceDescriptorV1(
                ref_kind=RefKind.VALUE,
                semantic_fingerprint=_sha("omega"),
                value_type="openui.omega",
            ),
        ),
        seed=7,
    )
    other_ref = next(entry.ref for entry in other_table.entries)
    write_b = fixture.declare(
        operator_id="openui.tx_base_b", write_target=other_ref
    )
    prepared_b = prepare_operator_action(
        pack=fixture.pack_with_library,
        library=fixture.library,
        state=other_state,
        reference_table=other_table,
        operator_id=write_b[0],
        arguments=write_b[1],
        provenance=ApplicationProvenanceV1(
            pack_id="openui",
            compiler_id="openui.transaction_fixture",
            compiler_version="v1",
            source_artifact_digest=_sha(other_state.source),
            request_id=fixture.request_id,
        ),
    )
    decision = fixture.build((prepared_a, prepared_b))
    assert not decision.succeeded
    assert decision.rejection is not None
    assert decision.rejection.kind is OperatorTransactionRejectionKind.STALE_BASE


def test_stale_reference_table_rejects_when_preparing_an_action() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.tx_stale", write_target=fixture.target("alpha")
    )
    stale_table = build_reference_table(
        request_id=fixture.request_id,
        state_digest=_sha("not-the-real-state"),
        branch_digest=fixture.branch_digest,
        descriptors=(),
        seed=1,
    )
    with pytest.raises(ValueError, match="stale"):
        prepare_operator_action(
            pack=fixture.pack_with_library,
            library=fixture.library,
            state=fixture.state,
            reference_table=stale_table,
            operator_id=write_a[0],
            arguments=write_a[1],
            provenance=fixture.provenance(),
        )


def test_partial_preparation_rejects_the_whole_batch_not_just_the_failure() -> None:
    fixture = _Fixture()
    good = fixture.declare(
        operator_id="openui.tx_partial_good", write_target=fixture.target("alpha")
    )
    # A declared operator with a precondition-bearing slot, but the caller
    # supplies no matching argument at all -> validate_arguments raises ->
    # dry_run comes back rejected, never succeeded.
    bad_id, _ = fixture.declare(
        operator_id="openui.tx_partial_bad",
        write_target=fixture.target("beta"),
        read_target=fixture.target("gamma"),
    )
    bad = (bad_id, ())
    decision = fixture.requests_transaction((good, bad))
    assert not decision.succeeded
    assert decision.rejection is not None
    assert (
        decision.rejection.kind is OperatorTransactionRejectionKind.PARTIAL_PREPARATION
    )


def test_opaque_id_hidden_duplicate_action_rejects_via_semantic_identity() -> None:
    fixture = _Fixture()
    operator_id, first_arguments = fixture.declare(
        operator_id="openui.tx_dup", read_target=fixture.target("alpha")
    )
    prepared_first = fixture.prepare(operator_id, first_arguments)

    # A different, freshly-permuted reference table (DSH3-03 permutation):
    # same request/state identity, but every opaque ID is reallocated.
    permuted_table = fixture.table.permuted(seed=99)
    permuted_alpha = next(
        entry.ref
        for entry in permuted_table.entries
        if entry.descriptor.value_type == "openui.alpha"
    )
    assert permuted_alpha.opaque_id != fixture.target("alpha").opaque_id
    prepared_second = prepare_operator_action(
        pack=fixture.pack_with_library,
        library=fixture.library,
        state=fixture.state,
        reference_table=permuted_table,
        operator_id=operator_id,
        arguments=(BoundArgumentV1("in", permuted_alpha),),
        provenance=fixture.provenance(),
    )
    # Different opaque wrapper, identical semantic target -> identical
    # semantic identity: opaque IDs never hide a duplicate action.
    assert prepared_first.semantic_action_id == prepared_second.semantic_action_id
    decision = fixture.build((prepared_first, prepared_second))
    assert not decision.succeeded
    assert decision.rejection is not None
    assert decision.rejection.kind is OperatorTransactionRejectionKind.DUPLICATE_ACTION


def test_fanout_exceeded_rejects() -> None:
    fixture = _Fixture()
    requests = []
    for index in range(OPERATOR_TRANSACTION_MAX_ACTIONS + 1):
        operator_id = f"openui.tx_fanout_{index}"
        requests.append(
            fixture.declare(operator_id=operator_id, write_target=None)
        )
    decision = fixture.requests_transaction(tuple(requests))
    assert not decision.succeeded
    assert decision.rejection is not None
    assert decision.rejection.kind is OperatorTransactionRejectionKind.FANOUT_EXCEEDED


def test_noncanonical_prepared_action_order_raises() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.tx_order_a", write_target=fixture.target("alpha")
    )
    write_b = fixture.declare(
        operator_id="openui.tx_order_b", write_target=fixture.target("beta")
    )
    decision = fixture.requests_transaction((write_a, write_b))
    assert decision.transaction is not None
    transaction = decision.transaction
    with pytest.raises(ValueError, match="canonically ordered"):
        replace(
            transaction, prepared_actions=tuple(reversed(transaction.prepared_actions))
        )


def test_noncanonical_dependency_and_conflict_graph_order_raises() -> None:
    fixture = _Fixture()
    first = fixture.declare(
        operator_id="openui.tx_triangle_a",
        write_target=fixture.target("alpha"),
        commutes_with=("openui.tx_triangle_b", "openui.tx_triangle_c"),
    )
    second = fixture.declare(
        operator_id="openui.tx_triangle_b",
        write_target=fixture.target("alpha"),
        commutes_with=("openui.tx_triangle_a", "openui.tx_triangle_c"),
    )
    third = fixture.declare(
        operator_id="openui.tx_triangle_c",
        write_target=fixture.target("alpha"),
        commutes_with=("openui.tx_triangle_a", "openui.tx_triangle_b"),
    )
    decision = fixture.requests_transaction((first, second, third))
    assert decision.succeeded
    transaction = decision.transaction
    assert transaction is not None
    assert len(transaction.dependency_edges) == 3
    assert len(transaction.conflict_graph) == 3
    with pytest.raises(ValueError, match="dependency edges must be canonically"):
        replace(
            transaction,
            dependency_edges=tuple(reversed(transaction.dependency_edges)),
        )
    with pytest.raises(ValueError, match="conflict graph must be canonically"):
        replace(
            transaction, conflict_graph=tuple(reversed(transaction.conflict_graph))
        )


def test_dependency_cycle_direct_construction_raises() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.tx_direct_cycle_a", write_target=fixture.target("alpha")
    )
    write_b = fixture.declare(
        operator_id="openui.tx_direct_cycle_b", write_target=fixture.target("beta")
    )
    decision = fixture.requests_transaction((write_a, write_b))
    assert decision.transaction is not None
    transaction = decision.transaction
    id0, id1 = (action.semantic_action_id for action in transaction.prepared_actions)
    filler_target = transaction.prepared_actions[0].effect.property_deltas[0].target
    filler_fingerprint = next(
        entry.descriptor.fingerprint
        for entry in fixture.table.entries
        if entry.ref == filler_target
    )
    cyclic_edges = (
        OperatorDependencyEdgeV1(id0, id1, "write_before_read", (filler_fingerprint,)),
        OperatorDependencyEdgeV1(id1, id0, "write_before_read", (filler_fingerprint,)),
    )
    with pytest.raises(ValueError, match="cycle"):
        replace(transaction, dependency_edges=cyclic_edges)


def test_no_execution_is_ever_wired_from_this_module() -> None:
    from slm_training.dsl.operators import transactions as transactions_module

    forbidden = {"apply_operator_transaction", "commit_operator_transaction", "execute"}
    assert not forbidden & set(transactions_module.__all__)

    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.tx_no_exec", write_target=fixture.target("alpha")
    )
    decision = fixture.requests_transaction((write_a,))
    assert decision.succeeded
    # Preparing/building a transaction never touches pack-authorized source.
    assert fixture.state.source == SOURCE
    assert (
        OperatorStateV1.from_source(fixture.pack, fixture.state.source)
        == fixture.state
    )


def test_serialization_round_trips_and_migration_guards_reject_unknown_schema() -> (
    None
):
    footprint = TargetFootprintV1(_sha("target"), ("property", "role"))
    assert TargetFootprintV1.from_dict(footprint.to_dict()) == footprint
    read_write = OperatorReadWriteSetV1(reads=(footprint,), writes=())
    assert OperatorReadWriteSetV1.from_dict(read_write.to_dict()) == read_write
    assert read_write.fingerprint == OperatorReadWriteSetV1.from_dict(
        read_write.to_dict()
    ).fingerprint

    argument = SemanticArgumentV1("slot", RefKind.VALUE, _sha("arg"))
    assert SemanticArgumentV1.from_dict(argument.to_dict()) == argument

    edge = OperatorDependencyEdgeV1(
        _sha("pred"), _sha("succ"), "write_before_read", (_sha("t"),)
    )
    assert OperatorDependencyEdgeV1.from_dict(edge.to_dict()) == edge

    conflict = OperatorConflictEdgeV1(
        *sorted((_sha("left"), _sha("right"))), (_sha("t"),), True
    )
    assert OperatorConflictEdgeV1.from_dict(conflict.to_dict()) == conflict

    stale_payload = dict(footprint.to_dict())
    stale_payload["schema"] = "operator_target_footprint/v0"
    with pytest.raises(ValueError, match="schema_unsupported"):
        TargetFootprintV1.from_dict(stale_payload)


def test_transaction_proof_checks_are_order_and_duplicate_stable() -> None:
    unsorted_proof = OperatorTransactionProofV1(
        proof_kind="transaction.prepared",
        checks=(
            "order.canonical",
            "base.consistent",
            "base.consistent",
            "footprints.exact",
        ),
        transaction_result_digest=_sha("result"),
        composite_effect_fingerprint=_sha("effect"),
    )
    assert unsorted_proof.checks == (
        "base.consistent",
        "footprints.exact",
        "order.canonical",
    )
    assert unsorted_proof.to_dict()["checks"] == list(unsorted_proof.checks)
    assert (
        OperatorTransactionProofV1.from_dict(unsorted_proof.to_dict())
        == unsorted_proof
    )


def test_derive_read_write_set_reuses_merge_style_write_categorization() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.tx_rw_write",
        write_target=fixture.target("alpha"),
        read_target=fixture.target("beta"),
    )
    prepared = fixture.prepare(*write_a)
    declaration = fixture.library.lookup("openui.tx_rw_write")
    lineage = {entry.ref: entry.descriptor.fingerprint for entry in fixture.table.entries}
    read_write = derive_read_write_set(
        declaration, prepared.dry_run.arguments, prepared.effect, lineage
    )
    assert read_write is not None
    assert read_write.write_targets == {
        lineage[fixture.target("alpha")]
    }
    assert read_write.read_targets == {lineage[fixture.target("beta")]}
    # Non-exact coverage is an unknown footprint, never a partial guess.
    inexact_effect = ActionEffectV1(compiler_coverage=CompilerCoverage.APPROXIMATE)
    assert (
        derive_read_write_set(
            declaration, prepared.dry_run.arguments, inexact_effect, lineage
        )
        is None
    )


def test_replay_operator_transaction_matches_recorded_decision() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.tx_replay_a", write_target=fixture.target("alpha")
    )
    write_b = fixture.declare(
        operator_id="openui.tx_replay_b", write_target=fixture.target("beta")
    )
    prepared = (fixture.prepare(*write_a), fixture.prepare(*write_b))
    recorded = fixture.build(prepared)
    assert recorded.succeeded
    replayed = replay_operator_transaction(
        library=fixture.library,
        state=fixture.state,
        reference_table=fixture.table,
        prepared_actions=prepared,
        provenance=fixture.provenance(),
        recorded=recorded,
    )
    assert replayed.decision_id == recorded.decision_id

    tampered = fixture.build(prepared[:1])
    with pytest.raises(ValueError, match="differs from recorded"):
        replay_operator_transaction(
            library=fixture.library,
            state=fixture.state,
            reference_table=fixture.table,
            prepared_actions=prepared,
            provenance=fixture.provenance(),
            recorded=tampered,
        )


def test_transaction_and_rejection_decision_ids_are_deterministic_digests() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.tx_digest_a", write_target=fixture.target("alpha")
    )
    decision = fixture.requests_transaction((write_a,))
    assert decision.succeeded
    assert decision.transaction is not None
    assert decision.decision_id == _fingerprint(
        decision.to_dict(include_decision_id=False)
    )
    assert decision.transaction.transaction_id == _fingerprint(
        decision.transaction.to_dict()
    )
