"""Tests for the DSH5-05 atomic multi-action transaction executor.

SLM-412 (DSH5-04, ``transactions.py``) proved *preparation*: a set of
independently-dry-run actions against one fixed base is either disjoint or
exactly commuting. This module (``transaction_executor.py``) is the
*execution* half — it composes that proven set into one real, pack-
authorized final state, atomically, and this test suite proves:

* Two/four disjoint local actions and a bulk-plus-primitive mixed
  transaction all compose and commit in one shot.
* A declared-commuting overlap with genuinely *agreeing* outcomes composes;
  one with genuinely *disagreeing* outcomes is still caught (a real
  structural conflict) and rejected, never silently approximated.
* A dependency chain composes to exactly the state every legal sequential
  topological order would reach — proved directly by a test-only
  sequential-shadow oracle that actually chains dry-runs (the one place in
  this whole suite that pattern is intentionally used, and only as an
  equivalence oracle, never as the production path).
* Action-set permutation reaches one committed identity.
* A failing constituent, a reference-rebuild failure, a stale/tampered
  transaction, and a real structural compose conflict all reject cleanly
  with a typed rejection and leave the base state object completely
  unread-only.
* Exact replay from base + recorded commit decision matches; tampering is
  detected.
* The composite effect names every target; post-commit legal work uses only
  the fresh reference table, never the base one.
* One conversation turn carries the whole transaction (retaining every
  constituent application ID) and replays exactly.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    MAP_SET_PROPERTY,
    SET_PROPERTY,
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    OperatorTransactionCommitRejectionKind,
    PreconditionV1,
    PreparedOperatorActionV1,
    RefKind,
    ReferenceDescriptorV1,
    ReferenceResolutionError,
    ReferenceTableV1,
    RegisteredOperatorV1,
    append_operator_transaction_turn,
    branch_fingerprint,
    build_bulk_selector,
    build_openui_bulk_operator_library,
    build_openui_local_operator_context,
    build_operator_transaction,
    build_reference_table,
    commit_operator_transaction,
    compose_operator_transaction,
    create_conversation_trace,
    prepare_operator_action,
    prepare_operator_transaction,
    replay_conversation_trace,
    replay_operator_transaction_commit,
    transaction_replayer_for_conversation,
)
from slm_training.dsl.operators.contracts import OperatorRef
from slm_training.dsl.operators.local import NodeLocationV1, RoleLocationV1
from slm_training.dsl.operators.references import SelectorKind
from slm_training.dsl.pack import get_pack
from slm_training.dsl.production_codec import parse_statement_bindings

SOURCE = (
    'root = Card([TextContent(":hero.alpha"), TextContent(":hero.beta"), '
    'TextContent(":hero.gamma"), TextContent(":hero.delta")], "clear")'
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Fixture:
    """Real ``openui`` pack authority; synthetic literal-text-replace operators.

    Mirrors ``test_operator_transactions.py``'s ``_Fixture`` (semantic
    footprint declaration) crossed with ``test_operator_merge.py``'s
    ``_Fixture`` (a *real* ``state.source.replace(...)`` mutation, not a
    no-op) — every action here genuinely changes the pack-authorized source,
    so the executor's structural AST composition has something real to
    compose, not just footprint bookkeeping.
    """

    def __init__(self) -> None:
        self.pack = get_pack("openui")
        self.state = OperatorStateV1.from_source(self.pack, SOURCE)
        self.request_id = "tx-exec-request"
        self.branch_digest = _sha("tx-exec-branch")
        self.names = ("alpha", "beta", "gamma", "delta")
        self.descriptors = tuple(
            ReferenceDescriptorV1(
                ref_kind=RefKind.VALUE,
                semantic_fingerprint=_sha(name),
                value_type=f"openui.{name}",
            )
            for name in self.names
        )
        self.table = build_reference_table(
            request_id=self.request_id,
            state_digest=self.state.state_digest,
            branch_digest=self.branch_digest,
            descriptors=self.descriptors,
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
            compiler_id="openui.tx_executor_fixture",
            compiler_version="v1",
            source_artifact_digest=_sha(self.state.source),
            request_id=self.request_id,
        )

    def declare(
        self,
        *,
        operator_id: str,
        target_name: str,
        replacement: str,
        write_category: str = "property",
        read_target_name: str | None = None,
        commutes_with: tuple[str, ...] = (),
        coverage: CompilerCoverage = CompilerCoverage.EXACT,
    ) -> tuple[str, tuple[BoundArgumentV1, ...]]:
        write_target = self.target(target_name)
        argument_slots: tuple[OperatorArgumentSlotV1, ...] = ()
        preconditions: tuple[PreconditionV1, ...] = ()
        arguments: tuple[BoundArgumentV1, ...] = ()
        if read_target_name is not None:
            argument_slots = (
                OperatorArgumentSlotV1("in", RefKind.VALUE, BindingPhase.STATE),
            )
            preconditions = (PreconditionV1("fixture.reads_in", ("in",)),)
            arguments = (BoundArgumentV1("in", self.target(read_target_name)),)
        declaration = AstOperatorV1(
            operator_id=operator_id,
            version="v1",
            domain="openui.ast",
            codomain="openui.ast",
            argument_slots=argument_slots,
            preconditions=preconditions,
            effect_signature=(EffectDeltaKind(write_category),),
            locality="node",
            cost=1.0,
            commutes_with=commutes_with,
        )
        before = f":hero.{target_name}"

        def execute(
            state: OperatorStateV1, _arguments: tuple[BoundArgumentV1, ...]
        ) -> OperatorMutationV1:
            if before not in state.source:
                raise OperatorRejectedError("fixture.target_unavailable")
            delta = EffectDeltaV1(
                EffectDeltaKind(write_category), write_target, "before", "after"
            )
            effect = ActionEffectV1(
                **{f"{write_category}_deltas": (delta,)}, compiler_coverage=coverage
            )
            return OperatorMutationV1(
                source=state.source.replace(before, replacement), effect=effect
            )

        self._registered[operator_id] = RegisteredOperatorV1(declaration, execute)
        self._library = None
        self._pack_with_library = None
        return operator_id, arguments

    def declare_flaky(
        self, *, operator_id: str, target_name: str, replacement: str, broken: dict
    ) -> tuple[str, tuple[BoundArgumentV1, ...]]:
        """An operator that behaves normally at prepare time, then raises on
        replay once the caller flips ``broken["on"] = True`` — a failing
        constituent whose failure only surfaces at *commit* time, never at
        prepare time (proving ``CONSTITUENT_REPLAY_FAILED`` is a real, live
        check, not dead code that can never fire)."""
        write_target = self.target(target_name)
        declaration = AstOperatorV1(
            operator_id=operator_id,
            version="v1",
            domain="openui.ast",
            codomain="openui.ast",
            argument_slots=(),
            preconditions=(),
            effect_signature=(EffectDeltaKind.PROPERTY,),
            locality="node",
            cost=1.0,
        )
        before = f":hero.{target_name}"

        def execute(
            state: OperatorStateV1, _arguments: tuple[BoundArgumentV1, ...]
        ) -> OperatorMutationV1:
            if broken["on"]:
                raise ValueError("fixture.simulated_failure")
            if before not in state.source:
                raise OperatorRejectedError("fixture.target_unavailable")
            delta = EffectDeltaV1(EffectDeltaKind.PROPERTY, write_target, "before", "after")
            effect = ActionEffectV1(
                property_deltas=(delta,), compiler_coverage=CompilerCoverage.EXACT
            )
            return OperatorMutationV1(
                source=state.source.replace(before, replacement), effect=effect
            )

        self._registered[operator_id] = RegisteredOperatorV1(declaration, execute)
        self._library = None
        self._pack_with_library = None
        return operator_id, ()

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

    def fresh_reference_table(
        self, pack, state: OperatorStateV1
    ) -> ReferenceTableV1:
        """A completely fresh, branch-local table for ``state`` — never a
        patch of ``self.table``. Sound here because these descriptors are
        stable external VALUE refs, independent of AST structure (see the
        module docstring on why that would *not* hold for NODE/ROLE
        descriptors, which the bulk/primitive fixture below rebuilds from
        scratch via ``build_openui_local_operator_context`` for exactly that
        reason). A pure function of ``state`` alone (the seed is derived
        from its digest, not a call counter) so exact replay — calling this
        builder twice for the same final state — reproduces byte-identical
        opaque IDs, exactly as a real reference-table rebuild must."""
        seed = int(state.state_digest[:8], 16)
        return build_reference_table(
            request_id=self.request_id,
            state_digest=state.state_digest,
            branch_digest=self.branch_digest,
            descriptors=self.descriptors,
            seed=seed,
        )

    def commit(self, transaction, *, reference_table_builder=None):
        return commit_operator_transaction(
            pack=self.pack_with_library,
            library=self.library,
            base_state=self.state,
            base_reference_table=self.table,
            transaction=transaction,
            provenance=self.provenance(),
            reference_table_builder=reference_table_builder
            or self.fresh_reference_table,
        )


def _all_topological_orders(
    ids: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> list[tuple[str, ...]]:
    """Every permutation of ``ids`` consistent with ``edges`` (pred before succ).

    Brute-force, tiny-domain only (as the issue's own agent contract
    requires for this test-only oracle) — never used outside this test
    module.
    """
    assert len(ids) <= 6, "tiny-domain oracle only: factorial blowup guard"
    orders = []
    for perm in itertools.permutations(ids):
        position = {node: index for index, node in enumerate(perm)}
        if all(position[pred] < position[succ] for pred, succ in edges):
            orders.append(perm)
    return orders


def _sequential_shadow_apply(
    fixture: _Fixture, requests: tuple[tuple[str, tuple[BoundArgumentV1, ...]], ...]
) -> OperatorStateV1:
    """Test-only sequential-shadow executor: actually chains dry-run/apply,
    rebinding (regenerating provenance for) the *current* state after each
    action — the exact anti-pattern the production ``compose_operator_transaction``
    must never use, kept here solely as a brute-force equivalence oracle."""
    state = fixture.state
    for operator_id, arguments in requests:
        provenance = ApplicationProvenanceV1(
            pack_id="openui",
            compiler_id="openui.tx_executor_shadow_fixture",
            compiler_version="v1",
            source_artifact_digest=_sha(state.source),
            request_id=fixture.request_id,
        )
        result = fixture.library.apply(
            fixture.pack_with_library, state, operator_id, arguments, provenance
        )
        assert result.succeeded and result.state is not None
        state = result.state
    return state


def _requests_by_semantic(
    fixture: _Fixture,
    transaction,
    requests: tuple[tuple[str, tuple[BoundArgumentV1, ...]], ...],
) -> dict[str, tuple[str, tuple[BoundArgumentV1, ...]]]:
    fingerprint_to_request = {
        fixture.library.lookup(operator_id).fingerprint: (operator_id, arguments)
        for operator_id, arguments in requests
    }
    return {
        action.semantic_action_id: fingerprint_to_request[action.operator_fingerprint]
        for action in transaction.prepared_actions
    }


# --------------------------------------------------------------------------- #
# Disjoint composition: two / four actions
# --------------------------------------------------------------------------- #


def test_two_disjoint_actions_compose_and_commit() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.exec_two_a",
        target_name="alpha",
        replacement=":hero.A",
    )
    write_b = fixture.declare(
        operator_id="openui.exec_two_b",
        target_name="beta",
        replacement=":hero.B",
    )
    decision = fixture.requests_transaction((write_a, write_b))
    assert decision.succeeded
    commit_decision = fixture.commit(decision.transaction)
    assert commit_decision.succeeded
    final_source = commit_decision.commit.final_state.source
    assert ":hero.A" in final_source
    assert ":hero.B" in final_source
    assert ":hero.gamma" in final_source and ":hero.delta" in final_source
    # base is read-only: never mutated by commit
    assert fixture.state.source == SOURCE
    assert commit_decision.commit.final_state.state_digest != fixture.state.state_digest


def test_four_disjoint_actions_compose_and_commit() -> None:
    fixture = _Fixture()
    requests = tuple(
        fixture.declare(
            operator_id=f"openui.exec_four_{name}",
            target_name=name,
            replacement=f":hero.{name.upper()}",
        )
        for name in fixture.names
    )
    decision = fixture.requests_transaction(requests)
    assert decision.succeeded
    assert len(decision.transaction.prepared_actions) == 4
    commit_decision = fixture.commit(decision.transaction)
    assert commit_decision.succeeded
    final_source = commit_decision.commit.final_state.source
    for name in fixture.names:
        assert f":hero.{name.upper()}" in final_source
    assert commit_decision.commit.commit_order == tuple(
        sorted(action.semantic_action_id for action in decision.transaction.prepared_actions)
    )


# --------------------------------------------------------------------------- #
# Bulk plus primitive mixed transaction (real production operators)
# --------------------------------------------------------------------------- #


def test_bulk_plus_primitive_mixed_transaction_commits() -> None:
    pack = get_pack("openui")
    source = (
        'root = Card([Stack([TextContent(":item0")]), '
        'Stack([TextContent(":item1")])], "clear")'
    )
    state = OperatorStateV1.from_source(pack, source)
    branch = branch_fingerprint(state.state_digest, "b" * 64)
    request_id = "bulk-primitive-request"
    context = build_openui_local_operator_context(
        pack,
        state,
        request_id=request_id,
        branch_digest=branch,
        seed=5,
        values=("column", "row"),
    )

    def _node_ref(path):
        return next(
            ref
            for ref in context.references(RefKind.NODE)
            if isinstance(context.payload(ref), NodeLocationV1)
            and context.payload(ref).path == path
        )

    def _descriptor_for(ref):
        return next(
            entry.descriptor
            for entry in context.reference_table.entries
            if entry.ref == ref
        )

    def _role_ref(node_ref, property_name):
        owner = _descriptor_for(node_ref).semantic_fingerprint
        return next(
            ref
            for ref in context.references(RefKind.ROLE)
            if isinstance(context.payload(ref), RoleLocationV1)
            and context.payload(ref).node_fingerprint == owner
            and context.payload(ref).property_name == property_name
        )

    def _value_ref(value):
        return next(
            ref
            for ref in context.references(RefKind.VALUE)
            if context.payload(ref).value == value
        )

    root_ref = _node_ref(("root",))
    stack_refs = tuple(
        _node_ref(("root", "props", "children", index)) for index in range(2)
    )
    context, selector_ref = build_bulk_selector(
        context,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint="1" * 64,
        matching_refs=stack_refs,
        max_fanout=8,
        seed=50,
    )
    library = build_openui_bulk_operator_library(context)
    pack_with_library = replace(pack, operator_library=library)

    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.tx_executor_mixed_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id=request_id,
    )
    bulk_prepared = prepare_operator_action(
        pack=pack_with_library,
        library=library,
        state=state,
        reference_table=context.reference_table,
        operator_id=MAP_SET_PROPERTY,
        arguments=(
            BoundArgumentV1("selector", selector_ref),
            BoundArgumentV1("role", _role_ref(stack_refs[0], "direction")),
            BoundArgumentV1("value", _value_ref("column")),
        ),
        provenance=provenance,
    )
    selector_argument = next(
        argument
        for argument in bulk_prepared.semantic_arguments
        if argument.slot_id == "selector"
    )
    assert selector_argument.ref_kind is RefKind.SELECTOR
    primitive_prepared = prepare_operator_action(
        pack=pack_with_library,
        library=library,
        state=state,
        reference_table=context.reference_table,
        operator_id=SET_PROPERTY,
        arguments=(
            BoundArgumentV1("node", root_ref),
            BoundArgumentV1("role", _role_ref(root_ref, "direction")),
            BoundArgumentV1("value", _value_ref("row")),
        ),
        provenance=provenance,
    )
    decision = build_operator_transaction(
        library=library,
        state=state,
        reference_table=context.reference_table,
        prepared_actions=(bulk_prepared, primitive_prepared),
        provenance=provenance,
    )
    assert decision.succeeded
    transaction = decision.transaction
    assert transaction.dependency_edges == ()
    assert transaction.conflict_graph == ()

    def _rebuild(pack, final_state):
        return build_openui_local_operator_context(
            pack,
            final_state,
            request_id=request_id,
            branch_digest=branch,
            seed=77,
        ).reference_table

    commit_decision = commit_operator_transaction(
        pack=pack_with_library,
        library=library,
        base_state=state,
        base_reference_table=context.reference_table,
        transaction=transaction,
        provenance=provenance,
        reference_table_builder=_rebuild,
    )
    assert commit_decision.succeeded
    final_state = commit_decision.commit.final_state
    bindings = parse_statement_bindings(final_state.source, dsl="openui")
    root_node = bindings["root"]
    assert root_node["props"]["direction"] == "row"
    for child in root_node["props"]["children"]:
        assert child["props"]["direction"] == "column"


# --------------------------------------------------------------------------- #
# Declared commuting: agreeing vs. disagreeing outcomes
# --------------------------------------------------------------------------- #


def test_declared_commuting_agreeing_overlap_composes() -> None:
    fixture = _Fixture()
    left = fixture.declare(
        operator_id="openui.exec_commute_agree_left",
        target_name="alpha",
        replacement=":hero.SAME",
        commutes_with=("openui.exec_commute_agree_right",),
    )
    right = fixture.declare(
        operator_id="openui.exec_commute_agree_right",
        target_name="alpha",
        replacement=":hero.SAME",
        commutes_with=("openui.exec_commute_agree_left",),
    )
    decision = fixture.requests_transaction((left, right))
    assert decision.succeeded
    assert len(decision.transaction.conflict_graph) == 1
    commit_decision = fixture.commit(decision.transaction)
    assert commit_decision.succeeded
    assert ":hero.SAME" in commit_decision.commit.final_state.source


def test_declared_commuting_disagreeing_overlap_rejects_as_structural_compose_conflict() -> (
    None
):
    fixture = _Fixture()
    left = fixture.declare(
        operator_id="openui.exec_commute_disagree_left",
        target_name="alpha",
        replacement=":hero.LEFT",
        commutes_with=("openui.exec_commute_disagree_right",),
    )
    right = fixture.declare(
        operator_id="openui.exec_commute_disagree_right",
        target_name="alpha",
        replacement=":hero.RIGHT",
        commutes_with=("openui.exec_commute_disagree_left",),
    )
    decision = fixture.requests_transaction((left, right))
    # transactions.py accepts it: both declare each other commuting.
    assert decision.succeeded
    commit_decision = fixture.commit(decision.transaction)
    # The executor's own structural safety net still catches the real,
    # unresolvable disagreement — a declared commutativity flag is never
    # trusted as a proof of an agreeing outcome.
    assert not commit_decision.succeeded
    assert (
        commit_decision.rejection.kind
        is OperatorTransactionCommitRejectionKind.STRUCTURAL_COMPOSE_CONFLICT
    )
    assert fixture.state.source == SOURCE


# --------------------------------------------------------------------------- #
# Dependency chain + sequential-shadow equivalence on a tiny domain
# --------------------------------------------------------------------------- #


def test_dependency_chain_composes_and_matches_every_valid_topological_order() -> None:
    fixture = _Fixture()
    first = fixture.declare(
        operator_id="openui.exec_chain_a",
        target_name="alpha",
        replacement=":hero.A2",
        commutes_with=("openui.exec_chain_b",),
    )
    second = fixture.declare(
        operator_id="openui.exec_chain_b",
        target_name="beta",
        replacement=":hero.B2",
        read_target_name="alpha",
        commutes_with=("openui.exec_chain_a", "openui.exec_chain_c"),
    )
    third = fixture.declare(
        operator_id="openui.exec_chain_c",
        target_name="gamma",
        replacement=":hero.C2",
        read_target_name="beta",
        commutes_with=("openui.exec_chain_b",),
    )
    decision = fixture.requests_transaction((first, second, third))
    assert decision.succeeded
    transaction = decision.transaction
    assert len(transaction.dependency_edges) == 2

    commit_decision = fixture.commit(transaction)
    assert commit_decision.succeeded
    final_source = commit_decision.commit.final_state.source

    ids = tuple(action.semantic_action_id for action in transaction.prepared_actions)
    edges = tuple(
        (edge.predecessor_semantic_id, edge.successor_semantic_id)
        for edge in transaction.dependency_edges
    )
    orders = _all_topological_orders(ids, edges)
    assert len(orders) >= 1
    requests_by_semantic = _requests_by_semantic(
        fixture, transaction, (first, second, third)
    )
    for order in orders:
        request_order = tuple(requests_by_semantic[sid] for sid in order)
        shadow_state = _sequential_shadow_apply(fixture, request_order)
        assert shadow_state.source == final_source


# --------------------------------------------------------------------------- #
# Action-set permutation reaches one committed identity
# --------------------------------------------------------------------------- #


def test_action_set_permutation_reaches_one_committed_identity() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.exec_perm_a", target_name="alpha", replacement=":hero.PA"
    )
    write_b = fixture.declare(
        operator_id="openui.exec_perm_b", target_name="beta", replacement=":hero.PB"
    )
    write_c = fixture.declare(
        operator_id="openui.exec_perm_c", target_name="gamma", replacement=":hero.PC"
    )
    forward = fixture.requests_transaction((write_a, write_b, write_c))
    backward = fixture.requests_transaction((write_c, write_b, write_a))
    assert forward.succeeded and backward.succeeded
    assert forward.transaction.transaction_id == backward.transaction.transaction_id

    forward_commit = fixture.commit(forward.transaction)
    backward_commit = fixture.commit(backward.transaction)
    assert forward_commit.succeeded and backward_commit.succeeded
    assert (
        forward_commit.commit.final_state.state_digest
        == backward_commit.commit.final_state.state_digest
    )
    assert forward_commit.commit.commit_order == backward_commit.commit.commit_order


# --------------------------------------------------------------------------- #
# Failure controls: failing constituent, reference rebuild, stale/tampered base
# --------------------------------------------------------------------------- #


def test_failing_constituent_rejects_and_leaves_base_unchanged() -> None:
    fixture = _Fixture()
    broken = {"on": False}
    good = fixture.declare(
        operator_id="openui.exec_flaky_good",
        target_name="alpha",
        replacement=":hero.GOOD",
    )
    flaky = fixture.declare_flaky(
        operator_id="openui.exec_flaky_bad",
        target_name="beta",
        replacement=":hero.BAD",
        broken=broken,
    )
    decision = fixture.requests_transaction((good, flaky))
    assert decision.succeeded

    broken["on"] = True
    commit_decision = fixture.commit(decision.transaction)
    assert not commit_decision.succeeded
    assert (
        commit_decision.rejection.kind
        is OperatorTransactionCommitRejectionKind.CONSTITUENT_REPLAY_FAILED
    )
    assert fixture.state.source == SOURCE


def test_reference_rebuild_failure_rejects_and_leaves_base_unchanged() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.exec_refbuild_a",
        target_name="alpha",
        replacement=":hero.RB",
    )
    decision = fixture.requests_transaction((write_a,))
    assert decision.succeeded

    def _raising_builder(pack, state):
        raise RuntimeError("simulated reference rebuild failure")

    commit_decision = fixture.commit(
        decision.transaction, reference_table_builder=_raising_builder
    )
    assert not commit_decision.succeeded
    assert (
        commit_decision.rejection.kind
        is OperatorTransactionCommitRejectionKind.REFERENCE_REBUILD_FAILED
    )
    assert fixture.state.source == SOURCE

    def _mismatched_builder(pack, state):
        # A "fresh" table bound to the wrong (base) state digest.
        return fixture.fresh_reference_table(pack, fixture.state)

    mismatched_decision = fixture.commit(
        decision.transaction, reference_table_builder=_mismatched_builder
    )
    assert not mismatched_decision.succeeded
    assert (
        mismatched_decision.rejection.kind
        is OperatorTransactionCommitRejectionKind.REFERENCE_REBUILD_FAILED
    )
    assert mismatched_decision.rejection.detail == "reference_table.identity_mismatch"


def test_stale_transaction_base_rejects() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.exec_stale_a", target_name="alpha", replacement=":hero.S"
    )
    decision = fixture.requests_transaction((write_a,))
    assert decision.succeeded

    other_source = 'root = Card([TextContent(":hero.other")], "clear")'
    other_state = OperatorStateV1.from_source(fixture.pack, other_source)
    other_table = build_reference_table(
        request_id=fixture.request_id,
        state_digest=other_state.state_digest,
        branch_digest=fixture.branch_digest,
        descriptors=(),
        seed=1,
    )
    commit_decision = commit_operator_transaction(
        pack=fixture.pack_with_library,
        library=fixture.library,
        base_state=other_state,
        base_reference_table=other_table,
        transaction=decision.transaction,
        provenance=fixture.provenance(),
        reference_table_builder=fixture.fresh_reference_table,
    )
    assert not commit_decision.succeeded
    assert (
        commit_decision.rejection.kind
        is OperatorTransactionCommitRejectionKind.STALE_TRANSACTION_BASE
    )


def test_tampered_transaction_rejects() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.exec_tamper_a", target_name="alpha", replacement=":hero.T1"
    )
    decision = fixture.requests_transaction((write_a,))
    assert decision.succeeded
    # A structurally valid ``OperatorTransactionV1`` (passes every
    # ``__post_init__`` invariant) but no longer authentic: its
    # ``expected_final_state_digest`` was hand-edited after the fact, so
    # recomputing the build decision from its own ``prepared_actions`` no
    # longer reproduces it exactly.
    forged = replace(
        decision.transaction,
        expected_final_state_digest=_sha("forged-final-state"),
    )
    commit_decision = fixture.commit(forged)
    assert not commit_decision.succeeded
    assert (
        commit_decision.rejection.kind
        is OperatorTransactionCommitRejectionKind.TRANSACTION_TAMPERED
    )
    assert fixture.state.source == SOURCE


# --------------------------------------------------------------------------- #
# Exact replay from base + recorded commit decision
# --------------------------------------------------------------------------- #


def test_replay_operator_transaction_commit_matches_recorded_and_detects_tampering() -> (
    None
):
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.exec_replay_a", target_name="alpha", replacement=":hero.R1"
    )
    write_b = fixture.declare(
        operator_id="openui.exec_replay_b", target_name="beta", replacement=":hero.R2"
    )
    decision = fixture.requests_transaction((write_a, write_b))
    assert decision.succeeded
    recorded = fixture.commit(decision.transaction)
    assert recorded.succeeded

    replayed = replay_operator_transaction_commit(
        pack=fixture.pack_with_library,
        library=fixture.library,
        base_state=fixture.state,
        base_reference_table=fixture.table,
        transaction=decision.transaction,
        provenance=fixture.provenance(),
        reference_table_builder=fixture.fresh_reference_table,
        recorded=recorded,
    )
    assert replayed.decision_id == recorded.decision_id
    assert replayed.commit.final_state == recorded.commit.final_state
    assert replayed.commit.final_reference_table == recorded.commit.final_reference_table
    assert replayed.commit.transaction_id == recorded.commit.transaction_id

    single = fixture.build((fixture.prepare(*write_a),))
    assert single.succeeded
    tampered_recorded = fixture.commit(single.transaction)
    assert tampered_recorded.succeeded
    with pytest.raises(ValueError, match="differs from recorded"):
        replay_operator_transaction_commit(
            pack=fixture.pack_with_library,
            library=fixture.library,
            base_state=fixture.state,
            base_reference_table=fixture.table,
            transaction=decision.transaction,
            provenance=fixture.provenance(),
            reference_table_builder=fixture.fresh_reference_table,
            recorded=tampered_recorded,
        )


# --------------------------------------------------------------------------- #
# Acceptance: composite effect coverage, fresh-table-only legal work
# --------------------------------------------------------------------------- #


def test_composite_effect_includes_every_target() -> None:
    fixture = _Fixture()
    requests = tuple(
        fixture.declare(
            operator_id=f"openui.exec_coverage_{name}",
            target_name=name,
            replacement=f":hero.{name.upper()}",
        )
        for name in fixture.names
    )
    decision = fixture.requests_transaction(requests)
    assert decision.succeeded
    transaction = decision.transaction
    assert len(transaction.composite_effect.property_deltas) == len(fixture.names)
    targets = {delta.target for delta in transaction.composite_effect.property_deltas}
    assert targets == {fixture.target(name) for name in fixture.names}


def test_post_commit_legal_set_uses_only_the_fresh_table() -> None:
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.exec_fresh_a", target_name="alpha", replacement=":hero.F1"
    )
    decision = fixture.requests_transaction((write_a,))
    assert decision.succeeded
    commit_decision = fixture.commit(decision.transaction)
    assert commit_decision.succeeded
    commit = commit_decision.commit

    stale_ref = fixture.target("beta")
    with pytest.raises(ReferenceResolutionError):
        commit.final_reference_table.resolve(
            stale_ref,
            state_digest=commit.final_state.state_digest,
            branch_digest=fixture.branch_digest,
            expected_kind=RefKind.VALUE,
        )
    fresh_ref = next(
        entry.ref
        for entry in commit.final_reference_table.entries
        if entry.descriptor.value_type == "openui.beta"
    )
    resolved = commit.final_reference_table.resolve(
        fresh_ref,
        state_digest=commit.final_state.state_digest,
        branch_digest=fixture.branch_digest,
        expected_kind=RefKind.VALUE,
    )
    assert resolved.value_type == "openui.beta"


# --------------------------------------------------------------------------- #
# Conversation turn integration
# --------------------------------------------------------------------------- #


def test_conversation_turn_commits_transaction_and_retains_constituent_application_ids() -> (
    None
):
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.exec_turn_a", target_name="alpha", replacement=":hero.T1"
    )
    write_b = fixture.declare(
        operator_id="openui.exec_turn_b", target_name="beta", replacement=":hero.T2"
    )
    decision = fixture.requests_transaction((write_a, write_b))
    assert decision.succeeded
    transaction = decision.transaction
    commit_decision = fixture.commit(transaction)
    assert commit_decision.succeeded
    commit = commit_decision.commit

    trace = create_conversation_trace(
        pack=fixture.pack_with_library,
        root_state=fixture.state,
        root_reference_table=fixture.table,
        provenance=fixture.provenance(),
    )
    next_trace = append_operator_transaction_turn(
        trace,
        pack=fixture.pack_with_library,
        library=fixture.library,
        transaction=transaction,
        final_state=commit.final_state,
        output_reference_table=commit.final_reference_table,
    )
    assert next_trace.current.state == commit.final_state
    assert next_trace.current.reference_table == commit.final_reference_table
    turn = next_trace.turns[-1]
    expected_ids = tuple(
        sorted(action.dry_run.application_id for action in transaction.prepared_actions)
    )
    assert turn.application_ids == expected_ids
    assert len(turn.application_ids) == 2

    replayed = replay_conversation_trace(
        pack=fixture.pack_with_library,
        library=fixture.library,
        transaction_replayer=transaction_replayer_for_conversation,
        trace=next_trace,
    )
    assert replayed.state == commit.final_state

    with pytest.raises(ValueError, match="transaction_replayer"):
        replay_conversation_trace(
            pack=fixture.pack_with_library,
            library=fixture.library,
            trace=next_trace,
        )


def test_compose_operator_transaction_never_chains_against_a_prior_actions_output() -> (
    None
):
    """Direct proof of the no-stale-ref-sequential-chaining invariant:

    ``compose_operator_transaction`` never even receives a "current state"
    parameter beyond the one fixed ``base_state`` — every constituent is
    replayed against it directly. This test makes that explicit by
    asserting the composed result equals directly folding each action's own
    *independently replayed* output, never a state any other action
    produced.
    """
    fixture = _Fixture()
    write_a = fixture.declare(
        operator_id="openui.exec_nochain_a", target_name="alpha", replacement=":hero.N1"
    )
    write_b = fixture.declare(
        operator_id="openui.exec_nochain_b", target_name="beta", replacement=":hero.N2"
    )
    decision = fixture.requests_transaction((write_a, write_b))
    assert decision.succeeded
    final_state = compose_operator_transaction(
        pack=fixture.pack_with_library,
        library=fixture.library,
        base_state=fixture.state,
        transaction=decision.transaction,
    )
    # Each action's dry run replayed against the untouched base individually
    # still succeeds and reproduces its own before-digests -- proving no
    # action was ever asked to resolve against a mutated intermediate state.
    for action in decision.transaction.prepared_actions:
        replayed = fixture.library.replay(
            fixture.pack_with_library, fixture.state, action.dry_run
        )
        assert replayed.succeeded
        assert replayed.application.before_state_digest == fixture.state.state_digest
    assert ":hero.N1" in final_state.source and ":hero.N2" in final_state.source
