"""Atomic multi-action operator transaction executor (DSH5-05).

SLM-412 (DSH5-04, ``transactions.py``) proved a schema/safety layer only: it
proves a set of independently-``dry_run`` actions against one fixed base is
either disjoint or exactly commuting, but it never executes anything — no
function there applies a mutation, and its own module docstring says so
explicitly. This module is the execution issue that schema layer was built
for: it composes an already-built, already-proven :class:`OperatorTransactionV1`
into one real, pack-authorized final state, atomically, without ever
re-dry-running one action against another action's result.

Composition algorithm (the actual "atomic AST composition" this issue is
about)
--------------------------------------------------------------------------
1. **Authenticate the transaction.** Reuse SLM-412's own
   :func:`~slm_training.dsl.operators.transactions.replay_operator_transaction`
   to recompute the complete build decision from ``transaction``'s own
   ``prepared_actions`` and require it to reproduce ``transaction`` exactly —
   before doing anything real, prove the record was not tampered with.
2. **Independently replay every constituent action** against the *same
   fixed* base state via ``OperatorLibraryV1.replay`` (never against another
   action's output — the exact anti-pattern this issue exists to avoid).
   Each replay reproduces that one action's own already-proven
   ``dry_run`` evidence, deterministically, from the base alone.
3. **Compose base-relative AST deltas, never re-execute sequentially.**
   Parse the base source once (``pack.backend.parse``), then fold every
   constituent's independently-replayed output through
   :func:`slm_training.dsl.operators.ast_merge.merge_ast_value` — the exact
   conservative base-relative 3-way merge DSH5-04's own predecessor,
   ``merge.py``, already uses for two-way branch merges. Folding it
   pairwise (``result = base; for output in outputs: result =
   merge_ast_value(base, result, output)``) generalizes that primitive to
   N-way composition: every fold step still diffs against the one fixed
   ``base``, never against a prior fold step's *content* as if it were a
   second base. The fold visits actions in one canonical topological order
   consistent with ``transaction.dependency_edges`` (ties broken
   lexicographically by ``semantic_action_id``) so composition genuinely
   "applies dependency order only when composition semantics require it" —
   for the disjoint/exactly-commuting subset this module accepts (already
   proven at the transactions.py layer), the merge is provably
   order-invariant, and ``test_operator_transaction_executor.py``'s
   sequential-shadow oracle checks exactly that on tiny domains.
4. **Real, unresolvable overlaps still reject — never approximate.** Even
   when two actions were declared mutually commuting (a *declarative* flag
   at the transactions.py layer, not an empirically-verified one), the
   structural merge only succeeds where the two independently-computed
   outputs actually agree wherever they overlap the base; a genuine
   disagreement raises ``ast_merge.StructuralMergeConflict`` here, which
   this module turns into a typed, fail-closed
   ``STRUCTURAL_COMPOSE_CONFLICT`` rejection — this is the safety net behind
   ``commutes_with`` being a declaration, not a proof.
5. **Validate the composed source with full pack authority** (parse /
   serialize / canonicalize / oracle / scope / property-order / round-trip —
   ``OperatorStateV1.from_source``'s existing pipeline, never a parallel
   check) and **rebuild a completely fresh, branch-local reference table**
   for the resulting state (never patch/reuse the base's descriptors: DSH3-04
   node identity is content-addressed through each node's own canonical
   structure, so *any* edit anywhere in an ancestor chain changes that
   ancestor's own persistent fingerprint — a fresh table is the only sound
   continuation, exactly the issue's "fresh branch-local reference table").

No commit ever applies a mutation to pack-authorized source directly:
every real change flows through the same ``OperatorStateV1.from_source``
authority boundary every other operator-execution path in this package
uses. Any failure at any step returns a typed
:class:`OperatorTransactionCommitRejectionV1` and leaves the caller's own
``base_state``/``base_reference_table`` completely untouched (they are never
mutated — only read).

Layering: why a shared ``ast_merge`` module
--------------------------------------------
``merge.py`` imports ``conversation.py`` (for ``ConversationStateNodeV1``);
``transactions.py`` imports ``merge.py`` (for ``_effect_targets``, reused
unmodified). If this module reused ``merge.py``'s merge primitive by
importing ``merge.py`` directly, and ``conversation.py`` in turn needed to
call into this module to replay a transaction-commit turn, the import graph
would close a cycle: ``conversation -> transaction_executor -> merge ->
conversation``. The fix is ``ast_merge.py``: the merge primitive extracted
with zero dependency on ``conversation.py``, imported by both ``merge.py``
(unchanged public behavior) and this module. ``conversation.py`` itself
never imports this module at all — see ``TransactionReplayer`` in
``conversation.py``, a caller-supplied callback (the same indirection
pattern as its existing ``OperatorAuthorityResolver``) that this module
fulfills via :func:`transaction_replayer_for_conversation` without
``conversation.py`` ever importing this one back.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from slm_training.dsl.operators.ast_merge import (
    StructuralMergeConflict,
    merge_ast_value,
)
from slm_training.dsl.operators.contracts import (
    ApplicationProvenanceV1,
    _fingerprint,
    _require_digest,
    _require_identifier,
)
from slm_training.dsl.operators.conversation import ConversationStateNodeV1
from slm_training.dsl.operators.references import ReferenceTableV1
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorReplayError,
    OperatorStateV1,
)
from slm_training.dsl.operators.transactions import (
    OPERATOR_TRANSACTION_MAX_ACTIONS,
    OperatorTransactionDecisionV1,
    OperatorTransactionV1,
    replay_operator_transaction,
)
from slm_training.dsl.pack import DslPack

#: A caller-supplied, DSL-specific rebuild of a *completely fresh*,
#: branch-local reference table for a brand-new state — never a patch of the
#: base table. Mirrors how ``build_openui_local_operator_context`` derives
#: descriptors from scratch for one state; this module stays DSL-agnostic by
#: never hardcoding that derivation itself.
ReferenceTableBuilder = Callable[[DslPack, OperatorStateV1], ReferenceTableV1]


class OperatorTransactionCommitRejectionKind(str, Enum):
    """Typed, deterministic reasons a transaction refused to commit."""

    #: The given base state/reference table/provenance no longer match what
    #: ``transaction`` was built against.
    STALE_TRANSACTION_BASE = "stale_transaction_base"
    #: Recomputing the transaction from its own prepared actions/base did not
    #: reproduce ``transaction`` exactly (SLM-412's own replay check).
    TRANSACTION_TAMPERED = "transaction_tampered"
    #: A constituent prepared action's own dry run did not replay against
    #: the base (a failing constituent).
    CONSTITUENT_REPLAY_FAILED = "constituent_replay_failed"
    #: The base-relative structural AST fold found a real, unresolvable
    #: overlap between two independently-prepared outputs.
    STRUCTURAL_COMPOSE_CONFLICT = "structural_compose_conflict"
    #: The composed source failed full pack authority (parse / canonicalize /
    #: oracle / scope / property-order / round-trip).
    FINAL_STATE_AUTHORITY_REJECTED = "final_state_authority_rejected"
    #: The fresh branch-local reference table could not be rebuilt, or the
    #: rebuilt table disagrees with the final state/request identity.
    REFERENCE_REBUILD_FAILED = "reference_rebuild_failed"


class _ConstituentReplayFailed(Exception):
    def __init__(self, semantic_action_id: str) -> None:
        self.semantic_action_id = semantic_action_id
        super().__init__(semantic_action_id)


class _ComposeConflict(Exception):
    def __init__(self, semantic_action_id: str, kind: Any) -> None:
        self.semantic_action_id = semantic_action_id
        self.kind = kind
        super().__init__(semantic_action_id)


@dataclass(frozen=True)
class OperatorTransactionCommitRejectionV1:
    """One typed, deterministic reason a transaction refused to commit."""

    kind: OperatorTransactionCommitRejectionKind
    transaction_id: str
    action_semantic_ids: tuple[str, ...] = ()
    detail: str = ""
    schema: str = "operator_transaction_commit_rejection/v1"

    def __post_init__(self) -> None:
        _require_digest(self.transaction_id, "transaction_id")
        for value in self.action_semantic_ids:
            _require_digest(value, "action_semantic_id")
        if len(set(self.action_semantic_ids)) != len(self.action_semantic_ids):
            raise ValueError("action semantic ids must be unique")
        if tuple(sorted(self.action_semantic_ids)) != self.action_semantic_ids:
            raise ValueError("action semantic ids must be canonical")

    @property
    def rejection_id(self) -> str:
        return _fingerprint(self.to_dict(include_rejection_id=False))

    def to_dict(self, *, include_rejection_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "kind": self.kind.value,
            "transaction_id": self.transaction_id,
            "action_semantic_ids": list(self.action_semantic_ids),
            "detail": self.detail,
        }
        if include_rejection_id:
            value["rejection_id"] = self.rejection_id
        return value


@dataclass(frozen=True)
class OperatorTransactionCommitProofV1:
    """One composite proof binding a committed transaction to its evidence."""

    proof_kind: str
    checks: tuple[str, ...]
    composite_effect_fingerprint: str
    committed_state_digest: str
    committed_ast_digest: str
    reference_table_fingerprint: str
    schema: str = "operator_transaction_commit_proof/v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(sorted(set(self.checks))))
        _require_identifier(self.proof_kind, "proof_kind")
        if not self.checks:
            raise ValueError("a commit proof requires at least one check")
        for check in self.checks:
            _require_identifier(check, "proof check")
        _require_digest(
            self.composite_effect_fingerprint, "composite_effect_fingerprint"
        )
        _require_digest(self.committed_state_digest, "committed_state_digest")
        _require_digest(self.committed_ast_digest, "committed_ast_digest")
        _require_digest(
            self.reference_table_fingerprint, "reference_table_fingerprint"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "proof_kind": self.proof_kind,
            "checks": sorted(set(self.checks)),
            "composite_effect_fingerprint": self.composite_effect_fingerprint,
            "committed_state_digest": self.committed_state_digest,
            "committed_ast_digest": self.committed_ast_digest,
            "reference_table_fingerprint": self.reference_table_fingerprint,
        }


def _state_dict(state: OperatorStateV1) -> dict[str, Any]:
    return {
        "schema": state.schema,
        "pack_id": state.pack_id,
        "source": state.source,
        "state_digest": state.state_digest,
        "ast_digest": state.ast_digest,
    }


@dataclass(frozen=True)
class OperatorTransactionCommitV1:
    """One real final state, fresh reference table, and proof for a transaction.

    ``commit_order`` is the exact canonical topological order (semantic
    action IDs) the structural fold used — recorded as evidence, never as
    authority a replay must reproduce independently (replay recomputes the
    *same* deterministic order from ``transaction`` itself).
    """

    transaction_id: str
    base_state_digest: str
    final_state: OperatorStateV1
    final_reference_table: ReferenceTableV1
    proof: OperatorTransactionCommitProofV1
    provenance: ApplicationProvenanceV1
    commit_order: tuple[str, ...]
    schema: str = "operator_transaction_commit/v1"

    def __post_init__(self) -> None:
        _require_digest(self.transaction_id, "transaction_id")
        _require_digest(self.base_state_digest, "base_state_digest")
        for value in self.commit_order:
            _require_digest(value, "commit_order semantic id")
        if len(set(self.commit_order)) != len(self.commit_order):
            raise ValueError("commit order must name each action exactly once")
        if self.final_reference_table.state_digest != self.final_state.state_digest:
            raise ValueError("final reference table is stale for the final state")
        if self.proof.committed_state_digest != self.final_state.state_digest:
            raise ValueError("commit proof does not bind the final state digest")
        if self.proof.committed_ast_digest != self.final_state.ast_digest:
            raise ValueError("commit proof does not bind the final ast digest")
        if (
            self.proof.reference_table_fingerprint
            != self.final_reference_table.fingerprint
        ):
            raise ValueError("commit proof does not bind the final reference table")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "transaction_id": self.transaction_id,
            "base_state_digest": self.base_state_digest,
            "final_state": _state_dict(self.final_state),
            "final_reference_table": self.final_reference_table.to_dict(),
            "proof": self.proof.to_dict(),
            "provenance": self.provenance.to_dict(),
            "commit_order": list(self.commit_order),
        }

    @property
    def commit_id(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class OperatorTransactionCommitDecisionV1:
    """Exactly one of a canonical commit or a typed rejection."""

    commit: OperatorTransactionCommitV1 | None = None
    rejection: OperatorTransactionCommitRejectionV1 | None = None
    schema: str = "operator_transaction_commit_decision/v1"

    def __post_init__(self) -> None:
        if (self.commit is None) == (self.rejection is None):
            raise ValueError("exactly one commit or rejection is required")

    @property
    def succeeded(self) -> bool:
        return self.commit is not None

    @property
    def decision_id(self) -> str:
        return _fingerprint(self.to_dict(include_decision_id=False))

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "commit": self.commit.to_dict() if self.commit is not None else None,
            "rejection": (
                self.rejection.to_dict() if self.rejection is not None else None
            ),
        }
        if include_decision_id:
            value["decision_id"] = self.decision_id
        return value


def _topological_commit_order(transaction: OperatorTransactionV1) -> tuple[str, ...]:
    """One deterministic linear extension of ``transaction.dependency_edges``.

    Ties are broken lexicographically by ``semantic_action_id`` so two
    transactions with the same action set and edges always fold in the same
    order — a second, independent determinism guarantee on top of
    ``OperatorTransactionV1``'s own canonical action ordering. The
    transaction is already proven acyclic at build time
    (``OperatorTransactionV1.__post_init__``/``build_operator_transaction``),
    so this never actually raises in practice; the check stays as defense in
    depth against a directly-constructed, non-canonical caller value.
    """
    ids = tuple(action.semantic_action_id for action in transaction.prepared_actions)
    predecessors: dict[str, set[str]] = {node: set() for node in ids}
    for edge in transaction.dependency_edges:
        predecessors[edge.successor_semantic_id].add(edge.predecessor_semantic_id)
    remaining = set(ids)
    order: list[str] = []
    while remaining:
        ready = sorted(
            node for node in remaining if not (predecessors[node] & remaining)
        )
        if not ready:
            raise ValueError("dependency edges contain a cycle")
        node = ready[0]
        order.append(node)
        remaining.discard(node)
    return tuple(order)


def _replay_constituents(
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    base_state: OperatorStateV1,
    transaction: OperatorTransactionV1,
) -> dict[str, OperatorStateV1]:
    """Independently replay every prepared action's own dry run against ``base_state``.

    Never chains: every action replays against the fixed ``base_state``
    directly, never against another action's output. A replay mismatch
    (a failing constituent) raises :class:`_ConstituentReplayFailed` naming
    the exact action, rather than silently skipping it.
    """
    outputs: dict[str, OperatorStateV1] = {}
    for action in transaction.prepared_actions:
        try:
            replayed = library.replay(pack, base_state, action.dry_run)
        except OperatorReplayError as exc:
            raise _ConstituentReplayFailed(action.semantic_action_id) from exc
        if not replayed.succeeded or replayed.state is None:
            raise _ConstituentReplayFailed(action.semantic_action_id)
        outputs[action.semantic_action_id] = replayed.state
    return outputs


def compose_operator_transaction(
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    base_state: OperatorStateV1,
    transaction: OperatorTransactionV1,
) -> OperatorStateV1:
    """Compose ``transaction``'s prepared actions into one final, authorized state.

    The shared composition core both :func:`commit_operator_transaction` and
    :func:`transaction_replayer_for_conversation` use. Every action is
    independently replayed against the *same fixed* ``base_state`` (never
    against another action's output), then folded pairwise through
    :func:`~slm_training.dsl.operators.ast_merge.merge_ast_value` in one
    canonical topological order. Raises on any failure (a failing
    constituent, a real structural conflict, or a pack-authority rejection
    of the composed source) — never returns a partial or approximate state.
    """
    if (
        base_state.state_digest != transaction.base_state_digest
        or base_state.ast_digest != transaction.base_ast_digest
    ):
        raise ValueError("transaction base differs from the given base state")
    outputs = _replay_constituents(
        pack=pack, library=library, base_state=base_state, transaction=transaction
    )
    order = _topological_commit_order(transaction)
    base_ast = pack.backend.parse(base_state.source)
    accumulator = base_ast
    for semantic_id in order:
        output_ast = pack.backend.parse(outputs[semantic_id].source)
        try:
            accumulator = merge_ast_value(base_ast, accumulator, output_ast)
        except StructuralMergeConflict as exc:
            raise _ComposeConflict(semantic_id, exc.kind) from exc
    final_source = pack.backend.serialize(accumulator)
    return OperatorStateV1.from_source(pack, final_source)


def transaction_replayer_for_conversation(
    pack: DslPack,
    library: OperatorLibraryV1,
    cursor: ConversationStateNodeV1,
    transaction: OperatorTransactionV1,
) -> OperatorStateV1:
    """``conversation.TransactionReplayer`` adapter: compose against ``cursor.state``.

    Passed as ``transaction_replayer=`` to
    ``conversation.replay_conversation_trace`` by callers that need to
    replay a trace containing ``TRANSACTION_COMMIT`` turns — ``conversation.py``
    itself never imports this module (see this module's docstring, "Layering").
    """
    return compose_operator_transaction(
        pack=pack, library=library, base_state=cursor.state, transaction=transaction
    )


def _reject(
    kind: OperatorTransactionCommitRejectionKind,
    transaction_id: str,
    action_semantic_ids: tuple[str, ...] = (),
    detail: str = "",
) -> OperatorTransactionCommitDecisionV1:
    return OperatorTransactionCommitDecisionV1(
        rejection=OperatorTransactionCommitRejectionV1(
            kind=kind,
            transaction_id=transaction_id,
            action_semantic_ids=tuple(sorted(set(action_semantic_ids))),
            detail=detail,
        )
    )


def commit_operator_transaction(
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    base_state: OperatorStateV1,
    base_reference_table: ReferenceTableV1,
    transaction: OperatorTransactionV1,
    provenance: ApplicationProvenanceV1,
    reference_table_builder: ReferenceTableBuilder,
    max_actions: int = OPERATOR_TRANSACTION_MAX_ACTIONS,
) -> OperatorTransactionCommitDecisionV1:
    """Atomically commit ``transaction``: one final state, one fresh table.

    Every step is data-dependent and fails closed to a typed rejection,
    leaving ``base_state``/``base_reference_table`` completely unread-only
    (never mutated) on any failure:

    1. ``STALE_TRANSACTION_BASE`` — the given base/provenance no longer
       match what ``transaction`` was built against.
    2. ``TRANSACTION_TAMPERED`` — replaying ``transaction`` through SLM-412's
       own ``replay_operator_transaction`` did not reproduce it exactly.
    3. ``CONSTITUENT_REPLAY_FAILED`` — a failing constituent action.
    4. ``STRUCTURAL_COMPOSE_CONFLICT`` — a real, unresolvable AST overlap the
       transactions.py read/write footprint analysis did not (or, for a
       declared-commuting pair, could not) catch in advance.
    5. ``FINAL_STATE_AUTHORITY_REJECTED`` — the composed source failed full
       pack authority.
    6. ``REFERENCE_REBUILD_FAILED`` — the fresh reference table could not be
       built, or disagrees with the final state/request identity.
    """
    transaction_id = transaction.transaction_id
    all_ids = tuple(
        sorted(action.semantic_action_id for action in transaction.prepared_actions)
    )
    if (
        base_state.state_digest != transaction.base_state_digest
        or base_state.ast_digest != transaction.base_ast_digest
        or base_reference_table.state_digest != base_state.state_digest
        or base_reference_table.fingerprint
        != transaction.base_reference_table_fingerprint
        or provenance.request_id != transaction.provenance.request_id
        or provenance.pack_id != transaction.provenance.pack_id
    ):
        return _reject(
            OperatorTransactionCommitRejectionKind.STALE_TRANSACTION_BASE,
            transaction_id,
            all_ids,
        )

    try:
        replay_operator_transaction(
            library=library,
            state=base_state,
            reference_table=base_reference_table,
            prepared_actions=transaction.prepared_actions,
            provenance=transaction.provenance,
            recorded=OperatorTransactionDecisionV1(transaction=transaction),
            max_actions=max_actions,
        )
    except ValueError:
        return _reject(
            OperatorTransactionCommitRejectionKind.TRANSACTION_TAMPERED,
            transaction_id,
            all_ids,
        )

    try:
        final_state = compose_operator_transaction(
            pack=pack,
            library=library,
            base_state=base_state,
            transaction=transaction,
        )
    except _ConstituentReplayFailed as exc:
        return _reject(
            OperatorTransactionCommitRejectionKind.CONSTITUENT_REPLAY_FAILED,
            transaction_id,
            (exc.semantic_action_id,),
        )
    except _ComposeConflict as exc:
        return _reject(
            OperatorTransactionCommitRejectionKind.STRUCTURAL_COMPOSE_CONFLICT,
            transaction_id,
            (exc.semantic_action_id,),
            detail=exc.kind.value,
        )
    except Exception as exc:  # noqa: BLE001 - any composed-source failure
        # rejects closed: OperatorAuthorityError from ``OperatorStateV1.from_source``
        # is the expected case, but a malformed merged AST can also fail
        # ``pack.backend.serialize`` itself before authority ever runs.
        return _reject(
            OperatorTransactionCommitRejectionKind.FINAL_STATE_AUTHORITY_REJECTED,
            transaction_id,
            all_ids,
            detail=str(exc),
        )

    try:
        final_table = reference_table_builder(pack, final_state)
    except Exception:  # noqa: BLE001 - any rebuild failure is a typed rejection
        return _reject(
            OperatorTransactionCommitRejectionKind.REFERENCE_REBUILD_FAILED,
            transaction_id,
            all_ids,
        )
    if (
        final_table.state_digest != final_state.state_digest
        or final_table.request_id != provenance.request_id
    ):
        return _reject(
            OperatorTransactionCommitRejectionKind.REFERENCE_REBUILD_FAILED,
            transaction_id,
            all_ids,
            detail="reference_table.identity_mismatch",
        )

    commit_order = _topological_commit_order(transaction)
    proof = OperatorTransactionCommitProofV1(
        proof_kind="transaction.committed",
        checks=(
            "transaction.authentic",
            "constituents.replayed",
            "composition.structural_merge",
            "final_state.pack_authority",
            "reference_table.fresh",
        ),
        composite_effect_fingerprint=transaction.composite_effect.fingerprint,
        committed_state_digest=final_state.state_digest,
        committed_ast_digest=final_state.ast_digest,
        reference_table_fingerprint=final_table.fingerprint,
    )
    commit = OperatorTransactionCommitV1(
        transaction_id=transaction_id,
        base_state_digest=base_state.state_digest,
        final_state=final_state,
        final_reference_table=final_table,
        proof=proof,
        provenance=provenance,
        commit_order=commit_order,
    )
    return OperatorTransactionCommitDecisionV1(commit=commit)


def replay_operator_transaction_commit(
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    base_state: OperatorStateV1,
    base_reference_table: ReferenceTableV1,
    transaction: OperatorTransactionV1,
    provenance: ApplicationProvenanceV1,
    reference_table_builder: ReferenceTableBuilder,
    recorded: OperatorTransactionCommitDecisionV1,
    max_actions: int = OPERATOR_TRANSACTION_MAX_ACTIONS,
) -> OperatorTransactionCommitDecisionV1:
    """Recompute the complete commit decision and require exact identity with ``recorded``.

    Exact replay from base + recorded transaction: this recomputes every
    step of :func:`commit_operator_transaction` from scratch (constituent
    replay, structural composition, pack authority, reference rebuild) and
    requires the recomputed decision's ``decision_id`` — which transitively
    binds the final state and fresh reference table identities — to match
    ``recorded`` exactly.
    """
    replayed = commit_operator_transaction(
        pack=pack,
        library=library,
        base_state=base_state,
        base_reference_table=base_reference_table,
        transaction=transaction,
        provenance=provenance,
        reference_table_builder=reference_table_builder,
        max_actions=max_actions,
    )
    if replayed.decision_id != recorded.decision_id:
        raise ValueError(
            "operator transaction commit replay differs from recorded decision"
        )
    return replayed


__all__ = [
    "OperatorTransactionCommitDecisionV1",
    "OperatorTransactionCommitProofV1",
    "OperatorTransactionCommitRejectionKind",
    "OperatorTransactionCommitRejectionV1",
    "OperatorTransactionCommitV1",
    "ReferenceTableBuilder",
    "commit_operator_transaction",
    "compose_operator_transaction",
    "replay_operator_transaction_commit",
    "transaction_replayer_for_conversation",
]
