# DSH5-05 atomic transaction execution: composition, commit, and replay

SLM-413 (DSH5-05) is milestone M2's second decision for "DSH5 — Bulk
Operators, Transactions & Control Plane": SLM-412 (DSH5-04,
`transactions.py`) proved that a set of independently-`dry_run` actions
against one fixed base is either disjoint or exactly commuting, but shipped
**no execution path** — its own module docstring says so explicitly. This
issue is that execution path: `src/slm_training/dsl/operators/transaction_executor.py`
composes an already-built, already-proven `OperatorTransactionV1` into one
real, pack-authorized final state, atomically, without ever re-dry-running
one action against another action's result.

## Decision and hypothesis

Can exact prepared branch effects be composed into one verifier-authorized
state without sequential stale-reference application? The hypothesis: for
disjoint or exactly commuting prepared actions, composing base-relative AST
deltas atomically reproduces every allowed sequential topological order,
validates once, emits one composite proof, and produces a fresh
continuation reference table.

**This hypothesis holds for the subset SLM-412 already proves preparable**
(disjoint footprints, or footprints resolved by mutual `commutes_with`) —
see "Coverage and what's excluded" below for the one honest caveat, and
"Stop-rule disposition" for why it does not trigger the issue's stop rule.

## Composition algorithm

`compose_operator_transaction` (the shared core both `commit_operator_transaction`
and the conversation `TransactionReplayer` adapter use):

1. **Independently replay every constituent action** against the *same
   fixed* base state via `OperatorLibraryV1.replay` — never against another
   action's output. Each replay reproduces that one action's own
   already-proven `dry_run` evidence, deterministically, from the base
   alone. A replay mismatch (a failing constituent) raises immediately,
   naming the exact action.
2. **Parse the base source once** (`pack.backend.parse`), then **fold**
   every constituent's independently-replayed output through
   `ast_merge.merge_ast_value` — the *exact* conservative base-relative
   3-way merge `merge.py` (DSH3-19/DSH3-20) already uses for two-way
   conversation-branch merges. Folding it pairwise —
   `result = base; for output in outputs: result = merge_ast_value(base, result, output)`
   — generalizes that primitive to N-way composition: every fold step still
   diffs against the one fixed `base`, never against a prior fold step's
   *content* as if it were a second base. This is directly analogous to a
   diff3 fold and is provably order-invariant for genuinely disjoint edits
   (proved directly by the sequential-shadow oracle described below).
3. The fold visits actions in **one canonical topological order** consistent
   with `transaction.dependency_edges` (ties broken lexicographically by
   `semantic_action_id`) — "apply dependency order only when composition
   semantics require it; independent actions remain order-invariant," per
   the issue's own wording. For every operator declared in this repository
   today, a "read" footprint is a *declared precondition argument-slot*
   annotation (validity bookkeeping), never a value an operator's mutation
   logic actually consumes from another action's fresh output — so, for
   this concrete operator set, the fold is provably order-invariant
   regardless of dependency edges too. The executor still computes and
   honors the topological order (defense in depth, and because a future
   operator *could* declare a real value-consuming read), but the
   dependency-chain test proves the *stronger, more useful* claim: the
   composed result equals what **every** legal sequential topological order
   would reach, not just the one order chosen.
4. **Real, unresolvable overlaps still reject — never approximate.** Even
   when two actions were declared mutually commuting (a *declarative* flag
   at the transactions.py layer, not an empirically-verified one), the
   structural merge only succeeds where the two independently-computed
   outputs actually agree wherever they overlap the base; a genuine
   disagreement raises `ast_merge.StructuralMergeConflict`, turned into a
   typed, fail-closed `STRUCTURAL_COMPOSE_CONFLICT` rejection. This is the
   safety net behind `commutes_with` being a declaration, not a proof —
   proved directly by `test_declared_commuting_disagreeing_overlap_rejects_as_structural_compose_conflict`.
5. **Validate the composed source with full pack authority** — parse /
   serialize / canonicalize / oracle / scope / property-order / round-trip,
   `OperatorStateV1.from_source`'s existing pipeline, never a parallel
   check — and **rebuild a completely fresh, branch-local reference table**
   for the resulting state via a caller-supplied `ReferenceTableBuilder`
   (never patch/reuse the base's descriptors: DSH3-04 node identity is
   content-addressed through each node's own canonical structure, so *any*
   edit anywhere in an ancestor chain changes that ancestor's own
   persistent fingerprint — a fresh table is the only sound continuation).

Any failure at any step returns a typed `OperatorTransactionCommitRejectionV1`
and leaves the caller's `base_state`/`base_reference_table` completely
unread-only (never mutated — the executor only ever reads them; the real
new state is a fresh `OperatorStateV1` value, never a mutation in place).

### Layering: why a shared `ast_merge` module

`merge.py` imports `conversation.py` (for `ConversationStateNodeV1`);
`transactions.py` imports `merge.py` (for `_effect_targets`, reused
unmodified, per DSH5-04). Reusing `merge.py`'s merge primitive by importing
`merge.py` directly from the new executor module, combined with
`conversation.py` needing to call into that executor to replay a
transaction-commit turn, would close an import cycle:
`conversation -> transaction_executor -> merge -> conversation`.

The fix is `src/slm_training/dsl/operators/ast_merge.py`: the merge
primitive (`MergeConflictKind`, `StructuralMergeConflict`, `merge_ast_value`
— formerly `merge.py`'s private `_merge_value`/`_StructuralConflict`)
extracted with **zero** dependency on `conversation.py`. `merge.py` now
imports from `ast_merge.py` instead of defining these locally — its own
public behavior and every existing `test_operator_merge.py` test are
unchanged (this is a pure extraction, proved by the full existing suite
still passing byte-for-byte). `transaction_executor.py` imports
`ast_merge.py` directly.

`conversation.py` itself **never imports `transaction_executor.py`** at
all. Its `TurnArtifactV1.transaction` field and the
`replay_conversation_trace(transaction_replayer=...)` parameter both use
`OperatorTransactionV1` only as a `TYPE_CHECKING`-guarded forward reference
(`from __future__ import annotations` already makes every annotation lazy)
and treat the value as duck-typed at runtime — the same indirection pattern
`OperatorAuthorityResolver` already established for pack/library
resolution. `transaction_executor.py` provides the concrete fulfillment,
`transaction_replayer_for_conversation`, without `conversation.py` ever
importing it back. This is the one structural change this issue makes to
files outside its own new module, and it is behavior-preserving for every
existing caller of `merge.py`/`conversation.py`.

## Coverage and what's excluded

**Covered, proven by the test suite:**

* Two/four disjoint local actions (pure property edits on distinct value
  targets).
* A bulk-plus-primitive mixed transaction using the *real* production
  operators (`openui.map_set_property` selector-bulk write plus
  `openui.set_property` primitive write on a disjoint node) — this
  surfaced and fixed a real pre-existing gap: `transactions.py`'s
  `_target_lineage` never included `ReferenceTableV1.selectors`, so no
  selector-argument operator (i.e. no bulk operator) could be *prepared*
  into a transaction at all before this issue. The fix is purely additive
  (adds selector entries to the lineage map; every existing non-selector
  lineage entry is unchanged) — see "Selector lineage fix" below.
* A declared-commuting overlap with genuinely agreeing outcomes composes.
* A declared-commuting overlap with genuinely disagreeing outcomes is
  caught by the structural safety net and rejected
  (`STRUCTURAL_COMPOSE_CONFLICT`) — never silently approximated.
* A dependency chain (write-before-read x2) composes to exactly the state
  every legal sequential topological order reaches, proved by a brute-force
  test-only sequential-shadow oracle over every valid permutation on a
  tiny (3-action) domain.
* Action-set permutation reaches one committed identity (same
  `final_state.state_digest`, same `commit_order`) regardless of submission
  order.
* A failing constituent, a reference-rebuild failure (both a raising
  builder and an identity-mismatched one), a stale base, and a tampered
  (hand-edited but structurally valid) transaction all reject cleanly with
  a typed, specific `OperatorTransactionCommitRejectionKind` and leave the
  base state's own source untouched.
* Exact replay: `replay_operator_transaction_commit` recomputes every step
  from base + recorded commit decision and requires identical
  `decision_id` (transitively binding the final state and fresh reference
  table); a tampered recorded decision is detected.
* The composite effect names every target (`property_deltas` count and
  target-fingerprint set match the action count/targets exactly).
* Post-commit legal work uses only the fresh table: a base-table ref for a
  target the fresh table also names resolves against the *base* table but
  raises `ReferenceResolutionError` against the *fresh* one (different
  opaque IDs by design), while the fresh table's own ref for the same
  semantic value resolves fine.
* One conversation turn (`TRANSACTION_COMMIT`) carries the whole
  transaction, retains every constituent `dry_run.application_id` via
  `TurnArtifactV1.application_ids`, and replays exactly through
  `replay_conversation_trace(transaction_replayer=transaction_replayer_for_conversation)`;
  omitting `transaction_replayer` on a trace that needs one fails closed.

**Deliberately excluded (not a stop-rule trigger — see below):**

* **No producer/consumer chaining across new entities.** Unchanged from
  DSH5-04: an action that would need to reference another action's
  *newly-produced* entity (e.g. a node an `ADD_CHILD`-shaped action would
  create) still cannot be expressed in one flat transaction, since that
  entity has no descriptor in the base table to resolve against, and this
  issue's Decision explicitly rejects "chain a second dry run against a
  first action's result" as the fix. Still simply unsupported today.
* **Real value-flow dependencies are not proven correct in general** — only
  proven correct for every operator actually declared in this repository,
  where a "read" footprint is validity bookkeeping, never a value an
  operator's mutation logic consumes from another action's fresh output
  (verified directly: every `PreconditionV1`-derived read only gates
  argument-slot *validity*, and no local/bulk/topology operator's executor
  ever reads the *current value* of a target another prepared action also
  writes). A future operator whose effect genuinely depended on reading
  another action's fresh output inside the *same* transaction would need a
  new decision — this module does not silently assume that case works.
* **Same-container topology overlaps that transactions.py's own footprint
  analysis would already reject as `CONFLICT`** (e.g. two undeclared-
  commuting `ADD_CHILD`s to the same parent+role) never reach this
  executor at all — they are rejected one layer up, at
  `build_operator_transaction`, exactly as DSH5-04 already tests. This
  executor's own structural safety net (item 4 above) is the fallback for
  the narrower case that *does* reach it: a pair declared commuting whose
  real outputs still disagree. `MergeConflictKind`'s specific taxonomy
  (`CHILD_ORDER`, `DELETE_MODIFY`, `SCOPE_BINDER`, …) is exhaustively
  proven for the shared `merge_ast_value` primitive itself by the existing
  `test_operator_merge.py::test_overlapping_effects_return_specific_typed_conflicts`
  suite (unchanged by this issue) — this issue does not re-prove that
  classification once more per kind; it proves only that its own composer
  correctly *propagates* whichever kind fires into a typed rejection
  (one representative case, `SAME_NODE_INCOMPATIBLE_EDIT`, is sufficient
  evidence for that propagation claim, since the propagation code path is
  identical for every kind).

### Selector lineage fix

`transactions.py::_target_lineage` originally read:

```python
def _target_lineage(table: ReferenceTableV1) -> dict[OperatorRef, str]:
    return {entry.ref: entry.descriptor.fingerprint for entry in table.entries}
```

`ReferenceTableV1.selectors` (DSH5-01) is a separate list from `.entries`,
so any operator with a `SelectorRef` argument slot — i.e. `openui.map_set_property`,
the only bulk operator in this repository (DSH5-02) — could never be
prepared into a transaction: `prepare_operator_action` calls
`_legal_semantic_action_id`, which requires every bound argument's ref to
resolve in the lineage map, and `derive_read_write_set`'s precondition
footprint has the same requirement for the selector's own
`"selector.non_empty"` precondition. Both would raise/return `None` before
a bulk operator ever got a chance to participate. This was invisible in
DSH5-04's own test suite because it never exercised a selector-argument
operator. The fix adds `table.selectors` to the same mapping:

```python
return {
    **{entry.ref: entry.descriptor.fingerprint for entry in table.entries},
    **{entry.ref: entry.descriptor.fingerprint for entry in table.selectors},
}
```

Purely additive — no existing (non-selector) lineage entry changes, and
every DSH5-04 test still passes unmodified.

## Sequential-shadow equivalence disposition

The issue asks for "sequential-shadow execution for tests only, rebinding
after each action and comparing valid topological orders on tiny domains."
Implemented in `tests/test_dsl/test_operator_transaction_executor.py` as
`_sequential_shadow_apply` (chains real `OperatorLibraryV1.apply` calls,
regenerating `ApplicationProvenanceV1` — i.e. "rebinding" — against the
*actual current* state after each hop) plus `_all_topological_orders`
(brute-force permutation enumeration filtered by `dependency_edges`,
appropriate only because the domain is tiny — three actions, six
permutations). `test_dependency_chain_composes_and_matches_every_valid_topological_order`
asserts the atomic composer's result equals **every** legal order's
sequential result, not just one. This oracle exists *only* in the test
module; production code never chains sequential applications — the
opposite is the entire point of `compose_operator_transaction`, proved
separately by
`test_compose_operator_transaction_never_chains_against_a_prior_actions_output`
(every constituent replays against the untouched `base_state` directly,
individually re-verified after composition).

## Matrix and controls

Covered in `tests/test_dsl/test_operator_transaction_executor.py`:

* Two/four disjoint local actions —
  `test_two_disjoint_actions_compose_and_commit`,
  `test_four_disjoint_actions_compose_and_commit`.
* Bulk plus primitive —
  `test_bulk_plus_primitive_mixed_transaction_commits`.
* Declared commuting overlap (agree/disagree) —
  `test_declared_commuting_agreeing_overlap_composes`,
  `test_declared_commuting_disagreeing_overlap_rejects_as_structural_compose_conflict`.
* Dependency chain —
  `test_dependency_chain_composes_and_matches_every_valid_topological_order`.
* Same-node/delete-modify/order/scope conflict — covered by reuse: the
  structural safety net's propagation path is generic over
  `MergeConflictKind`, and the underlying classification is exhaustively
  proven by the unchanged `test_operator_merge.py` suite (see "Coverage and
  what's excluded").
* Failing constituent —
  `test_failing_constituent_rejects_and_leaves_base_unchanged`.
* Reference rebuild failure —
  `test_reference_rebuild_failure_rejects_and_leaves_base_unchanged`
  (both a raising builder and an identity-mismatched one).
* Action-set permutation —
  `test_action_set_permutation_reaches_one_committed_identity`.
* Stale/tampered transaction —
  `test_stale_transaction_base_rejects`, `test_tampered_transaction_rejects`.
* Exact replay —
  `test_replay_operator_transaction_commit_matches_recorded_and_detects_tampering`.
* Composite effect coverage / fresh-table-only legal work —
  `test_composite_effect_includes_every_target`,
  `test_post_commit_legal_set_uses_only_the_fresh_table`.
* Conversation turn integration —
  `test_conversation_turn_commits_transaction_and_retains_constituent_application_ids`
  (also proves the fail-closed `transaction_replayer` requirement).
* No-stale-ref-sequential anti-pattern, directly —
  `test_compose_operator_transaction_never_chains_against_a_prior_actions_output`.

## Acceptance

* **Atomic replay-exact success** — every commit's `decision_id`
  (transitively the final state and fresh reference table) is reproduced
  exactly by `replay_operator_transaction_commit` from base + recorded
  transaction alone.
* **Failure leaves base unchanged** — every rejection path is proved to
  leave `fixture.state.source` (the caller's own base) untouched; the
  executor never mutates in place, only ever reads the base and returns a
  brand-new `OperatorStateV1` value on success.
* **Composite effect includes every target** —
  `test_composite_effect_includes_every_target`.
* **Accepted independent permutations reach one state/identity** —
  `test_action_set_permutation_reaches_one_committed_identity`.
* **Post-commit legal sets use only the fresh table** —
  `test_post_commit_legal_set_uses_only_the_fresh_table`.

## Stop-rule disposition

The stop rule fires "if exact composition cannot cover a useful primitive
subset." It does not fire here: exact composition covers every subset
SLM-412 already proves preparable (disjoint footprints, or footprints
resolved by mutual `commutes_with`) for every operator family in this
repository — local primitives, bulk selector operators, and mixed
primitive+bulk transactions all compose correctly, with a real structural
safety net (never a silent approximation) for the one case where a
*declared* commuting pair's real outputs still disagree. The one honest
gap this issue does not paper over is producer/consumer chaining across a
*newly-produced* entity within one transaction — that was already out of
scope at the DSH5-04 preparation layer (no base-table descriptor exists to
resolve such a reference against), and this issue does not invent a
provisional-descriptor mechanism to work around it. A well-documented
narrower executor that is honest about that one limit is the right outcome
here, not a forced general design that pretends to solve it.

## Scope notes (deliberately deferred)

* **Rollback fixtures beyond "the base is never mutated."** Every
  `OperatorStateV1`/`OperatorTransactionV1` in this package is already an
  immutable value; there is no in-place mutation to roll back from, so
  "rollback" reduces to "the executor never returns a partial state and
  never touches the caller's own base object" — proved directly by every
  rejection-path test asserting `fixture.state.source == SOURCE` afterward.
  A separate persisted-transaction-log rollback mechanism (for a caller
  that already committed a *previous* transaction and wants to unwind it)
  is out of scope; `conversation.py`'s existing `undo_conversation` already
  covers "return to a previous conversation state," including one produced
  by a `TRANSACTION_COMMIT` turn, with no new mechanism needed.
* **A general N-way sequential-shadow equivalence checker for arbitrary
  operator sets** is explicitly not shipped as production or reusable test
  infrastructure — the issue asks for it "for tests only" on "tiny
  domains," and the brute-force permutation approach used here would not
  scale past a handful of actions. It is scoped exactly to what the issue
  asks for.
* **Selector-lineage fix scope.** The `_target_lineage` fix only adds
  selector entries to the lineage map; it does not change selector
  freshness/fanout semantics, which remain entirely owned by DSH5-01/DSH5-02
  (`ReferenceTableV1.resolve_selector`, `OpenUILocalOperatorContextV1.resolve_selector`).

This work is compiler contract/unit fixtures exercised through the real
`openui` `DslPack` (parse/canonicalize/schema-oracle/round-trip authority is
the genuine pack backend, not a stub). No train, eval, benchmark, matrix,
checkpoint, model-card, ship-gate, or model-quality claim is produced.
