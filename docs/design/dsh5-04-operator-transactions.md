# DSH5-04 OperatorTransactionV1 base-state, dependency, and conflict contracts

SLM-412 (DSH5-04) is milestone M2's opening decision for "DSH5 — Bulk
Operators, Transactions & Control Plane": how can several actions predicted
against one base state remain valid after commit without reusing stale
request-local references? It extends the DSH3-01 operator contracts
(`ActionEffectV1`, `OperatorMutationV1`, `ApplicationProvenanceV1`), the
DSH3-02/DSH3-03 registry and reference-table machinery, and reuses the
DSH3-19/DSH3-20 `merge.py` conflict-detection pattern — it does not build a
parallel contract system, and it ships **no execution path**: this is a
schema/safety issue only, per the issue's own agent contract. No function in
`src/slm_training/dsl/operators/transactions.py` applies a mutation to
pack-authorized source; every action is dry-run (`OperatorLibraryV1.dry_run`)
against one fixed, unmodified base, never chained against another action's
result. A future issue owns actual multi-action commit execution.

## Decision and hypothesis

Naively re-running a second action's dry run against the *first* action's
post-commit AST would look like it "keeps references fresh," but it is
exactly the G0 staleness this issue's Decision names: action *N* would
resolve its opaque refs against an AST that action *N-1* already changed,
reusing request-local identity across a boundary that DSH3-03 explicitly
scopes to one state. The hypothesis this issue tests: a transaction contract
that (1) prepares every action against the *same* base, (2) derives semantic
read/write sets from each action's *exact* effect and its operator's
*declared* precondition footprint (never from operator names), (3) records
dependencies and conflicts from those footprints, and (4) commits one
composite proof, can solve staleness without pretending sequential
application stays fresh.

## Contract

`src/slm_training/dsl/operators/transactions.py`:

* **`PreparedOperatorActionV1`** — one action dry-run against a fixed base:
  `base_state_digest`/`base_ast_digest`, `operator_fingerprint`,
  `semantic_action_id`, `semantic_arguments` (tuple of `SemanticArgumentV1`:
  `slot_id`/`ref_kind`/`target_fingerprint`, never an opaque ref),
  `effect` (`ActionEffectV1`), `proof` (`ApplicationProofV1`), and `dry_run`
  (the full `OperatorApplicationV1`, kept as **evidence only** — see
  "Opaque IDs are evidence, never authority" below). `__post_init__` requires
  the dry run to have succeeded and requires `effect`/`proof` to equal the
  dry run's own `effect`/`proof` — a `PreparedOperatorActionV1` can never
  silently disagree with the dry run it claims to summarize.
* **`OperatorReadWriteSetV1`** — `reads`/`writes`, each a canonically
  ordered tuple of `TargetFootprintV1` (`target_fingerprint` +
  `categories`). Derived by `derive_read_write_set`:
  * **writes** reuse `merge.py`'s own `_effect_targets(effect, lineage)`
    directly (not a re-implementation) — every `consumed_roles`/
    `produced_roles`/`consumed_binders`/`produced_binders` ref and every
    delta target (`scope`/`cardinality`/`property`/`topology`), mapped to
    its stable `ReferenceDescriptorV1.fingerprint`.
  * **reads** are the *declared precondition argument-slot footprint*: for
    every `PreconditionV1.argument_slots` name on the operator's own
    `AstOperatorV1`, the bound argument at that slot is a read of its
    resolved target — never derived from `operator_id`/name.
  * Returns `None` — an **unknown footprint** — when `effect.compiler_coverage`
    is not `EXACT`, or when any referenced target is absent from the base
    table's lineage. The caller fails closed on `None`; it never guesses a
    partial footprint.
* **`OperatorTransactionV1`** — one base
  (`base_state_digest`/`base_ast_digest`/`base_reference_table_fingerprint`),
  canonically ordered `prepared_actions`, `dependency_edges`
  (`OperatorDependencyEdgeV1`), `conflict_graph`
  (`OperatorConflictEdgeV1` — evidence of *resolved* overlaps only),
  `composite_effect` (one concatenated `ActionEffectV1` across every
  prepared action, `EXACT` coverage), `expected_final_state_digest`/
  `expected_final_ast_digest` (deterministic *commitment* digests — see
  "Expected final digests are commitments, not executed hashes" below),
  `proof` (`OperatorTransactionProofV1`), `provenance`
  (`ApplicationProvenanceV1`), and `coverage` (`CompilerCoverage`, always
  `EXACT` for a constructed instance).
* **`OperatorTransactionDecisionV1`** — exactly one of `transaction` or
  `rejection`, mirroring `merge.py`'s `BranchMergeDecisionV1` envelope
  exactly (same "exactly one" invariant, same `decision_id` digest pattern).

## Canonicalization rule

`prepared_actions`, `dependency_edges`, and `conflict_graph` are always
ordered by `semantic_action_id` (actions) or `(predecessor, successor)` /
`(left, right)` pairs of semantic IDs (edges) — **never** by submission
order or any opaque ID. `build_operator_transaction` always sorts before
constructing; a directly constructed `OperatorTransactionV1` that violates
this raises `ValueError` immediately from `__post_init__`, mirroring exactly
how `merge.py`'s `BranchMergeConflictV1`/`BranchMergeArtifactV1` enforce
canonical branch/application ID order today. Two callers submitting the same
action *set* in different orders always build byte-identical instances —
proved directly by
`test_action_set_permutation_shares_canonical_transaction_identity`.

## Conflict rules, and how they compose with `merge.py`

For every pair of prepared actions, `build_operator_transaction` computes:

* `write_write` = left's writes ∩ right's writes,
* `write_read` = left's writes ∩ right's reads,
* `read_write` = right's writes ∩ left's reads.

If any of the three is non-empty, the pair conflicts **unless both
operators mutually declare the other in `commutes_with`** — the exact same
`mutually_commuting` gate `merge.py`'s `merge_conversation_branches` already
uses (`declarations[1].operator_id in declarations[0].commutes_with and`
`declarations[0].operator_id in declarations[1].commutes_with`). An
unresolved overlap rejects the **whole transaction** as
`OperatorTransactionRejectionKind.CONFLICT`, naming both actions' semantic
IDs and the overlapping target fingerprints — never a partial commit of the
non-conflicting subset.

**Direct reuse, not duplication:** the write-footprint extraction imports and
calls `merge.py`'s own `_effect_targets` unmodified. The mutual-commutativity
gate is the identical trust boundary `merge.py` already established (both
operators must name each other), applied here to a flat *N*-way, one-base
action set instead of a two-branch fork. Pairwise overlap *classification*
(is this a write/write, write/read, or read/write overlap?) is implemented
locally rather than by importing `merge.py`'s `_conflict_kind`, because that
helper's taxonomy (`DELETE_MODIFY` / `CHILD_ORDER` / `SCOPE_BINDER` / ...) is
specific to structurally merging two AST branches; a transaction only needs
to know *whether* an overlap is resolved and *which direction* implies a
commit-order dependency, not which of merge.py's fork-merge categories it
resembles.

A resolved overlap is recorded twice, for two different purposes:

* **`conflict_graph`** (`OperatorConflictEdgeV1`) — evidence that the pair
  overlapped and was resolved (`resolved_by_commutativity=True` always;
  `__post_init__` rejects constructing one any other way).
* **`dependency_edges`** (`OperatorDependencyEdgeV1`) — the *required commit
  order* this overlap implies:
  * `write_read` (left writes a target right reads) → edge left→right,
    `reason="write_before_read"`.
  * `read_write` (right writes a target left reads) → edge right→left,
    same reason.
  * A pure `write_write` overlap with no cross read/write direction has no
    natural producer/consumer direction, so it is ordered by a semantic-ID
    tie-break (`min → max`), `reason="write_write_order"`.
  * If a pair's overlap has both a read/write direction *and* a leftover
    write/write-only target, both reasons merge into one edge
    (`reason` becomes `"write_before_read-write_write_order"`).

Both directions of a `write_read`/`read_write` pair (e.g. left writes what
right reads, **and** right writes what left reads — genuinely
mutually-dependent, not just doubly-commuting) each add their own directed
edge. A DFS cycle check (`_has_cycle`) then runs over the full directed
graph; a real cycle — proven directly in
`test_mutual_write_read_overlap_forms_a_dependency_cycle_and_rejects` with
exactly this two-target, two-action, mutual-write/read scenario — rejects the
transaction as `DEPENDENCY_CYCLE` rather than picking an arbitrary,
unjustifiable order.

## Typed rejections

Two mechanisms, matching existing precedent exactly:

| Rejection | Mechanism | Precedent |
| --- | --- | --- |
| `stale_base` | `OperatorTransactionRejectionKind` via `OperatorTransactionDecisionV1.rejection` | `merge.py`'s `MergeConflictKind.STALE_REF` |
| `duplicate_action` | same | new — no merge.py analog |
| `dependency_cycle` | same (from the builder) **and** a structural `ValueError` if a caller directly constructs a cyclic `OperatorTransactionV1` | — |
| `unknown_footprint` | same | `merge.py` treats non-exact coverage as `UNSUPPORTED_EFFECT` |
| `conflict` | same | `merge.py`'s per-kind conflict taxonomy |
| `partial_preparation` | same (only from `prepare_operator_transaction`, the prepare+assemble convenience entry point) | new |
| `fanout_exceeded` | same | DSH5-02's `MAP_SET_PROPERTY_MAX_FANOUT` |
| `noncanonical_order` | **structural** `ValueError` from `OperatorTransactionV1.__post_init__` only — never a `OperatorTransactionRejectionKind` member | `merge.py`'s `BranchMergeConflictV1`/`BranchMergeArtifactV1` canonical-ID-order checks |

`noncanonical_order` is deliberately *not* a `Decision`-path rejection: the
builder always constructs canonical order itself, so the only way to
observe it is a directly (mis)constructed artifact — exactly the same
distinction `merge.py` already draws between its `MergeConflictKind` enum
(data-dependent conflict causes) and its plain `ValueError`s for canonical
ID ordering.

## Unknown effects and approximate coverage reject

Per the issue's own conflict-rules wording, this is deliberately **not**
folded into `PreparedOperatorActionV1` construction: a dry run can *succeed*
(a real `ApplicationProofV1`) while still declaring `BOUNDED`/`APPROXIMATE`
`compiler_coverage` on its effect — that is different from an outright
*rejected* dry run (`PARTIAL_PREPARATION`). `PreparedOperatorActionV1` only
requires the dry run to have *succeeded*; `derive_read_write_set` is the
single place that turns non-exact coverage into `UNKNOWN_FOOTPRINT`, proved
by `test_unknown_effect_and_approximate_coverage_reject_as_unknown_footprint`.

## Opaque IDs are evidence, never authority

`PreparedOperatorActionV1.dry_run` (a full `OperatorApplicationV1`) is kept
purely as replay evidence, the same reason every other
`OperatorApplicationV1` in this package is kept. No downstream identity —
`semantic_action_id`, `SemanticArgumentV1.target_fingerprint`, read/write
footprints, dependency/conflict-graph keys, canonical ordering — is ever
keyed by an opaque ref or its `opaque_id`; everything routes through
`ReferenceDescriptorV1.fingerprint` (via `_target_lineage`) instead.
`test_opaque_id_hidden_duplicate_action_rejects_via_semantic_identity` proves
this directly: the same read footprint, prepared through two *different*
reference tables (one a DSH3-03 `ReferenceTableV1.permuted()` reallocation of
the other, so every opaque ID differs) produces an *identical*
`semantic_action_id` and is caught as `DUPLICATE_ACTION` — an opaque-ID
disguise never hides a duplicate action from this contract. Symmetrically,
`test_conflict_derives_from_target_fingerprint_not_operator_name` proves two
operators with unrelated names still conflict when they target the same
semantic fingerprint — conflict detection never depends on operator naming.

## Expected final digests are commitments, not executed hashes

This issue never executes, so `OperatorTransactionV1.expected_final_state_digest`/
`expected_final_ast_digest` cannot be literal post-commit AST hashes (that
would require actually applying every action's mutation together). They are
instead **deterministic commitment digests** — `_fingerprint` over the base
digest, the composite effect's fingerprint, and the canonical action-ID list.
A future execution issue must reproduce these exact digests (or fail closed)
when it actually commits the transaction; this issue only records what a
correct execution is committed to producing, never a text-level guess.

## Application proof and provenance

`OperatorTransactionProofV1` binds `composite_effect_fingerprint` to the
composite effect and carries a `transaction_result_digest` hashing the base
digest and every prepared action's own `fingerprint` (each of which already
transitively binds its `dry_run`/`effect`/`proof`). `provenance`
(`ApplicationProvenanceV1`) is the one, transaction-wide identity — every
prepared action's own `dry_run.provenance.request_id` must equal it
(`STALE_BASE` otherwise), which is exactly "resolve all source refs against
the base table": one base, one request, one reference table.

## Matrix and controls

Covered in `tests/test_dsl/test_operator_transactions.py`:

* Disjoint property edits build one transaction, no conflicts
  (`test_disjoint_property_edits_build_one_transaction_no_conflicts`).
* Declared commuting write/write overlap resolves with a dependency edge
  (`test_declared_commuting_write_write_overlap_is_resolved_with_a_dependency_edge`).
* Write/write and write/read conflicts without commutativity reject
  (`test_write_write_conflict_without_commuting_rejects`,
  `test_write_read_conflict_without_commuting_rejects`).
* Dependency chain orders three actions by write-before-read
  (`test_dependency_chain_orders_actions_by_write_before_read`); a genuine
  mutual write/read overlap forms a dependency cycle and rejects
  (`test_mutual_write_read_overlap_forms_a_dependency_cycle_and_rejects`).
* Unknown/approximate-coverage effect rejects as unknown footprint
  (`test_unknown_effect_and_approximate_coverage_reject_as_unknown_footprint`).
* Action-set permutation shares canonical transaction identity
  (`test_action_set_permutation_shares_canonical_transaction_identity`).
* Rejected controls: name-derived footprints
  (`test_conflict_derives_from_target_fingerprint_not_operator_name`), mixed
  bases (`test_mixed_bases_reject_as_stale_base`), a stale reference table at
  prepare time
  (`test_stale_reference_table_rejects_when_preparing_an_action`),
  partial-as-exact
  (`test_partial_preparation_rejects_the_whole_batch_not_just_the_failure`),
  and opaque-ID-hidden duplicates
  (`test_opaque_id_hidden_duplicate_action_rejects_via_semantic_identity`).
* Fanout/budget boundary
  (`test_fanout_exceeded_rejects`, `OPERATOR_TRANSACTION_MAX_ACTIONS = 8`,
  matching the DSH5-02 precedent).
* Noncanonical order — prepared actions, dependency edges, and conflict
  graph each raise `ValueError` on direct construction
  (`test_noncanonical_prepared_action_order_raises`,
  `test_noncanonical_dependency_and_conflict_graph_order_raises`); a directly
  constructed cyclic edge pair raises too
  (`test_dependency_cycle_direct_construction_raises`).
* Serialization/digest/migration fixtures: every pure-digest schema
  (`TargetFootprintV1`, `OperatorReadWriteSetV1`, `SemanticArgumentV1`,
  `OperatorDependencyEdgeV1`, `OperatorConflictEdgeV1`) round-trips through
  `to_dict`/`from_dict`, and an unrecognized `schema` string fails closed
  (`test_serialization_round_trips_and_migration_guards_reject_unknown_schema`).
* No-execution fixture: `transactions.py`'s `__all__` never exposes an
  apply/commit/execute-shaped name, and preparing/building a transaction
  never mutates pack-authorized source
  (`test_no_execution_is_ever_wired_from_this_module`).
* Replay identity: recomputing a transaction from the same inputs matches a
  previously recorded decision exactly; a tampered `recorded` decision is
  detected (`test_replay_operator_transaction_matches_recorded_decision`).

## Acceptance

* **Canonical permutation-invariant transaction identity** —
  `test_action_set_permutation_shares_canonical_transaction_identity`.
* **Conflicts/dependencies derive from exact semantics** — every check
  routes through `ActionEffectV1`/declared preconditions and
  `ReferenceDescriptorV1.fingerprint`, never `operator_id`/name
  (`test_conflict_derives_from_target_fingerprint_not_operator_name`).
* **No post-first-commit base-ref reuse** — nothing in this module chains a
  second dry run against a first action's result; every prepared action's
  `base_state_digest`/`base_ast_digest` must equal the transaction's own, and
  `PreparedOperatorActionV1`'s opaque `dry_run` is never replayed to author
  a second application.
* **Unknown/partial preparation fails closed** — `UNKNOWN_FOOTPRINT`,
  `PARTIAL_PREPARATION`, and `FANOUT_EXCEEDED` are typed, never a silent
  subset commit.

## Stop-rule disposition

The stop rule fires if exact footprints cannot be derived for all inventory.
They can: every write footprint already has an exact-fingerprint source
(`ActionEffectV1`'s existing `compiler_coverage` field, already
`EXACT`-or-reject across this whole package since DSH3-01), and every read
footprint is derived from the operator's own already-existing
`PreconditionV1.argument_slots` declaration — no operator in this repository
needed a new declaration shape to participate. This issue does **not**
restrict the transaction contract to a proven subset; it applies uniformly
to every declared `AstOperatorV1`.

## Scope notes (deliberately deferred)

* **No commit/execution path.** Per the issue's own agent contract, this
  module never wires into `OperatorLibraryV1._execute`/`apply`, never applies
  a mutation, and exposes no apply/commit/execute-shaped name
  (`test_no_execution_is_ever_wired_from_this_module`). A future issue owns
  actually replaying a committed `OperatorTransactionV1` as one atomic
  multi-action AST edit and proving its real post-commit digests match
  `expected_final_state_digest`/`expected_final_ast_digest`.
* **No producer/consumer chaining across new entities.** Every prepared
  action resolves its arguments against the *same* base reference table;
  an action that would need to reference another action's *newly produced*
  entity (e.g. a node an `ADD_CHILD`-shaped action would create) cannot be
  expressed in one flat transaction today, since that entity has no
  descriptor in the base table to resolve against. This is the direct
  consequence of "never chain a second dry run against a first action's
  result" — solving it for real would require exactly the sequential
  chaining this issue's Decision rejects. A future issue may explore
  a provisional-descriptor mechanism for this case; today it is simply
  unsupported (the base table's `unknown_footprint`/`ref.missing` failure
  path already fails closed on it, never silently degrading).
* **Read/write categorization reuses `merge.py`'s write-side tags verbatim**
  (`"role"`/`"binder"`/`EffectDeltaKind.value`) rather than inventing a
  parallel taxonomy; read-side tags use each argument's own `RefKind.value`
  instead, since preconditions do not carry a `merge.py`-style delta kind.
  Category tags are evidence/documentation on `TargetFootprintV1`, never
  inputs to the conflict decision itself (only target-fingerprint overlap
  is).

These are compiler contract/unit fixtures exercised through the real
`openui` `DslPack` (parse/canonicalize/schema-oracle/round-trip authority is
the genuine pack backend, not a stub). No train, eval, benchmark, matrix,
checkpoint, model-card, ship-gate, or model-quality claim is produced.
