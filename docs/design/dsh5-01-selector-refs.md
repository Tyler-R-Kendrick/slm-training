# DSH5-01 exact SelectorRefV1 contracts and finite compiler-owned selector domains

SLM-409 (DSH5-01) is milestone M1's opening decision for "DSH5 — Bulk
Operators, Transactions & Control Plane": how can one model-visible argument
denote a bounded exact set of AST targets without introducing a free-form
query language or weakening reference freshness? It extends the DSH3-01
typed-reference contracts and the DSH3-03 permutation-invariant reference
table — it does not introduce a parallel reference system, and it ships no
bulk operator. The bulk operator that consumes a resolved selector is a
later M1 issue; nothing here executes a mutation.

## Contract

* `RefKind.SELECTOR` and `SelectorRef` (`src/slm_training/dsl/operators/contracts.py`)
  are one more opaque, request-local reference kind alongside `NodeRef` /
  `RoleRef` / ... — same `_OpaqueRef` shape, same "no display field" guarantee.
* `SelectorKind` (`src/slm_training/dsl/operators/references.py`) is the
  closed, compiler-owned predicate-class enum: `component_type_in_scope`,
  `schema_role_in_scope`, `symbol_references`, `descendants_with_role`. There
  is no free-form predicate string anywhere in this contract — a selector's
  semantics are always exactly one of these four classes.
* `SelectorFact` is a second closed enum for the allowlisted inference-visible
  facts a selector may carry (`selector.exact_finite`,
  `selector.fanout_bounded`, `selector.pack_authorized`,
  `selector.scope_rooted`). Arbitrary strings are structurally rejected —
  `SelectorFact("...")` raises `ValueError` for anything outside the enum.
* `SelectorDescriptorV1` commits to: `selector_kind`, `semantic_fingerprint`,
  `scope_fingerprint`, the **exact sorted** `target_fingerprints` (each one a
  target's `ReferenceDescriptorV1.fingerprint`, never a raw AST/node payload),
  `cardinality` (must equal `len(target_fingerprints)`), and `max_fanout`
  (`cardinality` must never exceed it — enforced in `__post_init__`, so an
  over-fanout descriptor cannot be constructed at all). No raw user query or
  target surface is ever a field.
* `SelectorEntryV1` pairs one `SelectorRef` with its `SelectorDescriptorV1`,
  same shape as `ReferenceEntryV1`/`ReferenceDescriptorV1`.
* `ReferenceTableV1` gained a `selectors: tuple[SelectorEntryV1, ...]` field
  and bumped its schema to `operator_reference_table/v2`. `__post_init__` now
  also checks selectors for cross-request reuse and opaque-ID duplication
  alongside ordinary entries, and rejects any selector whose
  `target_fingerprints` are not all already present among the table's own
  entry descriptor fingerprints (`selector.member_missing`) — this is the
  structural guard against gold-derived or otherwise invented selectors: a
  selector can only ever point at things the compiler already put in the
  table.

## Allocation

Selector IDs are allocated through the same permutation-safe path as every
other reference kind:

* `build_reference_table(..., selector_descriptors=...)` allocates entries and
  selectors together for a fresh request/state/branch, sorting each set by
  descriptor fingerprint before hashing an opaque ID and shuffling — input
  order and wire IDs carry no semantics, exactly as for `NodeRef`/`RoleRef`.
* `attach_selectors` / `attach_selector` (`references.py` / `selectors.py`)
  add selectors to an existing table **without disturbing already-issued
  entry refs** — only the selector set is reallocated (existing selectors get
  new opaque IDs too, same as a `.permuted()` call would produce, since they
  are re-hashed against the new seed). A caller who already handed out
  `NodeRef`s from the table keeps resolving them after a selector is added.
* `build_selector_descriptor` / `build_selector` (`selectors.py`) build one
  descriptor from a set of `matching_refs` that must already be entries of the
  given table — mismatched, invented, or duplicated refs fail closed before a
  `SelectorRef` is ever allocated.

## Resolution and stable failures

`ReferenceTableV1.resolve_selector` mirrors `.resolve()`'s freshness checks
and adds selector-specific membership drift checks. Order matters — cross
identity checks run before content checks:

| Code | Meaning |
| --- | --- |
| `selector.cross_request` | Selector belongs to another request. |
| `selector.stale_state` | Table/selector is being used against a changed state. |
| `selector.cross_branch` | Selector is being used on another branch. |
| `selector.type_incompatible` | A non-selector ref was presented as a selector. |
| `selector.missing` | Opaque ID is not present in the bound table. |
| `selector.duplicate` | Table contains an ambiguous selector opaque ID. |
| `selector.wrong_kind` | Resolution's expected `SelectorKind` does not match the descriptor. |
| `selector.scope_changed` | The freshly recomputed scope no longer matches the committed scope. |
| `selector.duplicate_target` | The current candidate set (or a descriptor's own targets) contains a repeat. |
| `selector.fanout_overflow` | The current candidate set exceeds `max_fanout` — checked both at build time (`SelectorDescriptorV1.__post_init__`, so an over-bound descriptor cannot exist) and at resolve time against a freshly recomputed set. |
| `selector.target_set_changed` | The freshly recomputed exact membership differs from the committed set. |
| `selector.unsorted_targets` | A descriptor's targets were not canonically sorted. |
| `selector.cardinality_mismatch` | A descriptor's claimed cardinality does not equal its target count. |
| `selector.member_missing` | A selector's target (or a build's `matching_refs`) is not an entry of the owning table. |
| `ref.table_schema_unsupported` | `ReferenceTableV1.from_dict` saw a schema string it does not recognize. |

No stale, cross-branch, wrong-kind, or drifted resolution ever returns a
descriptor — every failure raises `ReferenceResolutionError` with one of the
codes above before any target payload is exposed.

## Migration

`operator_reference_table/v1` payloads never carried selectors.
`ReferenceTableV1.from_dict` treats that schema string as an explicit
migration: `selectors` defaults to `()` rather than being guessed at from a
missing key. Any other unrecognized schema string (including a hypothetical
future `/v3`) is rejected outright with `ref.table_schema_unsupported` —
`from_dict` never silently guesses at an unknown shape. A round-trip test
(`test_selector_ref_table_migrates_legacy_payload_and_rejects_unknown_schema`)
covers both paths.

## Matrix and controls

Covered in `tests/test_dsl/test_operator_selectors.py`:

* Zero / one / many matches (`test_zero_one_and_many_match_selectors_build_exact_membership`).
* Nested scopes carry independent, non-interfering target sets
  (`test_nested_scopes_produce_independent_selectors`).
* Mixed selector kinds coexist in one table
  (`test_mixed_selector_kinds_coexist_in_one_table`).
* Fanout boundary — exactly at the bound succeeds, one over fails closed at
  build time (`test_fanout_boundary_at_and_over_the_bound`) and at resolve
  time against a freshly inflated candidate set
  (`test_current_fanout_overflow_fails_closed_on_resolve`).
* State / branch changes fail closed
  (`test_state_and_branch_changes_fail_closed`), as does cross-request reuse
  (`test_cross_request_reuse_fails_closed`).
* Opaque-ID and candidate-order permutation preserve resolution and semantic
  result (`test_opaque_id_and_candidate_order_permutation_preserve_resolution`),
  mirroring the DSH3-03 permutation evidence.
* Rejected by construction: arbitrary predicate strings
  (`test_selector_kind_is_closed_and_rejects_arbitrary_strings`), gold-derived
  selectors (`test_gold_derived_selectors_are_rejected_as_member_missing`),
  target-order semantics (`test_target_order_carries_no_semantics`), stale
  members via changed target sets
  (`test_changed_scope_and_changed_target_set_fail_closed`),
  truncated-complete claims
  (`test_truncated_scan_is_unknown_and_never_forceable`), and duplicate
  targets both at build time
  (`test_duplicate_targets_are_rejected`) and at resolve time
  (`test_duplicate_current_targets_fail_closed_on_resolve`).
* The context resolver returns independent, defensive-copy tuples, never a
  live view into table storage
  (`test_context_resolver_returns_defensive_copies_of_members`).

`build_selector_from_scope` is the bounded-budget entry point: it scans a
canonically ordered (by descriptor fingerprint) prefix of `table.entries` up
to `max_candidates_scanned`. If that budget truncates the scan before every
entry is checked, the result's `verdict` is `SelectorBuildVerdict.UNKNOWN` and
`descriptor` is `None` — there is no method that turns an `UNKNOWN` result
into a committed `SelectorRef`; the only way to get a resolved selector is to
scan the full candidate set. This mirrors the DSH3-06 legal-set treatment of
budget truncation (`OperatorSupportVerdict.UNKNOWN`, never a forced
singleton).

## Stop-rule disposition

The stop rule fires if exact finite selectors require unstable target
identities or hidden semantics. They do not: every target identity a selector
commits to is a `ReferenceDescriptorV1.fingerprint` already produced by the
DSH3-03 permutation-invariant allocation path, and every selector semantic is
one of four closed, compiler-declared classes with no free-form channel. Bulk
operator work is **not** stopped; it proceeds to the next M1 issue.

## Scope notes (deliberately deferred)

This issue delivers the contract, builder, and resolver layer only:

* No bulk operator consumes a `SelectorRef` yet — that is the next M1 issue,
  which nothing in this repository can start before this contract lands.
* No OpenUI-pack-specific scope/component-type/role extraction is wired here.
  `build_selector_descriptor` / `build_selector_from_scope` take an
  already-computed `matching_refs` set or a typed `predicate` callable over
  `ReferenceEntryV1` records — the caller (a future pack-integration change)
  is responsible for deciding *which* table entries satisfy a given
  `SelectorKind` using the pack's real scope/type/role facts. This keeps the
  fixture-level evidence in this change self-contained and consistent with
  the DSH3-03/DSH3-06 precedent of contract/unit-level fixtures with no
  train/eval/checkpoint/model-quality claim.
* `clone_reference_table_for_branch` (DSH3-08) does not carry selectors across
  a fork. Selectors are branch-bound; a forked branch's scope and membership
  need re-validation against the new branch state rather than being copied
  forward as stale evidence, so leaving them behind on fork is the
  fail-closed choice, not a gap.

These are compiler contract/unit fixtures. No train, eval, benchmark, matrix,
checkpoint, model-card, ship-gate, or model-quality claim is produced.

## Review follow-up (2026-07-25)

Two review findings against the initial cut were fixed in place rather than
deferred, since both were small and could otherwise silently corrupt the
guarantees this contract exists to make:

* `SelectorDescriptorV1.__post_init__` now normalizes `compiler_facts`
  (dedupe + sort by value) at construction, matching the ordering `to_dict`
  already serialized. Before this, a descriptor built from unsorted or
  duplicated facts satisfied every validation check and had the right
  `fingerprint`, but `from_dict(to_dict(x)) == x` could fail — the
  deserialized copy compared unequal to the original despite being
  semantically identical.
* `attach_selectors` now dedupes the combined existing+new descriptor list by
  `fingerprint` before allocating opaque IDs. Before this, re-attaching an
  identical selector (same kind/scope/targets, hence the same
  `descriptor_fingerprint` and opaque ID) produced two entries with the same
  key and `ReferenceTableV1.__post_init__` rejected the table with the
  entry-level `ref.duplicate` code — a misleading failure for an otherwise
  idempotent re-derivation of the same selector.

Added regression coverage for both, plus for the previously-untested
`selector.cardinality_mismatch` and `selector.type_incompatible` rejection
codes and the entry-ref-stability guarantee (`attach_selector` reallocates
only the selector set; a `NodeRef` a caller already holds keeps resolving to
the same descriptor afterward).
