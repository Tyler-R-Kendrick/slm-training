# DSH3-22 sanitized OperatorPolicyInputV1 model boundary

SLM-397 (DSH3-22) answers milestone M5's boundary question for "DSH3 — CAP2
Compiler-Owned AST Operators": which compiler-owned facts may a learned
policy consume without leaking identities, targets, proofs, or future
outcomes? It extends the DSH3-01 typed-reference contracts
(`src/slm_training/dsl/operators/contracts.py`), the DSH3-03 permutation-
invariant reference table (`references.py`), and the DSH3-06 legal-set
enumeration (`legal_set.py`) — it does not change any of their evidence
schemas, execute anything, or train anything. It defines the only allowed
input for all DSH3 M6/M7 learned-operator-policy work
(`src/slm_training/models/operator_policy_view.py`).

## Why

`src/slm_training/models/legal_edit_batch.py` hashes categorical strings and
successor fingerprints into floats (`_stable_scalar`) for the existing
dynamic-pointer bridge batch. The operator runtime carries much richer typed
facts (`ReferenceDescriptorV1`, `OperatorArgumentDomainV1`,
`OperatorLegalEntryV1`) but had no explicit model-safe boundary over them —
consuming those objects directly would leak `semantic_fingerprint`,
`application_id`, `proof_fingerprint`, and other opaque/identity fields a
policy must never see (SLM-398, the typed-embedding replacement for
`_stable_scalar`, is blocked on this boundary existing first).

## Contract

Three immutable, frozen dataclasses, all in
`src/slm_training/models/operator_policy_view.py`:

* `ReferenceModelViewV1` — one reference row: `ref_kind`, `value_type`,
  `compiler_facts`, and parent relation as a **row-local join**
  (`has_parent`, `parent_row: int | None`) rather than the raw
  `parent_fingerprint` hash. `relative_position` is carried only for
  `RefKind.INDEX` (the same small ordinal `ReferenceDescriptorV1.position`
  already stores; it is a genuine semantic integer, not an opaque identity).
* `OperatorArgumentSlotViewV1` — one operator argument slot's `ref_kind`,
  `binding_phase`, `required`/`repeated`, `domain_complete`
  (`OperatorArgumentDomainV1.complete`), and `candidate_rows: tuple[int, ...]`
  — row-local joins into the same table's reference rows, never `OperatorRef`
  objects or their opaque IDs.
* `OperatorActionViewV1` — one operator declaration's `operator_id`,
  `operator_version`, `locality`, `cost`, `effect_signature` (the
  `EffectDeltaKind` tuple only — never a bound `ActionEffectV1`'s concrete
  before/after deltas, which would reveal target content), its
  `argument_slots`, and the entry's `verdict` /
  `coverage` (`OperatorSupportVerdict` / `LegalSetCoverage`, exposed exactly
  as the legal-set enumeration computed them — never forced to "complete").
* `OperatorPolicyInputV1` — the top-level snapshot: `reference_rows`,
  `action_rows`, `ordinary_action_count` (a count only — the DSL's ordinary,
  non-operator action strings are already public grammar tokens, but nothing
  about routing among them is a DSH3-22 concern), and the aggregate
  `coverage`.

Excluded from every one of the above (enforced structurally — none of these
dataclasses has a field for them): `semantic_fingerprint`/`semantic_id`,
`application_id`, `proof_fingerprint`/`proof_checks`, `opaque_id`/
`request_id`, `before`/`after` effect deltas, `target`/`target_ast`,
`successor_fingerprint`, `rejection`/`rejection_samples`, and any planner
choice or future-witness field. `OperatorPolicyInputV1` never holds an
`OperatorApplicationV1`, `BoundArgumentV1`, or `OperatorRef` — only row
integers and the allowlisted scalars above.

## Recursive forbidden-field validation

`FORBIDDEN_FIELD_NAMES` is a frozenset of runtime evidence field names
(`validate_no_forbidden_fields`, `ForbiddenFieldError`). It walks any nested
`dict`/`list`/`tuple` recursively and raises on the first forbidden key at
any depth. `OperatorPolicyInputV1.__post_init__` calls it against its own
`to_dict()` output unconditionally — a future field that collides with a
forbidden name fails closed at construction time, not silently. This is
defense in depth for the hand-authored views above (which cannot currently
carry a forbidden field, since their dataclass shape has no such field) and
the load-bearing assertion for the adversarial-injection test matrix
(`tests/test_models/test_operator_policy_view.py::
test_forbidden_field_is_rejected_at_every_nesting_depth`, parametrized over
every name in `FORBIDDEN_FIELD_NAMES`, at both a shallow and a deeply nested
position).

## Building a view: freshness and registry fail-closed

`build_operator_policy_input(reference_table, legal_set, library)`:

* Fails closed with `policy_view.stale_reference_table` if
  `legal_set.reference_table_fingerprint != reference_table.fingerprint` —
  mirroring the runtime's own `ref.stale_state` guarantee, a stale or
  substituted table can never produce a view.
* Fails closed with `policy_view.registry_mismatch` if
  `legal_set.registry_fingerprint != library.registry_fingerprint`.
* Resolves each reference's parent relation by matching
  `descriptor.parent_fingerprint` against the *current* table's own
  `semantic_fingerprint`s only — a parent that is not itself a resolvable
  reference in this table yields `has_parent=True, parent_row=None`
  (explicit "known-to-exist-but-unresolvable", never guessed at or silently
  dropped).
* Pulls `operator_version`/`locality`/`cost`/`effect_signature` from the
  `OperatorLibraryV1` declaration (`library.lookup(entry.operator_id)`) —
  `OperatorLegalEntryV1` itself only carries the operator's fingerprint, not
  its typed metadata, so the declaration is the only place these come from.

## Canonical serialization order for evidence only

`OperatorPolicyInputV1.to_dict()` sorts `reference_rows` by allowlisted
content only (`ref_kind`, `value_type`, `compiler_facts`, `has_parent`,
`relative_position`) and `action_rows` by
`(operator_id, operator_version, verdict)`, then **remaps every
`parent_row`/`candidate_rows` join** through the new canonical numbering.
This is the property the permutation tests depend on: two
`OperatorPolicyInputV1` instances built from `ReferenceTableV1` and
`ReferenceTableV1.permuted(seed)` (same underlying descriptors, different
opaque IDs and build-time row order) serialize to byte-identical dicts. The
live object itself (`reference_rows`/`action_rows` as constructed, before
`to_dict()` canonicalizes) keeps whatever row order the caller's table gave
it — a scorer consuming the object directly must be equivariant under that
row order, exactly as `references.py`'s own permutation contract requires;
`to_dict()` only defines a canonical order for reproducible JSON evidence
artifacts, it is not claiming the live object's row order is itself
canonical.

`canonical_row_maps()` exposes that same live-to-canonical remapping for
evaluator-only supervision stored beside `to_dict()`. Callers must remap
accepted action rows, argument rows, and hard-negative rows through it;
otherwise a canonical evidence payload could retain a valid-looking label
that points at a different row. The maps contain only row integers and do not
extend the model-input allowlist.

`operator_policy_input_from_dict()` rehydrates only this canonical,
forbidden-field-checked payload. It lets a trainer consume the exact row order
the corpus labels target; it never rebuilds a policy view from runtime
objects or adds an identity/proof field to the model boundary.

## Verification matrix

Covered in `tests/test_models/test_operator_policy_view.py`:

* Local property / basic supported operator — allowlisted fields only, no
  identity fields present
  (`test_supported_operator_view_carries_only_allowlisted_facts`).
* Topology with an index reference and its parent — `parent_row` resolves to
  the parent's row, `relative_position` carries the index's ordinal, and a
  non-index reference never carries a position
  (`test_index_reference_exposes_parent_row_and_relative_position`).
* Partial legal set from an unbounded repeated slot — `UNKNOWN`/`PARTIAL` is
  explicit on both the action view and the aggregate input, and
  `domain_complete` is `False`, never silently promoted to complete
  (`test_partial_legal_set_from_unbounded_repeated_slot_is_explicit`).
* Fresh post-merge / stale-table rejection — a legal set built from one table
  cannot produce a view against a different table
  (`test_stale_reference_table_cannot_produce_a_view`).
* Opaque-ID and candidate-order permutation preserve the exact view
  (`test_opaque_id_and_candidate_order_permutation_preserve_the_view`).
* Canonical action/reference maps preserve evaluator-only joins after both
  axes reorder (`test_canonical_row_maps_keep_external_labels_joined_to_persisted_rows`).

## Adversarial controls

* Changing only the reference-table allocation seed changes every
  downstream application/proof/semantic hash (asserted disjoint) while the
  sanitized view stays byte-identical
  (`test_changing_only_allocation_seed_leaves_the_view_unchanged`).
* Permuted opaque refs/order preserve semantic equivalence (same test as the
  verification-matrix permutation case above — the runtime and the view
  boundary share one permutation contract).
* Every name in `FORBIDDEN_FIELD_NAMES` is rejected at both a shallow and a
  deeply nested position
  (`test_forbidden_field_is_rejected_at_every_nesting_depth`).
* Stale tables cannot produce a view (see verification matrix above).

## Stop-rule disposition

The stop rule fires if useful action semantics require successor or target
features. They do not: every allowed field above is either a current-state
descriptor fact already computed by the DSH3-01/03/06 contracts, or a
row-local join that permutes with its row and never resolves to an opaque
string. No field here depends on what an action *would* produce. Learned
DSH3 M6/M7 work is **not** stopped; it proceeds against this boundary.

## Scope notes (deliberately deferred)

* Runtime symbols (`RuntimeSymbolDescriptorV1`) are not modeled in this view
  at all yet — no DSH3 M6/M7 issue has asked for symbol-scope features as
  model input. Adding them is a follow-up extension to
  `ReferenceModelViewV1`, not a gap in this contract's own claims.
* `ordinary_action_count` is a bare count, not a per-token feature — routing
  between operator and ordinary DSL actions is out of scope for this
  boundary issue.
* No training, scorer, or head consumes `OperatorPolicyInputV1` yet; this
  issue delivers the contract and builder only, per its own non-goals.

These are compiler-contract/unit fixtures. No train, eval, benchmark,
matrix, checkpoint, model-card, ship-gate, or model-quality claim is
produced.
