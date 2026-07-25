# DSH5-02 exact atomic bulk set-property operator over SelectorRefV1

SLM-410 (DSH5-02) is milestone M1's second issue for "DSH5 — Bulk Operators,
Transactions & Control Plane": "Exact selectors and one bulk operator." DSH5-01
(SLM-409) shipped `SelectorRef`/`SelectorKind`/`SelectorDescriptorV1`/
`build_selector`/`SelectorContextV1` but explicitly shipped no operator that
consumed a resolved selector — that was left as "the next M1 issue." This
issue is that next issue: `openui.map_set_property(selector, role, value)`, one
exact atomic bulk operator that sets a scalar property uniformly across every
node a `SelectorRef` names.

## Contract

* New operator id `openui.map_set_property` (`MAP_SET_PROPERTY`,
  `src/slm_training/dsl/operators/bulk.py`), following the `openui.<snake_case>`
  naming convention of `openui.set_property` / `openui.add_child` / etc.
* Argument slots (`AstOperatorV1.argument_slots`):
  * `selector` — `RefKind.SELECTOR`, `BindingPhase.STATE`. The already-built
    `SelectorRef` naming the exact target node set.
  * `role` — `RefKind.ROLE`, `BindingPhase.STATE`. A node-scoped `RoleRef`
    whose `property_name` names the property to set on *every* selected
    node — not necessarily one belonging to a selector target itself; only
    its `property_name` is used (see "Execution" below for why a per-target
    `RoleRef` still has to be looked up separately).
  * `value` — `RefKind.VALUE`, `BindingPhase.APPLICATION`, matching
    `openui.set_property`'s own `value` slot exactly.
* One precondition, `PreconditionV1("schema.bulk_property", ("selector",
  "role", "value"))`.
* `effect_signature = (EffectDeltaKind.PROPERTY,)`.
* `compiler_coverage = CompilerCoverage.EXACT` unconditionally on every
  success path — there is no partial/approximate/bounded success for this
  operator. If exactness cannot be proven for some target, the whole
  application rejects instead (see "Execution / atomicity").
* No gap in `OperatorArgumentSlotV1` / `AstOperatorV1.validate_arguments` /
  `RefKind.SELECTOR` blocked this: a `SELECTOR`-kind argument slot validates
  and binds exactly like any other opaque-ref kind (`argument.value.KIND is
  RefKind.SELECTOR` matches the declared slot's `ref_kind`), confirming
  DSH5-01's own expectation that the contract layer already supported this
  without change.

## Execution / atomicity

`_executor` in `bulk.py` runs against the same
`OpenUILocalOperatorContextV1` DSH3-04/DSH5-01's own `local.py` operators use
— **no new context type was introduced**, because a selector already lives
inside `context.reference_table.selectors`; there is nothing else a bulk
property-set operator needs that `local.py`'s existing context does not
already expose (`references`, `payload`, `resolve`, `schema_defs`).

Steps, in order:

1. **Resolve the selector.** The bound `SelectorRef` is looked up in
   `context.reference_table.selectors`, then resolved through
   `ReferenceTableV1.resolve_selector` — never a hand-rolled membership
   lookup — and its exact target set is turned back into `NodeRef`s via
   `SelectorContextV1.resolve_members`, exactly the mechanism DSH5-01 built
   for this purpose. Every member is checked to be a `NodeRef` (rejecting
   `ref.type_incompatible` otherwise) — selectors in this repo have so far
   only ever been built over node entries, but nothing in the selector
   contract itself guarantees that, so this operator checks rather than
   assumes it.
2. **Zero targets rejects.** An empty selector is rejected with
   `bulk.no_targets` rather than treated as a vacuous success. This is a
   deliberate choice, not the only defensible one — see "Stop-rule
   disposition" below for why it was picked over the vacuous-success
   alternative the ticket also allowed.
3. **Extract the property name.** The bound `role` `RoleRef` is resolved
   through `context.resolve(..., RefKind.ROLE)` to get its
   `RoleLocationV1.property_name`. This does *not* require the role to
   belong to any selector target — `RoleRef` is inherently node-scoped (one
   instance per node per schema property; see `_resolve_role`'s ownership
   check in `local.py`), so a single `role` argument can never simultaneously
   *be* the correct per-node `RoleRef` for every target. Only its
   `property_name` is portable across targets.
4. **Validation pass (no mutation).** For every target, in the order
   `resolve_members` returns them (already canonical — see "Replay /
   order-invariance"):
   * Resolve the node (`_resolve_node`, reused unchanged from `local.py`).
   * Look up *that target's own* `RoleRef` for the extracted property name
     by scanning `context.references(RefKind.ROLE)` for the entry whose
     `RoleLocationV1.node_fingerprint` / `property_name` match — every node's
     every schema-declared property already has its own `RoleRef` entry from
     `build_openui_local_operator_context`, so this is a lookup over
     already-committed table entries, never a freshly minted reference. No
     match (the target's component simply has no such property) rejects
     `bulk.incompatible_role_member`.
   * Validate the new value against that property's schema
     (`_property_schema` + `_matches_scalar_schema`, both reused unchanged
     from `local.py`). A schema mismatch rejects `bulk.incompatible_role_member`.
   * If the target already holds exactly the new value, reject
     `bulk.no_change_member` — building an `EffectDeltaV1` with an unchanged
     value is structurally impossible anyway
     (`EffectDeltaV1.__post_init__` raises on `before == after`), so this
     mirrors `openui.set_property`'s own `local.no_change` precedent rather
     than silently dropping that target's delta and violating the "one delta
     per target" contract below.
   * Check the same positional-property-hole invariant
     `openui.set_property` checks (`_ordered_properties` /
     `_required_property`) before allowing the value to land.
   * **Nothing in `bindings` is touched during this pass.** Every read uses
     `node.get("props")`, never `node.setdefault(...)`, specifically so a
     later target's rejection cannot leave an earlier target's props
     partially initialized.
5. **Apply pass — only reached once every target above has passed.** Each
   validated target's `props[property_name]` is set in canonical (member)
   order. Because this pass only starts after full validation, one
   `execute()` call is genuinely all-or-nothing: either every target is
   rewritten, or the function raises before touching any node and
   `OperatorLibraryV1._execute` returns a rejection with `state=None`.
6. **One `ActionEffectV1`, one delta per target.** `property_deltas` holds
   exactly one `EffectDeltaV1` per target, matching `openui.set_property`'s
   own convention of keying a property delta on the `RoleRef` (not the
   `NodeRef`) — necessary here because the same conceptual property resolves
   to a *different* `RoleRef` per target node (`RoleRef` is node-scoped, so
   there is no single ref that could stand in for "this property on every
   target"). All deltas share the request id the selector, role, and value
   arguments already share, so `ActionEffectV1.__post_init__`'s
   single-request-id check passes without change, and `merge.py`'s
   `_refs(effect)` extraction (which reads `delta.target` off
   `property_deltas` generically) needed no change to keep working.

Every rejection code introduced follows the existing `<module>.<reason>`
convention (`local.no_change`, `local.unsupported_role`, ...):
`bulk.no_targets`, `bulk.incompatible_role_member`, `bulk.no_change_member`.

## Replay / order-invariance

`resolve_members` already returns targets in an order independent of the
order `matching_refs` were supplied when the selector was built — it walks
`descriptor.target_fingerprints`, which `SelectorDescriptorV1.__post_init__`
requires to be sorted, and that descriptor is itself content-addressed
(`build_selector_descriptor` sorts by descriptor fingerprint before hashing
the semantic fingerprint). So two selectors built from the same node set in
different orders produce byte-identical descriptors, hence identical member
order, hence identical `property_deltas` sequences and effect fingerprints —
verified directly by
`test_selector_construction_order_carries_no_semantics_for_the_effect`
(mirroring DSH5-01's own
`test_target_order_carries_no_semantics`). Nothing in this operator's own
code iterates a Python `set` or unordered `dict` view to decide target order.

Because `execute()` is a pure function of `(state, arguments)` closed only
over the already-fixed `OpenUILocalOperatorContextV1`,
`OperatorLibraryV1.replay()` reproduces the identical `application_id` when
re-run against the same `before_state_digest` —
`test_replay_reproduces_the_same_application_identity` applies once, replays,
and asserts identical `application_id` and resulting state.

## Stop-rule disposition

The ticket's stop rule: *"if atomic replay/exact attribution can't be
preserved, keep selectors diagnostic-only."* It did not fire. Every target
this operator touches is validated before any mutation, the resulting effect
is provably `CompilerCoverage.EXACT` (never a downgraded coverage value — the
operator only ever succeeds with `EXACT` or rejects), and replay reproduces
byte-identical evidence. Bulk operator work proceeds; scope was narrowed in
specific, documented ways below rather than the whole class of bulk operators
being deferred.

**Zero-target decision.** The ticket left this genuinely open ("pick one, be
consistent, and document"). This change rejects (`bulk.no_targets`) rather
than treats an empty selector as vacuously successful, for one concrete
reason: `_mutation()` (`local.py`, reused unchanged) already enforces
project-wide that an operator's canonical rewrite must be observably
different from the input source — every existing local/topology operator
that could produce a no-op (`openui.set_property`, `openui.reorder_children`)
rejects explicitly rather than relying on that generic check. A zero-target
bulk apply produces zero property deltas and an unchanged canonical source by
construction, so treating it as "vacuously successful" would mean returning
`OperatorApplyResultV1.succeeded == True` with a state identical to the input
and an `ActionEffectV1` claiming `EXACT` coverage over nothing — a success
report with no observable effect anywhere the compiler could check. Rejecting
it explicitly, before touching any node, keeps the "a successful application
changes something" invariant this repository already enforces everywhere
else, rather than being the one operator that quietly breaks it.

## Scope notes (deliberately deferred)

* **No OpenUI-pack scope/role extraction is wired here.** Deciding *which*
  nodes satisfy a `SelectorKind` predicate (e.g. "every `Stack` in this
  scope") stays exactly where DSH5-01 left it: the caller's job, using
  `build_selector_descriptor`'s `matching_refs` or
  `build_selector_from_scope`'s `predicate` callable. This operator only ever
  consumes an already-committed `SelectorRef` — it does not walk the AST to
  decide selector membership itself. Test fixtures build selectors directly
  with `build_selector_descriptor` + the new `attach_bulk_selector` helper,
  the same way DSH5-01's own tests do, not through any OpenUI-pack-specific
  discovery.
* **`resolve_selector`'s "fresh recompute" is self-referential, not
  independently re-derived.** `ReferenceTableV1.resolve_selector` is designed
  to check a selector's committed scope/membership against *freshly
  recomputed* compiler facts. Re-deriving those facts from live OpenUI AST
  semantics (e.g. re-walking the tree to confirm which nodes currently
  satisfy `COMPONENT_TYPE_IN_SCOPE`) is exactly the OpenUI-pack scope/role
  extraction machinery called out above as out of scope. This operator
  instead passes the selector's *own* stored `scope_fingerprint` and
  `target_fingerprints` (read off the same table entry) as the "current"
  values. This still exercises every state/branch/request/kind/fanout/
  duplicate integrity check `resolve_selector` performs — including the
  practically meaningful one, that the `state`/`branch_digest` passed to
  `execute()` at call time actually match the table the context was built
  for — but it does not independently prove the selector's membership is
  still *semantically* current against a possibly-since-mutated AST beyond
  what `state_digest` equality already guarantees. A caller that wants that
  stronger guarantee across an actual AST mutation in between selector build
  and bulk apply needs the (still deferred) pack-level scope re-derivation.
* **Registration is at the `OperatorLibraryV1` composition level, not
  `DslPack`'s own default construction.** `build_openui_bulk_operator_library`
  composes `openui_local_registered_operators(context)` with the new
  `MAP_SET_PROPERTY` entry, exactly the pattern `topology.py`'s
  `build_openui_topology_operator_library` already established for
  `DUPLICATE_SUBTREE` / `MOVE_NODE` / etc. Tests attach it the same way the
  existing local/topology test suites do
  (`replace(base_pack, operator_library=library)`) — this is the
  established integration boundary in this repository, not a narrower path
  invented for this ticket.
* **Only one bulk operator.** Ticket scope is exactly
  `openui.map_set_property`; no bulk unset, bulk topology, or transactional
  multi-operator envelope is introduced here — those are separate later M1
  issues.

These are compiler contract/unit fixtures exercised through a real
`OpenUILocalOperatorContextV1` + real `openui` `DslPack` + real
`OperatorLibraryV1`. No train, eval, benchmark, matrix, checkpoint,
model-card, ship-gate, or model-quality claim is produced.
