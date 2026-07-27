# DSH5-02 atomic bulk set-property operator over SelectorRefV1

SLM-410 (DSH5-02) is milestone M1's second decision for "DSH5 — Bulk
Operators, Transactions & Control Plane": can the existing operator registry
safely express one decision that mutates multiple verified AST nodes
atomically? It is the first consumer of the DSH5-01 `SelectorRefV1` contract
and extends the DSH3-02 registry, DSH3-03 reference tables, and DSH3-04 local
operators — it does not introduce a parallel execution path, and it declares
exactly one new operator, `openui.map_set_property(selector, role, value)`.

## Contract

* `MAP_SET_PROPERTY = "openui.map_set_property"`
  (`src/slm_training/dsl/operators/bulk.py`) is declared in the AST operator
  registry like every other compiler-owned operator (`AstOperatorV1`,
  `argument_slots`, `preconditions`, `effect_signature`) — never as a
  tokenizer special token. Its three argument slots are:
  * `selector` — `RefKind.SELECTOR`, `BindingPhase.STATE`: the exact, finite,
    pack-authorized target set (DSH5-01).
  * `role` — `RefKind.ROLE`, `BindingPhase.STATE`: one anchor role whose
    owner node must be a member of the selection; its `property_name` is
    projected onto every other selected node.
  * `value` — `RefKind.VALUE`, `BindingPhase.APPLICATION`: the literal to
    apply, validated against **every** target's own component schema.
* `effect_signature = (EffectDeltaKind.PROPERTY,)`, `locality =
  "selector.property"`, `idempotent = True`, matching the primitive
  `openui.set_property` this operator generalizes.
* `openui_bulk_registered_operators` / `build_openui_bulk_operator_library`
  compose the bulk operator with every DSH3-04 local primitive
  (`openui_local_registered_operators`) into one `OperatorLibraryV1` — the
  same composition pattern DSH3-05's topology module already uses, not a new
  one.
* No new `OpenUI`-pack-specific context type was needed:
  `OpenUILocalOperatorContextV1.reference_table` already carries a
  `selectors` field (DSH5-01), so the bulk executor consumes the *same*
  context object DSH3-04's primitives use. The only new machinery is one
  method, `OpenUILocalOperatorContextV1.resolve_selector` (`local.py`), plus
  two thin composition helpers in `bulk.py`:
  * `build_bulk_selector` / `attach_bulk_selector` — wire an existing
    `build_selector` / `attach_selector` (DSH5-01) call through a local
    operator context's own reference table; they add no new selector
    semantics.

## Enumeration and resolution

`resolve_selector` (added to `OpenUILocalOperatorContextV1`) is the DSH5-02
consumer of `ReferenceTableV1.resolve_selector`:

1. Reject a non-`SelectorRef` argument (`selector.type_incompatible`),
   an unknown opaque id (`selector.missing`), or an ambiguous one
   (`selector.duplicate`).
2. Look up the selector's own committed `SelectorKind` / scope / target set
   from the table (never guessed, never re-derived from an OpenUI-specific
   predicate — that extraction layer remains out of scope, per DSH5-01's own
   scope notes) and re-confirm it against the caller's live `state_digest`
   and `branch_digest` through the existing DSH5-01 freshness/cardinality
   checks (`selector.stale_state`, `selector.cross_branch`,
   `selector.fanout_overflow`, `selector.target_set_changed`,
   `selector.duplicate_target`, …). No new failure code is introduced at
   this layer — DSH5-02 inherits the DSH5-01 guarantee rather than
   re-implementing it.
3. Return the descriptor plus its exact member refs
   (`SelectorContextV1.resolve_members`), a fresh, independent tuple.

The `openui.map_set_property` executor (`bulk.py`) then, over that member
set:

1. Rejects an **empty** selector (`local.empty_selector`) — no explicit
   no-op policy is exposed by this operator's three-argument signature, so
   the ticket's "reject empty selectors unless an explicit no-op policy
   permits" resolves to: always reject. Adding an opt-in no-op policy is
   left to a future issue if a real use case needs it (see Scope notes).
2. Rejects a selector whose members are not all `RefKind.NODE`
   (`local.selector_not_node_kind`) — `SelectorDescriptorV1` itself is
   ref-kind-agnostic (DSH5-01), so this operator enforces its own domain.
3. Enforces `MAP_SET_PROPERTY_MAX_FANOUT = 8` independently of whatever
   `max_fanout` the selector itself was built with
   (`local.selector_fanout_exceeded`) — an operator-level policy, not a
   restatement of the selector's own bound. 8 is chosen to match the test
   matrix's largest fanout point, so the boundary is exercised directly.
4. Resolves the anchor `role` and requires its owner node to be a member of
   the selection (`local.role_not_selected`) — keeps the whole action
   self-contained to the exact selector membership rather than reaching for
   a role belonging to an unrelated, unselected node.
5. For every target, in the selector's own canonical (fingerprint-sorted,
   permutation-invariant) member order: looks up the node's own
   `_property_schema` for the anchor's `property_name`
   (`local.unsupported_role` if the node's component type never declared
   it — "every selected node exposes one compatible role"), validates the
   value against that node's own schema (`local.property_value_invalid` —
   "the value satisfies every target schema"), rejects an already-satisfied
   target (`local.no_change` — before == after is not a representable
   `EffectDeltaV1`, so an already-satisfied member is treated the same way
   the single-node `openui.set_property` primitive already treats a no-op:
   as a rejection, not a silent skip), and re-checks the existing positional
   canonical-property-order invariant (`canonical.positional_property`).
6. Only once every target validates does it return one mutation: all
   property writes plus one `ActionEffectV1` whose `property_deltas` name
   each target's own `RoleRef` (never the anchor), `compiler_coverage =
   CompilerCoverage.EXACT`, and `estimated_completion_cost =
   float(target_count)`.

## Atomicity proof

No target update is ever partially committed. This is not bespoke bulk-op
logic — it falls directly out of the existing `OperatorLibraryV1._execute`
contract (`registry.py`, unchanged by this issue):

* `state: OperatorStateV1` is a frozen dataclass; nothing in the executor can
  mutate it.
* `bindings = parse_statement_bindings(state.source, ...)` is a **scratch
  local dict**, freshly parsed on every call. The executor mutates only this
  local structure.
* A committed mutation is only ever returned via `_mutation(bindings,
  effect, state.source)`, called once, at the very end of the executor, after
  every target has already validated.
* Any exception raised at any point before that call — a schema mismatch on
  target 3 of 8, say — propagates out of `entry.execute(...)` in
  `_execute`. The registry's own exception handling converts it into a typed
  `OperatorRejectionV1` built from the **unmodified** `state`; the scratch
  `bindings` dict (however far its in-place mutation got) is simply
  discarded. `state.source` was never touched, so "no target update is ever
  partially committed" holds regardless of *where* in the per-target loop
  the rejection occurs, not because of any ordering trick in `bulk.py`.

Fixture: `test_incompatible_selected_node_missing_role_rejects_atomically`
and `test_incompatible_selected_node_value_fails_target_schema_rejects_atomically`
(`tests/test_dsl/test_bulk_operators.py`) assert `result.state is None`,
`result.application.after_state_digest is None`, and that `state.source` is
byte-identical to the original input after a rejected multi-target action.

## Replay and identity

`OperatorLibraryV1.replay` (unchanged) already reconstructs identical state
and `application_id` for any registered operator whose executor is a pure
function of `(state, arguments)` — which `openui.map_set_property` is (no
randomness, no wall-clock, no hidden ordering dependency).
`test_replay_reconstructs_identical_state_and_application_id` exercises this
directly for the bulk operator.

Target iteration order cannot affect identity: `SelectorDescriptorV1` always
stores `target_fingerprints` sorted by `ReferenceDescriptorV1.fingerprint`
regardless of the order `matching_refs` were supplied in when the selector
was built (DSH5-01 invariant), and `resolve_selector` returns members in that
same canonical order. `ActionEffectV1.to_dict()` additionally canonically
sorts every delta list before fingerprinting. Both are already-existing
invariants this issue relies on rather than re-derives;
`test_target_iteration_order_does_not_affect_application_identity`
(parametrized over all 6 permutations of a 3-target selection) proves the
resulting `application_id` and canonical `state.source` are identical no
matter which order `matching_refs` were passed in.

## Primitive-lowering equivalence (diagnostics only)

`lower_map_set_property_to_primitives` (`bulk.py`) applies the same edit as N
sequential `openui.set_property` primitive applications and is **never a
production dispatch path** — `openui.map_set_property` always commits in one
atomic pack-authorized application. It exists solely as an equivalence
oracle for tests.

A subtlety the lowering has to account for: `NodeRef` / `RoleRef` identity is
state-bound, and a property edit changes the *edited node's own* persistent
fingerprint (`persistent_node_fingerprint` folds a node's full canonical
structure, including its current `props`, into its identity). So a `NodeRef`
obtained from a context built against state *N* cannot be reused against
state *N+1* — this is by design (DSH3-03: references are single-state-scoped,
and the ticket's own "build a fresh post-commit reference table" requirement
is exactly this). The lowering therefore rebuilds a fresh
`OpenUILocalOperatorContextV1` before each sequential step and re-locates
each target by its **stable structural path** (`NodeLocationV1.path`), which
a property edit never moves.

`test_primitive_lowering_matches_bulk_atomic_result` (parametrized over
1/2/4 targets) asserts `lowered_state.source == bulk_result.state.source`
and `lowered_state.state_digest == bulk_result.state.state_digest` — bulk and
primitive lowering produce identical canonical state, satisfying the
acceptance criterion directly; no divergence was found.

## Fresh post-commit reference-table continuation

`test_post_commit_state_supports_fresh_reference_table_continuation` builds a
brand new `OpenUILocalOperatorContextV1` from a successful bulk application's
`result.state` (a fresh `request_id`, `branch_digest`, and `seed`) and
confirms it resolves and reflects the committed property values — the model
can keep issuing operators against the post-commit AST exactly as it would
after any primitive operator, with no bulk-specific continuation gap.

## Matrix and controls

Covered in `tests/test_dsl/test_bulk_operators.py`:

* 1 / 2 / 4 / 8 homogeneous targets, all succeeding atomically
  (`test_homogeneous_targets_update_every_node_atomically`, parametrized).
* Incompatible selected nodes: structurally missing role
  (`test_incompatible_selected_node_missing_role_rejects_atomically`, a
  `Stack` selection mixed with a bare `TextContent` leaf that has no
  `direction` property) and schema-value divergence on a shared role name
  (`test_incompatible_selected_node_value_fails_target_schema_rejects_atomically`,
  a `Card` + `Callout` selection sharing the `variant` role but disjoint
  enums).
* Already-satisfied subset (`test_already_satisfied_subset_rejects_as_no_change_not_partial_commit`)
  — one of three targets pre-set to the target value rejects the whole
  action as `local.no_change`, not a partial 2-of-3 commit.
* Nested scopes (`test_nested_scope_selectors_bound_mutation_to_exact_members`)
  — an outer 4-member selector and an inner 1-member selector coexist on one
  table; applying through the inner selector mutates exactly one node.
* Max fanout boundary at 8 (succeeds) and 9 (rejects)
  (`test_max_fanout_boundary_at_and_over_the_bound`), decoupled from the
  selector's own (larger) `max_fanout` to isolate the operator's own policy.
* Primitive-order / target-iteration permutations
  (`test_target_iteration_order_does_not_affect_application_identity`).
* Rejected by construction, never reaching the executor: duplicate targets
  (`test_duplicate_matching_refs_are_structurally_unrepresentable` —
  `SelectorDescriptorV1.__post_init__` already raises
  `selector.duplicate_target` before a selector with a repeated target can
  exist at all).
* Rejected at resolution: stale state, an unknown selector id, and a
  non-selector ref presented as one
  (`test_stale_state_and_unknown_selector_fail_closed_at_context_resolution`).
* Rejected at the operator's own domain boundary: an empty selector
  (`test_empty_selector_rejects_with_no_explicit_no_op_policy`), a selector
  over non-`NODE` refs (`test_selector_over_non_node_refs_rejects`), and an
  anchor role owned by a non-selected node
  (`test_role_anchor_must_belong_to_a_selected_node`).

## Acceptance

* Success updates every target; failure updates none — proved structurally
  (see Atomicity proof), not merely by example.
* Bulk and primitive lowering produce identical canonical state — proved by
  `test_primitive_lowering_matches_bulk_atomic_result`.
* Effects enumerate each target exactly once — `len(property_deltas) ==
  len(target_refs)` for every successful application (asserted in the
  homogeneous-count matrix test); selector membership uniqueness makes a
  duplicate target structurally impossible to reach the executor at all.
* Replay is order-invariant and reconstructs identical
  `application_id` / state — proved by the replay and permutation tests.
* Post-commit legal-set construction succeeds — a fresh
  `OpenUILocalOperatorContextV1` (the same machinery `enumerate_operator_legal_set`
  consumes) builds cleanly from the post-commit state.

## Stop-rule disposition

The stop rule fires if atomic replay and exact effect attribution cannot be
preserved. They are preserved: atomicity falls directly out of the existing,
unmodified `OperatorLibraryV1._execute` contract (no bespoke transaction
machinery was needed), effect attribution is exact (one `EffectDeltaV1` per
target, naming that target's own `RoleRef`, never the anchor or a shared
placeholder), and replay/identity are order-invariant by construction. Bulk
actions are **not** kept diagnostic-only; `openui.map_set_property` is a
fully registered, executable operator.

## Scope notes (deliberately deferred)

* **No explicit no-op policy for empty selectors.** The operator's
  three-argument signature (`selector, role, value`) has no slot for one;
  adding an opt-in "empty selection is a legal no-op" mode is left to a
  future issue if a real caller needs it. Today, an empty selector always
  rejects (`local.empty_selector`).
* **No OpenUI-pack-specific selector construction wiring.** Exactly as
  DSH5-01 scoped: `build_bulk_selector` takes an already-computed
  `matching_refs` tuple, not a real scope/component-type/role predicate over
  live OpenUI semantics. That extraction layer (deciding *which* table
  entries satisfy a given `SelectorKind` from real pack facts) remains a
  later issue; this one proves the *bulk-operator* half of the contract.
* **Selector domains are enumerated by the shared legal-set owner.** SLM-411
  extends `legal_set.py` to include the exact state-bound
  `reference_table.selectors` collection in both domain construction and
  semantic action identity. `openui.map_set_property` therefore enters a
  model-facing action surface only through a complete compiler-owned legal
  set; the selector descriptor's targets and opaque ID remain runtime-only.
* **No bulk unset/inverse operator.** `openui.map_set_property` declares no
  `inverse_operator_id`: inverting a bulk write would need each target's
  original per-node value, which the model does not supply as a single
  argument. A future bulk-unset or bulk-undo design is a separate decision.
* **Primitive lowering is diagnostic-only**, by design (per the ticket) —
  it is not exposed as an alternate execution path and is not registered in
  any operator library.

These are compiler contract/unit fixtures exercised through the real
`openui` `DslPack` (parse/canonicalize/schema-oracle/round-trip authority is
the genuine pack backend, not a stub). No train, eval, benchmark, matrix,
checkpoint, model-card, ship-gate, or model-quality claim is produced.
