# DSH5-07 branch sequence merge: N-step conservative merge over verified sequences

SLM-415 (DSH5-07) is milestone M3's decision for "DSH5 — Bulk Operators,
Transactions & Control Plane": the one-step conservative branch merge
(`merge.py`, SLM-377/DSH3-09, extended by SLM-394/DSH3-19's effect-derived
conflict classifier and SLM-395/DSH3-20's fresh reference-table
continuation) merges exactly *one* verified edit per branch. This issue
extends that bound from one step to N: `src/slm_training/dsl/operators/sequence_merge.py`
merges two verified multi-step `BranchApplicationSequenceV1` histories, or
returns a typed conflict, reusing every existing merge primitive unchanged.

## Decision and hypothesis

Which multi-step branch histories can merge deterministically, and which
must return typed conflicts? The hypothesis: collapsing each verified
branch sequence into an exact base-relative composite effect/read-write
set, then composing only disjoint or explicitly commuting changes, safely
extends the existing one-step merge while preserving full lineage and
fresh-reference continuation.

**This hypothesis holds for the scope this issue actually implements** —
see "Coverage and what's excluded" below for the precise, deliberately
conservative boundary, and "Stop-rule disposition" for why that boundary
is not itself a stop-rule trigger.

## `BranchApplicationSequenceV1`

The direct N-step generalization of `merge.BranchEditV1`:

```python
@dataclass(frozen=True)
class BranchApplicationSequenceV1:
    nodes: tuple[ConversationStateNodeV1, ...]       # N + 1 entries
    applications: tuple[OperatorApplicationV1, ...]  # N entries
```

`nodes[i]` is the input state of `applications[i]`; `nodes[i + 1]` is its
output. `__post_init__` proves the exact same per-edge invariants
`BranchEditV1` already proves (parent linkage, one shared `branch_digest`,
`application.succeeded`, before/after digests matching the edge) for
*every* step in the chain — a length-1 sequence carries exactly the same
evidence as one `BranchEditV1`. `sequence_from_branch_edits` chains an
ordered list of already-individually-proven `BranchEditV1`s, additionally
proving only the chain property (edit *i*'s output is edit *i + 1*'s
input) — the collapse step the issue's Implementation section describes
("Replay each branch from the common base and collapse AST-changing turns
... History-only turns modify cursor/branch lineage but not AST effects"):
a caller walking a real `ConversationTraceV1` skips `UNDO`/`REDO`/
`CHECKOUT_STATE` turns (they move the cursor over existing nodes and are
never evidence a sequence needs), builds one `BranchEditV1` per `AST_EDIT`
turn, and passes the ordered result to `sequence_from_branch_edits`.

## Conflict classification: composite effect over content-addressed lineage

The one-step classifier, `merge.classify_merge_effects`, takes exactly one
`ActionEffectV1` and one base-relative lineage map per side. Reusing it
unchanged for a sequence requires collapsing a branch's whole chain into
one composite effect and one lineage map — `_sequence_composite` does
this:

1. **`_sequence_fingerprint_targets(base, sequence)`** builds a
   `descriptor.fingerprint -> base target` map from the sequence's
   *immediate post-fork* input node (`nodes[0]`) via the existing
   `merge._base_targets` — exactly the node the one-step path already uses
   this way. Descriptor fingerprints are content-addressed (independent of
   the *opaque ref* a table build allocates to them), and this repository's
   own reference-table lifecycle reallocates opaque refs on every new table
   build (visible directly in the merge test fixture: an edit's output
   table is rebuilt from the same descriptors with a new seed) while
   preserving descriptor fingerprints for unchanged content. That is what
   makes this map still valid many steps later, against tables whose opaque
   refs have long since changed.
2. **`_resolve_sequence_chain(ref, node, fingerprint_targets)`** walks
   `ref`'s `parent_fingerprint` chain within `node`'s own table, mirroring
   `merge._base_target_lineage`'s all-or-nothing chain (every hop must
   resolve or the whole chain is discarded) but resolving each hop by
   content-addressed fingerprint instead of by-ref-vs-base comparison. It
   returns one of three outcomes, and the distinction between the last two
   is the crux of this issue's precision requirement:
   * **`None`** — `ref` is not registered in `node`'s own table at all.
     Genuinely stale or forged. The caller keeps it *in* the composite
     effect but *out* of lineage, so `classify_merge_effects`'s own
     existing "not fresh" check fires `STALE_REF` — exactly the one-step
     path's behavior for an unbound ref, unmodified.
   * **`()`** (empty) — `ref` exists in `node`'s table, but its chain never
     reaches a target that existed in `base`. This is content this
     branch's own sequence created (a node produced by an earlier step and
     only ever touched by a later step in the *same* branch, for example).
     The caller drops it from the composite effect entirely: such content
     cannot overlap the other branch, which forked from the same base and
     has no way to reference it, so silently including it would either
     force a spurious `STALE_REF` (wrong) or require inventing conflict
     semantics for content the other branch cannot possibly know about
     (also wrong). Dropping it is the only sound choice, not an
     optimistic shortcut.
   * **Non-empty tuple** — resolved; used exactly as the one-step path
     already uses a lineage chain (`chain[0]` as the direct target,
     `_effect_target_closure`'s union of the whole chain).
3. Every step's own effect must be present and `CompilerCoverage.EXACT`
   (mirroring the one-step gate) — `_sequence_composite` returns `None`
   (treated as `UNSUPPORTED_EFFECT`) otherwise, so one inexact step anywhere
   in either sequence makes the whole merge attempt refuse, not just that
   step's own contribution.
4. `merge_branch_application_sequences` additionally re-validates, **per
   step**, that the step's own `AstOperatorV1.effect_signature` covers the
   delta kinds its own effect actually used (`merge._effect_matches_declaration`,
   reused unmodified) *before* composing — this is strictly more precise
   than the one-step check (per-step, not once for the whole composite),
   and it is why the composite effect can safely be paired with a
   permissive internal stand-in declaration (`_PERMISSIVE_SEQUENCE_DECLARATION`,
   `effect_signature = every EffectDeltaKind`) when calling
   `classify_merge_effects`: that call's own signature check becomes a
   redundant pass-through, because the real check already ran per step.

The composite effect and lineage, once built for both sides, feed
`classify_merge_effects` **exactly as-is** — no change to that function,
and every existing `test_operator_merge.py` conflict-kind test continues
covering the underlying classification taxonomy (`SAME_NODE_INCOMPATIBLE_EDIT`,
`DELETE_MODIFY`, `ROLE_CARDINALITY`, `CHILD_ORDER`, `SCOPE_BINDER`) that
`_sequence_composite` merely *feeds into* rather than re-implements.

### One deliberate narrowing: no commuting override for composites

The one-step path lets a real `classify_merge_effects` overlap through if
both declarations mutually name each other in `commutes_with`
(`merge.py:711-716`). `merge_branch_application_sequences` does not
replicate this for a composite sequence: `commutes_with` is a precise
claim about *one pair of single operators*, and a composite sequence has
no single declaration to make that claim about (two different operators in
one branch's sequence could each declare different, unrelated commuting
partners). Treating any per-step commuting declaration as licensing the
*whole composite's* overlap would be a real weakening, not a
generalization. Any composite-level overlap is therefore final — strictly
more conservative than the one-step bound, consistent with "Unknown
overlap is a typed conflict, never optimistic merge" applying at least as
strongly to sequences as to single steps.

## AST composition and continuation

Once composite classification passes (no conflict, or a mutually-safe
disjoint footprint), the merge itself reuses `ast_merge.merge_ast_value`
unchanged — a 3-way structural merge over `(base_ast, left_output_ast,
right_output_ast)` that only needs each side's *final* AST, regardless of
how many steps produced it. `BranchSequenceMergeArtifactV1`/
`BranchSequenceMergeConflictV1`/`BranchSequenceMergeDecisionV1` mirror
their one-step counterparts field-for-field (canonically-sorted, arbitrary-
length `branch_state_ids`/`application_ids` tuples in place of the
one-step's fixed 2-tuples), and the fresh-reference continuation path is
*literally* the same `merge.BranchMergeContinuationV1` type, unmodified —
its shape (`merge_id`, `allocation_seed`, `merged_node`,
`reference_table_fingerprint`) never depended on how many steps were
merged, only on the merge's own identity and a caller-supplied
`MergeReferenceTableBuilder`.

## Coverage and what's excluded

**Covered, proven by `tests/test_dsl/test_operator_sequence_merge.py`:**

* A 2-step sequence (disjoint target, then a second edit on the same
  branch) merges against a 1-step sequence on a disjoint target — order
  invariant, replayable, byte-identical merged source
  (`test_disjoint_two_step_and_one_step_sequences_merge_and_are_order_invariant`).
* A conflict on the *second* step of a sequence (step 1 safe, step 2
  overlaps the other branch) is still detected — proves whole-sequence
  scanning, not just first/last-step scanning
  (`test_conflict_buried_in_a_later_step_is_still_detected`).
* The same "buried conflict" shape for `DELETE_MODIFY` specifically, since
  delete/modify classification depends on the *closure* union across the
  whole sequence, not a single step's own effect
  (`test_delete_modify_conflict_detected_when_delete_step_is_not_first`).
* Inexact (`CompilerCoverage.APPROXIMATE`) coverage on *any* step —
  anywhere in the chain, not just the first or last — refuses the whole
  merge as `UNSUPPORTED_EFFECT`
  (`test_inexact_coverage_anywhere_in_sequence_is_unsupported`).
* A genuinely stale/forged ref on a later step refuses as `STALE_REF`
  without touching the base — this is the test that caught the precision
  gap described above (an early implementation silently *dropped* a stale
  ref instead of surfacing it, since both "stale" and "legitimately
  intra-branch-only" looked identical before `_resolve_sequence_chain`
  was split into the three-outcome form)
  (`test_stale_ref_in_a_later_step_refuses_without_mutation`).
* Fresh-reference continuation after a successful sequence merge: old
  branch-local refs from either input sequence are rejected
  (`ref.stale_state`/`ref.cross_branch`) against the merged node, and one
  further real operator turn applies and replays cleanly from the merged
  continuation node through `replay_conversation_trace`
  (`test_sequence_merge_continuation_rebuilds_fresh_table_and_replays_followup_trace`).
* Conflict identity is deterministic, canonical, and provenance-complete,
  and `replay_branch_sequence_merge` reproduces the recorded decision
  exactly, order-swapped
  (`test_sequence_conflict_identity_is_deterministic_and_provenance_complete`).
* `BranchApplicationSequenceV1`/`sequence_from_branch_edits` reject an
  empty sequence and a broken chain (`test_branch_application_sequence_requires_chained_nonempty_edits`).

**Deliberately excluded (not a stop-rule trigger — see below):**

* **`TRANSACTION_COMMIT` turns are not representable in a sequence.**
  `BranchApplicationSequenceV1` only chains `AST_EDIT`-shaped
  `OperatorApplicationV1` steps. A branch history containing a
  `TRANSACTION_COMMIT` turn cannot be collapsed into one today; a caller
  walking a real trace with such a turn present cannot build a sequence
  across it. Composing a transaction's own already-composite effect into a
  *further* composite effect is a real, separate design question (nested
  composition, doubled read/write-set semantics) this issue does not
  answer speculatively.
* **`FORK`/`COPY_STATE` turns occurring after a branch's starting point**
  are equally unrepresentable — a sequence's `nodes` chain assumes every
  step is a real AST mutation on one fixed branch digest; a mid-sequence
  fork or state copy has no effect to compose and would need its own
  distinct evidence shape.
* **No general producer/consumer-across-branches resolution beyond
  fingerprint matching.** `_resolve_sequence_chain`'s content-addressing
  approach is sound for the common case (a ref's descriptor fingerprint
  is stable across a table rebuild when its underlying content is
  unchanged) but is not a claim that *every* possible pack-backend
  reference-table evolution strategy is covered — it is proven against
  this repository's actual `build_reference_table`/`clone_reference_table_for_branch`
  behavior (exercised directly by the fixture in
  `test_operator_sequence_merge.py`), not against a hypothetical one.

### Why fingerprint-based lineage instead of extending `merge._base_targets`

An earlier design considered generalizing `merge._base_targets`/
`_base_target_lineage` themselves to tolerate a same-branch node whose
table has diverged from the immediate post-fork clone (i.e. reuse the
*existing* function for every step, not just the first). That function's
strict whole-table-set-equality gate (`{entries} != {expected}` — an
all-or-nothing check, not a per-entry one) is deliberate, load-bearing
conservatism shared by the one-step merge path today; relaxing it would be
a change to *already-shipped, security-relevant* correctness logic with
no independent way in this issue's scope to prove no regression against
every existing one-step scenario beyond "the existing suite still passes."
Adding a new, sequence-specific resolution function
(`_resolve_sequence_chain`) that the one-step path never calls is strictly
lower-risk: `merge.py`'s own functions and their existing test coverage
are completely untouched by this issue (confirmed: `test_operator_merge.py`'s
full suite passes unmodified after this change).

## Matrix and controls

Covered in `tests/test_dsl/test_operator_sequence_merge.py`, described
above. The underlying per-conflict-kind classification matrix
(`SAME_NODE_INCOMPATIBLE_EDIT`/`ROLE_CARDINALITY`/`CHILD_ORDER`/
`SCOPE_BINDER`/`DELETE_MODIFY`) is exhaustively proven once, for the
shared `classify_merge_effects` primitive itself, by the unchanged
`test_operator_merge.py::test_overlapping_effects_return_specific_typed_conflicts`
suite — this issue proves only that composite-sequence effects correctly
*feed* that classifier (one property-category case plus one buried-conflict
and one delete/modify case is sufficient evidence for that propagation
claim, since the propagation code path from composite effect to conflict
kind is identical regardless of which kind fires, exactly the same
argument SLM-413/DSH5-05's design doc makes for its own structural safety
net).

## Acceptance

* **Every successful merge validates and replays through pack authority** —
  every merged state is produced by `pack.backend.serialize` +
  `OperatorStateV1.from_source` (the same full parse/canonicalize/oracle/
  scope/property-order/round-trip pipeline every other operator-produced
  state uses), and `replay_branch_sequence_merge` reproduces the recorded
  `decision_id` exactly from base + both sequences alone.
* **Disjoint supported branches are input-order invariant** — proved
  directly (`decision.decision_id == reversed_decision.decision_id`).
* **Unknown or overlapping unsupported effects produce typed conflicts
  with exact targets** — every conflict test asserts both `kind` and
  `target_fingerprints`.
* **Continuation uses only the fresh merged reference table** — old
  branch-local refs from either input sequence provably fail
  (`ref.stale_state`/`ref.cross_branch`) against the merged continuation
  node.
* **Negative, conflict, unsupported, and inconclusive cases remain in
  evidence** — every typed conflict (`STALE_REF`, `UNSUPPORTED_EFFECT`,
  `SAME_NODE_INCOMPATIBLE_EDIT`, `DELETE_MODIFY`,
  `REFERENCE_REBUILD_FAILED`) is a first-class, reproducible
  `BranchSequenceMergeDecisionV1`, never an exception or a silently
  dropped case.

## Stop-rule disposition

The issue's stop rule fires "if sequence collapse loses the precision
required for exact effect/read-write attribution." It does not fire here,
for the scope this issue actually implements: `_sequence_composite`'s
three-outcome resolution (stale/forged vs. legitimately-intra-branch vs.
resolved) preserves exactly the precision the one-step path already
guarantees for every ref it can classify, and the one case that could have
silently lost precision — a genuinely stale ref being mistaken for
legitimate intra-branch content — was caught by
`test_stale_ref_in_a_later_step_refuses_without_mutation` during
development and fixed before this issue shipped (see "Coverage" above).
The honest boundary this issue draws is narrower than "arbitrary branch
histories": `TRANSACTION_COMMIT`/mid-sequence `FORK`/`COPY_STATE` turns are
simply not representable as a `BranchApplicationSequenceV1` today, which
is the conservative "preserve the existing one-step merge bound... as
unsupported" outcome the stop rule itself names as acceptable, applied to
the specific sub-cases this issue does not attempt rather than to sequence
merge as a whole.

## Scope notes (deliberately deferred)

* **A generic trace-walking collapse function** (`ConversationTraceV1` in,
  `BranchApplicationSequenceV1` out, transparently skipping history-only
  turns and rejecting on an unsupported turn type) is not shipped in this
  issue. `sequence_from_branch_edits` proves the chain-collapsing
  invariant the issue's Implementation section describes; a caller
  building the ordered `BranchEditV1` list from a real trace (skipping
  `UNDO`/`REDO`/`CHECKOUT_STATE`, one `BranchEditV1` per `AST_EDIT` turn)
  is straightforward with existing `conversation.py` primitives
  (`TurnArtifactV1.operation`, `.input_state_id`/`.output_state_id`,
  `.application`) and is left to the caller rather than speculatively
  generalized here before a real caller exists.
* **Composing `TRANSACTION_COMMIT` turns into a sequence.** Flagged above
  as excluded; a follow-on issue would need to decide whether a
  transaction's own `composite_effect` (already built by DSH5-04/DSH5-05)
  can be folded into a sequence's composite effect directly, or whether it
  needs its own lineage-resolution pass analogous to
  `_sequence_fingerprint_targets`.
* **Associativity/symmetry beyond order-invariance of the two top-level
  sequences.** The issue asks these be "measured per supported subset,
  never assumed globally." This issue measures exactly one such property
  (swapping `left`/`right` reaches the same `decision_id`) and makes no
  claim about, for example, splitting one sequence into two smaller
  sequences and merging them independently before merging the result.

This work is compiler contract/unit fixtures exercised through the real
`openui` `DslPack` (parse/canonicalize/schema-oracle/round-trip authority
is the genuine pack backend, not a stub). No train, eval, benchmark,
matrix, checkpoint, model-card, ship-gate, or model-quality claim is
produced.
