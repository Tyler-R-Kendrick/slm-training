# DSH5-10: replay-grounded preference rows from undo/redo history (SLM-418)

**Status:** partial slice, in progress (fifth increment).
**Claim class:** `wiring`.
**Honest verdict:** not yet dispositioned -- this PR extends a scoped
subset, not the full issue.

SLM-418 asks whether exact undo, redo, checkout, and fork outcomes can
provide useful preference supervision for ambiguous follow-up instructions,
via: (1) a versioned preference-row schema over one exact input state, (2)
row extraction from seven verified conversation patterns, (3) matched
SFT/preference training against four context-view baselines, and (4)
held-out measurement of action/operator/argument/reference/branch accuracy,
calibration, and CAP0/CAP1/CAP2 retention.

This slice adds the seventh and final named pattern, **pronoun-focus-
followup** (see "Fifth slice (v6)" below), bringing extraction coverage to
7 of 7. All seven named patterns from the issue's own list now extract and
replay-verify. What remains is the issue's separate, still fully unattempted
training/measurement scope: SFT/preference training against the
DSH3-selected policy/control heads, the four-baseline comparison, held-out
benefit measurement, and turn-depth/context-view ablations.

**Update (sixth slice, below):** the turn-depth/context-view *structural*
ablation dimension and a bounded, fixture-scale matched context-view
comparison are now wired -- see "Sixth slice" for exactly what that does and
does not close. Real SFT/preference training against the DSH3-selected
policy/control heads, a powered/real corpus, and CAP0/CAP1/CAP2 retention
measurement remain unattempted.

## What this PR delivers

* `src/slm_training/dsl/operators/replay_preference.py`:
  * `OperatorReplayPreferenceRowV1` -- unchanged schema: one exact
    `input_state_id`, `chosen_action`/`rejected_action`, the resulting
    `chosen_output_state_id`, a typed `semantic_relation`, a
    `correction_reason`, and the `legal_set_fingerprint` the row was checked
    against.
  * `extract_replay_preference_rows` now scans a `ConversationTraceV1`'s
    turns for **five** of the issue's seven named patterns (one new since
    the second slice):
    * **edit-then-undo** and **undo-then-redo** (unchanged from v1/v2).
    * **partial-rollback** (new): a second, or later, *consecutive* `UNDO`
      turn -- i.e. the user keeps rolling back past the first undo instead
      of redoing, checking out elsewhere, or editing at that intermediate
      state. Distinct from `undo_then_redo` (which requires the *next* turn
      to be `REDO`) and from `edit_then_undo` (whose preceding turn must be
      an `AST_EDIT`, so it only ever fires for the *first* undo in a chain).
    * **checkout-another-state** (new): a `CHECKOUT_STATE` turn. Modeled as
      its own single-turn decision (no preceding-turn pairing needed, unlike
      the other three patterns) because choosing to `checkout` -- a distinct
      legal tool invocation from `undo`/`redo` -- over any other available
      action at that state is itself the preference signal, even when the
      checkout destination happens to coincide with what `undo` or `redo`
      would have reached. This pattern now only matches a **same-branch**
      checkout -- see `fork-then-choose-one-branch` below for the
      cross-branch case, which used to fall in here too.
    * **fork-then-choose-one-branch** (new): a `CHECKOUT_STATE` turn whose
      input or output state sits on a branch a recorded `FORK` turn opened
      (`fork_branch_digests`, the set of branch digests any `FORK` turn in
      the trace produced). Reuses the exact same `checkout:<state>` legal
      action as `checkout-another-state` -- the same
      `checkout_conversation_state` primitive -- but is classified
      separately because crossing a fork boundary (returning to the
      pre-fork branch, or moving between two sibling forks) is choosing
      between diverged branches, not merely relocating within one. `FORK`
      itself (the act of opening a branch) stays out of scope as a
      chosen/rejected candidate; only the *subsequent* branch-crossing
      checkout is modeled.
  * `_available_history_actions` now also enumerates
    `checkout:<state_id>` for every other state already materialized
    anywhere in the trace (ancestor, sibling, descendant, or cross-branch),
    per `checkout_conversation_state`'s actual authority (refuses only
    checkout-to-self). It is listed **alongside**, not instead of,
    `undo`/`redo:<child>` even at a shared destination, since those are
    distinct recorded turn operations and the choice between them is a real
    preference the issue asks for. (Unchanged this slice; the new
    classification lives entirely in `extract_replay_preference_rows`.)
  * `OperatorEventMemoryReportV1` -- counts of extracted rows by relation
    (unchanged schema; now counts five relations instead of four).
* Regression tests (`tests/test_dsl/test_replay_preference.py`), extending
  the existing coverage with:
  * `test_partial_rollback_yields_a_row_for_the_second_consecutive_undo`:
    a two-edit, two-undo trace produces exactly one `edit_then_undo` row
    (first undo) and one `partial_rollback` row (second undo), each
    grounded in its own exact input state.
  * `test_checkout_another_state_yields_a_row_preferring_checkout_over_undo`:
    an edit followed by `checkout` back to root produces one
    `checkout_another_state` row whose `chosen_action` is
    `checkout:<root_state_id>` even though `undo` was also legal and would
    have reached the same destination.
  * `test_checkout_row_replays_independently_to_its_recorded_output_state`:
    the same acceptance criterion as the v1 undo/redo tests, re-derived for
    checkout -- replaying `checkout_conversation_state` independently from
    the recorded `input_state_id` lands on exactly `chosen_output_state_id`.
  * `test_fork_then_return_to_original_branch_yields_a_distinct_relation`:
    edit, fork (new branch), then checkout back to the pre-fork state
    produces exactly one `fork_then_choose_one_branch` row, not
    `checkout_another_state`, even though the underlying primitive is
    identical.
  * `test_checkout_between_two_forked_branches_is_fork_then_choose_one_branch`:
    two forks opened from the same state, then a checkout directly between
    the two resulting (neither pre-fork) branches, is still classified
    `fork_then_choose_one_branch` -- the most literal reading of "choose one
    branch" among genuinely divergent forks.

## Explicitly out of scope for this PR

Per the issue's own scope, not attempted here:

* **Merge conflict as a preference row.** Deliberately not attempted --
  see "Fourth slice (v5)" for the honesty argument. This is a considered
  scope decision, not an oversight: modeling it would require inventing a
  "what the user did instead" action the trace never recorded.
* **SFT/preference training** against the DSH3-selected policy/control
  heads (`TypedOperatorPolicyScorer`,
  `src/slm_training/harnesses/experiments/typed_operator_policy.py:316`) or
  the `structured_objectives.py` / `decision_events_v2.py` `ObjectiveView`
  materializers -- no model, checkpoint, or training run is added.
* **The four-baseline comparison** (current-state-only, state+recent
  receipts, state+retrieved events, full transcript+state, last-three-text
  baseline) and the **held-out benefit measurement** the acceptance
  criteria require.
* **Turn-depth and context-view ablations** in `OperatorEventMemoryReportV1`
  -- this PR's report is row counts only.

No causal, calibration, or promotion claim is made. This PR is `wiring`
evidence for the row-extraction primitive only.

## Why a partial slice, not a full disposition

Unlike SLM-336 (AP-035) or SLM-419 (DSH5-11), SLM-418's own prerequisites
(DSH3 policy/control heads, the conversation/collapse/legal-set substrate)
are already merged and available -- there is no unmet upstream gate here.
The remaining scope is genuinely large (training + held-out evaluation
across a five-baseline, multi-metric matrix) and is left for follow-on work
rather than rushed to a false "Done." The issue should stay open against the
training/evaluation work enumerated above.

## Review fixes (v2)

* Rejection candidates are now drawn from the full legal set (operator
  actions and history controls such as `undo`/`redo:<state>` alike), not
  only operator actions -- a valid row is no longer dropped just because the
  only unchosen alternative at a state happens to be a control action.
* `OperatorEventMemoryReportV1` now carries and serializes a `version_stamp`
  (`dsl.operators.replay_preference`), matching the repository's result-artifact
  contract.

## Second slice (v3)

* Added `partial_rollback` and `checkout_another_state` to
  `ReplayPreferenceRelation` and their extraction logic (see above).
* `_available_history_actions` widened from `undo`/`redo:<child>` only to
  also include `checkout:<state>` for every other trace state -- required
  for `checkout_another_state` rows to verify as legal-set members, and
  incidentally widens the `rejected_action` candidate pool available to the
  two pre-existing patterns (no test asserted an exact `rejected_action`
  value, so this is compatible with v1/v2 behavior).
* `dsl.operators.replay_preference` bumped v2 -> v3 in
  `src/slm_training/resources/versions.json`.

## Third slice (v4)

* Added `fork_then_choose_one_branch` to `ReplayPreferenceRelation`. No new
  extraction loop is needed -- it reclassifies a subset of the existing
  `CHECKOUT_STATE` scan: a checkout whose input or output branch digest
  appears in the trace's `FORK`-opened branch set is
  `fork_then_choose_one_branch`; every other checkout stays
  `checkout_another_state`, unchanged from v3.
* Branch digests are opaque content-addressed fingerprints (never display
  names or state hashes carrying semantic meaning), so this classification
  cannot leak the target beyond what the trace's own recorded `FORK` turns
  already establish -- consistent with the issue's adversarial control
  ("branch display names and state hashes cannot leak the target").
* `dsl.operators.replay_preference` bumped v3 -> v4 in
  `src/slm_training/resources/versions.json`.
* Corrected the hardcoded SLM-418 evidence string in
  `src/slm_training/evals/advanced_operator_disposition.py` (previously
  fixed at "2 of 7" since the first slice; now "5 of 7", matching the
  actual coverage after the second and this third slice) via a `no-bump:`
  history note on `evals.advanced_operator_disposition` -- no disposition
  logic or schema changed, and the already-published
  `docs/design/dsh5-12-advanced-operator-disposition-20260727-local/`
  snapshot is untouched, staying immutable point-in-time evidence from when
  SLM-420 ran (before this and the prior SLM-418 slice landed).

## Fourth slice (v5)

* Added `extract_merge_preference_row`, a **standalone extraction function**
  -- not another branch inside `extract_replay_preference_rows`'s turn-scan
  loop -- because a merge attempt is never a recorded `ConversationTraceV1`
  turn. `merge_conversation_branches` (`merge.py`) operates directly on a
  shared `base` `ConversationStateNodeV1` and two independently verified
  `BranchEditV1` edges, and a successful merge starts a **fresh**
  continuation trace (`BranchMergeContinuationV1`, a new trace root) rather
  than appending to either input trace. This confirms the first slice's own
  prediction that merge-conflict detection "needs its own extraction path."
* `ReplayPreferenceRelation.MERGE_SUCCESS` (new). One row per successful
  merge attempt, grounded at the **left branch tip**
  (`left.output_node.state_id`): the legal set there is enumerated with
  `merge:<sorted-tip-pair>` (a new, order-independent canonical action
  name -- sorted so it serializes identically regardless of which edge is
  passed as `left` vs `right`, matching `merge_conversation_branches`'s own
  order-invariant `decision_id`) and `checkout:<right tip>` (plus `undo`,
  when a parent exists) offered alongside it, via the same
  `ordinary_nonoperator_actions` mechanism every other pattern in this
  module uses. `chosen_output_state_id` is the real
  `decision.continuation.merged_node.state_id` -- re-running
  `merge_conversation_branches` on the same `base`/`left`/`right`
  independently reproduces the identical merged state, satisfying the
  issue's replay-independence acceptance criterion exactly like every
  other relation.
* **Merge conflict is deliberately *not* modeled as a row.** The issue's
  own acceptance criterion requires every row to independently replay to
  its recorded `chosen_output_state_id`; a conflicting merge produces no
  successor state at all, so a row would have to invent a "what the user
  did instead" action the trace never recorded -- violating the issue's
  own adversarial control that chosen/rejected rows share exact, evidenced
  context. The issue's instruction to "mark rejected candidates as typed
  illegal/conflict controls outside the ranking denominator" is honored by
  **construction** instead: `extract_merge_preference_row` only ever adds
  `merge:<pair>` as a legal candidate action after
  `merge_conversation_branches` has already confirmed `decision.succeeded`,
  so a conflicting merge can never leak into any ranking denominator in the
  first place. `test_merge_conflict_never_yields_a_preference_row` proves
  this directly: a same-target-field conflict (`SAME_NODE_INCOMPATIBLE_
  EDIT`) yields `None`, not a fabricated row.
* `authority_resolver` (the same `BranchAuthorityResolver` type
  `merge_conversation_branches` itself takes) is resolved from
  `left.input_node` -- the same node the merge module's own internals
  resolve authority from -- since a `BranchEditV1` is one verified single-
  application edge and its input-state authority governs the actions legal
  at its output tip too.
* `dsl.operators.replay_preference` bumped v4 -> v5 in
  `src/slm_training/resources/versions.json`; `dsl.operators.contracts`
  gets a `no-bump:` history entry for the new `extract_merge_preference_row`
  re-export from `operators/__init__.py`.
* Corrected the hardcoded SLM-418 evidence string in
  `src/slm_training/evals/advanced_operator_disposition.py` (previously "5
  of 7"; now "6 of 7", with the remaining-gap claim narrowed from "5 of 7
  patterns not attempted" to "1 of 7 (pronoun/focus)" plus an explicit note
  that merge conflict is an honest non-row scope decision) via a
  `no-bump:` history note on `evals.advanced_operator_disposition` -- no
  disposition logic or schema changed, and the already-published
  `docs/design/dsh5-12-advanced-operator-disposition-20260727-local/`
  snapshot is untouched, staying immutable point-in-time evidence from
  before this slice landed.

## Fifth slice (v6)

* Added `pronoun_focus_followup` to `ReplayPreferenceRelation` -- the last
  of the issue's seven named patterns. Unlike merge-success, this **is**
  another branch inside `extract_replay_preference_rows`'s existing
  turn-pair scan loop: a second consecutive `AST_EDIT` turn.
* **Focus**, the module's only concept for it, is never a transcript
  pronoun or a semantic descriptor: it is `_touched_refs`, the exact
  `OperatorRef` values the *immediately preceding* `AST_EDIT` turn's own
  verified `OperatorApplicationV1.arguments` bound. A pair of consecutive
  edits is classified `PRONOUN_FOCUS_FOLLOWUP` only when (1) that focus set
  is non-empty (a zero-argument operator, like the base fixture every other
  pattern in this module uses, never establishes one), (2) the following
  edit's own bound arguments intersect it (the user kept operating on a ref
  they had just touched), and (3) the exact legal set at the shared decision
  state (`enumerate_operator_legal_set`, matched to the following turn's
  recorded application by `operator_fingerprint` and bound `arguments`)
  contains a **sibling**: another legal action for the *same operator* whose
  own bound refs do **not** overlap the focus set -- a genuinely available,
  equally legal "switch to something else" the user did not take. Without a
  real sibling candidate, no row is emitted, matching every other pattern's
  convention that undo/redo/checkout/continued-focus is never asserted
  preferred by default.
* This directly answers the issue's own "ambiguous sibling" and "pronoun
  focus" matrix rows: the pattern only ever fires when a second, disjoint
  legal target genuinely existed at that state, and the row records that
  the user's implicit "it" continuation was chosen over it.
* Deliberately does **not** attempt: switching to an explicit, different,
  legal reference (the issue's "exact named reference" matrix case) is
  honestly left unrowed rather than asserted a correction -- there is no
  "user was wrong" signal to record when they simply named something else.
  Multi-argument operators, transaction-commit turns, and any true
  natural-language pronoun/reference-resolution machinery over
  `ReferenceTableV1` remain out of scope; this slice is DAG-argument-set
  overlap only, exactly as adversarial control requires ("text history
  cannot reconstruct a different state than the DAG").
* `dsl.operators.replay_preference` bumped v5 -> v6 in
  `src/slm_training/resources/versions.json`.
* Corrected the hardcoded SLM-418 evidence string in
  `src/slm_training/evals/advanced_operator_disposition.py` (previously "6
  of 7"; now "7 of 7", with the remaining-gap claim narrowed from "1 of 7
  (pronoun/focus)" to the issue's training/measurement scope only) via a
  `no-bump:` history note on `evals.advanced_operator_disposition` -- no
  disposition logic or schema changed, and the already-published
  `docs/design/dsh5-12-advanced-operator-disposition-20260727-local/`
  snapshot is untouched, staying immutable point-in-time evidence from
  before this slice landed.

## Sixth slice (2026-07-27)

This slice closes the "turn-depth and context-view ablations" gap
`OperatorEventMemoryReportV1`'s own docstring named, and wires the issue's
"Compare current state only, state plus recent semantic receipts, state
plus retrieved relevant events, complete transcript plus state, and
existing last-three-text-history baseline" and "Train matched
SFT/preference variants ... event and state IDs are join/evidence keys,
never semantic embeddings" bullets -- at bounded, fixture-scale, honestly
reduced from the full issue.

### What was built

* `src/slm_training/dsl/operators/replay_preference_context_views.py`
  (new; `dsl.operators.replay_preference_context_views` v1). Pure, no-torch
  module defining:
  * `ContextView` -- the five context-view input representations the
    issue's own bullet names: `CURRENT_STATE_ONLY` (no history receipt is
    ever visible), `STATE_PLUS_RECENT_RECEIPTS` (the last `turn_depth`
    receipts in the trace's own chronological turn order -- may cross
    branch boundaries), `STATE_PLUS_RETRIEVED_EVENTS` (the last
    `turn_depth` receipts on the single-branch **ancestry** path that
    causally produced the decision state, walked via `parent_state_id` --
    distinct from recency, since it reflects the DAG's fixed structural
    lineage rather than chronological turn order), `FULL_TRACE_PLUS_STATE`
    (every receipt before the decision; `turn_depth` saturates), and
    `LAST_THREE_TEXT_HISTORY` (a fixed 3-receipt window with state ids
    stripped -- only the structural `action_kind` token survives, so this
    view can never reconstruct exact state by construction, directly
    satisfying the issue's adversarial control "text history cannot
    reconstruct a different state than the DAG").
  * `TURN_DEPTHS = (1, 2, 4, 8, 16)` -- the issue's own matrix.
  * `OperatorTurnReceiptV1` / `receipts_from_trace` / `state_lookup_from_trace`
    -- opaque join-key receipts (`action_kind` + the two state ids) built
    directly from a trace's own recorded turns; never transcript text,
    never a semantic embedding of any id.
  * `build_context_view_window` -- the core window builder. Disambiguates
    *which* occurrence of a possibly-revisited decision state a call is
    about via the row's own `chosen_output_state_id` (see "Bug fixed during
    this slice" below), not `input_state_id` alone.
  * `OperatorEventMemoryAblationReportV1` / `build_ablation_report` -- the
    **structural** (not yet trained) turn-depth x context-view grid: for
    every row and every `(view, turn_depth)` cell, how many history
    receipts that combination exposes and the most recently visible
    `action_kind`. This is the literal dimension `OperatorEventMemoryReportV1`'s
    docstring flagged as missing; `OperatorEventMemoryReportV1` itself is
    intentionally left unchanged (still row-counts-only) -- the ablation
    grid is a separate, sibling report built from it, matching this
    task's own "extend ... or add a sibling report type" instruction.
* `src/slm_training/harnesses/preference/replay_preference_context_view_variants.py`
  (new; `harness.preference.replay_preference_context_view_variants` v1).
  The bounded, fixture-scale training/comparison harness:
  * `synthesize_bounded_session_corpus()` -- a small, deterministic,
    **synthetic** corpus (never real user telemetry) of 8 conversation
    "sessions," each its own `group_id`: five `rollback_chain_<K>` sessions
    (`K` in `TURN_DEPTHS`, each `K` novel edits then a full `K`-undo
    rollback then one redo, giving genuine, real turn-depth variation for
    `EDIT_THEN_UNDO`/`PARTIAL_ROLLBACK`/`UNDO_THEN_REDO`), one
    `checkout_and_fork` session (`CHECKOUT_ANOTHER_STATE` +
    `FORK_THEN_CHOOSE_ONE_BRANCH`), one `pronoun_focus` session
    (`PRONOUN_FOCUS_FOLLOWUP`), and one `merge_success` session
    (`MERGE_SUCCESS`) -- all seven named relations appear at least once.
    States are built from a strictly-monotonic counter operator (never a
    small content-cycle) specifically so a chain of any length never
    collides with an earlier state's digest.
  * Splits sessions train/held-out via `split_for_group` (the same
    stable-hash mechanism `local_decisions.py` already uses for the same
    purpose) -- every row in one session shares that session's split, per
    the issue's adversarial control "conversation variants stay in one
    split." A real run: 8 sessions, 6 train / 2 held-out (`rollback_chain_1`,
    `checkout_and_fork`), 40 rows total.
  * `_features` -- exactly two structural features per candidate action:
    `history_repeat_score` (fraction of the visible window's receipts whose
    `action_kind` matches the candidate's own) and `is_history_control`
    (whether the candidate is a control action at all, computable from the
    decision state alone with zero history -- so `CURRENT_STATE_ONLY` is
    not information-free). Both are join-key/structural counts, never
    transcript text, never a semantic embedding.
  * `_train_pairwise_linear_scorer` -- full-batch gradient descent on
    pairwise logistic loss, two weights, no bias (bias cancels in a
    chosen-minus-rejected margin over the same window). This is
    deliberately **not** a DSH3 policy/control head or checkpoint -- wiring
    a real trained head against this corpus is out of scope for this slice.
  * `train_replay_preference_context_view_variants` -- trains and evaluates
    one scorer per `(view, turn_depth)` cell (25 cells), reporting pairwise
    chosen>rejected accuracy (overall and by semantic relation), a narrow
    pairwise-calibration proxy (mean Brier error against the always-1
    "chosen wins" label), corpus-composition rate (`undo_family_rate`), and
    an honest `held_out_benefit` verdict. Fails closed (raises) if either
    split is empty, and enforces `slm_training.levers.MAX_HARNESS_WALL_SECONDS`
    even though a real run completes in well under a second.
* `src/slm_training/evals/ambiguous_operator_followups.py` (new). Assembles
  the disposition: runs the comparison, builds the ablation grid per
  session, and packages `OUT_OF_SCOPE_METRICS` -- explicit, non-deletable
  notes for every issue-named metric this slice does not measure.
* `scripts/run_replay_preference_context_view_ablation.py` (new). CLI entry
  printing the disposition as JSON; used to produce the real numbers below.
* `tests/test_harnesses/preference/test_operator_history_pairs.py` (new, 16
  tests) and `tests/test_evals/test_ambiguous_operator_followups.py` (new,
  9 tests) -- the two test files the issue's own "Tests" section names as
  not yet existing.
* `src/slm_training/dsl/operators/__init__.py` -- re-exports the new
  module's public symbols, matching every prior slice's convention
  (`no-bump:` note on `dsl.operators.contracts`, which claims the
  directory-level path).
* `src/slm_training/evals/advanced_operator_disposition.py` -- corrected the
  now-stale `EVENT_MEMORY` claim's `dimension_reasons` (previously "No
  turn-depth or context-view ablation exists"), via a `no-bump:` note on
  `evals.advanced_operator_disposition` (no disposition logic or schema
  changed; the already-published
  `docs/design/dsh5-12-advanced-operator-disposition-20260727-local/`
  snapshot stays untouched, immutable point-in-time evidence).

### Bug fixed during this slice: revisited-decision-state disambiguation

While validating the corpus against real `rollback_chain` traces, the
initial `_decision_position` implementation (match the first receipt whose
`input_state_id` equals the decision state) was found to silently pick the
**wrong** occurrence whenever a state is later revisited -- which every
rollback chain does by construction (a state is first visited going forward
during the edit chain, then revisited going backward during the rollback).
`PARTIAL_ROLLBACK` rows' `state_plus_recent_receipts` windows were computed
against the *forward* occurrence instead of the *actual* decision turn,
silently reporting "operator" as the most-recent kind for every depth
instead of the correct "undo." Fixed by disambiguating on the row's own
recorded `chosen_output_state_id` in addition to `input_state_id` -- every
one of the seven named patterns derives `chosen_output_state_id` from a
real recorded turn's own `output_state_id`, so the exact `(input, output)`
edge always identifies the one turn a row is actually about. The ancestry
walk (`STATE_PLUS_RETRIEVED_EVENTS`) had the same class of bug (keyed by
output state alone, which collides when a state is later reproduced by an
undo/redo) and is now keyed by the exact `(parent_state_id, state)` edge,
which always resolves to the state's original creating turn regardless of
how many times it is later revisited.
`test_recent_receipts_most_recent_kind_reflects_the_immediately_prior_undo`
and `test_retrieved_events_is_ancestry_not_recency` are regression tests for
this fix.

### What was measured (real run, `python -m scripts.run_replay_preference_context_view_ablation`)

* **Corpus:** 8 sessions (6 train / 2 held-out), 40 rows total, all 7 named
  relations present. `undo_family_rate` (descriptive corpus composition,
  not a benefit claim): 0.9 -- i.e. 90% of rows are
  `edit_then_undo`/`undo_then_redo`/`partial_rollback`, reflecting the
  `rollback_chain` sessions' design (deep chains deliberately generate many
  `PARTIAL_ROLLBACK` rows to exercise the turn-depth axis).
* **Held-out pairwise preference accuracy: 1.0 for every one of the 25
  `(view, turn_depth)` cells**, including the `current_state_only` baseline.
  This is an honest **ceiling effect**, not evidence of benefit: the
  held-out split (`rollback_chain_1` + `checkout_and_fork`, 4 pairs) is
  small enough that `is_history_control` alone -- available with *zero*
  history, from the decision state alone -- already perfectly separates
  chosen from rejected in every held-out row. `held_out_benefit.verdict` is
  therefore honestly reported as **`no_benefit_fixture_scale`**
  (`baseline_accuracy=1.0`, `best_accuracy=1.0` at
  `state_plus_recent_receipts`/depth 1 -- tied with baseline, not exceeding
  it), per the acceptance criteria's own explicit permission to record an
  honest no-benefit result. The mean pairwise-calibration error (a Brier
  proxy, not a CAP-gated calibration measurement) is small and
  view-dependent (e.g. `0.00011` for `current_state_only` vs `0.00059` for
  `state_plus_recent_receipts` at depth 1, narrowing toward `0.00007` by
  depth 16), reflecting the trained weights' own confidence, not accuracy.
* **The structural ablation grid itself is genuinely informative** even
  though held-out accuracy ceilings: on a real `rollback_chain_8` trace, the
  `state_plus_recent_receipts` view's visible-receipt count for
  `PARTIAL_ROLLBACK` rows grows exactly as expected with `turn_depth`
  (1, 2, 4, 8, capping at the available history, e.g. 9-15 depending on
  rollback position for `turn_depth=16`), and its most-recent visible
  `action_kind` is consistently `"undo"` -- while the SAME rows'
  `state_plus_retrieved_events` (ancestry) view instead shows the *forward
  edit chain* (`action_kind="operator"`), a real, structurally distinct
  signal from recency. This is the concrete demonstration that the two
  views are not interchangeable, even though this slice's tiny held-out
  corpus cannot yet show one out-predicting the other.
* **Correction/undo rate:** the 0.9 `undo_family_rate` above; not
  independently cross-validated against a held-out distribution (the corpus
  is too small to split further).
* **Trace replay:** inherited, not independently re-measured this slice --
  `extract_replay_preference_rows`/`extract_merge_preference_row` are
  unchanged, and their own acceptance-criterion tests already prove every
  relation replays; this slice adds direct spot-check tests
  (`test_a_rollback_chain_row_replays_independently_to_its_recorded_output_state`
  in both new test files) confirming the *synthetic* corpus's own rows
  independently replay too, using the same pattern.
* **Unintended mutations:** structurally zero by construction -- the
  scorer only calls `_score()` over already-extracted legal candidates and
  never calls `OperatorLibraryV1.apply`.

### Explicitly out of scope (unchanged from, or newly identified by, this slice)

* Real SFT/preference training against the DSH3-selected policy/control
  heads (`TypedOperatorPolicyScorer`) -- the trained "variant" here is a
  two-parameter linear pairwise scorer, never a neural policy head or
  checkpoint. No model, checkpoint, or promotion is created by this slice.
* A powered or real (non-synthetic) corpus -- 40 rows across 8 sessions is
  wiring evidence, not a statistically powered held-out-benefit study.
* Real action/operator/argument accuracy against a trained policy -- only a
  pairwise ranking-accuracy proxy over two structural features is measured.
* CAP0/CAP1/CAP2 retention -- requires the full CAP-gated eval suite
  (`src/slm_training/evals/cap2_operator.py` and friends) integrated with a
  trained policy checkpoint; genuinely out of scope for a diagnostic linear
  scorer.
* Turn-depth padding for `CHECKOUT_ANOTHER_STATE`, `FORK_THEN_CHOOSE_ONE_BRANCH`,
  `MERGE_SUCCESS`, and `PRONOUN_FOCUS_FOLLOWUP` -- each gets exactly one
  synthetic session with no padded prior history; extending
  `sequence_merge.py`'s N-step machinery or a longer fork/checkout ladder to
  give these depth variation too is left for a future slice.
* `dsl.operators.replay_preference` itself (the row-extraction module) is
  **unchanged** this slice -- still v6, still 7 of 7 named patterns,
  `OperatorEventMemoryReportV1` still row-counts-only by design (the
  ablation grid lives in the new sibling module instead).

No causal, calibration, or promotion claim is made. No checkpoint or model
card update applies -- this slice creates no checkpoint.

## Review fixes (sixth slice)

CodeRabbit review on the PR surfaced four real issues, fixed here rather than
argued past:

* **NaN baseline could silently produce a "no benefit" verdict.** If the
  `current_state_only` baseline cell had zero held-out pairs,
  `pairwise_preference_accuracy` was `float("nan")`, and every
  `accuracy > baseline_accuracy` comparison against a NaN is `False` --
  producing a spurious, unearned `no_benefit_fixture_scale` verdict instead
  of failing closed. `train_replay_preference_context_view_variants` now
  raises `ValueError` when the baseline cell is unmeasured *or* NaN.
* **The `held_out_benefit_statistical_power` scope note hardcoded "8
  sessions."** `evaluate_ambiguous_operator_followups` accepts an arbitrary
  caller-supplied corpus (a test passes exactly two sessions), so a fixed
  "8 sessions" string was simply false for any other corpus size. The note
  is now derived per-call from `comparison.row_count` /
  `comparison.session_count`.
* **`docs/design/dsh5-10-replay-preference-rows.md` (this file) was missing
  from `harness.preference.replay_preference_context_view_variants`'s
  registered `paths`** in `versions.json`, leaving the experiment narrative
  outside that component's version/no-bump tracking contract. Added.
* **Two tests accepted every possible `held_out_benefit` verdict**,
  including `no_non_baseline_held_out_data` (the "we could not measure
  anything" outcome), which made a real corpus/split regression on the
  default full synthetic corpus indistinguishable from a genuine
  fixture-scale result. Both tests now assert the narrower
  `{benefit_observed_fixture_scale, no_benefit_fixture_scale}` set for that
  corpus; the excluded verdict remains a legitimate code path for a
  caller-supplied corpus too small to have non-baseline held-out data.

Also applied, both trivial and uncontroversial: an unchecked `int()` parse
in the counter fixture operator now raises `OperatorRejectedError` instead
of a bare `ValueError` on malformed input; the per-iteration `_pairs`
closure in the comparison loop is now a module-level `_view_depth_pairs`
function taking `view`/`depth` as explicit arguments instead of capturing
loop variables; and the unused `TURN_DEPTH_IS_BOUNDING` constant was
deleted.

Not applied, with reasons: prefixing the doc's own reproducibility commands
with `rtk` was skipped -- these blocks are exact, copy-pasteable
reproduction commands (the convention every prior slice in this file
follows), and `rtk` is a token-compression wrapper for an agent's own shell
usage, not part of the documented commands themselves. Promoting the test
modules' shared private fixture helpers (`_provenance`,
`_rollback_chain_trace`, `_bump`, `_counter_pack_and_root`, `_sha`,
`_table`) to a public/shared `conftest.py` API was also skipped as a
non-functional structural refactor left for a future slice, not a
correctness or honesty defect.

## Reproducibility

```bash
NODE_OPTIONS= pytest -q tests/test_dsl/test_operator_conversation.py tests/test_harnesses/preference/test_operator_history_pairs.py tests/test_evals/test_ambiguous_operator_followups.py tests/test_dsl/test_replay_preference.py
python -m scripts.run_replay_preference_context_view_ablation
python -m scripts.verify_version_stamps --check --base origin/main
python -m scripts.repo_policy
python -m scripts.verify_decode_invariants
ruff check src/slm_training/dsl/operators/replay_preference_context_views.py src/slm_training/harnesses/preference/replay_preference_context_view_variants.py src/slm_training/evals/ambiguous_operator_followups.py scripts/run_replay_preference_context_view_ablation.py tests/test_harnesses/preference/test_operator_history_pairs.py tests/test_evals/test_ambiguous_operator_followups.py src/slm_training/dsl/operators/__init__.py src/slm_training/evals/advanced_operator_disposition.py
```

Result (this PR, same fresh `.venv` -- Python 3.12, `pip install -e ".[dev,grammar]"`, plus `NODE_OPTIONS= npm ci` in `src/apps/openui_bridge`, run with `NODE_OPTIONS=` cleared for the same reason as the fifth slice): `tests/test_dsl/test_operator_conversation.py` + the two new test files: `39 passed`; `tests/test_dsl/test_replay_preference.py`: `17 passed` (56 total across the issue's own "Tests" command). `ruff check`: clean on every touched/created file. `python -m scripts.verify_version_stamps --check --base origin/main`: `ok (10 changed file(s), 4 component(s) touched)`. `python -m scripts.repo_policy`: `ok (tracked + untracked)`. `python -m scripts.verify_decode_invariants`: clean. The ablation script's real output: 8 sessions (6 train / 2 held-out), 40 rows, `undo_family_rate=0.9`, `held_out_benefit={"verdict": "no_benefit_fixture_scale", "baseline_accuracy": 1.0, "best_view": "state_plus_recent_receipts", "best_turn_depth": 1, "best_accuracy": 1.0, ...}` -- all 25 `(view, turn_depth)` cells report `pairwise_preference_accuracy=1.0` (the ceiling effect explained above).

## Reproducibility (fifth slice)

```bash
NODE_OPTIONS= pytest -q tests/test_dsl/test_replay_preference.py tests/test_dsl/test_operator_merge.py tests/test_dsl/test_operator_conversation.py tests/test_evals/test_advanced_operator_disposition.py tests/test_scripts/test_validate_advanced_operator_disposition.py
```

Result (this PR, real run in a fresh `.venv` -- Python 3.12, `pip install -e ".[dev,grammar]"`, plus `NODE_OPTIONS= npm ci` in `src/apps/openui_bridge` for the G2/G8 schema-oracle gates the pack authority requires; the ambient `--import tsx` `NODE_OPTIONS` is rejected by this Node 22 build both for `npm ci` and for `pytest`, unrelated to this change): `61 passed`. Also verified: `ruff check` clean on every changed file; `python -m scripts.verify_version_stamps --check --base origin/claude/great-dirac-v82ph9` -- `ok (2 component(s) touched)`; `python -m scripts.repo_policy` -- `ok`; `python -m scripts.verify_decode_invariants` -- clean.
