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

## Reproducibility

```bash
NODE_OPTIONS= pytest -q tests/test_dsl/test_replay_preference.py tests/test_dsl/test_operator_merge.py tests/test_dsl/test_operator_conversation.py tests/test_evals/test_advanced_operator_disposition.py tests/test_scripts/test_validate_advanced_operator_disposition.py
```

Result (this PR, real run in a fresh `.venv` -- Python 3.12, `pip install -e ".[dev,grammar]"`, plus `NODE_OPTIONS= npm ci` in `src/apps/openui_bridge` for the G2/G8 schema-oracle gates the pack authority requires; the ambient `--import tsx` `NODE_OPTIONS` is rejected by this Node 22 build both for `npm ci` and for `pytest`, unrelated to this change): `61 passed`. Also verified: `ruff check` clean on every changed file; `python -m scripts.verify_version_stamps --check --base origin/claude/great-dirac-v82ph9` -- `ok (2 component(s) touched)`; `python -m scripts.repo_policy` -- `ok`; `python -m scripts.verify_decode_invariants` -- clean.
