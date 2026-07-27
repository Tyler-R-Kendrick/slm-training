# DSH5-10: replay-grounded preference rows from undo/redo history (SLM-418)

**Status:** partial slice, in progress (third increment).
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

* **Two of seven patterns**: merge success/conflict and pronoun/focus
  follow-ups. Merge-conflict detection in particular lives in `merge.py`,
  not `conversation.py`/`collapse.py`, and needs its own extraction path.
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
The remaining scope is genuinely large (two more patterns, plus training +
held-out evaluation across a five-baseline, multi-metric matrix) and is left
for follow-on work rather than rushed to a false "Done." The issue should
stay open against the patterns and training/evaluation work enumerated
above.

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

## Reproducibility

```bash
pytest -q tests/test_dsl/test_replay_preference.py tests/test_dsl/test_operator_conversation.py tests/test_evals/test_advanced_operator_disposition.py tests/test_scripts/test_validate_advanced_operator_disposition.py
```

Result (this PR, sandboxed run with `NODE_OPTIONS` cleared -- the ambient
`--import tsx` flag is rejected by this Node 22 build, unrelated to this
change): `40 passed`.
