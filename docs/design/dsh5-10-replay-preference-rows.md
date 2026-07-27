# DSH5-10: replay-grounded preference rows from undo/redo history (SLM-418)

**Status:** partial slice, in progress (ninth increment).
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

The fifth slice added the seventh and final named pattern, bringing
extraction coverage to 7 of 7 (see "Fifth slice (v6)" below). The sixth
slice added the first (and, until then, entirely missing) converter from an
extracted row to the `PreferencePair` shape `scripts/train_preference.py`
actually consumes. The seventh slice ran the first real
(`fixture_or_scratch`) end-to-end pass: a scratch SFT checkpoint, a demo
replay-preference pairs corpus covering 3 of the 7 named patterns, and one
bounded `scripts/train_preference.py train` call against it. This slice
extends the demo corpus to a fourth pattern, `merge_success` -- the one
pattern the seventh slice's trace-scan corpus could not reach -- and reruns
the same training chain against the now-richer 3-pair corpus; see "Eighth
slice" below. This slice takes the first real step onto the issue's actual
named training target: `typed_operator_policy.py`'s
`TypedOperatorPolicyScorer`, not the generic TwoTower pair format the
sixth/seventh/eighth slices used for tooling compatibility. Only one of the
seven named patterns (`pronoun_focus_followup`) is honestly representable
there today -- see "Ninth slice" below for why, and for a real, structural
(not a bug) null-training finding this exposed. What remains is a *real*
pairs corpus (no captured conversation-trace data exists anywhere in this
repo -- see below), a scope decision for the other six patterns' history-
control/merge actions (none has a row in the typed policy's action space),
the four-baseline comparison, held-out benefit measurement, and turn-depth/
context-view ablations.

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

## Sixth slice

* New module `src/slm_training/harnesses/preference/replay_pairs.py`:
  `render_replay_preference_pair` renders one `OperatorReplayPreferenceRowV1`
  into a `PreferencePair` (`prompt`/`chosen`/`rejected`, the shape
  `scripts/train_preference.py`'s `build-pairs`/`train` path consumes), and
  `render_replay_preference_pairs` batches a report's rows. Before this
  slice **nothing in the repo converted a row into any trainable shape** --
  confirmed by grepping the whole tree for `ReplayPreferenceRelation`,
  `extract_replay_preference_rows`, and `extract_merge_preference_row`: only
  the operators module itself, its own test file, the version-stamp
  registry, and this doc referenced them.
* **Never fabricates a state.** `chosen_output_state_id` is always an
  already-materialized node by construction (true of every row every prior
  slice has produced). The rejected side is resolved the same
  replay-grounded way, with no shortcut:
  * `undo` -> the input state's own parent (an existing node -- no
    computation).
  * `redo:<state_id>` / `checkout:<state_id>` -> that state is already
    materialized; the id is read directly out of the action string.
  * an operator action's serialized form (the `pronoun_focus_followup`
    sibling case) -> recomputes the exact legal set at the input state
    (`enumerate_operator_legal_set`, the identical call
    `extract_replay_preference_rows` itself makes), matches the row's
    `rejected_action` string against it, and actually applies it through
    the pack-authorized `OperatorLibraryV1.apply` -- the same executor
    every other application in this module goes through. This is a real,
    independently-reproducible state, not a guess.
  * `merge:<pair>` as a *rejected* action is deliberately left unrendered
    (returns `None`). It cannot occur under today's single-merge-candidate
    extraction (`extract_merge_preference_row` only ever offers one
    `merge:<pair>` candidate, and it is always the *chosen* side of a
    `MERGE_SUCCESS` row -- see `test_rejected_merge_action_is_never_rendered`
    for the defensive-branch proof), but the renderer refuses to guess a
    merged state instead of honestly declining if that ever changes.
* **Two call shapes, one renderer.** Trace-turn-scan rows
  (`extract_replay_preference_rows`) resolve nodes via
  `resolve_node=trace.node` directly. `MERGE_SUCCESS` rows
  (`extract_merge_preference_row`) never live on a shared
  `ConversationTraceV1` -- `left`/`right` are independently-verified
  `BranchEditV1` edges and `decision.continuation.merged_node` is a fresh
  node -- so the new `merge_node_resolver(left, right, decision)` builds the
  equivalent `NodeResolver` over exactly those nodes instead.
* **Open, explicitly-flagged modeling choice:** the pair's `prompt` is set
  to the input state's own DSL source (score chosen/rejected
  *continuations* of the current AST against it). This is a first,
  documented cut for compatibility with the existing generic TwoTower pair
  format, not a validated training-objective decision -- the disposition's
  own remaining-scope note names the DSH3-selected policy/control heads
  (`typed_operator_policy.py`) as the actual training target, and whoever
  wires a real training run should treat the prompt shape as open rather
  than inherited from this slice.
* **Still wiring only.** This slice adds the converter and its tests; it
  does not build a pairs corpus from real conversation traces, does not run
  `scripts/train_preference.py`, and makes no training or held-out-benefit
  claim. `harness.preference.replay_pairs` registered fresh (`v1`, initial
  registration) in `src/slm_training/resources/versions.json`; no existing
  component's behavior changed (`dsl.operators.replay_preference` stays at
  `v6` -- this slice only *consumes* its existing public API).

## Seventh slice

* New script `scripts/build_replay_preference_pairs.py`: builds one small,
  deterministic, honestly-labeled scratch conversation (`build_demo_trace`)
  exercising three of the seven named patterns (edit-then-undo,
  undo-then-redo, checkout-another-state) with a toy zero-argument cycling
  operator, extracts rows, renders them via the sixth slice's
  `render_replay_preference_pairs`, and writes a real `pairs.jsonl` via the
  existing `write_pairs`. This is the **first real, on-disk pairs corpus
  this feature line has ever produced** -- everything before this slice was
  either an in-memory row/pair in a unit test, or a function that could
  render one but had never been run outside `pytest`.
* **No real corpus exists to build from.** Confirmed again this slice (grep
  for `"conversation_trace"` / `"schema": "conversation_trace` across
  `src/slm_training/resources/`): zero persisted `ConversationTraceV1`
  records anywhere in this repo, and no harness ingests captured
  conversation history (`build_symbolic_operator_corpus` in
  `harnesses/train_data/operator_corpus.py` *synthesizes* traces
  combinatorially from existing gold DSL records; it does not read
  real/captured usage). The demo trace here is explicitly scratch, not a
  stand-in for that missing corpus.
* **2 of 3 rows render, honestly.** `edit_then_undo` and
  `checkout_another_state` render real, non-degenerate pairs.
  `undo_then_redo` never does, for a structural reason, not a bug: for
  *any* deterministic, zero-argument operator, `redo` and "reapply the same
  operator at the same input state" are, by construction, the identical
  resulting text, so the renderer's own dedup guard
  (`render_replay_preference_pair`) correctly declines rather than emitting
  a self-contradictory pair. The script's own printed report says so
  (`pairs_dropped: 1`) rather than silently hiding it.
* **First real training run using DSH5-10 rows, full pipeline, one command
  chain:**
  ```bash
  python -m scripts.train_model --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
    --model twotower --context-backend scratch --steps 8 \
    --run-id replay_pref_sft_ckpt --no-sync-checkpoints --device cpu --seed 0
  python -m scripts.build_replay_preference_pairs \
    --out outputs/data/preference/replay_demo_pairs.jsonl
  python -m scripts.train_preference train \
    --checkpoint outputs/runs/replay_pref_sft_ckpt/checkpoints/last.pt \
    --pairs outputs/data/preference/replay_demo_pairs.jsonl \
    --out-dir outputs/runs/replay_pref_dpo --steps 6 --device cpu
  ```
  SFT step: `last_loss=32.610084533691406` -- identical to every prior
  `wf_smoke_v2`/seed-0/8-step row in
  `docs/design/autotrain-loop-ledger-20260725.md` (16+ prior independent
  reproductions), confirming this checkpoint is the same deterministic
  artifact those rows already verified, not a new unverified path.
  Preference step: `{"steps": 6, "last_loss": 1.0767018795013428,
  "mean_loss": 0.9917331635951996, "n_pairs": 2, "reference_free": true}`
  (`outputs/runs/replay_pref_dpo/preference_summary.json`, not committed --
  `outputs/` is gitignored). Both commands completed in well under
  `MAX_RUN_MINUTES=3` (SFT ~10s per the ledger's own prior timings for this
  exact recipe; the 6-step preference pass over 2 pairs on CPU is
  comparably fast).
* **Still not a training or held-out-benefit claim.** `n_pairs=2` on a
  scratch fixture with no held-out split is `fixture_or_scratch` wiring
  evidence that the pipeline *runs end to end for real* -- SFT checkpoint
  in, DSH5-10-extracted-and-rendered pairs in, a real
  `train_preference.py train` loss trajectory out. It says nothing about
  whether this signal helps the model, generalizes, or should train the
  DSH3-selected policy head the issue actually asks about.
* `harness.preference.replay_pairs` bumped `v1` -> `v2` in
  `src/slm_training/resources/versions.json` (adds the new script + test to
  its watched paths).

## Eighth slice

* Adds `build_demo_merge_scenario` to `scripts/build_replay_preference_pairs.py`:
  a second scratch fixture -- two branches forked from a shared base editing
  disjoint node refs (title vs body) -- mirroring the exact disjoint-target
  shape `tests/test_dsl/test_operator_merge.py` already verifies merges
  cleanly, replayably, and order-invariantly. This reaches `merge_success`,
  the one named pattern the seventh slice's single-trace corpus structurally
  cannot: `extract_merge_preference_row` never operates on a shared
  `ConversationTraceV1` (see the sixth slice's `merge_node_resolver`), so it
  needs its own two-branch construction rather than another turn in the
  same trace.
* `main()` now combines both sources into one report and one `pairs.jsonl`:
  4 rows total (3 from the trace-scan corpus + 1 `merge_success`), 3 render.
  `undo_then_redo` is still the only drop, for the same structural reason
  the seventh slice documented (never a bug to fix on this fixture family).
* Reran the full training chain against the now-richer corpus:
  ```bash
  python -m scripts.train_model --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
    --model twotower --context-backend scratch --steps 8 \
    --run-id replay_pref_sft_ckpt2 --no-sync-checkpoints --device cpu --seed 0
  python -m scripts.build_replay_preference_pairs \
    --out outputs/data/preference/replay_demo_pairs_v2.jsonl
  python -m scripts.train_preference train \
    --checkpoint outputs/runs/replay_pref_sft_ckpt2/checkpoints/last.pt \
    --pairs outputs/data/preference/replay_demo_pairs_v2.jsonl \
    --out-dir outputs/runs/replay_pref_dpo2 --steps 9 --device cpu
  ```
  SFT step: `last_loss=32.610084533691406` again -- the same deterministic
  artifact every prior `wf_smoke_v2`/seed-0/8-step row in the smoke-loop
  ledger reproduces. Preference step, now over 3 pairs instead of 2:
  `{"steps": 9, "last_loss": 0.5314897894859314, "mean_loss":
  0.7201318964362144, "n_pairs": 3, "reference_free": true}`
  (`outputs/runs/replay_pref_dpo2/preference_summary.json`, not committed).
  Both commands again well under `MAX_RUN_MINUTES=3`.
* **Still not a training or held-out-benefit claim** -- `n_pairs=3` on a
  scratch fixture is a larger, more structurally diverse smoke corpus (now
  covering 4 of the 7 named patterns instead of 3), not evidence the signal
  helps the model or generalizes. The three remaining un-exercised-in-a-script
  patterns (`partial_rollback`, `fork_then_choose_one_branch`, and
  `pronoun_focus_followup`) are left for a future slice rather than piling
  more scratch fixtures onto this one; see the doc's still-open
  training/measurement scope above.
* `harness.preference.replay_pairs` bumped `v2` -> `v3` in
  `src/slm_training/resources/versions.json`.

## Ninth slice

Every slice from the sixth onward has flagged the same gap: the sixth
through eighth slices' `PreferencePair`/TwoTower path exists only for
tooling compatibility, since the disposition's own remaining-scope note
names `typed_operator_policy.py`'s `TypedOperatorPolicyScorer` -- not
TwoTower -- as the issue's actual training target. This slice takes the
first real step onto that target, and honestly narrows what's reachable
there.

* New module `src/slm_training/harnesses/experiments/argument_preference.py`:
  * `build_argument_preference_example` renders one row into a
    `TypedOperatorArgumentPreferenceExampleV1` -- but **only for
    `pronoun_focus_followup`**. `OperatorPolicyInputV1.action_rows` is built
    only from `legal_set.entries` (operator-registry actions;
    `build_operator_policy_input` in
    `slm_training.models.operator_policy_view`). The other six named
    patterns' `chosen_action`/`rejected_action` are history controls
    (`undo`, `redo:<id>`, `checkout:<id>`) or `merge:<pair>` -- none of
    which has a row in that space at all. `pronoun_focus_followup` is the
    one pattern whose chosen and rejected actions are the *same* operator
    with a different bound argument for the *same* slot: a genuine
    argument-selection preference the typed policy's
    `argument_head` (`CandidateScoringHead`) can score. Wiring the other
    six patterns would require a real scope decision about what an "action
    row" even means for a control action -- left open here, not guessed at.
  * `typed_operator_argument_preference_loss` is a Bradley-Terry pairwise
    margin, `-log_sigmoid(chosen_logit - rejected_logit)`, over the two
    candidates' `CandidateScoringHead` logits for the differing slot.
    Surrogate preference loss, not textbook DPO -- the same honesty note
    `scripts/train_preference.py`'s own `dpo_loss` already carries for the
    generic TwoTower path.
  * `train_typed_operator_argument_preference` mirrors
    `train_typed_operator_policy`'s own matched full-batch schedule exactly,
    over this new loss.
* **Real, structural finding, not a bug:** training on the actual
  `pronoun_focus_followup` fixture
  (`tests/test_dsl/test_replay_preference.py`'s `_pronoun_focus_fixture`)
  provably cannot move the loss. Both sibling refs share every field
  `ReferenceModelViewV1` exposes (`ref_kind=VALUE`, `value_type=
  openui.string`, no parent, no position) -- they differ only by
  `semantic_fingerprint`, which `FORBIDDEN_FIELD_NAMES` in
  `slm_training.models.operator_policy_view` deliberately strips from every
  model input as anti-identity-leakage. `OperatorFeatureEncoder.
  _reference_embeddings` is a pure function of exactly those allowed
  fields, with no row-index feature, so two feature-identical candidates
  get byte-identical embeddings through the shared-weight
  `CandidateScoringHead` regardless of any parameter update: the loss sits
  at `-log_sigmoid(0) = ln(2)` structurally, provably, for as many steps as
  you run it. `test_training_cannot_move_the_loss_when_candidates_are_feature_identical`
  proves this is exact (`pytest.approx`, not "roughly unchanged"); a
  second test with a synthetic, feature-*distinguishable* pair (differing
  `relative_position`) proves the loss function and gradient flow
  themselves work correctly (`test_training_reduces_the_pairwise_loss_when_candidates_differ`)
  -- this is a property of *this fixture's* candidates, not of the
  mechanism.
  * Consequence for the open training/measurement scope above: any real
    corpus of `pronoun_focus_followup` rows will only carry a learnable
    argument-preference signal for pairs whose candidates differ in
    `ref_kind`/`value_type`/`compiler_facts`/`has_parent`/
    `relative_position`/selector fields -- feature-identical siblings
    (plausibly common for repeated same-type VALUE refs, exactly the
    minimal case this fixture represents) are structurally unlearnable
    signal by this scorer's own anti-leakage design, not a data-quantity
    problem. Worth surfacing before anyone builds a real corpus and is
    puzzled why training on it plateaus.
* No harness code outside the new module changed; `typed_operator_policy.py`
  itself is unchanged (only consumed, not modified).
* `harness.experiments.argument_preference` registered fresh (`v1`, initial
  registration) in `src/slm_training/resources/versions.json`.

## Reproducibility

```bash
NODE_OPTIONS= pytest -q tests/test_dsl/test_replay_preference.py tests/test_dsl/test_operator_merge.py tests/test_dsl/test_operator_conversation.py tests/test_harnesses/preference/test_replay_pairs.py tests/test_scripts/test_build_replay_preference_pairs.py tests/test_harnesses/experiments/test_argument_preference.py tests/test_harnesses/experiments/test_typed_operator_policy.py tests/test_evals/test_advanced_operator_disposition.py tests/test_scripts/test_validate_advanced_operator_disposition.py
```

Result (fifth-slice PR, real run in a fresh `.venv` -- Python 3.12, `pip install -e ".[dev,grammar]"`, plus `NODE_OPTIONS= npm ci` in `src/apps/openui_bridge` for the G2/G8 schema-oracle gates the pack authority requires; the ambient `--import tsx` `NODE_OPTIONS` is rejected by this Node 22 build both for `npm ci` and for `pytest`, unrelated to this change): `61 passed`. Also verified: `ruff check` clean on every changed file; `python -m scripts.verify_version_stamps --check --base origin/claude/great-dirac-v82ph9` -- `ok (2 component(s) touched)`; `python -m scripts.repo_policy` -- `ok`; `python -m scripts.verify_decode_invariants` -- clean.

Result (sixth-slice PR #1124, real run in a fresh `.venv-dsh510` -- Python 3.12, `pip install -e ".[dev,grammar]"`, plus `env -u NODE_OPTIONS npm ci` in `src/apps/openui_bridge` -- the ambient `NODE_OPTIONS="--import tsx" --max-old-space-size=8192` is rejected outright by Node for both `npm ci` and `pytest` in this environment, so it has to be unset, not just locally overridden, unlike the fifth slice's note above): `69 passed` against `main` HEAD `5f94b92` (includes the fifth slice, already merged). Also verified: `ruff check` clean on both new files; `python -m scripts.verify_version_stamps --check --base origin/main` -- `ok (1 component(s) touched)`; `python -m scripts.repo_policy` -- `ok`; `python -m scripts.verify_decode_invariants` -- clean. No training run in this slice; `outputs/` untouched.

Result (seventh-slice PR #1125, real run in a fresh `.venv-dsh510`, same environment recipe as the sixth slice above, stacked on top of PR #1124 which was still unmerged when this slice started): `71 passed` (69 from the sixth slice + 2 new). Also verified: `ruff check` clean on both new files; `python -m scripts.verify_version_stamps --check --base origin/main` -- `ok (1 component(s) touched)`; `python -m scripts.repo_policy` -- `ok`. Plus the real training run described above (SFT checkpoint + demo pairs + preference-training pass, both commands well under `MAX_RUN_MINUTES=3`); its `outputs/runs/replay_pref_sft_ckpt/` and `outputs/runs/replay_pref_dpo/` are not committed (`outputs/` is gitignored) per this repo's checked-not-committed convention for scratch run artifacts.

Result (eighth-slice PR #1126, real run in a fresh `.venv-dsh510`, same environment recipe as above, stacked on top of PR #1125 which was still unmerged when this slice started): `72 passed` (71 from the seventh slice + 1 new). Also verified: `ruff check` clean; `python -m scripts.verify_version_stamps --check --base origin/main` -- `ok (1 component(s) touched)`; `python -m scripts.repo_policy` -- `ok`; `python -m scripts.verify_decode_invariants` -- clean. Plus the reran training chain described above; `outputs/runs/replay_pref_sft_ckpt2/` and `outputs/runs/replay_pref_dpo2/` are not committed (`outputs/` is gitignored).

Result (this PR, ninth slice, real run in the same `.venv-dsh510`, stacked on top of PR #1126 which was still unmerged when this slice started): `92 passed` (72 from the prior slices + 6 new in `test_argument_preference.py`, plus `test_typed_operator_policy.py`'s own 14 pre-existing tests now included in this suite's reproduction command for the first time since this slice touches that module's consumer surface). Also verified: `ruff check` clean on both new files; `python -m scripts.verify_version_stamps --check --base origin/main` -- `ok (2 component(s) touched)`; `python -m scripts.repo_policy` -- `ok`; `python -m scripts.verify_decode_invariants` -- clean. No end-to-end training run in this slice (the `train_typed_operator_argument_preference` calls are inside the test suite itself, proving the mechanism works on a synthetic distinguishable pair and correctly plateaus on the real fixture's feature-identical pair -- not a separate `outputs/`-writing run).
