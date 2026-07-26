# DSH5-10: replay-grounded preference rows from undo/redo history (SLM-418)

**Status:** partial slice, in progress.
**Claim class:** `wiring`.
**Honest verdict:** not yet dispositioned -- this PR delivers a scoped subset,
not the full issue.

SLM-418 asks whether exact undo, redo, checkout, and fork outcomes can
provide useful preference supervision for ambiguous follow-up instructions,
via: (1) a versioned preference-row schema over one exact input state, (2)
row extraction from seven verified conversation patterns, (3) matched
SFT/preference training against four context-view baselines, and (4)
held-out measurement of action/operator/argument/reference/branch accuracy,
calibration, and CAP0/CAP1/CAP2 retention.

## What this PR delivers

* `src/slm_training/dsl/operators/replay_preference.py`:
  * `OperatorReplayPreferenceRowV1` -- the versioned preference row the issue
    asks for: one exact `input_state_id`, `chosen_action`/`rejected_action`,
    the resulting `chosen_output_state_id`, a typed `semantic_relation`, a
    `correction_reason`, and the `legal_set_fingerprint` the row was checked
    against.
  * `extract_replay_preference_rows` -- scans a `ConversationTraceV1`'s
    turns for two of the issue's seven named patterns:
    **edit-then-undo** and **undo-then-redo**. Each emitted row's chosen and
    rejected actions are verified members of the exact legal set at the
    shared input state (`enumerate_operator_legal_set`, including the
    trace's own available `undo`/`redo:<state>` control actions) -- never
    inferred from transcript text, and never emitted unless an actual
    unchosen alternative existed in that legal set (per the issue's
    adversarial control: undo/redo is not asserted preferred by default).
  * `OperatorEventMemoryReportV1` -- counts of extracted rows by relation.
* Regression tests (`tests/test_dsl/test_replay_preference.py`) proving:
  the acceptance criterion "every valid preference pair shares one exact
  input state and independently replays" (`chosen_output_state_id` is
  exactly the state the conversation turn's own replay-verified transition
  produced); that no row is emitted for an edit-only trace with no history
  operation; and that both patterns extract exactly the row their trace
  supports.

## Explicitly out of scope for this PR

Per the issue's own scope, not attempted here:

* **Five of seven patterns**: partial rollback, checkout-another-state,
  fork-then-choose-one-branch, merge success/conflict, and pronoun/focus
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
The remaining scope is genuinely large (training + held-out evaluation
across a five-pattern, five-baseline, multi-metric matrix) and is left for
follow-on work rather than rushed to a false "Done." The issue should stay
open against the patterns and training/evaluation work enumerated above.

## Reproducibility

```bash
pytest -q tests/test_dsl/test_replay_preference.py tests/test_dsl/test_operator_conversation.py
```
