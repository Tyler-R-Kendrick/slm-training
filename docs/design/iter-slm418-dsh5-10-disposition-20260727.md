# SLM-418 (DSH5-10) final disposition: replay-grounded preference signal from undo/redo/fork history

Date: 2026-07-27
Status: **final issue-level disposition — falsification close.**
Companion JSON: `docs/design/iter-slm418-dsh5-10-disposition-20260727.json`
(`operator_event_memory_report/v1`, real `version_stamp`).
Honesty: fixture-scale measured result, honestly labeled. Not a capability,
ship, or promotion claim. No checkpoint was created by this issue; no model
card update applies.

## Decision

**`no_held_out_benefit_at_fixture_scale_retain_dag_only`**

The issue's own falsification / stop rule is invoked, verbatim:

> If replay-grounded history yields no held-out benefit, retain the event
> DAG for runtime/evaluation only and do not add preference training
> complexity.

This is the issue's designed, legitimate negative close — not a failure.
The conversation event DAG (`ConversationTraceV1`) is retained as the sole
state authority for runtime and evaluation; the extraction machinery stays
as evaluated wiring; **no SFT/preference training complexity is added.**

## What the issue asked and what was delivered

SLM-418 asked whether exact undo/redo/checkout/fork outcomes can provide
useful preference supervision for ambiguous follow-ups without making
transcript text the artifact authority. Delivered across merged slices
(`docs/design/dsh5-10-replay-preference-rows.md`, v1–v7):

### Pattern coverage: 7 of 7 named patterns extracted and tested

`src/slm_training/dsl/operators/replay_preference.py` (v7):

| # | Pattern (`ReplayPreferenceRelation`) | Extraction | Replay-verified tests |
| --- | --- | --- | --- |
| 1 | `edit_then_undo` | turn-pair scan (`AST_EDIT` → `UNDO`) | ✅ |
| 2 | `undo_then_redo` | turn-pair scan (`UNDO` → `REDO`) | ✅ |
| 3 | `partial_rollback` | second+ consecutive `UNDO` | ✅ |
| 4 | `checkout_another_state` | same-branch `CHECKOUT_STATE` | ✅ |
| 5 | `fork_then_choose_one_branch` | cross-branch checkout over a `FORK` boundary | ✅ |
| 6 | `merge_success` | standalone `extract_merge_preference_row` on a verified merge | ✅ |
| 7 | `pronoun_focus_followup` | consecutive `AST_EDIT` with focus-set overlap and a legal sibling | ✅ |

Merge *conflict* is deliberately not a row (no successor state to replay
to); it is excluded from every ranking denominator **by construction**
(merge candidates only enter the legal set after
`merge_conversation_branches` confirms success), proven by
`test_merge_conflict_never_yields_a_preference_row`.

Every row shares one exact `input_state_id`, is checked against the exact
legal set at that state (`legal_set_fingerprint`), and independently
replays to its recorded `chosen_output_state_id` — the issue's first two
acceptance criteria.

### The ablation that measured no benefit (PR #1129, merged `57f5bdbb`)

Real run of `python -m scripts.run_replay_preference_context_view_ablation`
on the bounded deterministic synthetic corpus (never real user telemetry):

- **Corpus:** 8 sessions (6 train / 2 held-out), 40 rows, all 7 relations
  present; group-stable split (conversation variants stay in one split).
- **Grid:** the issue's own 5 context views × turn depths {1,2,4,8,16} =
  25 cells.
- **`undo_family_rate` = 0.9** (corpus composition: 90% of rows are the
  undo family, by rollback-chain design).
- **Held-out pairwise accuracy: 1.0 in all 25 cells, including the
  `current_state_only` baseline** → verdict **`no_benefit_fixture_scale`**.
  This is an honest **ceiling effect**: `is_history_control`, computable
  from the decision state with zero history, already perfectly separates
  chosen from rejected on the 4-pair held-out split; history context has
  nothing left to add at this scale.
- **Calibration proxy (Brier, not CAP-gated):** 0.00011
  (`current_state_only`) vs 0.00059 (`state_plus_recent_receipts`, depth 1),
  narrowing toward 0.00007 by depth 16 — reflects the two-parameter
  scorer's own confidence, not accuracy.
- **Structural grid (genuinely informative):** on `rollback_chain_8`
  `PARTIAL_ROLLBACK` rows, `state_plus_recent_receipts` (recency) shows
  most-recent `action_kind="undo"`, while `state_plus_retrieved_events`
  (ancestry) shows the forward edit chain (`"operator"`) — the two views
  are not interchangeable, even though the fixture corpus cannot show one
  out-predicting the other.

Re-verified in this session: `no_benefit_fixture_scale`,
`undo_family_rate=0.9`, 8 sessions / 40 rows, all 25 cells at 1.0
(see "Reproducibility").

### The adapter gap (design note, merged)

`docs/design/dsh5-10-policy-scorer-adapter-gap-20260727.md`: 6 of 7
relations choose history-control actions (`undo`, `redo:<state>`,
`checkout:<state>`, merge tokens) that **cannot be expressed as
`OperatorActionViewV1` rows** in `TypedOperatorPolicyExampleV1` at all;
only `pronoun_focus_followup` maps as-is. Training against the
DSH3-selected `TypedOperatorPolicyScorer` therefore requires an adapter
design (recommended: a separate history-control head, option B) — but the
fixture-scale measurement above shows no benefit that would justify
building it. Identified, not warranted: no evidence, no training.

## Acceptance criteria — final status

| Criterion | Status |
| --- | --- |
| Every valid preference pair shares one exact input state and independently replays | **Met** (per-relation replay tests) |
| State/event authority never depends on transcript reconstruction | **Met by construction** (join keys only; `LAST_THREE_TEXT_HISTORY` strips state ids so text history cannot reconstruct a different state than the DAG) |
| At least one receipt/event context improves held-out ambiguous follow-ups | **Not met at fixture scale** (baseline-ceilinged 1.0 everywhere) |
| Negative and no-effect results remain in evidence | **Met** (this disposition + the sixth-slice measured results) |

## Non-goals honored

- No transcript as event store or artifact authority — the
  `ConversationTraceV1` DAG remains the sole state authority throughout.
- No free-form model summaries as authority.
- No permanent semantic use of state ids (opaque join/evidence keys only).
- **No preference training complexity added** — no SFT/preference training
  run, no checkpoint, no `TypedOperatorPolicyScorer` or `ObjectiveView`
  wiring, no schema change to the frozen `OperatorPolicyInputV1`.

## Successor conditions (what would re-open preference training)

Per goal-drift guard I14, this closes the *approach*, never the *goal*.
Preference training on replay history re-opens when:

1. A **real, argument-bound corpus build** exists — the VAR3-04/05 pattern
   (`docs/design/var3-04-turn-disposition-real-corpus-20260727.md`):
   point the unmodified extraction/conversion pipeline at
   `build_symbolic_operator_corpus` real admitted documents instead of the
   8-session synthetic fixture, and measure against
   `current_state_only` / derived-only baselines.
2. A **powered held-out split** large enough that a state-only feature
   (`is_history_control`) does not already ceiling — the fixture's 4-pair
   held-out split is exhausted by it, which is exactly why this run is
   honest `no_benefit_fixture_scale` and not a powered claim either way.
3. If re-opened, prototype adapter-gap **option B** (a separate
   `HistoryControlPolicyInputV1` head) first; never extend the frozen
   `OperatorActionViewV1` schema unilaterally.

## Reproducibility

```bash
python -m scripts.run_replay_preference_context_view_ablation
NODE_OPTIONS= pytest -q tests/test_dsl/test_replay_preference.py \
  tests/test_harnesses/preference/test_operator_history_pairs.py \
  tests/test_evals/test_ambiguous_operator_followups.py
python -m scripts.verify_version_stamps --check
python -m scripts.repo_policy
```

Result (this session, on `slm-418-dsh5-10-disposition` branched from
`origin/main` `0fd83214`): ablation run re-produced
`no_benefit_fixture_scale` (8 sessions, 40 rows, `undo_family_rate=0.9`,
all 25 cells 1.0, best `state_plus_recent_receipts` depth 1 at 1.0 tied
with baseline). Tests: **50 passed**. `verify_version_stamps --check`,
`repo_policy`, and `git diff --check`: clean (see PR body).

Note: the task-brief claim of a merged demo pairs builder
(`scripts/build_replay_preference_pairs.py`, PRs #1125/#1127/#1128) does
not hold on `origin/main` — those commits live only on an unmerged branch
(`origin/claude/great-dirac-ni43oh`). The disposition does not depend on
them; every number above comes from merged main.
