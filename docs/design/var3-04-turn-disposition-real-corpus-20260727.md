# VAR3-04 (SLM-433): real argument-bound corpus for TurnDispositionHead training

Date: 2026-07-27
Status: implemented; fixture/small-scale wiring evidence only
Scope: `scripts/run_slm433_04_turn_disposition_real_corpus.py` (new). No
change to `data/flow/turn_disposition_corpus.py`,
`harnesses/experiments/turn_disposition_training.py`,
`harnesses/train_data/operator_corpus.py`, or any scoring function --
this issue only wires an existing, unmodified pipeline to a richer real
source.
Honesty: **fixture_or_scratch / wiring, not a capability or ship claim.** No
promotion, gate change, or production-readiness claim is made anywhere in
this document.

## Decision

VAR3-03 (`docs/design/var3-03-turn-disposition-corpus-training-20260727.md`)
trained the first real `TurnDispositionClassifierV1` end to end but
reported `no_difference`: its own "What a real capability run would
require" section named the reason and the fix explicitly --

> A corpus whose ambiguous turns carry genuine per-row distinguishing
> content (e.g. operators with argument slots binding to different
> reference-table candidates per turn, so the sanitized
> `OperatorPolicyInputV1` view actually varies row to row) is required
> before this comparison can support a directional capability claim in
> either direction.

This issue builds exactly that corpus -- not with new fixture operators,
but by pointing VAR3-03's own, unmodified training/evaluation path at
`harnesses.train_data.operator_corpus.build_symbolic_operator_corpus`
(DSH3-24/SLM-399), the same real-argument-slot pipeline
`run_dsh3_28_typed_operator_policy.py` already uses in production runs of
this repo. That function applies the real, production `openui` local
operator library (genuine `node`/`role`/`value` argument slots, from
`dsl.operators.local`) to real admitted train-root documents
(`openui_verified_v1`) and a disjoint held-out record set
(`e763_symbol_only_eval_r2_20260722/suites/held_out`) -- never a toy
zero-argument competing library.

## What changed vs. VAR3-03, and what did not

* **Unchanged:** `build_turn_disposition_rows`, `corpus_action_frequency`,
  `assert_leakage_safe_split`, `TurnDispositionClassifierV1`,
  `train_turn_disposition_head`, `evaluate_turn_disposition_arms`, and
  `score_disposition_predictions`. Every one of these is called exactly as
  VAR3-03 called them, with the identical two-pass frequency-then-train
  structure (build with an empty frequency table first, since the
  frequency statistic itself can only come from already-accepted actions;
  rebuild train rows and derive dev rows from the train-only statistic
  second).
* **Changed:** the *source* of the `(ConversationTraceV1,
  CollapsedInstructionV1, authority_resolver)` triples fed into
  `build_turn_disposition_rows`. VAR3-03 used a hand-written, zero-argument,
  two-operator fixture library. This issue uses
  `build_symbolic_operator_corpus`'s real `on_collapsed_trace` callback
  against real admitted documents -- the same real per-root cost the
  DSH3-24/DSH3-28 pipeline already pays elsewhere in this repo.

## Corpus admission: a real, honestly-reported minority is skipped

Not every admitted-document root's 2-turn extension survives replay +
symbolic collapse under this repo's real operator library and pack static
oracle. `_admit_raw_traces` (new, in the script) tries each candidate root
in deterministic sorted-id order, keeps the real
`(trace, collapse, authority_resolver)` triple for every root whose
`build_symbolic_operator_corpus` call succeeds, and skips (never retries or
patches) any root whose replay/collapse raises -- recording the skipped
root id rather than silently absorbing it. This is necessary because
`build_symbolic_operator_corpus`'s own multi-root batch call fails closed
on the *first* failing root in the batch (by design -- "build only after
every generated transition and trace replays exactly"), and this repo's own
sorted-by-id root order happens to put one such root first. The measured
run below admits 8 of 8 requested train roots and 4 of 4 requested dev
roots, skipping 2 candidate train roots (both the same duplicated source
record id, `07e122e7030b65a8_scope`) along the way -- see
`corpus.train_roots_skipped` in the committed JSON.

`max_combinations_per_operator` is set to **32** (not VAR3-03's 64, and not
the 512 several other real-corpus scripts in this repo use for `_run_arm`
five-family matrices): a real per-root timing probe run before this script
was finalized measured ~9-25s per root at `max_combinations_per_operator=
512` (roughly 10 roots needed for train+dev already approaches
`MAX_RUN_MINUTES`), versus ~2s per root at 8-32. 32 keeps the full script
(admission + two-pass row derivation + 30-step training + both evaluations)
at **~90 wall seconds**, comfortably inside the 3-minute cap, while still
admitting genuine multi-candidate legal sets (this run's train corpus
carries 5 distinct `(ref_kind, value_type)` reference-row signatures and 6
distinct action operator ids -- see below).

## Real evidence this is not VAR3-03's degenerate case

VAR3-03's `no_difference` outcome was explicitly attributed to a structural
fact: its competing library's every non-forced row shared one feature
vector and one gold label, so a classifier could not do better than a
constant policy. This run's committed JSON reports, as a direct, measured
fact (not an assumption): `corpus.distinct_reference_signatures = 5` and
`corpus.distinct_action_operator_ids = 6` across the 16 real train rows --
genuine per-row variety exists in this corpus, unlike VAR3-03's.

## Result: `held_out_improvement`, honestly bounded

| Arm | Split | composite_penalized_error_rate | wrong_op_rate | abstention_rate |
| --- | --- | ---: | ---: | ---: |
| `disposition_off` / `derived_only` | held-out (n=8) | 0.5625 | 0.750 | 0.000 |
| `disposition_trained` | held-out (n=8) | 0.4375 | 0.375 | 0.625 |
| `disposition_off` / `derived_only` | train (n=16) | 0.375 | 0.500 | 0.000 |
| `disposition_trained` | train (n=16) | 0.125 | 0.000 | 0.500 |

`disposition_trained`'s held-out `composite_penalized_error_rate` (0.4375)
is lower than both always-emit baselines' (0.5625) on this real,
argument-bound corpus -- `disposition = "held_out_improvement"`. This is
reported exactly as measured, with its scale made explicit rather than
rounded away: **n_dev = 8 rows, from 4 held-out root traces.** `claim_class`
stays `"wiring"`, not `"capability"`: this is one seed, one small held-out
split, no statistical test, no cross-seed replication, and the trained
arm's held-out behavior leans heavily on abstaining (`clarify`) rather than
on a more accurate `emit` (`abstention_rate` 0.625 vs. the baselines'
0.0) -- consistent with, but not proof of, the classifier learning that
uncertain rows are better deferred than guessed. `disposition_off` and
`derived_only` again coincide exactly, for the same structural reason
VAR3-03 documented: this corpus generator only emits AST-mutating steps
(no `out_of_scope`/`answer` turns), and every forced-singleton row's only
live candidate is also the naive always-emit top pick.

## Non-goals

No SFT/production training run, no cross-seed or cross-split replication,
no statistical significance test, no promotion or ship-gate claim, no
change to `derive_turn_disposition`'s I1 precedence order or
`dsl/operators/legal_set.py`. A single `held_out_improvement` result at
n=8 is directional evidence that a richer corpus *can* produce a
non-degenerate held-out signal -- it is not evidence that this signal
generalizes, replicates across seeds, or would survive a larger held-out
split.

## Tests

`pytest -q tests/test_scripts/test_run_slm433_04_turn_disposition_real_corpus.py`
(new, 1 test): builds a tiny 2-root real corpus via
`build_symbolic_operator_corpus` against `openui_verified_v1` and asserts
its reference rows actually carry more than one distinct
`(ref_kind, value_type)` signature -- the structural property this whole
issue depends on, checked directly rather than only inferred from a full
training run's side effects.

## Required artifacts

This JSON/Markdown pair
(`docs/design/var3-04-turn-disposition-real-corpus-20260727.{json,md}`).
No separate `quality_report.json`/`rejected.jsonl` triplet: like VAR3-03,
this harness does not use the `scripts.build_train_data` pipeline that
convention governs; the analogous corpus-quality counts (admitted/skipped
roots, row/rejection counts per split, forced-singleton counts) are
reported inline in the committed JSON's `corpus` block instead.

## Acceptance criteria

- [x] The corpus source is the real, production `openui` local operator
  library applied to real admitted documents -- never a hand-written
  zero-argument fixture.
- [x] `build_turn_disposition_rows`, the classifier, its training loop, and
  `score_disposition_predictions` are reused completely unmodified.
- [x] The held-out split is trace-level leakage-safe
  (`assert_leakage_safe_split`, and structurally guaranteed here since
  train/dev roots come from wholly disjoint source-record files).
- [x] All three arms (`disposition_off`, `derived_only`,
  `disposition_trained`) are reported on the identical held-out split at
  matched budget.
- [x] Both `wrong_op_rate` and `abstention_rate` are reported for every
  arm, never blended into a single hidden number.
- [x] Skipped candidate roots are recorded (`corpus.*_roots_skipped`),
  never silently dropped.
- [x] No promotion or ship claim is made (`claim_class: "wiring"`,
  `honesty: "fixture_or_scratch"`), regardless of the directional result.

## Falsification / stop rule

If a future rerun at a larger held-out size or across multiple seeds shows
`disposition_trained` regress to `no_difference` or `held_out_regression`,
that is the legitimate outcome to report -- this run's single
`held_out_improvement` at n=8 does not license skipping straight to a
capability or promotion claim. Per VAR3-03's own stop rule, no further
trained-head complexity should be added on top of this single small-n
result before a larger, cross-seed held-out measurement exists.
