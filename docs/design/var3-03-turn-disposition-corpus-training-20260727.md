# VAR3-03 (SLM-433): a real corpus for TurnDispositionHead + held-out measurement

Date: 2026-07-27
Status: implemented; fixture/small-scale wiring evidence only
Scope: `data/flow/turn_disposition_corpus.py` (new), `harnesses/experiments/
turn_disposition_training.py` (new), `scripts/run_slm433_turn_disposition_training.py`
(new).
Honesty: **fixture_or_scratch / wiring, not a capability or ship claim.** No
promotion, gate change, or production-readiness claim is made anywhere in
this document.

## Decision

VAR3-02 (SLM-430) shipped `TurnDispositionLabel`, `turn_disposition_losses`,
and `TurnDispositionReportV1` in `models/turn_disposition_head.py`, but its
own design doc stated explicitly: "No corpus builder or training run is
added by this issue -- the head's loss and report are the training-ready
surface a future corpus-labeling issue would target, tested here against
synthetic tensors only." This issue is that future issue: it builds the
first real (non-synthetic-label) corpus of turn dispositions and trains the
first actual classifier over `turn_disposition_losses`, then measures
whether it beats two matched always-emit baselines on a trace-level
held-out split.

## Corpus: real corpus, real derivation, never a hand-labeled shortcut

`data/flow/turn_disposition_corpus.build_turn_disposition_rows` re-enumerates
the live legal set fresh at every step of an executed conversation trace
(mirroring `data/flow/operator_policy_corpus.build_operator_policy_rows`'s
own re-enumeration discipline) and derives each row's gold
`TurnDispositionTargetV1` through the exact same
`dsl.operators.turn_disposition.derive_turn_disposition` the REPL variant's
own decode loop consults -- never a hand-authored label.

* **`out_of_scope` is structurally unreachable in this corpus.** Every row
  starts from a step whose recorded operator application was already proven
  live against a freshly re-enumerated legal set (the same
  `ACCEPTED_ACTION_NOT_LIVE` rejection `build_operator_policy_rows` uses); a
  legal set that admits an accepted action can never be empty.
  `build_turn_disposition_rows` asserts this invariant with a hard failure
  rather than silently trusting it, and
  `test_accepted_action_absent_from_a_re_enumerated_library_is_rejected_not_mislabeled`
  proves the only way history could ever disagree with a fresh legal set --
  rejection, never a coerced disposition.
* **`answer` never appears** either: this corpus generator only emits
  AST-mutating operator applications, so `is_state_query` is always `False`
  -- an explicit, honestly-stated scope limit.
* **`clarify` margin labels come from a real corpus statistic**
  (`corpus_action_frequency`): each live candidate's frequency as the
  historically accepted choice elsewhere in the *train* split only, never a
  synthetic score manufactured per example to force an outcome.
* **Leakage-safe split**: `assert_leakage_safe_split` fails closed if any
  source trace (`row.trace_id`, the whole collapsed conversation) appears in
  more than one split -- turns within one trace share state history, so the
  leakage-safe unit is the trace, never the individual row. Mirrors
  `operator_policy_corpus.freeze_collapse_negative_ablation`'s own
  split-disjointness check.

## Trained classifier

`harnesses/experiments/turn_disposition_training.TurnDispositionClassifierV1`
is the first actual `nn.Module` trained against `turn_disposition_losses`: a
linear head over one pooled `OperatorFeatureEncoder` (DSH3-23) action
embedding -- reused, not reimplemented. `train_turn_disposition_head` runs
full-batch Adam, matching `harnesses.experiments.typed_operator_policy
.train_typed_operator_policy`'s schedule discipline; forced-singleton rows
are masked (`mask_forced=True`, `turn_disposition_losses`'s own default), so
a corpus made entirely of forced rows trains nothing --
`test_masked_forced_singleton_examples_train_to_a_flat_zero_loss` proves the
mask is wired end to end.

**I1 bypass at evaluation, not just training.** A COMPLETE-coverage legal
set with exactly one candidate is a deterministic bypass that outranks any
learned score (this repository's non-negotiable I1 ordering). The trained
classifier is therefore *never consulted* for a forced-singleton example,
even at evaluation time --
`evaluate_turn_disposition_arms`'s `_trained_row` short-circuits to `emit`
for every forced row before ever calling the classifier.
`test_forced_singleton_examples_never_consult_the_classifier` proves this
with a classifier double that raises `AssertionError` if `forward` is ever
invoked, mirroring this repository's own
`test_out_of_scope_never_consults_score_provider` convention.

## Three-arm held-out comparison

`evaluate_turn_disposition_arms` scores `disposition_off` / `derived_only` /
`disposition_trained` on one identical example set, all through the
unchanged `evals.cap2_operator.score_disposition_predictions`:

* `disposition_off` -- VAR3-02's own always-emit baseline.
* `derived_only` -- the free `out_of_scope`/`answer`/forced-singleton rules
  only; `clarify` never fires.
* `disposition_trained` -- this issue's trained classifier (I1-bypassed on
  forced rows, as above).

**`disposition_off` and `derived_only` are numerically identical on this
corpus, and this is reported honestly rather than manufactured apart.** The
corpus has no `out_of_scope`/`answer` turns by construction, and a
forced-singleton row's only live candidate is also the naive always-emit top
pick, so the two named baselines coincide here. The informative comparison
in this corpus is emit-vs-clarify disposition awareness
(`disposition_trained` against either always-emit baseline), not
forced-singleton awareness.

## Held-out run and honest disposition

`scripts/run_slm433_turn_disposition_training.py` builds 8 train traces / 4
dev traces (alternating an always-ambiguous two-operator library with a
sequential-precondition library that forces a singleton at every step, so
the corpus has genuine structural variety rather than one homogeneous case
shape), trains the classifier for 30 full-batch steps, and reports the
three arms on both splits. Committed result:
`docs/design/var3-03-turn-disposition-corpus-training-20260727.json`.

Result: **`no_difference`** on the held-out split -- `disposition_trained`'s
`composite_penalized_error_rate` scores identically to the always-emit
baselines'. This is an expected, legitimate outcome at this scale, not
reframed as inconclusive: this fixture's two competing operators carry no
per-row distinguishing argument content (no argument slots at all), so
every non-forced row shares one feature vector and one gold label -- there
is no genuine per-row disambiguation signal available for the classifier to
learn beyond a constant policy. `claim_class` is accordingly `"wiring"`,
not `"capability"`: this run demonstrates that the corpus builder, the
trained classifier, the I1 bypass, and the three-arm scoring are wired
correctly end to end on real (not synthetic-label) data, but it is not
evidence of a held-out disposition benefit either way.

## What a real capability run would require

A corpus whose ambiguous turns carry genuine per-row distinguishing content
(e.g. operators with argument slots binding to different reference-table
candidates per turn, so the sanitized `OperatorPolicyInputV1` view actually
varies row to row) is required before this comparison can support a
directional capability claim in either direction. That corpus-richness work
is out of scope for this issue.

## Tests

`pytest -q tests/test_models/test_turn_disposition_head.py
tests/test_evals/test_cap2_operator_turn_disposition.py
tests/test_dsl/test_turn_disposition.py` (existing, pass unmodified) plus
new: `tests/test_data/flow/test_turn_disposition_corpus.py` (6 tests: forced
rows always derive to `emit`, real-frequency-driven margin skew,
rejection-not-mislabeling, leakage-safe split, frequency aggregation,
allowlisted-field validation) and `tests/test_harnesses/experiments/
test_turn_disposition_training.py` (4 tests: end-to-end three-arm training +
evaluation, masked-forced-loss regression, and the I1-bypass-at-evaluation
regression).

## Required artifacts

This JSON/Markdown pair. No separate `quality_report.json`/`rejected.jsonl`
triplet: this harness does not use the `scripts.build_train_data` pipeline
that convention governs; the analogous corpus-quality counts (row/rejection
counts per split, forced-singleton counts per split) are reported inline in
the committed JSON's `corpus` block instead.

## Acceptance criteria

- [x] `out_of_scope`/`answer` corpus labels are provably derived, never
  model-predicted (structurally unreachable / never emitted here).
- [x] The held-out split is trace-level leakage-safe
  (`assert_leakage_safe_split`, tested).
- [x] All three arms (`disposition_off`, `derived_only`,
  `disposition_trained`) are reported on the identical held-out split at
  matched budget.
- [x] Both `wrong_op_rate` and `abstention_rate` are reported for every arm,
  never blended into a single hidden number (`score_disposition_predictions`
  unchanged).
- [x] No promotion or ship claim is made regardless of outcome
  (`claim_class: "wiring"`, `honesty: "fixture_or_scratch"`).

## Falsification / stop rule

`disposition_trained` showed no held-out improvement over `derived_only` at
this scale -- reported as the legitimate negative (`no_difference`) result
it is. Per this issue's own stop rule, no further trained-head complexity is
added to this surface; a richer corpus (see above) is the honest
prerequisite for re-testing the hypothesis, not more model capacity on the
current one.

## Non-goals

No natural-language generation for `clarify`/`answer` turns. No change to
`derive_turn_disposition`'s I1 precedence order or `dsl/operators/legal_set.py`.
No promotion or ship-gate claim.
