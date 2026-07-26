# DSH3-31 replay-grounded operator-decode failure prediction

SLM-406 asks whether an operator decode's eventual replay-grounded failure
can be predicted early from already-existing `DecodeStats` counters, and
whether a compact subset of those counters beats an entropy/margin-only
baseline and a context-free frequency baseline. This module is
shadow-analysis only: it adds a new, self-contained measurement/
classification library over `DecodeStats`-shaped counters. It does **not**
change any production decode/abort behavior, does not touch
`src/slm_training/models/decode_stats.py`,
`src/slm_training/harnesses/experiments/typed_operator_policy.py`, or
`src/slm_training/models/operator_policy_objective.py`, and does not wire
anything into a live decode loop, compiler lattice, or constrained decode
path.

`src/slm_training/evals/cap2_operator_policy_rebase.py`'s own
forward-reference inventory (its `item(...)` entry for
`models.decode_stats.DecodeStats`) already named
`compiler_lattice_false_hard_eliminations`,
`compiler_lattice_selector_regret`,
`compiler_lattice_invalid_selected_over_valid`, and `constrained_dead_ends`
(with related counters) as "reusable as DSH3-31 features" and stated plainly
that "no operator-specific replay-outcome label ... exists yet on top of
them." This module closes that forward reference: it adds the label
taxonomy and the feature-arm evaluation harness, and reuses the counters
exactly as they already exist.

## Module placement

The new module lives at `src/slm_training/evals/operator_failure_prediction.py`
(not `src/slm_training/models/`). DSH3-29 and DSH3-30 (already on this
branch) put their modules under `models/` because they wrap model/head
classes (`TernaryECOCHead`, `OperatorActionViewV1` scoring). DSH3-31 is pure
measurement and classification over `DecodeStats`-shaped counters -- there is
no model or head here, only a taxonomy, a deterministic proxy scorer, and an
AUROC/AUPRC/calibration report, which is structurally the same shape as other
`src/slm_training/evals/*.py` report modules (e.g.
`cap2_operator_policy_rebase.py`'s `Cap2CapabilityDispositionV1`). The
matching test lives at `tests/test_evals/test_operator_failure_prediction.py`,
matching the issue's own literal `pytest` invocation.

## Module

`src/slm_training/evals/operator_failure_prediction.py` adds:

* `ReplayOutcomeLabel` -- the 7 labels the issue names: `ILLEGAL_SELECTION`,
  `EXECUTOR_REJECTION`, `REPLAY_FAILURE`, `SEMANTIC_MISS`, `DEAD_END`,
  `TIMEOUT`, `SAFE_DEFER`. No "success" sentinel is included by design.
* `classify_replay_outcome(*, timed_out=False, illegal_selection=False,
  executor_rejected=False, replay_failed=False, dead_end=False,
  semantic_miss=False, credited_defer=False)` -- a strict-precedence
  classifier mirroring `harnesses.model_build.decode_outcome
  .classify_decode_outcome`'s "strict precedence over one struct" shape.
  Precedence: `TIMEOUT > ILLEGAL_SELECTION > EXECUTOR_REJECTION >
  REPLAY_FAILURE > DEAD_END > SEMANTIC_MISS > SAFE_DEFER`. A timeout is
  checked first because it can mask any other in-flight signal; illegal
  selection outranks executor/replay because it is detectable earlier in the
  pipeline (at selection time, before execution/replay ever runs); a dead
  end outranks a semantic miss because it is a harder, structural failure;
  safe DEFER is the fallback only when nothing else fired and the defer was
  credited (mirroring `DeferFallbackAttributionV1.credited_defer`'s
  route-plus-fallback-success concept). Calling with no signal and
  `credited_defer=False` raises `ValueError` -- a genuinely successful
  decode is out of scope for this 7-label failure/defer taxonomy.
* `PREREGISTERED_COMPACT_FEATURES` -- the exact 5-name compact feature subset
  DSH3-31's hypothesis is about:
  `compiler_lattice_false_hard_eliminations`,
  `compiler_lattice_selector_regret`,
  `compiler_lattice_invalid_selected_over_valid`, `constrained_dead_ends`,
  `constrained_last_legal_candidates`.
* `ALL_COUNTER_FEATURES` -- every numeric (`int`/`float`) field on
  `DecodeStats`, discovered via `dataclasses.fields` introspection (134
  fields today) rather than a hand-maintained list, so it can never silently
  drift from the real dataclass.
* `extract_features(stats, *, feature_names)` -- pulls named fields off a
  `DecodeStats` instance via `getattr` into a flat float dict.
* `score_from_features(features, feature_names)` -- a fixed, documented,
  equal-weighted sum of the named features (every named `DecodeStats`
  counter here is already a "bad is high" counter, so a plain sum is a
  well-defined, arm-comparable risk score). This is a fixture-scale
  deterministic proxy, never a trained model (same honesty framing as
  `operator_hard_negative_training.score_action_view`). An empty
  `feature_names` (the `context_free_baseline` arm) always scores `0.0`.
* `auroc(scored)` -- pure-Python rank-based AUROC over `(score,
  is_positive)` pairs via the standard Mann-Whitney U / average-rank
  formula (ties resolved to the tied group's average rank, so an
  all-identical-score set gives exactly `0.5`). Returns `None` when all
  labels are one class.
* `auprc(scored)` -- precision-recall AUC via a descending-score threshold
  sweep (ties broken by input order) with plain trapezoidal integration over
  recall, starting from the conventional `(recall=0, precision=1)` point.
  Returns `None` when there are no positives.
* `assert_group_disjoint_split(train_group_ids, held_out_group_ids)` -- a
  small, directly testable set-intersection check enforcing "group split by
  request/target cluster and checkpoint to prevent trace leakage"; raises
  `ValueError` on any overlap.
* `OperatorFailurePredictionArmResultV1` / `OperatorFailurePredictionReportV1`
  (`to_dict()`) and `evaluate_failure_prediction_arms(rows, arms)` -- scores
  every named feature arm over the same fixed `(features, label)` rows,
  treating every label except `SAFE_DEFER` as "positive" (failure), and
  reports AUROC, AUPRC, and calibration error (via the reused
  `slm_training.evals.judge_independence.calibration_error`, fed a min-max
  normalized `[0, 1]` "confidence of failure" per row) per arm. Picks the
  best arm by AUROC, ties broken by arm name.

## 1st result (2026-07-26)

Fixture: 18 hand-built synthetic `(features, ReplayOutcomeLabel)` rows -- 2
rows each for `TIMEOUT`, `ILLEGAL_SELECTION`, `EXECUTOR_REJECTION`,
`REPLAY_FAILURE`, `DEAD_END`, `SEMANTIC_MISS` (12 replay-grounded failures
total, all 7 labels represented once `SAFE_DEFER` is added), plus 6
`SAFE_DEFER` rows. Failure rows carry large
`compiler_lattice_false_hard_eliminations` /
`compiler_lattice_invalid_selected_over_valid` / `constrained_dead_ends` /
`constrained_last_legal_candidates` counters and a `compiler_lattice_selector
_regret` deliberately overlapping the SAFE_DEFER rows' own regret range (both
span roughly 0.40-0.60), by construction -- so the compact/all-counter arms'
signal comes from the non-regret counters, not from regret alone. Report:
[`dsh3-31-operator-failure-prediction-2026-07-26/report.json`](dsh3-31-operator-failure-prediction-2026-07-26/report.json),
[`summary.md`](dsh3-31-operator-failure-prediction-2026-07-26/summary.md).
Reproduce: `PYTHONPATH=src:. python scripts/run_dsh3_31_operator_failure_prediction.py --output-dir docs/design/dsh3-31-operator-failure-prediction-2026-07-26`
(runs in well under a second, far under `MAX_RUN_MINUTES`).

| Arm | # features | AUROC | AUPRC | Calibration error | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all_counters` | 134 | 1.000 | 1.000 | 0.157 | 18 |
| `compact_subset` | 5 | 1.000 | 1.000 | 0.174 | 18 |
| `entropy_margin_baseline` | 1 | 0.451 | 0.622 | 0.336 | 18 |
| `context_free_baseline` | 0 | 0.500 | 1.000 | 0.167 | 18 |

`compact_subset`'s AUROC (1.000) is strictly above
`entropy_margin_baseline`'s (0.451) on this fixture -- a compact 5-counter
subset separates replay-grounded failures from safe defers perfectly here,
while `compiler_lattice_selector_regret` alone, with its deliberately
overlapping range across both classes, does not. `all_counters` ties
`compact_subset`'s AUROC (unsurprising: it is a superset that includes the
same 5 counters, so it inherits the same by-construction separation) and
wins the tie-break by AUROC (equal scores are broken by arm name only when
AUROC itself ties -- `all_counters` sorts before `compact_subset`
alphabetically). `context_free_baseline`'s constant zero score is
undiscriminative by construction (AUROC exactly 0.5, the documented
tie-handling degenerate value). See
`tests/test_evals/test_operator_failure_prediction.py::test_compact_subset_strictly_beats_entropy_margin_baseline_on_fixture`
for the isolated, deterministic 6-row version of this same mechanism used as
the unit-test proof.

**Decision: `fixture-scale-positive-signal`.** This is real, deterministic
evidence that a compact `DecodeStats` compiler-lattice/constrained-decode
counter subset can predict a by-construction-separable synthetic
replay-grounded-failure/safe-defer split far better than a
selector-regret-only margin baseline -- but it is fixture-scale mechanism
evidence on one small, hand-built synthetic row set, not a claim about real
held-out generalization, real checkpoint cross-generalization, or a
production early-abort deployment decision. No time-indexed "earliest decode
fraction reaching target precision" metric is computed -- that requires real
time-indexed decode traces (a stream of per-step `DecodeStats` snapshots with
a known eventual outcome) which do not exist yet in this repo; this is
explicitly out of scope here, not approximated. DSH3-31's own stop rule ("if
no feature set beats entropy/margin within confidence bounds, reject
telemetry-based early abort") requires real decode-trace-scale evidence this
fixture cannot provide, so this result does **not** by itself justify wiring
any telemetry-based early-abort mechanism into a production decode loop. The
re-test trigger is once real per-decision `DecodeStats` rows with
replay-grounded outcome labels (illegal selection / executor rejection /
replay failure / semantic miss / dead end / timeout / safe DEFER, established
by real replay evidence, not a synthetic construction) exist at meaningful
volume -- only then should this ablation be re-run against real traces, with
group-disjoint train/held-out splits enforced via
`assert_group_disjoint_split`, and confidence bounds computed before any
production decision is made.

## Scope

This is a unit-tested taxonomy/classifier/scoring/report contract plus one
small synthetic-fixture script run, not a train, eval, matrix, checkpoint, or
ship claim. `decode_stats.py`, `typed_operator_policy.py`, and
`operator_policy_objective.py` are untouched; this module does not plug into
any decode loop or training pipeline.
