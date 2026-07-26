# DSH3-30 CONFLICT vs DIFFERENT_RESULT collapse hard-negative ablation

SLM-405 tests DSH3-24/SLM-399's own re-projection mechanism
(`slm_training.data.flow.operator_policy_corpus`) for a structural signal it
already carries but nothing yet measures: a `CollapsedHardNegativeV1` whose
outcome is `CONFLICT` means the adjacent-swap replay raised (the alternate
ordering was outright illegal at that state), while `DIFFERENT_RESULT` means
the swap replayed successfully but disagreed. Re-projected onto one step's
freshly re-enumerated live legal set as an `OperatorPolicyHardNegativeV1`,
this becomes concrete: a `CONFLICT` row's alternate operator is very likely
*not* among that step's live `action_rows` (`alternate_action_row is None`),
while a `DIFFERENT_RESULT` row's alternate operator *was* a legal, live
candidate (`alternate_action_row is not None`). This module makes that
distinction explicit, testable, and measurable across five matched
hard-negative training arms, without training anything and without touching
DSH3-28's frozen `typed_operator_policy_loss`.

This does not change legal-set enumeration, collapse detection, hard
negative re-projection, or any existing scorer's loss. It does not add a
real hard-negative-aware training loop. It is one new, self-contained module
plus a small synthetic-fixture script.

## Module

`src/slm_training/models/operator_hard_negative_training.py` adds:

* `HardNegativeArm` — the five matched arms: `none` (baseline: no
  hard-negative supervision at all), `conflict_only`,
  `different_result_only`, `equal_mix` (balances to
  `min(count_conflict, count_different_result)` of each type, deterministic
  under a fixed seed), and `curriculum_mix` (all `DIFFERENT_RESULT` rows
  before all `CONFLICT` rows — a scoped simplification of "easy negatives
  first," since there is no real iterative training loop here to stage a
  curriculum across steps; this only fixes a presentation order over one
  static row sequence).
* `select_hard_negative_rows(rows, arm, seed=0)` — filters/selects which
  rows contribute hard-negative supervision under `arm`. Rows with
  `hard_negative is None` are never selected by any arm.
* `score_action_view(action)` — a small, deterministic, label-free proxy
  score (`-cost - 0.1 * len(effect_signature)`) computed only from
  `OperatorActionViewV1`'s own sanitized fields, never from the
  outcome/target being predicted. This is a fixture-scale stand-in for a
  trained scorer, not a real model.
* `hard_negative_margin_loss(accepted_score, alternate_score, margin=1.0)` —
  a plain hinge loss; returns `None` when `alternate_score is None`, which is
  the load-bearing case for `CONFLICT` rows structurally lacking a live
  alternate.
* `evaluate_hard_negative_arm(rows, arm, seed=0, margin=1.0)` /
  `evaluate_hard_negative_arms(rows, arms=...)` — evaluate one or all arms
  over the same fixed row set and return a `HardNegativeArmResultV1` /
  `CollapseNegativeAblationReportV1` (`to_dict()`), reporting `total_rows`,
  `usable_rows` (a live alternate score is computable),
  `unusable_rows`, `negative_type_counts`, `ranking_correct`,
  `mean_margin_loss`, and `usable_negative_rate = usable_rows / total_rows`
  — the central "denser supervision per example" metric DSH3-30's
  hypothesis is about.

## 1st result (2026-07-26)

Fixture: 9 hand-built synthetic `OperatorPolicyRowV1` rows — 4
`DIFFERENT_RESULT` (each with a live alternate action row present in
`policy_input.action_rows`, accepted/alternate declared costs alternating
0.5/1.5 and 1.5/0.5 so the deterministic proxy scorer sees a mix of
"accepted was cheaper" and "accepted was pricier" rows rather than every row
tying identically), 2 `CONFLICT` (each with the alternate operator genuinely
absent from `action_rows`, matching how the real corpus builder produces a
swap partner that never re-enumerates as legal), and 3 no-hard-negative
rows. Report:
[`dsh3-30-collapse-negative-ablation-2026-07-26/report.json`](dsh3-30-collapse-negative-ablation-2026-07-26/report.json),
[`summary.md`](dsh3-30-collapse-negative-ablation-2026-07-26/summary.md).
Reproduce: `PYTHONPATH=src:. python scripts/run_dsh3_30_collapse_negative_ablation.py --output-dir docs/design/dsh3-30-collapse-negative-ablation-2026-07-26`
(runs in well under a second, far under `MAX_RUN_MINUTES`).

| Arm | Total | Usable | Unusable | Usable rate | Ranking correct | Mean margin loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 0 | 0 | 0 | 0.000 | 0 | 0.000 |
| `conflict_only` | 2 | 0 | 2 | 0.000 | 0 | 0.000 |
| `different_result_only` | 4 | 4 | 0 | 1.000 | 2 | 1.000 |
| `equal_mix` | 4 | 2 | 2 | 0.500 | 1 | 1.000 |
| `curriculum_mix` | 6 | 4 | 2 | 0.667 | 2 | 1.000 |

`different_result_only`'s `usable_negative_rate` (1.000) is strictly above
`conflict_only`'s (0.000) — every `DIFFERENT_RESULT` row in this fixture has
a live alternate to rank against, and every `CONFLICT` row has none, by
construction, matching the real re-projection mechanism's own structure. See
`tests/test_models/test_operator_hard_negative_training.py::test_different_result_only_has_strictly_higher_usable_negative_rate_than_conflict_only`
for the isolated, deterministic single-row-per-type version of this
mechanism. `equal_mix` and `curriculum_mix` land between the two pure arms,
as expected from mixing usable and unusable rows.

**Decision: `fixture-scale-positive-signal`.** This is real, deterministic
evidence that `DIFFERENT_RESULT` collapse hard negatives structurally
support pairwise ranking/margin supervision far more densely than `CONFLICT`
ones on this hand-built fixture — but it is fixture-scale wiring/mechanism
evidence on one small synthetic row set, not a claim about downstream
trained-model quality (no real gradient training happened; the scorer is a
fixed deterministic label-free proxy, not a trained model) and not a
real-corpus-scale incidence measurement. Per DSH3-24's own documented scope
gap, real verified-conversation-trace incidence of `CONFLICT` versus
`DIFFERENT_RESULT` hard negatives remains unmeasured — this module does not
attempt to close that gap. This result does **not** by itself justify wiring
hard-negative-aware loss into DSH3-28's frozen
`typed_operator_policy.typed_operator_policy_loss` at production scale: that
would need the real corpus's actual `CONFLICT`/`DIFFERENT_RESULT` incidence
measured first, which remains the acknowledged gap from DSH3-24. The
re-test trigger is to re-run this ablation once `operator_policy_corpus`
supplies real, natural hard-negative rows at meaningful volume, and to
measure real incidence before proposing any production training-loop change.

## Scope

This is a unit-tested selection/scoring/report contract plus one small
synthetic-fixture script run, not a train, eval, matrix, checkpoint, or ship
claim. `typed_operator_policy.py` (DSH3-28's frozen scorer) is untouched;
this module does not plug into its training loop.
