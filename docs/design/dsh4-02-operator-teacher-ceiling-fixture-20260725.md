# DSH4-02: one-teacher ceiling over exact operator legal sets (SLM-387)

**Status:** fixture / wiring only.
**Claim class:** `wiring`.
**Honest verdict:** `no_go_defer` -- **stop rule triggered on this run**.
**Depends on:** SLM-386 (DSH4-01, Done, landed as commit `8efcbe0`).

This change answers the DSH4-02 decision question -- *does a teacher rank
accepted operator actions better than frequency, deterministic compiler
order, and the current scorer, robustly enough to justify distillation* --
**before any distillation work happens**, exactly as the issue's "Decision"
section requires. It adds one new module,
`src/slm_training/harnesses/distill/operator_teacher_ceiling.py`, that
consumes SLM-386's `OperatorDecisionStateTraceV1` exports and implements the
full comparison harness: baselines, metrics, perturbation-robustness tests,
paired significance, and the stop-rule gate.

## Honesty scoping (read first)

**There is no live external LLM teacher wired into this repository or this
environment.** `SyntheticDescriptorTeacherV1` is a deterministic, non-learned
**stand-in** for a real teacher, documented as such in its own docstring. It
scores candidates from structural features already exported on
`LegalOperatorActionV1` (`operator_id`, argument arity, `proof_checks`) plus
a content-identity term derived from `semantic_id`, mixed with deterministic
pseudo-random noise -- **never** from opaque ids, presentation order, natural
-language descriptions, `current_scores`, or `accepted_application_ids`. It
plugs into `OperatorTeacherAdapter`, the same interface a real prompted LLM
teacher would implement, so swapping one in later requires no harness change.

**Every number and verdict in this report is fixture/wiring evidence
(`claim_class: wiring`), never a production KD-readiness claim.** The stop
rule was evaluated honestly against the synthetic teacher's real (weak, on
this fixture) output -- it was not tuned, and the fixture's latent gold-label
distribution (`GOLD_WEIGHTS` in the runner script) was fixed before the
harness was run, not chosen after seeing results. A production KD-readiness
claim requires: (1) a real teacher behind `OperatorTeacherAdapter`, (2)
re-running this exact harness against it, and (3) an honest read of the stop
rule over that real output.

## What changed

* `src/slm_training/harnesses/distill/operator_teacher_ceiling.py`
  * `RankingV1` -- the ranking contract shared by every comparator and the
    teacher: a best-first `application_order`, optional aligned
    `probabilities`, and explicit `reliability` / `reliability_reason` /
    `approximate` metadata bound to the DSH3-06 legal-set coverage of the
    trace it ranks (never fabricated when a comparator cannot honestly
    produce a probability distribution, e.g. pure-ordinal compiler order).
  * `DecisionFamily` -- eight preregistered supervision-source x
    legal-set-size slices (`classify_decision_family`), fixed in code before
    any comparison runs.
  * `TeacherQueryV1` / `PerturbationKind` / `perturb_*` /
    `run_perturbation_robustness` -- the perturbation-robustness harness:
    candidate-order shuffle, opaque-id relabeling, description-length
    variation, and prompt-template-hash variation, each measured as a
    pairwise ranking-distance (Kendall-tau-style discordant-pair fraction)
    between the canonical and perturbed ranking.
  * `OperatorTeacherAdapter` (Protocol) + `SyntheticDescriptorTeacherV1` --
    the teacher interface and its synthetic stand-in implementation.
  * `CompilerOrderBaselineV1`, `FrequencyBaselineV1` (+ `FrequencyTableV1` /
    `build_frequency_table`), `CurrentScorerBaselineV1`,
    `DescriptorSimilarityBaselineV1`, `OracleAcceptedSetComparatorV1` -- the
    required comparators (frequency, deterministic compiler order, current
    scorer, simple descriptor similarity) plus the oracle upper-bound
    reference.
  * Metrics: `accepted_set_mass`, `top_k_recall`, `reciprocal_rank` (MRR),
    `ndcg_at_k`, `expected_calibration_error`, `selective_risk_aurc`,
    `pearson_correlation` / `spearman_correlation` (used for verifier-regret
    correlation). Every metric returns `None` -- never a fabricated number --
    when its inputs are insufficient (empty accepted set, no probabilities,
    zero variance).
  * `PairedComparisonV1` + a paired bootstrap (`_bootstrap_ci`, 2000
    resamples by default) for teacher-vs-baseline confidence evidence, both
    overall and per preregistered `DecisionFamily`.
  * `StopRuleVerdictV1` + `evaluate_stop_rule` -- the stop-rule gate as a
    pure function over already-computed evidence (paired comparisons,
    perturbation report, verifier-regret correlation), independently
    unit-testable without any trace/teacher machinery.
  * `compare_operator_teacher_ceiling` -- top-level orchestration; enforces
    that `frequency_training_traces` is disjoint from `eval_traces` (a
    frequency prior trained on its own evaluation set would leak gold labels
    into a "baseline").
  * `write_operator_teacher_ceiling_report` -- JSON writer.
* `scripts/run_dsh4_02_operator_teacher_ceiling_fixture.py` -- deterministic
  fixture runner: builds 30 frequency-training states and 30 disjoint
  evaluation states over a small compiler operator with a fixed, reused
  five-candidate value vocabulary, a fixed latent gold-label distribution
  (`GOLD_WEIGHTS`, invisible to the teacher and the descriptor-similarity
  baseline), and noisy `current_scores` that roughly track the same latent
  distribution (a realistic "already-decent" current scorer, not a
  strawman). Prints and writes the harness's honest stop-rule verdict.
* `tests/test_harnesses/distill/test_operator_teacher_ceiling.py` -- 39
  tests covering every baseline comparator, every metric (including empty
  legal set / single candidate / tie edge cases), perturbation-robustness
  invariance, the stop-rule gate (a genuine passing scenario, a genuine
  failing scenario, and an insufficient-data scenario, all from hand-built
  evidence objects), reliability metadata on approximate/partial-coverage
  traces, the frequency train/eval leakage guard, and full end-to-end report
  shape / JSON round-trip.
* `src/slm_training/resources/versions.json` -- bumped `harness.distill` to
  `v4` (adds the new module/script/tests/docs to its watched paths).

## Methodology

1. **Teacher adapter interface.** `OperatorTeacherAdapter.rank(query)` takes
   a `TeacherQueryV1` (trace + candidate presentation order + cosmetic
   opaque-id labels + cosmetic description lengths + prompt-template hash)
   and returns a `RankingV1 | None`. This is the exact interface a real
   prompted LLM teacher would implement -- the presentation fields exist so
   perturbation-robustness tests have something concrete to vary.
2. **Synthetic stand-in teacher.** `SyntheticDescriptorTeacherV1` scores each
   legal action as `(1 - noise_weight) * descriptor_similarity + noise_weight
   * deterministic_noise(semantic_id)`, softmaxes, and returns **exact**
   finite-set logits when DSH3-06 enumeration achieved
   `LegalSetCoverage.COMPLETE`, or an explicitly reliability-marked
   **approximate** preference when coverage was budget-truncated
   (`PARTIAL`). It never reads `current_scores`, `accepted_application_ids`,
   or any `TeacherQueryV1` presentation field.
3. **Baselines.** `FrequencyBaselineV1` (Laplace-smoothed historical
   acceptance frequency by `semantic_id`, trained on a disjoint pool),
   `CompilerOrderBaselineV1` (raw DSH3-06 enumeration order, no scores),
   `CurrentScorerBaselineV1` (softmax over `trace.current_scores`, with
   explicit partial-score reliability downgrade), `DescriptorSimilarityBaselineV1`
   (the same structural features as the teacher, without its noise term),
   and `OracleAcceptedSetComparatorV1` (perfect ranking of the accepted set
   -- an upper-bound reference, never a baseline the teacher must beat).
4. **Metrics**, computed per trace and aggregated overall and per
   `DecisionFamily`: accepted-set mass, top-3 recall, MRR, NDCG@3
   (rank-based, preregistered primary/gating metric is MRR -- it is the only
   one defined for every comparator including the score-free compiler-order
   baseline), calibration ECE and selective-risk AURC (pooled over
   candidates), and verifier-regret correlation (Spearman correlation
   between a comparator's per-candidate probability and whether that
   candidate was in the accepted/verifier-backed set, pooled across all
   candidates and traces).
5. **Paired significance.** For each required baseline and each metric, a
   2000-resample paired bootstrap over per-trace `teacher - baseline`
   differences yields a 95% CI; `teacher_significantly_better` requires
   `ci_low > 0.0` and `n_paired > 0` (an empty comparison is reported as
   `insufficient_data`, never silently counted as a win).
6. **Perturbation robustness.** For each of candidate-order shuffle,
   opaque-id relabeling, description-length variation, and prompt-template
   variation, the teacher is queried on the canonical and perturbed query
   for every multi-candidate trace; the maximum discordant-pair ranking
   distance across all traces and kinds must stay within the declared bound
   (`DECLARED_PERTURBATION_BOUND = 0.0`).
7. **Stop rule.** `evaluate_stop_rule` requires **all** of: the teacher
   significantly beats every required baseline on paired MRR (overall);
   perturbation robustness stays within bound; and teacher scores positively
   correlate with verifier-backed outcomes. Any single failure forces
   `go=False` -- the gate never partially passes.

## Fixture result (honest, unmodified harness output)

```bash
python -m scripts.run_dsh4_02_operator_teacher_ceiling_fixture \
  --run-id dsh4-02-20260725-fixture
```

Evidence: [dsh4-02-operator-teacher-ceiling-fixture-20260725.json](dsh4-02-operator-teacher-ceiling-fixture-20260725.json).
Output: `outputs/runs/dsh4-02-operator-teacher-ceiling/dsh4-02-20260725-fixture/`
(`summary.json`, `report.json`).

### Overall metrics (30 eval traces, 5 legal actions each)

| Comparator | MRR | top-3 recall | NDCG@3 | accepted-set mass | calibration ECE | selective-risk AURC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle:accepted_set` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `baseline:current_scorer` | 0.594 | 0.767 | 0.601 | 0.216 | 0.037 | 0.577 |
| `baseline:frequency` | 0.580 | 0.633 | 0.534 | 0.241 | 0.011 | 0.556 |
| `baseline:descriptor_similarity` | 0.435 | 0.633 | 0.423 | 0.200 | 0.000 | 0.703 |
| `baseline:compiler_order` | 0.418 | 0.767 | 0.469 | n/a (no scores) | n/a | n/a |
| **`teacher:synthetic_descriptor_v1`** | **0.343** | 0.467 | 0.280 | 0.197 | 0.063 | 0.963 |

### Paired MRR, teacher vs. each required baseline (2000-resample bootstrap, n=30)

| Baseline | mean diff | 95% CI | significantly better? |
| --- | ---: | --- | :---: |
| `baseline:frequency` | -0.237 | [-0.400, -0.065] | no |
| `baseline:compiler_order` | -0.075 | [-0.133, -0.017] | no |
| `baseline:current_scorer` | -0.251 | [-0.421, -0.078] | no |
| `baseline:descriptor_similarity` | -0.092 | [-0.186, -0.006] | no |

### Perturbation robustness

| Kind | max ranking distance | bound | within bound |
| --- | ---: | ---: | :---: |
| candidate order | 0.000 | 0.000 | yes |
| opaque-id relabel | 0.000 | 0.000 | yes |
| description length | 0.000 | 0.000 | yes |
| prompt-template hash | 0.000 | 0.000 | yes |

### Verifier-regret correlation (pooled Spearman, probability vs. accepted)

| Comparator | correlation |
| --- | ---: |
| `oracle:accepted_set` | 1.000 |
| `baseline:current_scorer` | 0.197 |
| `baseline:frequency` | 0.137 |
| **`teacher:synthetic_descriptor_v1`** | **-0.153** |
| `baseline:compiler_order` | n/a (no scores) |
| `baseline:descriptor_similarity` | n/a (no scores; see note) |

### Stop rule

```json
{
  "go": false,
  "recommendation": "no_go_defer",
  "beats_required_baselines": false,
  "failing_baselines": [
    "baseline:compiler_order", "baseline:current_scorer",
    "baseline:descriptor_similarity", "baseline:frequency"
  ],
  "perturbation_within_bound": true,
  "correlation_positive": false,
  "verifier_regret_correlation": -0.1532064692570853
}
```

**Honest verdict: DEFER. Do not distill `SyntheticDescriptorTeacherV1`.** It
did not significantly beat any required baseline on paired MRR, and its
scores negatively correlate with verifier-backed (accepted) outcomes.
Perturbation robustness is perfect (`0.0` on every kind) -- the teacher is
architecturally, not just empirically, immune to candidate-order and
opaque-id fragility, since it never reads those fields. That one criterion
passing does not offset the other two failing; the gate requires all three.

## Why this result is not surprising, and is not fudged

On this fixture, every legal action is an application of the *same* single
-argument operator (`openui.dsh4_02_fixture`) -- so `LegalOperatorActionV1`'s
structural features (`operator_id`, argument arity, `proof_checks`) are
**identical across all five candidates in every state**; the only thing that
differs between candidates is the bound value's content identity
(`semantic_id`), which `descriptor_similarity` deliberately does not treat as
a raw feature (using a value-identity hash directly as a "similarity"
feature would not be a real structural signal, just frequency-baseline logic
by another name). `DescriptorSimilarityBaselineV1` is therefore an honest
tie on this fixture, and the synthetic teacher (similarity + noise) reduces
to essentially noise. `FrequencyBaselineV1` and `CurrentScorerBaselineV1`,
by contrast, have a legitimate path to the fixture's latent skew (`GOLD_WEIGHTS`)
-- frequency learns it from a disjoint training pool, and `current_scores`
was constructed to roughly track it (a realistic "already-decent" scorer,
not a strawman) -- so both outperform a teacher with no comparable signal.
This is the harness correctly identifying that its synthetic teacher has no
legitimate information advantage on this fixture, not a harness bug. A real
LLM teacher, given a natural-language description of what each candidate
*means*, would have signal the synthetic teacher structurally cannot see;
that is exactly the follow-up this fixture is scoped to require, not to
simulate.

## Honest caveats

* **Wiring-only fixture; synthetic teacher.** No external teacher model,
  checkpoint, or ship gate is loaded, called, or evaluated.
  `SyntheticDescriptorTeacherV1` is a deterministic non-learned stand-in;
  see the "Honesty scoping" section above.
* **Stop rule triggered (DEFER).** This is the harness's real, unmodified
  finding on this fixture -- it is reported, not hidden or worked around,
  per this repo's culture of documenting negative/rejected findings
  (AGENTS.md Iron Law; see prior `not_authorized`/`CERT_*` gate-closeout
  precedent, e.g. SLM-340/341/343/344/362).
* **Single-operator fixture.** All legal actions share one operator
  declaration; `descriptor_similarity`'s structural features are therefore
  uninformative by construction on this fixture (see above) -- this is a
  property of the fixture's scope, not a general claim that descriptor
  similarity is never useful.
* **`LegalOperatorActionV1` does not carry `ActionEffectV1` for unapplied
  candidates** (only opaque proof/effect digests) -- a teacher with true
  effect-level visibility would need a live dry-run/replay per candidate,
  out of scope here; the synthetic teacher's visibility is bounded by the
  same DSH3-06 export.
* **No production KD-readiness claim.** A production claim requires a real
  teacher behind `OperatorTeacherAdapter` and a re-run of this exact
  harness; per the SLM-387 stop rule, no distillation work should follow
  from this synthetic-teacher result.

## Verification commands

```bash
python -m pytest tests/test_harnesses/distill/test_operator_teacher_ceiling.py -q
python -m pytest tests/test_harnesses/distill/ tests/test_dsl/test_operator_legal_set.py -q
python -m scripts.run_dsh4_02_operator_teacher_ceiling_fixture --run-id dsh4-02-20260725-fixture
ruff check src/slm_training/harnesses/distill/operator_teacher_ceiling.py \
  scripts/run_dsh4_02_operator_teacher_ceiling_fixture.py \
  tests/test_harnesses/distill/test_operator_teacher_ceiling.py
python -m scripts.verify_version_stamps --check
```

All commands passed on this branch at the time of writing (see the final
report for exact pass/fail counts).
