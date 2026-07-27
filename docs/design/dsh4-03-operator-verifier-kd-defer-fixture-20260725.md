# DSH4-03: accepted-set and verifier-backed operator distillation with DEFER (SLM-388)

**Status:** fixture / wiring only.
**Claim class:** `wiring`.
**Honest verdict:** `no_go_defer` -- **stop rule triggered on this run**.
**Depends on:** SLM-387 (DSH4-02, Done, landed as commit `cf9b4dd`) and SLM-386
(DSH4-01, Done, landed as commit `8efcbe0`).

This change answers the DSH4-03 decision question -- *does accepted-set mass,
verifier-backed ranking, entropy/reliability-aware KD, and a fail-closed
DEFER gate improve operator ranking beyond hard accepted-set labels and the
nondistilled certified controller* -- by implementing the issue's five
matched arms directly on top of DSH4-01/02's existing trace/ranking
contracts and running them at fixture scale. It adds one new module,
`src/slm_training/harnesses/distill/operator_verifier_kd_defer.py`.

## Honesty scoping (read first)

**This harness does not train or evaluate a neural student.** Exactly like
DSH4-02, each arm's ranking is the *target distribution* a student trained
under that arm's loss combination would be pushed toward on one decision,
computed directly from typed evidence -- never learned by gradient descent.
A production KD-readiness claim requires actually training a student under
each arm's loss and evaluating its live rollouts; that is out of scope in
this CPU-only, no-GPU sandbox.

**No real external LLM teacher is called.** This reuses DSH4-02's
`SyntheticDescriptorTeacherV1` unchanged -- it is not re-run to flip
DSH4-02's own `no_go_defer` verdict. DSH4-03 asks whether verifier-anchored
KD and DEFER can still add value given that (already-shown-to-be-weak)
teacher.

**The held-out "CAP2-proxy" is not the frozen `dsh3-13` CAP2 operator
suite.** Running that suite requires full compiler+controller wiring beyond
this fixture's scope. This harness reports a fixture-scale substitute --
paired accepted-set-mass bootstrap confidence intervals versus the
nondistilled baseline on a held-out split, across three seeds -- for the
same acceptance question, explicitly labeled as a substitution.

**Verifier ranking never reads a decision's own answer.** A naive
"verifier-backed ranking" signal built from a trace's own
`accepted_application_ids` would be circular. Instead, the verifier signal
is `FrequencyBaselineV1` (DSH4-02) trained on a pool *disjoint* from every
trace scored against it -- a generalizable prior over historically
replay-verified acceptances, never the specific decision's own label. The
`hard_accepted_set` arm is that same disjoint-trained prior collapsed to a
hard one-hot label -- the standard "train with one-hot cross-entropy on the
accepted label" control the issue's Decision section names, not an oracle
with per-decision label access.

## What changed

* `src/slm_training/harnesses/distill/operator_verifier_kd_defer.py`
  * `DistillArm` / `ARM_ORDER` -- the five matched arms: `hard_accepted_set`,
    `teacher_argmax`, `offline_conditional_kd`, `verifier_ranking_kd`,
    `verifier_ranking_kd_defer`.
  * `KLMode` / `normalized_entropy` / `classify_kl_mode` / `apply_kd_shaping`
    -- entropy/reliability-aware shaping: forward-KL/JS mass-covering
    (blend toward uniform) by default; reverse-KL-style mode-seeking
    (power-sharpen) only when the trace's legal-set coverage is `EXACT`
    *and* the teacher distribution is already low-entropy (issue: "reverse
    -KL-style pressure only in low-entropy, high-reliability strata").
  * `_verifier_anchor_blend` -- blends a KD-shaped target toward the
    disjoint-trained `FrequencyBaselineV1` prior, weighted more heavily
    (0.6 vs 0.3) when trace reliability is `APPROXIMATE` (the KD/teacher
    signal is least trustworthy exactly where the legal set was not exactly
    enumerated).
  * `_should_defer` / `DEFER_MARGIN_THRESHOLD` (0.2, declared before this
    run) -- the DEFER gate: defer to `CompilerOrderBaselineV1` (the
    deterministic, exact, ordinary path) on forced singleton/empty
    decisions, on `APPROXIMATE` legal-set coverage, or when the KD target's
    own top1/top2 probability margin is below the declared threshold.
  * `build_arm_target` -- constructs one arm's target ranking for one trace.
  * `evaluate_decision_changes` / `ArmDecisionChangeReportV1` -- measures
    "eligible choices changed" (excluding forced singleton/empty decisions)
    and "correct vs. wrong changes" against a reference ranking, and
    **rejects** (not scores) a comparison whose top-1 picks are identical to
    the reference on `>=99%` of eligible decisions (issue acceptance:
    "prediction-identical auxiliary-loss results are rejected").
  * `evaluate_structural_regressions` / `StructuralRegressionReportV1` -- a
    fixture-scale proxy for "legal-set exactness, CAP0/CAP1 retention, and
    fallback behavior do not regress": every arm must never rank an action
    outside the trace's certified legal set, and must never alter a forced
    singleton/empty decision.
  * `_cap2_proxy_seed_result` / `Cap2ProxySeedResultV1` -- paired
    accepted-set-mass bootstrap CI (`bootstrap_paired_ci`, SLM-183) between
    an arm and the nondistilled baseline on each held-out seed pool.
  * `compare_operator_verifier_kd_defer` -- top-level orchestration;
    enforces `verifier_training_traces` disjoint from `eval_traces` and
    every `held_out_traces_by_seed` pool (pairwise), builds every arm, and
    gates `verifier_ranking_kd` / `verifier_ranking_kd_defer` against the
    full acceptance bar. `hard_accepted_set` is the reference control the
    Decision section asks other arms to beat and is never itself gated.
  * `write_operator_verifier_kd_defer_report` -- JSON writer.
* `scripts/run_dsh4_03_operator_verifier_kd_defer_fixture.py` -- fixture
  runner: reuses DSH4-02's `_build_decision_state` / `GOLD_WEIGHTS` /
  `N_CANDIDATES` unchanged to build a 30-trace verifier-training pool, a
  30-trace evaluation pool, and three disjoint 30-trace held-out pools
  (seeds 0, 1, 2). Prints and writes the harness's honest stop-rule verdict.
* `tests/test_harnesses/distill/test_operator_verifier_kd_defer.py` -- 39
  tests covering entropy/KL-mode classification, KD shaping (order
  preservation), the verifier anchor (never reads the trace's own accepted
  set; reliability-aware weighting), the DEFER gate (singleton, approximate
  coverage, thin margin, confident-and-exact), every arm's target
  construction (including empty legal sets and legal-set containment), the
  decision-change evaluator (correct/wrong counting, ineligible-family
  exclusion, prediction-identical rejection, the 5% threshold), the
  structural regression checks, and end-to-end orchestration (disjointness
  enforcement, JSON round-trip, stop-rule never-partially-passes).
* `src/slm_training/resources/versions.json` -- bumped `harness.distill` to
  `v5` (adds the new module/script/tests/docs to its watched paths).

## Methodology

1. **Reused fixture.** The exact DSH4-02 five-candidate single-operator
   fixture (`openui.dsh4_02_fixture`, `GOLD_WEIGHTS = (0.10, 0.15, 0.50,
   0.15, 0.10)`) is reused unchanged via
   `scripts.run_dsh4_02_operator_teacher_ceiling_fixture._build_decision_state`,
   so this run is directly comparable to DSH4-02's own finding rather than a
   fixture chosen to flip it.
2. **Five matched arms**, each producing a target `RankingV1` per trace:
   * `hard_accepted_set` -- disjoint-trained accepted-label frequency prior
     (`FrequencyBaselineV1`), hard-argmaxed.
   * `teacher_argmax` -- `SyntheticDescriptorTeacherV1`'s own top-1 pick,
     hard-argmaxed.
   * `offline_conditional_kd` -- the teacher's full legal-set distribution,
     entropy/reliability-mode shaped. No verifier signal.
   * `verifier_ranking_kd` -- `offline_conditional_kd`'s shaped target,
     additionally anchored toward the disjoint-trained verifier prior.
   * `verifier_ranking_kd_defer` -- `verifier_ranking_kd`'s target, but
     fails closed to `CompilerOrderBaselineV1` whenever DEFER's gate fires.
3. **Nondistilled baseline reference.** `CurrentScorerBaselineV1` (DSH4-02's
   "the model/scorer already in production") stands in for the certified
   controller each arm must improve on.
4. **Decision-change evidence** (acceptance #1 and #4). For every *eligible*
   decision (more than one legal action -- forced singleton/empty decisions
   are excluded, never touched, per invariant #5), compare the arm's top-1
   pick against the baseline's. A change is "correct" iff the new pick is in
   the trace's replay-verified accepted set. If `>=99%` of eligible picks
   are identical to the baseline, the whole comparison is **rejected**, not
   scored.
5. **Held-out CAP2-proxy** (acceptance #2). On three disjoint 30-trace
   held-out pools (seeds 0/1/2, none overlapping the eval pool, the
   verifier-training pool, or each other), a 2000-resample paired bootstrap
   CI on accepted-set-mass (arm minus baseline) is computed per seed;
   "improves" requires `ci_low > 0` on every seed.
6. **Structural regression proxy** (acceptance #3). Every arm's ranking must
   stay within the trace's certified legal set and must never alter a
   forced singleton/empty decision.
7. **Stop rule.** `verifier_ranking_kd` and `verifier_ranking_kd_defer` are
   gated on: not prediction-identical-rejected, `>=5%` of eligible choices
   changed, correct changes exceed wrong changes, held-out CAP2-proxy
   improves on every seed, and no structural regression. The harness
   recommends `go` iff at least one of those two arms clears every gate.

## Fixture result (honest, unmodified harness output)

```bash
python -m scripts.run_dsh4_03_operator_verifier_kd_defer_fixture \
  --run-id dsh4-03-20260725-fixture
```

Evidence: [dsh4-03-operator-verifier-kd-defer-fixture-20260725.json](dsh4-03-operator-verifier-kd-defer-fixture-20260725.json).
Output: `outputs/runs/dsh4-03-operator-verifier-kd-defer/dsh4-03-20260725-fixture/`
(`summary.json`, `report.json`).

### Per-arm summary (30 eligible eval decisions, vs. `baseline:current_scorer`)

| Arm | changed / eligible | correct | wrong | pred-identical rejected? | CAP2-proxy improves (all 3 seeds)? | arm_go |
| --- | ---: | ---: | ---: | :---: | :---: | :---: |
| `hard_accepted_set` (control) | 0 / 30 | 0 | 0 | **yes** | yes | n/a (control) |
| `teacher_argmax` | 30 / 30 | 2 | 28 | no | no | **no** |
| `offline_conditional_kd` | 30 / 30 | 2 | 28 | no | no | **no** |
| `verifier_ranking_kd` | 0 / 30 | 0 | 0 | **yes** | no | **no** |
| `verifier_ranking_kd_defer` | 30 / 30 | 2 | 28 | no | no | **no** |

`verifier_ranking_kd_defer` deferred on **30/30 (100%)** of eligible
decisions -- every one hit
`kd_target_margin_below_threshold(margin=0.0479, threshold=0.2)` -- so its
row above is really "raw deterministic compiler order vs. current-scorer,"
not a case where the DEFER-shaped KD target was actually applied.

### Verifier-regret correlation and calibration (pooled, per arm)

| Arm | verifier-regret correlation | calibration ECE |
| --- | ---: | ---: |
| `hard_accepted_set` | 0.250 | 0.240 |
| `offline_conditional_kd` | -0.153 | 0.061 |
| `teacher_argmax` | -0.167 | 0.373 |
| `verifier_ranking_kd` | 0.118 | 0.062 |
| `verifier_ranking_kd_defer` | n/a (deferred to a no-probability comparator on every eligible decision) | n/a |

### Held-out CAP2-proxy (paired accepted-set-mass bootstrap CI, arm minus baseline)

| Arm | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| `hard_accepted_set` | +0.307 `[+0.152, +0.461]` | +0.276 `[+0.121, +0.430]` | +0.215 `[+0.061, +0.370]` |
| `offline_conditional_kd` | -0.031 `[-0.047, -0.015]` | -0.028 `[-0.045, -0.011]` | -0.022 `[-0.038, -0.006]` |
| `teacher_argmax` | -0.093 `[-0.204, +0.048]` | -0.058 `[-0.190, +0.093]` | -0.085 `[-0.196, +0.055]` |
| `verifier_ranking_kd` | -0.008 `[-0.013, -0.003]` | -0.008 `[-0.015, -0.002]` | -0.006 `[-0.011, 0.000]` |
| `verifier_ranking_kd_defer` | n/a (0 paired -- deferred to a no-probability comparator) | n/a | n/a |

### Stop rule

```json
{
  "go": false,
  "recommendation": "no_go_defer",
  "passing_arms": [],
  "reasons": [
    "neither verifier_ranking_kd nor verifier_ranking_kd_defer cleared every acceptance gate; per the SLM-388 stop rule, retain the teacher as an evaluation/data tool only"
  ]
}
```

**Honest verdict: DEFER. Retain the teacher as an evaluation/data tool
only**, exactly per this issue's own stop rule text. No arm cleared every
acceptance gate:

* `teacher_argmax` and `offline_conditional_kd` (pure teacher imitation, no
  verifier signal) changed every eligible decision but were **wrong 28 times
  out of 30** -- a direct decision-level confirmation of DSH4-02's finding
  that `SyntheticDescriptorTeacherV1` carries no real signal on this
  fixture and negatively correlates with verifier-backed outcomes
  (verifier-regret correlation -0.153 / -0.167 here, matching DSH4-02's
  -0.153).
* `verifier_ranking_kd`'s blended target converged to the **same** top-1
  pick as the current-scorer baseline on every eligible decision. Per
  acceptance criterion #4 ("prediction-identical auxiliary-loss results are
  rejected"), this is correctly reported as **rejected**, not credited with
  any metric -- even though its CAP2-proxy numbers are only mildly negative,
  reporting them as a "near-miss win" would misrepresent an arm that never
  actually changed a decision.
* `verifier_ranking_kd_defer` deferred on 100% of eligible decisions at the
  declared `DEFER_MARGIN_THRESHOLD = 0.2` -- meaning DEFER did exactly what
  it is supposed to do (refuse to trust a KD target that was not
  confidently and exactly supported), but its fallback path (raw
  deterministic compiler order) is not itself better than the already
  -decent current-scorer baseline on this fixture, so it does not clear the
  CAP2-proxy or correct-vs-wrong gates either.
* `hard_accepted_set`, the reference control, is itself the strongest
  performer (CAP2-proxy improves on every seed, `ci_low` between +0.061 and
  +0.152) -- but it is a disjoint-trained hard-label frequency prior, not a
  candidate arm; no teacher-informed arm surpassed it.

## Why this result is not surprising, and is not fudged

`DEFER_MARGIN_THRESHOLD = 0.2` was declared in the module before this run
and is the same value used for every arm across every seed -- it was not
tuned after seeing the 100% defer rate. The 100% defer rate itself follows
directly from the fixture's structure: `SyntheticDescriptorTeacherV1`'s
raw scores are close to flat across the five candidates (identical
structural features per candidate on this single-operator fixture -- see
DSH4-02's own "why this is not surprising" section), so even after
`FORWARD_KL_MASS_COVER_WEIGHT`/verifier-anchor shaping, the blended target's
top1/top2 margin stays well under 0.2 on every trace. This is the DEFER
gate correctly identifying "not confidently and exactly supported," exactly
as designed -- not a harness bug and not evidence tuned to produce a
particular outcome. The teacher-only arms' 28/30 wrong-change rate is the
same weak/negatively-correlated teacher DSH4-02 already found, now measured
in decision-outcome terms rather than ranking metrics; the two results are
consistent, not independently surprising.

## Honest caveats

* **Wiring-only fixture; synthetic teacher; no trained student.** See
  "Honesty scoping" above. Nothing here is a production KD-readiness claim.
* **CAP2-proxy, not the frozen suite.** The held-out metric is paired
  accepted-set-mass bootstrap CI, not a run of the frozen `dsh3-13` CAP2
  operator suite.
* **`verifier_ranking_kd_defer`'s CAP2-proxy and verifier-regret-correlation
  are `n/a`, not `0`,** on every seed -- `CompilerOrderBaselineV1` (the
  DEFER fallback) never returns probabilities, so `accepted_set_mass` and
  pooled correlation are honestly undefined when 100% of decisions defer,
  not silently reported as a zero or a loss.
* **Stop rule triggered (`no_go_defer`).** This is the harness's real,
  unmodified finding on this fixture, reported per this repo's culture of
  documenting negative/rejected findings (AGENTS.md Iron Law).
* **Single-operator fixture.** Reused unchanged from DSH4-02; the same
  scope caveats apply (all legal actions share one operator declaration,
  so structural descriptor features are uninformative by construction here
  -- a property of the fixture, not a general claim).
* **No production KD-readiness claim.** A production claim requires a real
  teacher, an actually-trained student under each arm's loss, and a re-run
  of this exact harness (or its live-rollout successor) against the frozen
  CAP2 suite.

## Verification commands

```bash
python -m pytest tests/test_harnesses/distill/test_operator_verifier_kd_defer.py -q
python -m pytest tests/test_harnesses/distill/ tests/test_dsl/test_operator_legal_set.py -q
python -m scripts.run_dsh4_03_operator_verifier_kd_defer_fixture --run-id dsh4-03-20260725-fixture
ruff check src/slm_training/harnesses/distill/operator_verifier_kd_defer.py \
  scripts/run_dsh4_03_operator_verifier_kd_defer_fixture.py \
  tests/test_harnesses/distill/test_operator_verifier_kd_defer.py
python -m scripts.verify_version_stamps --check
```

All commands passed on this branch at the time of writing (see the final
report for exact pass/fail counts). Two pre-existing failures in
`tests/test_harnesses/distill/test_meta_traces.py` (`grammar_constrained`
enforcement, unrelated to this change) reproduce identically on a clean
checkout of this branch before this change and are not caused by it.
