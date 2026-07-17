# E269 — minimum-norm decision-kind safe set FTPO

Date: 2026-07-17
Status: **completed; common train descent fails held-out stratified guard; no 30-step run**

E269 replaces pairwise PCGrad with the minimum-norm convex gradient
combination from Sener and Koltun's multi-objective optimization method
([paper](https://arxiv.org/abs/1810.04650),
[reference implementation](https://github.com/isl-org/MultiObjectiveOptimization)).
The harness uses deterministic Frank-Wolfe optimization over the grammar/AST
`decision_kind` gradient simplex and records an explicit common-descent
certificate before the existing held-out stratified guard.

The first one-step preflight fetched/rebased latest `origin/main`, was clean,
and proved `0 behind / 1 ahead` at `9aaeb8c`. It took 212.89s and accepted
0/1 updates. The preflight correctly exposed a solver integration defect: one
decision kind had an exactly zero FTPO gradient, so the unfiltered simplex
selected that zero vertex (`norm_sq=0`, `common_descent=false`). AdamW then
tested weight-decay-only updates, and all five scales were rejected. The parent
was restored exactly; five full-eval gates failed and AgentV remained 2/5.

The generalized repair excludes zero-norm inactive objectives from the
minimum-norm simplex while requiring weak non-conflict for them and strict
common descent for every active objective. Regression tests cover both a
two-objective analytic solution and inactive-task filtering. A corrected
one-step preflight is required before any 30-step run.

The second preflight used 13 active and one inactive objective. After the
initial 250-iteration budget, Frank-Wolfe had not converged and did not certify
common descent (`norm_sq=5.6099`, `min_active_task_dot=-3.7646`). The proposal
was rejected, but the harness still spent five trials probing the uncertified
direction. The final repair raises and reports the convergence budget and
bypasses AdamW plus held-out probing whenever no certificate exists. A final
one-step preflight will distinguish under-convergence from a genuine absence
of common descent; a 30-step run remains disallowed.

Machine-readable invalid-preflight evidence:
[`quality-matrix-v10-e269-one-step-inactive-objective-results.json`](quality-matrix-v10-e269-one-step-inactive-objective-results.json).
Second-preflight evidence:
[`quality-matrix-v10-e269-one-step-fw250-results.json`](quality-matrix-v10-e269-one-step-fw250-results.json).

## Final preflight result

The final one-step preflight repeated the latest-main clean gate and proved
`0 behind / 3 ahead`. With the 5,000-iteration convergence budget, MGDA found
a direction with 13 active and one inactive objective. The direction has a
strict train-objective common-descent certificate (`min_active_task_dot=4.2178`,
`norm_sq=4.3637`); Frank-Wolfe's remaining duality gap was `0.1305`.

All five optimizer-consistent scales were nevertheless rejected. Each scale
regressed eight held-out metrics across the same four grammar/AST categories:
`component_bound`, `grammar_comma`, `lit`, and `sym`. The final local stage took
219.11s for one proposal. No update was accepted, all held-out deltas are zero
after restoration, and the serialized checkpoint SHA is
`518d4736571df2f3842ffd338801cfcc4a855d50358c87bd7563facb191935ba`.

Full evaluation matches the current parent: syntax is 1.0 on all five suites,
five ship thresholds fail, and AgentV passes 2/5 with zero execution errors.

## Decision

Reject E269 and do not run 30 matched steps, sync, or promote. MGDA proves that
a common descent direction exists for the committed train-event objectives,
but that direction does not transfer to the held-out per-kind Pareto contract
at even `1/16` scale. A 30-step run would cost roughly 110 minutes while its
first step is guaranteed to be rejected.

The next generalized investigation should diagnose train/held-out gradient
alignment by decision kind and data provenance before proposing another
optimizer. Do not tune duration, learning rate, Frank-Wolfe iterations, or
individual literal cases.

Final preflight evidence:
[`quality-matrix-v10-e269-one-step-results.json`](quality-matrix-v10-e269-one-step-results.json).
