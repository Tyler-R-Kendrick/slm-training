# E269 — minimum-norm decision-kind safe set FTPO

Date: 2026-07-17
Status: **in progress; two one-step preflights repaired solver fail-closed behavior**

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
