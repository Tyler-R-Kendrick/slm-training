# E273 — metric-complete preference feasibility

Date: 2026-07-17
Status: **completed; full guard objective set is Pareto-conflicted**

E273 profiles the frozen E228 parent without training. For every grammar/AST
`decision_kind`, it differentiates the four exact objectives enforced by the
held-out guard: loss and bad probability mass are minimized; good probability
mass and mean margin are maximized. It then solves the minimum-norm convex
combination across all train objectives and checks every held-out objective.

The profile fetched/rebased latest `origin/main`, was clean, and proved
`0 behind / 1 ahead` at `7fa4dfc`.

## Result

The train split contains 56 objectives, 55 with nonzero gradients. After 5,000
Frank-Wolfe iterations, the minimum-norm vector is essentially zero
(`norm_sq=3.8993e-8`) but still has a negative active-task dot
(`-3.4113e-7`), so no nonzero common-descent certificate exists. The remaining
duality gap is `1.3028e-5`.

The solution is dominated by `component_bound` probability-mass objectives:
good mass receives weight `0.7449` and bad mass `0.2271`. Twelve held-out
objectives oppose the near-stationary combination, including
`grammar_comma` loss/margin/bad mass, populated-bracket loss/margin/good mass,
`grammar_rpar` good mass, and `lit` loss/bad mass.

## Decision

Do not run metric-complete MGDA training. The current guard objective set is
already Pareto-conflicted at the frozen parent, so another optimizer or smaller
step cannot satisfy it.

Before changing data or architecture, verify the probability-space contract.
Good/bad mass currently comes from a full-vocabulary softmax, while constrained
decoding selects only among each event's `legal_token_ids`. If the guard is
intended to assess the symbolic decision, it should likely use legal-candidate
conditional probability. The next diagnostic must compare full-vocabulary and
legal-conditioned metrics without weakening ship gates or changing syntax
ownership.

Machine-readable evidence:
[`quality-matrix-v10-e273-metric-complete-results.json`](quality-matrix-v10-e273-metric-complete-results.json).
