# E275 — normalized legal-metric geometry

Date: 2026-07-17
Status: **completed; scale artifact removed, three held-out conflicts remain**

E275 repeats E274's frozen-parent, read-only profile with legal-token-conditioned
probability mass and one generalized geometry correction: every nonzero metric
gradient is scaled to unit L2 norm before the minimum-norm convex combination.
The resulting direction is checked against the original, unscaled train and
held-out gradients, so normalization cannot hide a regression by changing the
acceptance units.

The run used CPU, the unchanged E228 checkpoint, all 239 independently judged
E261 events, objective `ftpo_set`, 56 train objectives (55 active), 48 held-out
objectives, and at most 5,000 deterministic Frank-Wolfe iterations. The branch
was clean, rebased on current `origin/main`, and proved `0 behind / 1 ahead` at
`0045257` immediately before the profile.

## Result

The unit-normalized solver finds strict train common descent
(`norm_sq=4.5356e-4`, minimum active normalized-task dot `3.7717e-4`). Every
active original train objective also has positive alignment; `train_regressions`
is empty. The solver reached the iteration cap with duality gap `9.0017e-5`, so
the result is a positive feasibility certificate rather than a convergence
claim.

Normalization removes E274's pathological `0.9964` weight on
`lit:good_probability_mass`. The largest weights are now distributed across
grammar-derived objectives: `grammar_comma:mean_margin` (`0.4228`),
`grammar_rpar:good_probability_mass` (`0.1520`), populated-bracket loss
(`0.1134`), and root-populated-bracket margin (`0.0899`).

Held-out regressions fall from eleven to three:

- `component_bound:bad_probability_mass`
- `component_bound:good_probability_mass`
- `lit:loss`

Thus legal conditioning and unit normalization resolve the two evaluator and
unit-geometry artifacts, but the direction is still unsafe at infinitesimal
scale. No optimizer run is authorized.

## Decision

The next diagnostic should stratify objectives by a reusable decision signature
derived from event semantics—`decision_kind` plus legal/good/bad token sets—so
different grammar/AST states are not averaged into one kind-level gradient.
That directly tests whether the last three conflicts are hidden within-kind
mixtures. Do not add component/literal special cases, weaken the held-out guard,
or increase training duration.

Machine-readable evidence:
[`quality-matrix-v10-e275-normalized-legal-metric-results.json`](quality-matrix-v10-e275-normalized-legal-metric-results.json).
