# E274 — legal-conditioned metric feasibility

Date: 2026-07-17
Status: **completed; full-vocabulary mass was partly invalid, training still blocked**

E274 repeats E273's read-only metric-complete gradient profile on the frozen
E228 parent and the same 239 independently judged E261 decision events. The
only changed variable is the probability denominator: good and bad mass are
computed over each event's grammar-derived `legal_token_ids`, matching the
candidate set the deterministic constrained decoder can actually select.
Loss and margin definitions, data splits, solver, and ship gates are unchanged.

The profile ran on CPU with objective `ftpo_set`, 56 train objectives (55
active), 48 held-out objectives, and at most 5,000 deterministic Frank-Wolfe
iterations. Immediately beforehand the branch was clean, rebased on current
`origin/main`, and proved `0 behind / 1 ahead` at `a6f41cc`.

## Result

Legal conditioning materially changes the diagnosis. E273's full-vocabulary
profile had no common-descent certificate (`norm_sq=3.8993e-8`, minimum active
task dot `-3.4113e-7`). E274 finds a nonzero train common-descent vector
(`norm_sq=3.8089e-4`, minimum active task dot `3.3631e-4`). The solver stopped
at 5,000 iterations with duality gap `4.7899e-5`, but the positive certificate
is strict.

This proves that full-vocabulary probability mass was an invalid proxy for the
constrained symbolic decision and created a false train-side Pareto conflict.
Eight E273 held-out regressions disappear, including populated-bracket
loss/margin/good-mass regressions and `lit` bad mass.

The corrected direction is still not safe. Eleven held-out objectives oppose
it. Four persist from E273 (`grammar_comma` bad mass/loss/margin and `lit`
loss), while seven newly selected conflicts include both component-bound mass
objectives, `grammar_comma` good mass, empty-bracket good mass/loss/margin, and
root-populated-bracket mean margin. The minimum-norm mixture is also almost
entirely assigned to `lit:good_probability_mass` (`0.9964`), showing that raw
objective-gradient scale dominates the mixture.

## Decision

Treat legal-token-conditioned probability as the correct semantic measurement
space for constrained decisions, but do not launch a training run from this
direction. The next diagnostic should remove objective-unit scale as a
confounder by profiling normalized metric gradients, then check both train
common descent and held-out alignment before any optimizer mutation. This is a
general metric-geometry correction, not a token or grammar-case special case.

Machine-readable evidence:
[`quality-matrix-v10-e274-legal-conditioned-metric-results.json`](quality-matrix-v10-e274-legal-conditioned-metric-results.json).
