# DSH3-29 cost-aware ECOC coverage protocol

## Decision

Measure whether the existing ternary ECOC head detects costly wrong actions
under controlled legal-set degradation without allowing a PARTIAL public view
to force an action.

## Frozen evaluator boundary

`evaluator_ecoc_cost_matrix` in the shared typed-policy harness accepts only
compiler-visible declaration cost, locality, effect signature, and a
separately supplied canonical-AST cost tuple. It rejects incomplete cost
coverage and does not accept target distance, gold labels, or shadow truth.
The tuple remains evaluator-only and is never added to
`OperatorPolicyInputV1`.

For each COMPLETE shadow state, use
`build_controlled_partial_fixture` at retained 100/75/50/25% budgets with a
pre-registered, deterministic action order. The public view is PARTIAL at
every budget; the complete shadow is evaluation-only. Record compiler-PARTIAL
defer separately from ECOC invalid-code abstention and only credit a defer
when its exact fallback succeeds.

## Status

Implementation is default-off. The next bounded local matrix must use matched
zero/random/shuffled/enabled arms and the five existing heads. It must report
ordinary and cost-weighted selective risk together, invalid codewords,
coverage, shadow provenance, and permutation checks. No result here is a ship
claim; the positive claim requires the issue's held-out matrix.

## Local bounded result (2026-07-26)

The two explicit current-surface probes, [form](dsh3-29-operator-ecoc-coverage-20260726-local/report.json)
and [dual-card](dsh3-29-operator-ecoc-coverage-20260726-local-dual/report.json),
used the five matched head families, zero/random/shuffled/enabled controls,
four CPU steps, and the local 512-combination cap. Each had two COMPLETE train
rows but zero COMPLETE held-out rows. Both reports therefore reject the
coverage matrix before it can form evaluator-only 100/75/50/25% shadows;
their AgentV bundle passes the legality, singleton, and stop-rule checks and
fails `controlled-partial-defer` specifically for the missing held-out shadow.

This is an honest negative coverage result, not evidence for a cost-aware ECOC
improvement or a ship claim. The new harness keeps the cost matrix and shadow
truth evaluator-only and fails rather than treating a vacuous PARTIAL check as
success.
