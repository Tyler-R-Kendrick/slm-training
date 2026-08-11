# Proof-carrying goal support fixture results

Claim class: **fixture/diagnostic only**. This is bounded wiring evidence, not a model-quality, ship-readiness, global-unrealizability, or production-performance claim.

- Source commit: `af122ace147e7ad6ff29d479c4ec1d9c0744a714` (`dirty=false`)
- Suite: `n=4`, seed `0`, local CPU/in-process, no model
- Solver bounds: `{"max_backtracks": 64, "max_depth": 8, "max_nodes": 64, "max_tokens": 4096, "max_verifier_calls": 64}`
- Exact action cap: `2`; goal-query cap per arm: `32`
- Canonical raw bundle: `outputs/autoresearch/proof-carrying-goal-support-fixture`
- Manifest: `07b0c1070a3530ccf94bd8a2e61ffe34c3796dca12077d76ba8a498f5d65fc7f`
- Canonical result digest: `3a1ab8e174ab6ff922841acf435cb5d63e95cf7220028020f6f4181b1d12a235`
- Deterministic rerun: `PASS`

## Arm results

| arm | coverage | supported | unsupported | unknown | unobserved | selection regret | inadequate under bounds | false hard prune | verifier calls | expanded nodes | wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| structural_support_reference | 0.888889 | 0.777778 | 0 | 0.111111 | 0.111111 | 0 | 0 | 0 | 14 | 16 | 0.011084 |
| goal_support_production_exact_diagnostic | 0.888889 | 0.222222 | 0.555556 | 0.111111 | 0.111111 | 0.25 | 0.25 | 0 | 26 | 29 | 0.04517 |
| goal_support_evaluation_oracle_diagnostic | 0.888889 | 0.0 | 0.0 | 0.888889 | 0.111111 | 0.0 | 0.0 | 0 | 21 | 24 | 0.027489 |
| goal_support_certified_fixture | 0.888889 | 0.222222 | 0.555556 | 0.111111 | 0.111111 | 0.25 | 0.25 | 0 | 37 | 40 | 0.065191 |

## Interpretation

The production-exact diagnostic arm distinguishes the satisfying and violating alternatives, emits a replayable bounded obstruction for the fully observed inadequate case, preserves incomplete/unknown candidates, and reports the candidate-cap case as coverage uncertainty. The evaluation-oracle arm requires `G11`; its mandatory skip remains UNKNOWN and it is never supplied to certified closure.

Certified fixture pruning removed `3` replay-valid compiler-hard UNSUPPORTED actions, produced `1` bounded empty domain, and recorded `0` false hard prunes. Obstruction-core replay rate was `1.0` with mean core size `1.0`.

No checkpoint was created, so `docs/MODEL_CARD.md` and its README summary were intentionally unchanged. No AgentV model-eval bundle was required because this command performs no model evaluation.

Remaining limitation: these are small exact fixtures over terminal completions. Successor work should measure the default-off diagnostic/certified path on real complete OpenUI completion forests without changing the pinned verifier or using fixture evidence for promotion.
